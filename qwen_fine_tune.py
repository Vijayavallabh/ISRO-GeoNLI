import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from typing import Dict, List
import random

class VRSBenchDataset(Dataset):
    """Dataset class for VRS Bench annotations."""
    
    def __init__(self, image_dir: str, annotation_dir: str, processor, max_samples=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.processor = processor
        
        # Load all annotation files
        self.annotations = []
        for ann_file in os.listdir(annotation_dir):
            if ann_file.endswith('.json'):
                with open(os.path.join(annotation_dir, ann_file), 'r') as f:
                    ann = json.load(f)
                    self.annotations.append(ann)
        
        if max_samples:
            self.annotations = self.annotations[:max_samples]
        
        print(f"Loaded {len(self.annotations)} samples")
    
    def __len__(self):
        return len(self.annotations)
    
    def create_training_samples(self, annotation: Dict) -> List[Dict]:
        """Create multiple training samples from one annotation."""
        samples = []
        
        # 1. Caption generation task
        samples.append({
            "prompt": "Describe this image in detail.",
            "response": annotation["caption"]
        })
        
        # 2. Object referring tasks
        for obj in annotation["objects"]:
            samples.append({
                "prompt": f"Describe the {obj['obj_cls']} in this image.",
                "response": obj["referring_sentence"]
            })
        
        # 3. QA pairs
        for qa in annotation["qa_pairs"]:
            samples.append({
                "prompt": qa["question"],
                "response": qa["answer"]
            })
        
        return samples
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, annotation["image"])
        image = Image.open(image_path).convert("RGB")
        
        # Create training samples and randomly select one
        training_samples = self.create_training_samples(annotation)
        sample = random.choice(training_samples)
        
        # Format as conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": sample["prompt"]}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample["response"]}
                ]
            }
        ]
        
        # Process inputs
        text = self.processor.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=False,
            return_tensors="pt"
        )
        
        # Prepare labels (mask the prompt part)
        labels = inputs["input_ids"].clone()
        
        # Find where assistant response starts
        assistant_token = self.processor.tokenizer.encode("assistant", add_special_tokens=False)[0]
        for i, token_id in enumerate(inputs["input_ids"][0]):
            if token_id == assistant_token:
                # Mask everything before assistant response
                labels[0, :i+2] = -100
                break
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "image_grid_thw": inputs["image_grid_thw"].squeeze(0),
            "labels": labels.squeeze(0)
        }

def collate_fn(batch):
    """Custom collate function to handle variable-length sequences and vision inputs."""
    # Extract components
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # For vision inputs, we need to handle them carefully
    # pixel_values can have different shapes due to different image sizes
    pixel_values_list = [item["pixel_values"] for item in batch]
    image_grid_thw_list = [item["image_grid_thw"] for item in batch]
    
    # Pad text sequences
    max_len = max(len(ids) for ids in input_ids)
    
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    
    pad_token_id = 0  # Qwen2-VL uses 0 for padding
    
    for ids, mask, lab in zip(input_ids, attention_mask, labels):
        padding_length = max_len - len(ids)
        padded_input_ids.append(torch.cat([ids, torch.full((padding_length,), pad_token_id, dtype=ids.dtype)]))
        padded_attention_mask.append(torch.cat([mask, torch.zeros(padding_length, dtype=mask.dtype)]))
        padded_labels.append(torch.cat([lab, torch.full((padding_length,), -100, dtype=lab.dtype)]))
    
    # Stack text tensors
    batch_dict = {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
        "labels": torch.stack(padded_labels)
    }
    
    # Handle vision inputs - concatenate along batch dimension
    if len(pixel_values_list) > 0:
        # pixel_values shape: [num_patches, channels, height, width]
        # We need to concatenate all images' patches
        all_pixel_values = []
        all_image_grid_thw = []
        
        for pv, thw in zip(pixel_values_list, image_grid_thw_list):
            # Ensure tensors have correct dimensions
            if pv.dim() == 3:  # Missing batch dimension
                pv = pv.unsqueeze(0)
            if thw.dim() == 1:  # Missing batch dimension
                thw = thw.unsqueeze(0)
            all_pixel_values.append(pv)
            all_image_grid_thw.append(thw)
        
        batch_dict["pixel_values"] = torch.cat(all_pixel_values, dim=0)
        batch_dict["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)
    
    return batch_dict

def main():
    # Configuration
    MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"  # Using 7B as 8B might be a typo
    IMAGE_DIR = "VRS_image_val"  # Update this
    ANNOTATION_DIR = "VRS_annotations_val"  # Update this
    OUTPUT_DIR = "./qwen2vl_lora_finetuned"
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model with quantization
    print("Loading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # Use flash attention if available
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=256*28*28,
        max_pixels=1280*28*28,
        padding_side="right"
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    # Find all attention layer names in the language model
    # Qwen2-VL structure: model.layers.{i}.self_attn.{q,k,v,o}_proj
    target_modules = []
    for name, module in model.named_modules():
        if "model.layers" in name and "self_attn" in name and any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            # Extract just the relative module name
            if "visual" not in name:  # Exclude vision encoder
                module_name = name.split(".")[-1]
                if module_name not in target_modules:
                    target_modules.append(module_name)
    
    print(f"Target modules for LoRA: {target_modules}")
    
    # LoRA configuration - targeting only language model attention layers
    lora_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,  # Scaling factor
        target_modules=target_modules,  # ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=None,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = VRSBenchDataset(IMAGE_DIR, ANNOTATION_DIR, processor)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Reduced to 1 for stability
        gradient_accumulation_steps=4,  # Increased to maintain effective batch size
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="tensorboard",
        ddp_find_unused_parameters=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving model...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    print(f"Training complete! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
