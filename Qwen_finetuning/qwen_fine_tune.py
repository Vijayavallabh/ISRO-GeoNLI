import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen2_5_VLForConditionalGeneration, 
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from typing import Dict, List, Any
import random

# --- CHANGED: Updated Dataset Class to handle multiple sources ---
class VRSDataset(Dataset):    
    def __init__(self, dataset_configs: List[Dict[str, str]], processor, max_samples=None): 
        """
        Args:
            dataset_configs: List of dicts, e.g., 
                             [{"image_dir": "path/A", "annotation_dir": "path/B"}, ...]
            processor: AutoProcessor instance
            max_samples: Optional limit for debugging
        """
        self.processor = processor
        self.annotations = []
        
        for config in dataset_configs:
            img_dir = config["image_dir"]
            ann_dir = config["annotation_dir"]
            
            print(f"Loading annotations from: {ann_dir}")
            
            current_source_anns = []
            if os.path.exists(ann_dir):
                for ann_file in os.listdir(ann_dir):
                    if ann_file.endswith('.json'):
                        with open(os.path.join(ann_dir, ann_file), 'r') as f:
                            ann = json.load(f)
                            # IMPORTANT: Store the specific image root for this file
                            # so we know where to look for the image later
                            ann['_root_image_dir'] = img_dir 
                            current_source_anns.append(ann)
            else:
                print(f"Warning: Directory {ann_dir} not found.")

            print(f" -> Found {len(current_source_anns)} samples in {ann_dir}")
            self.annotations.extend(current_source_anns)
        
        if max_samples:
            self.annotations = self.annotations[:max_samples]
            
        print(f"Total Combined Samples: {len(self.annotations)}")
    
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
        
        # 3. QA pairs
        if "qa_pairs" in annotation:
            for qa in annotation["qa_pairs"]:
                samples.append({
                    "prompt": qa["question"],
                    "response": qa["answer"]
                })
        
        return samples
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # --- CHANGED: Load image using the specific root dir for this annotation ---
        image_path = os.path.join(annotation["_root_image_dir"], annotation["image"])
        
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            # Fallback or error handling if path is wrong
            print(f"Error: Image not found at {image_path}")
            # Create a black image to prevent crash, or raise error depending on preference
            image = Image.new('RGB', (224, 224), color='black')

        # Get all training samples
        training_samples = self.create_training_samples(annotation)
        
        conversation = []
        
        for i, sample in enumerate(training_samples):
            user_content = []
            if i == 0:
                user_content.append({"type": "image"})
            user_content.append({"type": "text", "text": sample["prompt"]})
            
            conversation.append({
                "role": "user",
                "content": user_content
            })
            conversation.append({
                "role": "assistant",
                "content": [{"type": "text", "text": sample["response"]}]
            })
        
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
        
        # Masking logic
        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()
        labels[:] = -100
        
        assistant_token_id = self.processor.tokenizer.encode("assistant", add_special_tokens=False)[0]
        eos_token_id = self.processor.tokenizer.eos_token_id
        
        i = 0
        while i < len(input_ids):
            if input_ids[i] == assistant_token_id:
                start_response = i + 2 
                end_response = len(input_ids)
                for j in range(start_response, len(input_ids)):
                    if input_ids[j] == eos_token_id:
                        end_response = j + 1
                        break
                labels[start_response:end_response] = input_ids[start_response:end_response]
                i = end_response
            else:
                i += 1
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "image_grid_thw": inputs["image_grid_thw"].squeeze(0),
            "labels": labels
        }

def collate_fn(batch):
    # (Same as your original code)
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    pixel_values_list = [item["pixel_values"] for item in batch]
    image_grid_thw_list = [item["image_grid_thw"] for item in batch]
    
    max_len = max(len(ids) for ids in input_ids)
    
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    pad_token_id = 0 
    
    for ids, mask, lab in zip(input_ids, attention_mask, labels):
        padding_length = max_len - len(ids)
        padded_input_ids.append(torch.cat([ids, torch.full((padding_length,), pad_token_id, dtype=ids.dtype)]))
        padded_attention_mask.append(torch.cat([mask, torch.zeros(padding_length, dtype=mask.dtype)]))
        padded_labels.append(torch.cat([lab, torch.full((padding_length,), -100, dtype=lab.dtype)]))
    
    batch_dict = {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
        "labels": torch.stack(padded_labels)
    }
    
    if len(pixel_values_list) > 0:
        all_pixel_values = []
        all_image_grid_thw = []
        for pv, thw in zip(pixel_values_list, image_grid_thw_list):
            if pv.dim() == 3: pv = pv.unsqueeze(0)
            if thw.dim() == 1: thw = thw.unsqueeze(0)
            all_pixel_values.append(pv)
            all_image_grid_thw.append(thw)
        
        batch_dict["pixel_values"] = torch.cat(all_pixel_values, dim=0)
        batch_dict["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)
    
    return batch_dict

def main():
    # Configuration
    MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
    OUTPUT_DIR = "outputs/qwen2_5_8"
    
    # --- CHANGED: Define multiple dataset configurations here ---
    DATASET_CONFIGS = [
        {
            "image_dir": "VRS/train", 
            "annotation_dir": "VRS_annotations/train"
        }
    ]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device_map = {"": local_rank} if local_rank != -1 else "auto"

    print(f"Loading model... on {local_rank}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map=device_map,
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



    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    target_modules = []
    for name, module in model.named_modules():
        if "model.layers" in name and "self_attn" in name and any(x in name for x in ["q_proj", "v_proj"]):
            if "visual" not in name:
                module_name = name.split(".")[-1]
                if module_name not in target_modules:
                    target_modules.append(module_name)
    
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=target_modules, 
        lora_dropout=0, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # --- CHANGED: Pass the config list to the dataset ---
    print("Loading datasets...")
    train_dataset = VRSDataset(DATASET_CONFIGS, processor)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=32, 
        gradient_accumulation_steps=2, 
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        save_total_limit=1,
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name="vrs_covt_ft",
        ddp_find_unused_parameters=False,
        overwrite_output_dir=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )
    
    print("Starting training...")
    trainer.train()
    
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
