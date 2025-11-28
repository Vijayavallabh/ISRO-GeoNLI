from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from PIL import Image
import logging
from typing import Dict, Sequence
import torch.nn.functional as F
from trl import SFTTrainer, SFTConfig
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants (from ovis/util/constants.py and model config)
IGNORE_ID = -100
IMAGE_TOKEN = "<image>"

# Special token IDs (from Ovis2.5 configuration)
IMAGE_TOKEN_ID = 151665
VISUAL_INDICATOR_IDS = [151666, 151667, 151668, 151669, 151670]

logger.info("Loading model...")

# Load model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "AIDC-AI/Ovis2.5-9B",
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto"
)

# Use AutoProcessor
processor = AutoProcessor.from_pretrained(
    "AIDC-AI/Ovis2.5-9B",
    trust_remote_code=True
)

# Access components
text_tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

# Add missing methods
model.get_input_embeddings = lambda: model.llm.get_input_embeddings()
model.get_output_embeddings = lambda: model.llm.get_output_embeddings()

# Freeze vision encoder
for param in model.visual_tokenizer.parameters():
    param.requires_grad = False
for param in model.vte.parameters():
    param.requires_grad = False

logger.info("Vision encoder frozen ✓")

# Prepare for training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=r".*llm\.model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))",
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
    use_rslora=True,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Update references after LoRA wrapping
base_model = model.base_model.model
visual_tokenizer = base_model.visual_tokenizer

# Verification
logger.info("\n=== Detailed Verification ===")
vision_trainable = sum(p.numel() for p in base_model.visual_tokenizer.parameters() if p.requires_grad)
vte_trainable = sum(p.numel() for p in base_model.vte.parameters() if p.requires_grad)
llm_base_trainable = sum(p.numel() for n, p in model.llm.named_parameters() if p.requires_grad and 'lora' not in n)
llm_lora_trainable = sum(p.numel() for n, p in model.llm.named_parameters() if p.requires_grad and 'lora' in n)

logger.info(f"Visual tokenizer trainable params: {vision_trainable:,} (should be 0)")
logger.info(f"Visual embedding trainable params: {vte_trainable:,} (should be 0)")
logger.info(f"LLM base trainable params: {llm_base_trainable:,}")
logger.info(f"LLM LoRA trainable params: {llm_lora_trainable:,}")

import os
import json
from datasets import Dataset

logger.info("\nLoading VRSBench dataset from Annotations_train...")

# Load samples from Annotations_train folder
annotations_dir = "VRSBench/Annotations_train"
samples = []
for filename in os.listdir(annotations_dir):
    if filename.endswith('.json'):
        with open(os.path.join(annotations_dir, filename), 'r') as f:
            sample = json.load(f)
            # Only extract caption and image keys
            samples.append({
                "image": sample["image"],
                "caption": sample["caption"]
            })

# Create Dataset from list of samples
dataset = Dataset.from_list(samples)

instruction = "Describe the content shown in the image in detail."

# Training hyperparameters (matching official TrainingArguments)
SINGLE_IMAGE_MIN_PIXELS = 448 * 448
SINGLE_IMAGE_MAX_PIXELS = 1792 * 1344
MULTIMODAL_MAX_LENGTH = 4096

def preprocess_function(sample, idx):
    """
    Preprocess each sample following the official Ovis CaptionDataset pattern.
    Returns input_ids, pixel_values, grid_thws, attention_mask, and labels.
    """
    try:
        image_path = "VRSBench/Images_train/" + sample["image"]
        caption = sample["caption"]
        
        # Create caption template with image token
        caption_template = f"User: {IMAGE_TOKEN}\n{instruction}\nAssistant: "
        
        # Split template around IMAGE_TOKEN
        head, tail = caption_template.split(IMAGE_TOKEN)
        
        # Tokenize text components (without special tokens)
        head_ids = text_tokenizer(head, add_special_tokens=False).input_ids
        tail_ids = text_tokenizer(tail, add_special_tokens=False).input_ids
        text_ids = text_tokenizer(caption, add_special_tokens=False).input_ids
        
        # Process image
        image = Image.open(image_path).convert("RGB")
        pixel_values, grid_thws = visual_tokenizer.preprocess(
            image=image,
            min_pixels=SINGLE_IMAGE_MIN_PIXELS,
            max_pixels=SINGLE_IMAGE_MAX_PIXELS
        )
        
        # Calculate number of visual tokens needed
        # grid_thws shape: [1, 3] where dims are [temporal, height, width]
        num_image_atoms = grid_thws[0].prod().item()
        
        # Account for visual tokenizer architecture
        hidden_stride = visual_tokenizer.vit.config.hidden_stride
        temporal_patch_size = visual_tokenizer.vit.config.temporal_patch_size
        
        num_image_atoms //= (hidden_stride ** 2)
        num_image_atoms //= temporal_patch_size
        
        # Create image placeholder tokens
        # Format: [indicator_0] + [atom] * num_atoms + [indicator_1]
        # indicator_0 marks start of image, indicator_1 marks end
        image_placeholders = [VISUAL_INDICATOR_IDS[0]] + [VISUAL_INDICATOR_IDS[0]] * num_image_atoms + [VISUAL_INDICATOR_IDS[1]]
        
        # Construct full input_ids sequence
        # Format: head_text + image_placeholders + tail_text + caption_text
        input_ids = head_ids + image_placeholders + tail_ids + text_ids
        
        # Create labels: mask everything except the caption text
        # Only train on the assistant's response (caption)
        labels = [IGNORE_ID] * (len(input_ids) - len(text_ids)) + text_ids
        
        # Verify no padding token in input
        if text_tokenizer.pad_token_id is not None and text_tokenizer.pad_token_id in input_ids:
            logger.warning(f"Sample {idx} contains padding token")
            raise ValueError("Padding token in input")
            
    except Exception as e:
        logger.error(f'Processing sample failed with idx: {idx}, error: {str(e)}')
        # Return dummy values on error (will be skipped in training)
        pixel_values, grid_thws = None, None
        input_ids = [0]
        labels = [IGNORE_ID]
    
    # Truncate to maximum length
    input_ids = input_ids[:MULTIMODAL_MAX_LENGTH]
    labels = labels[:MULTIMODAL_MAX_LENGTH]
    
    # Convert to tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.full_like(input_ids, fill_value=True, dtype=torch.bool)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "grid_thws": grid_thws,
        "attention_mask": attention_mask,
        "labels": labels
    }

# Apply preprocessing
logger.info("Preprocessing dataset...")

# Use map with batched=False for individual sample processing
processed_dataset = dataset.map(
    preprocess_function,
    with_indices=True,
    remove_columns=dataset.column_names,
)
class DataCollatorForMultimodalDataset:
    """Official Ovis data collator"""
    
    def __init__(self, text_tokenizer):
        self.text_tokenizer = text_tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        keys = ("input_ids", "pixel_values", "grid_thws", "attention_mask", "labels")
        input_ids, pixel_values, grid_thws, attention_mask, labels = (
            tuple(instance[key] for instance in instances) for key in keys
        )
        
        # Pad input_ids
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.text_tokenizer.pad_token_id
        )
        
        # Concatenate pixel_values across batch
        pixel_values = [x for x in pixel_values if x is not None]
        pixel_values = torch.cat(pixel_values, dim=0) if len(pixel_values) > 0 else None
        
        # Concatenate grid_thws across batch
        grid_thws = [x for x in grid_thws if x is not None]
        grid_thws = torch.cat(grid_thws, dim=0) if len(grid_thws) > 0 else None
        
        # Pad attention_mask
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=False
        )
        
        # Pad labels
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_ID
        )
        
        # Add padding if no padding token exists in attention_mask
        if 0 not in attention_mask:
            input_ids = F.pad(input_ids, (0, 1), value=self.text_tokenizer.pad_token_id)
            attention_mask = F.pad(attention_mask, (0, 1), value=False)
            labels = F.pad(labels, (0, 1), value=IGNORE_ID)
        
        if torch.all(labels == IGNORE_ID):
            logging.warning('[DataCollatorForMultimodalDataset] All samples in the current batch are ignored.')
            
        return dict(
            input_ids=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            attention_mask=attention_mask,
            labels=labels
        )

training_args = SFTConfig(
    output_dir="./ovis2.5-vrsbench-lora",
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 8,   # effective batch size ~= 16 on 1 GPU
        # Train for full epochs instead of a tiny max_steps
        num_train_epochs = 3,
        # learning rate & schedule tuned for LoRA on 8B models
        learning_rate = 1e-4,
        warmup_ratio = 0.03,
        lr_scheduler_type = "cosine",
        weight_decay = 0.0,

        logging_steps = 10,
        optim = "adamw_8bit",
        seed = 3407,
        report_to = "none",
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 1024,   # enough for instruction + long captions
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    data_collator=DataCollatorForMultimodalDataset(text_tokenizer=text_tokenizer),
)

# Start training
logger.info("Starting training...")
trainer.train()
