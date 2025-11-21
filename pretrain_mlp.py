import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict, Any
import random
import torch.nn.utils.rnn as rnn_utils

# This patch must run BEFORE we import from transformers
import transformers.activations as activations
from transformers.activations import NewGELUActivation, GELUActivation
try:
    # Try to import the new name
    from transformers.activations import PytorchGELUTanh
except ImportError:
    # If it fails, import the old name and "alias" it
    from transformers.activations import GELUTanh
    activations.PytorchGELUTanh = GELUTanh
    PytorchGELUTanh = GELUTanh
# ==========================================

from transformers import (
    AutoModel, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    AutoProcessor
)

# ==========================================
# 2. & 3. Model & Projector 
# ==========================================
class MoonQwenProjector(nn.Module):
    def __init__(self, vision_dim, llm_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
    def forward(self, x):
        return self.net(x)

class MoonQwenVLM(nn.Module):
    def __init__(self, vision_model_path, llm_model_path):
        super().__init__()
        
        print(f"Loading Vision Tower: {vision_model_path}...")
        self.vision_tower = AutoModel.from_pretrained(
            vision_model_path, 
            trust_remote_code=True, 
            dtype=torch.bfloat16, # Using updated 'dtype'
        )
        self.vision_tower.requires_grad_(False)

        print(f"Loading LLM: {llm_model_path}...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16, # Using updated 'dtype'
            attn_implementation="flash_attention_2"
        )
        self.llm.requires_grad_(False)

        self.vision_dim = self.vision_tower.config.hidden_size * 4 ## multiply by 4
        print('vision dimension:',self.vision_dim)
        self.llm_dim = self.llm.config.hidden_size

        self.projector = MoonQwenProjector(self.vision_dim, self.llm_dim).to(dtype=torch.bfloat16)
        self.projector.requires_grad_(True)


    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None, image_grid_hws=None):
        # Only the vision_tower call should be in no_grad()
        with torch.no_grad():
            vis_outputs = self.vision_tower(
                pixel_values=pixel_values,
                grid_hws=image_grid_hws
            )
            # vis_outputs is a LIST of tensors [img1_patches, img2_patches, ...]
             
        image_embeds_list = []
        patch_counts = []
        
        for image_features in vis_outputs:
            # image_features shape is [num_patches_for_this_image, 4, 1152]
            
            # 1. Store the patch count for this image
            patch_counts.append(image_features.shape[0])
            
            # 2. Flatten the features
            # Shape becomes [num_patches, 4608]
            image_features_flat = image_features.view(image_features.shape[0], -1)
            
            # 3. Project the features (THIS IS NOW TRACKED FOR GRADIENTS)
            # Shape becomes [num_patches, 4096]
            image_embed_flat = self.projector(image_features_flat)
            
            # 4. Add to our list
            image_embeds_list.append(image_embed_flat)

        # 3. Pad the list of projected tensors
        # This creates a single, batched tensor
        # Shape: [batch_size, max_patches, llm_dim]
        padded_image_embeds = rnn_utils.pad_sequence(
            image_embeds_list, 
            batch_first=True, 
            padding_value=0.0
        )

        # 4. Get text embeddings (this is also no-grad, as LLM is frozen)
        with torch.no_grad():
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # 5. Concatenate
        combined_embeds = torch.cat([padded_image_embeds, inputs_embeds], dim=1)

        # 6. Create the new attention mask
        batch_size = padded_image_embeds.shape[0]
        max_patches = padded_image_embeds.shape[1]
        text_len = inputs_embeds.shape[1]
        
        image_attn_mask = torch.arange(max_patches, device=padded_image_embeds.device)[None, :] < torch.tensor(patch_counts, device=padded_image_embeds.device)[:, None]
        
        text_attn_mask = attention_mask
        if text_attn_mask is None:
             text_attn_mask = torch.ones((batch_size, text_len), device=combined_embeds.device, dtype=torch.long)

        combined_mask = torch.cat([image_attn_mask.to(torch.long), text_attn_mask], dim=1)

        # 7. Create new labels (ignore image patches)
        img_labels = torch.full((batch_size, max_patches), -100, device=combined_embeds.device, dtype=torch.long)
        text_labels = labels
        if text_labels is None:
            text_labels = input_ids.clone()
            
        combined_labels = torch.cat([img_labels, text_labels], dim=1)

        # 8. Run the LLM
        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=combined_labels
        )
        
        return outputs
# ==========================================
# 4. Dataset Loading (UNCHANGED)
# ==========================================
class PretrainDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        _ = self.data_list[idx] 
        
        # --- MOCK DATA (Variable Resolution) ---
        width = random.randint(300, 800)
        height = random.randint(300, 800)
        image = Image.new('RGB', (width, height), color='red') 
        caption = f"A description of the red image of size {width}x{height}."
        
        return {
            "image": image,
            "text": caption
        }

# 5. Custom Data Collator (MODIFIED)
# ==========================================
class CustomDataCollator:
    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = [item.pop("image") for item in batch]
        texts = [item.pop("text") for item in batch]

        # Process texts
        text_inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding="longest",
            truncation=True, 
            max_length=128
        )

        image_inputs = self.image_processor(
            images=images, 
            return_tensors="pt"
        )
        
        labels = text_inputs.input_ids.clone()
        
        # Merge the two dictionaries
        final_batch = {
            "pixel_values": image_inputs.pixel_values.to(torch.bfloat16),
            "input_ids": text_inputs.input_ids,
            "attention_mask": text_inputs.attention_mask,
            "labels": labels,
            "image_grid_hws": image_inputs.image_grid_hws 
        }
            
        return final_batch
# ==========================================
# 6. Main Training Function 
# ==========================================
def train():
    VISION_ID = "moonshotai/MoonViT-SO-400M"
    LLM_ID = "Qwen/Qwen3-4B"
    
    tokenizer = AutoTokenizer.from_pretrained(LLM_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    try:
        image_processor = AutoProcessor.from_pretrained(VISION_ID, trust_remote_code=True)
    except:
        from transformers import SiglipImageProcessor
        image_processor = SiglipImageProcessor.from_pretrained("google/siglip-so4B4m-patch14-384")

    print("Creating mock data split...")
    all_data_indices = [i for i in range(600)]
    train_data = all_data_indices[:-50]
    val_data = all_data_indices[-50:]
    
    train_dataset = PretrainDataset(train_data)
    val_dataset = PretrainDataset(val_data)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    data_collator = CustomDataCollator(tokenizer, image_processor)
    model = MoonQwenVLM(VISION_ID, LLM_ID)

    training_args = TrainingArguments(
        output_dir="./checkpoints/moon-qwen-align-variable-res",
        per_device_train_batch_size=4, 
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=12,
        eval_accumulation_steps=1, 
        num_train_epochs=1,
        learning_rate=1e-3, 
        bf16=True, 
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="tensorboard",
        logging_steps=3, ## Change this
        eval_strategy="steps",
        eval_steps=3,
        save_steps=3,
        save_safetensors=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("Starting Training with Validation & Plotting...")
    trainer.train()
    
    torch.save(model.projector.state_dict(), "moon_qwen_projector_variable_res.bin")

if __name__ == "__main__":
    train()