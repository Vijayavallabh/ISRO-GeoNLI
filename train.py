import os
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
from tqdm import tqdm
import argparse


class MLPAdapter(nn.Module):
    """Two-layer MLP adapter for vision-language alignment [web:18]"""
    def __init__(self, vision_hidden_size, text_hidden_size, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = text_hidden_size  # Default to text_hidden_size for simplicity
        self.mlp = nn.Sequential(
            nn.Linear(vision_hidden_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, text_hidden_size),
            nn.LayerNorm(text_hidden_size)
        )
    
    def forward(self, x, input_embeddings=None, output_embeddings=None):
        x = self.mlp(x)
        # Apply soft-token bottleneck if embeddings are provided
        if input_embeddings is not None and output_embeddings is not None:
            logits = output_embeddings(x)
            
            x = torch.softmax(logits, dim=-1) @ input_embeddings.weight
            
        return x

class Qwen3VLWithAdapter(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=None,
        )
        
        # Get actual hidden sizes from the model
        self.vision_hidden_size = 1152 #self.base_model.model.visual.config.hidden_size
        self.text_hidden_size = 4096#self.base_model.config.text_config.hidden_size
        
        # Store original visual forward for potential reuse
        
        # Replace deepstack_merger_list with identity layers for each deepstack index

        self.base_model.model.visual.merger = nn.Identity()
        if hasattr(self.base_model.model.visual, 'deepstack_visual_indexes'):
            num_deepstack = len(self.base_model.model.visual.deepstack_visual_indexes)
            self.base_model.model.visual.deepstack_merger_list = nn.ModuleList([
                nn.Identity() for _ in range(num_deepstack)
            ])
        else:
            self.base_model.model.visual.deepstack_merger_list = nn.ModuleList()

        self.original_visual_forward = self.base_model.model.visual.forward
        # Freeze base model
        for p in self.base_model.parameters():
            p.requires_grad = False

        base_dtype = next(self.base_model.parameters()).dtype
        self.adapter = MLPAdapter(self.vision_hidden_size, self.text_hidden_size)
        self.adapter.to(dtype=base_dtype)

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw, labels):
        # Get text embeddings - use language_model's embedding layer
        text_embeds = self.base_model.language_model.get_input_embeddings()(input_ids)
        
        # Process vision through frozen encoder
        with torch.no_grad():
            vision_outputs = self.original_visual_forward(
                pixel_values.to(dtype=self.base_model.dtype),
                grid_thw=image_grid_thw,
            )
        
        if isinstance(vision_outputs, tuple):
            vision_feats = vision_outputs[0]  # (total_patches, 1152)
        elif hasattr(vision_outputs, "last_hidden_state"):
            vision_feats = vision_outputs.last_hidden_state
        else:
            vision_feats = vision_outputs

        merge_size = self.base_model.model.visual.spatial_merge_size  # usually 2
        m2 = merge_size ** 2
        seq_len, dim = vision_feats.shape
        assert dim == self.vision_hidden_size

        # Group 2x2 patches -> one token
        assert seq_len % m2 == 0, "seq_len must be divisible by merge_size**2"
        vision_feats_grouped = vision_feats.view(-1, m2, dim)        # (num_tokens, 4, 1152)
        vision_feats_pooled = vision_feats_grouped.mean(1)           # (num_tokens, 1152)

        # Now apply your adapter: (num_tokens, 4096)
        input_embeddings = self.base_model.language_model.get_input_embeddings()
        output_embeddings = self.base_model.get_output_embeddings()
        adapted_features = self.adapter(
            vision_feats_pooled,
            input_embeddings=input_embeddings,
            output_embeddings=output_embeddings,
        )
        # Find image token positions and replace with adapted features
        image_token_id = self.base_model.config.image_token_id
        image_positions = (input_ids == image_token_id)

        combined_embeds = text_embeds.clone()

        batch_size = input_ids.size(0)
        total_img_tokens = image_positions.sum().item()
        n_image_features = adapted_features.size(0)

        if total_img_tokens != n_image_features:
            raise ValueError(
                f"Mismatch between image features ({n_image_features}) "
                f"and <image> tokens in input_ids ({total_img_tokens})"
            )

        # Sequentially fill image features across the batch
        feat_ptr = 0
        for b in range(batch_size):
            img_mask = image_positions[b]
            num_img_tokens = img_mask.sum().item()
            if num_img_tokens > 0:
                combined_embeds[b, img_mask] = adapted_features[
                    feat_ptr : feat_ptr + num_img_tokens
                ]
                feat_ptr += num_img_tokens
                
        # Forward through language model with combined embeddings
        outputs = self.base_model.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
        )
        
        hidden_states = outputs[0]
        logits = self.base_model.get_output_embeddings()(hidden_states)
        
        # Calculate loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        # Return in expected format
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )
    
class LLaVAPretrainDataset(Dataset):
    """Dataset for LLaVA-style pretraining data [web:18][web:19]"""
    def __init__(self, json_path, processor):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.processor = processor
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path =  item['image']
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Format conversation for Qwen3-VL
        messages = []
        for conv in item['conversations']:
            role = conv['from']
            content = conv['value']
            
            if role == 'human':
                # Replace <image> placeholder with proper format
                content_list = []
                if '<image>' in content:
                    content_list.append({"type": "image", "image": image})
                    content = content.replace('<image>', '').strip()
                if content:
                    content_list.append({"type": "text", "text": content})
                messages.append({"role": "user", "content": content_list})
            else:  # gpt/assistant
                messages.append({"role": "assistant", "content": content})
        
        return messages, image


def collate_fn(batch, processor):
    """Custom collate function for batch processing [web:1]"""
    all_messages = [item[0] for item in batch]
    
    # Process batch with processor
    texts = processor.apply_chat_template(
        all_messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    # Tokenize and prepare inputs
    inputs = processor(
        text=texts,
        images=[item[1] for item in batch],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    
    # Create labels (shift input_ids for causal LM)
    labels = inputs['input_ids'].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'pixel_values': inputs['pixel_values'],
        'image_grid_thw': inputs['image_grid_thw'],
        'labels': labels
    }


def setup_distributed():
    """Initialize distributed training [web:17]"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()


def train_epoch(model, dataloader, optimizer, scheduler, epoch, local_rank):
    """Training loop for one epoch [web:19]"""
    model.train()
    total_loss = 0
    
    if local_rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader
    
    for step, batch in enumerate(pbar):
        # Move batch to device
        batch = {k: v.to(local_rank) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Logging
        if local_rank == 0 and step % 10 == 0:
            avg_loss = total_loss / (step + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default='LLaVA-Pretrain/blip_laion_cc_sbu_558k_fixed.json',
        help='Path to JSON dataset'
    )
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen3-VL-8B-Instruct')

    # Per-GPU batch size; choose this so that
    # batch_size * world_size * grad_accum_steps ≈ 256
    parser.add_argument('--batch_size', type=int, default=4)

    # LLaVA pretrain uses 1 epoch over the 558K dataset
    parser.add_argument('--num_epochs', type=int, default=1)

    # LLaVA pretrain lr
    parser.add_argument('--lr', type=float, default=1e-3)

    # Optional; currently unused, but you can use it if you later add a warmup scheduler
    parser.add_argument('--warmup_steps', type=int, default=0)

    args = parser.parse_args()
    
    # Setup distributed training
    local_rank = setup_distributed()
    # Load processor
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = 'right'
    
    # Create dataset and dataloader [web:18]
    dataset = LLaVAPretrainDataset(args.data_path,  processor)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=lambda batch: collate_fn(batch, processor),
        num_workers=4,
        pin_memory=True
    )
    
    model = Qwen3VLWithAdapter(args.model_path).to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    # Optimizer and scheduler (only adapter parameters)
    optimizer = torch.optim.AdamW(
        model.module.adapter.parameters(),
        lr=args.lr,
        weight_decay=0
    )
    
    total_steps = len(dataloader) * args.num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-5
    )
    
    # Training loop [web:19]
    if local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Training on {dist.get_world_size()} GPUs")
        print(f"Total steps: {total_steps}")
    
    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        avg_loss = train_epoch(model, dataloader, optimizer, scheduler, epoch, local_rank)
        
        if local_rank == 0:
            print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'adapter_state_dict': model.module.adapter.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss
            }
            torch.save(
                checkpoint,
                os.path.join(args.output_dir, f'adapter_epoch_{epoch}.pt')
            )
    
    cleanup_distributed()


if __name__ == '__main__':
    main()
