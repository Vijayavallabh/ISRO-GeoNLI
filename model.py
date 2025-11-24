"""
VLM with 2-layer MLP Adapter supporting any vision encoder and language decoder.
"""
import torch
import torch.nn as nn
from transformers import (
    AutoModel, 
    AutoTokenizer,
    AutoProcessor,
)
from transformers.activations import GELUTanh
import transformers.activations as activations
activations.PytorchGELUTanh = GELUTanh
PytorchGELUTanh = GELUTanh
from typing import Optional, Dict, Any
from config import ModelConfig


class MLPAdapter(nn.Module):
    """
    2-layer MLP adapter to project vision features to language decoder space.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        self.pre_norm = nn.LayerNorm(input_dim)
        self.linear_1 = nn.Linear(input_dim * 4, input_dim * 4)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(input_dim * 4, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Vision features [batch_size, num_tokens, vision_dim]
        Returns:
            Projected features [batch_size, num_tokens/4, language_dim]
        """
        x = self.pre_norm(x)
        
        # Patch merging (2x2 pooling)
        B, L, D = x.shape
        
        # Handle potential CLS token
        H = int(L ** 0.5)
        if H * H != L:
            H = int((L - 1) ** 0.5)
            if H * H == L - 1:
                x = x[:, 1:, :]
                L = L - 1
        
        W = L // H
        x = x.view(B, H, W, D)
        x = x.view(B, H // 2, 2, W // 2, 2, D)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, -1, D * 4)
        
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x


class VLMWithAdapter(nn.Module):
    """
    Vision-Language Model with frozen encoders and trainable MLP adapter.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load vision encoder
        self.vision_encoder = self._load_vision_encoder()
        if config.freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        
        # Load language decoder
        self.language_decoder = AutoModel.from_pretrained(
            config.language_decoder_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        ).language_model
        if config.freeze_language_decoder:
            for param in self.language_decoder.parameters():
                param.requires_grad = False
        
        # Get dimensions
        self.vision_dim = self._get_vision_dim()
        self.language_dim = self.language_decoder.config.hidden_size
        
        # Initialize MLP adapter (this is the ONLY trainable component)
        self.adapter = MLPAdapter(
            input_dim=self.vision_dim,
            output_dim=self.language_dim
        )
        
        # Load processors
        self.image_processor = AutoProcessor.from_pretrained(config.vision_encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.language_decoder_name)
        
        # Set special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _load_vision_encoder(self) -> nn.Module:
        """Load the appropriate vision encoder based on config."""
        return AutoModel.from_pretrained(self.config.vision_encoder_name).vision_tower()
    
    def _get_vision_dim(self) -> int:
        """Get vision encoder output dimension."""
        if hasattr(self.vision_encoder.config, 'hidden_size'):
            return self.vision_encoder.config.hidden_size
        elif hasattr(self.vision_encoder.config, 'projection_dim'):
            return self.vision_encoder.config.projection_dim
        else:
            raise ValueError("Cannot determine vision encoder output dimension")
    
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode images using vision encoder.
        
        Args:
            pixel_values: Preprocessed images [batch_size, channels, height, width]
        Returns:
            Vision features [batch_size, num_patches, vision_dim]
        """
        with torch.no_grad() if self.config.freeze_vision_encoder else torch.enable_grad():
            outputs = self.vision_encoder(pixel_values)
            if hasattr(outputs, 'last_hidden_state'):
                vision_features = outputs.last_hidden_state
            elif hasattr(outputs, 'pooler_output'):
                vision_features = outputs.pooler_output.unsqueeze(1)
            else:
                raise ValueError("Cannot extract vision features from encoder")
        
        return vision_features
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VLM.
        
        Args:
            pixel_values: Images [batch_size, 3, H, W]
            input_ids: Text tokens [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target tokens for loss computation [batch_size, seq_len]
        
        Returns:
            Dictionary with loss and logits
        """
        batch_size = pixel_values.shape[0]
        
        # 1. Encode images
        vision_features = self.encode_image(pixel_values)  # [B, num_patches, vision_dim]
        
        # 2. Project through adapter
        projected_features = self.adapter(vision_features)  # [B, num_patches, language_dim]
        
        # 3. Get text embeddings
        text_embeds = self.language_decoder.get_input_embeddings()(input_ids)  # [B, seq_len, language_dim]
        
        # 4. Concatenate visual and text embeddings
        # Vision tokens come first, then text tokens
        num_vision_tokens = projected_features.shape[1]
        combined_embeds = torch.cat([projected_features, text_embeds], dim=1)  # [B, num_patches + seq_len, language_dim]
        
        # 5. Create attention mask for combined sequence
        vision_attention_mask = torch.ones(
            batch_size, num_vision_tokens,
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        combined_attention_mask = torch.cat([vision_attention_mask, attention_mask], dim=1)
        
        # 6. Forward through language decoder
        outputs = self.language_decoder(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            labels=self._prepare_labels(labels, num_vision_tokens) if labels is not None else None,
            return_dict=True
        )
        
        return {
            "loss": outputs.loss if labels is not None else None,
            "logits": outputs.logits,
            "vision_features": vision_features,
            "projected_features": projected_features
        }
    
    def _prepare_labels(self, labels: torch.Tensor, num_vision_tokens: int) -> torch.Tensor:
        """
        Prepare labels by padding with -100 for vision tokens.
        
        Args:
            labels: Original labels [batch_size, seq_len]
            num_vision_tokens: Number of vision tokens to prepend
        
        Returns:
            Padded labels [batch_size, num_vision_tokens + seq_len]
        """
        batch_size = labels.shape[0]
        vision_labels = torch.full(
            (batch_size, num_vision_tokens),
            -100,  # Ignore index for cross-entropy
            dtype=labels.dtype,
            device=labels.device
        )
        return torch.cat([vision_labels, labels], dim=1)
    
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 50,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text given image and text prompt.
        
        Args:
            pixel_values: Images
            input_ids: Prompt tokens
            attention_mask: Attention mask
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Generated token IDs
        """
        # Encode and project image
        vision_features = self.encode_image(pixel_values)
        projected_features = self.adapter(vision_features)
        
        # Get text embeddings
        text_embeds = self.language_decoder.get_input_embeddings()(input_ids)
        
        # Combine embeddings
        batch_size = pixel_values.shape[0]
        num_vision_tokens = projected_features.shape[1]
        combined_embeds = torch.cat([projected_features, text_embeds], dim=1)
        
        # Create combined attention mask
        vision_attention_mask = torch.ones(
            batch_size, num_vision_tokens,
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        combined_attention_mask = torch.cat([vision_attention_mask, attention_mask], dim=1)
        
        # Generate
        outputs = self.language_decoder.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        
        return outputs
    
    def get_trainable_parameters(self):
        """Return only the adapter parameters for optimization."""
        return self.adapter.parameters()
    
    def print_trainable_parameters(self):
        """Print the number of trainable parameters."""
        trainable_params = sum(p.numel() for p in self.adapter.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || "
              f"Trainable%: {100 * trainable_params / all_params:.2f}%")
