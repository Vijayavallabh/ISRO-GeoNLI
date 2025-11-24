"""
Data loading utilities for VQA datasets with distributed support.
"""
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from datasets import load_dataset, concatenate_datasets
from PIL import Image
from typing import Dict, List, Optional, Any
from config import DataConfig
import io


class VQADataset(Dataset):
    """
    Generic VQA dataset that works with HuggingFace datasets.
    Supports multiple dataset formats.
    """
    def __init__(
        self,
        dataset_names: List[str],
        splits: List[str],
        image_processor,
        tokenizer,
        max_length: int = 512,
        streaming: bool = False,
        cache_dir: Optional[str] = None
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load and concatenate datasets
        datasets = []
        for name, split in zip(dataset_names, splits):
            print(f"Loading dataset: {name} (split: {split})")
            ds = load_dataset(
                name,
                split=split,
                streaming=streaming,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            datasets.append(ds)
        
        # Concatenate if multiple datasets
        if len(datasets) > 1:
            self.dataset = concatenate_datasets(datasets)
        else:
            self.dataset = datasets[0]
        
        print(f"Total samples: {len(self.dataset) if not streaming else 'streaming'}")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single VQA sample.
        
        Returns dict with:
            - pixel_values: Processed image tensor
            - input_ids: Tokenized question
            - attention_mask: Attention mask
            - labels: Tokenized answer (for training)
        """
        item = self.dataset[idx]
        
        # Extract image, question, and answer
        # Adapt these keys based on your dataset format
        image = self._load_image(item)
        question = self._extract_text(item, 'question')
        answer = self._extract_text(item, 'answer')
        
        # Process image
        pixel_values = self.image_processor(
            images=image,
            return_tensors="pt"
        )['pixel_values'].squeeze(0)
        
        # Format prompt: "Question: <question> Answer:"
        prompt = f"Question: {question} Answer:"
        full_text = f"{prompt} {answer}"
        
        # Tokenize
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length // 2,  # Leave space for answer
            add_special_tokens=True
        )
        
        full_tokens = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            add_special_tokens=True,
            return_tensors='pt'
        )
        
        input_ids = full_tokens['input_ids'].squeeze(0)
        attention_mask = full_tokens['attention_mask'].squeeze(0)
        
        # Create labels (mask prompt tokens with -100)
        labels = input_ids.clone()
        prompt_length = len(prompt_tokens['input_ids'])
        labels[:prompt_length] = -100  # Don't compute loss on prompt
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def _load_image(self, item: Dict) -> Image.Image:
        """Load image from dataset item."""
        if 'image' in item:
            img = item['image']
        elif 'img' in item:
            img = item['img']
        else:
            raise KeyError("No image field found in dataset")
        
        # Handle different image formats
        if isinstance(img, Image.Image):
            return img.convert('RGB')
        elif isinstance(img, bytes):
            return Image.open(io.BytesIO(img)).convert('RGB')
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")
    
    def _extract_text(self, item: Dict, field: str) -> str:
        """Extract text field from dataset item."""
        # Common field names for questions and answers
        if field == 'question':
            possible_keys = ['question', 'query', 'text', 'prompt']
        else:  # answer
            possible_keys = ['answer', 'response', 'label', 'multiple_choice_answer']
        
        for key in possible_keys:
            if key in item:
                value = item[key]
                # Handle list of answers (take first one)
                if isinstance(value, list):
                    return str(value[0])
                return str(value)
        
        raise KeyError(f"No {field} field found in dataset")


def get_dataloaders(
    config: DataConfig,
    image_processor,
    tokenizer,
    world_size: int = 1,
    rank: int = 0
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders with distributed support.
    
    Args:
        config: Data configuration
        image_processor: Image processor from vision encoder
        tokenizer: Tokenizer from language decoder
        world_size: Number of distributed processes
        rank: Current process rank
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Training dataset
    train_dataset = VQADataset(
        dataset_names=config.dataset_names,
        splits=config.dataset_splits,
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_length=config.max_length,
        streaming=config.streaming,
        cache_dir=config.cache_dir
    )
    
    # Validation dataset
    val_dataset = VQADataset(
        dataset_names=[config.val_dataset_name],
        splits=[config.val_split],
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_length=config.max_length,
        streaming=False,  # Don't stream validation
        cache_dir=config.cache_dir
    )
    
    # Create samplers for distributed training
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size if hasattr(config, 'batch_size') else 16,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size if hasattr(config, 'batch_size') else 16,
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
