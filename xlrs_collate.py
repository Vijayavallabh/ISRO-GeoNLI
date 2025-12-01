import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def custom_collate(batch):
    # Custom collate to handle variable-sized images
    images = [item['image'] for item in batch]
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    bbox_pixels = torch.stack([item['bbox_pixel'] for item in batch])
    bbox_normalizeds = torch.stack([item['bbox_normalized'] for item in batch])
    original_sizes = torch.stack([item['original_size'] for item in batch])
    target_sizes = torch.stack([item['target_size'] for item in batch])
    scale_factors = [item['scale_factor'] for item in batch]
    types = [item['type'] for item in batch]
    
    return {
        'image': images,  # List of PIL images
        'question': questions,
        'answer': answers,
        'bbox_pixel': bbox_pixels,
        'bbox_normalized': bbox_normalizeds,
        'original_size': original_sizes,
        'target_size': target_sizes,
        'scale_factor': scale_factors,
        'type': types
    }

class XLRSBenchmarkDataset(Dataset):
    def __init__(self, base_path, use_viz=False):
        self.base_path = base_path
        self.use_viz = use_viz
        self.samples = []
        
        # Load all samples
        for sample_dir in sorted(os.listdir(base_path)):
            sample_path = os.path.join(base_path, sample_dir)
            if os.path.isdir(sample_path):
                json_file = os.path.join(sample_path, "data.json")
                if os.path.exists(json_file):
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    for item in data:
                        item['sample_dir'] = sample_dir
                        self.samples.append(item)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        sample_dir = item['sample_dir']
        img_dir = "Viz" if self.use_viz else "Clean"
        img_path = os.path.join(self.base_path, sample_dir, img_dir, item['filename'])
        
        # Load image as PIL RGB
        image = Image.open(img_path).convert('RGB')
        
        # Prepare bbox as tensor
        bbox_pixel = torch.tensor(item['bbox_pixel'], dtype=torch.float32)
        bbox_normalized = torch.tensor(item['bbox_normalized'], dtype=torch.float32)
        
        return {
            'image': image,
            'question': item['question'],
            'answer': item['answer'],
            'bbox_pixel': bbox_pixel,
            'bbox_normalized': bbox_normalized,
            'original_size': torch.tensor(item['original_size'], dtype=torch.int32),
            'target_size': torch.tensor(item['target_size'], dtype=torch.int32),
            'scale_factor': item['scale_factor'],
            'type': item['type']
        }

# Example usage
base_path = "/.cache/kagglehub/datasets/lokeshop/xlrs-downsampled-bench/versions/1/By_Sample"

dataset = XLRSBenchmarkDataset(base_path, use_viz=False)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)

# Example to get one batch
for batch in dataloader:
    print([img.size for img in batch['image']])  # List of sizes
    print(batch['question'][0])
    print(batch['bbox_pixel'][0])
    break
