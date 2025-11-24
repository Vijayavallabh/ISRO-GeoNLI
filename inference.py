"""
Inference script to test trained VLM adapter.
"""
import torch
from PIL import Image
from model import VLMWithAdapter
from config import VLMAdapterTrainConfig, ModelConfig
from pathlib import Path
import argparse


def load_trained_adapter(checkpoint_path: str, config: ModelConfig):
    """Load model with trained adapter weights."""
    # Create model
    model = VLMWithAdapter(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.adapter.load_state_dict(checkpoint['adapter_state_dict'])
    
    return model


def generate_answer(
    model: VLMWithAdapter,
    image_path: str,
    question: str,
    device: torch.device,
    max_new_tokens: int = 50
) -> str:
    """Generate answer for image and question."""
    model.eval()
    model = model.to(device)
    
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    pixel_values = model.image_processor(
        images=image,
        return_tensors="pt"
    )['pixel_values'].to(device)
    
    # Prepare prompt
    prompt = f"Question: {question} Answer:"
    inputs = model.tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=3
        )
    
    # Decode
    answer = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Remove the prompt from answer
    answer = answer.replace(prompt, "").strip()
    
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--question', type=str, required=True, help='Question to ask')
    parser.add_argument('--vision_encoder', type=str, default='openai/clip-vit-large-patch14')
    parser.add_argument('--language_decoder', type=str, default='Qwen/Qwen2-0.5B')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Create config
    config = ModelConfig(
        vision_encoder_name=args.vision_encoder,
        language_decoder_name=args.language_decoder
    )
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_trained_adapter(args.checkpoint, config)
    
    # Generate answer
    print(f"\nImage: {args.image}")
    print(f"Question: {args.question}")
    print("Generating answer...")
    
    answer = generate_answer(
        model,
        args.image,
        args.question,
        torch.device(args.device)
    )
    
    print(f"Answer: {answer}")


if __name__ == '__main__':
    main()
