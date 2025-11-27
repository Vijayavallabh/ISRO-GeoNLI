from unsloth import FastVisionModel
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False,  # keep vision frozen for caption-only
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,

    # LoRA hyperparameters (beefed up, but still safe for 8B)
    r = 8,                 # more capacity than 8, still lightweight
    lora_alpha = 16,        # >= r helps stability
    lora_dropout = 0,    # small dropout to reduce overfitting
    bias = "none",
    random_state = 3407,
    use_rslora = True,      # rank-stabilized LoRA for better training
    loftq_config = None,
    # target_modules = "all-linear",
)

from datasets import load_dataset
from PIL import Image

dataset = load_dataset("VRSBench", split = "train", streaming = True)

instruction = "Describe the content shown in the image in detail."

def convert_to_conversation(sample):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text",  "text": instruction},
                {"type": "image", "image": Image.open(
                    "VRSBench/Images_train/" + sample["image"]
                ).convert("RGB")},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": sample["caption"]},
            ],
        },
    ]
    return {"messages": conversation}

converted_dataset = [convert_to_conversation(sample) for sample in dataset]

from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
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
        output_dir = "outputs_vrsbench_cap",
        report_to = "none",
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 1024,   # enough for instruction + long captions
    ),
)

trainer_stats = trainer.train()

