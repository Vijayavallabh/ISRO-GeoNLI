###############################################################################
#### WARNING: THIS CODE ONLY GENERATES RESPONSES AND DOES NOT EVALUATE ########
##############################################################################

import argparse
import os
import json
import csv
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from typing import List, Dict, Tuple
from tqdm import tqdm
from collections import defaultdict

MODEL_DIR = "./qwen2vl_lora_finetuned"
ANNOTATION_DIR = "VRS_val_Annotation"
IMAGE_DIR = "VRS_val_image"
OUTPUT_DIR = "VRS_val_Annotation_outputs"

# --- FIX: Helper to move dict to device safely ---
def move_to_device(batch, device):
    new_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            new_batch[k] = v.to(device)
        else:
            new_batch[k] = v
    return new_batch

def collate_batch(batch_data: List[Dict], processor) -> Dict:
    texts = [item["text"] for item in batch_data]

    # --- FIX: Load images here (Just-In-Time) to save RAM ---
    images = []
    for item in batch_data:
        img_path = item["image_path"] # Changed key from 'image' to 'image_path'
        images.append(Image.open(img_path).convert("RGB"))

    inputs = processor(
            text=texts,
            images=images,
            padding=True,
            min_pixels=256*28*28,  # For 512x512 images
            max_pixels=512*28*28,  # Prevents upscaling
            return_tensors="pt"
            )
    return inputs

import time
def run_batch_inference(processor, model, batch_data: List[Dict]) -> List[str]:
    if len(batch_data) == 0:
        return []

    # --- FIX: Move to device explicitly ---
    inputs = collate_batch(batch_data, processor)
    inputs = move_to_device(inputs, model.device)

    with torch.no_grad():
        outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id, # Explicitly set this
                do_sample=False,
                use_cache=True
                )

    results = []
    # --- FIX: Handle input lengths correctly for batch generation ---
    input_ids_len = inputs["input_ids"].shape[1]

    for i, output in enumerate(outputs):
        decoded = processor.tokenizer.decode(
                output[input_ids_len:], # Slice off the prompt
                skip_special_tokens=True
                )
        results.append(decoded.strip())

    return results

def prepare_prompt(text: str, processor) -> str:
    # Removed image arg, we just need text formatting here
    turn = [
            {
                "role": "user",
                "content": [{"type": "image"}] + [{"type": "text", "text": text}]
                }
            ]
    chat_text = processor.apply_chat_template(
            turn, tokenize=False, add_generation_prompt=True
            )
    return chat_text

def collect_tasks_by_type(ann_files: List[str], processor) -> Dict[int, List[Dict]]: 
    # --- FIX: Use defaultdict to prevent IndexError ---
    tasks = defaultdict(list)

    for ann_file in tqdm(ann_files, desc="Collecting tasks"):
        ann_path = os.path.join(ANNOTATION_DIR, ann_file)
        with open(ann_path, 'r') as f:
            ann = json.load(f)

        # --- FIX: Store PATH, not Image object (RAM fix) ---
        image_path = os.path.join(IMAGE_DIR, ann["image"])

        # Caption task (ID 0)
        caption_text = prepare_prompt("Describe this image in detail.", processor)
        tasks[0].append({
            "file_name": ann_file,
            "image_path": image_path, # Store path
            "text": caption_text,
            "gt": ann.get("caption", []),
            })

        # QA tasks
        qa_pairs = ann.get("qa_pairs", [])
        for qa in qa_pairs:
            qa_text = prepare_prompt(qa["question"], processor)
            # Use ques_id as key
            tasks[qa["ques_id"]].append({
                "file_name": ann_file,
                "image_path": image_path, # Store path
                "text": qa_text,
                "gt": qa["answer"],
                })

    return tasks

def process_task_type_batch(
        task_id: int,
        tasks: List[Dict], 
        processor,
        model,
        batch_size: int
        ) -> List[Dict]:

    results = []
    if task_id == 0: print("processing caption")
    else : print(f"processing ques_id {task_id}")

    for i in tqdm(range(0, len(tasks), batch_size)):
        batch = tasks[i:i + batch_size]
        # --- FIX: Pass 'image_path' ---
        batch_data = [{"image_path": task["image_path"], "text": task["text"]} for task in batch]

        predictions = run_batch_inference(processor, model, batch_data)


        for task, pred in zip(batch, predictions):
            results.append({
                "file_name": task["file_name"],
                "task_id": task_id,
                "prediction": pred,                
                })

    return results

def organize_results(all_results: List[Dict]) -> Tuple[Dict, Dict]:
    outputs_dict = defaultdict(lambda: {"caption_pred": None, "qa_pairs": []})
    #scores_dict = defaultdict(lambda: {"caption_score": None, "qa_scores": []})

    for result in all_results:
        file_name = result["file_name"]

        if result["task_id"] == 0:
            outputs_dict[file_name]["caption_pred"] = result["prediction"]
            #scores_dict[file_name]["caption_score"] = result["score"]
        else:
            outputs_dict[file_name]["qa_pairs"].append({
                "ques_id": result["task_id"], # --- FIX: Access task_id, not ques_id ---
                "pred_answer": result["prediction"],
                })
            '''
            #scores_dict[file_name]["qa_scores"].append({
                "ques_id": result["task_id"], # --- FIX: Access task_id ---
                "score": result["score"],
                })
            '''    
    # Sorting logic remains valid
    for file_name in outputs_dict:
        outputs_dict[file_name]["qa_pairs"].sort(key=lambda x: x["ques_id"])
    '''
    for file_name in scores_dict:
        scores_dict[file_name]["qa_scores"].sort(key=lambda x: x["ques_id"])
    '''
    return dict(outputs_dict)

# ... (update_output_json and write_scores_to_csv remain mostly correct) ...
# Just ensuring imports are clean for the bottom part
def update_output_json(file_name: str, outputs: Dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, file_name.replace(".json", "_out.json"))
    ann_path = os.path.join(ANNOTATION_DIR, file_name)

    with open(ann_path, 'r') as f: ann = json.load(f)

    if os.path.exists(output_file):
        with open(output_file, 'r') as f: output_ann = json.load(f)
    else:
        output_ann = ann.copy()

    # Handle caption safely
    if outputs["caption_pred"]:
        output_ann["caption"] = outputs["caption_pred"]

    existing_preds = {qa.get("ques_id"): qa for qa in output_ann.get("qa_pairs", [])}
    for qa_pred in outputs["qa_pairs"]:
        ques_id = qa_pred["ques_id"]
        if ques_id in existing_preds:
            existing_preds[ques_id]["answer"] = qa_pred["pred_answer"]
        else:
            # Handle case where output JSON didn't have this QA pair originally
            existing_preds[ques_id] = {"ques_id": ques_id, "question": "Unknown", "answer": qa_pred["pred_answer"]}

    output_ann["qa_pairs"] = sorted(existing_preds.values(), key=lambda x: x["ques_id"])

    with open(output_file, 'w') as f:
        json.dump(output_ann, f, indent=2)

def write_scores_to_csv(scores_dict: Dict[str, Dict], output_csv: str):
    max_qa = 0
    for scores in scores_dict.values():
        if scores.get("qa_scores"):
            max_qa = max(max_qa, max(qa["ques_id"] for qa in scores["qa_scores"]))

    headers = ["file_name", "caption_score"] + [f"qa_{i}" for i in range(max_qa + 1)] # +1 to include max_qa

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for file_name, scores in sorted(scores_dict.items()):
            row = {
                    "file_name": file_name,
                    "caption_score": scores.get("caption_score", "")
                    }
            qa_scores_map = {qa["ques_id"] : qa["score"] for qa in scores.get("qa_scores", [])}
            for i in range(max_qa + 1):
                row[f"qa_{i}"] = qa_scores_map.get(i, "")
            writer.writerow(row)
    print(f"Scores written to {output_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    # parser.add_argument("--output_csv", type=str, default="evaluation_scores.csv")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    device_map = {"": local_rank} if local_rank != -1 else "auto"

    print(f"Loading model on device_map: {device_map}...")

    processor = AutoProcessor.from_pretrained(MODEL_DIR, padding_side="left")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_DIR, torch_dtype=torch.bfloat16, device_map=device_map, attn_implementation="flash_attention_2"
            ).eval()


    # evaluator = GeoNLIEvaluator()
    all_ann_files = [f for f in os.listdir(ANNOTATION_DIR) if f.endswith('.json')]
    all_ann_files.sort()
    ann_files = all_ann_files[global_rank::world_size]

    print(f"[GPU {global_rank}] Processing {len(ann_files)} / {len(all_ann_files)} files")

    # 1. Collect
    tasks_dict = collect_tasks_by_type(ann_files, processor)
    print("Tasks collected!")

    all_results = []
    # --- FIX: Iterate over Dictionary items, not range(len) ---
    for task_id, task_list in tasks_dict.items():
        results = process_task_type_batch(
                task_id,
                task_list, 
                processor,
                model,              
                args.batch_size
                )
        all_results.extend(results)

    print("\nOrganizing results...")
    outputs_dict = organize_results(all_results)

    print("Updating JSONs...")
    for file_name, outputs in tqdm(outputs_dict.items()):
        update_output_json(file_name, outputs)

    '''
    # write_scores_to_csv(scores_dict, args.output_csv)

    # Summary
    caption_scores = [s["caption_score"] for s in scores_dict.values() if s.get("caption_score") is not None]
    if caption_scores:
        print(f"Caption Avg: {sum(caption_scores) / len(caption_scores):.4f}")
    '''
if __name__ == "__main__":
    main()
