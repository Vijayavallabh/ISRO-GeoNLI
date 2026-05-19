"""
CLI entry point: ``geonli-run --config <path>``
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure repo root on path so adapters can import ISRO code
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from geonli.core.config import ExperimentConfig
from geonli.core.registry import get_vlm, get_segmenter, get_task, get_dataset
from geonli.core.pipeline_impl import DefaultGeoNLIPipeline
from geonli.prompts.manager import PromptManager


def build_pipeline_from_config(cfg: ExperimentConfig):
    """Instantiate models, tasks, and pipeline from an ExperimentConfig."""
    # -- Models -----------------------------------------------------------
    vlm_cfg = cfg.models.get("vlm")
    seg_cfg = cfg.models.get("segmenter")

    vlm_kwargs = {k: v for k, v in vlm_cfg.model_dump().items() if k != "name"}
    vlm_kwargs.update(vlm_cfg.extra)
    vlm = get_vlm(vlm_cfg.name, **vlm_kwargs) if vlm_cfg else None

    seg_kwargs = {k: v for k, v in seg_cfg.model_dump().items() if k != "name"}
    seg_kwargs.update(seg_cfg.extra)
    segmenter = get_segmenter(seg_cfg.name, **seg_kwargs) if seg_cfg else None

    # -- Tasks ------------------------------------------------------------
    tasks = []
    for t_cfg in cfg.tasks:
        if not t_cfg.enabled:
            continue
        t_kwargs = {k: v for k, v in t_cfg.model_dump().items() if k not in ("name", "enabled")}
        t_kwargs.pop("extra", None)
        if t_cfg.name == "captioning":
            tasks.append(get_task("captioning", vlm=vlm, **t_kwargs))
        elif t_cfg.name == "grounding":
            tasks.append(get_task("grounding", vlm=vlm, segmenter=segmenter, **t_kwargs))
        elif t_cfg.name == "vqa":
            tasks.append(get_task("vqa", vlm=vlm, segmenter=segmenter, **t_kwargs))
        else:
            # Generic lookup via registry
            tasks.append(get_task(t_cfg.name, vlm=vlm, segmenter=segmenter, **t_kwargs))

    return DefaultGeoNLIPipeline(tasks=tasks, vlm=vlm, segmenter=segmenter)


def run_inference(cfg: ExperimentConfig):
    pipeline = build_pipeline_from_config(cfg)
    dataset = None
    if cfg.dataset:
        ds_kwargs = {k: v for k, v in cfg.dataset.model_dump().items() if k != "name"}
        ds_kwargs.pop("extra", None)
        dataset = get_dataset(cfg.dataset.name, **ds_kwargs)

    if dataset is None:
        print("No dataset configured. Use --config with a dataset section.")
        sys.exit(1)

    out_dir = Path(cfg.output.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        image_id = sample["image_id"]
        print(f"[{idx+1}/{len(dataset)}] Processing {image_id} ...")
        results = pipeline.run(
            image=sample["image"],
            queries=sample["queries"],
            metadata=sample.get("metadata"),
        )
        serializable = {k: {"query": v.query, "response": _serialize(v.response), "metadata": v.metadata}
                        for k, v in results.items()}
        record = {
            "image_id": image_id,
            "metadata": sample.get("metadata", {}),
            "results": serializable,
        }
        all_results.append(record)

    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results -> {out_path}")


def _serialize(obj):
    """Make TaskResult.response JSON-serializable."""
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    if isinstance(obj, list) and obj and hasattr(obj[0], "__dict__"):
        return [o.__dict__ for o in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description="GeoNLI Inference Runner")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML/JSON config")
    parser.add_argument("--override", "-o", action="append", default=[], help="Key=value overrides")
    parser.add_argument("--prompt-dir", "-p", default=None, help="Path to custom prompt templates directory")
    args = parser.parse_args()

    # Load built-in + user prompt templates
    import geonli
    builtin_prompt_dir = os.path.join(os.path.dirname(geonli.__file__), "prompts", "templates")
    PromptManager(builtin_prompt_dir)
    if args.prompt_dir:
        PromptManager(args.prompt_dir)

    cfg_path = args.config
    if cfg_path.endswith(".yaml") or cfg_path.endswith(".yml"):
        cfg = ExperimentConfig.from_yaml(cfg_path)
    else:
        cfg = ExperimentConfig.from_json(cfg_path)

    # Simple dot-path overrides: key.subkey=value
    for ov in args.override:
        if "=" not in ov:
            continue
        key_path, val = ov.split("=", 1)
        _set_nested(cfg, key_path, val)

    run_inference(cfg)


def _set_nested(obj, key_path: str, val: str):
    """Crude override utility; for production use Hydra or OmegaConf."""
    keys = key_path.split(".")
    for k in keys[:-1]:
        obj = getattr(obj, k, None)
        if obj is None:
            return
    # Try int/float/bool, else str
    casted = val
    for caster in (int, float):
        try:
            casted = caster(val)
            break
        except ValueError:
            pass
    if val.lower() in ("true", "false"):
        casted = val.lower() == "true"
    setattr(obj, keys[-1], casted)


if __name__ == "__main__":
    main()
