"""Part B: TRL GRPOTrainer + LoRA smoke test, 5 steps, fp16, num_generations=8.

Reward = exact match with correct_answer (1/0).
Trains on the *train* split so val stays clean for later evaluation.
Appends a Part B section to reports/smoke_test.md.
"""
import argparse
import json
import math
import os
import sys
import time

import torch
from transformers import TrainerCallback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

TRAIN_SPLIT = "train"
MAX_COMPLETION = 24
NUM_GENERATIONS = 8
MAX_STEPS = 5
LR = 5e-5

# Language-side projections. The Qwen2.5-VL *vision* tower also contains
# gate_proj/up_proj/down_proj, so it must be excluded explicitly or LoRA
# would silently attach to the vision encoder.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
EXCLUDE_MODULES = r".*visual.*"


def load_train_samples(n, seed=C.SEED):
    """Spatial rows from the train split via the cached parquet pool.

    NOTE: `load_dataset("array/SAT", split="train", streaming=True)` does NOT work
    here -- SAT_train.parquet has a single 4.96 GiB row group and pyarrow raises
    `ArrowNotImplementedError: Nested data conversions not implemented for chunked
    array outputs`. See smoke/fetch_train_pool.py for the details and workaround.
    """
    from fetch_train_pool import fetch, sample

    pool = fetch(pool_size=max(256, n * 4))
    rows = []
    for row in sample(pool, n, seed=seed):
        imgs = row["image_bytes"]
        if not isinstance(imgs, list):
            imgs = [imgs]
        s = {
            "images": [C._to_pil(i) for i in imgs],
            "question": row["question"],
            "answers": list(row["answers"]),
            "question_type": row["question_type"],
            "correct_answer": row["correct_answer"],
        }
        rows.append(
            {
                "prompt": [{"role": "user", "content": C.build_prompt(s)}],
                "images": s["images"],
                "correct_answer": s["correct_answer"],
                "answers": s["answers"],
                "question_type": s["question_type"],
            }
        )
    return rows


def to_hf_dataset(rows):
    """TRL requires a `Dataset`/`IterableDataset`, not a list.

    Explicit features keep `images` as an Image list (decoded back to PIL on
    access) and keep the reward columns as plain strings.
    """
    from datasets import Dataset, Features, Value
    from datasets import Image as HFImage

    feats = Features(
        {
            "prompt": [{"role": Value("string"), "content": Value("string")}],
            "images": [HFImage()],
            "correct_answer": Value("string"),
            "answers": [Value("string")],
            "question_type": Value("string"),
        }
    )
    return Dataset.from_list(rows, features=feats)


def _text(completion):
    """Completions are conversational: [{"role": "assistant", "content": ...}]."""
    if isinstance(completion, list) and completion:
        c = completion[-1].get("content", "")
        if isinstance(c, list):
            return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        return c
    return completion if isinstance(completion, str) else ""


def exact_match_reward(completions, correct_answer, **kwargs):
    """1.0 iff the parsed choice equals correct_answer, else 0.0."""
    norm = lambda s: " ".join(str(s).lower().split()).strip(" .,!?:;\"'")
    out = []
    for comp, gold in zip(completions, correct_answer):
        raw = norm(_text(comp))
        out.append(1.0 if raw == norm(gold) else 0.0)
    return out


class StepRecorder(TrainerCallback):
    def __init__(self):
        self.steps = []
        self.nonfinite = []
        self._t0 = None

    def on_step_begin(self, args, state, control, **kw):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def on_log(self, args, state, control, logs=None, **kw):
        logs = logs or {}
        if "loss" not in logs and "reward" not in logs:
            return
        torch.cuda.synchronize()
        dt = (time.perf_counter() - self._t0) if self._t0 else float("nan")
        rec = {
            "step": state.global_step,
            "loss": logs.get("loss"),
            "grad_norm": logs.get("grad_norm"),
            "reward_mean": logs.get("reward"),
            "reward_std": logs.get("reward_std"),
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "seconds": dt,
            "completion_len": logs.get("completions/mean_length"),
        }
        self.steps.append(rec)

        bad = [k for k in ("loss", "grad_norm", "reward_mean", "reward_std")
               if isinstance(rec[k], (int, float)) and not math.isfinite(rec[k])]
        line = (f"[step {rec['step']}] loss={rec['loss']} grad_norm={rec['grad_norm']} "
                f"reward={rec['reward_mean']}±{rec['reward_std']} "
                f"peak={rec['peak_vram_gib']:.2f}GiB {dt:.1f}s")
        print(line, flush=True)
        if bad:
            self.nonfinite.append({"step": rec["step"], "fields": bad})
            print(f"!!! NaN/inf detected at step {rec['step']} in {bad} — STOPPING", flush=True)
            control.should_training_stop = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=MAX_STEPS)
    ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--report", default="reports/smoke_test.md")
    ap.add_argument("--no-append", action="store_true")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from trl import GRPOConfig, GRPOTrainer

    C.set_seed()
    print(f"GPU: {C.gpu_name()} ({C.capability()})  torch {torch.__version__}")

    rows = load_train_samples(args.n_train)
    train_ds = to_hf_dataset(rows)
    import collections as _c
    print(f"Loaded {len(rows)} train-split (spatial) samples, seed={C.SEED}")
    print("  question_type:", dict(_c.Counter(r["question_type"] for r in rows)))
    print("  n_images:", dict(_c.Counter(len(r["images"]) for r in rows)))

    processor = AutoProcessor.from_pretrained(C.MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        C.MODEL_ID, dtype=torch.float16, attn_implementation="sdpa"
    )

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
        exclude_modules=EXCLUDE_MODULES,
    )
    model = get_peft_model(model, lora)

    # ---- pre-training report -------------------------------------------------
    adapters = sorted({n.split(".lora_")[0].split(".")[-1] for n, _ in model.named_parameters() if "lora_" in n})
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    n_train_p = sum(p.numel() for _, p in trainable)
    n_all_p = sum(p.numel() for p in model.parameters())
    vis_trainable = [n for n, _ in trainable if "visual" in n]
    vis_total = sum(p.numel() for n, p in model.named_parameters() if "visual" in n)
    lora_layers = sorted({n.split(".lora_")[0] for n, _ in model.named_parameters() if "lora_" in n})
    n_lora_sites = len(lora_layers)

    pre = {
        "target_modules": TARGET_MODULES,
        "exclude_modules": EXCLUDE_MODULES,
        "adapter_module_types": adapters,
        "lora_injection_sites": n_lora_sites,
        "trainable_params": n_train_p,
        "all_params": n_all_p,
        "trainable_pct": 100 * n_train_p / n_all_p,
        "vision_trainable_params": len(vis_trainable),
        "vision_total_params": vis_total,
        "vision_frozen": len(vis_trainable) == 0,
        "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    }
    print("\n=== Pre-training ===")
    print(f"target_modules  : {TARGET_MODULES}")
    print(f"exclude_modules : {EXCLUDE_MODULES!r}")
    print(f"adapter types   : {adapters}")
    print(f"injection sites : {n_lora_sites}")
    print(f"trainable params: {n_train_p:,} / {n_all_p:,} ({pre['trainable_pct']:.4f}%)")
    print(f"vision encoder  : {'FROZEN' if pre['vision_frozen'] else 'NOT FROZEN'} "
          f"({vis_total:,} params, {len(vis_trainable)} trainable tensors)")
    print("====================\n")

    cfg = GRPOConfig(
        output_dir="outputs/partb",
        max_steps=args.steps,
        per_device_train_batch_size=NUM_GENERATIONS,
        gradient_accumulation_steps=1,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION,
        learning_rate=LR,
        fp16=True, bf16=False,           # sm_75: fp16 AMP only
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        seed=C.SEED,
        beta=0.0,                        # no reference model
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=exact_match_reward,
        args=cfg,
        train_dataset=train_ds,
    )

    rec = StepRecorder()
    trainer.add_callback(rec)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    trainer.train()
    wall = time.perf_counter() - t0

    # peak stats are reset per step, so the true peak is the max over steps
    overall_peak = max([s["peak_vram_gib"] for s in rec.steps], default=0.0)
    write_report(args.report, pre, rec, overall_peak, wall, args, append=not args.no_append)


def fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "inf"
        # GRPO losses sit near 0 (~1e-8); fixed-point would render them all as 0.0000
        if v != 0 and abs(v) < 10 ** -nd:
            return f"{v:.3e}"
        return f"{v:.{nd}f}"
    return str(v)


def write_report(path, pre, rec, overall_peak, wall, args, append=True):
    rows = "\n".join(
        f"| {s['step']} | {fmt(s['loss'])} | {fmt(s['grad_norm'])} | {fmt(s['reward_mean'],3)} | "
        f"{fmt(s['reward_std'],3)} | {s['peak_vram_gib']:.2f} | {s['seconds']:.1f} | {fmt(s['completion_len'],1)} |"
        for s in rec.steps
    )
    n_zero_grad = sum(1 for s in rec.steps if s.get("reward_std") == 0.0)
    nf = rec.nonfinite
    nf_line = ("None — all loss / grad_norm / reward values finite across every step."
               if not nf else
               "DETECTED, training stopped — " + "; ".join(f"step {x['step']}: {x['fields']}" for x in nf))

    sec = f"""
---

# Part B: GRPO Training (TRL GRPOTrainer + LoRA)

Trained on **`{TRAIN_SPLIT}`** split (val left untouched for later evaluation) ·
{len(rec.steps)} of {args.steps} steps · fp16 AMP · `num_generations={NUM_GENERATIONS}` ·
per_device_train_batch_size={NUM_GENERATIONS}, grad_accum=1 → **1 prompt × {NUM_GENERATIONS} generations per step**

TRL {__import__('trl').__version__} · PEFT {__import__('peft').__version__} · loss_type `dapo` · `beta=0.0` (no reference model) · prompts never truncated (TRL 1.13 has no `max_prompt_length`) · lr {LR} · max_completion_length {MAX_COMPLETION}

Reward: **exact match with `correct_answer` → 1.0 / 0.0**

## LoRA configuration

| | |
|---|---|
| `target_modules` | `{pre['target_modules']}` |
| `exclude_modules` | `{pre['exclude_modules']}` |
| adapter module types actually matched | `{pre['adapter_module_types']}` |
| LoRA injection sites | {pre['lora_injection_sites']} |
| r / alpha / dropout | {pre['lora_r']} / {pre['lora_alpha']} / {pre['lora_dropout']} |
| **trainable params** | **{pre['trainable_params']:,}** / {pre['all_params']:,} ({pre['trainable_pct']:.4f}%) |
| **vision encoder** | **{'FROZEN' if pre['vision_frozen'] else 'NOT FROZEN'}** — {pre['vision_total_params']:,} params, {pre['vision_trainable_params']} trainable tensors |

> The Qwen2.5-VL vision tower also contains `gate_proj` / `up_proj` / `down_proj`, so without
> `exclude_modules={pre['exclude_modules']!r}` LoRA would have attached to the vision encoder as well.

## Per-step metrics

| step | loss | grad_norm | reward mean | reward std | peak VRAM (GiB) | seconds | mean completion len |
|---|---|---|---|---|---|---|---|
{rows}

- **NaN/inf:** {nf_line}
- **Zero-gradient steps:** {n_zero_grad}/{len(rec.steps)} — steps where all {NUM_GENERATIONS} generations
  scored the same reward, so `reward_std=0`, advantages vanish and `grad_norm=0` (see note below).
- Overall peak VRAM: **{overall_peak:.2f} GiB** of {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GiB
- Total wall time for {len(rec.steps)} steps: **{wall:.1f}s** ({wall/max(len(rec.steps),1):.1f}s/step)

### Notes

- **Loss magnitude near zero is expected, not a failure.** On the first optimizer pass over each
  generation batch the policy equals the sampling policy, so the importance ratio is exactly 1 and
  the clipped DAPO objective starts at ~0. `grad_norm` is the useful signal here, and it is non-zero
  on every step that had reward variance.
- **{n_zero_grad} of {len(rec.steps)} steps produced no gradient at all.** GRPO normalizes advantages
  within each group, so when all {NUM_GENERATIONS} generations of a prompt receive the same binary
  reward the advantages are identically zero and the step is wasted (`frac_reward_zero_std=1`,
  `grad_norm=0`). With a 1/0 exact-match reward on binary yes/no questions this will happen often —
  worth addressing before a real run via a larger group, harder prompt sampling, or a shaped
  (partial-credit) reward.
- Peak VRAM is sampled per step (the CUDA peak counter is reset at each step boundary), so the
  figure above is the max across steps rather than a single cumulative reading.
"""
    mode = "a" if append and os.path.exists(path) else "w"
    with open(path, mode) as f:
        f.write(sec)
    os.makedirs("reports/raw", exist_ok=True)
    with open("reports/raw/part_b.json", "w") as f:
        json.dump({"pre": pre, "steps": rec.steps, "nonfinite": nf,
                   "overall_peak_gib": overall_peak, "wall_s": wall}, f, indent=2)
    print(f"\n{'Appended to' if mode=='a' else 'Wrote'} {path}")


if __name__ == "__main__":
    main()
