"""Part C: GRPO cost with a long, reasoning-style completion format.

Same config as Part B (Qwen2.5-VL-3B, TRL GRPOTrainer, LoRA on the LLM decoder,
vision frozen, fp16, 5 steps, 1 prompt x 8 generations, same train samples/seed).
Only the output format changes between the two modes:

  --mode baseline   Part B exactly: short answer, max_completion_length=24,
                    reward = exact match of the whole completion.
  --mode reasoning  system prompt asks for <description>/<reasoning>/<answer>,
                    max_completion_length=1024, reward = exact match of the text
                    inside <answer> (0 if it can't be parsed). Format is NOT rewarded.

Each mode runs in its own process and dumps reports/raw/part_c_<mode>.json.
`--write-report` then appends the Part C section from both dumps.

Goal is time and memory per step, not accuracy.
"""
import argparse
import json
import math
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import part_b_grpo as B

RAW = C.RAW_DIR
MAX_COMPLETION = {"baseline": B.MAX_COMPLETION, "reasoning": 1024}

SYSTEM_PROMPT = (
    "You are a careful visual spatial reasoner. Always respond in exactly this format:\n"
    "<description>what is in the image relevant to the question</description>\n"
    "<reasoning>step-by-step spatial reasoning</reasoning>\n"
    "<answer>one of the given choices</answer>\n"
    "The text inside <answer> must be exactly one of the given choices, copied verbatim."
)

TAGS = ("description", "reasoning", "answer")
TAG_RE = {t: re.compile(rf"<{t}>(.*?)</{t}>", re.DOTALL) for t in TAGS}


def build_prompt_reasoning(sample):
    # Part B's prompt ends with "Reply with the choice text only, no explanation.",
    # which contradicts the reasoning format, so that sentence is dropped here.
    choices = "\n".join(f"- {c}" for c in sample["answers"])
    return (
        f"{sample['question']}\n\nChoices:\n{choices}\n\n"
        "Answer with exactly one choice from the list above."
    )


def load_rows(mode, n):
    rows = B.load_train_samples(n)
    if mode == "reasoning":
        for r in rows:
            # recover the raw question from Part B's prompt, then rebuild
            q = r["prompt"][0]["content"].split("\n\nChoices:\n")[0]
            r["prompt"] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt_reasoning({"question": q, "answers": r["answers"]})},
            ]
    return rows


def norm(s):
    return " ".join(str(s).lower().split()).strip(" .,!?:;\"'")


class Probe:
    """Collects per-generation-batch completion stats from inside the reward function."""

    def __init__(self, mode, eos_ids):
        self.mode = mode
        self.eos_ids = set(eos_ids)
        self.batches = []

    def reward(self, completions, correct_answer, completion_ids=None, **kwargs):
        texts = [B._text(c) for c in completions]
        lens = [len(ids) for ids in completion_ids]
        trunc = [len(ids) >= MAX_COMPLETION[self.mode] and (not ids or ids[-1] not in self.eos_ids)
                 for ids in completion_ids]
        rewards, all_tags, answers = [], [], []
        for t, gold in zip(texts, correct_answer):
            if self.mode == "baseline":
                ans = t
                all_tags.append(None)
            else:
                found = {k: TAG_RE[k].search(t) for k in TAGS}
                all_tags.append(all(found.values()))
                ans = found["answer"].group(1) if found["answer"] else None
            answers.append(ans)
            rewards.append(1.0 if ans is not None and norm(ans) == norm(gold) else 0.0)
        self.batches.append({
            "texts": texts, "lens": lens, "truncated": trunc, "all_tags": all_tags,
            "answers": answers, "rewards": rewards, "gold": list(correct_answer),
        })
        return rewards


class TimedGRPOTrainer:
    """Mixin factory: wraps GRPOTrainer to time generation vs. the training forward/backward."""

    @staticmethod
    def make(base):
        class T(base):
            phase = None

            def _reset(self):
                self.phase = {"gen_s": 0.0, "score_s": 0.0, "train_s": 0.0,
                              "gen_peak": 0.0, "train_peak": 0.0}

            def _generate(self, prompts):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                out = super()._generate(prompts)
                torch.cuda.synchronize()
                self.phase["gen_s"] += time.perf_counter() - t0
                self.phase["gen_peak"] = max(self.phase["gen_peak"], torch.cuda.max_memory_allocated() / 1024**3)
                return out

            def _generate_and_score_completions(self, inputs):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = super()._generate_and_score_completions(inputs)
                torch.cuda.synchronize()
                # everything in here that isn't _generate: tokenisation, rewards, advantages
                self.phase["score_s"] += time.perf_counter() - t0
                return out

            def training_step(self, model, inputs, num_items_in_batch):
                if self.phase is None:
                    self._reset()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                score_before = self.phase["score_s"]
                out = super().training_step(model, inputs, num_items_in_batch)
                torch.cuda.synchronize()
                total = time.perf_counter() - t0
                # training_step = _prepare_inputs (generate+score) + forward + backward
                self.phase["train_s"] += total - (self.phase["score_s"] - score_before)
                return out

        return T


class Recorder(B.StepRecorder):
    def __init__(self, trainer_ref):
        super().__init__()
        self.tr = trainer_ref

    def on_step_begin(self, args, state, control, **kw):
        self.tr[0]._reset()
        super().on_step_begin(args, state, control, **kw)

    def on_log(self, args, state, control, logs=None, **kw):
        n = len(self.steps)
        super().on_log(args, state, control, logs=logs, **kw)
        if len(self.steps) > n:
            ph = self.tr[0].phase
            s = self.steps[-1]
            s["gen_s"] = ph["gen_s"]
            s["score_s"] = ph["score_s"] - ph["gen_s"]
            s["train_s"] = ph["train_s"]
            s["gen_peak_gib"] = ph["gen_peak"]
            s["trl"] = {k: (logs or {}).get(k) for k in (
                "completions/mean_length", "completions/max_length", "completions/clipped_ratio",
                "frac_reward_zero_std", "step_time")}


def run(mode, steps, n_train):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from trl import GRPOConfig, GRPOTrainer

    C.set_seed()
    rows = load_rows(mode, n_train)
    train_ds = B.to_hf_dataset(rows)

    processor = AutoProcessor.from_pretrained(C.MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        C.MODEL_ID, dtype=torch.float16, attn_implementation="sdpa"
    )
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=B.TARGET_MODULES, exclude_modules=B.EXCLUDE_MODULES,
    ))
    assert not any(p.requires_grad for n, p in model.named_parameters() if "visual" in n)

    tok = processor.tokenizer
    eos = {tok.eos_token_id, tok.pad_token_id, tok.convert_tokens_to_ids("<|im_end|>")} - {None}
    probe = Probe(mode, eos)

    cfg = GRPOConfig(
        output_dir=f"outputs/partc_{mode}",
        max_steps=steps,
        per_device_train_batch_size=B.NUM_GENERATIONS,
        gradient_accumulation_steps=1,
        num_generations=B.NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION[mode],
        learning_rate=B.LR,
        fp16=True, bf16=False,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        seed=C.SEED,
        beta=0.0,
    )
    # plain function so TRL logs it under the same reward name as Part B
    def exact_match_reward(completions, correct_answer, completion_ids=None, **kw):
        return probe.reward(completions, correct_answer, completion_ids=completion_ids, **kw)

    Trainer = TimedGRPOTrainer.make(GRPOTrainer)
    trainer = Trainer(model=model, processing_class=processor, reward_funcs=exact_match_reward,
                      args=cfg, train_dataset=train_ds)
    ref = [trainer]
    rec = Recorder(ref)
    trainer.add_callback(rec)

    status, err = "ok", None
    t0 = time.perf_counter()
    try:
        trainer.train()
    except torch.OutOfMemoryError as e:
        status, err = "oom", str(e).splitlines()[0]
        print(f"!!! OOM — STOPPING: {err}", flush=True)
    wall = time.perf_counter() - t0
    if rec.nonfinite:
        status = "nonfinite"

    # pair each logged step with its generation batch (one generation per step here)
    for s, b in zip(rec.steps, probe.batches):
        s["len_mean"] = sum(b["lens"]) / len(b["lens"])
        s["len_max"] = max(b["lens"])
        s["trunc_frac"] = sum(b["truncated"]) / len(b["truncated"])
        s["tags_frac"] = (None if mode == "baseline"
                          else sum(b["all_tags"]) / len(b["all_tags"]))
        s["gold"] = b["gold"][0]

    os.makedirs(RAW, exist_ok=True)
    out = {"mode": mode, "status": status, "error": err, "wall_s": wall,
           "max_completion_length": MAX_COMPLETION[mode],
           "steps": rec.steps, "nonfinite": rec.nonfinite, "batches": probe.batches,
           "prompt_example": rows[0]["prompt"]}
    with open(f"{RAW}/part_c_{mode}.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[{mode}] status={status} wall={wall:.1f}s -> {RAW}/part_c_{mode}.json")
    return status


# ---------------------------------------------------------------------------------- report

def f(v, nd=2):
    return B.fmt(v, nd) if v is not None else "—"


def step_table(d):
    head = ("| step | s/step | gen s | score s | train (fwd+bwd) s | peak VRAM GiB | gen peak GiB | "
            "len mean | len max | cut off @max | all tags | reward mean | reward std | loss | grad_norm |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines = []
    for s in d["steps"]:
        tags = "n/a" if s["tags_frac"] is None else f"{s['tags_frac']:.3f}"
        lines.append(
            f"| {s['step']} | {s['seconds']:.1f} | {s['gen_s']:.1f} | {s['score_s']:.2f} | {s['train_s']:.1f} | "
            f"{s['peak_vram_gib']:.2f} | {s['gen_peak_gib']:.2f} | {s['len_mean']:.1f} | {s['len_max']} | "
            f"{s['trunc_frac']:.3f} | {tags} | "
            f"{f(s['reward_mean'],3)} | {f(s['reward_std'],3)} | {f(s['loss'],4)} | {f(s['grad_norm'],4)} |")
    return head + "\n" + "\n".join(lines)


def summary(d):
    st = d["steps"]
    if not st:
        return {}
    n = len(st)
    return {
        "n": n,
        "s_step": sum(s["seconds"] for s in st) / n,
        "gen": sum(s["gen_s"] for s in st) / n,
        "train": sum(s["train_s"] for s in st) / n,
        "peak": max(s["peak_vram_gib"] for s in st),
        "len_mean": sum(s["len_mean"] for s in st) / n,
        "len_max": max(s["len_max"] for s in st),
        "trunc": sum(s["trunc_frac"] for s in st) / n,
        "zero_std": sum(1 for s in st if s["reward_std"] == 0.0),
    }


def fence(text):
    ticks = "```"
    while ticks in text:
        ticks += "`"
    return f"{ticks}text\n{text}\n{ticks}"


def pick_examples(d):
    """First completion with reward 1 and first with reward 0, scanning steps in order."""
    flat = [(bi + 1, i, b) for bi, b in enumerate(d["batches"]) for i in range(len(b["texts"]))]
    good = next((x for x in flat if x[2]["rewards"][x[1]] == 1.0), None)
    bad = next((x for x in flat if x[2]["rewards"][x[1]] == 0.0), None)
    picked = [x for x in (good, bad) if x is not None]
    for x in flat:
        if len(picked) >= 2:
            break
        if x not in picked:
            picked.append(x)
    return picked


def write_report(path):
    b = json.load(open(f"{RAW}/part_c_baseline.json"))
    r = json.load(open(f"{RAW}/part_c_reasoning.json"))
    sb, sr = summary(b), summary(r)
    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    same_prompts = [s["gold"] for s in b["steps"]] == [s["gold"] for s in r["steps"]]

    def status_line(d):
        if d["status"] == "ok":
            return f"completed {len(d['steps'])}/5 steps, no NaN/inf, no OOM"
        if d["status"] == "oom":
            return f"**STOPPED: out of memory** after {len(d['steps'])} logged steps — `{d['error']}`"
        return f"**STOPPED: NaN/inf** — {d['nonfinite']}"

    def ratio(a, c):
        return f"{c / a:.1f}×" if a else "—"

    ex = []
    for step, i, bt in pick_examples(r):
        ex.append(
            f"**Step {step}, generation {i}** — reward {bt['rewards'][i]:.0f}, "
            f"{bt['lens'][i]} tokens, cut off: {'yes' if bt['truncated'][i] else 'no'}, "
            f"all three tags parsed: {'yes' if bt['all_tags'][i] else 'no'}, "
            f"correct_answer: `{bt['gold'][i]}`\n\n{fence(bt['texts'][i])}\n")

    sec = f"""
---

# Part C: GRPO cost with a reasoning-style output format

Goal: **time and memory per step when completions are long** — not accuracy.

Same setup as Part B in both runs: Qwen2.5-VL-3B-Instruct · TRL {__import__('trl').__version__} GRPOTrainer ·
LoRA r16 on the LLM decoder only (vision encoder frozen) · fp16 AMP · 5 steps ·
per_device_train_batch_size=8, grad_accum=1, `num_generations=8` (1 prompt × 8 generations per step) ·
lr {B.LR} · `beta=0.0` · seed {C.SEED} · same train-split samples. Both runs were executed in this session,
each in a fresh process, on {C.gpu_name()} ({C.capability()}) in the project venv `.venv/` (torch {torch.__version__}).
The step prompts were identical across the two runs: **{'yes' if same_prompts else 'NO'}**
(checked by comparing each step's `correct_answer`).

| | Step 1: baseline (Part B rerun) | Step 2: reasoning format |
|---|---|---|
| prompt | Part B prompt (no system prompt) | system prompt below + Part B prompt without its last sentence |
| `max_completion_length` | {b['max_completion_length']} | {r['max_completion_length']} |
| reward | exact match of the whole completion, 1/0 | exact match of the text inside `<answer>`, 1/0; 0 if `<answer>` doesn't parse; format itself not rewarded |
| status | {status_line(b)} | {status_line(r)} |

Step 2 system prompt:

{fence(SYSTEM_PROMPT)}

Part B's user prompt ends with *"Reply with the choice text only, no explanation."*, which contradicts the
reasoning format, so in Step 2 that one sentence is removed; the question and choice list are unchanged.

## Summary (mean over steps)

| | baseline | reasoning | ratio |
|---|---|---|---|
| seconds / step | {sb.get('s_step', 0):.1f} | {sr.get('s_step', 0):.1f} | {ratio(sb.get('s_step'), sr.get('s_step', 0))} |
| generation s / step | {sb.get('gen', 0):.1f} | {sr.get('gen', 0):.1f} | {ratio(sb.get('gen'), sr.get('gen', 0))} |
| train fwd+bwd s / step | {sb.get('train', 0):.1f} | {sr.get('train', 0):.1f} | {ratio(sb.get('train'), sr.get('train', 0))} |
| peak VRAM (GiB, max over steps) | {sb.get('peak', 0):.2f} | {sr.get('peak', 0):.2f} | of {gpu_total:.1f} GiB |
| completion tokens: mean / max | {sb.get('len_mean', 0):.1f} / {sb.get('len_max', 0)} | {sr.get('len_mean', 0):.1f} / {sr.get('len_max', 0)} | |
| fraction cut off at max length | {sb.get('trunc', 0):.3f} | {sr.get('trunc', 0):.3f} | |
| zero-gradient steps (reward std = 0) | {sb.get('zero_std', 0)}/{sb.get('n', 0)} | {sr.get('zero_std', 0)}/{sr.get('n', 0)} | |

## Per step — Step 1: baseline (Part B rerun)

{step_table(b)}

## Per step — Step 2: reasoning format

{step_table(r)}

**How the columns are measured.** `s/step` is wall time from `on_step_begin` to the step's log, as in Part B.
`gen s` is time inside TRL's `_generate` (sampling all 8 completions). `score s` is the rest of
`_generate_and_score_completions` (tokenisation, reward, advantages). `train (fwd+bwd) s` is `training_step` minus
those two, i.e. the policy forward + backward. The optimizer step and logging are the small remainder.
Peak VRAM is the CUDA allocator peak for the whole step; `gen peak` is the peak during generation only.
`cut off @max` counts completions that hit `max_completion_length` without an end-of-turn token. `all tags`
is the fraction where `<description>…</description>`, `<reasoning>…</reasoning>` and `<answer>…</answer>` all parse.
Token lengths come from the `completion_ids` TRL passes to the reward function.

## Two full Step 2 completions (unedited)

Selection rule, fixed before looking: the first completion with reward 1 and the first with reward 0,
scanning steps and generations in order (falls back to the next completions if one kind doesn't exist).

{chr(10).join(ex)}"""
    with open(path, "a") as fh:
        fh.write(sec)
    print(f"Appended Part C to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "reasoning"])
    ap.add_argument("--steps", type=int, default=B.MAX_STEPS)
    ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--write-report", action="store_true")
    ap.add_argument("--report", default="reports/smoke_test.md")
    a = ap.parse_args()
    if a.write_report:
        write_report(a.report)
        return
    status = run(a.mode, a.steps, a.n_train)
    sys.exit(0 if status == "ok" else 3)


if __name__ == "__main__":
    main()
