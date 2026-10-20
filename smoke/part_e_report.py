"""Part E report: reasoning format with the instruction in the user message (E1) and
E1 + a "<description>" prefill (E2), for 3B and 7B.

Runs are produced by part_c_reasoning.py --mode e1/e2 with SMOKE_MODEL_ID and
SMOKE_RAW_DIR=reports/raw/{3b,7b}. This script reads those dumps (plus the Part C/D
reasoning dumps for comparison) and appends the Part E section to reports/smoke_test.md.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import part_c_reasoning as PC

RUNS = [  # (label, raw dir, mode)
    ("3B Part C", "reports/raw/3b", "reasoning"),
    ("3B-E1", "reports/raw/3b", "e1"),
    ("3B-E2", "reports/raw/3b", "e2"),
    ("7B Part D", "reports/raw/7b", "reasoning"),
    ("7B-E1", "reports/raw/7b", "e1"),
    ("7B-E2", "reports/raw/7b", "e2"),
]


def load(raw, mode):
    p = f"{raw}/part_c_{mode}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def counts(d):
    """Counts over every completion in the run (40 for 5 steps x 8 generations)."""
    texts = [(b.get("prefix", "") + t) for b in d["batches"] for t in b["texts"]]
    return {
        "n": len(texts),
        "all_tags": sum(1 for b in d["batches"] for x in b["all_tags"] if x),
        "has_answer": sum(1 for t in texts if "<answer>" in t),
        "cut": sum(1 for b in d["batches"] for x in b["truncated"] if x),
        "reward1": sum(1 for b in d["batches"] for x in b["rewards"] if x == 1.0),
    }


def status_line(d):
    if d is None:
        return "**not run** (an earlier run stopped)"
    if d["status"] == "ok":
        return f"completed {len(d['steps'])}/5 steps, no NaN/inf, no OOM"
    if d["status"] == "oom":
        return f"**STOPPED: out of memory** after {len(d['steps'])} logged steps — `{d['error']}`"
    return f"**STOPPED: NaN/inf** — {d['nonfinite']}"


def summary_table(data):
    labels = [l for l, _, _ in RUNS]
    S = {l: (PC.summary(data[l]) if data[l] and data[l]["steps"] else {}) for l in labels}
    K = {l: (counts(data[l]) if data[l] else {}) for l in labels}

    def row(name, fn):
        return f"| {name} | " + " | ".join(fn(l) for l in labels) + " |"

    def s(f, fmt):
        return lambda l: fmt(S[l][f]) if f in S[l] else "—"

    def k(f):
        return lambda l: f"{K[l][f]}/{K[l]['n']}" if K[l] else "—"

    out = ["| | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|",
           row("prompt", lambda l: {"reasoning": "system prompt", "e1": "user msg + example",
                                    "e2": "E1 + `<description>` prefill"}[dict((x, m) for x, _, m in RUNS)[l]]),
           row("seconds / step", s("s_step", lambda v: f"{v:.1f}")),
           row("generation s / step", s("gen", lambda v: f"{v:.1f}")),
           row("train fwd+bwd s / step", s("train", lambda v: f"{v:.1f}")),
           row("peak VRAM (GiB, max over steps)", s("peak", lambda v: f"{v:.2f}")),
           row("completion tokens: mean", s("len_mean", lambda v: f"{v:.1f}")),
           row("completion tokens: max", s("len_max", lambda v: f"{v}")),
           row("all three tags", k("all_tags")),
           row("contains `<answer>`", k("has_answer")),
           row("cut off at 1024", k("cut")),
           row("reward 1", k("reward1")),
           row("zero-gradient steps (reward std = 0)",
               lambda l: f"{S[l]['zero_std']}/{S[l]['n']}" if S[l] else "—")]
    return "\n".join(out)


def run_section(label, d):
    if d is None:
        return f"\n### {label}\n\n{status_line(d)}\n"
    k = counts(d)
    sm = PC.summary(d) if d["steps"] else {}
    sec = f"""
### {label}

Status: {status_line(d)}

- Out of {k['n']} completions: **{k['all_tags']}** have all three tags, **{k['has_answer']}** contain `<answer>`,
  **{k['cut']}** were cut off at 1024 tokens, {k['reward1']} got reward 1.
- Completion length: mean **{sm.get('len_mean', 0):.1f}**, max **{sm.get('len_max', 0)}** tokens.

{PC.step_table(d) if d['steps'] else '(no steps logged)'}

**Two full completions (unedited).** Same selection rule as Part C: the first completion with reward 1 and
the first with reward 0, scanning steps and generations in order (falls back to the next completions if one
kind doesn't exist).{' Shown exactly as generated. The prefilled `<description>` is part of the prompt, so it is not repeated here. Every completion below is preceded by it.' if d.get('batches') and d['batches'][0].get('prefix') else ''}

"""
    for step, i, bt in PC.pick_examples(d):
        sec += (f"**Step {step}, generation {i}** — reward {bt['rewards'][i]:.0f}, "
                f"{bt['lens'][i]} tokens, cut off: {'yes' if bt['truncated'][i] else 'no'}, "
                f"all three tags parsed: {'yes' if bt['all_tags'][i] else 'no'}, "
                f"correct_answer: `{bt['gold'][i]}`\n\n{PC.fence(bt['texts'][i])}\n\n")
    return sec


def write(path):
    data = {l: load(raw, m) for l, raw, m in RUNS}
    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    example_prompt = next((data[l]["prompt_example"] for l in ("3B-E1", "7B-E1") if data[l]), None)
    user_text = example_prompt[0]["content"] if example_prompt else "(E1 did not run)"

    sec = f"""
---

# Part E: getting the model to use the reasoning format

Part C (3B) and Part D (7B) showed that a system prompt alone barely changes the output: 3B used all three
tags in 2/40 completions and 7B in 0/40. Part E changes only the prompt and repeats the Part C Step 2 run
for both models. The samples, seed, LoRA, GRPO settings, `max_completion_length=1024`, reward (exact match
inside `<answer>` only, 0 if it doesn't parse, format not rewarded) and measurements are unchanged.
GPU: {C.gpu_name()} ({C.capability()}), {gpu_total:.1f} GiB. Each run is a fresh process.

**E1: instruction in the user message, no system prompt.** The format instruction plus one worked example
about a different question go at the end of the user message, after the question and choices. With no system
message, the Qwen2.5-VL chat template inserts its default `You are a helpful assistant.`, the same as in
Parts B/C baseline. The full user message for the first training prompt:

{PC.fence(user_text)}

**E2: E1 + `<description>` prefill.** The prompt is the same as E1, but every completion is forced to begin
with `<description>`. How this is implemented in TRL 1.13:

- `GRPOTrainer` always renders conversational prompts with `apply_chat_template(..., add_generation_prompt=True)`.
  A trailing assistant message therefore can't act as a prefill: it would be closed with `<|im_end|>`, and a
  new assistant turn would start after it.
- Without `tools`, the trainer passes `chat_template=None`, so it uses the processor's own template.
- Before building the trainer, the generation-prompt clause of `processor.chat_template` (and
  `processor.tokenizer.chat_template`) is changed from
  `{{% if add_generation_prompt %}}<|im_start|>assistant\\n{{% endif %}}` to
  `{{% if add_generation_prompt %}}<|im_start|>assistant\\n<description>{{% endif %}}`.
  The script asserts that the rendered prompt ends with `<|im_start|>assistant\\n<description>`. Tokenized, the
  prompt ends `<|im_start|> assistant \\n < description >`.
- The prefill tokens are therefore part of `prompt_ids`: the model generates from after them, they aren't in
  `completion_ids`, they don't count toward completion length or the 1024 limit, and they get no loss.
- Because the completion text TRL passes to the reward does not include the prefill, the reward function
  prepends `<description>` before parsing tags and `<answer>`. The tag counts below do the same.

## Summary: Part E next to the Part C / Part D reasoning runs

Counts are over all 40 completions per run (5 steps × 8 generations). Timings and VRAM are means over steps,
except peak VRAM, which is the maximum.

{summary_table(data)}

## Per run
"""
    for label in ("3B-E1", "3B-E2", "7B-E1", "7B-E2"):
        sec += run_section(label, data[label])
    with open(path, "a") as fh:
        fh.write(sec)
    print(f"Appended Part E to {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="reports/smoke_test.md")
    write(ap.parse_args().report)
