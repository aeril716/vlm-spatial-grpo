"""Part D report: Part A + Part C repeated with Qwen2.5-VL-7B-Instruct.

The runs themselves reuse the Part A / Part C scripts unchanged, with
SMOKE_MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct and SMOKE_RAW_DIR=reports/raw/7b.
This script only reads those dumps (plus the 3B Part C dumps in reports/raw/3b)
and appends the Part D section to reports/smoke_test.md.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import part_c_reasoning as PC

MODEL_7B = "Qwen/Qwen2.5-VL-7B-Instruct"


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def step0(a):
    n = len(a["fp16"])
    has32 = bool(a.get("fp32"))
    nonfinite = sum(1 for r in a["fp16"] if r["nonfinite_logits"])
    acc16 = sum(1 for r in a["fp16"] if r["parsed"] == r["correct_answer"])
    ok16 = sum(1 for r in a["fp16"] if r["parse_ok"])
    pk, sec = a["peak_gib"], a["seconds"]
    if has32:
        agree = sum(1 for x, y in zip(a["fp16"], a["fp32"]) if x["parsed"] == y["parsed"])
        acc32 = sum(1 for r in a["fp32"] if r["parsed"] == r["correct_answer"])
        ok32 = sum(1 for r in a["fp32"] if r["parse_ok"])
        col32 = [f"{pk['fp32']:.2f}", f"{sec['fp32']:.1f}", f"{sec['fp32']/n:.2f}", f"{ok32}/{n}", f"{acc32}/{n}"]
        agree_line = f"- **fp16 vs fp32 agreement: {agree}/{n}**"
    else:
        col32 = ["skipped"] * 5
        agree_line = "- **fp32 skipped: it did not fit in GPU memory**"
    col16 = [f"{pk['fp16']:.2f}", f"{sec['fp16']:.1f}", f"{sec['fp16']/n:.2f}", f"{ok16}/{n}", f"{acc16}/{n}"]
    names = ["peak VRAM (GiB)", "total time (s)", "seconds / sample", "parse success", "matches correct_answer"]
    rows = "\n".join(f"| {m} | {a16} | {a32} |" for m, a16, a32 in zip(names, col16, col32))

    per = []
    for i, x in enumerate(a["fp16"]):
        y = a["fp32"][i] if has32 else None
        per.append(f"| {i} | {x['question_type']} | {x['n_images']} | {x['parsed'] or '—'} | "
                   f"{'—' if y is None else (y['parsed'] or '—')} | {x['correct_answer']} | "
                   f"{'yes' if x['nonfinite_logits'] else 'no'} | "
                   f"{'' if y is None else ('=' if x['parsed'] == y['parsed'] else 'DIFF')} |")
    return f"""## Step 0: fp16 sanity check (as Part A)

Same 20 SAT `val` samples (streaming, shuffle seed {C.SEED}), greedy, max_new_tokens=24, SDPA.

| metric | fp16 | fp32 (reference) |
|---|---|---|
{rows}

{agree_line}
- **fp16 non-finite (NaN/inf) logits: {nonfinite}/{n} sample(s)**

| # | question_type | imgs | fp16 answer | fp32 answer | correct_answer | fp16 NaN/inf | agree |
|---|---|---|---|---|---|---|---|
{chr(10).join(per)}
""", nonfinite


def summary_table(b7, r7, b3, r3):
    S = {k: PC.summary(d) if d else {} for k, d in
         (("b3", b3), ("b7", b7), ("r3", r3), ("r7", r7))}

    def g(k, f, fmt):
        v = S[k].get(f)
        return "—" if v is None else fmt(v)

    lines = [
        ("seconds / step", "s_step", lambda v: f"{v:.1f}"),
        ("generation s / step", "gen", lambda v: f"{v:.1f}"),
        ("train fwd+bwd s / step", "train", lambda v: f"{v:.1f}"),
        ("peak VRAM (GiB, max over steps)", "peak", lambda v: f"{v:.2f}"),
        ("completion tokens: mean", "len_mean", lambda v: f"{v:.1f}"),
        ("completion tokens: max", "len_max", lambda v: f"{v}"),
        ("fraction cut off at max length", "trunc", lambda v: f"{v:.3f}"),
    ]
    out = ["| | baseline 3B (Part C) | baseline 7B | reasoning 3B (Part C) | reasoning 7B |",
           "|---|---|---|---|---|"]
    for name, f, fmt in lines:
        out.append(f"| {name} | {g('b3', f, fmt)} | {g('b7', f, fmt)} | {g('r3', f, fmt)} | {g('r7', f, fmt)} |")
    out.append("| zero-gradient steps (reward std = 0) | " + " | ".join(
        f"{S[k]['zero_std']}/{S[k]['n']}" if S[k] else "—" for k in ("b3", "b7", "r3", "r7")) + " |")
    return "\n".join(out)


def tag_counts(r):
    texts = [t for b in r["batches"] for t in b["texts"]]
    all_tags = sum(1 for b in r["batches"] for x in b["all_tags"] if x)
    starts = sum(1 for t in texts if t.lstrip().startswith("<description>"))
    return len(texts), all_tags, starts


def status_line(d):
    if d is None:
        return "not run"
    if d["status"] == "ok":
        return f"completed {len(d['steps'])}/5 steps, no NaN/inf, no OOM"
    if d["status"] == "oom":
        return f"**STOPPED: out of memory** after {len(d['steps'])} logged steps — `{d['error']}`"
    return f"**STOPPED: NaN/inf** — {d['nonfinite']}"


def write(path, raw7, raw3):
    a = load(f"{raw7}/part_a.json")
    b7, r7 = load(f"{raw7}/part_c_baseline.json"), load(f"{raw7}/part_c_reasoning.json")
    b3, r3 = load(f"{raw3}/part_c_baseline.json"), load(f"{raw3}/part_c_reasoning.json")
    s0, _ = step0(a)
    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    same = (b7 and r7 and b3 and
            [s["gold"] for s in b7["steps"]] == [s["gold"] for s in b3["steps"]] ==
            [s["gold"] for s in r7["steps"]])

    sec = f"""
---

# Part D: Part A + Part C with Qwen2.5-VL-7B-Instruct

Only the model changes: **{MODEL_7B}** instead of Qwen2.5-VL-3B-Instruct. Everything else is identical to
Part C: the project venv `.venv/` (torch {torch.__version__}, TRL {__import__('trl').__version__}), same samples, seed {C.SEED}, LoRA config
(r16, decoder only, `exclude_modules='.*visual.*'`), GRPO settings, prompts, system prompt, reward and
measurements. The scripts are the Part A / Part C scripts, run unchanged with `SMOKE_MODEL_ID` set to the 7B
model. GPU: {C.gpu_name()} ({C.capability()}), {gpu_total:.1f} GiB.

{s0}
## Steps 1 and 2: GRPO, 5 steps each

| | Step 1: baseline (short answer, max 24) | Step 2: reasoning format (max 1024) |
|---|---|---|
| status | {status_line(b7)} | {status_line(r7)} |

Each step's prompt was the same as in the 3B Part C runs: **{'yes' if same else 'NO'}**
(checked by comparing each step's `correct_answer`).

### Summary (mean over steps), 3B vs 7B

{summary_table(b7, r7, b3, r3)}
"""
    if b7 and b7["steps"]:
        sec += f"\n### Per step — Step 1: baseline, 7B\n\n{PC.step_table(b7)}\n"
    if r7 and r7["steps"]:
        n, nt, ns = tag_counts(r7)
        sec += f"\n### Per step — Step 2: reasoning format, 7B\n\n{PC.step_table(r7)}\n"
        sec += f"""
Columns are measured exactly as in Part C (see the note under Part C's per-step tables).

### Step 2 format use (7B)

- Completions with all three tags parsed: **{nt}/{n}**
- Completions that start with `<description>` (ignoring leading whitespace): **{ns}/{n}**

### Two full Step 2 completions, 7B (unedited)

Same selection rule as Part C: the first completion with reward 1 and the first with reward 0,
scanning steps and generations in order (falls back to the next completions if one kind doesn't exist).

"""
        for step, i, bt in PC.pick_examples(r7):
            sec += (f"**Step {step}, generation {i}** — reward {bt['rewards'][i]:.0f}, "
                    f"{bt['lens'][i]} tokens, cut off: {'yes' if bt['truncated'][i] else 'no'}, "
                    f"all three tags parsed: {'yes' if bt['all_tags'][i] else 'no'}, "
                    f"correct_answer: `{bt['gold'][i]}`\n\n{PC.fence(bt['texts'][i])}\n\n")
    with open(path, "a") as fh:
        fh.write(sec)
    print(f"Appended Part D to {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="reports/smoke_test.md")
    ap.add_argument("--raw7", default="reports/raw/7b")
    ap.add_argument("--raw3", default="reports/raw/3b")
    a = ap.parse_args()
    write(a.report, a.raw7, a.raw3)
