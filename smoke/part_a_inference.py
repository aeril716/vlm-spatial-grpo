"""Part A: fp16 vs fp32 inference smoke test on 20 array/SAT val samples.

fp32 is the reference. Greedy decoding, short answers.
Writes reports/smoke_test.md.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

MAX_NEW_TOKENS = 24


def run_dtype(samples, dtype, label):
    C.set_seed()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model, processor = C.load_model(dtype)
    rows, t_total = [], 0.0

    for idx, s in enumerate(samples):
        msgs = C.to_messages(s)
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=s["images"], return_tensors="pt").to("cuda")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,          # greedy
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
                return_dict_in_generate=True,
                output_logits=True,
            )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        t_total += dt

        # NaN/inf check over every generated step's logits
        bad = False
        for lg in out.logits:
            if not torch.isfinite(lg).all():
                bad = True
                break

        gen_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
        raw = processor.decode(gen_ids, skip_special_tokens=True).strip()
        parsed, ok = C.parse_answer(raw, s["answers"])

        rows.append(
            {
                "i": idx,
                "question_type": s["question_type"],
                "n_images": len(s["images"]),
                "raw": raw,
                "parsed": parsed,
                "parse_ok": ok,
                "correct_answer": s["correct_answer"],
                "nonfinite_logits": bad,
                "seconds": round(dt, 3),
            }
        )
        print(f"[{label}] {idx+1}/{len(samples)} {dt:5.2f}s nonfinite={bad} :: {raw[:60]!r}", flush=True)

    peak = torch.cuda.max_memory_allocated() / 1024**3
    del model, processor
    torch.cuda.empty_cache()
    return rows, peak, t_total


def md_table(hdr, rows):
    out = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def cell(s, n=34):
    s = "—" if s is None else str(s).replace("|", "\\|").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=C.N_SAMPLES)
    ap.add_argument("--out", default="reports/smoke_test.md")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA unavailable — cannot run the smoke test.")

    print(f"GPU: {C.gpu_name()} ({C.capability()})  torch {torch.__version__}")
    print(f"Loading {args.n} samples from {C.DATASET_ID} split={C.SPLIT} (streaming, seed={C.SEED})")
    samples = C.load_samples(args.n)
    print(f"Loaded {len(samples)} samples.")

    r16, peak16, t16 = run_dtype(samples, torch.float16, "fp16")
    r32, peak32, t32 = run_dtype(samples, torch.float32, "fp32")

    n = len(samples)
    agree = sum(1 for a, b in zip(r16, r32) if a["parsed"] == b["parsed"])
    nonfinite = sum(1 for r in r16 if r["nonfinite_logits"])
    ok16 = sum(1 for r in r16 if r["parse_ok"])
    ok32 = sum(1 for r in r32 if r["parse_ok"])
    acc16 = sum(1 for r in r16 if r["parsed"] == r["correct_answer"])
    acc32 = sum(1 for r in r32 if r["parsed"] == r["correct_answer"])

    per_sample = [
        [
            a["i"],
            a["question_type"],
            a["n_images"],
            cell(a["parsed"]),
            cell(b["parsed"]),
            cell(a["correct_answer"]),
            "yes" if a["nonfinite_logits"] else "no",
            "=" if a["parsed"] == b["parsed"] else "DIFF",
        ]
        for a, b in zip(r16, r32)
    ]

    body = f"""# VLM Spatial Reasoning — Smoke Test (Part A: Inference)

Model **{C.MODEL_ID}** · dataset **{C.DATASET_ID}** split **`{C.SPLIT}`** · {n} samples, seed {C.SEED}

GPU **{C.gpu_name()}** ({C.capability()}) · torch {torch.__version__} · attn `sdpa` · greedy decoding, max_new_tokens={MAX_NEW_TOKENS}

> `array/SAT` splits are `train` / `static` / `val` / `test` — there is no split named
> `validation`, so **`val`** (4001 examples) is used as validation. Samples are drawn with
> `streaming=True` + `.shuffle(seed={C.SEED}, buffer_size={C.SHUFFLE_BUFFER}).take({n})`, so the
> full split is never downloaded.

## Summary

| metric | fp16 | fp32 (reference) |
|---|---|---|
| peak VRAM (GiB) | {peak16:.2f} | {peak32:.2f} |
| total time (s) | {t16:.1f} | {t32:.1f} |
| seconds / sample | {t16/n:.2f} | {t32/n:.2f} |
| parse success | {ok16}/{n} ({100*ok16/n:.0f}%) | {ok32}/{n} ({100*ok32/n:.0f}%) |
| matches correct_answer | {acc16}/{n} | {acc32}/{n} |

- **fp16 vs fp32 agreement: {agree}/{n} ({100*agree/n:.0f}%)**
- **fp16 non-finite (NaN/inf) logits: {nonfinite}/{n} sample(s)**

## Per sample

{md_table(["#", "question_type", "imgs", "fp16 answer", "fp32 answer", "correct_answer", "fp16 NaN/inf", "agree"], per_sample)}

## Raw model output

{md_table(["#", "fp16 raw", "fp32 raw"], [[a["i"], cell(a["raw"], 60), cell(b["raw"], 60)] for a, b in zip(r16, r32)])}
"""
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(body)
    os.makedirs("reports/raw", exist_ok=True)
    with open("reports/raw/part_a.json", "w") as f:
        json.dump({"fp16": r16, "fp32": r32,
                   "peak_gib": {"fp16": peak16, "fp32": peak32},
                   "seconds": {"fp16": t16, "fp32": t32}}, f, indent=2)

    print(f"\nWrote {args.out}")
    print(f"agreement {agree}/{n} · fp16 non-finite {nonfinite}/{n} · peak {peak16:.2f} / {peak32:.2f} GiB")


if __name__ == "__main__":
    main()
