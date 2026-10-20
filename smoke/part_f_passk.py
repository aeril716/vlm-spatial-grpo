"""Part F: base-model pass@k diagnosis. No training.

For each of 200 stratified train-pool questions: 1 greedy answer + 8 samples at the
GRPO sampling settings (T=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0), max 1024
new tokens, Part E E1 prompt with the copyable answer placeholder replaced by "...".
Score = exact match inside <answer>, 0 if it doesn't parse.

  python smoke/part_f_passk.py --model 3B     # writes reports/raw/part_f/<model>.jsonl
  python smoke/part_f_passk.py --model 7B

Question selection is written once to data/passk_f_ids.json and reused by both models.
"""
import argparse
import json
import os
import pickle
import random
import sys
import time

import torch
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import part_c_reasoning as PC
from fetch_train_pool import CACHE, SPATIAL_START

N_QUESTIONS = 200
K = 8
MAX_NEW = 1024
IDS_PATH = "data/passk_f_ids.json"
OUT_DIR = "reports/raw/part_f"

# E1 instruction with the copyable placeholder removed; the worked example is kept.
INSTRUCTION = PC.E_INSTRUCTION.replace("<answer>one of the choices, copied exactly</answer>", "<answer>...</answer>")
assert INSTRUCTION != PC.E_INSTRUCTION


def load_pool():
    """The pool pickle holds consecutive train rows starting at SPATIAL_START (see fetch_train_pool)."""
    with open(CACHE, "rb") as f:
        pool = pickle.load(f)
    return [dict(r, row_id=SPATIAL_START + i) for i, r in enumerate(pool)]


def stratified_ids(pool, n=N_QUESTIONS, seed=C.SEED):
    """Equal per question_type as far as possible: small types are taken whole and the
    remainder is split evenly over the larger ones (water-filling)."""
    by_type = {}
    for r in pool:
        by_type.setdefault(r["question_type"], []).append(r["row_id"])
    types = sorted(by_type, key=lambda t: len(by_type[t]))
    quota, left = {}, n
    for i, t in enumerate(types):
        share = left // (len(types) - i)
        quota[t] = min(share, len(by_type[t]))
        left -= quota[t]
    rng = random.Random(seed)
    picked = {t: sorted(rng.sample(sorted(by_type[t]), quota[t])) for t in sorted(by_type)}
    return {
        "description": "Part F pass@k questions: stratified sample of the SAT train spatial pool",
        "pool": CACHE, "pool_first_row_id": SPATIAL_START, "pool_size": len(pool),
        "seed": seed, "n": sum(quota.values()),
        "pool_counts": {t: len(v) for t, v in sorted(by_type.items())},
        "quota": dict(sorted(quota.items())),
        "row_ids": sorted(x for v in picked.values() for x in v),
        "row_ids_by_type": picked,
    }


def messages(r):
    choices = "\n".join(f"- {c}" for c in r["answers"])
    text = f"{r['question']}\n\nChoices:\n{choices}\n\n{INSTRUCTION}"
    # images before the text, as TRL's prepare_multimodal_messages lays them out
    content = [{"type": "image", "image": C._to_pil(i)} for i in r["image_bytes"]]
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}], text


def score(text, gold):
    found = {k: PC.TAG_RE[k].search(text) for k in PC.TAGS}
    ans = found["answer"].group(1) if found["answer"] else None
    return {
        "parsed": ans,
        "all_tags": all(found.values()),
        "correct": 1 if ans is not None and PC.norm(ans) == PC.norm(gold) else 0,
    }


class NonFinite(RuntimeError):
    pass


class NanGuard(LogitsProcessor):
    """Raise on the first decoding step whose logits contain NaN/inf."""

    def __call__(self, input_ids, scores):
        if not torch.isfinite(scores).all():
            raise NonFinite(f"non-finite logits at decode position {input_ids.shape[1]}")
        return scores


def decode(processor, seqs, prompt_len, eos_ids):
    out = []
    for s in seqs:
        gen = s[prompt_len:].tolist()
        end = next((i for i, t in enumerate(gen) if t in eos_ids), None)
        ids = gen if end is None else gen[:end]
        out.append({
            "text": processor.decode(ids, skip_special_tokens=True),
            "n_tokens": len(ids) + (0 if end is None else 1),
            "cut_off": end is None and len(gen) >= MAX_NEW,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["3B", "7B"], required=True)
    ap.add_argument("--batch", type=int, default=8, help="questions per generate() call")
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first N questions")
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    pool = load_pool()
    if not os.path.exists(IDS_PATH):
        with open(IDS_PATH, "w") as f:
            json.dump(stratified_ids(pool), f, indent=2)
    sel = json.load(open(IDS_PATH))
    by_id = {r["row_id"]: r for r in pool}
    rows = [by_id[i] for i in sel["row_ids"]][: a.limit]

    model_id = f"Qwen/Qwen2.5-VL-{a.model}-Instruct"
    C.set_seed()
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.float16, attn_implementation="sdpa").to("cuda").eval()

    tok = processor.tokenizer
    eos_ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id, tok.pad_token_id} - {None}
    common = dict(max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id,
                  eos_token_id=tok.eos_token_id, disable_compile=True)
    # same fields TRL's GRPOTrainer sets for HF generation (GRPOConfig defaults); replaces the
    # model's own generation_config (repetition_penalty=1.05, temperature=1e-6)
    sample_cfg = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, min_p=None,
                                  repetition_penalty=1.0, num_return_sequences=K, **common)
    greedy_cfg = GenerationConfig(do_sample=False, repetition_penalty=1.0, num_beams=1, **common)
    guard = LogitsProcessorList([NanGuard()])

    os.makedirs(a.out_dir, exist_ok=True)
    name = f"Qwen2.5-VL-{a.model}-Instruct"
    out_path = f"{a.out_dir}/{name}.jsonl"
    meta = {"model": model_id, "n_questions": len(rows), "k": K, "max_new_tokens": MAX_NEW,
            "batch_questions": a.batch, "dtype": "float16", "attn": "sdpa",
            "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0},
            "instruction": INSTRUCTION, "status": "ok", "error": None, "batches": []}
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    with open(out_path, "w") as fout:
        for b0 in range(0, len(rows), a.batch):
            batch = rows[b0: b0 + a.batch]
            built = [messages(r) for r in batch]
            inputs = processor.apply_chat_template(
                [m for m, _ in built], tokenize=True, return_dict=True, add_generation_prompt=True,
                padding=True, return_tensors="pt").to("cuda")
            plen = inputs["input_ids"].shape[1]
            try:
                torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.inference_mode():
                    g = model.generate(**inputs, generation_config=greedy_cfg, logits_processor=guard)
                torch.cuda.synchronize(); t1 = time.perf_counter()
                with torch.inference_mode():
                    s = model.generate(**inputs, generation_config=sample_cfg, logits_processor=guard)
                torch.cuda.synchronize(); t2 = time.perf_counter()
            except (NonFinite, torch.OutOfMemoryError) as e:
                meta["status"] = "nonfinite" if isinstance(e, NonFinite) else "oom"
                meta["error"] = f"questions {b0}-{b0 + len(batch) - 1}: {str(e).splitlines()[0]}"
                print(f"!!! {meta['status']} — STOPPING: {meta['error']}", flush=True)
                break
            greedy = decode(processor, g, plen, eos_ids)
            samples = decode(processor, s, plen, eos_ids)  # grouped: K consecutive per question
            for qi, r in enumerate(batch):
                gr = dict(greedy[qi], **score(greedy[qi]["text"], r["correct_answer"]))
                sm = [dict(x, **score(x["text"], r["correct_answer"])) for x in samples[qi * K:(qi + 1) * K]]
                fout.write(json.dumps({
                    "row_id": r["row_id"], "question_type": r["question_type"], "question": r["question"],
                    "choices": list(r["answers"]), "correct_answer": r["correct_answer"],
                    "n_images": len(r["image_bytes"]), "prompt_text": built[qi][1],
                    "greedy": gr, "samples": sm, "n_correct": sum(x["correct"] for x in sm),
                }, ensure_ascii=False) + "\n")
            fout.flush()
            meta["batches"].append({"first": b0, "n": len(batch), "greedy_s": t1 - t0, "sample_s": t2 - t1})
            done = b0 + len(batch)
            print(f"[{name}] {done}/{len(rows)} greedy {t1 - t0:.1f}s sample {t2 - t1:.1f}s "
                  f"peak {torch.cuda.max_memory_allocated() / 1024**3:.2f}GiB "
                  f"elapsed {time.perf_counter() - t_start:.0f}s", flush=True)
    meta["wall_s"] = time.perf_counter() - t_start
    meta["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    meta["gpu"] = f"{C.gpu_name()} ({C.capability()})"
    with open(f"{a.out_dir}/{name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{name}] status={meta['status']} wall={meta['wall_s']:.0f}s peak={meta['peak_vram_gib']:.2f}GiB")
    sys.exit(0 if meta["status"] == "ok" else 3)


if __name__ == "__main__":
    main()
