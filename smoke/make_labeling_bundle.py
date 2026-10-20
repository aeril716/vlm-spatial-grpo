"""Labeling bundle for a human rubric check of Part G 7B sampled answers. No generation.

Selects 60 sampled answers from reports/raw/part_g/Qwen2.5-VL-7B-Instruct.jsonl, exports their
images from the SAT train pool and writes labeling/items.json + labeling/selection.json.

  python smoke/make_labeling_bundle.py
"""
import hashlib
import io
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import part_c_reasoning as PC
import part_f_passk as PF

SRC = "reports/raw/part_g/Qwen2.5-VL-7B-Instruct.jsonl"
ORDER_PATH = "data/passk_g_choice_order.json"
OUT = "labeling"
PER_TYPE = 15
MAX_SIDE = 768
JPEG_QUALITY = 85
SEED = C.SEED

RULE = (
    "Source: every sampled answer (not greedy) in {src}. Eligible: all three tags parse "
    "(<description>, <reasoning>, <answer>) and the <answer> text equals one of the question's choices after "
    "the scoring normalisation (lowercase, collapsed whitespace, stripped punctuation at the ends). "
    "Per question_type (sorted by name) take {per} answers; types alternate between 8 correct + 7 incorrect "
    "and 7 correct + 8 incorrect, starting with 8 correct for the first type. At most one answer per row_id. "
    "With random.Random({seed}), separately for each type in sorted order: of the two buckets (correct, incorrect), "
    "fill the one with fewer candidate questions first; questions are drawn by rng.sample from the sorted "
    "candidate row_ids not already used, and within a question one eligible sample of that bucket is drawn with "
    "rng.choice from its sample indices in order. Items are then numbered 1..60 by (question_type, row_id)."
)


def tag(text, t):
    m = PC.TAG_RE[t].search(text)
    return m.group(1) if m else None


def eligible(q, s):
    return s["all_tags"] and s["parsed"] is not None and any(
        PC.norm(s["parsed"]) == PC.norm(c) for c in q["choices"])


def select(Q, rng):
    types = sorted({q["question_type"] for q in Q})
    picked = []
    for ti, t in enumerate(types):
        n_ok = 8 if ti % 2 == 0 else 7
        need = {1: n_ok, 0: PER_TYPE - n_ok}
        qs = [q for q in Q if q["question_type"] == t]
        cand = {b: {q["row_id"]: [i for i, s in enumerate(q["samples"]) if eligible(q, s) and s["correct"] == b]
                    for q in qs} for b in (1, 0)}
        cand = {b: {r: v for r, v in c.items() if v} for b, c in cand.items()}
        used = set()
        for b in sorted((1, 0), key=lambda b: (len(cand[b]), -b)):
            pool = sorted(r for r in cand[b] if r not in used)
            assert len(pool) >= need[b], f"{t}: only {len(pool)} questions with an eligible score-{b} answer"
            for r in rng.sample(pool, need[b]):
                used.add(r)
                picked.append((t, r, rng.choice(cand[b][r])))
    return sorted(picked)


def save_image(raw, path):
    im = C._to_pil(raw)
    im.thumbnail((MAX_SIDE, MAX_SIDE))  # only shrinks, keeps aspect ratio
    im.save(path, "JPEG", quality=JPEG_QUALITY)
    return list(im.size)


def main():
    Q = [json.loads(l) for l in open(SRC)]
    by_id = {q["row_id"]: q for q in Q}
    orders = json.load(open(ORDER_PATH))["shuffled_choices"]
    pool = {r["row_id"]: r for r in PF.load_pool()}
    os.makedirs(f"{OUT}/images", exist_ok=True)

    items = []
    for n, (t, rid, si) in enumerate(select(Q, random.Random(SEED)), 1):
        q, s = by_id[rid], by_id[rid]["samples"][si]
        choices = orders[str(rid)]
        assert choices == q["choices"], rid  # the order the model saw
        files = []
        for k, raw in enumerate(pool[rid]["image_bytes"]):
            p = f"{OUT}/images/{rid}_{k}.jpg"
            save_image(raw, p)
            files.append(p)
        items.append({
            "id": n, "row_id": rid, "sample_index": si, "question_type": t, "question": q["question"],
            "choices": choices, "correct_answer": q["correct_answer"], "model_correct": bool(s["correct"]),
            "image_files": files,
            "description": tag(s["text"], "description"), "reasoning": tag(s["text"], "reasoning"),
            "answer": tag(s["text"], "answer"), "full_text": s["text"],
        })

    with open(f"{OUT}/items.json", "w") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    counts = {}
    for it in items:
        c = counts.setdefault(it["question_type"], {"correct": 0, "incorrect": 0})
        c["correct" if it["model_correct"] else "incorrect"] += 1
    with open(f"{OUT}/selection.json", "w") as f:
        json.dump({
            "description": "Labeling bundle: 60 Part G 7B sampled answers for a human rubric check",
            "source": SRC, "source_sha256": hashlib.sha256(open(SRC, "rb").read()).hexdigest(),
            "choice_order": ORDER_PATH, "script": "smoke/make_labeling_bundle.py", "seed": SEED,
            "rule": RULE.format(src=SRC, per=PER_TYPE, seed=SEED),
            "images": f"SAT train split (array/SAT, SAT_train.parquet), row index = row_id; "
                      f"saved as images/<row_id>_<image_index>.jpg, image_index 0-based in the dataset's order; "
                      f"longer side at most {MAX_SIDE} px (never upscaled), RGB JPEG quality {JPEG_QUALITY}",
            "counts": counts, "n": len(items),
        }, f, indent=2)
    print(json.dumps(counts), len(items), "items,",
          sum(len(it["image_files"]) for it in items), "images")


if __name__ == "__main__":
    main()
