"""Shared helpers for the SAT smoke tests (Qwen2.5-VL, sm_75 / fp16+SDPA)."""
import io
import os
import random

import numpy as np
import torch

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_ID = "array/SAT"
# NOTE: array/SAT has splits: train / static / val / test. There is NO "validation".
# "val" (4001 examples) is the validation split.
SPLIT = "val"
N_SAMPLES = 20
SEED = 0
SHUFFLE_BUFFER = 1000


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_samples(n: int = N_SAMPLES, seed: int = SEED):
    """Stream the val split so the full 4001-example split is never downloaded."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split=SPLIT, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER)
    out = []
    for row in ds.take(n):
        imgs = row["image_bytes"]
        if not isinstance(imgs, list):
            imgs = [imgs]
        out.append(
            {
                "images": [_to_pil(i) for i in imgs],
                "question": row["question"],
                "answers": list(row["answers"]),
                "question_type": row["question_type"],
                "correct_answer": row["correct_answer"],
            }
        )
    return out


def _to_pil(obj):
    from PIL import Image

    if hasattr(obj, "convert"):
        return obj.convert("RGB")
    if isinstance(obj, dict) and obj.get("bytes"):
        return Image.open(io.BytesIO(obj["bytes"])).convert("RGB")
    if isinstance(obj, (bytes, bytearray)):
        return Image.open(io.BytesIO(obj)).convert("RGB")
    raise TypeError(f"unrecognised image payload: {type(obj)}")


def build_prompt(sample):
    """Multiple-choice prompt; answer with the choice text verbatim."""
    choices = "\n".join(f"- {c}" for c in sample["answers"])
    return (
        f"{sample['question']}\n\nChoices:\n{choices}\n\n"
        "Answer with exactly one choice from the list above. "
        "Reply with the choice text only, no explanation."
    )


def to_messages(sample):
    content = [{"type": "image", "image": img} for img in sample["images"]]
    content.append({"type": "text", "text": build_prompt(sample)})
    return [{"role": "user", "content": content}]


def load_model(dtype: torch.dtype):
    """sm_75 (Turing): fp16 + SDPA only. No bf16, no FlashAttention2."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        attn_implementation="sdpa",
        device_map=None,
    ).to("cuda").eval()
    return model, processor


def parse_answer(text: str, choices):
    """Return (parsed_choice_or_None, ok). Tries exact, substring, then A/B/C/D index."""
    t = (text or "").strip()
    norm = lambda s: " ".join(s.lower().split()).strip(" .,!?:;\"'")
    tn = norm(t)
    for c in choices:
        if tn == norm(c):
            return c, True
    hits = [c for c in choices if norm(c) and norm(c) in tn]
    if len(hits) == 1:
        return hits[0], True
    first = tn[:1].upper()
    if first and "A" <= first <= "Z":
        i = ord(first) - ord("A")
        if i < len(choices) and (len(tn) == 1 or not tn[1:2].isalnum()):
            return choices[i], True
    if hits:
        return hits[0], True
    return None, False


def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def capability():
    if not torch.cuda.is_available():
        return "n/a"
    maj, minor = torch.cuda.get_device_capability(0)
    return f"sm_{maj}{minor}"
