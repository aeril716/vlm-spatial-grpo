"""Part G: Part F pass@k diagnosis redone with shuffled choices and a non-yes/no worked example.

Identical to part_f_passk.py (same 200 row IDs, sampling settings, scoring, output format)
except:
  1. the answer choices of every question are shuffled with a fixed seed. One order per
     row_id, saved to data/passk_g_choice_order.json and reused for greedy, all 8 samples
     and both models;
  2. the worked example in the instruction answers "the lamp" instead of "no".

  python smoke/part_g_passk.py --model 3B     # writes reports/raw/part_g/<model>.jsonl
  python smoke/part_g_passk.py --model 7B
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import part_f_passk as PF

ORDER_PATH = "data/passk_g_choice_order.json"
OUT_DIR = "reports/raw/part_g"

_EXAMPLE_F = PF.INSTRUCTION[PF.INSTRUCTION.index("Example of the format"):]
EXAMPLE_G = (
    "Example of the format, for a different question (\"Which is closer to the camera, the lamp or the sofa?\" "
    "with choices the lamp / the sofa):\n"
    "<description>A grey sofa stands against the back wall. A floor lamp stands in front of it, near the bottom "
    "of the image.</description>\n"
    "<reasoning>The lamp is lower in the image and overlaps the sofa, so it is in front of the sofa. That means "
    "the lamp is closer to the camera.</reasoning>\n"
    "<answer>the lamp</answer>"
)
INSTRUCTION = PF.INSTRUCTION.replace(_EXAMPLE_F, EXAMPLE_G)
assert INSTRUCTION.endswith(EXAMPLE_G) and "<answer>no</answer>" not in INSTRUCTION


def choice_orders(pool, row_ids, seed=C.SEED):
    """One permutation per question, drawn in row_id order from a single seeded RNG."""
    by_id = {r["row_id"]: r for r in pool}
    rng = random.Random(seed)
    orders = {}
    for i in sorted(row_ids):
        perm = list(range(len(by_id[i]["answers"])))
        rng.shuffle(perm)
        orders[str(i)] = perm
    shuffled = {i: [by_id[int(i)]["answers"][j] for j in p] for i, p in orders.items()}
    first_ok = sum(1 for i, ch in shuffled.items() if ch[0] == by_id[int(i)]["correct_answer"])
    return {
        "description": "Part G choice order: shuffled[i] = original answers[order[i]] for each row_id",
        "seed": seed, "n": len(orders),
        "correct_first": first_ok,
        "order": orders,
        "shuffled_choices": shuffled,
    }


def load_pool_shuffled():
    pool = _load_pool()
    if not os.path.exists(ORDER_PATH):
        ids = json.load(open(PF.IDS_PATH))["row_ids"]
        with open(ORDER_PATH, "w") as f:
            json.dump(choice_orders(pool, ids), f, indent=2, ensure_ascii=False)
    orders = json.load(open(ORDER_PATH))["order"]
    out = []
    for r in pool:
        p = orders.get(str(r["row_id"]))
        if p is not None:
            r = dict(r, answers=[r["answers"][j] for j in p], original_answers=list(r["answers"]))
        out.append(r)
    return out


# part_f_passk.main reads these as module globals at call time
_load_pool = PF.load_pool
PF.INSTRUCTION = INSTRUCTION
PF.OUT_DIR = OUT_DIR
PF.load_pool = load_pool_shuffled

if __name__ == "__main__":
    PF.main()
