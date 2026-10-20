"""Part G report: Part F pass@k diagnosis with shuffled choices and a non-yes/no example.

Reads reports/raw/part_g/*.jsonl (and reports/raw/part_f/*.jsonl for the comparison) and
appends the Part G section to reports/smoke_test.md.
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import part_c_reasoning as PC
import part_f_passk as PF
import part_f_report as FR
import part_g_passk as PG

MODELS = FR.MODELS
K = PF.K
F_RAW = "reports/raw/part_f"  # not PF.OUT_DIR: importing part_g_passk repoints it to part_g
pct = FR.pct


def load(raw):
    rows = {}
    for m in MODELS:
        p = f"{raw}/{m}.jsonl"
        if os.path.exists(p):
            rows[m] = ([json.loads(l) for l in open(p)], json.load(open(f"{raw}/{m}_meta.json")))
    return rows


def is_first(q, parsed):
    return parsed is not None and PC.norm(parsed) == PC.norm(q["choices"][0])


def in_choices(q, parsed):
    return parsed is not None and any(PC.norm(parsed) == PC.norm(c) for c in q["choices"])


def stats(Q):
    n = len(Q)
    counts = [q["n_correct"] for q in Q]
    mean = sum(counts) / n
    S = [s for q in Q for s in q["samples"]]
    return {
        "greedy": sum(q["greedy"]["correct"] for q in Q), "n": n,
        "mean": mean, "var": sum((c - mean) ** 2 for c in counts) / n,
        "zero": sum(1 for c in counts if c == 0), "full": sum(1 for c in counts if c == K),
        "mixed": sum(1 for c in counts if 0 < c < K),
        "fmt_s": sum(s["all_tags"] for s in S) / len(S),
        "fmt_g": sum(q["greedy"]["all_tags"] for q in Q) / n,
    }


def comparison(F, G):
    lines = ["| | " + " | ".join(f"{m.replace('Qwen2.5-VL-', '')} {p}" for m in MODELS for p in ("F", "G")) + " |",
             "|---" * (2 * len(MODELS) + 1) + "|"]
    st = {(m, p): stats(R[m][0]) for m in MODELS for p, R in (("F", F), ("G", G)) if m in R}

    def row(name, fn):
        lines.append(f"| {name} | " + " | ".join(fn(st[(m, p)]) if (m, p) in st else "—"
                                                 for m in MODELS for p in ("F", "G")) + " |")

    row("greedy accuracy", lambda s: pct(s["greedy"], s["n"]))
    row("mean correct / 8", lambda s: f"{s['mean']:.2f}")
    row("variance of correct / 8 (binomial 2.00)", lambda s: f"{s['var']:.2f}")
    row("0/8", lambda s: pct(s["zero"], s["n"]))
    row("mixed 1–7/8", lambda s: pct(s["mixed"], s["n"]))
    row("8/8", lambda s: pct(s["full"], s["n"]))
    row("format rate, sampled (all three tags)", lambda s: f"{100 * s['fmt_s']:.1f}%")
    row("format rate, greedy (all three tags)", lambda s: f"{100 * s['fmt_g']:.1f}%")
    return "\n".join(lines)


def position_bias(G):
    out = ["| model | question_type | n | first-listed is correct | greedy picks first | sampled picks first "
           "(of all 8×n) | sampled picks first (of answers that are a choice) |", "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        if m not in G:
            continue
        Q = G[m][0]
        types = sorted({q["question_type"] for q in Q})
        for t, qs in [(t, [q for q in Q if q["question_type"] == t]) for t in types] + [("**all**", Q)]:
            ok = sum(1 for q in qs if PC.norm(q["correct_answer"]) == PC.norm(q["choices"][0]))
            gf = sum(1 for q in qs if is_first(q, q["greedy"]["parsed"]))
            sf = sum(1 for q in qs for s in q["samples"] if is_first(q, s["parsed"]))
            sv = sum(1 for q in qs for s in q["samples"] if in_choices(q, s["parsed"]))
            out.append(f"| {m.replace('Qwen2.5-VL-', '')} | {t} | {len(qs)} | {pct(ok, len(qs))} | "
                       f"{pct(gf, len(qs))} | {pct(sf, K * len(qs))} | {pct(sf, sv)} |")
    return "\n".join(out)


def yes_no(F, G):
    """Answers that are exactly yes / no (after normalisation) on questions where neither is a choice."""
    def count(Q, t):
        qs = [q for q in Q if (t is None or q["question_type"] == t)
              and not {"yes", "no"} & {PC.norm(c) for c in q["choices"]}]
        S = [s["parsed"] for q in qs for s in q["samples"]]
        g = [q["greedy"]["parsed"] for q in qs]
        f = lambda xs, w: sum(1 for x in xs if x is not None and PC.norm(x) == w)
        return len(qs), f(S, "yes"), f(S, "no"), f(g, "yes") + f(g, "no")

    types = sorted({q["question_type"] for R in (F, G) for Q, _ in R.values() for q in Q})
    out = ["| model | question_type | questions | Part F sampled yes / no (of 8×n) | Part G sampled yes / no | "
           "Part F greedy yes+no | Part G greedy yes+no |", "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        for t in types + [None]:
            nf, yf, xf, gf = count(F[m][0], t)
            ng, yg, xg, gg = count(G[m][0], t)
            if nf == 0:
                continue
            out.append(f"| {m.replace('Qwen2.5-VL-', '')} | {t or '**all**'} | {nf} | "
                       f"{yf} / {xf} ({100 * (yf + xf) / (K * nf):.1f}%) | "
                       f"{yg} / {xg} ({100 * (yg + xg) / (K * ng):.1f}%) | {gf}/{nf} | {gg}/{ng} |")
    return "\n".join(out)


def example_leak(R):
    """Answers that copy the new worked example's answer choices."""
    out = []
    for m, (Q, _) in R.items():
        S = [s["parsed"] for q in Q for s in q["samples"]] + [q["greedy"]["parsed"] for q in Q]
        c = collections.Counter(PC.norm(x) for x in S if x is not None and PC.norm(x) in ("the lamp", "the sofa"))
        out.append(f"{m.replace('Qwen2.5-VL-', '')}: `the lamp` ×{c['the lamp']}, `the sofa` ×{c['the sofa']}")
    return "; ".join(out)


def write(path):
    F, G = load(F_RAW), load(PG.OUT_DIR)
    order = json.load(open(PG.ORDER_PATH))
    Q0 = next(iter(G.values()))[0]
    meta0 = next(iter(G.values()))[1]
    by_t = collections.defaultdict(list)
    for q in Q0:
        by_t[q["question_type"]].append(PC.norm(q["correct_answer"]) == PC.norm(q["choices"][0]))
    first_tbl = "\n".join(f"| {t} | {len(v)} | {pct(sum(v), len(v))} |" for t, v in sorted(by_t.items()))
    allv = [x for v in by_t.values() for x in v]
    unchanged = sum(1 for p in order["order"].values() if p == sorted(p))

    sec = f"""
---

# Part G: Part F pass@k diagnosis with shuffled choices and a new worked example (no training)

Everything is as in Part F (same 2 base models, same 200 row IDs from `{PF.IDS_PATH}`, fp16, SDPA, on
{meta0['gpu']}, 1 greedy + k = 8 samples at temperature 1.0 / top_p 1.0 / top_k 0 / repetition_penalty 1.0,
`max_new_tokens=1024`, same scoring, same batching of 8 questions per `generate()` call, seed 0), with two changes:

1. **Shuffled choices.** For each question, the choices are permuted with one `random.Random(0)` stream, drawn in
   row_id order. Each question gets one order, used for its greedy answer, all 8 samples and both models. The
   order is saved per row_id in `{PG.ORDER_PATH}` (`order[row_id]` gives indices into the original `answers` list, and
   `shuffled_choices[row_id]` gives the resulting choice texts). Every question has 2 choices, so each permutation is either
   kept or swapped: {unchanged}/{order['n']} kept the original order.
2. **New worked example** whose answer is not yes/no (`<answer>the lamp</answer>` instead of `<answer>no</answer>`).

Generation stops on NaN/inf logits (checked at every decode step) or OOM. Status for each model is below.

## Where the correct answer ended up

| question_type | n | correct answer listed first |
|---|---|---|
{first_tbl}
| **all** | {len(allv)} | **{pct(sum(allv), len(allv))}** |

(In Part F the correct answer was listed first in 200/200 questions.)

## Prompt

Same as Part F except for the example. The full user text for the first question (row {Q0[0]['row_id']}; its order
is {'unchanged' if order['order'][str(Q0[0]['row_id'])] == [0, 1] else 'swapped'}):

{PC.fence(Q0[0]['prompt_text'])}

## Part F vs Part G

{comparison(F, G)}

Variance is the population variance of the per-question correct count (n = 200). Under pure guessing
it would be Binomial(8, 0.5), with variance 2.00.

## Position bias

How often each model answers with the first-listed choice, next to how often the first-listed choice is correct.
With an unbiased model the "picks first" columns would match "first-listed is correct". With a model
that ignores the content they would sit near a constant regardless of it.

{position_bias(G)}

## yes / no answers where yes / no is not a choice

Answers whose `<answer>` text is exactly `yes` or `no` (after the usual normalisation) on questions whose choices
contain neither. These always score 0. Only action_consequence offers yes / no as choices, so it is mostly excluded here
(its one non-yes/no question is included).

{yes_no(F, G)}

Answers in Part G that copy the new example's choices (greedy + sampled): {example_leak(G)}.

## Summary (Part G)

{FR.summary([(m, Q, meta) for m, (Q, meta) in G.items()])}
"""
    for m, (Q, meta) in G.items():
        s = FR.model_section(m, Q, meta)
        sec += s.replace(f"\n## {m}\n", f"\n## {m} (Part G)\n", 1)
    sec += f"""

Raw answers for every question (greedy + 8 samples, full text, with scores, token counts, cut-off flags and the
shuffled `choices` as shown in the prompt) are in `{PG.OUT_DIR}/<model>.jsonl`, with run metadata in
`{PG.OUT_DIR}/<model>_meta.json`.
"""
    with open(path, "a") as fh:
        fh.write(sec)
    print(f"Appended Part G to {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="reports/smoke_test.md")
    write(ap.parse_args().report)
