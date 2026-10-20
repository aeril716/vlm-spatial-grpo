"""Part F report: base-model pass@k diagnosis from reports/raw/part_f/*.jsonl.

Appends the Part F section to reports/smoke_test.md.
"""
import argparse
import collections
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import part_c_reasoning as PC
import part_f_passk as PF

MODELS = ["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct"]
NO_ANSWER = "(no `<answer>`)"
NOT_CHOICE = "(not one of the choices)"


def label(qtype, choice):
    """Collapse a choice text to the label that answer bias is measured on."""
    t = PC.norm(choice)
    if qtype == "goal_aim":
        return "straight" if "straight" in t else ("left" if t.startswith("left") else "right")
    if qtype == "obj_movement":
        if t == "no objects moved":
            return "no objects moved"
        m = re.search(r"moved (left|right) and (towards|away)", t)
        return f"moved {m.group(1)} + {m.group(2)}" if m else t
    return t  # action_consequence: yes / no; action_sequence: the action text itself


def pick_label(q, parsed):
    if parsed is None:
        return NO_ANSWER
    for c in q["choices"]:
        if PC.norm(parsed) == PC.norm(c):
            return label(q["question_type"], c)
    return NOT_CHOICE


def binom_pmf(n, k, p):
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "—"


def model_section(name, Q, meta):
    K = PF.K
    types = sorted({q["question_type"] for q in Q})
    by_t = {t: [q for q in Q if q["question_type"] == t] for t in types}
    n = len(Q)
    out = [f"\n## {name}\n", f"Status: **{meta['status']}**" + (f" — `{meta['error']}`" if meta.get("error") else "")
           + f" · {n} questions answered\n"]

    # 1. greedy accuracy
    out += ["### 1. Greedy accuracy\n", "| question_type | n | greedy correct |", "|---|---|---|"]
    for t in types:
        out.append(f"| {t} | {len(by_t[t])} | {pct(sum(q['greedy']['correct'] for q in by_t[t]), len(by_t[t]))} |")
    out.append(f"| **all** | {n} | **{pct(sum(q['greedy']['correct'] for q in Q), n)}** |")

    # 2. buckets
    out += ["\n### 2. pass@8 buckets (8 samples per question)\n",
            "| question_type | n | 0/8 | mixed 1–7/8 | 8/8 | mean correct / 8 |", "|---|---|---|---|---|---|"]
    for t, qs in list(by_t.items()) + [("**all**", Q)]:
        z = sum(1 for q in qs if q["n_correct"] == 0)
        f = sum(1 for q in qs if q["n_correct"] == K)
        out.append(f"| {t} | {len(qs)} | {pct(z, len(qs))} | {pct(len(qs) - z - f, len(qs))} | {pct(f, len(qs))} | "
                   f"{sum(q['n_correct'] for q in qs) / len(qs):.2f} |")

    # 3. distribution vs. pure guessing
    ps = [1 / len(q["choices"]) for q in Q]
    obs = collections.Counter(q["n_correct"] for q in Q)
    exp = [sum(binom_pmf(K, k, p) for p in ps) for k in range(K + 1)]
    out += ["\n### 3. Correct-count distribution vs. pure guessing\n",
            "Expected = sum over questions of Binomial(8, p = 1 / number of choices). Every question in this "
            "sample has 2 choices, so p = 0.5 throughout.\n",
            "| correct / 8 | observed questions | expected if guessing |", "|---|---|---|"]
    for k in range(K + 1):
        out.append(f"| {k} | {obs.get(k, 0)} | {exp[k]:.1f} |")
    counts = [q["n_correct"] for q in Q]
    mean = sum(counts) / n
    var_obs = sum((c - mean) ** 2 for c in counts) / n
    # variance of a mixture of binomials: E[Var] + Var[E]
    means = [K * p for p in ps]
    var_exp = sum(K * p * (1 - p) for p in ps) / n + sum((m - sum(means) / n) ** 2 for m in means) / n
    out += ["", f"- Mean correct per question: observed **{mean:.2f}**, guessing {sum(means) / n:.2f}",
            f"- Variance of correct counts: observed **{var_obs:.2f}**, binomial (guessing) **{var_exp:.2f}** "
            f"→ ratio {var_obs / var_exp:.2f}× (population variance, n = {n})"]

    # 4. answer bias
    out += ["\n### 4. Answer bias\n",
            "`picked` is over all 8 sampled answers per question. `correct` is the share of questions whose "
            "correct_answer has that label. The last two columns count, among questions whose correct label is "
            "this one, how many scored 0/8 and 8/8. Labels: goal_aim → left / right / straight; obj_movement → "
            "direction + towards/away, or no objects moved; the other types use the choice text.\n"]
    for t, qs in by_t.items():
        picked = collections.Counter(pick_label(q, s["parsed"]) for q in qs for s in q["samples"])
        corr = collections.Counter(label(t, q["correct_answer"]) for q in qs)
        labels = sorted(set(picked) | set(corr), key=lambda x: (x in (NO_ANSWER, NOT_CHOICE), -corr.get(x, 0), x))
        tot = sum(picked.values())
        out += [f"**{t}** ({len(qs)} questions, {tot} sampled answers)\n",
                "| label | picked | correct | correct-label questions: 0/8 | 8/8 |", "|---|---|---|---|---|"]
        for lb in labels:
            grp = [q for q in qs if label(t, q["correct_answer"]) == lb]
            z = sum(1 for q in grp if q["n_correct"] == 0)
            f = sum(1 for q in grp if q["n_correct"] == K)
            out.append(f"| {lb} | {100 * picked.get(lb, 0) / tot:.0f}% ({picked.get(lb, 0)}) | "
                       f"{100 * corr.get(lb, 0) / len(qs):.0f}% ({corr.get(lb, 0)}) | "
                       f"{f'{z}/{len(grp)}' if grp else '—'} | {f'{f}/{len(grp)}' if grp else '—'} |")
        first = sum(1 for q in qs for s in q["samples"]
                    if s["parsed"] is not None and PC.norm(s["parsed"]) == PC.norm(q["choices"][0]))
        first_ok = sum(1 for q in qs if PC.norm(q["correct_answer"]) == PC.norm(q["choices"][0]))
        nc = collections.Counter(PC.norm(s["parsed"]) for q in qs for s in q["samples"]
                                 if pick_label(q, s["parsed"]) == NOT_CHOICE)
        out.append(f"\nPosition: picks the first-listed choice in {100 * first / tot:.0f}% of samples; the first-listed "
                   f"choice is correct in {100 * first_ok / len(qs):.0f}% of questions.")
        if nc:
            out.append("Most common answers that are not one of the choices: "
                       + ", ".join(f"`{a}` ×{c}" for a, c in nc.most_common(3)) + ".")
        out.append("")

    # 5. format
    S = [s for q in Q for s in q["samples"]]
    G = [q["greedy"] for q in Q]
    out += ["### 5. Format\n", "| | greedy | sampled |", "|---|---|---|"]
    for nm, f in (("all three tags", lambda x: x["all_tags"]), ("contains `<answer>`", lambda x: x["parsed"] is not None),
                  ("cut off at 1024", lambda x: x["cut_off"])):
        out.append(f"| {nm} | {pct(sum(1 for x in G if f(x)), len(G))} | {pct(sum(1 for x in S if f(x)), len(S))} |")
    out.append(f"| completion tokens mean / max | {sum(x['n_tokens'] for x in G) / len(G):.0f} / "
               f"{max(x['n_tokens'] for x in G)} | {sum(x['n_tokens'] for x in S) / len(S):.0f} / "
               f"{max(x['n_tokens'] for x in S)} |")

    # 6. time / memory
    gs = sum(b["greedy_s"] for b in meta["batches"])
    ss = sum(b["sample_s"] for b in meta["batches"])
    out += ["\n### 6. Time and memory\n", "| | |", "|---|---|",
            f"| total wall time | {meta['wall_s'] / 60:.1f} min |",
            f"| time per question (greedy + 8 samples) | {meta['wall_s'] / n:.1f} s |",
            f"| of which greedy / sampling | {gs / n:.1f} s / {ss / n:.1f} s |",
            f"| peak VRAM | {meta['peak_vram_gib']:.2f} GiB |",
            "", f"Questions are generated {meta['batch_questions']} at a time, i.e. {meta['batch_questions']} greedy "
            f"sequences and then {meta['batch_questions']}×8 sampled sequences per `generate()` call. Per-question "
            "times are wall time divided by the number of questions, not the latency of one question alone."]

    # examples
    flat = [(qi, si, q) for qi, q in enumerate(Q) for si in range(K)]
    good = next((x for x in flat if x[2]["samples"][x[1]]["correct"] == 1), None)
    bad = next((x for x in flat if x[2]["samples"][x[1]]["correct"] == 0), None)
    out += ["\n### Two sampled answers (unedited)\n",
            "Same selection rule as Part C: the first sampled answer that scored 1 and the first that scored 0, "
            "scanning questions in file order and samples in order.\n"]
    for x in (good, bad):
        if x is None:
            continue
        qi, si, q = x
        s = q["samples"][si]
        out.append(f"**row_id {q['row_id']} ({q['question_type']}), sample {si}**: score {s['correct']}, "
                   f"{s['n_tokens']} tokens, all three tags: {'yes' if s['all_tags'] else 'no'}, "
                   f"correct_answer: `{q['correct_answer']}`, choices: {q['choices']}\n\n{PC.fence(s['text'])}\n")
    return "\n".join(out)


def summary(rows):
    head = "| | " + " | ".join(m.replace("Qwen2.5-VL-", "") for m, _, _ in rows) + " |"
    lines = [head, "|---" * (len(rows) + 1) + "|"]

    def r(name, fn):
        lines.append(f"| {name} | " + " | ".join(fn(Q, meta) for _, Q, meta in rows) + " |")

    K = PF.K
    r("questions answered", lambda Q, m: f"{len(Q)} ({m['status']})")
    r("greedy accuracy", lambda Q, m: pct(sum(q["greedy"]["correct"] for q in Q), len(Q)))
    r("0/8", lambda Q, m: pct(sum(1 for q in Q if q["n_correct"] == 0), len(Q)))
    r("mixed 1–7/8", lambda Q, m: pct(sum(1 for q in Q if 0 < q["n_correct"] < K), len(Q)))
    r("8/8", lambda Q, m: pct(sum(1 for q in Q if q["n_correct"] == K), len(Q)))
    r("sampled answers correct", lambda Q, m: f"{100 * sum(q['n_correct'] for q in Q) / (K * len(Q)):.1f}%")
    r("sampled: all three tags", lambda Q, m: f"{100 * sum(s['all_tags'] for q in Q for s in q['samples']) / (K * len(Q)):.1f}%")
    r("sampled: cut off at 1024", lambda Q, m: f"{sum(s['cut_off'] for q in Q for s in q['samples'])}/{K * len(Q)}")
    r("time per question", lambda Q, m: f"{m['wall_s'] / len(Q):.1f} s")
    r("peak VRAM", lambda Q, m: f"{m['peak_vram_gib']:.2f} GiB")
    return "\n".join(lines)


def write(path, raw=PF.OUT_DIR):
    sel = json.load(open(PF.IDS_PATH))
    rows = []
    for m in MODELS:
        p = f"{raw}/{m}.jsonl"
        if os.path.exists(p):
            rows.append((m, [json.loads(l) for l in open(p)], json.load(open(f"{raw}/{m}_meta.json"))))
    meta0 = rows[0][2]
    prompt_example = rows[0][1][0]["prompt_text"]
    pool_tbl = "\n".join(
        f"| {t} | {sel['pool_counts'][t]} | 2 | {sel['quota'][t]} |" for t in sorted(sel["pool_counts"]))
    sec = f"""
---

# Part F: base-model pass@k diagnosis (no training)

Base models with no adapter, fp16, SDPA, on {meta0['gpu']}. For each question: **1 greedy answer** and
**k = 8 sampled answers** at the GRPO sampling settings: TRL's `GRPOConfig` defaults are `temperature=1.0`,
`top_p=1.0`, `top_k=0` and `repetition_penalty=1.0`, and they replace the model's own generation_config,
whose `repetition_penalty=1.05` is therefore not used. `max_new_tokens=1024` for both greedy and sampled
answers. Score = exact match of the text inside `<answer>` with correct_answer (same normalisation as Parts B–E), 0 if
`<answer>` doesn't parse. Generation stops on NaN/inf logits (checked at every decode step) or OOM; neither
happened.

## Questions

Pool: `{sel['pool']}`, the first {sel['pool_size']} spatial rows of the SAT train split (row IDs
{sel['pool_first_row_id']}–{sel['pool_first_row_id'] + sel['pool_size'] - 1}; the row ID is the row index in
`SAT_train.parquet`). This is only the first slice of the spatial part of the split, not all of it.

| question_type | questions in pool | answer choices | sampled |
|---|---|---|---|
{pool_tbl}
| **total** | {sel['pool_size']} | | **{sel['n']}** |

Every question in the pool has exactly 2 answer choices. Stratified sample with seed {sel['seed']}: equal per
type as far as possible. The two 48-question types are taken whole and the remaining 104 are split between
the other two, capped at goal_aim's 50. The row IDs are saved in `{PF.IDS_PATH}`, total and by type.

## Prompt

Part E E1 prompt (instruction + worked example at the end of the user message, no system prompt). The only
change is that `<answer>one of the choices, copied exactly</answer>` became `<answer>...</answer>`. The full user text
for the first question:

{PC.fence(prompt_example)}

## Summary

{summary(rows)}
"""
    for m, Q, meta in rows:
        sec += model_section(m, Q, meta)
    sec += f"""

Raw answers for every question (greedy + 8 samples, full text, with scores, token counts and cut-off flags)
are in `{raw}/<model>.jsonl`, with run metadata in `{raw}/<model>_meta.json`.
"""
    with open(path, "a") as fh:
        fh.write(sec)
    print(f"Appended Part F to {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="reports/smoke_test.md")
    write(ap.parse_args().report)
