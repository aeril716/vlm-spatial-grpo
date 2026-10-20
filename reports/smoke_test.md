# VLM Spatial Reasoning — Smoke Test (Part A: Inference)

Model **Qwen/Qwen2.5-VL-3B-Instruct** · dataset **array/SAT** split **`val`** · 20 samples, seed 0

GPU **Quadro RTX 8000** (sm_75) · torch 2.11.0+cu128 · attn `sdpa` · greedy decoding, max_new_tokens=24

> `array/SAT` splits are `train` / `static` / `val` / `test` — there is no split named
> `validation`, so **`val`** (4001 examples) is used as validation. Samples are drawn with
> `streaming=True` + `.shuffle(seed=0, buffer_size=1000).take(20)`, so the
> full split is never downloaded.

## Summary

| metric | fp16 | fp32 (reference) |
|---|---|---|
| peak VRAM (GiB) | 7.29 | 14.23 |
| total time (s) | 6.0 | 13.3 |
| seconds / sample | 0.30 | 0.67 |
| parse success | 20/20 (100%) | 20/20 (100%) |
| matches correct_answer | 14/20 | 14/20 |

- **fp16 vs fp32 agreement: 20/20 (100%)**
- **fp16 non-finite (NaN/inf) logits: 0/20 sample(s)**

## Per sample

| # | question_type | imgs | fp16 answer | fp32 answer | correct_answer | fp16 NaN/inf | agree |
|---|---|---|---|---|---|---|---|
| 0 | goal_aim | 1 | left by 49 degrees | left by 49 degrees | right by 49 degrees | no | = |
| 1 | action_sequence | 2 | rotated left | rotated left | rotated left | no | = |
| 2 | action_consequence | 1 | yes | yes | no | no | = |
| 3 | action_sequence | 2 | rotated left and moved forward | rotated left and moved forward | rotated left and moved forward | no | = |
| 4 | action_consequence | 1 | yes | yes | no | no | = |
| 5 | goal_aim | 1 | left by 58 degrees | left by 58 degrees | left by 58 degrees | no | = |
| 6 | goal_aim | 1 | left by 29 degrees | left by 29 degrees | left by 29 degrees | no | = |
| 7 | obj_movement | 2 | no objects moved | no objects moved | no objects moved | no | = |
| 8 | goal_aim | 1 | left by 30 degrees | left by 30 degrees | left by 30 degrees | no | = |
| 9 | action_consequence | 1 | yes | yes | no | no | = |
| 10 | goal_aim | 1 | left by 41 degrees | left by 41 degrees | left by 41 degrees | no | = |
| 11 | action_sequence | 2 | did not move | did not move | did not move | no | = |
| 12 | action_consequence | 1 | yes | yes | yes | no | = |
| 13 | action_consequence | 1 | yes | yes | no | no | = |
| 14 | obj_movement | 2 | no objects moved | no objects moved | no objects moved | no | = |
| 15 | action_sequence | 2 | rotated right | rotated right | rotated right | no | = |
| 16 | goal_aim | 1 | left by 54 degrees | left by 54 degrees | left by 54 degrees | no | = |
| 17 | obj_movement | 2 | no objects moved | no objects moved | no objects moved | no | = |
| 18 | action_sequence | 2 | rotated right and moved forward | rotated right and moved forward | rotated right and moved forward | no | = |
| 19 | action_consequence | 1 | yes | yes | no | no | = |

## Raw model output

| # | fp16 raw | fp32 raw |
|---|---|---|
| 0 | left by 49 degrees | left by 49 degrees |
| 1 | rotated left | rotated left |
| 2 | yes | yes |
| 3 | rotated left and moved forward | rotated left and moved forward |
| 4 | yes | yes |
| 5 | left by 58 degrees | left by 58 degrees |
| 6 | left by 29 degrees | left by 29 degrees |
| 7 | no objects moved | no objects moved |
| 8 | left by 30 degrees | left by 30 degrees |
| 9 | yes | yes |
| 10 | left by 41 degrees | left by 41 degrees |
| 11 | did not move | did not move |
| 12 | yes | yes |
| 13 | yes | yes |
| 14 | no objects moved | no objects moved |
| 15 | rotated right | rotated right |
| 16 | left by 54 degrees | left by 54 degrees |
| 17 | no objects moved | no objects moved |
| 18 | rotated right and moved forward | rotated right and moved forward |
| 19 | yes | yes |

---

# Part B: GRPO Training (TRL GRPOTrainer + LoRA)

Trained on **`train`** split (val left untouched for later evaluation) ·
5 of 5 steps · fp16 AMP · `num_generations=8` ·
per_device_train_batch_size=8, grad_accum=1 → **1 prompt × 8 generations per step**

TRL 1.13.0 · PEFT 0.20.0 · loss_type `dapo` · `beta=0.0` (no reference model) · prompts never truncated (TRL 1.13 has no `max_prompt_length`) · lr 5e-05 · max_completion_length 24

Reward: **exact match with `correct_answer` → 1.0 / 0.0**

## LoRA configuration

| | |
|---|---|
| `target_modules` | `['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']` |
| `exclude_modules` | `.*visual.*` |
| adapter module types actually matched | `['down_proj', 'gate_proj', 'k_proj', 'o_proj', 'q_proj', 'up_proj', 'v_proj']` |
| LoRA injection sites | 252 |
| r / alpha / dropout | 16 / 32 / 0.05 |
| **trainable params** | **29,933,568** / 3,784,556,544 (0.7909%) |
| **vision encoder** | **FROZEN** — 668,684,288 params, 0 trainable tensors |

> The Qwen2.5-VL vision tower also contains `gate_proj` / `up_proj` / `down_proj`, so without
> `exclude_modules='.*visual.*'` LoRA would have attached to the vision encoder as well.

## Per-step metrics

| step | loss | grad_norm | reward mean | reward std | peak VRAM (GiB) | seconds | mean completion len |
|---|---|---|---|---|---|---|---|
| 1 | -3.406e-08 | 0.5170 | 0.250 | 0.463 | 9.70 | 6.4 | 7.0 |
| 2 | 0.0000 | 0.0000 | 1.000 | 0.000 | 12.15 | 11.2 | 4.0 |
| 3 | 1.490e-08 | 2.1936 | 0.375 | 0.518 | 9.86 | 5.6 | 2.0 |
| 4 | 0.0000 | 0.0000 | 1.000 | 0.000 | 12.16 | 11.1 | 4.0 |
| 5 | 1.490e-08 | 2.2510 | 0.375 | 0.518 | 9.87 | 5.6 | 2.0 |

- **NaN/inf:** None — all loss / grad_norm / reward values finite across every step.
- **Zero-gradient steps:** 2/5 — steps where all 8 generations
  scored the same reward, so `reward_std=0`, advantages vanish and `grad_norm=0` (see note below).
- Overall peak VRAM: **12.16 GiB** of 47.3 GiB
- Total wall time for 5 steps: **40.7s** (8.1s/step)

### Notes

- **Loss magnitude near zero is expected, not a failure.** On the first optimizer pass over each
  generation batch the policy equals the sampling policy, so the importance ratio is exactly 1 and
  the clipped DAPO objective starts at ~0. `grad_norm` is the useful signal here, and it is non-zero
  on every step that had reward variance.
- **2 of 5 steps produced no gradient at all.** GRPO normalizes advantages
  within each group, so when all 8 generations of a prompt receive the same binary
  reward the advantages are identically zero and the step is wasted (`frac_reward_zero_std=1`,
  `grad_norm=0`). With a 1/0 exact-match reward on binary yes/no questions this will happen often —
  worth addressing before a real run via a larger group, harder prompt sampling, or a shaped
  (partial-credit) reward.
- Peak VRAM is sampled per step (the CUDA peak counter is reset at each step boundary), so the
  figure above is the max across steps rather than a single cumulative reading.

---

# Part C: GRPO cost with a reasoning-style output format

Goal: **time and memory per step when completions are long** — not accuracy.

Same setup as Part B in both runs: Qwen2.5-VL-3B-Instruct · TRL 1.13.0 GRPOTrainer ·
LoRA r16 on the LLM decoder only (vision encoder frozen) · fp16 AMP · 5 steps ·
per_device_train_batch_size=8, grad_accum=1, `num_generations=8` (1 prompt × 8 generations per step) ·
lr 5e-05 · `beta=0.0` · seed 0 · same train-split samples. Both runs were executed in this session,
each in a fresh process, on Quadro RTX 8000 (sm_75) in the project venv `.venv/` (torch 2.11.0+cu128).
The step prompts were identical across the two runs: **yes**
(checked by comparing each step's `correct_answer`).

| | Step 1: baseline (Part B rerun) | Step 2: reasoning format |
|---|---|---|
| prompt | Part B prompt (no system prompt) | system prompt below + Part B prompt without its last sentence |
| `max_completion_length` | 24 | 1024 |
| reward | exact match of the whole completion, 1/0 | exact match of the text inside `<answer>`, 1/0; 0 if `<answer>` doesn't parse; format itself not rewarded |
| status | completed 5/5 steps, no NaN/inf, no OOM | completed 5/5 steps, no NaN/inf, no OOM |

Step 2 system prompt:

```text
You are a careful visual spatial reasoner. Always respond in exactly this format:
<description>what is in the image relevant to the question</description>
<reasoning>step-by-step spatial reasoning</reasoning>
<answer>one of the given choices</answer>
The text inside <answer> must be exactly one of the given choices, copied verbatim.
```

Part B's user prompt ends with *"Reply with the choice text only, no explanation."*, which contradicts the
reasoning format, so in Step 2 that one sentence is removed; the question and choice list are unchanged.

## Summary (mean over steps)

| | baseline | reasoning | ratio |
|---|---|---|---|
| seconds / step | 7.9 | 11.5 | 1.4× |
| generation s / step | 2.0 | 5.3 | 2.6× |
| train fwd+bwd s / step | 5.8 | 6.0 | 1.0× |
| peak VRAM (GiB, max over steps) | 12.16 | 12.39 | of 47.3 GiB |
| completion tokens: mean / max | 3.8 / 7 | 16.3 / 191 | |
| fraction cut off at max length | 0.000 | 0.000 | |
| zero-gradient steps (reward std = 0) | 2/5 | 3/5 | |

## Per step — Step 1: baseline (Part B rerun)

| step | s/step | gen s | score s | train (fwd+bwd) s | peak VRAM GiB | gen peak GiB | len mean | len max | cut off @max | all tags | reward mean | reward std | loss | grad_norm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6.3 | 2.0 | 0.06 | 4.1 | 9.70 | 7.91 | 7.0 | 7 | 0.000 | n/a | 0.250 | 0.463 | -3.406e-08 | 0.5170 |
| 2 | 11.2 | 2.7 | 0.10 | 8.4 | 12.15 | 8.69 | 4.0 | 4 | 0.000 | n/a | 1.000 | 0.000 | 0.0000 | 0.0000 |
| 3 | 5.6 | 1.3 | 0.06 | 4.1 | 9.86 | 8.23 | 2.0 | 2 | 0.000 | n/a | 0.375 | 0.518 | 1.490e-08 | 2.1936 |
| 4 | 11.1 | 2.7 | 0.09 | 8.3 | 12.16 | 8.70 | 4.0 | 4 | 0.000 | n/a | 1.000 | 0.000 | 0.0000 | 0.0000 |
| 5 | 5.6 | 1.3 | 0.06 | 4.2 | 9.87 | 8.23 | 2.0 | 2 | 0.000 | n/a | 0.375 | 0.518 | 1.490e-08 | 2.2510 |

## Per step — Step 2: reasoning format

| step | s/step | gen s | score s | train (fwd+bwd) s | peak VRAM GiB | gen peak GiB | len mean | len max | cut off @max | all tags | reward mean | reward std | loss | grad_norm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6.6 | 2.3 | 0.06 | 4.1 | 9.81 | 7.98 | 7.5 | 10 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 | 0.0000 |
| 2 | 11.7 | 2.8 | 0.09 | 8.7 | 12.39 | 8.86 | 4.0 | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 | 0.0000 |
| 3 | 9.4 | 4.9 | 0.05 | 4.4 | 10.18 | 8.29 | 17.5 | 65 | 0.000 | 0.125 | 0.125 | 0.354 | -0.9594 | 2.3130 |
| 4 | 12.6 | 4.7 | 0.10 | 7.9 | 12.17 | 8.86 | 8.1 | 36 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 | 0.0000 |
| 5 | 17.1 | 12.0 | 0.06 | 5.1 | 11.78 | 8.29 | 44.2 | 191 | 0.000 | 0.125 | 0.125 | 0.354 | -1.1722 | 1.3167 |

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

**Step 3, generation 4** — reward 1, 65 tokens, cut off: no, all three tags parsed: yes, correct_answer: `yes`

```text
yes

<description>the door on the left side of the image is facing 4.</description>
<reasoning>Assuming the perspective is starting from 4’ and that you are turning left, it would mean you’re turning your back to the door.</reasoning>
<answer>yes</answer>
```

**Step 1, generation 0** — reward 0, 7 tokens, cut off: no, all three tags parsed: no, correct_answer: `right by 15 degrees`

```text
left by 15 degrees
```

### Notes

- **The main goal wasn't reached: completions in Step 2 were not long.** Across the 40 Step 2
  completions the mean was 16.3 tokens and the longest was 191; none came near 1024 and none were cut off.
  33/40 were ≤10 tokens, i.e. a bare answer like `yes` or `left by 15 degrees`. Only 2/40 contained all three
  tags, and those 2 are the only completions where `<answer>` parsed. Qwen2.5-VL-3B-Instruct largely ignores
  the format in the system prompt. Typically it writes the bare answer first and sometimes then adds a
  description and reasoning, often as `description:` / `reasoning:` / `answer:` instead of XML tags. So the
  numbers above are the cost of *slightly* longer completions. They are **not** a measurement of 1024-token
  rollouts, and they should not be used to budget a real reasoning-format run.
- **What the numbers do show.** Generation time grows with the longest completion in the group (HF `generate`
  runs until the last of the 8 sequences finishes). For example, step 5 had a 191-token maximum, took 12.0 s of
  generation vs. 1.3 s in the baseline, and accounted for 70% of that step. Training fwd+bwd time barely
  moved (5.8 → 6.0 s/step mean). Peak VRAM rose only 0.2 GiB, and peak memory is set by the training pass,
  not generation (gen peak ≤ 8.9 GiB vs. step peak ≤ 12.4 GiB).
- **The baseline reproduced Part B exactly.** Loss, grad_norm, reward mean/std and completion lengths are
  identical at every step, and peak VRAM and s/step match to within 0.1 GiB / 0.5 s. So the two Part C runs
  are comparable with each other and with Part B.
- **Zero-gradient steps got worse (3/5 vs. 2/5).** With almost no parsable `<answer>`, all 8 generations
  mostly score 0. Because the format is deliberately not rewarded, the policy gets no signal to adopt it
  unless a completion happens to use it *and* be correct.
- The odd per-step pattern (steps 2 and 4 ≈ 11–12 s and ≈ 12.2 GiB in both runs) comes from those prompts
  having 2 images, which roughly doubles the vision tokens in the training forward pass.
