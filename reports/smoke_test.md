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

---

# Part D: Part A + Part C with Qwen2.5-VL-7B-Instruct

Only the model changes: **Qwen/Qwen2.5-VL-7B-Instruct** instead of Qwen2.5-VL-3B-Instruct. Everything else is identical to
Part C: the project venv `.venv/` (torch 2.11.0+cu128, TRL 1.13.0), same samples, seed 0, LoRA config
(r16, decoder only, `exclude_modules='.*visual.*'`), GRPO settings, prompts, system prompt, reward and
measurements. The scripts are the Part A / Part C scripts, run unchanged with `SMOKE_MODEL_ID` set to the 7B
model. GPU: Quadro RTX 8000 (sm_75), 47.3 GiB.

## Step 0: fp16 sanity check (as Part A)

Same 20 SAT `val` samples (streaming, shuffle seed 0), greedy, max_new_tokens=24, SDPA.

| metric | fp16 | fp32 (reference) |
|---|---|---|
| peak VRAM (GiB) | 15.80 | 31.43 |
| total time (s) | 8.4 | 24.4 |
| seconds / sample | 0.42 | 1.22 |
| parse success | 20/20 | 20/20 |
| matches correct_answer | 11/20 | 11/20 |

- **fp16 vs fp32 agreement: 20/20**
- **fp16 non-finite (NaN/inf) logits: 0/20 sample(s)**

| # | question_type | imgs | fp16 answer | fp32 answer | correct_answer | fp16 NaN/inf | agree |
|---|---|---|---|---|---|---|---|
| 0 | goal_aim | 1 | right by 49 degrees | right by 49 degrees | right by 49 degrees | no | = |
| 1 | action_sequence | 2 | rotated right | rotated right | rotated left | no | = |
| 2 | action_consequence | 1 | yes | yes | no | no | = |
| 3 | action_sequence | 2 | rotated left and moved forward | rotated left and moved forward | rotated left and moved forward | no | = |
| 4 | action_consequence | 1 | yes | yes | no | no | = |
| 5 | goal_aim | 1 | left by 58 degrees | left by 58 degrees | left by 58 degrees | no | = |
| 6 | goal_aim | 1 | left by 29 degrees | left by 29 degrees | left by 29 degrees | no | = |
| 7 | obj_movement | 2 | no objects moved | no objects moved | no objects moved | no | = |
| 8 | goal_aim | 1 | right by 30 degrees | right by 30 degrees | left by 30 degrees | no | = |
| 9 | action_consequence | 1 | no | no | no | no | = |
| 10 | goal_aim | 1 | left by 41 degrees | left by 41 degrees | left by 41 degrees | no | = |
| 11 | action_sequence | 2 | rotated right and moved forward | rotated right and moved forward | did not move | no | = |
| 12 | action_consequence | 1 | no | no | yes | no | = |
| 13 | action_consequence | 1 | yes | yes | no | no | = |
| 14 | obj_movement | 2 | no objects moved | no objects moved | no objects moved | no | = |
| 15 | action_sequence | 2 | rotated right | rotated right | rotated right | no | = |
| 16 | goal_aim | 1 | left by 54 degrees | left by 54 degrees | left by 54 degrees | no | = |
| 17 | obj_movement | 2 | Bed was moved right and towards the camera in the first frame | Bed was moved right and towards the camera in the first frame | no objects moved | no | = |
| 18 | action_sequence | 2 | rotated right and moved forward | rotated right and moved forward | rotated right and moved forward | no | = |
| 19 | action_consequence | 1 | yes | yes | no | no | = |

## Steps 1 and 2: GRPO, 5 steps each

| | Step 1: baseline (short answer, max 24) | Step 2: reasoning format (max 1024) |
|---|---|---|
| status | completed 5/5 steps, no NaN/inf, no OOM | completed 5/5 steps, no NaN/inf, no OOM |

Each step's prompt was the same as in the 3B Part C runs: **yes**
(checked by comparing each step's `correct_answer`).

### Summary (mean over steps), 3B vs 7B

| | baseline 3B (Part C) | baseline 7B | reasoning 3B (Part C) | reasoning 7B |
|---|---|---|---|---|
| seconds / step | 7.9 | 10.6 | 11.5 | 11.1 |
| generation s / step | 2.0 | 2.9 | 5.3 | 2.9 |
| train fwd+bwd s / step | 5.8 | 7.6 | 6.0 | 8.1 |
| peak VRAM (GiB, max over steps) | 12.16 | 22.44 | 12.39 | 22.84 |
| completion tokens: mean | 3.8 | 4.9 | 16.3 | 3.8 |
| completion tokens: max | 7 | 15 | 191 | 7 |
| fraction cut off at max length | 0.000 | 0.000 | 0.000 | 0.000 |
| zero-gradient steps (reward std = 0) | 2/5 | 1/5 | 3/5 | 5/5 |

### Per step — Step 1: baseline, 7B

| step | s/step | gen s | score s | train (fwd+bwd) s | peak VRAM GiB | gen peak GiB | len mean | len max | cut off @max | all tags | reward mean | reward std | loss | grad_norm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 8.5 | 2.6 | 0.06 | 5.7 | 19.08 | 16.75 | 7.0 | 7 | 0.000 | n/a | 0.750 | 0.463 | 2.980e-08 | 0.6994 |
| 2 | 14.6 | 4.4 | 0.09 | 10.0 | 21.96 | 18.00 | 9.5 | 15 | 0.000 | n/a | 0.500 | 0.535 | 0.5415 | 1.0916 |
| 3 | 7.6 | 1.9 | 0.05 | 5.7 | 19.29 | 17.12 | 2.0 | 2 | 0.000 | n/a | 0.375 | 0.518 | 1.490e-08 | 3.0418 |
| 4 | 14.5 | 3.7 | 0.09 | 10.8 | 22.44 | 18.01 | 4.0 | 4 | 0.000 | n/a | 1.000 | 0.000 | 0.0000 | 0.0000 |
| 5 | 7.7 | 1.9 | 0.06 | 5.7 | 19.30 | 17.13 | 2.0 | 2 | 0.000 | n/a | 0.375 | 0.518 | 1.490e-08 | 3.0397 |

### Per step — Step 2: reasoning format, 7B

| step | s/step | gen s | score s | train (fwd+bwd) s | peak VRAM GiB | gen peak GiB | len mean | len max | cut off @max | all tags | reward mean | reward std | loss | grad_norm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 8.8 | 2.7 | 0.06 | 5.9 | 19.45 | 16.89 | 7.0 | 7 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 | 0.0000 |
| 2 | 15.3 | 3.8 | 0.09 | 11.4 | 22.82 | 18.23 | 4.0 | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 | 0.0000 |
| 3 | 8.0 | 2.0 | 0.06 | 5.9 | 19.65 | 17.26 | 2.0 | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 | 0.0000 |
| 4 | 15.4 | 3.9 | 0.09 | 11.4 | 22.84 | 18.24 | 4.0 | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 | 0.0000 |
| 5 | 8.1 | 2.0 | 0.06 | 6.0 | 19.67 | 17.27 | 2.0 | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 | 0.0000 |

Columns are measured exactly as in Part C (see the note under Part C's per-step tables).

### Step 2 format use (7B)

- Completions with all three tags parsed: **0/40**
- Completions that start with `<description>` (ignoring leading whitespace): **0/40**

### Two full Step 2 completions, 7B (unedited)

Same selection rule as Part C: the first completion with reward 1 and the first with reward 0,
scanning steps and generations in order (falls back to the next completions if one kind doesn't exist).

**Step 1, generation 0** — reward 0, 7 tokens, cut off: no, all three tags parsed: no, correct_answer: `right by 15 degrees`

```text
left by 15 degrees
```

**Step 1, generation 1** — reward 0, 7 tokens, cut off: no, all three tags parsed: no, correct_answer: `right by 15 degrees`

```text
left by 15 degrees
```

Both completions shown are reward 0. No Step 2 completion got reward 1, so the selection rule fell back to
the next completion in order. They are representative: every 7B Step 2 completion was a bare answer like these.

### Notes

- **The 7B model ignored the reasoning format completely: 0/40 completions used the tags, 0/40 started with
  `<description>`, and none added any explanation at all.** For comparison, 3B used all three tags in 2/40 and
  also never started with `<description>`. Every 7B completion was a bare 2–7-token answer, the same as its
  baseline. 24/40 of these bare answers were actually correct but received reward 0, because the reward only
  reads `<answer>`. So all 5 Step 2 steps had reward std 0 and grad_norm 0, and **no learning happened**.
  This also means Step 2 does not measure long-completion cost for 7B either. Its timing and memory are
  effectively a second baseline run, with the system prompt adding a few prompt tokens.
- **7B cost vs 3B, short completions (baseline):** 10.6 vs 7.9 s/step (1.3×). Generation time is 2.9 vs 2.0 s
  and train fwd+bwd is 7.6 vs 5.8 s. Peak VRAM is 22.44 vs 12.16 GiB, which leaves about 25 GiB of headroom on
  this 47.3 GiB GPU. Peak is again set by the training pass (gen peak ≤ 18.2 GiB).
- **Step 0:** fp16 is safe for 7B on sm_75: 20/20 agreement with fp32 and no NaN/inf. fp32 fit (31.43 GiB peak),
  so it was not skipped. On these 20 samples 7B matched `correct_answer` 11/20 vs 3B's 14/20. With n=20 this
  is noise-level and not a capability comparison.
- The baseline 7B run trained normally: 4/5 steps had non-zero gradient, and no NaN/inf or OOM.

---

# Part E: getting the model to use the reasoning format

Part C (3B) and Part D (7B) showed that a system prompt alone barely changes the output: 3B used all three
tags in 2/40 completions and 7B in 0/40. Part E changes only the prompt and repeats the Part C Step 2 run
for both models. The samples, seed, LoRA, GRPO settings, `max_completion_length=1024`, reward (exact match
inside `<answer>` only, 0 if it doesn't parse, format not rewarded) and measurements are unchanged.
GPU: Quadro RTX 8000 (sm_75), 47.3 GiB. Each run is a fresh process.

**E1: instruction in the user message, no system prompt.** The format instruction plus one worked example
about a different question go at the end of the user message, after the question and choices. With no system
message, the Qwen2.5-VL chat template inserts its default `You are a helpful assistant.`, the same as in
Parts B/C baseline. The full user message for the first training prompt:

```text
I need to go to blue colour armchair (near the mark 0 in the image). Which direction should I turn to face the object?

Choices:
- left by 62 degrees
- right by 62 degrees

First describe what in the image is relevant, then reason step by step, then give the final answer. Use exactly this format:
<description>...</description>
<reasoning>...</reasoning>
<answer>one of the choices, copied exactly</answer>

Example of the format, for a different question ("Is the lamp to the left of the sofa?" with choices yes / no):
<description>A grey sofa stands against the back wall. A floor lamp is next to its right arm.</description>
<reasoning>The lamp is beside the sofa's right arm, which is on the right side of the image from the camera's view. So the lamp is to the right of the sofa, not the left.</reasoning>
<answer>no</answer>
```

**E2: E1 + `<description>` prefill.** The prompt is the same as E1, but every completion is forced to begin
with `<description>`. How this is implemented in TRL 1.13:

- `GRPOTrainer` always renders conversational prompts with `apply_chat_template(..., add_generation_prompt=True)`.
  A trailing assistant message therefore can't act as a prefill: it would be closed with `<|im_end|>`, and a
  new assistant turn would start after it.
- Without `tools`, the trainer passes `chat_template=None`, so it uses the processor's own template.
- Before building the trainer, the generation-prompt clause of `processor.chat_template` (and
  `processor.tokenizer.chat_template`) is changed from
  `{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}` to
  `{% if add_generation_prompt %}<|im_start|>assistant\n<description>{% endif %}`.
  The script asserts that the rendered prompt ends with `<|im_start|>assistant\n<description>`. Tokenized, the
  prompt ends `<|im_start|> assistant \n < description >`.
- The prefill tokens are therefore part of `prompt_ids`: the model generates from after them, they aren't in
  `completion_ids`, they don't count toward completion length or the 1024 limit, and they get no loss.
- Because the completion text TRL passes to the reward does not include the prefill, the reward function
  prepends `<description>` before parsing tags and `<answer>`. The tag counts below do the same.

## Summary: Part E next to the Part C / Part D reasoning runs

Counts are over all 40 completions per run (5 steps × 8 generations). Timings and VRAM are means over steps,
except peak VRAM, which is the maximum.

| | 3B Part C | 3B-E1 | 3B-E2 | 7B Part D | 7B-E1 | 7B-E2 |
|---|---|---|---|---|---|---|
| prompt | system prompt | user msg + example | E1 + `<description>` prefill | system prompt | user msg + example | E1 + `<description>` prefill |
| seconds / step | 11.5 | 25.6 | 23.1 | 11.1 | 33.2 | 33.7 |
| generation s / step | 5.3 | 18.3 | 15.8 | 2.9 | 22.6 | 23.1 |
| train fwd+bwd s / step | 6.0 | 7.3 | 7.2 | 8.1 | 10.4 | 10.5 |
| peak VRAM (GiB, max over steps) | 12.39 | 16.87 | 14.29 | 22.84 | 24.33 | 23.93 |
| completion tokens: mean | 16.3 | 170.9 | 149.3 | 3.8 | 195.8 | 195.9 |
| completion tokens: max | 191 | 438 | 354 | 7 | 315 | 374 |
| all three tags | 2/40 | 39/40 | 39/40 | 0/40 | 40/40 | 38/40 |
| contains `<answer>` | 2/40 | 39/40 | 40/40 | 0/40 | 40/40 | 40/40 |
| cut off at 1024 | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 |
| reward 1 | 2/40 | 20/40 | 22/40 | 0/40 | 21/40 | 26/40 |
| zero-gradient steps (reward std = 0) | 3/5 | 0/5 | 0/5 | 5/5 | 0/5 | 0/5 |

## Per run

### 3B-E1

Status: completed 5/5 steps, no NaN/inf, no OOM

- Out of 40 completions: **39** have all three tags, **39** contain `<answer>`,
  **0** were cut off at 1024 tokens, 20 got reward 1.
- Completion length: mean **170.9**, max **438** tokens.

| step | s/step | gen s | score s | train (fwd+bwd) s | peak VRAM GiB | gen peak GiB | len mean | len max | cut off @max | all tags | reward mean | reward std | loss | grad_norm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 24.7 | 18.8 | 0.06 | 5.7 | 13.06 | 8.08 | 140.0 | 286 | 0.000 | 1.000 | 0.500 | 0.535 | -0.2238 | 0.3200 |
| 2 | 39.4 | 28.8 | 0.09 | 10.5 | 16.87 | 9.04 | 212.1 | 438 | 0.000 | 1.000 | 0.625 | 0.518 | 0.0918 | 0.2767 |
| 3 | 20.3 | 14.7 | 0.05 | 5.5 | 12.45 | 8.39 | 179.9 | 230 | 0.000 | 1.000 | 0.625 | 0.518 | 0.1656 | 0.3139 |
| 4 | 24.2 | 14.9 | 0.10 | 9.2 | 13.54 | 9.04 | 153.2 | 213 | 0.000 | 0.875 | 0.500 | 0.535 | -0.0580 | 0.3259 |
| 5 | 19.7 | 14.1 | 0.06 | 5.5 | 12.28 | 8.39 | 169.1 | 219 | 0.000 | 1.000 | 0.250 | 0.463 | 0.0371 | 0.3061 |

**Two full completions (unedited).** Same selection rule as Part C: the first completion with reward 1 and
the first with reward 0, scanning steps and generations in order (falls back to the next completions if one
kind doesn't exist).

**Step 1, generation 0** — reward 1, 141 tokens, cut off: no, all three tags parsed: yes, correct_answer: `right by 15 degrees`

```text
<description>There is a brown square dining table in the center of the image, and three chairs are positioned around it, with red numbers marking their positions: 13, 8, 5, and 3.</description>
<reasoning>Since the dining table is in the center of the image, my objective is to find its location by rotating my perspective. Looking at the table and the surrounding chairs, I can see that they are in a clockwise direction from my initial viewpoint. Therefore, to face the table, I need to turn left. I can turn 15 degrees towards the right side.</reasoning>
<answer>right by 15 degrees</answer>
```

**Step 1, generation 1** — reward 0, 106 tokens, cut off: no, all three tags parsed: yes, correct_answer: `right by 15 degrees`

```text
<description>The image shows a virtual room with a square dining table, chairs, and a window. Mark 5 is labeled on the table's surface.</description>
<reasoning>The dining table is the primary object of interest. It is centered in the image. The label "5" is directly attached to the table, making it easy to identify. Clicking on the table should likely lead to the actual investigation related to the given marks.</reasoning>
<answer>one of the choices, copied exactly</answer>
```


### 3B-E2

Status: completed 5/5 steps, no NaN/inf, no OOM

- Out of 40 completions: **39** have all three tags, **40** contain `<answer>`,
  **0** were cut off at 1024 tokens, 22 got reward 1.
- Completion length: mean **149.3**, max **354** tokens.

| step | s/step | gen s | score s | train (fwd+bwd) s | peak VRAM GiB | gen peak GiB | len mean | len max | cut off @max | all tags | reward mean | reward std | loss | grad_norm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 23.1 | 17.3 | 0.06 | 5.7 | 12.87 | 8.08 | 157.1 | 273 | 0.000 | 1.000 | 0.500 | 0.535 | -0.1763 | 0.3018 |
| 2 | 21.4 | 12.0 | 0.09 | 9.2 | 12.86 | 9.04 | 122.6 | 163 | 0.000 | 0.875 | 0.375 | 0.518 | 0.1691 | 0.3279 |
| 3 | 23.2 | 17.4 | 0.06 | 5.8 | 13.07 | 8.39 | 186.0 | 272 | 0.000 | 1.000 | 0.500 | 0.535 | 0.0415 | 0.2894 |
| 4 | 19.8 | 10.7 | 0.10 | 9.0 | 12.78 | 9.05 | 114.5 | 138 | 0.000 | 1.000 | 0.625 | 0.518 | 0.0221 | 0.3953 |
| 5 | 28.1 | 21.8 | 0.06 | 6.2 | 14.29 | 8.39 | 166.1 | 354 | 0.000 | 1.000 | 0.750 | 0.463 | -0.2068 | 0.2637 |

**Two full completions (unedited).** Same selection rule as Part C: the first completion with reward 1 and
the first with reward 0, scanning steps and generations in order (falls back to the next completions if one
kind doesn't exist). Shown exactly as generated. The prefilled `<description>` is part of the prompt, so it is not repeated here. Every completion below is preceded by it.

**Step 1, generation 0** — reward 1, 208 tokens, cut off: no, all three tags parsed: yes, correct_answer: `right by 15 degrees`

```text
On a tiled wall are three wood-framed windows. A long grey side table and a shorter one are next to a table beneath the windows. A square painting hangs over the side table, facing the viewer. The table is on a large floor and has four chairs. Spheres are placed on the table with numbers indicating their position above them.</description>
<reasoning>The brown square dining table is near the marker 5 in the image. The table is positioned in the middle of the scene, highlighted by the number 5. To face the dining table, one must move toward where the table is located, which is the center of the scene. However, the furniture arrangement around the table includes three chairs. To face the table squarely, it would be necessary to turn by an appropriate angle to align with the table's center. The space between the table, the chairs, and the walls indicates that we are to turn right to face the table directly.</reasoning>
<answer>right by 15 degrees</answer>
```

**Step 1, generation 1** — reward 0, 159 tokens, cut off: no, all three tags parsed: yes, correct_answer: `right by 15 degrees`

```text
Three red spots are placed on the table and its nearby chairs. Number 8 is behind chairs B and C, and there are two blue items on the wineadero.</description>
<reasoning>The dining table with red spots marked gives a clearer instruction. If the red spots are placed on different chairs, one of them marks the dining table or the chair. Number five (5) is beside the table, showing that it is facing or near the dining table. Considering the chair placements, and the fact that number five is close to the dining table,椅子 B and C must be part of the dining table setup, making number five a good candidate that's near an area that could potentially be the dining area.</reasoning>
<answer>left by 15 degrees</answer>
```


### 7B-E1

Status: completed 5/5 steps, no NaN/inf, no OOM

- Out of 40 completions: **40** have all three tags, **40** contain `<answer>`,
  **0** were cut off at 1024 tokens, 21 got reward 1.
- Completion length: mean **195.8**, max **315** tokens.

| step | s/step | gen s | score s | train (fwd+bwd) s | peak VRAM GiB | gen peak GiB | len mean | len max | cut off @max | all tags | reward mean | reward std | loss | grad_norm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 30.2 | 21.6 | 0.06 | 8.4 | 21.80 | 17.08 | 218.0 | 280 | 0.000 | 1.000 | 0.625 | 0.518 | -0.0942 | 0.2173 |
| 2 | 43.7 | 29.7 | 0.09 | 13.9 | 24.33 | 18.55 | 197.8 | 315 | 0.000 | 1.000 | 0.750 | 0.463 | 0.1631 | 0.2549 |
| 3 | 27.0 | 18.7 | 0.05 | 8.2 | 21.56 | 17.45 | 191.5 | 245 | 0.000 | 1.000 | 0.125 | 0.354 | -0.0175 | 0.2367 |
| 4 | 33.0 | 20.0 | 0.09 | 12.8 | 23.72 | 18.56 | 154.1 | 201 | 0.000 | 1.000 | 0.500 | 0.535 | -0.0205 | 0.2707 |
| 5 | 32.0 | 23.2 | 0.06 | 8.7 | 22.50 | 17.46 | 217.6 | 306 | 0.000 | 1.000 | 0.625 | 0.518 | -0.0542 | 0.2332 |

**Two full completions (unedited).** Same selection rule as Part C: the first completion with reward 1 and
the first with reward 0, scanning steps and generations in order (falls back to the next completions if one
kind doesn't exist).

**Step 1, generation 1** — reward 1, 276 tokens, cut off: no, all three tags parsed: yes, correct_answer: `right by 15 degrees`

```text
<description>The image shows a room with a tiled wall and a room divider. On the far left, there is a small table with a red marker labeled "13". In the center and right of the image, there is a wooden square dining table with a light blue and red object on it, surrounded by four wooden chairs. Red markers are placed on each chair, with the chair at the center of the table marked "5", and the chairs to its right marked "3" and "8". Another red marker labeled "13" is on the small table to the left. The dining table is in the center-right part of the image.</description>
<reasoning>To face the brown square dining table (near the mark 5), you should rotate towards the right. Since the chair marked "5" is at the center of the table and the landmark is labeled "5", you would need to turn right to face it directly. Given the choices, the correct direction to turn is "right by 15 degrees" because it matches the right turn required to face the correct landmark (5).

Choosing the other direction (left by 15 degrees) would mean you are moving away from the center point of the dining table, which is the goal to face it.
</reasoning>
<answer>right by 15 degrees</answer>
```

**Step 1, generation 0** — reward 0, 160 tokens, cut off: no, all three tags parsed: yes, correct_answer: `right by 15 degrees`

```text
<description>A room is depicted with a dining area in the center. There is a wooden dining table with four chairs arranged around it. The table has several objects on it, including a bottle, a bowl, and two red markers labeled 5 and 3. There is a wall with a window on the left and a partial view of another table and wall on the right side. A painting hangs on the left wall.</description>
<reasoning>Directly positioned in front of the table with the display of markers 5 and 3, is to the left convex part of the orientation of the dinette.  Hence, you'll need to make a small turn to align perfectly in front of the table.</reasoning>
<answer>left by 15 degrees</answer>
```


### 7B-E2

Status: completed 5/5 steps, no NaN/inf, no OOM

- Out of 40 completions: **38** have all three tags, **40** contain `<answer>`,
  **0** were cut off at 1024 tokens, 26 got reward 1.
- Completion length: mean **195.9**, max **374** tokens.

| step | s/step | gen s | score s | train (fwd+bwd) s | peak VRAM GiB | gen peak GiB | len mean | len max | cut off @max | all tags | reward mean | reward std | loss | grad_norm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 26.5 | 18.3 | 0.06 | 8.1 | 21.09 | 17.08 | 184.5 | 233 | 0.000 | 1.000 | 0.375 | 0.518 | -0.0007 | 0.2377 |
| 2 | 34.4 | 21.2 | 0.10 | 13.1 | 23.81 | 18.57 | 194.1 | 216 | 0.000 | 1.000 | 0.750 | 0.463 | 0.0177 | 0.3027 |
| 3 | 33.3 | 24.4 | 0.06 | 8.9 | 22.74 | 17.46 | 220.5 | 322 | 0.000 | 0.875 | 0.625 | 0.518 | -0.0060 | 0.2185 |
| 4 | 36.3 | 23.0 | 0.09 | 13.1 | 23.93 | 18.58 | 168.6 | 236 | 0.000 | 1.000 | 0.875 | 0.354 | 0.1412 | 0.2670 |
| 5 | 37.9 | 28.4 | 0.06 | 9.4 | 23.54 | 17.46 | 211.8 | 374 | 0.000 | 0.875 | 0.625 | 0.518 | -0.0744 | 0.2378 |

**Two full completions (unedited).** Same selection rule as Part C: the first completion with reward 1 and
the first with reward 0, scanning steps and generations in order (falls back to the next completions if one
kind doesn't exist). Shown exactly as generated. The prefilled `<description>` is part of the prompt, so it is not repeated here. Every completion below is preceded by it.

**Step 1, generation 2** — reward 1, 154 tokens, cut off: no, all three tags parsed: yes, correct_answer: `right by 15 degrees`

```text
There is a brown square dining table with a number 5 on it. Several chairs are arranged around the table, and one of the chairs has a number 3 on it. There is another table next to it with a number 13 on it, and a window and a barbecue grill can be seen in the background.</description>
<reasoning>To move from your position towards the table marked with the number 5, you should turn right. You are currently facing the walls, and the table is to your right. To face the brown square dining table (marked with the number 5), you need to turn right by 15 degrees to align yourself with the table.</reasoning>
<answer>right by 15 degrees</answer>
```

**Step 1, generation 0** — reward 0, 189 tokens, cut off: no, all three tags parsed: yes, correct_answer: `right by 15 degrees`

```text
There is a dining table near the center of the image, marked with the number "5". The table is rectangular and surrounded by four wooden chairs. The floor is plain and light-colored, and there is a small table with a red object on it in the upper left corner, marked with the number "13". The wall is tiled with beige tiles, and there is a painting hanging on the wall next to the window.</description>
<reasoning>From your position facing away from the table, you need to turn to face the table, which is to your left from this perspective. Since you need to face the table, you should turn your body to the left by approximately 90 degrees. To achieve this, you would turn left by a smaller angle around 15 degrees, as you are already in a position to be facing the table.</reasoning>
<answer>left by 15 degrees</answer>
```

### Notes

- **The format now works for both models.** Moving the instruction plus one example into the user message (E1)
  took tag use from 2/40 to 39/40 (3B) and from 0/40 to 40/40 (7B). The `<description>` prefill (E2) added
  little on top: 39/40 and 38/40 have all three tags, and 40/40 contain `<answer>` in both. Every step in all
  four runs had reward variance, so there were no zero-gradient steps (Part C: 3/5, Part D: 5/5). Reward 1
  counts (20–26/40) are about the same as the bare-answer accuracy, which is expected after 5 steps. The
  E1 vs E2 differences are within noise at n=40.
- **Completions are longer but still nowhere near 1024 tokens.** Means are 149–196 tokens and the maximum is 438,
  with 0/160 cut off. So these numbers measure cost at about 150–450 tokens, not at the 1024 limit:
  - s/step is 23–26 for 3B (vs. 7.9 baseline, ~3×) and 33–34 for 7B (vs. 10.6, ~3×).
  - Generation is now most of the step: 16–18 s (3B) and 23 s (7B).
  - Train fwd+bwd rose only ~25–30% (3B 5.8 → 7.2–7.3 s, 7B 7.6 → 10.4–10.5 s).
  - Peak VRAM rose by 2.1–4.7 GiB for 3B (max 16.87 GiB) and 1.5–1.9 GiB for 7B (max 24.33 GiB), still under
    about half of the 47.3 GiB GPU. Generation peak stayed ≤ 9.1 GiB (3B) and ≤ 18.6 GiB (7B), so the
    training pass still sets the peak.
- **Generation time tracks the longest completion in the group.** Measured `gen s / longest completion
  tokens` per step is 62–78 ms/token for 3B and 76–100 ms/token for 7B. *Extrapolation, not measured:* a
  group where one completion runs to 1024 tokens would spend roughly 65–80 s (3B) and 80–100 s (7B) per step
  just on generation. Memory at that length remains unmeasured.
- **3B sometimes copies the template's placeholder as its answer.** `<answer>one of the choices, copied
  exactly</answer>` appears verbatim in 6/40 (E1) and 4/40 (E2) of the 3B completions (see the 3B-E1 reward 0
  example above). 7B never does this. Replacing the placeholder with something that can't be copied (e.g.
  `<answer>[the exact text of one choice]</answer>`) or dropping it in favour of the worked example would
  likely remove these failures.
- The remaining untagged completions: 1 from 3B-E1 used `description:` / `reasoning:` / `answer:` instead of tags.
  1 from 3B-E2 had `<answer>` but never closed `</reasoning>`, and so did 2 from 7B-E2. No E2 completion
  repeated `<description>`, which confirms that the prefill was in the prompt and the model continued from it.

---

# Part F: base-model pass@k diagnosis (no training)

Base models with no adapter, fp16, SDPA, on Quadro RTX 8000 (sm_75). For each question: **1 greedy answer** and
**k = 8 sampled answers** at the GRPO sampling settings: TRL's `GRPOConfig` defaults are `temperature=1.0`,
`top_p=1.0`, `top_k=0` and `repetition_penalty=1.0`, and they replace the model's own generation_config,
whose `repetition_penalty=1.05` is therefore not used. `max_new_tokens=1024` for both greedy and sampled
answers. Score = exact match of the text inside `<answer>` with correct_answer (same normalisation as Parts B–E), 0 if
`<answer>` doesn't parse. Generation stops on NaN/inf logits (checked at every decode step) or OOM; neither
happened.

## Questions

Pool: `data/train_spatial_pool.pkl`, the first 256 spatial rows of the SAT train split (row IDs
127405–127660; the row ID is the row index in
`SAT_train.parquet`). This is only the first slice of the spatial part of the split, not all of it.

| question_type | questions in pool | answer choices | sampled |
|---|---|---|---|
| action_consequence | 110 | 2 | 54 |
| action_sequence | 48 | 2 | 48 |
| goal_aim | 50 | 2 | 50 |
| obj_movement | 48 | 2 | 48 |
| **total** | 256 | | **200** |

Every question in the pool has exactly 2 answer choices. Stratified sample with seed 0: equal per
type as far as possible. The two 48-question types are taken whole and the remaining 104 are split between
the other two, capped at goal_aim's 50. The row IDs are saved in `data/passk_f_ids.json`, total and by type.

## Prompt

Part E E1 prompt (instruction + worked example at the end of the user message, no system prompt). The only
change is that `<answer>one of the choices, copied exactly</answer>` became `<answer>...</answer>`. The full user text
for the first question:

```text
Were any of the objects in the initial frame that you can still see in the second frame moved from their original positions?

Choices:
- green tapered square potted houseplant was moved right and towards the camera in the first frame
- green tapered square potted houseplant was moved left and away from the camera in the first frame

First describe what in the image is relevant, then reason step by step, then give the final answer. Use exactly this format:
<description>...</description>
<reasoning>...</reasoning>
<answer>...</answer>

Example of the format, for a different question ("Is the lamp to the left of the sofa?" with choices yes / no):
<description>A grey sofa stands against the back wall. A floor lamp is next to its right arm.</description>
<reasoning>The lamp is beside the sofa's right arm, which is on the right side of the image from the camera's view. So the lamp is to the right of the sofa, not the left.</reasoning>
<answer>no</answer>
```

## Summary

| | 3B-Instruct | 7B-Instruct |
|---|---|---|
| questions answered | 200 (ok) | 200 (ok) |
| greedy accuracy | 121/200 (60%) | 119/200 (60%) |
| 0/8 | 19/200 (10%) | 9/200 (4%) |
| mixed 1–7/8 | 180/200 (90%) | 179/200 (90%) |
| 8/8 | 1/200 (0%) | 12/200 (6%) |
| sampled answers correct | 43.1% | 52.2% |
| sampled: all three tags | 89.2% | 96.8% |
| sampled: cut off at 1024 | 0/1600 | 0/1600 |
| time per question | 8.8 s | 11.7 s |
| peak VRAM | 14.84 GiB | 27.11 GiB |

## Qwen2.5-VL-3B-Instruct

Status: **ok** · 200 questions answered

### 1. Greedy accuracy

| question_type | n | greedy correct |
|---|---|---|
| action_consequence | 54 | 35/54 (65%) |
| action_sequence | 48 | 28/48 (58%) |
| goal_aim | 50 | 33/50 (66%) |
| obj_movement | 48 | 25/48 (52%) |
| **all** | 200 | **121/200 (60%)** |

### 2. pass@8 buckets (8 samples per question)

| question_type | n | 0/8 | mixed 1–7/8 | 8/8 | mean correct / 8 |
|---|---|---|---|---|---|
| action_consequence | 54 | 2/54 (4%) | 51/54 (94%) | 1/54 (2%) | 4.41 |
| action_sequence | 48 | 1/48 (2%) | 47/48 (98%) | 0/48 (0%) | 3.42 |
| goal_aim | 50 | 2/50 (4%) | 48/50 (96%) | 0/50 (0%) | 3.72 |
| obj_movement | 48 | 14/48 (29%) | 34/48 (71%) | 0/48 (0%) | 2.12 |
| **all** | 200 | 19/200 (10%) | 180/200 (90%) | 1/200 (0%) | 3.45 |

### 3. Correct-count distribution vs. pure guessing

Expected = sum over questions of Binomial(8, p = 1 / number of choices). Every question in this sample has 2 choices, so p = 0.5 throughout.

| correct / 8 | observed questions | expected if guessing |
|---|---|---|
| 0 | 19 | 0.8 |
| 1 | 15 | 6.2 |
| 2 | 27 | 21.9 |
| 3 | 37 | 43.8 |
| 4 | 39 | 54.7 |
| 5 | 34 | 43.8 |
| 6 | 20 | 21.9 |
| 7 | 8 | 6.2 |
| 8 | 1 | 0.8 |

- Mean correct per question: observed **3.45**, guessing 4.00
- Variance of correct counts: observed **3.63**, binomial (guessing) **2.00** → ratio 1.81× (population variance, n = 200)

### 4. Answer bias

`picked` is over all 8 sampled answers per question. `correct` is the share of questions whose correct_answer has that label. The last two columns count, among questions whose correct label is this one, how many scored 0/8 and 8/8. Labels: goal_aim → left / right / straight; obj_movement → direction + towards/away, or no objects moved; the other types use the choice text.

**action_consequence** (54 questions, 432 sampled answers)

| label | picked | correct | correct-label questions: 0/8 | 8/8 |
|---|---|---|---|---|
| no | 55% (239) | 56% (30) | 0/30 | 1/30 |
| yes | 36% (155) | 43% (23) | 1/23 | 0/23 |
| modern stainless steel refrigerator(marked 0 in the image) | 0% (0) | 2% (1) | 1/1 | 0/1 |
| blue colour sofa(near the mark 6 in the image) | 0% (2) | 0% (0) | — | — |
| (no `<answer>`) | 4% (19) | 0% (0) | — | — |
| (not one of the choices) | 4% (17) | 0% (0) | — | — |

Position: picks the first-listed choice in 55% of samples; the first-listed choice is correct in 100% of questions.
Most common answers that are not one of the choices: `the floating cabinet will move further from the camera if i rotated right and moved forward` ×1, `hanging pair of shoes (not shown) will move further away from the camera` ×1, `modern stainless steel refrigerator (marked 0 in the image)` ×1.

**action_sequence** (48 questions, 384 sampled answers)

| label | picked | correct | correct-label questions: 0/8 | 8/8 |
|---|---|---|---|---|
| rotated right and moved forward | 18% (69) | 31% (15) | 0/15 | 0/15 |
| rotated left | 14% (52) | 25% (12) | 0/12 | 0/12 |
| did not move | 9% (35) | 17% (8) | 1/8 | 0/8 |
| rotated left and moved forward | 10% (39) | 17% (8) | 0/8 | 0/8 |
| rotated right | 12% (46) | 10% (5) | 0/5 | 0/5 |
| rotated left and rotated left | 1% (2) | 0% (0) | — | — |
| (no `<answer>`) | 12% (45) | 0% (0) | — | — |
| (not one of the choices) | 25% (96) | 0% (0) | — | — |

Position: picks the first-listed choice in 43% of samples; the first-listed choice is correct in 100% of questions.
Most common answers that are not one of the choices: `no` ×25, `rotated left and moved forward` ×7, `rotated right` ×3.

**goal_aim** (50 questions, 400 sampled answers)

| label | picked | correct | correct-label questions: 0/8 | 8/8 |
|---|---|---|---|---|
| right | 30% (121) | 52% (26) | 0/26 | 0/26 |
| left | 34% (135) | 34% (17) | 0/17 | 0/17 |
| straight | 8% (34) | 14% (7) | 2/7 | 0/7 |
| (no `<answer>`) | 2% (8) | 0% (0) | — | — |
| (not one of the choices) | 26% (102) | 0% (0) | — | — |

Position: picks the first-listed choice in 46% of samples; the first-listed choice is correct in 100% of questions.
Most common answers that are not one of the choices: `right by 90 degrees` ×11, `left by 90 degrees` ×10, `no` ×6.

**obj_movement** (48 questions, 384 sampled answers)

| label | picked | correct | correct-label questions: 0/8 | 8/8 |
|---|---|---|---|---|
| no objects moved | 26% (98) | 62% (30) | 0/30 | 0/30 |
| moved left + towards | 2% (7) | 17% (8) | 5/8 | 0/8 |
| moved right + towards | 0% (1) | 17% (8) | 7/8 | 0/8 |
| moved left + away | 3% (12) | 2% (1) | 1/1 | 0/1 |
| moved right + away | 2% (9) | 2% (1) | 1/1 | 0/1 |
| (no `<answer>`) | 20% (77) | 0% (0) | — | — |
| (not one of the choices) | 47% (180) | 0% (0) | — | — |

Position: picks the first-listed choice in 27% of samples; the first-listed choice is correct in 100% of questions.
Most common answers that are not one of the choices: `yes` ×57, `no` ×36, `yes objects moved` ×3.

### 5. Format

| | greedy | sampled |
|---|---|---|
| all three tags | 198/200 (99%) | 1427/1600 (89%) |
| contains `<answer>` | 198/200 (99%) | 1451/1600 (91%) |
| cut off at 1024 | 2/200 (1%) | 0/1600 (0%) |
| completion tokens mean / max | 143 / 1024 | 158 / 694 |

### 6. Time and memory

| | |
|---|---|
| total wall time | 29.3 min |
| time per question (greedy + 8 samples) | 8.8 s |
| of which greedy / sampling | 1.1 s / 7.7 s |
| peak VRAM | 14.84 GiB |

Questions are generated 8 at a time, i.e. 8 greedy sequences and then 8×8 sampled sequences per `generate()` call. Per-question times are wall time divided by the number of questions, not the latency of one question alone.

### Two sampled answers (unedited)

Same selection rule as Part C: the first sampled answer that scored 1 and the first that scored 0, scanning questions in file order and samples in order.

**row_id 127406 (action_sequence), sample 0**: score 1, 118 tokens, all three tags: yes, correct_answer: `rotated left`, choices: ['rotated left', 'rotated right']

```text
<description>The image shows an interior room with a table in the foreground, a laptop on the table, and a mountain scene on a framed picture in the background. The room has a dark-colored wall.</description>
<reasoning>The position of the table, the laptop, and the framed picture have shifted from their original positions. The table is now to the right side of the image, the laptop has moved slightly, and the framed picture with the mountain scene is now on the floor to the left side of the room.</reasoning>
<answer>rotated left</answer>
```

**row_id 127405 (obj_movement), sample 0**: score 0, 128 tokens, all three tags: no, correct_answer: `green tapered square potted houseplant was moved right and towards the camera in the first frame`, choices: ['green tapered square potted houseplant was moved right and towards the camera in the first frame', 'green tapered square potted houseplant was moved left and away from the camera in the first frame']

```text
description:The image shows a living room with features including a potted plant next to a yellow sectional couch, a coffee table in the corner with two bottles of wine, and pictures on the wall.

reasoning:In the initial frame,green tapered square potted houseplant is sitting on a table. By the second frame, the potted houseplant is still on the table but has been moved slightly to the right and closer to the camera. This results in a reduced size of the plant in the second frame compared to the first frame.

answer:green tapered square potted houseplant was moved right and towards the camera in the first frame
```

## Qwen2.5-VL-7B-Instruct

Status: **ok** · 200 questions answered

### 1. Greedy accuracy

| question_type | n | greedy correct |
|---|---|---|
| action_consequence | 54 | 29/54 (54%) |
| action_sequence | 48 | 33/48 (69%) |
| goal_aim | 50 | 40/50 (80%) |
| obj_movement | 48 | 17/48 (35%) |
| **all** | 200 | **119/200 (60%)** |

### 2. pass@8 buckets (8 samples per question)

| question_type | n | 0/8 | mixed 1–7/8 | 8/8 | mean correct / 8 |
|---|---|---|---|---|---|
| action_consequence | 54 | 2/54 (4%) | 48/54 (89%) | 4/54 (7%) | 4.41 |
| action_sequence | 48 | 2/48 (4%) | 40/48 (83%) | 6/48 (12%) | 4.77 |
| goal_aim | 50 | 1/50 (2%) | 48/50 (96%) | 1/50 (2%) | 4.62 |
| obj_movement | 48 | 4/48 (8%) | 43/48 (90%) | 1/48 (2%) | 2.88 |
| **all** | 200 | 9/200 (4%) | 179/200 (90%) | 12/200 (6%) | 4.18 |

### 3. Correct-count distribution vs. pure guessing

Expected = sum over questions of Binomial(8, p = 1 / number of choices). Every question in this sample has 2 choices, so p = 0.5 throughout.

| correct / 8 | observed questions | expected if guessing |
|---|---|---|
| 0 | 9 | 0.8 |
| 1 | 22 | 6.2 |
| 2 | 24 | 21.9 |
| 3 | 23 | 43.8 |
| 4 | 25 | 54.7 |
| 5 | 33 | 43.8 |
| 6 | 28 | 21.9 |
| 7 | 24 | 6.2 |
| 8 | 12 | 0.8 |

- Mean correct per question: observed **4.18**, guessing 4.00
- Variance of correct counts: observed **5.04**, binomial (guessing) **2.00** → ratio 2.52× (population variance, n = 200)

### 4. Answer bias

`picked` is over all 8 sampled answers per question. `correct` is the share of questions whose correct_answer has that label. The last two columns count, among questions whose correct label is this one, how many scored 0/8 and 8/8. Labels: goal_aim → left / right / straight; obj_movement → direction + towards/away, or no objects moved; the other types use the choice text.

**action_consequence** (54 questions, 432 sampled answers)

| label | picked | correct | correct-label questions: 0/8 | 8/8 |
|---|---|---|---|---|
| no | 72% (311) | 56% (30) | 0/30 | 4/30 |
| yes | 25% (108) | 43% (23) | 1/23 | 0/23 |
| modern stainless steel refrigerator(marked 0 in the image) | 0% (0) | 2% (1) | 1/1 | 0/1 |
| blue colour sofa(near the mark 6 in the image) | 0% (2) | 0% (0) | — | — |
| (no `<answer>`) | 1% (4) | 0% (0) | — | — |
| (not one of the choices) | 2% (7) | 0% (0) | — | — |

Position: picks the first-listed choice in 55% of samples; the first-listed choice is correct in 100% of questions.
Most common answers that are not one of the choices: `modern stainless steel refrigerator` ×2, `blue colour sofa (near the mark 6 in the image)` ×2, `the modern stainless steel refrigerator (marked 0) will move further from the camera if you rotated right and moved forward` ×1.

**action_sequence** (48 questions, 384 sampled answers)

| label | picked | correct | correct-label questions: 0/8 | 8/8 |
|---|---|---|---|---|
| rotated right and moved forward | 30% (115) | 31% (15) | 0/15 | 3/15 |
| rotated left | 22% (85) | 25% (12) | 0/12 | 2/12 |
| did not move | 3% (12) | 17% (8) | 2/8 | 0/8 |
| rotated left and moved forward | 18% (71) | 17% (8) | 0/8 | 1/8 |
| rotated right | 14% (52) | 10% (5) | 0/5 | 0/5 |
| rotated left and rotated left | 4% (14) | 0% (0) | — | — |
| rotated right and rotated right | 0% (1) | 0% (0) | — | — |
| (no `<answer>`) | 1% (2) | 0% (0) | — | — |
| (not one of the choices) | 8% (32) | 0% (0) | — | — |

Position: picks the first-listed choice in 60% of samples; the first-listed choice is correct in 100% of questions.
Most common answers that are not one of the choices: `rotated left and moved forward` ×10, `rotated left` ×4, `rotated right and moved forward` ×3.

**goal_aim** (50 questions, 400 sampled answers)

| label | picked | correct | correct-label questions: 0/8 | 8/8 |
|---|---|---|---|---|
| right | 38% (153) | 52% (26) | 0/26 | 1/26 |
| left | 30% (120) | 34% (17) | 0/17 | 0/17 |
| straight | 22% (87) | 14% (7) | 1/7 | 0/7 |
| (no `<answer>`) | 2% (9) | 0% (0) | — | — |
| (not one of the choices) | 8% (31) | 0% (0) | — | — |

Position: picks the first-listed choice in 58% of samples; the first-listed choice is correct in 100% of questions.
Most common answers that are not one of the choices: `left by 90 degrees` ×9, `right by 90 degrees` ×8, `left` ×2.

**obj_movement** (48 questions, 384 sampled answers)

| label | picked | correct | correct-label questions: 0/8 | 8/8 |
|---|---|---|---|---|
| no objects moved | 29% (112) | 62% (30) | 1/30 | 1/30 |
| moved left + towards | 10% (40) | 17% (8) | 0/8 | 0/8 |
| moved right + towards | 10% (40) | 17% (8) | 2/8 | 0/8 |
| moved left + away | 9% (35) | 2% (1) | 0/1 | 0/1 |
| moved right + away | 8% (32) | 2% (1) | 1/1 | 0/1 |
| (no `<answer>`) | 2% (7) | 0% (0) | — | — |
| (not one of the choices) | 31% (118) | 0% (0) | — | — |

Position: picks the first-listed choice in 36% of samples; the first-listed choice is correct in 100% of questions.
Most common answers that are not one of the choices: `no objects were moved` ×2, `yes objects moved` ×2, `yes, some objects moved` ×2.

### 5. Format

| | greedy | sampled |
|---|---|---|
| all three tags | 200/200 (100%) | 1549/1600 (97%) |
| contains `<answer>` | 200/200 (100%) | 1578/1600 (99%) |
| cut off at 1024 | 0/200 (0%) | 0/1600 (0%) |
| completion tokens mean / max | 164 / 268 | 193 / 498 |

### 6. Time and memory

| | |
|---|---|
| total wall time | 39.0 min |
| time per question (greedy + 8 samples) | 11.7 s |
| of which greedy / sampling | 1.5 s / 10.2 s |
| peak VRAM | 27.11 GiB |

Questions are generated 8 at a time, i.e. 8 greedy sequences and then 8×8 sampled sequences per `generate()` call. Per-question times are wall time divided by the number of questions, not the latency of one question alone.

### Two sampled answers (unedited)

Same selection rule as Part C: the first sampled answer that scored 1 and the first that scored 0, scanning questions in file order and samples in order.

**row_id 127405 (obj_movement), sample 2**: score 1, 131 tokens, all three tags: yes, correct_answer: `green tapered square potted houseplant was moved right and towards the camera in the first frame`, choices: ['green tapered square potted houseplant was moved right and towards the camera in the first frame', 'green tapered square potted houseplant was moved left and away from the camera in the first frame']

```text
<description>In the first frame, a green tapered square potted houseplant is on a yellow surface. In the second frame, the plant appears to be on the same surface but slightly closer to the camera.</description>
<reasoning>The plant is more prominent in the second frame, indicating it has moved closer to the camera. The position of the plant on the surface remains consistent between the two frames, as does the surface itself. This shows a positional change rather than the plant being moved to a different surface.</reasoning>
<answer>green tapered square potted houseplant was moved right and towards the camera in the first frame</answer>
```

**row_id 127405 (obj_movement), sample 0**: score 0, 155 tokens, all three tags: yes, correct_answer: `green tapered square potted houseplant was moved right and towards the camera in the first frame`, choices: ['green tapered square potted houseplant was moved right and towards the camera in the first frame', 'green tapered square potted houseplant was moved left and away from the camera in the first frame']

```text
<description>In the first frame, a green tapered square potted houseplant is visible near the corner of a table, slightly to the right of the camera's view. In the second frame, the same houseplant appears closer to the left side of the frame towards the camera, indicating it has been moved closer to the camera and shifted slightly to the left.</description>

<reasoning>The plant in the first frame is on the right side of the camera's view, while in the second frame it is closer to the center and slightly more to the left. This suggests the plant has been moved leftwards and towards the camera.</reasoning>

<answer>green tapered square potted houseplant was moved left and away from the camera in the first frame</answer>
```


Raw answers for every question (greedy + 8 samples, full text, with scores, token counts and cut-off flags)
are in `reports/raw/part_f/<model>.jsonl`, with run metadata in `reports/raw/part_f/<model>_meta.json`.

## Notes

- **The correct answer is always the first-listed choice in this pool.** In all 256 questions of
  `data/train_spatial_pool.pkl`, `correct_answer == answers[0]` (checked directly on the pickle, not only on the
  200 sampled questions). No script in Parts B–F shuffles the choices, so every training prompt so far listed the
  correct answer first. Two consequences:
  - In this sample, "picks the first-listed choice" and "is correct" are the same event. That is why the
    Position lines say 100%. The accuracies above therefore can't separate spatial ability from a preference for
    the first option.
  - In GRPO training on this data without shuffling, "always answer the first choice" earns reward 1 on every
    question. That is a shortcut the policy can learn instead of spatial reasoning. Shuffling the choice order
    per prompt (and per epoch) before real training would close it. Nothing was changed here. This part only
    reports it.
- **Variance ratio.** If each model guessed uniformly, each question's correct count would be Binomial(8, 0.5),
  with variance 2.00. The observed variance is 1.81× that for 3B and 2.52× for 7B, and both the 0/8 and 8/8 tails
  are much heavier than guessing predicts (3B: 19 vs 0.8 at 0/8; 7B: 9 vs 0.8 at 0/8 and 12 vs 0.8 at 8/8). So
  per-question success rates differ: the models are consistently right on some questions and consistently wrong on
  others, rather than flipping a coin everywhere. The mean, however, is near or below chance (3B 3.45/8, 7B
  4.18/8). The 3B mean is pulled down by answers that are not one of the choices or have no `<answer>`.
- **GRPO signal.** With 8 generations per prompt, a group only produces a gradient when its rewards are mixed.
  That is the case for 90% of questions for both models (180/200 and 179/200). On these 200 questions, reward
  diversity is not the bottleneck for either model.
- **"no"/"yes" answers that aren't choices.** The worked example in the prompt ends with `<answer>no</answer>`.
  Both models sometimes answer `no`/`yes` to questions where neither is a choice. This is mostly 3B: obj_movement
  `yes` ×57 and `no` ×36, action_sequence `no` ×25, goal_aim `no` ×6. 7B does it far less. This looks like the
  example's yes/no answer leaking into other question types. The example was kept unchanged as instructed. Those
  answers score 0, and they are the main reason for 3B's 29% 0/8 rate on obj_movement.
- **Answers that paraphrase a choice** (e.g. `right by 90 degrees` for goal_aim, `rotated left and moved forward`
  when that wasn't offered, a choice missing its `(marked 0 in the image)` suffix) score 0 under exact match. This
  is the same scoring as the GRPO reward.
- **Format.** Sampled answers had all three tags in 89% (3B) and 97% (7B) of cases. Nothing sampled was cut off.
  Two 3B greedy answers ran to the 1024 limit, both as repetition loops (greedy uses `repetition_penalty=1.0`, as
  specified).
- **Batching.** Greedy decoding was run 8 questions per `generate()` call with left padding. In fp16, padded
  batches can produce slightly different logits than single-question runs, so a greedy answer could in rare cases
  differ from an unbatched run. Sampling is random anyway (seed 0 set once per model run).
- **Pool coverage.** The pool is only the first 256 spatial rows of SAT train (row IDs 127405–127660). Two types
  are taken whole (48 each), so the per-type results describe this slice, not the full split. The pool contains no
  question types other than these four, and every question has exactly 2 choices.
