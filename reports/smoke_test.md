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
