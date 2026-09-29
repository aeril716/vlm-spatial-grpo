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
