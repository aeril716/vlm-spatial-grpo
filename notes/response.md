Qwen2.5-VL을 GRPO로 학습시켜 공간 추론 능력을 높이는 프로젝트(`vlm-spatial-grpo`)이고, 지금은 스모크 테스트 두 개를 끝낸 상태예요. 마지막 커밋은 9/29입니다. 이 프로젝트에 대해 저장해 둔 메모리는 없어서 git 기록과 `reports/smoke_test.md`를 보고 정리했어요.

## 완료한 것

**Part A: fp16 추론이 정상 동작하는지 확인 (SAT val 20개)**
- Quadro RTX 8000(sm_75)에서 fp16과 SDPA로 돌렸어요.
- fp16과 fp32 결과가 20개 모두 같았고, 비정상 logit(NaN/inf)은 0개였어요.
- VRAM은 fp16이 7.29 GiB, fp32가 14.23 GiB였어요.

**Part B: GRPO + LoRA를 5 step 돌려보기**
- 5 step 모두 NaN/inf 없이 끝났어요. 피크 VRAM은 12.16 GiB(47.3 GiB 중)이고, step당 8.1초 걸렸어요.
- LoRA 학습 파라미터는 약 2,990만 개(전체의 0.79%)이고, vision encoder는 완전히 고정했어요.
- 작업 중에 알게 된 것:
  - Qwen2.5-VL의 vision tower에도 `gate_proj`/`up_proj`/`down_proj`가 있어요. 그래서 `exclude_modules=".*visual.*"`가 없으면 LoRA가 vision encoder에도 붙어요.
  - `SAT_train.parquet`은 datasets streaming으로 읽히지 않아요. 그래서 `smoke/fetch_train_pool.py`를 만들어서 읽고, 결과를 `data/train_spatial_pool.pkl`로 저장했어요.
  - train split은 순서대로 정렬돼 있어서 0~127404번 행은 전부 "other" 유형이에요. 공간 관련 문제는 127405번부터 시작해요.

## 남은 문제
5 step 중 2 step은 gradient가 0이었어요. yes/no 문제에 맞으면 1, 틀리면 0을 주는 보상을 쓰다 보니, 한 그룹의 생성 8개가 전부 같은 점수를 받으면 `reward_std`가 0이 되고 advantage가 사라지기 때문이에요.

## 다음 단계 후보
1. 위의 zero-gradient 문제 해결. 방법은 그룹 크기 늘리기, 어려운 문제 위주로 뽑기, 부분 점수를 주는 보상 중에서 고르면 돼요.
2. 실제 학습 설정 정하고 본 학습 실행
3. README 채우기 (지금은 placeholder예요)

어디부터 할까요? 원하시면 이번 진행 상황을 메모리에 저장해서 다음에 바로 이어갈 수 있게 해둘게요.