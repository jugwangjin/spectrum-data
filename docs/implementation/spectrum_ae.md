# Spectral softmax autoencoder (deprecated, archived)

## 상태

- **`au_region_analysis.py`는 PyTorch를 import하지 않는다.** torch 없는 환경에서 메인 파이프라인만으로 동작한다.
- AE 구현은 **`src/deprecated/spectrum_autoencoder.py`** 에만 남겨 두었고, 참고·단독 실험용이다.
- **`au_region_analysis_ae.py`** 는 deprecated 래퍼이며 `au_region_analysis.main()`을 호출한다(경고만 추가).

## 예전 동작 요약 (아카이브)

- MLP 인코더 `n_wl→1024→256→128→K` 로짓, **softmax(K)** 병목, 대칭 디코더.
- 손실: 재구성 + β_kl·mean KL(q_i‖U) + β_ent·mean H(q) + β_batch·KL(q̄‖U) (전 픽셀 full-batch).
- 산출: 히트맵·mixing 스캐터·NPY·선택적 재구성 샘플 PNG 등은 과거 파이프라인에서 생성되었음.

## 다시 쓰려면

1. `torch` 환경에서 `deprecated/spectrum_autoencoder.py`의 `train_spectrum_softmax_ae`를 import.
2. 큐브 `y`, 성분 수 `K` 등에 맞춰 학습 후, 히트맵·meta 연동은 **별도 스크립트**에서 처리해야 한다(메인 CLI에 통합되어 있지 않음).
