# Au region 스펙트럼 큐브 분석 — 사용법

## Python 환경

Windows에서 Anaconda base 예시:

`C:\Users\gwangjin\AppData\Local\anaconda3\python.exe`

(`where python`으로 본 경로와 동일하게 쓰면 됩니다.)

의존성:

```text
pip install -r requirements.txt
```

## 실행

프로젝트 루트에서:

```text
python src/au_region_analysis.py 40_SC_20per_1s_sample_77K_Au_region.txt.txt
```

기본 출력 폴더는 입력 파일명의 stem과 동일합니다(예: `40_SC_20per_1s_sample_77K_Au_region.txt.txt` → `40_SC_20per_1s_sample_77K_Au_region.txt/`). 다른 경로를 쓰려면 `--out 디렉터리`를 지정합니다.

- **기본**은 픽셀별 PNG 1600장을 **쓰지 않습니다**(HTML 호버로 스펙트럼). 파일이 필요하면 `--pixel-pngs`.

```text
python src/au_region_analysis.py ... --pixel-pngs
```

- **K-means / NMF / FastICA**(기본: 성분 10·**K-means**·**L2 행 정규화**; 끄려면 `--ica-n-components 0`):

```text
python src/au_region_analysis.py 입력.txt.txt
python src/au_region_analysis.py 입력.txt.txt --spectral-kmeans-normalize none
python src/au_region_analysis.py 입력.txt.txt --ica-method nmf
python src/au_region_analysis.py 입력.txt.txt --ica-method nmf --no-nmf-envelope-residual
python src/au_region_analysis.py 입력.txt.txt --ica-method fastica
python src/au_region_analysis.py 입력.txt.txt --ica-n-components 0
```

- x축을 파장 대신 `1..N` 인덱스로 그리려면 `--x-index` 를 추가합니다.

## 결과물 (기본: 입력 stem 이름의 폴더)

| 파일 | 설명 |
|------|------|
| `index.html` | activation·스펙트럼 분해·modal RMS·피크/골 맵 등 40×40 격자; 호버 시 스펙트럼 팝업. `meta`에 맵·`ica`(분해 실행 시) 포함 |
| `spectra.bin` | 동일 큐브의 바이너리 사본 |
| `meta.json` | 파장, 활성도, ROI 피크/골 목록·개수, clear max/min, 주도 피크·골 파장/폭 |
| `heatmap_activation_40x40.png` | ROI 피크+골 prominence 점수 + colorbar(백분위 + **기본 로그** 스케일; `--no-heatmap-activation-log` 로 선형) |
| `heatmap_valley_wavelength_40x40.png` | 주도 골 파장(nm) + colorbar |
| `heatmap_valley_width_40x40.png` | 주도 골 폭(nm) + colorbar |
| `heatmap_peak_wavelength_40x40.png` | ROI 주도 피크 파장(nm) + colorbar |
| `heatmap_peak_width_40x40.png` | 주도 피크 폭(nm) + colorbar |
| `heatmap_peak_count_roi_40x40.png` | ROI 안 검출 피크 개수(다중 피크) + colorbar |
| `heatmap_robust_peak_persistence_frac_40x40.png` 등 | 이전 LP와의 피크/골 일치 비율, 이전 단계 피크 수, **잔차** 피크·골 수, activation×persistence (보조) |
| `heatmap_modal_deviation_rms_40x40.png` | 파장별 최빈 진폭 대비 RMS 편차 + colorbar(**선형**; 최빈/진폭 축은 log 미적용) |
| `heatmap_ica_component*_40x40.png`, `ica_mixing_*_scatter.png`, `ica_ref_*_spectrum_scatter.png`, `ica_* .npy` | 기본 **K-means·L2**·성분 10·엔벨롭 참조 등; `--ica-n-components` 0이면 생략 |
| `pixels/pixel_rXX_cYY.png` | `--pixel-pngs` 일 때만 1600장 |

## `index.html` 여는 방법

`{입력 stem}/index.html`을 더블클릭해 `file://`로 열면 됩니다. 스펙트럼은 HTML에 임베드되어 있습니다. Chart.js는 CDN을 사용합니다.

## 파라미터 요약

- `--roi-lo` / `--roi-hi` (기본 675–775 nm): 특징 점수를 계산할 파장 구간.
- `--windows`: 저역통과 단계별 Savitzky–Golay 창 길이(홀수로 자동 보정).
- `--prominence-frac`: prominence 하한 스케일(기본 0.020, **낮출수록 더 민감**).
- `--spread-metric`: **각 파장(행)에서 1600픽셀만** 대상으로 `var` / `iqr` / `std` / `mad` (전체 값 한 덩어리 variance 아님).
- `--prominence-spread-mode`: `inverse`(기본) — 전체 대비 spread **낮은** 파장은 prominence 하한을 크게 올려 억제, **높은** 파장은 하한을 낮춤. `direct` — spread에 비례한 하한(레거시).
- `--prominence-spread-ref`: inverse 전용. 전 파장 spread의 기준값 — `median` / `p75`(기본) / `p90`. 분위수가 높을수록 «평균적으로 조용한» 파장이 기준 대비 더 크게 벌어져 억제가 강해짐.
- `--prominence-spread-gamma`: inverse 전용, 기본 2.5. `(ref/safe)`에 거듭제곱; **크면** 전체보다 spread가 낮은 구간에서 하한이 훨씬 빨리 커짐.
- `--spread-mix-local`: 0~1 (기본 0.5). **inverse**: `shape_core` 가중치; 클수록 저 spread 억제·고 spread 완화가 강함. **direct**: 파장별 spread와 median 혼합 비율.
- `--spread-smooth-window`: 파장축 이동평균(기본 21, `<3`이면 끔). 파장별 IQR 들쭉날쭉함 완화.
- `--peak-distance`: 인접 피크·골 최소 간격 샘플 수 (기본 12, **작을수록 더 촘촘히** 검출).
- `--peak-wlen`: `find_peaks`의 `wlen`(0=끔, 기본). 0이 아니면 prominence 평가 폭 제한.
- `--robust-persistence-match-samples`: 이전 LP 피크/골과의 ±샘플 허용 폭(기본 5).
- `--robust-median-detrend-window`: 파장축 `median_filter` 창(기본 21). 0 또는 &lt;3이면 잔차 피크·골 보조 지표 생략.
- `--robust-residual-prom-scale`: 잔차에 쓰는 `prom_vec` 배수(기본 0.35).
- `--clear-prominence-factor`: “명확한” 국소 최대·최소 판정 배수 (기본 1.65).
- `--modal-hist-bins`: 파장별 최빈 진폭 히스토그램 빈 수 (기본 64).
- `--pixel-pngs` / `--no-pixel-pngs`: 픽셀별 PNG 1600장. **기본 꺼짐**; 필요 시 `--pixel-pngs`.
- `--no-progress`: `tqdm` 진행 표시 비활성화.
- `--x-index`: 플롯·산점도 x축을 nm 대신 `1..N` 샘플 인덱스로.
- `--heatmap-activation-pct-lo` / `--heatmap-activation-pct-hi`: activation PNG 색 범위 백분위(기본 2 / 98). 히트맵이 한쪽에만 몰리면 `pct_hi`를 90~95로.
- `--heatmap-activation-log` / `--no-heatmap-activation-log`: activation 히트맵 colorbar(기본 **로그**). 선형이 필요하면 `--no-heatmap-activation-log`.
- `--viz-medium-prominence-frac`: `spectra.bin`·HTML 팝업에 넣을 «중간» 피크/골( clear 미만·이 비율×clear 이상 ). 0이면 clear 화살표만.
- `--ica-n-components`: 0이면 분해 생략. 기본 10(`min(요청, n_λ, 1600)`).
- `--ica-method`: `kmeans`(기본), `nmf`(비음수 혼합), `fastica`.
- `--spectral-kmeans-normalize`: `kmeans`만(다른 방법은 무시). `l2`(기본, 형태 위주) 또는 `none`(진폭 유지).
- `--ref-envelope-pct-lo` / `--ref-envelope-pct-hi`: 엔벨롭 참조 곡선 분위(기본 10 / 90, `0 ≤ lo < hi ≤ 100`).
- `--ref-envelope-smooth-window`: 엔벨롭용 λ축 중앙값 스무딩 창(기본 21, `<3`이면 중앙값만).
- `--ica-random-state`: 난수 시드(기본 0).
- `--ica-no-standardize`: **FastICA**만; 파장축 `StandardScaler` 생략.
- `--ica-max-iter`: 분해 반복 상한 NMF·K-means·FastICA 공통(기본 1000).
- `--ica-linear-score-heatmap`: FastICA는 대칭 선형(asinh 끔). NMF는 log colorbar 끄고 선형.
- `--nmf-alpha`: NMF 정규화 강도(기본 0.05; 0이면 l1_ratio만으로는 페널티 없음).
- `--nmf-l1-ratio`: NMF L1 비중 0~1(기본 0.65; 클수록 더 희소 경향).
- `--nmf-envelope-residual` / `--no-nmf-envelope-residual`: **NMF만**. 기본 **켬** — 입력을 `max(raw − 엔벨롭 참조, 0)`로 두어 공통 스펙트럼 형태 대비 특이한 부분 위주로 분해. 전체 진폭 형태로 해석하려면 `--no-nmf-envelope-residual`.

자세한 구현은 `docs/internal/IMPLEMENTATION_au_region.md` 를 참고하세요.
