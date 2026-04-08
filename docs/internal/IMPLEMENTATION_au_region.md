# Au region 파이프라인 구현 메모

## 데이터 형식

- 입력: 탭 구분 텍스트, **행 = 파장 샘플**, **열0 = 파장(nm)**, **열 1..1600 = 40×40 픽셀 진폭**.
- `numpy.loadtxt(..., delimiter="\t")` → shape `(n_wl, 1601)`.

## 픽셀 인덱스

- 열 `j` (0..1599) ↔ 테이블 위치 `row = j // 40`, `col = j % 40` (0 기반).
- `index.html` / `plot_pixel_figure` 모두 동일 규칙.

## 출력 경로·ROI 기본값

- `--out` 생략 시 현재 작업 디렉터리 아래 `입력파일.stem` 이름의 폴더(예: `…Au_region.txt.txt` → `…Au_region.txt/`).
- ROI 기본: 675–775 nm.
- **clear_maxima / clear_minima**: `prominence ≥ clear_prominence_factor × prom_vec[i]` 인 ROI 내 피크·골만 플롯/HTML에 강조(다중 피크·골 동시 표시). 기본 `clear_prominence_factor`는 과강조를 줄이기 위해 보수적으로 둠.
- **n_peaks_roi / n_valleys_roi**: ROI에서 `find_peaks`로 잡힌 전체 개수.
- 주도 피크: ROI 내 prominence 최대 피크 → 파장·`peak_widths` 폭 히트맵; 피크 개수 히트맵으로 다중 피크 분포 요약.

## 파장별 최빈 진폭 (modal) 곡선

- 각 파장 인덱스 `i`에서 `y[i, :]` (1600 픽셀)에 대해 히스토그램(`--modal-hist-bins`, 기본 64)을 만들고 **최빈 빈의 중심**을 해당 파장의 대표 진폭으로 둠.
- 픽셀 `j`의 점수: 전체 파장에 대해 `sqrt(mean((y[:,j] - modal)^2))` (RMS). 피팅 없음.
- **색 스케일**: 최빈 진폭·modal RMS 맵·팝업 속 최빈 곡선은 **로그 colorbar를 쓰지 않음**. 둘 다 원 데이터와 같은 **진폭 차원**이고 0 근처·해석 직관을 위해 선형이 맞음. 로그는 **activation**(prominence로 쌓인 양수 점수, 역수 분포 완화용)만 적용.

## 저역통과 (다단계)

- 각 픽셀에 대해 `scipy.signal.savgol_filter`를 `--windows` 리스트만큼 반복 적용.
- 창 길이는 홀수이며 `n_wl`을 넘지 않도록 잘라냄.
- **특징 검출**은 가장 강한 스무딩(마지막 단계) 곡선에 대해 수행.

## 국소 최대/최소 및 점수

- `scipy.signal.find_peaks` 로 피크, `-y` 로 골.
- **파장별 prominence 하한**: `y` shape `(n_λ, n_pix)`일 때 **항상 axis=1**(픽셀만)으로 파장 `i`마다 spread를 구함 — `var`·IQR·std·MAD 모두 «그 주파수에서 픽셀 간 퍼짐». **전체 행렬을 펼친 단일 variance는 사용하지 않음.**
- `spread_to_prominence_curve`의 `global_scale = median(spread_s)`는 «파장별 spread 스칼라 n_λ개»의 중앙값이지, 전 큐브 진폭의 분산이 아님.
- **prominence vs spread (`--prominence-spread-mode`)**
  - **inverse (기본)**: «전체 파장 spread 분포» 대비 해당 파장 spread가 **많이 낮으면** prominence 하한을 **비선형으로** 크게 올려 거의 검출되지 않게 함. 기준 `ref`는 `--prominence-spread-ref`(median / p75 / p90)로 잡고, `safe = max(spread_s, ref·1e-5)`, `shape_core = min((ref/safe)^gamma, 8000)`, `shape = mix_local·shape_core + (1-mix_local)`, `prom = prominence_frac·ref·shape`. `gamma`=`--prominence-spread-gamma`(기본 2.5). `gamma=1`이면 (ref/safe)에 대한 선형 비율과 동일 차수.
  - **direct (레거시)**: `spread_used = mix_local * spread_s + (1-mix_local) * median`, `prom = prominence_frac * spread_used` — spread 큰 파장이 더 까다로움.
- 이후: 미소 하한 → 선택적 파장축 이동평균 → 위 모드에 따른 블렌딩 → `find_peaks(..., prominence=벡터, distance=peak_distance)`.
- **튜닝**: 저분산 파장이 여전히 시끄러우면 `--prominence-spread-gamma`↑, `--prominence-spread-ref p90`, 또는 `--spread-mix-local`↑. 고분산 구간까지 너무 둔해지면 `gamma`↓·`ref median`·`mix`↓.
- ROI 안 피크/골로 activation score.
- **clear max/min**: `prominence >= clear_prominence_factor * prom_vec[i]`.
- 산출: `prominence_spread_effective.npy` (= `prom_vec / prominence_frac`, 파장별 «effective scale»; inverse에서는 `ref * shape`).
- **선택 `find_peaks` `wlen`**: `--peak-wlen`>0 이면 prominence 인접 구간 평가 폭을 제한(기본 0 = 미사용, 기존과 동일).

## 견고성 보조 (기본 ROI 피크/골·주도 파장·activation 정의는 유지)

- **다단계 persistence**: `len(windows)≥2`일 때 마지막 스무딩과 **바로 이전** 단계 각각에 동일 `prom_vec`·`distance`·`wlen`로 `find_peaks`. ROI 안 «주 피크» 인덱스가 이전 단계 피크와 `±robust_persistence_match_samples` 샘플 이내면 1건으로 매칭. 픽셀별 `robust_peak_persistence_frac` = 매칭 수 / max(ROI 피크 수, 1), 골도 동일.
- **잔차 피크·골**: `--robust-median-detrend-window`≥3(홀수 보정)이면 `y_smooth − median_filter(y_smooth, size=window)`에 대해 `prominence = max(prom_vec * robust_residual_prom_scale, 바닥)`로 보조 `find_peaks`. 느린 배경 기울기와 좁은 돌출을 분리해 참고용 개수 `n_peaks_roi_residual` 등을 산출. 0 또는 <3이면 잔차 브랜치 생략.
- **파생 히트맵·npy**: `heatmap_robust_*`, `heatmap_peak_count_roi_prev_smooth_*`, `heatmap_valley_count_roi_prev_smooth_*`, `heatmap_*_residual_*`, `heatmap_activation_x_peak_persistence_*` 및 대응 `.npy`. `activation_x_peak_persistence` = `score * (0.2 + 0.8 * robust_peak_persistence_frac)` (persistence 없으면 nan).
- **meta / HTML**: `pixels[]`에 위 스칼라 필드 병합; `index.html` 하단에 견고성 격자 섹션.

## 바이너리 `spectra.bin`

헤더 16바이트, little-endian uint32 네 개:

- `magic = 0x41555247` (`'AURG'`)
- `n_wavelength`, `n_pixel` (=1600), `n_chan` (= **1 + len(windows) + 4**)

이후 `float32`, C 순서, 논리 shape `(n_pixel, n_wavelength, n_chan)`:

- 채널 0: raw
- 채널 1..k: 스무딩 단계(약한 LP → 강한 LP 순, k=len(windows))
- 마지막 4채널(HTML 팝업 화살표): `viz_clear_peak_y`, `viz_clear_valley_y`, `viz_medium_peak_y`, `viz_medium_valley_y` — 대부분 **NaN**, 마킹된 λ만 **raw 진폭**(화살표 끝). clear=기존 clear_max/min; medium=`--viz-medium-prominence-frac`×clear ≤ prom < clear 인 ROI 피크/골.

픽셀 `p`의 시작 오프셋: `16 + p * n_wavelength * n_chan * 4`.

## HTML 뷰어 (`index.html`)

- `index_template.html` → `index.html`: `meta` JSON + `spectra.bin` 전체를 **gzip 후 base64**로 인라인.
- 브라우저 `DecompressionStream('gzip')`으로 복원 후 `ArrayBuffer`를 그대로 파싱 (`file://`에서도 동작).
- Chart.js 4(CDN): scatter(raw + LP) + line 화살표. **번들 채널 수**가 `1 + len(windows) + meta.spectra_viz_channels`와 일치하면 화살표는 **spectra.bin** 마지막 4채널에서만 읽음(clear 빨강/녹 + 중간 주황/청록). 예전 번들(채널 수 짧음)은 `pixels[].clear_maxima/minima`(meta) 폴백.
- 픽셀 메타에 `valley_primary_nm`, `valley_width_nm` 추가(툴팁·팝업 제목).
- **activation 격자 색**: `meta.heatmap_activation_log`, `heatmap_activation_pct_lo`, `heatmap_activation_pct_hi`를 읽어 PNG `heatmap_activation_40x40.png`와 같은 백분위·로그 규칙으로 `activations`를 스칼라→색으로 매핑(팔레트는 matplotlib `magma`와 다를 수 있으나 스케일은 동일). 예전 `meta`에 키가 없으면 로그 켜고 pct 2/98로 간주.
- **세로 순서(요약)**: (1) activation (로그) (2) 스펙트럼 분해 블록(`meta.ica` 있을 때) (3) modal RMS (4) 주도 피크/골 **파장** (5) 소제목 «개수·폭·견고성» 아래 피크 개수·폭·골 폭·견고성·activation×persistence.
- **추가 히트맵 격자**(PNG와 동일 데이터): `peak_wavelength_nm`, `peak_width_nm`, `valley_wavelength_nm`, `valley_width_nm`, `peak_count_roi` — 길이 `grid*grid`, `nan`은 JSON `null`. 브라우저에서 2–98 백분위 선형(또는 파장은 HSL)으로 색만 근사; 모든 격자에서 호버 시 동일 스펙트럼 팝업. 팝업 제목에 해당 픽셀의 맵 요약 한 줄 포함.
- **견고성 격자**: `robust_peak_persistence_frac`, `robust_valley_persistence_frac`, `peak_count_roi_prev_smooth`, `peak_count_roi_residual`, `valley_count_roi_residual`, `activation_x_peak_persistence` — LP 한 단계뿐이거나 잔차 끄면 일부가 전부 `null`/어두운 격자.
- **분해 블록**(옵션): `meta.ica`가 있으면 activation 바로 아래에 mixing 차트 + 성분별 픽셀 점수 격자 + 체크박스 토글(«스펙트럼 분해» 절).

## 주도 골(dominant valley) 맵

- ROI 안에서 `find_peaks(-y_smooth)`로 잡힌 골 중 **prominence가 최대인 것**을 주도 골로 선택.
- **파장 맵**: 그 인덱스의 `wl` (nm).
- **폭 맵**: `scipy.signal.peak_widths(-y_smooth, [i0], rel_height=0.5)`의 폭(샘플) × `median(diff(wl))` → 대략 nm.

## 히트맵 색 스케일 (activation)

- `save_heatmap_colorbar`: 기본 **전 픽셀 유효값의 백분위**로 `vmin`/`vmax` 설정(기본 2–98). **소수 픽셀만 점수가 매우 크면** 상한이 그쪽으로 잡혀, 나머지 대부분이 colormap 하단에 몰려 **검게** 보일 수 있음(수치 `activations.npy`는 그대로).
- activation 전용: `--heatmap-activation-pct-hi`를 90–95로 낮추면 중간 대비↑. 로그 스케일은 **기본**(끄려면 `--no-heatmap-activation-log`).
- `imshow(..., interpolation='nearest')` — 픽셀 경계를 흐리게 만드는 건 아님. 부드럽게 보이면 이웃 픽셀 점수가 비슷한 것.

## 성능

- `matplotlib` 백엔드는 `Agg` 고정.
- 픽셀 PNG 1600장은 수 분 단위일 수 있어 **기본 비생성**; `--pixel-pngs`일 때만 `pixels/*.png` 작성.

## 스펙트럼 분해 FastICA / NMF / K-means (기본: `--ica-n-components` 10, `--ica-method` kmeans; 0이면 생략)

- **`--ica-method`**
  - **kmeans (기본)**: `sklearn.cluster.KMeans`. 픽셀당 **정확히 한 성분에만 1**(exclusive one-hot), 나머지 0. 성분 곡선=**클러스터 중심**. 선형 혼합 `X≈WH`가 아니라 Voronoi 할당. `--spectral-kmeans-normalize` 기본 **`l2`**: 행 **L2 정규화 후** 군집(진폭보다 **형태** 위주); `none`이면 진폭 유지. 성분(클러스터)마다 픽셀이 **통상 ≥1개**(빈 클러스터는 드묻). «각 λ마다 한 성분만 켜지게» 같은 **파장축 one-hot**은 별도 모델(예: 엄격한 스파스 코딩·맞춤 제약)이 필요하고 본 파이프라인에는 없음.
  - **nmf**: `sklearn.decomposition.NMF` — `X≈W·H`, **W,H ≥ 0**. 기본 입력은 **엔벨롭 참조 `env(λ)` 대비 양의 잔차** `X = max(y.T − env, 0)` 후 `max(·, 1e-12)`(공통 형태를 빼고 «특이» 돌출·피크 쪽에 NMF가 기울기 쉬움). `--no-nmf-envelope-residual`이면 예전처럼 **raw 양수 클립만**. 표준화 없음. `--nmf-alpha`, `--nmf-l1-ratio`로 L1/L2 페널티. `meta.ica.nmf_envelope_residual`.
  - **fastica**: `sklearn.decomposition.FastICA` (`whiten="unit-variance"`). **음수** mixing·점수는 정상(부호·스케일 모호성: 성분과 mixing 열을 동시에 뒤집으면 동일 모델). 해석은 크기·상대 형태 위주.
- **FastICA 전처리**: 기본 파장축 `StandardScaler` (`--ica-no-standardize`로 끔). NMF·K-means는 위 규칙대로 raw/정규화만.
- **출력 해석**
  - **FastICA**: `mixing_` **열** ≈ λ 패턴; `transform` 점수 = 픽셀별 축 강도. **고유값 없음**; 순위는 점수 **분산** 내림차순.
  - **NMF**: `components_`의 행 = λ 패턴 H(≥0), `transform` = W(≥0).
  - **K-means**: `meta.ica.cluster_sizes_ordered`, `ica_kmeans_labels_ordered.npy`, 점수 맵은 0/1.
- **잘림**: `K = min(n_components, n_pixel, n_wl)`; `K < 1`이면 생략.
- **mean / median / 엔벨롭 참조**: 모두 **원본 `y`**에서 만든 **비교용 곡선**(성분 아님).
  - **엔벨롭**: 파장마다 픽셀 축 **중앙값** `med(λ)`을 λ축 이동평균으로 스무딩한 `baseline(λ)`과 비교. `med > baseline`이면 «피크형»으로 보고 그 λ에서 픽셀 축 **하위 분위**(`--ref-envelope-pct-lo`, 기본 10%); `med < baseline`이면 «골형»으로 **상위 분위**(`--ref-envelope-pct-hi`, 기본 90%); 같으면 중앙값. 창 `<3`이면 엔벨롭=중앙값만. 산출: `ica_ref_envelope_spectrum_scatter.png`, `ica_ref_envelope_spectrum_y.npy`, `ica_ref_envelope_peak_like.npy`, `meta.ica.envelope_ref_*`.
- **픽셀 점수 히트맵·HTML**
  - FastICA: 기본 **asinh** 대칭 스케일; `--ica-linear-score-heatmap`이면 대칭 선형(coolwarm).
  - NMF: 기본 **log colorbar + magma**; `--ica-linear-score-heatmap`이면 선형 magma.
  - K-means: **선형 magma**(0/1).
- **mixing 대표 형상 PNG**: 성분마다 `ica_mixing_component{00..}_scatter.png`.
- **파일**: `heatmap_ica_component{00..}_40x40.png`, scatter·ref PNG, `ica_* .npy`.
- **meta / HTML**: `decomposition_backend`, `score_sign`, `ica_score_display`, NMF 시 `nmf_*`·`nmf_envelope_residual`, K-means 시 `kmeans_normalize`·`cluster_sizes_ordered`·`pixel_code`. 뷰어 **체크박스** 토글 동일.

## 소스

- `src/au_region_analysis.py`
- `src/index_template.html`
