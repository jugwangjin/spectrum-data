"""
40x40 (1600) spectral cube: tab-separated rows = wavelength samples, col0 = λ (nm), cols 1..1600 = amplitude.

Outputs under --out (기본: 입력 txt의 stem 디렉터리): index.html (스펙트럼 gzip+base64 임베드, 로컬 서버 불필요),
heatmaps + colorbars, meta.json, spectra.bin, optional per-pixel PNGs.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks, peak_widths, savgol_filter
from tqdm import tqdm

GRID = 40
N_PIXEL = 1600
MAGIC = 0x41555247  # 'AURG'


def load_cube(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter="\t", dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 1 + N_PIXEL:
        raise ValueError(
            f"expected shape (n_wl, {1 + N_PIXEL}), got {data.shape} for {path}"
        )
    wl = data[:, 0].astype(np.float64)
    y = data[:, 1:].astype(np.float64)
    return wl, y


def lowpass_levels(
    y: np.ndarray, windows: list[int], poly: int = 3
) -> list[np.ndarray]:
    n = len(y)
    out: list[np.ndarray] = []
    for w in windows:
        w = int(w)
        if w % 2 == 0:
            w += 1
        w = max(w, poly + 2)
        if w >= n:
            w = n - 1 if (n - 1) % 2 == 1 else n - 2
            w = max(w, poly + 2)
        out.append(savgol_filter(y, w, poly))
    return out


def roi_mask(wl: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (wl >= lo) & (wl <= hi)


def modal_amplitude_curve(y: np.ndarray, n_bins: int) -> np.ndarray:
    """
    각 파장(행)마다 1600픽셀 진폭의 히스토그램에서 최빈 구간 중심을 대표 진폭으로 둔 곡선.
    y: (n_wavelength, n_pixel)
    """
    n_wl = y.shape[0]
    out = np.empty(n_wl, dtype=np.float64)
    nb = max(8, int(n_bins))
    for i in range(n_wl):
        row = y[i, :]
        lo, hi = float(np.min(row)), float(np.max(row))
        if hi <= lo or not np.isfinite(lo):
            out[i] = float(np.median(row))
            continue
        counts, edges = np.histogram(row, bins=nb, range=(lo, hi))
        k = int(np.argmax(counts))
        if counts[k] == 0:
            out[i] = float(np.median(row))
        else:
            out[i] = 0.5 * (edges[k] + edges[k + 1])
    return out


def modal_deviation_rms(y: np.ndarray, modal: np.ndarray) -> np.ndarray:
    """각 픽셀 스펙트럼과 modal 곡선의 RMS 차이 (전 파장)."""
    d = y - modal[:, np.newaxis]
    return np.sqrt(np.mean(d * d, axis=0))


def per_wavelength_cross_spread(y: np.ndarray, metric: str) -> np.ndarray:
    """
    각 파장(주파수) 행마다, 그 파장에서 1600 픽셀 진폭만 모아 통계 — axis=1 (픽셀 방향).
    전 큐브를 펼친 단일 variance / 전체 분포가 아님.
    y: (n_wavelength, n_pixel)
    """
    y = np.asarray(y, dtype=np.float64)
    if metric == "var":
        # 분산(진폭²); prominence와 단위 맞추려면 std가 동일 스케일. var 쓸 때 prominence_frac 를 더 작게 잡는 것이 일반적.
        return np.var(y, axis=1, ddof=0).astype(np.float64)
    if metric == "iqr":
        q75 = np.percentile(y, 75.0, axis=1)
        q25 = np.percentile(y, 25.0, axis=1)
        return (q75 - q25).astype(np.float64)
    if metric == "std":
        return np.std(y, axis=1, ddof=0).astype(np.float64)
    if metric == "mad":
        med = np.median(y, axis=1, keepdims=True)
        return (np.median(np.abs(y - med), axis=1) * 1.4826).astype(np.float64)
    raise ValueError(f"spread metric: {metric!r} (use var, iqr, std, mad)")


def _cross_spread_reference_level(spread_s: np.ndarray, kind: str) -> float:
    """파장축으로 구한 spread 시퀀스에 대한 전역 기준(중앙값·분위수)."""
    s = np.asarray(spread_s, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 1.0
    if kind == "median":
        return float(np.median(s))
    if kind == "p75":
        return float(np.percentile(s, 75.0))
    if kind == "p90":
        return float(np.percentile(s, 90.0))
    raise ValueError(f"prominence spread ref: {kind!r} (use median, p75, p90)")


def spread_to_prominence_curve(
    spread: np.ndarray,
    prominence_frac: float,
    mix_local: float,
    smooth_window: int,
    mode: str = "inverse",
    inverse_gamma: float = 2.5,
    inverse_ref: str = "median",
) -> tuple[np.ndarray, dict]:
    """
    파장별 cross-spread로 prominence 하한 곡선. 선택적 파장축 스무딩 후:

    - inverse (기본): 전역 기준 대비 spread가 작은 파장은 하한을 **강하게** 올림.
      ref = median 또는 p75/p90(전체 파장 spread 분포), safe = max(spread_s, ref*rho_floor),
      shape_core = min((ref/safe)^gamma, cap), shape = mix*shape_core + (1-mix),
      prom = prominence_frac * ref * shape. gamma>1이면 «전체보다 많이 낮은» 구간이 선형보다 훨씬 덜 민감.
    - direct (레거시): spread에 비례해 하한 — spread 큰 파장이 더 까다로움.
    mix_local: inverse에서 shape_core 비중(클수록 저분산 억제·고분산 완화 효과 강함).
    global_cross_spread_median: 진단용 median(spread_s) (inverse의 ref와 다를 수 있음).
    smooth_window: 홀수 길이 이동평균(파장 방향), 0이면 끔.
    """
    spread = np.asarray(spread, dtype=np.float64)
    pos = spread[spread > 0]
    floor = float(np.median(pos) * 1e-4) if pos.size else 1.0
    spread_eff = np.fmax(spread, floor)

    w = int(smooth_window)
    if w >= 3:
        if w % 2 == 0:
            w += 1
        spread_s = uniform_filter1d(spread_eff, size=w, mode="nearest")
    else:
        spread_s = spread_eff

    global_scale = float(np.median(spread_s))
    mix = float(np.clip(mix_local, 0.0, 1.0))

    prom_inv_gamma: float | None = None
    prom_inv_ref: str | None = None
    prom_inv_ref_val: float | None = None

    if mode == "direct":
        spread_used = mix * spread_s + (1.0 - mix) * global_scale
        prom_vec = prominence_frac * spread_used
    elif mode == "inverse":
        ref = _cross_spread_reference_level(spread_s, inverse_ref)
        if not np.isfinite(ref) or ref <= 0.0:
            ref = max(global_scale, 1e-12)
        prom_inv_gamma = float(inverse_gamma)
        prom_inv_ref = inverse_ref
        prom_inv_ref_val = float(ref)
        rho_floor = 1e-5
        safe = np.fmax(spread_s, ref * rho_floor)
        ratio = ref / safe
        g = float(max(inverse_gamma, 1e-6))
        shape_core = np.power(ratio, g)
        shape_core = np.fmin(shape_core, 8000.0)
        shape = mix * shape_core + (1.0 - mix) * 1.0
        prom_vec = prominence_frac * ref * shape
    else:
        raise ValueError(f"spread vs prominence mode: {mode!r} (use inverse, direct)")

    prom_vec = prom_vec.astype(np.float64)
    info = {
        "global_cross_spread_median": global_scale,
        "mix_local": mix,
        "smooth_window": w if w >= 3 else 0,
        "prominence_spread_mode": mode,
        "prominence_inverse_gamma": prom_inv_gamma,
        "prominence_inverse_ref": prom_inv_ref,
        "prominence_inverse_ref_value": prom_inv_ref_val,
    }
    return prom_vec, info


def features_for_pixel(
    wl: np.ndarray,
    y_raw: np.ndarray,
    y_smooth: np.ndarray,
    roi_lo: float,
    roi_hi: float,
    prom_vec: np.ndarray,
    clear_prominence_factor: float,
    distance: int,
    peak_wlen: int = 0,
) -> dict:
    """Peaks/valleys on y_smooth. prom_vec: 파장별 prominence 하한 (전 큐브 분포 기반)."""
    dist = max(1, int(distance))
    if prom_vec.shape[0] != y_smooth.shape[0]:
        raise ValueError("prom_vec length must match y_smooth")

    fp_kw: dict = {}
    wlen = int(peak_wlen)
    if wlen > 0:
        fp_kw["wlen"] = wlen
    pk_i, pk_prop = find_peaks(
        y_smooth, prominence=prom_vec, distance=dist, **fp_kw
    )
    vl_i, vl_prop = find_peaks(
        -y_smooth, prominence=prom_vec, distance=dist, **fp_kw
    )

    roi = roi_mask(wl, roi_lo, roi_hi)
    pk_prom = pk_prop["prominences"]
    vl_prom = vl_prop["prominences"]

    def in_roi(indices: np.ndarray) -> np.ndarray:
        return np.array([bool(roi[i]) for i in indices], dtype=bool)

    pk_roi = in_roi(pk_i)
    vl_roi = in_roi(vl_i)

    score = 0.0
    if np.any(pk_roi):
        score += float(np.sum(pk_prom[pk_roi] ** 2) ** 0.5)
    if np.any(vl_roi):
        score += float(np.sum(vl_prom[vl_roi] ** 2) ** 0.5)

    peaks_detail = [
        {
            "index": int(i),
            "wavelength": float(wl[i]),
            "amplitude_smooth": float(y_smooth[i]),
            "prominence": float(p),
            "kind": "max",
        }
        for i, p in zip(pk_i, pk_prom)
        if roi[i]
    ]
    valleys_detail = [
        {
            "index": int(i),
            "wavelength": float(wl[i]),
            "amplitude_smooth": float(y_smooth[i]),
            "prominence": float(p),
            "kind": "min",
        }
        for i, p in zip(vl_i, vl_prom)
        if roi[i]
    ]

    clear_vec = clear_prominence_factor * prom_vec
    clear_minima = [
        v for v in valleys_detail if v["prominence"] >= clear_vec[v["index"]]
    ]
    clear_maxima = [
        v for v in peaks_detail if v["prominence"] >= clear_vec[v["index"]]
    ]

    valley_primary_nm: float | None = None
    valley_width_nm: float | None = None
    if np.any(vl_roi):
        roi_j = np.flatnonzero(vl_roi)
        k = int(np.argmax(vl_prom[roi_j]))
        j = int(roi_j[k])
        i0 = int(vl_i[j])
        valley_primary_nm = float(wl[i0])
        neg = -y_smooth.astype(np.float64)
        w_idx, _, _, _ = peak_widths(neg, np.array([i0], dtype=np.intp), rel_height=0.5)
        dw = float(np.median(np.diff(wl)))
        if dw > 0 and np.isfinite(w_idx[0]):
            valley_width_nm = float(w_idx[0]) * dw

    peak_primary_nm: float | None = None
    peak_width_nm: float | None = None
    if np.any(pk_roi):
        roi_j = np.flatnonzero(pk_roi)
        k = int(np.argmax(pk_prom[roi_j]))
        j = int(roi_j[k])
        ip = int(pk_i[j])
        peak_primary_nm = float(wl[ip])
        ys = y_smooth.astype(np.float64)
        w_idx, _, _, _ = peak_widths(ys, np.array([ip], dtype=np.intp), rel_height=0.5)
        dw = float(np.median(np.diff(wl)))
        if dw > 0 and np.isfinite(w_idx[0]):
            peak_width_nm = float(w_idx[0]) * dw

    return {
        "score": score,
        "peaks": peaks_detail,
        "valleys": valleys_detail,
        "clear_minima": clear_minima,
        "clear_maxima": clear_maxima,
        "n_peaks_roi": len(peaks_detail),
        "n_valleys_roi": len(valleys_detail),
        "prominence_threshold": float(np.median(prom_vec)),
        "valley_primary_nm": valley_primary_nm,
        "valley_width_nm": valley_width_nm,
        "peak_primary_nm": peak_primary_nm,
        "peak_width_nm": peak_width_nm,
    }


def _count_persistence_matches(
    main_roi_idx: np.ndarray, prev_peaks: np.ndarray, tol: int
) -> int:
    if main_roi_idx.size == 0:
        return 0
    if prev_peaks.size == 0:
        return 0
    prev = np.sort(prev_peaks.astype(np.intp))
    n_ok = 0
    for i in main_roi_idx.astype(np.intp):
        lo = int(np.searchsorted(prev, i - tol, side="left"))
        hi = int(np.searchsorted(prev, i + tol, side="right"))
        if hi > lo:
            n_ok += 1
    return int(n_ok)


def robust_extension_features(
    wl: np.ndarray,
    y_main: np.ndarray,
    y_prev: np.ndarray | None,
    prom_vec: np.ndarray,
    roi_lo: float,
    roi_hi: float,
    distance: int,
    persistence_tol: int,
    median_detrend_w: int,
    residual_prom_scale: float,
    peak_wlen: int = 0,
) -> dict[str, float | int | None]:
    """
    기본 피크/골 정의는 바꾸지 않고, 견고성 참고 지표만 추가.
    - 이전 LP 단계와의 인덱스 근접 일치(persistence)
    - 파장축 median_filter 베이스라인 제거 잔차 위 find_peaks (좁은 스파이크·슬로우 드리프트 분리)
    """
    dist = max(1, int(distance))
    roi = roi_mask(wl, roi_lo, roi_hi)
    fp_kw: dict = {}
    if int(peak_wlen) > 0:
        fp_kw["wlen"] = int(peak_wlen)

    pk_m, _ = find_peaks(y_main, prominence=prom_vec, distance=dist, **fp_kw)
    vl_m, _ = find_peaks(-y_main, prominence=prom_vec, distance=dist, **fp_kw)
    pk_roi_idx = pk_m[roi[pk_m]]
    vl_roi_idx = vl_m[roi[vl_m]]
    n_pk_roi = int(pk_roi_idx.size)
    n_vl_roi = int(vl_roi_idx.size)

    out: dict[str, float | int | None] = {}

    tol = max(0, int(persistence_tol))
    if y_prev is not None and y_prev.shape == y_main.shape:
        pk_p, _ = find_peaks(y_prev, prominence=prom_vec, distance=dist, **fp_kw)
        vl_p, _ = find_peaks(-y_prev, prominence=prom_vec, distance=dist, **fp_kw)
        out["n_peaks_roi_prev_smooth"] = int(np.sum(roi[pk_p]))
        out["n_valleys_roi_prev_smooth"] = int(np.sum(roi[vl_p]))
        out["robust_peak_persistence_count"] = _count_persistence_matches(
            pk_roi_idx, pk_p, tol
        )
        out["robust_valley_persistence_count"] = _count_persistence_matches(
            vl_roi_idx, vl_p, tol
        )
        out["robust_peak_persistence_frac"] = float(
            out["robust_peak_persistence_count"] / max(n_pk_roi, 1)
        )
        out["robust_valley_persistence_frac"] = float(
            out["robust_valley_persistence_count"] / max(n_vl_roi, 1)
        )
    else:
        out["n_peaks_roi_prev_smooth"] = None
        out["n_valleys_roi_prev_smooth"] = None
        out["robust_peak_persistence_count"] = None
        out["robust_valley_persistence_count"] = None
        out["robust_peak_persistence_frac"] = None
        out["robust_valley_persistence_frac"] = None

    wdt = int(median_detrend_w)
    if wdt >= 3:
        if wdt % 2 == 0:
            wdt += 1
        ys = y_main.astype(np.float64)
        baseline = median_filter(ys, size=wdt, mode="nearest")
        res = ys - baseline
        roi_ix = np.flatnonzero(roi)
        med_prom = (
            float(np.median(prom_vec[roi]))
            if roi_ix.size > 0
            else float(np.median(prom_vec))
        )
        floor = max(med_prom * 0.05, 1e-12)
        prom_r = np.fmax(prom_vec * float(residual_prom_scale), floor)
        pk_r, _ = find_peaks(res, prominence=prom_r, distance=dist, **fp_kw)
        vl_r, _ = find_peaks(-res, prominence=prom_r, distance=dist, **fp_kw)
        out["n_peaks_roi_residual"] = int(np.sum(roi[pk_r]))
        out["n_valleys_roi_residual"] = int(np.sum(roi[vl_r]))
    else:
        out["n_peaks_roi_residual"] = None
        out["n_valleys_roi_residual"] = None

    return out


SPECTRA_VIZ_CHANNELS = 4


def build_spectra_viz_channels(
    n_wl: int,
    y_col: np.ndarray,
    feat: dict,
    clear_vec: np.ndarray,
    medium_prominence_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    HTML 번들용 sparse 채널: 마킹된 λ만 raw 진폭, 나머지 NaN.
    clear_* = clear_maxima/minima; medium_* = prominence in [frac·clear, clear).
    """
    yf = y_col.astype(np.float64)
    cp = np.full(n_wl, np.nan, dtype=np.float32)
    cv = np.full(n_wl, np.nan, dtype=np.float32)
    mp = np.full(n_wl, np.nan, dtype=np.float32)
    mv = np.full(n_wl, np.nan, dtype=np.float32)
    clear_pi = {int(v["index"]) for v in feat.get("clear_maxima", [])}
    clear_vi = {int(v["index"]) for v in feat.get("clear_minima", [])}
    for v in feat.get("clear_maxima", []):
        i = int(v["index"])
        cp[i] = float(yf[i])
    for v in feat.get("clear_minima", []):
        i = int(v["index"])
        cv[i] = float(yf[i])
    mfrac = float(medium_prominence_frac)
    if mfrac > 0.0:
        for v in feat.get("peaks", []):
            i = int(v["index"])
            cvi = float(clear_vec[i])
            thr = cvi * mfrac
            p = float(v["prominence"])
            if p >= thr and p < cvi and i not in clear_pi:
                mp[i] = float(yf[i])
        for v in feat.get("valleys", []):
            i = int(v["index"])
            cvi = float(clear_vec[i])
            thr = cvi * mfrac
            p = float(v["prominence"])
            if p >= thr and p < cvi and i not in clear_vi:
                mv[i] = float(yf[i])
    return cp, cv, mp, mv


def write_spectra_bin(
    path: Path,
    raw: np.ndarray,
    smooth_stack: np.ndarray,
    per_pixel: list[dict],
    prom_vec: np.ndarray,
    clear_prominence_factor: float,
    viz_medium_prominence_frac: float,
) -> None:
    """
    raw, smooth_stack: (n_wl, n_pixel). smooth_stack shape (n_levels, n_wl, n_pixel).
    채널: raw, LP1..LPk, viz_clear_peak_y, viz_clear_valley_y, viz_medium_peak_y, viz_medium_valley_y
    (viz는 NaN sparse, 값=해당 λ의 raw 진폭 → 팝업 화살표).
    """
    n_wl, n_pix = raw.shape
    levels = smooth_stack.shape[0]
    n_chan = 1 + levels + SPECTRA_VIZ_CHANNELS
    clear_vec = clear_prominence_factor * prom_vec
    base = np.stack([raw.T] + [smooth_stack[i].T for i in range(levels)], axis=-1).astype(
        np.float32
    )
    viz_layers = np.empty((n_pix, n_wl, SPECTRA_VIZ_CHANNELS), dtype=np.float32)
    for j in range(n_pix):
        cp, cv, mp, mv = build_spectra_viz_channels(
            n_wl,
            raw[:, j],
            per_pixel[j],
            clear_vec,
            viz_medium_prominence_frac,
        )
        viz_layers[j, :, 0] = cp
        viz_layers[j, :, 1] = cv
        viz_layers[j, :, 2] = mp
        viz_layers[j, :, 3] = mv
    cube = np.concatenate([base, viz_layers], axis=-1)
    blob = cube.tobytes(order="C")
    header = struct.pack("<IIII", MAGIC, n_wl, n_pix, n_chan)
    path.write_bytes(header + blob)


def write_meta(path: Path, meta: dict) -> None:
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _pixel_map_for_json(a: np.ndarray) -> list[float | None]:
    """길이 N_PIXEL 1D 배열을 meta.json용으로 직렬화(nan → null, JSON 표준 호환)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    out: list[float | None] = []
    for x in a:
        xf = float(x)
        out.append(None if np.isnan(xf) else xf)
    return out


def plot_pixel_figure(
    wl: np.ndarray,
    y_raw: np.ndarray,
    smooths: list[np.ndarray],
    feat: dict,
    out_path: Path,
    x_use_index: bool,
    modal_curve: np.ndarray | None = None,
) -> None:
    x = np.arange(1, len(y_raw) + 1, dtype=np.float64) if x_use_index else wl
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=120)
    if modal_curve is not None and modal_curve.shape[0] == y_raw.shape[0]:
        ax.plot(
            x,
            modal_curve,
            c="#a855e8",
            lw=1.8,
            alpha=0.9,
            zorder=1,
            label="λ-wise mode (line)",
        )
        ax.scatter(
            x,
            modal_curve,
            s=12,
            c="#e9b4ff",
            edgecolors="#6b21a8",
            linewidths=0.6,
            alpha=0.95,
            zorder=2,
            label="λ-wise mode (scatter)",
        )
    ax.scatter(
        x, y_raw, s=4, c="#0066ff", alpha=0.95, label="raw", edgecolors="none", zorder=3
    )

    alphas = [0.45, 0.28, 0.16, 0.09]
    colors = ["#ff6600", "#ff9933", "#ffcc66", "#ffe0aa"]
    for k, ys in enumerate(smooths):
        a = alphas[k] if k < len(alphas) else 0.06
        c = colors[k] if k < len(colors) else "#aaaaaa"
        ax.scatter(
            x, ys, s=3, c=c, alpha=a, label=f"LP {k+1}", edgecolors="none", zorder=3
        )

    ypad = max(np.ptp(y_raw), 1) * 0.08
    for v in feat["clear_minima"]:
        xi = x[v["index"]]
        yi = float(y_raw[v["index"]])
        ax.annotate(
            "",
            xy=(xi, yi),
            xytext=(xi, yi + ypad),
            textcoords="data",
            arrowprops=dict(
                arrowstyle="->", color="#00cc44", lw=1.5, shrinkA=0, shrinkB=4
            ),
        )
    for v in feat["clear_maxima"]:
        xi = x[v["index"]]
        yi = float(y_raw[v["index"]])
        ax.annotate(
            "",
            xy=(xi, yi),
            xytext=(xi, yi - ypad),
            textcoords="data",
            arrowprops=dict(
                arrowstyle="->", color="#ff3366", lw=1.5, shrinkA=0, shrinkB=4
            ),
        )

    ax.set_xlabel("index (1..N)" if x_use_index else "wavelength (nm)")
    ax.set_ylabel("amplitude")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    mr = feat.get("modal_rms")
    mr_s = f" · modal RMS={mr:.2f}" if mr is not None else ""
    rp = feat.get("robust_peak_persistence_frac")
    rv = feat.get("robust_valley_persistence_frac")
    rsum = ""
    if rp is not None and rv is not None:
        rsum = f" · persist p={rp:.2f} v={rv:.2f}"
    if feat.get("n_peaks_roi_residual") is not None:
        rsum += f" · res p/v={feat['n_peaks_roi_residual']}/{feat['n_valleys_roi_residual']}"
    ax.set_title(
        f"score={feat['score']:.1f}{mr_s}{rsum} | "
        f"strong max(red↑) {len(feat['clear_maxima'])} · "
        f"strong min(green↓) {len(feat['clear_minima'])} · "
        f"ROI {feat['n_peaks_roi']}p/{feat['n_valleys_roi']}v"
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_index_html(template_path: Path, out_html: Path, meta: dict, bin_path: Path) -> None:
    tpl = template_path.read_text(encoding="utf-8")
    meta_json = json.dumps(meta, separators=(",", ":"))
    raw = bin_path.read_bytes()
    b64 = base64.b64encode(gzip.compress(raw, compresslevel=9)).decode("ascii")
    html = tpl.replace("__META_JSON__", meta_json)
    html = html.replace("__BUNDLE_B64__", b64)
    out_html.write_text(html, encoding="utf-8")


def save_heatmap_colorbar(
    data: np.ndarray,
    out_path: Path,
    title: str,
    cbar_label: str,
    cmap: str,
    *,
    percentile_lo: float = 2.0,
    percentile_hi: float = 98.0,
    log_scale: bool = False,
) -> None:
    """data shape (40, 40); NaN masked. Colorbar: linear percentiles or optional log (양수만)."""
    z = np.ma.masked_invalid(data.astype(np.float64))
    finite = z.compressed()
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=150, layout="constrained")
    if finite.size == 0:
        lo, hi = 0.0, 1.0
        im = ax.imshow(
            z, origin="upper", interpolation="nearest", cmap=cmap, vmin=lo, vmax=hi
        )
    elif log_scale:
        z_plot = np.ma.masked_where(z <= 0.0, z)
        pos = z_plot.compressed()
        if pos.size == 0:
            lo, hi = 1e-12, 1.0
            im = ax.imshow(
                z_plot,
                origin="upper",
                interpolation="nearest",
                cmap=cmap,
                norm=LogNorm(vmin=lo, vmax=hi),
            )
        else:
            lo = float(np.percentile(pos, percentile_lo))
            hi = float(np.percentile(finite, percentile_hi))
            lo = max(lo, 1e-12)
            if hi <= lo:
                hi = lo * 10.0
            im = ax.imshow(
                z_plot,
                origin="upper",
                interpolation="nearest",
                cmap=cmap,
                norm=LogNorm(vmin=lo, vmax=hi),
            )
    else:
        lo = float(np.percentile(finite, percentile_lo))
        hi = float(np.percentile(finite, percentile_hi))
        if hi <= lo:
            hi = lo + 1e-9
        im = ax.imshow(
            z, origin="upper", interpolation="nearest", cmap=cmap, vmin=lo, vmax=hi
        )
    ax.set_title(title)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "txt_path",
        type=Path,
        default=Path("40_SC_20per_1s_sample_77K_Au_region.txt.txt"),
        nargs="?",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="출력 디렉터리 (미지정 시 입력 txt 경로의 stem과 같은 이름의 폴더)",
    )
    p.add_argument("--roi-lo", type=float, default=675.0)
    p.add_argument("--roi-hi", type=float, default=775.0)
    p.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[9, 25, 55, 111],
        help="Savitzky–Golay window lengths (odd enforced)",
    )
    p.add_argument(
        "--prominence-frac",
        type=float,
        default=0.020,
        help="블렌딩된 effective spread에 곱해 prominence 하한 곡선을 만듦 (클수록 덜 민감)",
    )
    p.add_argument(
        "--clear-prominence-factor",
        type=float,
        default=1.65,
        help="명확한 국소 최대/최소: 해당 파장 prominence 하한의 이 배수 이상",
    )
    p.add_argument(
        "--spread-metric",
        type=str,
        choices=("var", "iqr", "std", "mad"),
        default="iqr",
        help="각 파장(행)에서 1600픽셀만 대상: var / iqr / std / mad (전 큐브 단일 variance 아님)",
    )
    p.add_argument(
        "--spread-mix-local",
        type=float,
        default=0.5,
        help="0~1. inverse: shape_core=(ref/safe)^gamma 가중; 클수록 저분산 억제·고분산 완화가 강함. direct: 파장별/median 혼합",
    )
    p.add_argument(
        "--spread-smooth-window",
        type=int,
        default=21,
        help="파장축 이동평균 창(홀수, <3이면 스무딩 끔)",
    )
    p.add_argument(
        "--prominence-spread-mode",
        type=str,
        choices=("inverse", "direct"),
        default="inverse",
        help="inverse: cross-spread 큰 파장에서 피크·골 검출 우선(하한 낮음). direct: spread에 비례한 하한(이전 동작)",
    )
    p.add_argument(
        "--prominence-spread-ref",
        type=str,
        choices=("median", "p75", "p90"),
        default="p75",
        help="inverse: 전 파장 spread 분포의 기준(중앙값·분위수). p75/p90일수록 «전체보다 낮은» 파장 억제 강함",
    )
    p.add_argument(
        "--prominence-spread-gamma",
        type=float,
        default=2.5,
        help="inverse: (ref/spread_safe)^gamma 에 쓰는 지수. 클수록 저 spread 파장 prominence 하한이 급격히 증가",
    )
    p.add_argument(
        "--peak-distance",
        type=int,
        default=12,
        help="인접 피크·골 최소 간격(샘플); 클수록 덜 촘촘히 검출",
    )
    p.add_argument(
        "--peak-wlen",
        type=int,
        default=0,
        help="find_peaks wlen(샘플). 0이면 미사용(기본, 기존과 동일). >0이면 prominence 평가 구간 제한으로 겹침·넓은 돌출에 덜 민감",
    )
    p.add_argument(
        "--robust-persistence-match-samples",
        type=int,
        default=5,
        help="바로 이전 LP 단계 피크/골과 ±이 샘플 이내면 동일 특징으로 간주(다단계 windows일 때만)",
    )
    p.add_argument(
        "--robust-median-detrend-window",
        type=int,
        default=21,
        help="0 또는 <3이면 끔. 홀수 권장. 스무딩−median_filter 베이스 잔차에서 보조 피크/골 개수(n_*_residual) 계산",
    )
    p.add_argument(
        "--robust-residual-prom-scale",
        type=float,
        default=0.35,
        help="잔차 find_peaks prominence = max(prom_vec*이 값, 작은 바닥)",
    )
    p.add_argument(
        "--heatmap-activation-pct-lo",
        type=float,
        default=2.0,
        help="activation 히트맵 색 vmin 백분위. 로그 모드에서는 양수 점수만으로 하한 계산",
    )
    p.add_argument(
        "--heatmap-activation-pct-hi",
        type=float,
        default=98.0,
        help="activation 히트맵 색 vmax 백분위. 소수 초고점수 픽셀만 있으면 98이어도 나머지가 거의 검게 보일 수 있음→ 90~95 권장",
    )
    p.add_argument(
        "--heatmap-activation-log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="activation 히트맵 colorbar: 기본 로그. 선형은 --no-heatmap-activation-log",
    )
    p.add_argument(
        "--x-index",
        action="store_true",
        help="use 1..N for plot x instead of wavelength",
    )
    p.add_argument(
        "--no-pixel-pngs",
        action="store_true",
        help="skip writing output/pixels/*.png (faster; use HTML hover for spectra)",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="tqdm 진행 표시 끄기",
    )
    p.add_argument(
        "--modal-hist-bins",
        type=int,
        default=64,
        help="파장별 최빈 진폭용 히스토그램 빈 개수",
    )
    p.add_argument(
        "--viz-medium-prominence-frac",
        type=float,
        default=0.62,
        help="spectra.bin·HTML 팝업: clear 미만이나 이 비율×clear 이상인 ROI 피크/골을 «중간» 화살표로 표시. 0이면 clear만",
    )
    args = p.parse_args()
    if args.prominence_spread_gamma <= 0.0:
        p.error("--prominence-spread-gamma must be > 0")
    if args.peak_wlen < 0:
        p.error("--peak-wlen must be >= 0")
    if args.robust_residual_prom_scale <= 0.0:
        p.error("--robust-residual-prom-scale must be > 0")
    if args.viz_medium_prominence_frac < 0.0:
        p.error("--viz-medium-prominence-frac must be >= 0")

    txt = args.txt_path.resolve()
    out = (args.out if args.out is not None else Path(txt.stem)).resolve()
    out.mkdir(parents=True, exist_ok=True)
    pixels_dir = out / "pixels"
    pixels_dir.mkdir(exist_ok=True)

    def pxiter(desc: str):
        it = range(N_PIXEL)
        if args.no_progress:
            return it
        return tqdm(it, desc=desc, unit="px", mininterval=0.2)

    wl, y = load_cube(txt)
    n_wl = y.shape[0]

    modal_curve = modal_amplitude_curve(y, args.modal_hist_bins)
    modal_rms = modal_deviation_rms(y, modal_curve)

    cross_spread = per_wavelength_cross_spread(y, args.spread_metric)
    prom_vec, spread_info = spread_to_prominence_curve(
        cross_spread,
        args.prominence_frac,
        args.spread_mix_local,
        args.spread_smooth_window,
        args.prominence_spread_mode,
        args.prominence_spread_gamma,
        args.prominence_spread_ref,
    )
    spread_used = prom_vec / max(args.prominence_frac, 1e-15)
    np.save(out / "per_wavelength_cross_spread.npy", cross_spread)
    np.save(out / "prominence_spread_effective.npy", spread_used)
    np.save(out / "prominence_min_per_wavelength.npy", prom_vec)

    windows = list(args.windows)
    n_level = len(windows)
    smooth_stack = np.zeros((n_level, n_wl, N_PIXEL), dtype=np.float64)
    for j in pxiter("Savitzky–Golay"):
        ys = lowpass_levels(y[:, j], windows)
        for li, s in enumerate(ys):
            smooth_stack[li, :, j] = s
    y_smooth_main = smooth_stack[-1]
    y_smooth_prev = smooth_stack[-2] if n_level >= 2 else None

    activations = np.zeros(N_PIXEL, dtype=np.float64)
    peak_persist_frac = np.full(N_PIXEL, np.nan, dtype=np.float64)
    valley_persist_frac = np.full(N_PIXEL, np.nan, dtype=np.float64)
    n_peaks_prev_roi = np.full(N_PIXEL, np.nan, dtype=np.float64)
    n_valleys_prev_roi = np.full(N_PIXEL, np.nan, dtype=np.float64)
    n_peaks_residual_roi = np.full(N_PIXEL, np.nan, dtype=np.float64)
    n_valleys_residual_roi = np.full(N_PIXEL, np.nan, dtype=np.float64)
    activation_x_peak_persist = np.full(N_PIXEL, np.nan, dtype=np.float64)

    per_pixel: list[dict] = []
    for j in pxiter("피크/골·특징"):
        y_col = y_smooth_main[:, j]
        y_pr = y_smooth_prev[:, j] if y_smooth_prev is not None else None
        feat = features_for_pixel(
            wl,
            y[:, j],
            y_col,
            args.roi_lo,
            args.roi_hi,
            prom_vec,
            args.clear_prominence_factor,
            args.peak_distance,
            args.peak_wlen,
        )
        aux = robust_extension_features(
            wl,
            y_col,
            y_pr,
            prom_vec,
            args.roi_lo,
            args.roi_hi,
            args.peak_distance,
            args.robust_persistence_match_samples,
            args.robust_median_detrend_window,
            args.robust_residual_prom_scale,
            args.peak_wlen,
        )
        feat.update(aux)
        activations[j] = feat["score"]
        feat["modal_rms"] = float(modal_rms[j])
        if aux["robust_peak_persistence_frac"] is not None:
            peak_persist_frac[j] = float(aux["robust_peak_persistence_frac"])
            valley_persist_frac[j] = float(aux["robust_valley_persistence_frac"])
            n_peaks_prev_roi[j] = float(aux["n_peaks_roi_prev_smooth"])
            n_valleys_prev_roi[j] = float(aux["n_valleys_roi_prev_smooth"])
            activation_x_peak_persist[j] = feat["score"] * (
                0.2 + 0.8 * float(aux["robust_peak_persistence_frac"])
            )
        if aux["n_peaks_roi_residual"] is not None:
            n_peaks_residual_roi[j] = float(aux["n_peaks_roi_residual"])
            n_valleys_residual_roi[j] = float(aux["n_valleys_roi_residual"])
        per_pixel.append(feat)

    act_map = activations.reshape(GRID, GRID)
    act_ht_title = "activation (ROI peak+valley prominence score)"
    if args.heatmap_activation_log:
        act_ht_title += " [log color scale]"
    save_heatmap_colorbar(
        act_map,
        out / "heatmap_activation_40x40.png",
        act_ht_title,
        "score (a.u.)",
        "magma",
        percentile_lo=args.heatmap_activation_pct_lo,
        percentile_hi=args.heatmap_activation_pct_hi,
        log_scale=args.heatmap_activation_log,
    )

    save_heatmap_colorbar(
        modal_rms.reshape(GRID, GRID),
        out / "heatmap_modal_deviation_rms_40x40.png",
        "RMS vs per-λ histogram mode spectrum (all pixels)",
        "RMS amplitude",
        "inferno",
    )

    valley_nm = np.full(N_PIXEL, np.nan, dtype=np.float64)
    valley_w_nm = np.full(N_PIXEL, np.nan, dtype=np.float64)
    for j, feat in enumerate(per_pixel):
        if feat["valley_primary_nm"] is not None:
            valley_nm[j] = feat["valley_primary_nm"]
        if feat["valley_width_nm"] is not None:
            valley_w_nm[j] = feat["valley_width_nm"]

    save_heatmap_colorbar(
        valley_nm.reshape(GRID, GRID),
        out / "heatmap_valley_wavelength_40x40.png",
        "dominant valley wavelength (ROI, highest prominence)",
        "wavelength (nm)",
        "turbo",
    )
    save_heatmap_colorbar(
        valley_w_nm.reshape(GRID, GRID),
        out / "heatmap_valley_width_40x40.png",
        "dominant valley width (FWHM @ 0.5 rel. height on −smoothed)",
        "width (nm)",
        "viridis",
    )

    peak_nm = np.full(N_PIXEL, np.nan, dtype=np.float64)
    peak_w_nm = np.full(N_PIXEL, np.nan, dtype=np.float64)
    peak_count = np.zeros(N_PIXEL, dtype=np.float64)
    for j, feat in enumerate(per_pixel):
        if feat["peak_primary_nm"] is not None:
            peak_nm[j] = feat["peak_primary_nm"]
        if feat["peak_width_nm"] is not None:
            peak_w_nm[j] = feat["peak_width_nm"]
        peak_count[j] = float(feat["n_peaks_roi"])

    save_heatmap_colorbar(
        peak_nm.reshape(GRID, GRID),
        out / "heatmap_peak_wavelength_40x40.png",
        "dominant peak wavelength (ROI, highest prominence)",
        "wavelength (nm)",
        "turbo",
    )
    save_heatmap_colorbar(
        peak_w_nm.reshape(GRID, GRID),
        out / "heatmap_peak_width_40x40.png",
        "dominant peak width (FWHM @ 0.5 rel. height on smoothed)",
        "width (nm)",
        "plasma",
    )
    save_heatmap_colorbar(
        peak_count.reshape(GRID, GRID),
        out / "heatmap_peak_count_roi_40x40.png",
        "number of peaks in ROI (find_peaks on smoothed)",
        "count",
        "cividis",
    )

    save_heatmap_colorbar(
        peak_persist_frac.reshape(GRID, GRID),
        out / "heatmap_robust_peak_persistence_frac_40x40.png",
        "peak persistence: frac of ROI peaks matched on prev LP (±samples)",
        "fraction",
        "viridis",
    )
    save_heatmap_colorbar(
        valley_persist_frac.reshape(GRID, GRID),
        out / "heatmap_robust_valley_persistence_frac_40x40.png",
        "valley persistence: frac of ROI valleys matched on prev LP",
        "fraction",
        "viridis",
    )
    save_heatmap_colorbar(
        n_peaks_prev_roi.reshape(GRID, GRID),
        out / "heatmap_peak_count_roi_prev_smooth_40x40.png",
        "ROI peak count on penultimate Savitzky–Golay level",
        "count",
        "cividis",
    )
    save_heatmap_colorbar(
        n_valleys_prev_roi.reshape(GRID, GRID),
        out / "heatmap_valley_count_roi_prev_smooth_40x40.png",
        "ROI valley count on penultimate Savitzky–Golay level",
        "count",
        "cividis",
    )
    save_heatmap_colorbar(
        n_peaks_residual_roi.reshape(GRID, GRID),
        out / "heatmap_peak_count_roi_residual_40x40.png",
        "ROI peaks on median-detrended residual (auxiliary)",
        "count",
        "plasma",
    )
    save_heatmap_colorbar(
        n_valleys_residual_roi.reshape(GRID, GRID),
        out / "heatmap_valley_count_roi_residual_40x40.png",
        "ROI valleys on median-detrended residual (auxiliary)",
        "count",
        "plasma",
    )
    save_heatmap_colorbar(
        activation_x_peak_persist.reshape(GRID, GRID),
        out / "heatmap_activation_x_peak_persistence_40x40.png",
        "activation score × (0.2 + 0.8·peak persistence frac)",
        "score (a.u.)",
        "magma",
        percentile_lo=args.heatmap_activation_pct_lo,
        percentile_hi=args.heatmap_activation_pct_hi,
        log_scale=args.heatmap_activation_log,
    )

    np.save(out / "activations.npy", activations)
    np.save(out / "modal_amplitude_curve.npy", modal_curve)
    np.save(out / "modal_deviation_rms.npy", modal_rms)
    np.save(out / "valley_wavelength_nm.npy", valley_nm)
    np.save(out / "valley_width_nm.npy", valley_w_nm)
    np.save(out / "peak_wavelength_nm.npy", peak_nm)
    np.save(out / "peak_width_nm.npy", peak_w_nm)
    np.save(out / "peak_count_roi.npy", peak_count)
    np.save(out / "robust_peak_persistence_frac.npy", peak_persist_frac)
    np.save(out / "robust_valley_persistence_frac.npy", valley_persist_frac)
    np.save(out / "peak_count_roi_prev_smooth.npy", n_peaks_prev_roi)
    np.save(out / "valley_count_roi_prev_smooth.npy", n_valleys_prev_roi)
    np.save(out / "peak_count_roi_residual.npy", n_peaks_residual_roi)
    np.save(out / "valley_count_roi_residual.npy", n_valleys_residual_roi)
    np.save(out / "activation_x_peak_persistence.npy", activation_x_peak_persist)

    bin_path = out / "spectra.bin"
    write_spectra_bin(
        bin_path,
        y,
        smooth_stack,
        per_pixel,
        prom_vec,
        args.clear_prominence_factor,
        args.viz_medium_prominence_frac,
    )

    meta = {
        "grid": GRID,
        "n_wavelength": int(n_wl),
        "spectra_viz_channels": int(SPECTRA_VIZ_CHANNELS),
        "viz_medium_prominence_frac": float(args.viz_medium_prominence_frac),
        "roi_lo": args.roi_lo,
        "roi_hi": args.roi_hi,
        "windows": windows,
        "spread_metric": args.spread_metric,
        "spread_mix_local": float(args.spread_mix_local),
        "spread_smooth_window": spread_info["smooth_window"],
        "prominence_spread_mode": spread_info["prominence_spread_mode"],
        "prominence_inverse_gamma": spread_info["prominence_inverse_gamma"],
        "prominence_inverse_ref": spread_info["prominence_inverse_ref"],
        "prominence_inverse_ref_value": spread_info["prominence_inverse_ref_value"],
        "global_cross_spread_median": spread_info["global_cross_spread_median"],
        "clear_prominence_factor": float(args.clear_prominence_factor),
        "prominence_min_per_wavelength": prom_vec.astype(float).tolist(),
        "modal_hist_bins": int(args.modal_hist_bins),
        "x_axis": "index" if args.x_index else "wavelength",
        "wavelength": wl.astype(float).tolist(),
        "activations": activations.astype(float).tolist(),
        "heatmap_activation_log": bool(args.heatmap_activation_log),
        "heatmap_activation_pct_lo": float(args.heatmap_activation_pct_lo),
        "heatmap_activation_pct_hi": float(args.heatmap_activation_pct_hi),
        "modal_amplitude_curve": modal_curve.astype(float).tolist(),
        "modal_deviation_scores": modal_rms.astype(float).tolist(),
        "peak_wavelength_nm": _pixel_map_for_json(peak_nm),
        "peak_width_nm": _pixel_map_for_json(peak_w_nm),
        "valley_wavelength_nm": _pixel_map_for_json(valley_nm),
        "valley_width_nm": _pixel_map_for_json(valley_w_nm),
        "peak_count_roi": [float(x) for x in peak_count.tolist()],
        "n_smooth_levels": int(n_level),
        "peak_wlen": int(args.peak_wlen),
        "robust_persistence_match_samples": int(args.robust_persistence_match_samples),
        "robust_median_detrend_window": int(args.robust_median_detrend_window),
        "robust_residual_prom_scale": float(args.robust_residual_prom_scale),
        "robust_peak_persistence_frac": _pixel_map_for_json(peak_persist_frac),
        "robust_valley_persistence_frac": _pixel_map_for_json(valley_persist_frac),
        "peak_count_roi_prev_smooth": _pixel_map_for_json(n_peaks_prev_roi),
        "valley_count_roi_prev_smooth": _pixel_map_for_json(n_valleys_prev_roi),
        "peak_count_roi_residual": _pixel_map_for_json(n_peaks_residual_roi),
        "valley_count_roi_residual": _pixel_map_for_json(n_valleys_residual_roi),
        "activation_x_peak_persistence": _pixel_map_for_json(
            activation_x_peak_persist
        ),
        "pixels": per_pixel,
    }
    write_meta(out / "meta.json", meta)

    tpl = Path(__file__).resolve().parent / "index_template.html"
    build_index_html(tpl, out / "index.html", meta, bin_path)

    if not args.no_pixel_pngs:
        for j in pxiter("픽셀 PNG"):
            r, c = divmod(j, GRID)
            plot_pixel_figure(
                wl,
                y[:, j],
                [smooth_stack[i, :, j] for i in range(smooth_stack.shape[0])],
                per_pixel[j],
                pixels_dir / f"pixel_r{r:02d}_c{c:02d}.png",
                args.x_index,
                modal_curve=modal_curve,
            )

    extra = "" if args.no_pixel_pngs else ", pixels/*.png"
    print(
        f"Wrote: {out / 'index.html'} (embedded spectra), meta.json, spectra.bin, "
        f"heatmap_*_40x40.png{extra}"
    )


if __name__ == "__main__":
    main()
