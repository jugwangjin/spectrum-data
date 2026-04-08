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
import numpy as np
from scipy.ndimage import uniform_filter1d
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


def spread_to_prominence_curve(
    spread: np.ndarray,
    prominence_frac: float,
    mix_local: float,
    smooth_window: int,
    mode: str = "inverse",
) -> tuple[np.ndarray, dict]:
    """
    파장별 cross-spread로 prominence 하한 곡선. 선택적 파장축 스무딩 후:

    - inverse (기본): spread 큰 파장 → 하한 낮음(피크·골 잘 잡음), spread 작은 파장 → 하한 높음(억제).
      shape = mix * (median(spread_s) / spread_s_safe) + (1-mix)*1, prom = prominence_frac * median * shape.
    - direct (레거시): spread에 비례해 하한 — spread 큰 파장이 더 까다로움.
    mix_local: inverse에서 1이면 파장별 대비가 최대, 0이면 전 파장 동일 하한(prominence_frac * median).
    global_scale = median(spread_s) 는 파장별 spread 스칼라들의 중앙값.
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

    if mode == "direct":
        spread_used = mix * spread_s + (1.0 - mix) * global_scale
        prom_vec = prominence_frac * spread_used
    elif mode == "inverse":
        med = global_scale
        safe = np.fmax(spread_s, med * 1e-4)
        ratio = med / safe
        shape = mix * ratio + (1.0 - mix) * 1.0
        prom_vec = prominence_frac * med * shape
    else:
        raise ValueError(f"spread vs prominence mode: {mode!r} (use inverse, direct)")

    prom_vec = prom_vec.astype(np.float64)
    info = {
        "global_cross_spread_median": global_scale,
        "mix_local": mix,
        "smooth_window": w if w >= 3 else 0,
        "prominence_spread_mode": mode,
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
) -> dict:
    """Peaks/valleys on y_smooth. prom_vec: 파장별 prominence 하한 (전 큐브 분포 기반)."""
    dist = max(1, int(distance))
    if prom_vec.shape[0] != y_smooth.shape[0]:
        raise ValueError("prom_vec length must match y_smooth")

    pk_i, pk_prop = find_peaks(y_smooth, prominence=prom_vec, distance=dist)
    vl_i, vl_prop = find_peaks(-y_smooth, prominence=prom_vec, distance=dist)

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


def write_spectra_bin(
    path: Path,
    raw: np.ndarray,
    smooth_stack: np.ndarray,
) -> None:
    """
    raw, smooth_stack: (n_wl, n_pixel). smooth_stack shape (n_levels, n_wl, n_pixel).
    File: magic, n_wl, n_pixel, n_chan, float32 column-major [wl, pixel] per channel flattened C-order:
    actually store (n_pixel, n_wl, n_chan) for easy mmap per pixel in JS.
    """
    n_wl, n_pix = raw.shape
    levels = smooth_stack.shape[0]
    n_chan = 1 + levels
    cube = np.stack([raw.T] + [smooth_stack[i].T for i in range(levels)], axis=-1)
    cube = cube.astype(np.float32)
    blob = cube.tobytes(order="C")
    header = struct.pack("<IIII", MAGIC, n_wl, n_pix, n_chan)
    path.write_bytes(header + blob)


def write_meta(path: Path, meta: dict) -> None:
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


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
    ax.set_title(
        f"score={feat['score']:.1f}{mr_s} | "
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
) -> None:
    """data shape (40, 40); NaN masked. Colorbar scaled to finite percentiles."""
    z = np.ma.masked_invalid(data.astype(np.float64))
    finite = z.compressed()
    if finite.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = np.percentile(finite, 2), np.percentile(finite, 98)
        if hi <= lo:
            hi = lo + 1e-9
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=150, layout="constrained")
    im = ax.imshow(z, origin="upper", interpolation="nearest", cmap=cmap, vmin=lo, vmax=hi)
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
        default=0.012,
        help="블렌딩된 effective spread에 곱해 prominence 하한 곡선을 만듦",
    )
    p.add_argument(
        "--clear-prominence-factor",
        type=float,
        default=1.5,
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
        default=0.35,
        help="0~1. 파장별 spread 비중; 나머지는 전 스펙트럼 median(spread) (작을수록 덜 민감)",
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
    p.add_argument("--peak-distance", type=int, default=8)
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
    args = p.parse_args()

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

    activations = np.zeros(N_PIXEL, dtype=np.float64)
    per_pixel: list[dict] = []
    for j in pxiter("피크/골·특징"):
        feat = features_for_pixel(
            wl,
            y[:, j],
            y_smooth_main[:, j],
            args.roi_lo,
            args.roi_hi,
            prom_vec,
            args.clear_prominence_factor,
            args.peak_distance,
        )
        activations[j] = feat["score"]
        feat["modal_rms"] = float(modal_rms[j])
        per_pixel.append(feat)

    act_map = activations.reshape(GRID, GRID)
    save_heatmap_colorbar(
        act_map,
        out / "heatmap_activation_40x40.png",
        "activation (ROI peak+valley prominence score)",
        "score (a.u.)",
        "magma",
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

    np.save(out / "activations.npy", activations)
    np.save(out / "modal_amplitude_curve.npy", modal_curve)
    np.save(out / "modal_deviation_rms.npy", modal_rms)
    np.save(out / "valley_wavelength_nm.npy", valley_nm)
    np.save(out / "valley_width_nm.npy", valley_w_nm)
    np.save(out / "peak_wavelength_nm.npy", peak_nm)
    np.save(out / "peak_width_nm.npy", peak_w_nm)
    np.save(out / "peak_count_roi.npy", peak_count)

    bin_path = out / "spectra.bin"
    write_spectra_bin(bin_path, y, smooth_stack)

    meta = {
        "grid": GRID,
        "n_wavelength": int(n_wl),
        "roi_lo": args.roi_lo,
        "roi_hi": args.roi_hi,
        "windows": windows,
        "spread_metric": args.spread_metric,
        "spread_mix_local": float(args.spread_mix_local),
        "spread_smooth_window": spread_info["smooth_window"],
        "prominence_spread_mode": spread_info["prominence_spread_mode"],
        "global_cross_spread_median": spread_info["global_cross_spread_median"],
        "clear_prominence_factor": float(args.clear_prominence_factor),
        "prominence_min_per_wavelength": prom_vec.astype(float).tolist(),
        "modal_hist_bins": int(args.modal_hist_bins),
        "x_axis": "index" if args.x_index else "wavelength",
        "wavelength": wl.astype(float).tolist(),
        "activations": activations.astype(float).tolist(),
        "modal_amplitude_curve": modal_curve.astype(float).tolist(),
        "modal_deviation_scores": modal_rms.astype(float).tolist(),
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
