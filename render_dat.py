"""Render Zhongshan 8-lead, lead-major little-endian int16 DAT (500 Hz)."""
from __future__ import annotations
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample_poly
from wfdb import processing as wfdb_processing
FS=500
TARGET_SAMPLES=5000
TOTAL_WIDTH=6000
TOTAL_HEIGHT=4000
XIAMEN_DAT_GAIN_MV=2.44/1000
LEAD_NAMES=["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
def _load_zhongshan_dat_signal(path: Path) -> tuple[np.ndarray, int]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Zhongshan DAT file: {path}")

    sampling_rate = 500
    num_stored_leads = 8
    num_samples = 5000
    expected_values = num_stored_leads * num_samples
    raw_values = np.fromfile(path, dtype="<i2")
    if raw_values.size != expected_values:
        raise ValueError(
            f"Unexpected DAT payload size for {path}: "
            f"expected {expected_values} int16 values, got {raw_values.size}"
        )

    stored = raw_values.reshape(num_stored_leads, num_samples).astype(np.float32, copy=False)
    lead_i = stored[0]
    lead_ii = stored[1]
    derived = [
        lead_i,
        lead_ii,
        lead_ii - lead_i,
        -0.5 * (lead_ii + lead_i),
        lead_i - 0.5 * lead_ii,
        lead_ii - 0.5 * lead_i,
        stored[2],
        stored[3],
        stored[4],
        stored[5],
        stored[6],
        stored[7],
    ]
    signal_mv = np.vstack(derived).astype(np.float32, copy=False) * XIAMEN_DAT_GAIN_MV
    return signal_mv, sampling_rate
def draw_background_fast(ax: plt.Axes) -> None:
    for x in np.arange(0, TOTAL_WIDTH + 100, 100):
        ax.axvline(x=x, color="red", linewidth=1)
    for y in np.arange(0, TOTAL_HEIGHT + 100, 100):
        ax.axhline(y=y, color="red", linewidth=1)

    x_dots = np.arange(0, TOTAL_WIDTH, 20)
    y_dots = np.arange(0, TOTAL_HEIGHT, 20)
    xx, yy = np.meshgrid(x_dots, y_dots)
    ax.scatter(xx.ravel(), yy.ravel(), c="black", s=1, marker=".", linewidths=0)

def _resample_signal(signal: np.ndarray, src_fs: int) -> np.ndarray:
    if src_fs <= 0 or src_fs == FS:
        return signal.astype(np.float32, copy=False)
    return resample_poly(signal, up=FS, down=src_fs, axis=1).astype(np.float32, copy=False)

def pad_or_trim_signal(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 2:
        raise ValueError(f"Expected 2D signal, got shape {signal.shape}")
    if signal.shape[1] >= TARGET_SAMPLES:
        return signal[:, :TARGET_SAMPLES]
    pad_width = TARGET_SAMPLES - signal.shape[1]
    return np.pad(signal, ((0, 0), (0, pad_width)), mode="constant")

def build_display_signal(raw_signal_mv: np.ndarray) -> np.ndarray:
    all_ecg_concat = [[], [], []]
    for lead_idx in range(12):
        segment = raw_signal_mv[lead_idx][(lead_idx // 3) * 1250 : (lead_idx // 3 + 1) * 1250]
        all_ecg_concat[lead_idx % 3].append(segment)

    display_rows = [np.hstack(row) for row in all_ecg_concat]
    display_rows.append(raw_signal_mv[1])
    return np.vstack(display_rows)

def detect_peaks(signal: np.ndarray) -> np.ndarray:
    try:
        peaks = wfdb_processing.xqrs_detect(sig=signal, fs=FS, verbose=False)
    except Exception:
        peaks = np.array([], dtype=int)
    return np.asarray(peaks, dtype=int)

def choose_peaks(ecg_plot: np.ndarray, raw_signal_mv: np.ndarray) -> np.ndarray:
    peaks_v5 = detect_peaks(raw_signal_mv[10])
    peaks_ii = detect_peaks(ecg_plot[-1])
    if len(peaks_v5) > 0:
        return peaks_v5
    return peaks_ii

def _build_xiamen_display_signal(raw_signal_mv: np.ndarray) -> np.ndarray:
    all_ecg_concat = [[], [], []]
    for lead_idx in range(12):
        segment = raw_signal_mv[lead_idx][(lead_idx // 3) * 1250 : (lead_idx // 3 + 1) * 1250]
        all_ecg_concat[lead_idx % 3].append(segment)

    display_rows = [np.hstack(row) for row in all_ecg_concat]
    display_rows.append(raw_signal_mv[1])
    return np.vstack(display_rows)

def draw_xiamen_visual_layout(ax: plt.Axes, ecg_plot: np.ndarray, peaks: np.ndarray) -> None:
    ax.set_xlim(0, TOTAL_WIDTH)
    ax.set_ylim(0, TOTAL_HEIGHT)
    draw_background_fast(ax)

    x_base = np.linspace(500, 5500, ecg_plot.shape[1])
    peak_times = peaks / FS * 1000 if len(peaks) > 0 else np.array([])
    intervals = np.diff(peak_times) if len(peak_times) > 1 else np.array([])

    for i in range(4):
        row = 3 - i
        y_shift = row * 1000 + 500
        ecg_data_i = ecg_plot[i] * 200 + y_shift
        ax.plot(x_base, ecg_data_i, color="black", linewidth=1.5)

        if i != 3:
            ax.text(500, y_shift + 100, LEAD_NAMES[i], fontsize=24)
            ax.text(1750, y_shift + 100, LEAD_NAMES[3 + i], fontsize=24)
            ax.text(3000, y_shift + 100, LEAD_NAMES[6 + i], fontsize=24)
            ax.text(4250, y_shift + 100, LEAD_NAMES[9 + i], fontsize=24)
            continue

        ax.text(500, y_shift + 100, LEAD_NAMES[1], fontsize=24)
        for j, interval in enumerate(intervals):
            peak_idx = min(int(peaks[j]), len(ecg_data_i) - 1)
            x_mid = 500 + (int(peaks[j]) + int(peaks[j + 1])) / 2
            y_anchor = ecg_data_i[peak_idx]
            ax.annotate(
                f"{int(interval)}",
                (x_mid, y_anchor),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=24,
            )

    ax.text(1700, 0, "25mm/s      10mm/mv      5mm/grid", fontsize=36)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

def render(source, destination):
    signal, fs = _load_zhongshan_dat_signal(source)
    signal = pad_or_trim_signal(_resample_signal(signal, fs))
    peaks = choose_peaks(build_display_signal(signal), signal)
    fig, ax = plt.subplots(figsize=(30,20), dpi=100)
    try:
        fig.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.05)
        draw_xiamen_visual_layout(ax, _build_xiamen_display_signal(signal), peaks)
        destination.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(destination,dpi=100,bbox_inches=None,facecolor="white")
    finally:
        plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,default=Path("data/dat"))
    p.add_argument("--output",type=Path,default=Path("data/images"))
    a=p.parse_args()
    files=sorted(a.input.glob("*.dat"))
    if not files: raise SystemExit("No .dat files found in " + str(a.input))
    for source in files:
        destination=a.output/(source.stem+".png")
        if not destination.exists(): render(source,destination)
    print(f"Rendered/retained {len(files)} images")
if __name__=="__main__": main()
