"""
轴承振动诊断分析脚本

对样本 A/B/C/D 做轴承故障频率谱分析，判断故障类型。

轴承几何参数 (SKF 6205-2RS):
  - 滚动体数 N = 9
  - 球径 d = 7.940 mm
  - 节圆直径 PW = 39.040 mm
  - 接触角 alpha = 0°

故障频率公式 (alpha=0):
  BPFO = N/2 * (1 - d/PW*cos(alpha)) * fr
  BPFI = N/2 * (1 + d/PW*cos(alpha)) * fr
  BSF  = N/2 * (1 - d/PW)^2 * fr
  FTF  = (1/2) * (1 - d/PW*cos(alpha)) * fr
  fr   = RPM/60

分析路线:
  1. 读取 .mat 信号, 做频谱分析 (PSD + 峰值检测)
  2. 在 1x-15x 范围内比对 BPFO/BPFI/BSF/FTF 与理论值
  3. 检测各阶谐波是否同时增强
  4. 结合时域特征 (冲击性、包络解调) 综合判断
"""

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch
import os, json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 轴承参数 ──────────────────────────────────────────────
N_BALLS   = 9
D_BALL    = 7.940e-3   # m
PW_PITCH  = 39.040e-3  # m
ALPHA     = 0.0        # rad

def bearing_frequencies(rpm):
    """返回各特征频率 (Hz)。"""
    fr = rpm / 60.0
    ratio = D_BALL / PW_PITCH
    cos_a = np.cos(ALPHA)
    FTF  = 0.5 * (1 - ratio * cos_a) * fr
    BPFO = N_BALLS / 2 * (1 - ratio * cos_a) * fr
    BPFI = N_BALLS / 2 * (1 + ratio * cos_a) * fr
    BSF  = N_BALLS / 2 * (1 - ratio) ** 2 * fr
    return dict(ftf=FTF, bpfo=BPFO, bpfi=BPFI, bsf=BSF, fr=fr, rpm=rpm)

# ── 信号读取 ─────────────────────────────────────────────
def load_sample(name: str):
    path = os.path.join("samples", f"sample_{name}.mat")
    d = loadmat(path)
    # 找信号键 (排除 RPM 键)
    sig_key = next(k for k in d if not k.startswith("__") and not k.endswith("RPM"))
    x = d[sig_key].ravel()
    # 转速 (可能是 0-d 或 1-d 数组)
    rpm_key = [k for k in d if not k.startswith("__") and "RPM" in k]
    rpm = float(d[rpm_key[0]].flatten()[0]) if rpm_key else 0.0
    return x, rpm, sig_key

# ── 频谱分析 ─────────────────────────────────────────────
def power_spectrum(x, fs, nperseg=8192, nfft=16384):
    f, Pxx = welch(x, fs=fs, nperseg=min(nperseg, len(x)), nfft=nfft, window="hann")
    return f, Pxx

def detect_peaks(f, Pxx, min_freq=5.0, max_order=15):
    """在 [min_freq, max_order*fr] 内找显著峰值。"""
    Pdb = 10 * np.log10(Pxx + 1e-30)
    mean_level = np.median(Pdb[Pdb < 0])
    thresh = mean_level + 10  # 比中位数高 10 dB 才算峰值
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(Pdb, height=thresh, distance=2)
    # 限制频率范围
    mask = (f >= min_freq) & (f <= max_order * 50)
    p_idx = peaks[mask[peaks]]
    pairs = list(zip(f[p_idx], Pdb[p_idx]))
    pairs.sort(key=lambda x: -x[1])
    return pairs

# ── 包络解调 ─────────────────────────────────────────────
def envelope_spectrum(x, fs, fr, nfft=16384):
    """简易包络: 信号 - 均值, 取绝对值, FFT。"""
    env = np.abs(x - np.mean(x))
    env = env - env.mean()
    f, Pxx = welch(env, fs=fs, nperseg=min(8192, len(env)), nfft=nfft)
    return f, Pxx

# ── 主分析函数 ───────────────────────────────────────────
def analyze_sample(name, fs_expected):
    print(f"\n{'='*60}\n样本 {name}\n{'='*60}")
    x, mat_rpm, sig_key = load_sample(name)
    fs = fs_expected

    N = len(x)
    duration = N / fs
    print(f"  信号键: {sig_key}, 点数 {N}, 时长 {duration:.1f}s, fs={fs} Hz")
    print(f"  .mat 记录 RPM: {mat_rpm}")

    # 以 .mat 记录值为准 (更可信)
    rpm = mat_rpm
    freqs = bearing_frequencies(rpm)
    print(f"\n  理论特征频率 (rpm={rpm}):")
    for k, v in freqs.items():
        print(f"    {k:6s}: {v:8.2f} Hz")

    # 主频谱
    f, Pxx = power_spectrum(x, fs)
    peaks = detect_peaks(f, Pxx, min_freq=5.0, max_order=15)

    print(f"\n  主频谱显著峰值 (前 20):")
    print(f"  {'频率(Hz)':>10s}  {'dB':>6s}  {'频率/rpm':>10s}")
    for pf, pdb in peaks[:20]:
        print(f"  {pf:10.2f}  {pdb:6.1f}  {pf/rpm*60:10.2f}")

    # 检查特征频率的谐波是否突出
    print(f"\n  特征频率谐波比对:")
    features = {"BPFO": freqs["bpfo"], "BPFI": freqs["bpfi"],
                "BSF":  freqs["bsf"],  "FTF":  freqs["ftf"]}
    for feat_name, feat_f in features.items():
        harmonics = []
        for k in range(1, 9):
            target = feat_f * k
            if target < 2.0 or target > fs * 0.4:
                continue
            idx = np.argmin(np.abs(f - target))
            amp = Pxx[idx]
            harmonics.append((k, target, amp))
        # 看这些点是否比周围背景高
        bg = np.median(Pxx[(f > 1.0) & (f < feat_f*0.5)])
        for k, target, amp in harmonics:
            if amp > bg * 10:
                print(f"    {feat_name} x{k}: {target:8.2f} Hz  amp/bg = {amp/bg:.1f}  [ALERT]")
            else:
                print(f"    {feat_name} x{k}: {target:8.2f} Hz  amp/bg = {amp/bg:.1f}")

    # 包络频谱
    ef, ePxx = envelope_spectrum(x, fs, rpm)
    e_peaks = detect_peaks(ef, ePxx, min_freq=0.5, max_order=30)
    print(f"\n  包络频谱显著峰值 (前 15):")
    for pf, pdb in e_peaks[:15]:
        print(f"    {pf:8.2f} Hz  {pdb:6.1f} dB")

    # 包络 vs 特征频率比对
    print(f"\n  包络中特征频率比对:")
    for feat_name, feat_f in features.items():
        idx = np.argmin(np.abs(ef - feat_f))
        amp = ePxx[idx]
        bg = np.median(ePxx[(ef > 0.5) & (ef < feat_f*0.5)]) if feat_f*0.5 > 0.5 else np.median(ePxx[:200])
        if bg < 1e-30:
            bg = np.median(ePxx[:100])
        print(f"    {feat_name}: {feat_f:8.2f} Hz  幅值/背景 = {amp/max(bg,1e-30):.2f}")

    # 时域统计
    rms = np.sqrt(np.mean(x**2))
    peak_val = np.max(np.abs(x))
    crest = peak_val / rms if rms > 0 else 0
    kurt = np.mean(((x - x.mean()) / (x.std() + 1e-30))**4)
    print(f"\n  时域统计: RMS={rms:.4f} g, Peak={peak_val:.4f} g, "
          f"Crest={crest:.1f}, Kurtosis={kurt:.1f}")

    return {
        "name": name, "fs": fs, "rpm": rpm,
        "N": N, "duration_s": duration,
        "features": freqs,
        "peaks_main": peaks[:20],
        "peaks_env": e_peaks[:15],
        "rms": rms, "peak": peak_val, "crest": crest, "kurt": kurt,
        "features_main": {k: v for k, v in features.items()},
        "features_env": {k: v for k, v in features.items()},
    }


def save_charts(name, res, f, Pxx, ef, ePxx):
    os.makedirs("results", exist_ok=True)
    freqs = res["features"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

    # 1. 功率谱 (跳过时域波形, 避免额外加载)
    fdb = 10*np.log10(Pxx+1e-30)
    axes[0].plot(f, fdb, color="darkorange", lw=0.5)
    for feat_name, fv in [("BPFO",freqs["bpfo"]),("BPFI",freqs["bpfi"]),("BSF",freqs["bsf"]),("FTF",freqs["ftf"])]:
        for k in range(1, 9):
            axes[0].axvline(fv*k, color="red", alpha=0.3, ls="--")
        axes[0].axvline(fv, color="red", label=f"{feat_name}={fv:.1f}Hz")
    axes[0].set_ylabel("dB PSD")
    axes[0].set_title(f"Sample {name} — Power spectrum with theoretical feature harmonics (red dashed)")
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(0, min(500, res["fs"]*0.4))

    # 2. 包络谱
    efdb = 10*np.log10(ePxx+1e-30)
    axes[1].plot(ef, efdb, color="green", lw=0.5)
    for feat_name, fv in [("BPFO",freqs["bpfo"]),("BPFI",freqs["bpfi"]),("BSF",freqs["bsf"]),("FTF",freqs["ftf"])]:
        for k in range(1, 9):
            axes[1].axvline(fv*k, color="red", alpha=0.3, ls="--")
    axes[1].set_ylabel("dB")
    axes[1].set_title("Envelope spectrum with theoretical feature harmonics")
    axes[1].set_xlim(0, min(200, res["fs"]*0.4))

    # 3. 包络谱局部放大 (0-200 Hz)
    efdb_zoom = 10*np.log10(ePxx+1e-30)
    m = ef <= 200
    axes[2].plot(ef[m], efdb_zoom[m], color="green", lw=0.5)
    for feat_name, fv in [("BPFO",freqs["bpfo"]),("BPFI",freqs["bpfi"]),("BSF",freqs["bsf"]),("FTF",freqs["ftf"])]:
        for k in range(1, 9):
            axes[2].axvline(fv*k, color="red", alpha=0.3, ls="--")
    axes[2].set_ylabel("dB")
    axes[2].set_title("Envelope spectrum zoom (0–200 Hz)")
    axes[2].set_xlim(0, 200)

    plt.tight_layout()
    out = f"results/sample_{name}_spectrum.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  图表已保存: {out}")
    return out


def run_all(fs_map):
    all_results = {}
    for name, fs in fs_map.items():
        res = analyze_sample(name, fs)
        # 重新读取信号用于画图
        x, _, _ = load_sample(name)
        f, Pxx = power_spectrum(x, fs)
        ef, ePxx = envelope_spectrum(x, fs, res["rpm"])
        chart = save_charts(name, res, f, Pxx, ef, ePxx)
        res["chart"] = chart
        all_results[name] = res
    return all_results


if __name__ == "__main__":
    fs_map = {"A": 12000, "B": 48000, "C": 12000, "D": 12000}
    results = run_all(fs_map)

    # 打印摘要 JSON
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for name, r in results.items():
        summary[name] = {
            "fs": r["fs"], "rpm": r["rpm"], "N": r["N"],
            "duration_s": round(r["duration_s"], 2),
            "features": {k: round(v, 2) for k, v in r["features"].items()},
            "rms_g": round(r["rms"], 5), "peak_g": round(r["peak"], 5),
            "crest": round(r["crest"], 2), "kurtosis": round(r["kurt"], 2),
            "top_main_peaks": [(round(p,2), round(d,1)) for p,d in r["peaks_main"][:10]],
            "top_env_peaks": [(round(p,2), round(d,1)) for p,d in r["peaks_env"][:10]],
            "chart": r["chart"],
        }
    def _serialize(v):
        if isinstance(v, (np.integer, np.uint8, np.uint16, np.int64)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (list, tuple)):
            return [_serialize(i) for i in v]
        if isinstance(v, dict):
            return {k: _serialize(x) for k, x in v.items()}
        return v

    summary_out = _serialize(summary)
    with open(os.path.join(out_dir, "analysis_data.json"), "w", encoding="utf-8") as fh:
        json.dump(summary_out, fh, indent=2, ensure_ascii=False)
    print(f"\n分析数据已写入 {out_dir}/analysis_data.json")
