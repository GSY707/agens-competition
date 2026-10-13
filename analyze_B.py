"""
样本 B 深入分析
B 特殊结构: 含 X098 / X099 两套传感器, 无 RPM 字段, 采样率 48000 Hz
SAMPLE_PARAMETERS.md 标称 B 转速约 1750 rpm
"""
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch, hilbert, find_peaks
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_BALLS = 9
D_BALL = 7.940e-3
PW = 39.040e-3
ALPHA = 0.0

def bearing_freqs(rpm):
    fr = rpm/60.0
    r = D_BALL/PW
    ca = np.cos(ALPHA)
    return dict(ftf=0.5*(1-r*ca)*fr, bpfo=N_BALLS/2*(1-r*ca)*fr,
                bpfi=N_BALLS/2*(1+r*ca)*fr, bsf=N_BALLS/2*(1-r)**2*fr,
                fr=fr, rpm=rpm)

def pst(f, Pxx):
    Pdb = 10*np.log10(Pxx+1e-30)
    base = np.median(Pdb[(f>0.5)&(f<Pxx.max()*f*0.01+0.1)])  # rough
    return Pdb

def hilbert_envelope_spectrum(x, fs):
    """Hilbert 包络 + FFT, 用于冲击类故障 (比 |x-mean| 更稳健)"""
    x = x - np.mean(x)
    analytic = hilbert(x)
    env = np.abs(analytic)
    f, Pxx = welch(env, fs=fs, nperseg=min(16384, len(env)),
                   nfft=32768, noverlap=0)
    return f, Pxx

d = loadmat('samples/sample_B.mat')
fs = 48000
signals = {}
for k in ['X098_DE_time','X098_FE_time','X099_DE_time','X099_FE_time']:
    signals[k] = d[k].ravel()

# 由于没有 RPM, 尝试多个候选转速估计特征频率位置
# 标称 1750 rpm
candidates = [1740, 1745, 1750, 1755, 1760]
print("候选转速及其特征频率 (Hz):")
for c in candidates:
    fr = bearing_freqs(c)
    print(f"  rpm={c}: FTF={fr['ftf']:.2f} BPFO={fr['bpfo']:.2f} "
          f"BPFI={fr['bpfi']:.2f} BSF={fr['bsf']:.2f} fr={fr['fr']:.2f}")

# 对 DE 信号做包络分析 (X098_DE 与 X099_DE)
for signame in ['X098_DE_time','X099_DE_time']:
    x = signals[signame]
    N = len(x); dur = N/fs
    rms = np.sqrt(np.mean(x**2)); pk = np.max(np.abs(x))
    kurt = np.mean(((x-x.mean())/(x.std()+1e-30))**4)
    crest = pk/rms
    print(f"\n===== {signame} =====")
    print(f"N={N} dur={dur:.1f}s rms={rms:.4f} peak={pk:.4f} "
          f"crest={crest:.2f} kurt={kurt:.2f}")

    # 主谱
    f, Pxx = welch(x, fs=fs, nperseg=16384, nfft=32768, window="hann")
    Pdb = 10*np.log10(Pxx+1e-30)
    thresh = np.median(Pdb) + 8
    pk_idx, _ = find_peaks(Pdb, height=thresh, distance=4)
    mask = (f[pk_idx] > 1.0)
    idx2 = pk_idx[mask]
    pairs = sorted(zip(f[idx2], Pdb[idx2]), key=lambda p:-p[1])
    print("主谱显著峰值 (前12):")
    for pf,pd in pairs[:12]:
        print(f"  {pf:9.2f} Hz  {pd:6.1f} dB")

    # 包络谱
    ef, ePxx = hilbert_envelope_spectrum(x, fs)
    ePdb = 10*np.log10(ePxx+1e-30)
    ethresh = np.median(ePdb) + 8
    epk, _ = find_peaks(ePdb, height=ethresh, distance=4)
    emask = (ef[epk] > 0.5)
    eidx = epk[emask]
    epairs = sorted(zip(ef[eidx], ePdb[eidx]), key=lambda p:-p[1])
    print("包络谱显著峰值 (前12):")
    for pf,pd in epairs[:12]:
        print(f"  {pf:9.2f} Hz  {pd:6.1f} dB")

    # 各候选转速下特征频率在包络谱中的幅值/背景比
    ebg = np.median(ePxx[(ef>0.5)&(ef<40)])
    print("包络中各特征频率幅值/背景比 (按候选rpm):")
    for c in candidates:
        fr = bearing_freqs(c)
        line = f"  rpm={c}: "
        for feat in ['bpfo','bpfi','bsf','ftf']:
            fv = fr[feat]
            ii = np.argmin(np.abs(ef - fv))
            ratio = ePxx[ii]/ebg if ebg>0 else np.nan
            line += f"{feat.upper()}={fv:7.2f}(x{ratio:7.1f}) "
        print(line)

print("\n分析完成")
