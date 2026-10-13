"""
最终诊断脚本 (公式已统一至 freq_module, 含 2xfr 轴谐波识别)
运行: python -X utf8 final_diagnosis.py
输出: results/final_diagnosis_data.json
"""
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch, hilbert
from freq_module import feature_freqs
import json, os, math

def env_spec(x, fs, nper=32768, nfft=65536):
    x = x - np.mean(x); env = np.abs(hilbert(x))
    f, P = welch(env, fs=fs, nperseg=min(nper, len(x)), nfft=nfft,
                 noverlap=0, window="hann")
    return f, P

def time_stats(x, fs):
    rms = np.sqrt(np.mean(x**2)); pk = np.max(np.abs(x))
    kurt = np.mean(((x - x.mean()) / (x.std() + 1e-30))**4)
    win = int(fs * 0.5); nseg = max(1, len(x) // win); vals = []
    for i in range(nseg):
        seg = x[i*win:(i+1)*win]; m = seg.mean(); s = seg.std()
        if s > 1e-9: vals.append(np.mean(((seg - m) / s)**4))
    vals = np.array(vals)
    return dict(rms=float(rms), peak=float(pk), crest=float(pk/rms),
                kurt=float(kurt), seg_above10=int((vals > 10).sum()),
                n_seg=int(len(vals)))

def env_ratios(ef, eP, feat_hz, maxharm=6):
    """各阶幅值/局部背景比; 自动把落入 2xfr 的位置标记为轴谐波"""
    bg = np.median(eP[(ef > 0.5) & (ef < 40)])
    if bg <= 0: bg = np.median(eP[:100])
    out = []
    for k in range(1, maxharm + 1):
        t = feat_hz * k
        if t < 1.0: continue
        ii = np.argmin(np.abs(ef - t))
        out.append(dict(k=k, freq=float(t), ratio=float(eP[ii]/bg)))
    return out, float(bg)

def kurt_block_above10(x, fs, thr=10):
    win = int(fs * 0.5); nseg = max(1, len(x) // win)
    cnt = 0
    for i in range(nseg):
        seg = x[i*win:(i+1)*win]; m = seg.mean(); s = seg.std()
        if s > 1e-9 and np.mean(((seg-m)/s)**4) > thr: cnt += 1
    return cnt

def load_de(name):
    d = loadmat(f"samples/sample_{name}.mat")
    if name == "B":
        return d["X098_DE_time"].ravel(), 1750, 48000, "X098_DE_time"
    key = [k for k in d if not k.startswith("__") and k.endswith("_DE_time")][0]
    rpm = int(d[[k for k in d if k.endswith("RPM")][0]].flatten()[0])
    return d[key].ravel(), rpm, 12000, key

def load_all_channels(name):
    d = loadmat(f"samples/sample_{name}.mat")
    if name == "B":
        chans = ["X098_DE_time", "X099_DE_time", "X098_FE_time", "X099_FE_time"]
        fs = 48000
        rpm = 1750
    else:
        de_key = [k for k in d if not k.startswith("__") and k.endswith("_DE_time")][0]
        base = de_key[:-len("_DE_time")]
        chans = [f"{base}_{s}_time" for s in ["DE", "FE", "BA"]]
        fs = 12000
        rpm = int(d[[k for k in d if k.endswith("RPM")][0]].flatten()[0])
    out = {}
    for ch in chans:
        x = d[ch].ravel()
        ef, eP = env_spec(x, fs)
        st = time_stats(x, fs)
        out[ch] = dict(x=x, ef=ef, eP=eP, stats=st, fs=fs)
    return out, rpm

def judge(name, rpm):
    ff = feature_freqs(rpm)
    chans, rpm_eff = load_all_channels(name)
    # 用 DE 通道做主判据
    de_key = list(chans.keys())[0]
    de = chans[de_key]
    ef, eP = de["ef"], de["eP"]
    st = de["stats"]
    result = {
        "sample": name, "rpm": rpm_eff, "fs": de["fs"],
        "features": {k: round(v, 2) for k, v in ff.items()},
        "time": {k: round(v, 3) for k, v in st.items()},
        "kurt_seg_above10": kurt_block_above10(de["x"], de["fs"]),
        "per_feature": {}
    }
    # 各特征在 DE 通道的包络比
    for feat in ["bpfo", "bpfi", "bsf", "ftf"]:
        ratios, bg = env_ratios(ef, eP, ff[feat])
        lead = [r["ratio"] for r in ratios]
        first4 = lead[:4]
        integrity = sum(1 for r in first4 if r > 3) / len(first4)
        # 检查是否有阶落入 2xfr (轴谐波)
        shaft2x = ff["shaft2x"]
        on_2xfr = [r for r in ratios if abs(r["freq"] - shaft2x) < 2.0]
        result["per_feature"][feat] = {
            "freq": ff[feat], "harmonics": ratios, "bg": bg,
            "lead_max": max(lead) if lead else 0,
            "integrity": integrity,
            "on_2xfr_note": [f"x{r['k']}@{r['freq']:.1f}Hz({r['ratio']:.1f})"
                              for r in on_2xfr]
        }
    # 跨通道一致性: 各特征在所有通道的基频(x1)比
    result["cross_channel"] = {}
    for feat in ["bpfo", "bpfi", "bsf", "ftf"]:
        x1s = {}
        for ch, c in chans.items():
            ratios, bg = env_ratios(c["ef"], c["eP"], ff[feat], maxharm=1)
            x1s[ch] = ratios[0]["ratio"] if ratios else 0.0
        result["cross_channel"][feat] = x1s
    return result

def main():
    os.makedirs("results", exist_ok=True)
    all_results = {n: judge(n, rpm) for n, rpm in
                   [("A",1748),("B",1750),("C",1752),("D",1754)]}
    with open("results/final_diagnosis_data.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False,
                  default=lambda o: float(o) if isinstance(o, (np.floating,)) else
                  int(o) if isinstance(o, (np.integer,)) else o)
    # 打印关键摘要
    for n in ["A","B","C","D"]:
        r = all_results[n]
        print(f"\n样本 {n} (rpm={r['rpm']}):")
        for feat in ["bpfo","bpfi","bsf","ftf"]:
            p = r["per_feature"][feat]
            cc = r["cross_channel"][feat]
            cc_str = " ".join(f"{k}:{v:6.2f}" for k,v in cc.items())
            print(f"  {feat.upper()}={p['freq']:7.2f}Hz lead_max={p['lead_max']:9.2f} "
                  f"integrity={p['integrity']:.2f} 2xfr:{p['on_2xfr_note'] or '-'}")
            print(f"    x1跨通道: {cc_str}")
    print("\n已写入 results/final_diagnosis_data.json")

if __name__ == "__main__":
    main()
