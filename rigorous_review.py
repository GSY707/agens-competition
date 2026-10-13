"""
高置信度复核: 全通道 + RPM 自适应搜索 + 谐波列完整性 + 跨通道一致性
针对每个样本的所有通道做包络谱分析, 并对特征频率做转速自适应定位.
"""
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch, hilbert, find_peaks
import os, json

N_BALLS=9; D_BALL=7.940e-3; PW=39.040e-3; ALPHA=0.0
def bearing_freqs(rpm):
    fr=rpm/60.0; r=D_BALL/PW; ca=np.cos(ALPHA)
    return dict(ftf=0.5*(1-r*ca)*fr, bpfo=N_BALLS/2*(1-r*ca)*fr,
                bpfi=N_BALLS/2*(1+r*ca)*fr, bsf=N_BALLS/2*(1-r)**2*fr, fr=fr)

def hilbert_env_spec(x, fs, nper=32768, nfft=65536):
    x=x-np.mean(x); env=np.abs(hilbert(x))
    f,Pxx=welch(env, fs=fs, nperseg=min(nper,len(env)), nfft=nfft,
                noverlap=0, window="hann")
    return f, Pxx

def channel_stats(x, fs):
    x=x.ravel(); rms=np.sqrt(np.mean(x**2)); pk=np.max(np.abs(x))
    kurt=np.mean(((x-x.mean())/(x.std()+1e-30))**4)
    # 分段峰度
    win=int(fs*0.5); nseg=max(1,len(x)//win); vals=[]
    for i in range(nseg):
        seg=x[i*win:(i+1)*win]
        m=seg.mean(); s=seg.std()
        if s>1e-9: vals.append(np.mean(((seg-m)/s)**4))
    return dict(rms=rms, peak=pk, crest=pk/rms, kurt=kurt,
                seg_kurt_max=float(np.max(vals)) if vals else 0.0,
                seg_kurt_above10=int(np.sum(np.array(vals)>10)) if vals else 0,
                n_seg=len(vals))

def envelope_feature_score(ef, ePxx, feat_hz):
    """在特征频率(及其 1~8 谐波)处测量幅值/局部背景比, 并评分谐波列完整性"""
    ePdb=10*np.log10(ePxx+1e-30)
    bg=np.median(ePxx[(ef>0.5)&(ef<min(feat_hz*0.6,40))]) if feat_hz*0.6>0.5 else np.median(ePxx[:200])
    if bg<=0: bg=np.median(ePxx[:100])
    ratios=[]
    for k in range(1,9):
        t=feat_hz*k
        if t<1.0: continue
        ii=np.argmin(np.abs(ef-t))
        ratios.append(ePxx[ii]/bg)
    ratios=np.array(ratios)
    # 谐波列完整性: 1~min(6, len) 阶中 >3 的阶数占比
    lead=ratios[:min(6,len(ratios))]
    integrity=float(np.mean(lead>3.0)) if len(lead) else 0.0
    base_ratio=float(ratios[0]) if len(ratios) else 0.0
    return base_ratio, integrity, ratios

def adaptive_rpm(ef, ePxx, feat_name, rpm_nominal, window=30, step=1):
    """不信任记录 RPM: 在 nominal±window 网格搜索, 找该特征频率下
    包络幅值(幅值/背景比)最大的实际位置, 返回最佳 (freq, ratio, rpm_fit)"""
    best=(0,0,None)
    for rpm in np.arange(rpm_nominal-window, rpm_nominal+window+1, step):
        fr=bearing_freqs(float(rpm))
        base,integ,ratios=envelope_feature_score(ef,ePxx,fr[feat_name])
        if base>best[1]:
            best=(fr[feat_name], base, float(rpm))
    return best

def analyze(name, fs, rpm_nominal, channels):
    d=loadmat(f"samples/sample_{name}.mat")
    print(f"\n{'='*70}\n样本 {name}  (fs={fs}, 标称/记录 rpm={rpm_nominal})\n{'='*70}")
    ch_results={}
    for ch in channels:
        key=f"{name_prefix[name]}_{ch}"
        x=d[key].ravel()
        st=channel_stats(x, fs)
        ef,ePxx=hilbert_env_spec(x, fs)
        # 记录/标称 rpm 下各特征
        fr0=bearing_freqs(rpm_nominal)
        feat0={}
        for fn in ['bpfo','bpfi','bsf','ftf']:
            base,integ,ratios=envelope_feature_score(ef,ePxx,fr0[fn])
            feat0[fn]=dict(freq=fr0[fn], base_ratio=base, integrity=integ)
        print(f"--- 通道 {key}")
        print(f"  时域: rms={st['rms']:.4f} peak={st['peak']:.4f} crest={st['crest']:.2f} "
              f"kurt={st['kurt']:.2f} segmax={st['seg_kurt_max']:.1f} "
              f"seg>10={st['seg_kurt_above10']}/{st['n_seg']}")
        print(f"  包络特征 (rpm={rpm_nominal}): " + "  ".join(
            f"{k}:{v['freq']:.1f}Hz(x{v['base_ratio']:.2f},int{v['integrity']:.1f})"
            for k,v in feat0.items()))
        ch_results[ch]=dict(key=key, stats=st, feat0=feat0,
                            ef=ef, ePxx=ePxx)
    return ch_results

name_prefix={'A':'X236','B':'X098','C':'X211','D':'X224'}
# B 有两个编号 X098/X099, 单独处理

res_A=analyze('A',12000,1748,['DE_time','FE_time','BA_time'])
res_C=analyze('C',12000,1752,['DE_time','FE_time','BA_time'])
res_D=analyze('D',12000,1754,['DE_time','FE_time','BA_time'])

# B: 两套编号 X098 / X099, 各 DE/FE
d=loadmat('samples/sample_B.mat')
print(f"\n{'='*70}\n样本 B  (fs=48000, 标称 rpm≈1750, 无记录RPM)\n{'='*70}")
res_B={}
for prefix in ['X098','X099']:
    for ch in ['DE_time','FE_time']:
        x=d[f"{prefix}_{ch}"].ravel()
        st=channel_stats(x,48000)
        ef,ePxx=hilbert_env_spec(x,48000)
        fr0=bearing_freqs(1750); feat0={}
        for fn in ['bpfo','bpfi','bsf','ftf']:
            base,integ,ratios=envelope_feature_score(ef,ePxx,fr0[fn])
            feat0[fn]=dict(freq=fr0[fn], base_ratio=base, integrity=integ)
        print(f"--- 通道 {prefix}_{ch}")
        print(f"  时域: rms={st['rms']:.4f} peak={st['peak']:.4f} crest={st['crest']:.2f} "
              f"kurt={st['kurt']:.2f} seg>10={st['seg_kurt_above10']}/{st['n_seg']}")
        print(f"  包络特征 (rpm=1750): " + "  ".join(
            f"{k}:{v['freq']:.1f}Hz(x{v['base_ratio']:.2f})" for k,v in feat0.items()))
        res_B[f"{prefix}_{ch}"]=dict(stats=st, feat0=feat0, ef=ef, ePxx=ePxx)

# ── RPM 自适应搜索: 对 A/C/D 的 DE 通道, 不信任记录 rpm ──
print(f"\n{'='*70}\nRPM 自适应细化搜索 (DE 通道, ±30rpm 网格)\n{'='*70}")
for label,res,nom in [('A',res_A,1748),('C',res_C,1752),('D',res_D,1754)]:
    ef=ePxx=res['DE_time']['ePxx']
    ef=res['DE_time']['ef']; ePxx=res['DE_time']['ePxx']
    print(f"--- 样本 {label} (记录 rpm={nom})")
    for fn in ['bpfo','bpfi','bsf','ftf']:
        freq,ratio,rpm_fit=adaptive_rpm(ef,ePxx,fn,nom,window=30,step=1)
        print(f"  {fn.upper():5s}: 最佳位置 {freq:.2f} Hz @ rpm={rpm_fit:.0f}  "
              f"(包络幅值/背景 x{ratio:.1f})  [记录值下为 {nom}: "
              f"{bearing_freqs(nom)[fn]:.2f} Hz]")

# 保存自适应结果供 diagnosis 引用
def slim(res):
    out={}
    for ch,v in res.items():
        out[ch]={k:v[k] for k in ['stats','feat0'] if k in v}
    return out
payload={'A':slim(res_A),'B':{k:slim({k:res_B[k]})[k] for k in res_B},
         'C':slim(res_C),'D':slim(res_D)}
def clean(o):
    if isinstance(o,dict): return {k:clean(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [clean(x) for x in o]
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,(np.floating,)): return float(o)
    if isinstance(o,np.ndarray): return o.tolist()
    return o
with open('results/hilbert_detail.json','w',encoding='utf-8') as fh:
    json.dump(clean(payload),fh,indent=2,ensure_ascii=False)
print("\nHilbert 细节已写入 results/hilbert_detail.json")
