"""
B/D 专项复核 (含公式修正)
修正: BSF = N/2 * (1 - (d/PW)^2) * fr   (而非错误的 N/2*(1-d/PW)^2*fr)
新增判据: 包络冲击间隔自相关 + 正确 BSF 位置核查
"""
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch, hilbert, find_peaks, correlate
import os

N_BALLS=9; D_BALL=7.940e-3; PW=39.040e-3; ALPHA=0.0
ratio=D_BALL/PW
def bearing_freqs(rpm):
    fr=rpm/60.0; ca=np.cos(ALPHA)
    return dict(ftf=0.5*(1-ratio*ca)*fr,
                bpfo=N_BALLS/2*(1-ratio*ca)*fr,
                bpfi=N_BALLS/2*(1+ratio*ca)*fr,
                bsf=N_BALLS/2*(1-ratio**2)*fr,      # 修正后的正确公式
                fr=fr, ratio=ratio)

def print_freq_table(rpm):
    fr=bearing_freqs(rpm)
    print(f"  rpm={rpm}: fr={fr['fr']:.3f}  "
          f"FTF={fr['ftf']:.2f}  BPFO={fr['bpfo']:.2f}  "
          f"BPFI={fr['bpfi']:.2f}  BSF={fr['bsf']:.2f}  (BSF 周期={1/fr['bsf']*1000:.1f} ms)")

def hilbert_env_spec(x, fs, nper=32768, nfft=65536):
    x=x-np.mean(x); env=np.abs(hilbert(x))
    f,Pxx=welch(env, fs=fs, nperseg=min(nper,len(env)), nfft=nfft, noverlap=0, window="hann")
    return f, Pxx, env

def kurt_stats(x, fs):
    rms=np.sqrt(np.mean(x**2)); pk=np.max(np.abs(x))
    kurt=np.mean(((x-x.mean())/(x.std()+1e-30))**4)
    win=int(fs*0.5); nseg=max(1,len(x)//win); vals=[]
    for i in range(nseg):
        seg=x[i*win:(i+1)*win]; m=seg.mean(); s=seg.std()
        if s>1e-9: vals.append(np.mean(((seg-m)/s)**4))
    vals=np.array(vals)
    return dict(rms=rms,peak=pk,crest=pk/rms,kurt=kurt,
                seg_max=float(vals.max()) if len(vals) else 0,
                seg_above10=int((vals>10).sum()) if len(vals) else 0, nseg=len(vals))

def env_feature(ef, ePxx, feat_hz, harmonics=8):
    bg=np.median(ePxx[(ef>0.5)&(ef<min(feat_hz*0.6,40))]) if feat_hz*0.6>0.5 else np.median(ePxx[:200])
    if bg<=0: bg=np.median(ePxx[:100])
    out=[]
    for k in range(1,harmonics+1):
        t=feat_hz*k
        if t<1.0: continue
        ii=np.argmin(np.abs(ef-t)); out.append((k, feat_hz, ePxx[ii]/bg))
    return out, bg

def impact_autocorrelation(env, fs, feat_hz_list, half_ms=200):
    """测量包络冲击的重复周期: 对包络做自相关, 找最接近各特征周期的峰值间隔"""
    env=env-env.mean()
    n=len(env)
    # 限制到 half_ms 内的自相关, 避免长周期轴频污染
    lag_max=int(fs*half_ms/1000)
    ac=np.correlate(env, env, mode='full')[n-1:n-1+lag_max]
    ac=ac/ac[0]
    # 各特征周期 (ms) 对应的 lag
    print("    包络自相关峰值间隔 (对照特征周期):")
    for feat, fv in feat_hz_list:
        period_ms=1000.0/fv
        lag=int(fs*period_ms/1000)
        if lag>=len(ac): continue
        # 在 lag±2 处取局部最大值
        lo=max(0,lag-2); hi=min(len(ac)-1,lag+2)
        local_peak=ac[lo:hi+1].max()
        print(f"      {feat} 周期 {period_ms:8.2f} ms @ lag={lag}: "
              f"自相关={local_peak:.3f}")
    return ac, ac[0:len(ac)]

def review(name, fs, rpm_nominal, channels, labels):
    print(f"\n{'='*70}\n样本 {name}  fs={fs}  rpm(名义/记录)={rpm_nominal}\n{'='*70}")
    print_freq_table(rpm_nominal)
    d=loadmat(f"samples/sample_{name}.mat")
    res={}
    for ch in channels:
        key=f"{labels}_{ch}"
        x=d[key].ravel()
        st=kurt_stats(x, fs)
        ef,ePxx,env=hilbert_env_spec(x, fs)
        fr0=bearing_freqs(rpm_nominal)
        # 各特征在记录转速下的包络比
        feat_lines=[]
        for fn in ['bpfo','bpfi','bsf','ftf']:
            ratios,bg=env_feature(ef,ePxx,fr0[fn])
            feat_lines.append((fn,fr0[fn],bg,ratios))
        print(f"--- 通道 {key}")
        print(f"  时域: rms={st['rms']:.4f} peak={st['peak']:.4f} crest={st['crest']:.2f} "
              f"kurt={st['kurt']:.2f} seg>10={st['seg_above10']}/{st['nseg']}")
        print(f"  包络幅值/背景比 (各阶, rpm={rpm_nominal}):")
        for fn,fv,bg,ratios in feat_lines:
            rstr=" ".join(f"x{k}={r:7.2f}" for k,_,r in ratios)
            print(f"    {fn.upper()}={fv:7.2f} Hz  bg={bg:.3e}  {rstr}")
        res[ch]=dict(key=key,stats=st,ef=ef,ePxx=ePxx,env=env)
    return res

print("### 修正后的特征频率表 ###")
print_freq_table(1748); print_freq_table(1750); print_freq_table(1752); print_freq_table(1754)

res_B=review('B',48000,1750,['DE_time','FE_time','X099_DE_time','X099_FE_time'],'X098') if False else None
# B 特殊: 两套编号
d=loadmat('samples/sample_B.mat')
print("\nB 的 ans 字段 (弱参考标注):", d['ans'].item(), f"  dtype={d['ans'].dtype}")

for ch in ['X098_DE_time','X099_DE_time','X098_FE_time','X099_FE_time']:
    x=d[ch].ravel(); st=kurt_stats(x,48000)
    ef,ePxx,env=hilbert_env_spec(x,48000); fr0=bearing_freqs(1750)
    print(f"--- B 通道 {ch}")
    print(f"  时域: rms={st['rms']:.4f} kurt={st['kurt']:.2f} seg>10={st['seg_above10']}/{st['nseg']}")
    for fn in ['bpfo','bpfi','bsf','ftf']:
        ratios,bg=env_feature(ef,ePxx,fr0[fn])
        rstr=" ".join(f"x{k}={r:7.2f}" for k,_,r in ratios)
        print(f"    {fn.upper()}={fr0[fn]:7.2f} Hz  bg={bg:.3e}  {rstr}")

# B 包络冲击自相关
for ch in ['X098_DE_time']:
    x=d[ch].ravel(); ef,ePxx,env=hilbert_env_spec(x,48000); fr0=bearing_freqs(1750)
    print(f"\n  B {ch} 包络冲击自相关:")
    impact_autocorrelation(env,48000,[('FTF',fr0['ftf']),('BPFO',fr0['bpfo']),
                                       ('BPFI',fr0['bpfi']),('BSF',fr0['bsf'])])

res_C=review('C',12000,1752,['DE_time','FE_time','BA_time'],'X211')
res_D=review('D',12000,1754,['DE_time','FE_time','BA_time'],'X224')

# D 的冲击自相关
for ch in ['X224_DE_time']:
    dmat=loadmat('samples/sample_D.mat'); x=dmat[ch].ravel()
    ef,ePxx,env=hilbert_env_spec(x,12000); fr0=bearing_freqs(1754)
    print(f"\n  D {ch} 包络冲击自相关:")
    impact_autocorrelation(env,12000,[('FTF',fr0['ftf']),('BPFO',fr0['bpfo']),
                                       ('BPFI',fr0['bpfi']),('BSF',fr0['bsf'])])

# C 的冲击自相关
for ch in ['X211_DE_time']:
    dmat=loadmat('samples/sample_C.mat'); x=dmat[ch].ravel()
    ef,ePxx,env=hilbert_env_spec(x,12000); fr0=bearing_freqs(1752)
    print(f"\n  C {ch} 包络冲击自相关 (参照):")
    impact_autocorrelation(env,12000,[('FTF',fr0['ftf']),('BPFO',fr0['bpfo']),
                                       ('BPFI',fr0['bpfi']),('BSF',fr0['bsf'])])

print("\n复核完成 (BSF 公式已修正, 冲击自相关已补充)")
