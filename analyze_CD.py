"""
C / D 严谨复核: Hilbert 包络谱 + 多 rpm 候选 + 时域冲击统计
关键: 包络谱幅值/背景比 是最可靠的冲击判据
"""
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch, hilbert, find_peaks

N_BALLS=9; D_BALL=7.940e-3; PW=39.040e-3; ALPHA=0.0
def bearing_freqs(rpm):
    fr=rpm/60.0; r=D_BALL/PW; ca=np.cos(ALPHA)
    return dict(ftf=0.5*(1-r*ca)*fr,bpfo=N_BALLS/2*(1-r*ca)*fr,
                bpfi=N_BALLS/2*(1+r*ca)*fr,bsf=N_BALLS/2*(1-r)**2*fr,fr=fr)

def hilbert_env_spec(x,fs,nper=32768,nfft=65536):
    x=x-np.mean(x); env=np.abs(hilbert(x))
    f,Pxx=welch(env,fs=fs,nperseg=min(nper,len(env)),nfft=nfft,noverlap=0,window="hann")
    return f,Pxx

def kurtos_block(x,fs):
    # 分段峰度, 捕捉间歇冲击
    win=int(fs*0.5); nseg=len(x)//win
    vals=[]
    for i in range(nseg):
        seg=x[i*win:(i+1)*win]
        m=seg.mean(); s=seg.std()
        if s>1e-9:
            vals.append(np.mean(((seg-m)/s)**4))
    return np.array(vals)

def analyze(name, key, fs, rpm_recorded, rpm_candidates):
    d=loadmat(f"samples/sample_{name}.mat")
    x=d[key].ravel()
    N=len(x); dur=N/fs
    rms=np.sqrt(np.mean(x**2)); pk=np.max(np.abs(x))
    kurt=np.mean(((x-x.mean())/(x.std()+1e-30))**4)
    print(f"\n========== 样本 {name} ({key}) ==========")
    print(f"N={N} dur={dur:.1f}s fs={fs} rpm_recorded={rpm_recorded}")
    print(f"RMS={rms:.4f} peak={pk:.4f} crest={pk/rms:.2f} kurt={kurt:.2f}")
    kb=kurtos_block(x,fs)
    print(f"分段峰度: mean={kb.mean():.1f} max={kb.max():.1f} "
          f"(>10 的段数={np.sum(kb>10)}/{len(kb)})")

    f,Pxx=welch(x,fs=fs,nperseg=32768,nfft=65536,window="hann")
    ef,ePxx=hilbert_env_spec(x,fs)
    ePdb=10*np.log10(ePxx+1e-30)
    eth=np.median(ePdb)+6
    epk,_=find_peaks(ePdb,height=eth,distance=4)
    em=(ef[epk]>0.5); eidx=epk[em]
    epairs=sorted(zip(ef[eidx],ePdb[eidx]),key=lambda p:-p[1])
    print("包络谱前10峰值:")
    for pf,pd in epairs[:10]:
        print(f"  {pf:9.2f} Hz  {pd:6.1f} dB")

    ebg=np.median(ePxx[(ef>0.5)&(ef<40)])
    print("包络特征频率幅值/背景比 (rpm候选):")
    for rc in rpm_candidates:
        fr=bearing_freqs(rc)
        cells=[]
        for feat in ['bpfo','bpfi','bsf','ftf']:
            fv=fr[feat]; ii=np.argmin(np.abs(ef-fv))
            cells.append(f"{feat.upper()}={fv:7.2f}(x{ePxx[ii]/ebg:7.2f})")
        print(f"  rpm={rc}: "+" ".join(cells))

for name,key,fs,rc,rms_ in [
    ("C","X211_DE_time",12000,1752,[1745,1748,1752,1756]),
    ("D","X224_DE_time",12000,1754,[1745,1750,1754,1760]),
]:
    analyze(name,key,fs,rc,rms_)

print("\n完成")
