"""
第三轮复核: 正确 BSF 公式 + 全特征全谐波系统性异常扫描
不再预设任何"误报/伪影", 只报告数据中的异常并留待解释.
"""
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch, hilbert

N=9; D=7.940e-3; PW=39.040e-3; ALPHA=0.0
R=D/PW; CA=np.cos(ALPHA)
def bearing_freqs(rpm):
    fr=rpm/60.0
    return dict(ftf=0.5*(1-R*CA)*fr,
                bpfo=N/2*(1-R*CA)*fr,
                bpfi=N/2*(1+R*CA)*fr,
                bsf=0.5*(PW/D - R*CA*CA)*fr,   # 正确公式 (不含 N)
                fr=fr)

def hilbert_env_spec(x, fs, nper=32768, nfft=65536):
    x=x-np.mean(x); env=np.abs(hilbert(x))
    f,Pxx=welch(env,fs=fs,nperseg=min(nper,len(env)),nfft=nfft,noverlap=0,window="hann")
    return f,Pxx

def env_ratios(ef,ePxx,fr_dict, feat, maxharm=8):
    """返回该特征 1~maxharm 阶的 (k, freq, ratio) 列表, 用各自局部中位数背景"""
    fv=fr_dict[feat]
    bg=np.median(ePxx[(ef>0.5)&(ef<max(fv*0.6,40))])
    if bg<=0: bg=np.median(ePxx[:100])
    out=[]
    for k in range(1,maxharm+1):
        t=fv*k
        if t<1.0: continue
        ii=np.argmin(np.abs(ef-t))
        out.append((k, t, ePxx[ii]/bg))
    return out, bg

def scan(name, fs, rpm, channels, labels):
    print(f"\n{'='*72}\n样本 {name}  fs={fs}  rpm={rpm}  "
          f"(正确 BSF={bearing_freqs(rpm)['bsf']:.2f} Hz)\n{'='*72}")
    d=loadmat(f"samples/sample_{name}.mat")
    anomalies=[]
    for ch in channels:
        key=f"{labels}_{ch}"
        x=d[key].ravel()
        ef,ePxx=hilbert_env_spec(x,fs)
        fr0=bearing_freqs(rpm)
        print(f"--- {key}")
        for feat in ['bpfo','bpfi','bsf','ftf']:
            ratios,bg=env_ratios(ef,ePxx,fr0,feat)
            # 报告每一阶; 比值>10 且谐波列不完整时标记为"待解释异常"
            lead=[r for r in ratios]
            lead_max=max(r[2] for r in lead)
            # 列完整性: 1~4 阶中 >3 的占比
            first4=lead[:4]
            integrity=sum(1 for r in first4 if r[2]>3)/len(first4)
            markers=[]
            for k,frq,r in lead:
                if r>=10:
                    markers.append(f"x{k}={frq:.1f}Hz({r:.1f})")
            anom = ''
            if lead_max>=10 and integrity<0.5:
                anom='  [孤立突出, 列不完整 -> 待解释]'
            rstr="  ".join(f"x{k}={r:6.2f}" for k,_,r in lead)
            print(f"    {feat.upper()}={fr0[feat]:7.2f} bg={bg:.2e}  {rstr}{anom}")
            if lead_max>=10 and integrity<0.5:
                anomalies.append(dict(sample=name, channel=key, feature=feat,
                                      freq=fr0[feat], lead_max=lead_max,
                                      integrity=integrity, isolated=markers))
    return anomalies

all_anoms=[]
all_anoms+=scan('A',12000,1748,['DE_time','FE_time','BA_time'],'X236')
all_anoms+=scan('C',12000,1752,['DE_time','FE_time','BA_time'],'X211')
all_anoms+=scan('D',12000,1754,['DE_time','FE_time','BA_time'],'X224')
# B 特殊
d=loadmat('samples/sample_B.mat')
print(f"\n{'='*72}\n样本 B  fs=48000  rpm≈1750(标称)  "
      f"(正确 BSF={bearing_freqs(1750)['bsf']:.2f} Hz)\n{'='*72}")
for ch in ['X098_DE_time','X099_DE_time','X098_FE_time','X099_FE_time']:
    x=d[ch].ravel(); ef,ePxx=hilbert_env_spec(x,48000); fr0=bearing_freqs(1750)
    print(f"--- {ch}")
    for feat in ['bpfo','bpfi','bsf','ftf']:
        ratios,bg=env_ratios(ef,ePxx,fr0,feat)
        lead_max=max(r[2] for r in ratios)
        first4=ratios[:4]; integrity=sum(1 for r in first4 if r[2]>3)/len(first4)
        markers=[]
        for k,frq,r in ratios:
            if r>=10: markers.append(f"x{k}={frq:.1f}Hz({r:.1f})")
        anom=''
        if lead_max>=10 and integrity<0.5: anom='  [孤立突出, 列不完整 -> 待解释]'
        rstr="  ".join(f"x{k}={r:6.2f}" for k,_,r in ratios)
        print(f"    {feat.upper()}={fr0[feat]:7.2f} bg={bg:.2e}  {rstr}{anom}")
        if lead_max>=10 and integrity<0.5:
            all_anoms.append(dict(sample='B',channel=ch,feature=feat,
                                  freq=fr0[feat],lead_max=lead_max,
                                  integrity=integrity,isolated=markers))

print("\n\n===== 待解释异常清单 (比值>=10 且谐波列不完整) =====")
for a in all_anoms:
    print(f"  样本{a['sample']} {a['channel']} {a['feature'].upper()} "
          f"@{a['freq']:.1f}Hz lead_max={a['lead_max']:.1f} "
          f"integrity={a['integrity']:.2f}  {a['isolated']}")
print(f"共 {len(all_anoms)} 项")
