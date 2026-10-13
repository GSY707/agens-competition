"""
最后一次诊断检查: 剔除 2xfr 轴谐波污染后的纯净特征列判读
重点: 不把 2xfr(58.2Hz) 误当 FTF x5; 确认各样本真实主导特征.
"""
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch, hilbert
from freq_module import feature_freqs

def env_spec(x, fs):
    x=x-np.mean(x); env=np.abs(hilbert(x))
    f,P=welch(env,fs=fs,nperseg=32768,nfft=65536,noverlap=0,window='hann')
    return f,P

def clean_ratios(ef,eP,feat,rpm,maxharm=6):
    ff=feature_freqs(rpm); fv=ff[feat]; bg=np.median(eP[(ef>0.5)&(ef<40)])
    s2=ff['shaft2x']
    out=[]
    for k in range(1,maxharm+1):
        t=fv*k
        if t<1.0: continue
        # 若该阶位置落在 2xfr 附近 (±2Hz), 标记为轴谐波污染
        contaminated = abs(t-s2)<2.0
        ii=np.argmin(np.abs(ef-t)); r=eP[ii]/bg
        out.append((k,t,r,contaminated))
    return out, bg, s2

def review(name,rpm,fs,de_key):
    d=loadmat(f'samples/sample_{name}.mat')
    x=d[de_key].ravel(); ef,eP=env_spec(x,fs)
    print(f"\n===== 样本 {name} ({de_key}, rpm={rpm}, fs={fs}) =====")
    ff=feature_freqs(rpm)
    for feat in ['bpfo','bpfi','bsf','ftf']:
        out,bg,s2=clean_ratios(ef,eP,feat,rpm)
        clean=[(k,t,r) for k,t,r,c in out if not c]
        cont=[(k,t,r) for k,t,r,c in out if c]
        # 纯净列完整性 (剔除污染后, 1~4 阶 >3 占比)
        c4=[r for k,t,r in clean[:4]]; integ=sum(1 for r in c4 if r>3)/len(c4) if c4 else 0
        print(f"  {feat.upper()}={ff[feat]:6.2f}Hz  纯净列: " +
              " ".join(f"x{k}={r:6.1f}" for k,t,r in clean))
        if cont:
            print(f"          ^ 剔除轴谐波(2xfr={s2:.1f}Hz): " +
                  " ".join(f"x{k}@{t:.1f}Hz={r:.1f}" for k,t,r in cont))
        print(f"          纯净列integrity={integ:.2f}  (剔除 2xfr 污染后)")

review('A',1748,12000,'X236_DE_time')
review('B',1750,48000,'X098_DE_time')
review('C',1752,12000,'X211_DE_time')
review('D',1754,12000,'X224_DE_time')

print("\n\n===== 最终判读 =====")
print("规则: 某特征纯净列 integrity>=0.5 且 lead 显著(>10) -> 该特征主导")
