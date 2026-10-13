"""
滚动体是否正常? -- 专项验证
判据 (球体缺陷 spalling/pitting on the ball):
  1. BSF 基频 + 谐波列 (x1~x5 同向增强) -- 滚动体自转频率
  2. BSF 调制 BPFO/BPFI 产生的边带 (BPFO +/- k*fr, 球体故障特有)
  3. 时域峰度 (球体故障冲击通常周期较长, 峰度中等)
对照 A/B/C/D 四样本, 判断 D 的滚动体是否正常.
"""
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch, hilbert

N=9; D=7.940e-3; PW=39.040e-3; ALPHA=0.0
R=D/PW; CA=np.cos(ALPHA)
def bearing_freqs(rpm):
    fr=rpm/60.0
    return dict(ftf=0.5*(1-R*CA)*fr, bpfo=N/2*(1-R*CA)*fr,
                bpfi=N/2*(1+R*CA)*fr, bsf=0.5*(PW/D - R*CA*CA)*fr, fr=fr)

def env_spec(x, fs):
    x=x-np.mean(x); env=np.abs(hilbert(x))
    f,P=welch(env,fs=fs,nperseg=32768,nfft=65536,noverlap=0,window='hann')
    return f,P

# 加载各样本 DE 通道
samples={}
for n,rpm,fs in [('A',1748,12000),('B',1750,48000),('C',1752,12000),('D',1754,12000)]:
    d=loadmat(f'samples/sample_{n}.mat')
    if n=='B':
        key='X098_DE_time'
    else:
        key=[k for k in d if not k.startswith('__') and k.endswith('_DE_time')][0]
    x=d[key].ravel()
    ef,eP=env_spec(x,fs)
    fr0=bearing_freqs(rpm)
    samples[n]=dict(ef=ef,eP=eP,fr=fr0,fs=fs,key=key)

print("="*64)
print("判据1: BSF 谐波列 (x1~x5 包络幅值/背景比, 滚动体自转)")
print("  (真滚动体故障: 多阶同向增强; 正常: 各阶均接近背景)")
print("="*64)
for n in ['A','B','C','D']:
    s=samples[n]; ef=s['ef']; eP=s['eP']; fr0=s['fr']
    bg=np.median(eP[(ef>0.5)&(ef<40)])
    bsf=fr0['bsf']
    ratios=[]
    for k in range(1,6):
        ii=np.argmin(np.abs(ef-bsf*k)); ratios.append(eP[ii]/bg)
    line="  ".join(f"x{k}={r:7.2f}" for k,r in zip(range(1,6),ratios))
    verdict="同向增强" if sum(1 for r in ratios if r>5)>=3 else "无列"
    print(f"  {n}({s['key'][-8:]}): BSF={bsf:6.2f}Hz  {line}  -> {verdict}")

print()
print("="*64)
print("判据2: 球体故障特有调制边带 (BPFO 上下 k*fr 边带)")
print("  (滚动体缺陷时 BPFO 敲击被 BSF 调制, 出现 BPFO±fr 边带)")
print("="*64)
for n in ['A','B','C','D']:
    s=samples[n]; ef=s['ef']; eP=s['eP']; fr0=s['fr']
    bg=np.median(eP[(ef>0.5)&(ef<40)])
    bpfo,fr=fr0['bpfo'],fr0['fr']
    def ratio(fq):
        ii=np.argmin(np.abs(ef-fq)); return eP[ii]/bg
    main=ratio(bpfo)
    plus=ratio(bpfo+fr) if bpfo+fr<ef.max() else np.nan
    minus=ratio(bpfo-fr) if bpfo-fr>1 else np.nan
    print(f"  {n}: BPFO={bpfo:6.2f} main={main:8.1f}  "
          f"+fr@{bpfo+fr:6.1f}={plus:8.1f}  -fr@{bpfo-fr:6.1f}={minus:8.1f}")

print()
print("="*64)
print("判据3: 时域 (滚动体故障冲击周期 1/BSF ≈ %.1f ms)" % (1000/samples['D']['fr']['bsf']))
print("  (对照: 峰度/峰值因子, 判断 D 是否有球体冲击痕迹)")
print("="*64)
for n in ['A','B','C','D']:
    s=samples[n]
    x=np.array(loadmat(f"samples/sample_{'B' if n=='B' else n}.mat")[s['key']]).ravel()
    rms=np.sqrt(np.mean(x**2)); pk=np.max(np.abs(x))
    kurt=np.mean(((x-x.mean())/(x.std()+1e-30))**4)
    print(f"  {n}: rms={rms:.4f} peak={pk:.4f} crest={pk/rms:.2f} kurt={kurt:.2f}")

print()
print("="*64)
print("汇总: D 滚动体是否正常?")
print("="*64)
