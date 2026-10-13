"""
唯一权威轴承特征频率模块 (SKF 6205-2RS, alpha=0°)
所有脚本必须 import 此模块, 禁止各自硬编码公式.

几何参数 (SAMPLE_PARAMETERS.md):
  N=9, d=7.940mm, PW=39.040mm, alpha=0°

标准公式 (alpha=0°):
  fr   = rpm/60
  FTF  = 0.5 * (1 - (d/PW)) * fr
  BPFO = N/2 * (1 - (d/PW)) * fr
  BPFI = N/2 * (1 + (d/PW)) * fr
  BSF  = 0.5 * (PW/d - (d/PW)) * fr        # 滚动体自转, 不含 N (α=0°)
"""
import numpy as np

N_BALLS = 9
D_BALL  = 7.940e-3   # m
PW_PITCH= 39.040e-3  # m
ALPHA   = 0.0        # rad
_RATIO  = D_BALL / PW_PITCH
_CA     = np.cos(ALPHA)

def feature_freqs(rpm):
    """返回 {ftf, bpfo, bpfi, bsf, fr, shaft2x} @ 给定 rpm。"""
    fr = float(rpm) / 60.0
    ftf  = 0.5 * (1 - _RATIO * _CA) * fr
    bpfo = N_BALLS / 2 * (1 - _RATIO * _CA) * fr
    bpfi = N_BALLS / 2 * (1 + _RATIO * _CA) * fr
    bsf  = 0.5 * (PW_PITCH / D_BALL - _RATIO * _CA ** 2) * fr  # 正确, 不含 N
    return dict(ftf=ftf, bpfo=bpfo, bpfi=bpfi, bsf=bsf,
                fr=fr, shaft2x=2 * fr)   # 2x 轴转频 (系统性背景, 非轴承特征)

def feature_freqs_all():
    return {1748: feature_freqs(1748), 1750: feature_freqs(1750),
            1752: feature_freqs(1752), 1754: feature_freqs(1754)}

if __name__ == "__main__":
    for rpm, f in feature_freqs_all().items():
        print(f"rpm={rpm}: fr={f['fr']:.3f}  2xfr={f['shaft2x']:.2f}  "
              f"FTF={f['ftf']:.2f}  BPFO={f['bpfo']:.2f}  "
              f"BPFI={f['bpfi']:.2f}  BSF={f['bsf']:.2f}")
