# 轴承振动诊断 — 可复现分析说明

## 环境

- Python 3.13 + `numpy`, `scipy` (welch/hilbert/find_peaks), `scipy.io.loadmat`
- 安装命令 (Windows):
  ```powershell
  pip install numpy scipy matplotlib
  ```

## 运行命令

```powershell
# 全样本分析 (A/C/D 使用 12kHz, B 使用 48kHz)
python -X utf8 bearing_diagnosis.py

# 样本 B 深入分析 (双传感器, 无 RPM, 48kHz)
python -X utf8 analyze_B.py

# 样本 C / D 严谨复核 (Hilbert 包络 + 多 rpm 候选 + 分段峰度)
python -X utf8 analyze_CD.py
```

## 输出

- `results/diagnosis.md` — 逐样本诊断报告
- `results/summary.json` — 逐样本 status / confidence / evidence
- `results/analysis_data.json` — 主分析脚本的量化中间数据
- `results/sample_A_spectrum.png` ~ `results/sample_D_spectrum.png` — 各样本图表

## 方法要点

1. **读取**: `scipy.io.loadmat` 读取 `.mat`。A/C/D 含单一 DE 信号 + RPM 键；
   B 含 X098/X099 两套 DE/FE 信号且无 RPM 键。
2. **主功率谱**: Welch PSD (Hann 窗, 16k~32k 点), 用于观察宽带背景与谐波列。
3. **包络解调谱 (核心判据)**: `scipy.signal.hilbert` 取解析信号包络, FFT。
   冲击类故障 (外圈/内圈/滚动体/保持架) 在包络谱中表现为对应特征频率
   (BPFO/BPFI/BSF/FTF) 及其谐波列的显著增强。
4. **时域冲击统计**: RMS、峰值因子 (Crest)、整体峰度 (Kurtosis)、
   分段峰度 (每 0.5 s 一段, >10 的段数) 用于确认间歇冲击。
5. **特征频率定位**: 无 RPM 记录时 (样本 B) 用标称值与多候选转速复核,
   取幅值/背景比在候选转速间一致居首的特征。

## 判读原则

- 主功率谱的 `amp/bg` 比值在背景噪声极低时会失真 (如样本 D), 不可作为判据。
- 最终诊断以**包络谱特征峰 + 时域冲击统计**为准。
