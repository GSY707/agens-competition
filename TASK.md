# 轴承振动诊断任务

你是一名旋转机械状态监测与轴承故障诊断工程师。

`samples/` 目录中将提供四份匿名轴承振动记录，样本代号为 A、B、C、D。请结合 `SAMPLE_PARAMETERS.md` 中的实验与轴承参数，独立分析各样本的运行状态，判断其属于正常、内圈故障、外圈故障、滚动体故障，或因证据不足而无法可靠分类。

你可以通过 Agnes Harness 使用终端、安装 Python 及所需科学计算依赖、自行编写并运行分析程序、生成图表。分析路线和验证方式由你决定。

请在工作区中保存可复现的分析代码，并将结果写入 `results/`：

- `diagnosis.md`：逐样本列出数据概况、分析方法、关键可核查证据、最终判断、置信度与不确定性，并概述整体结论。
- `summary.json`：逐样本记录 `status`、`confidence` 和 `evidence`。`status` 使用 `normal`、`inner_race`、`outer_race`、`ball`、`inconclusive` 之一。
- 必要的图表，以及重现分析结果的运行命令。
