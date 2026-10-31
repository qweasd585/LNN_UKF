# LNN_UKF

## LNN+UKF：可解释的水下 AUV 协同定位基准

这是依据 `LNN.docx` 实现的可运行研究代码。它将 LNN 的内部连续状态作为 UKF 的状态空间，用 Sigma 点传播非线性液态动力学，再用声学测距量测更新外部位置与内部协方差。代码还包含 UKF、CKF、EKF、LNN（无 UKF）、LSTM、CTGRU 和 CTRNN 的统一比较，消融、置信度校准和可解释性实验。

当前工作区没有挂载 `LNN_Lack` 文件夹，也没有真实数据文件。因此仓库内置了可复现的合成水下场景，用来验证算法、输出路径和图表。合成结果会在报告中明确标记，不能作为真实 AUV 试验结论。真实数据接入格式见下文“接入 LNN_Lack 数据”。

## 模型对应文档的方式

设 LNN 内部状态为 (h_t\in\mathbb{R}^n)，外部物理位置为 (x_t\in\mathbb{R}^2)，控制/里程计为 (u_t)，量测为到固定信标的距离 (z_t)。每一个 UKF Sigma 点都经过液态时间常数动力学：

\[
\dot h=\frac{\tanh(W_hh+W_uu+b)-h}{\tau(h,u)},\qquad
h_{t+1}=h_t+\bigl(1-e^{-\Delta t/\tau_t}\bigr)(\tilde h_t-h_t).
\]

读出层只表达未建模位移残差：

\[
x_{t+1}^{(i)}=x_t+u_t+A h_{t+1}^{(i)}+b.
\]

默认约束 (AA^\top=I)，即行半正交。这样即使隐藏维数大于二维外部状态，也不会错误地假设矩形矩阵可逆。量测为非线性声学测距：

\[
z_{t,j}=\lVert x_t-b_j\rVert+v_{t,j}.
\]

`CholeskyCovariancePredictor` 使用**仅包含过去新息**的两个 LSTM 分别输出过程噪声和量测噪声的下三角因子：

\[
Q_t=L_{Q,t}L_{Q,t}^\top+\epsilon I,\qquad
R_t=L_{R,t}L_{R,t}^\top+\epsilon I.
\]

其对角线经过 `softplus`，因此始终正定。量测更新同时使用 (P_{hz}) 和 (P_{xz}) 的 Sigma 点交叉协方差，更新隐藏均值、隐藏协方差、物理位置和物理协方差。它是对文档“外部协方差反馈到内部状态”的高维可行实现；不依赖不存在的 (A^{-1})。

## 安装与运行

Python 3.10 以上即可。CPU PyTorch 的安装方式取决于平台；以下命令适用于官方 CPU wheel 源：

```bash
python -m venv .venv
.venv/bin/pip install --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install numpy scipy pandas matplotlib pytest
.venv/bin/python -m pytest -q
```

快速端到端验证（包含比较、四项 LNN+UKF 消融、图表与数据表）：

```bash
.venv/bin/python run_experiment.py --fast --threads 4 --output results/synthetic_fast
```

较完整的合成实验可增加轮数和随机种子：

```bash
.venv/bin/python run_experiment.py \
  --epochs 30 --ablation-epochs 18 --seeds 1 2 3 4 5 \
  --threads 4 --output results/synthetic_multiseed
```

执行结束后，`REPORT.md` 只汇总本次实际产生的数值；`metrics_summary.csv` 与 `ablation_metrics_summary.csv` 提供每航迹 bootstrap 95% CI；`figures/` 保存轨迹、误差、校准、液态时间常数和内部节点贡献图。`trajectory_comparison_six.png` 使用六条局部裁切轨迹叠加全部模型，不纳入远处锚点；`tau_input_vector_norms_six.png` 比较六条轨迹的 tau 向量模值和标准化输入向量模值变化。若要在 Excel 中修改字体或颜色，请使用 `tools/csv_to_xlsx.py` 批量转换 CSV，而不要把格式化结果当作算法输出的一部分提交。

## 接入 LNN_Lack 数据

为了避免猜测旧工程的列顺序，真实数据使用显式的 NPZ 契约：

| 键 | 形状 | 含义 |
| --- | --- | --- |
| `controls` | `[N, T-1, 2]` | 已知名义平面位移，如 DVL/INS 里程计积分 |
| `measurements` | `[N, T, M]` | 到 (M\ge3) 个固定声学信标的距离；缺测为 `NaN` |
| `states` | `[N, T, 2]` | 真值位置，仅用于训练/验证/离线评估 |
| `beacons` | `[M, 2]` | 固定信标坐标 |
| `dt` | 标量，可选 | 采样间隔，默认 1 秒 |

将旧数据的**副本**转换为这一结构，例如：

```python
import numpy as np

# 以下变量必须由旧项目的明确字段映射得到，不能凭列号猜测。
np.savez_compressed(
    "auv_localization.npz",
    controls=controls,          # [N, T-1, 2]
    measurements=ranges,        # [N, T, M]
    states=ground_truth_xy,     # [N, T, 2]
    beacons=beacon_xy,          # [M, 2]
    dt=np.asarray(sample_dt),
)
```

然后运行：

```bash
.venv/bin/python run_experiment.py --data /absolute/path/auv_localization.npz \
  --seeds 1 2 3 4 5 --epochs 40 --ablation-epochs 25 --output results/real_data
```

切分始终按**完整航迹**进行，归一化或噪声调参只能从训练/验证集得到。不要把未来时刻、测试航迹或真值位置当作在线滤波输入。

## 对照与消融

主表包括：

- `EKF`、`UKF`、`CKF`：同一 (x_{t+1}=x_t+u_t) 物理基线、同一声学量测函数，固定 (Q,R) 仅在验证集网格搜索。
- `LNN`、`LSTM`、`CTGRU`、`CTRNN`：同一隐藏宽度、名义位移输入和起始位置，执行开环长期位移残差预测。
- `LNN+UKF`：使用在线声学量测、内部 Sigma 点、动态 (Q,R) 和内部协方差反馈。

直接预测模型和在线滤波器回答不同问题，报告中会分开解释：前者检验动力学外推，后者检验协同定位。若要比较“同样使用量测的网络”，应在真实数据实验中加入同样的因果量测编码器，而不是将未来量测拼接进输入。

消融包括：固定 (Q,R)、取消内部协方差反馈、固定 (	au)、取消半正交读出，以及主表中的无 UKF LNN。每个变体独立训练，不能只在训练好的完整模型上关闭模块后宣称消融结果。

## 数学可解释性实验

运行器会生成以下可量化诊断，而不只展示轨迹图：

1. **完整液态时间向量与完整输入向量**：保存所有单元形成的 \(\tau(t)\in\mathbb{R}^H\)，并与直接输入的控制东/北分量、滤波位置及由控制导出的速度和连续航向角计算完整的两两 Spearman 矩阵，结果在 `tau_input_spearman.csv`。`tau_input_change_comparison.csv` 保存 \(\|\Delta\tau\|_2\) 与标准化 \(\|\Delta z(u)\|_2\) 的对齐变化；`tau_input_vector_norms_six.png` 展示六条轨迹的向量模值变化，原始模值和逐轨迹 z-score 在 `tau_input_vector_norms_by_trajectory.csv`。逐轨迹 Spearman 明细保留在 `tau_input_spearman_by_trajectory.csv`。
2. **输入到 tau 的时间滞后**：对每条选中轨迹计算 \(\rho_{ij}(\ell)=\operatorname{Spearman}(\Delta\tau_i(t),\Delta u_j(t-\ell))\)。正 \(\ell\) 表示输入变化先于 tau 变化；完整结果、跨轨迹汇总和峰值延迟分别在 `tau_input_lagged_spearman_by_trajectory.csv`、`tau_input_lagged_spearman_summary.csv`、`tau_input_delay_summary.csv`，图为 `tau_input_lag_diagnostics.png`。仅当跨轨迹平均峰值 \(|\rho|\ge0.20\) 时标记方向，否则为 `no_clear_lag`。该分析量化时序关联，不能单独证明因果关系。
3. **(\tau) 干预与真实轨迹**：固定输入、初始值和量测，将全部动态 \(\tau(t)\) 乘以 \(0.5,0.75,1,1.25,1.5\)，在 `tau_interventions.csv` 中同时报告相对真实轨迹的 RMSE、MAE、ADE、FDE，以及相对未干预动态 \(\tau(t)\) 的平均/末端偏离、时间对齐重合率
   \[
   O_\epsilon=\frac1T\sum_t\mathbf{1}\{\|\hat x_t^{cf}-\hat x_t\|\le\epsilon\}
   \]
   与栅格占据 IoU。重合率和 IoU 用于衡量干预敏感性；定位性能以真实轨迹误差为准。默认合成场景使用 \(\epsilon=0.10\) m 和 0.50 m 栅格；真实数据应按任务允许定位误差预先设定。独立训练的固定 \(\tau\) 消融在 `ablation_metrics_summary.csv` 的 `Fixed tau` 行。见 `tau_interventions.csv` 和 `tau_trajectory_interventions.png`。
4. **内部节点到位移的闭合与跨轨迹角色**：对每个单元计算 (c_i(t)=A_{:,i}h_i(t))，展示热图并数值检查
   \[
   Ah_t+b=\sum_i c_i(t)+b.
   \]
   `readout_contributions.png` 展示各单元贡献模值。`readout_role_consistency.csv` 使用时间均值有符号贡献向量的跨轨迹两两余弦和平均贡献幅值变异系数，判定 `consistent`、`trajectory_dependent` 或 `weak`；图为 `readout_role_consistency.png`。
5. **协方差闭环与校准**：记录 (|AP_hA^\top-P_x\|_F/\|P_x\|_F)、最小特征值、NEES、NIS 和覆盖率。`lnn_ukf_calibration.png` 只展示实际运行产生的分布。
6. **动态噪声是否有意义**：画出 `tr(Q_t)`、`tr(R_t)` 与新息范数，并报告二者与新息的 Spearman 相关；结果在 `adaptive_noise_diagnostics.png` 与 `noise_adaptation_correlation.csv`。

真实论文实验还应按任务/环境分组切分，覆盖量测稀疏、丢包、离群值、洋流强度、训练样本比例和未见环境等压力条件；每个条件使用至少 5–10 个训练种子，逐航迹配对 bootstrap 或 Wilcoxon 检验，并对多模型比较进行 Holm 校正。

## 保持可解释性的改良建议

1. **改用 square-root UKF**：维护 Cholesky 因子并使用 QR/秩一更新，减少长航时下协方差非正定和条件数恶化；(Q,R) 仍由可解释的下三角因子表示。
2. **增广物理状态而非任意增大黑箱隐藏层**：把洋流偏置、DVL 比例误差或声速偏差作为有单位的随机游走状态，并让 LNN 仅学习剩余动力学。这使每个不确定性来源有明确物理含义。
3. **使用可观测子空间约束**：保留 (AA^\top=I)，对未观测隐藏子空间单独报告方差；不要声称矩形 (A) 能唯一把外部协方差反推为全部内部协方差。
4. **把动态 (Q,R) 校准纳入目标**：除定位 MSE 外使用因果新息似然、NIS 覆盖率和协方差平滑正则化，防止网络仅靠放大 (R_t) 逃避误差。
5. **将 (	au) 与可测机动量绑定**：对 (	au) 的单调性或分段稀疏性加入软约束，并预注册“转弯/加速/稳定航行”片段上的相关与干预检验，而不是事后挑选漂亮曲线。
6. **对声学离群值采用显式鲁棒量测模型**：在 UKF 量测更新中使用创新门控或 Student-t 似然，并报告触发率。它保持“物理量测函数 + 协方差”的解释，而非用不可见的后处理网络覆盖异常。
7. **检验单元语义跨种子稳定性**：按读出向量和 (	au) 响应匹配隐藏单元，只有稳定的单元或子空间才用于物理解释。
