# 模型 v3.2：运输引导的槽重聚合（TGSR）

状态：独立候选实现，真实数据性能待验证。这里的 v3.2 是用户指定的新版本名，
不是历史 v3.2 的复现，也不是冻结 DCT-Reg v3.10 的重命名。
`dct_reg` 和默认主方法仍指向 v3.10；唯一论文主稿暂不改写为新方法。

## 1. 研究问题与实现边界

研究假设：跨模态对应能否帮助槽重新选择原始输入中的信息，而不只用于已压缩槽之间的融合？

沿用已有的特征编码、模态内槽注意力、共享原型坐标和最终阶段三几何 OT 预测头。
本版本唯一结构变量是：中间槽如何通过一次反馈重新读取原始特征。
这里的“原始特征”指 WSI 投影后的 patch token 与组学编码后的通路 token，
不是原始图像像素或未处理表达矩阵。

不采用风险锚点、方向损失、剂量单调损失、IPCW 排序或重构损失；
不修改已有 HPSA 文件，也不声称继承其剪枝/因果解释能力。
四个结构臂的训练目标均为共享训练器提供的生存 NLL：

$$L = L_{\mathrm{NLL}}/B.$$

保留原 NLL 的 `alpha_surv=0.15` 设置。模型返回的辅助损失严格为零。
旧基类内的一些参考缓冲区为兼容训练器仍存在，参考配置钩子为空操作，缓冲区保持空/未观察状态。
数据集仍独立在训练折拟合 NLL 时间分箱。v3.2 forward 不更新锚点、
不读取患者结局、不执行低/高风险干预分支，也不输出伪造的 DCR/DMR 字段。

## 2. 计算路径

1. 模态内槽注意力提取局部槽，经过已有共享原型聚合，得到初始槽 $S^w,S^o$。
2. 用归一化余弦代价在两组槽之间求解反馈 OT：

$$C_{ij}=1-\langle\bar S_i^w,\bar S_j^o\rangle,\qquad
P=\operatorname{Sinkhorn}(C,a,b;\varepsilon).$$

反馈计划的边缘是均匀的 $a_i=1/K_w,b_j=1/K_o$，默认 $\varepsilon=0.10$、50 次迭代。
这是一个受控、简洁的初始设计，不是自适应容量、不平衡 OT 或槽重要性估计。
数值上报告实际边缘误差，不把有限迭代结果当作精确解。
最终预测头仍用原模型的阶段条件、三几何、非均匀边缘 OT；二者不能混称。

3. 由同一个计划构建双向匹配上下文：

$$c_i^w=\frac{\sum_jP_{ij}S_j^o}{\sum_jP_{ij}+\epsilon},\qquad
c_j^o=\frac{\sum_iP_{ij}S_i^w}{\sum_iP_{ij}+\epsilon}.$$

4. 将匹配上下文加到槽查询，重新读取对应模态的原始 token $X$：

$$q_i=W_q\operatorname{LN}(S_i)+\gamma W_c\operatorname{LN}(c_i),\qquad
A_{in}=\operatorname{softmax}_{i}\left(q_i^\top W_k\operatorname{LN}(X_n)/\sqrt d\right),$$

$$\widehat A_{in}=A_{in}/(\sum_{n'}A_{in'}+\epsilon),\qquad
u_i=\sum_n\widehat A_{in}W_v\operatorname{LN}(X_n).$$

用 GRU 将 $u_i$ 与旧槽结合，再应用残差 MLP。
默认反馈强度 $\gamma=0.25$，两侧使用同一轮更新前的状态，避免顺序更新偏差。
默认仅一轮重聚合；可选择 1–8 轮，跨轮共享参数并重新求解反馈计划。
槽数量固定，不宣称动态槽数。

5. 重聚合后的槽进入原阶段 OT 和 hazard 预测头。最终标量风险仍为
$-\sum_t\prod_{u\le t}(1-\sigma(z_u))$，不更换风险定义。

反馈计划和重聚合都保留梯度。可选记录 `reaggregation.capture_attention=True`，
导出最后一轮 token 分配和池化权重（detach），仅用于表示诊断，不能当成临床因果贡献。
默认关闭大注意力图记录，保留小型计划和标量诊断。
独立组件支持 token padding mask；整个模型仍遵循旧数据加载器的真实 bag 输入契约，
不支持向初始槽注意力直接传入任意 padding 或缺失模态。

## 3. 四个配对结构对照

| 名称 | 参数值 | 新增操作 |
|---|---|---|
| A `baseline` | `none` | 无反馈，保持初始槽与末端 OT |
| B `self_update` | `self` | 重新读取原始 token，但无跨模态上下文 |
| C `attention_feedback` | `attention` | 相同余弦/温度的双向 softmax 上下文，随后重读 |
| D `ot_feedback` | `ot` | OT 上下文，随后重读（完整候选） |

四臂实例化相同参数，以同种子保证共同骨干初始化一致。
A 的重读参数不参与运算；B 的上下文映射不参与运算，因此不能声称有效参数数或 FLOPs 完全相同。
B/C 是额外更新及反馈算子的针对性对照；应实测时间/显存，不把相同轮数等同于相同计算量。
C/D 使用同一代价几何和温度，主要区别在于独立行归一化与双边缘约束。
`ot` 的反馈强度为零时，槽输出应严格退化到相同权重的 `self`。

旧 DCT-Reg Full 作为额外参考单独保留。0.7175 等旧验证结果不能填到 A/B/C/D 上。
此方案也不能直接宣称首创：SlotSPE 已有迭代槽交互，MESH 已连接槽注意力与 OT。
候选差异是匹配上下文参与后续原始 token 到槽的重聚合，仍需系统查重和实验支持。

## 4. 运行方式

先在配置数据的训练机器上检查环境（本地 `results/` 不包含另一台机器的预测）：

```bash
python scripts/run_dct_v32_experiments.py doctor --cancers blca --data-root /data1/TCGA-UNI2-h-features --data-csv-root data/dataset_csv
python scripts/run_dct_v32_experiments.py plan
python scripts/run_dct_v32_experiments.py smoke --cancers blca --folds 0
python scripts/run_dct_v32_experiments.py run --cancers blca --folds 0,1,2,3,4
```

`plan` 只打印命令，默认 4 臂 × BLCA 5 折 = 20 个任务；`smoke` 为每臂首折 2 epoch、每 epoch 最多 2 batch。
`run` 默认 30 epoch、seed=3；可用 `--seed 11` 或 `--rounds 2` 建立独立实验。
可用 `--variants ot_feedback` 只运行一个臂。GPU/解释器/数据目录均可通过相同命令选项指定。
队列按失败即停、GPU 与任务锁控制执行。默认跳过相同身份已完成的任务；`--force` 明确允许重跑。

输出隔离在 `results/dct_v3.2/<variant>/<cancer>/seed<seed>_<fingerprint>/`，
smoke 则在 `results/dct_v3.2_smoke/`。指纹包含代码版本、当前 Python 源内容、配置与任务覆盖值。
数据内容变更仍须核对训练器归档的 split/data 信息，不能只依赖路径指纹。
模型 checkpoint 记录反馈模式、轮数、强度与求解设置，配置不匹配时拒绝加载。
不要用 `strict=False` 将旧权重冒充 v3.2 已训练权重。

## 5. 验证与证据状态

2026-09-05 本地执行：`python -m pytest -q` 为 **45 passed**；
直接脚本入口（含从其他目录调用）、四臂配置解析及 CLI 候选注册通过。
数据检查明确返回缺少 BLCA 特征、临床 CSV 和 `5fold_uni2h` 划分；未启动真实患者训练。
同时将旧 v3.10 队列测试的一处断言同步到远端已有的 `ABLATION_PARENT` 修正，
未修改冻结模型或其训练脚本。

```bash
python -m pytest tests -q
python -m survot_rank.cli methods --status candidate
```

测试覆盖实际原始 token 重读、跨模态上下文改变 token 分配、双向反传、零反馈对照、
槽/输入形状、有限数、变长批次、token 顺序不变性、组件 mask、CPU 混合精度、
Pathways 模态、NLL/优化器步进、确定性评估、checkpoint 重载与版本隔离、20 任务配置解析。
同时验证 A 在同权重下等于旧骨干的事实预测路径。

这些是软件与合成张量验证，不是真实患者效果。v3.2 目前没有可报告的 C-index 或优越性结论。
现有训练入口仍是按验证集选择 epoch 的开发协议；正式论文需另设未参与选择的外层测试/嵌套评估，
不能把本队列输出改名为独立测试证据。

先比较 D 相对 A/B/C 的完整配对折差异和计算开销；无稳定收益就不扩大实验。
若通过，再扩展癌种、重复种子和独立测试，并做输入噪声/错误模态配对检查。
本候选不承诺风险单调性，因此 DMR 不是本轮通过条件；原 v3.10 的失败审计不得隐藏或改写。
