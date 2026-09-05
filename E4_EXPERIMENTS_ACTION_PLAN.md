# E4实验行动计划 - 一键运行准备

**时间**: 2026年9月4日  
**状态**: ⚠️ 需要进一步适配模型加载代码

---

## 📋 实验概述

**目标**: 直接验证"方向一致性"机制 - 当手动将样本朝低风险方向传输时，预测风险是否单调下降

**关键意义**:
- Direction Only消融实验表现出色（仅比Full Model低1.2%）
- 证明方向约束已经学到了有效的预后方向
- E4实验可以直接展示这种方向一致性

---

## ✅ 已完成

1. **消融实验全部完成** (15个训练任务)
   - Direction Only: 0.7087 ± 0.048 (-1.2%)
   - IPCW Only: 0.6777 ± 0.046 (-5.6%)
   - NLL Only: 0.6824 ± 0.056 (-4.9%)

2. **Checkpoint文件可用**
   - Direction Only: 5 folds × 1 checkpoint = 5个
   - IPCW Only: 5 folds × 1 checkpoint = 5个
   - Full Model: 5 folds × 1 checkpoint = 5个
   - 文件格式: `model_best_s{fold}.pth`

3. **E4脚本框架已创建**
   - `/data1/DCT-Reg/scripts/e4_audit_adapted.py` (494行)
   - `/data1/DCT-Reg/scripts/visualize_e4_results.py` (208行)
   - 包含核心逻辑：锚点提取、插值、风险预测、单调性分析

4. **运行脚本已准备**
   - `/data1/DCT-Reg/run_all_e4_experiments.sh`
   - 批量运行所有变体和folds
   - 自动生成可视化

---

## ⚠️ 当前问题

### 模型加载问题

E4脚本在加载模型时遇到以下技术挑战：

1. **模型实例化困难**
   - Checkpoint只包含`state_dict`（OrderedDict格式）
   - 需要知道正确的模型类和初始化参数
   - 不同变体使用不同的父类：
     - Direction Only/IPCW Only/NLL Only: `dct_transport_intervention_consistency`
     - Full Model: `dct_v310_directional_regularized_transport`

2. **数据加载依赖复杂**
   - 需要正确的`SurvivalDatasetFactory`参数
   - 数据路径: `/data1/DCT-Reg/data/dataset_csv` (CSV数据)
   - WSI特征: `/data1/TCGA-UNI2-h-features` (UNI2-h编码器)
   - Split格式: `5fold_uni2h`

3. **模型forward逻辑需要适配**
   - 需要从模型中提取embedding
   - 需要找到预后锚点的存储位置
   - 需要正确的风险预测路径

### 解决方案选项

#### 选项A: 深入适配E4脚本（推荐，2-3天）

**工作内容**:
1. 查看训练代码，了解模型实例化的完整流程
2. 提取checkpoint中隐含的args或config
3. 复现模型创建逻辑（使用`get_model`函数）
4. 适配embedding提取和风险预测
5. 测试和调试

**优点**:
- 一旦成功，可以自动化运行所有实验
- 可重用于未来其他分析
- 结果最可靠

**缺点**:
- 需要2-3天的调试时间
- 依赖于对模型架构的深入理解

#### 选项B: 使用训练脚本的评估模式（替代方案，1天）

**思路**: 
修改训练脚本，添加"干预审计"模式，在测试集上执行干预实验

**优点**:
- 复用现有的模型加载逻辑
- 所有依赖已经解决
- 实现相对简单

**缺点**:
- 需要修改核心训练代码
- 可能影响代码的可维护性

#### 选项C: 启动一个专门的调试子任务（推荐，今天）

**使用Agent Mode的Task工具**:
- 启动一个专门的subagent来适配E4脚本
- 聚焦于模型加载和前向传播
- 完成后返回可直接运行的脚本

---

## 🎯 推荐行动

### 立即执行（今天，2小时）

**使用Task工具启动模型加载适配任务**:

```python
Task(
    description="Adapt E4 model loading",
    prompt="""
    目标：修复 /data1/DCT-Reg/scripts/e4_audit_adapted.py 中的模型加载问题
    
    需要完成：
    1. 查看 survot_rank/training/train_runner.py，了解模型加载流程
    2. 查看 survot_rank/training/model_factory.py 的 get_model 函数
    3. 适配 load_model_and_data 函数：
       - 正确实例化模型
       - 加载checkpoint的state_dict
       - 提取precompute的预后锚点
    4. 适配 extract_embeddings 函数：
       - 从模型forward中提取embedding
    5. 适配 compute_risk_prediction 函数：
       - 从embedding计算风险分数
    6. 在Direction Only fold 0上测试，确保能成功运行
    
    成功标准：
    - 脚本能成功加载checkpoint
    - 能提取test set的embeddings
    - 能找到预后锚点
    - 能计算干预后的风险分数
    - 生成有效的CSV结果
    
    输出：
    - 修复后的 e4_audit_adapted.py
    - 测试运行日志
    - 简短的使用说明
    """,
    subagent_type="generalPurpose",
    environment="local"
)
```

### 明天执行（9月5日，全天）

一旦模型加载问题解决：

1. **运行单个测试** (1小时)
   ```bash
   cd /data1/DCT-Reg
   bash run_all_e4_experiments.sh test  # 测试mode
   ```

2. **运行完整实验** (6-8小时)
   ```bash
   cd /data1/DCT-Reg
   nohup bash run_all_e4_experiments.sh run > e4_experiments.log 2>&1 &
   ```

3. **生成可视化** (1小时)
   ```bash
   python3 scripts/visualize_e4_results.py
   ```

4. **分析结果** (2小时)
   - 计算单调性率
   - 检查方向一致性
   - 撰写实验报告

---

## 📊 预期结果

如果E4实验成功，预期看到：

1. **单调递减率（towards low-risk）> 70%**
   - 大多数患者在朝低风险锚点移动时，风险单调下降

2. **单调递增率（towards high-risk）> 70%**
   - 大多数患者在朝高风险锚点移动时，风险单调上升

3. **平均风险变化显著**
   - α=1.0时，低风险方向: 风险显著下降
   - α=1.0时，高风险方向: 风险显著上升

4. **可视化展示清晰的趋势**
   - 干预曲线呈现预期的单调性
   - 聚合趋势图显示整体方向一致性

---

## 📁 文件清单

### 脚本文件
- ✅ `scripts/e4_audit_adapted.py` - 主实验脚本（⚠️ 需要修复模型加载）
- ✅ `scripts/visualize_e4_results.py` - 可视化脚本
- ✅ `run_all_e4_experiments.sh` - 批量运行脚本

### Checkpoint文件
```
results/dct_v3.10_experiments/robust/
├── direction_only/blca/blca/.../model_best_s{0-4}.pth  ✅
├── ipcw_only/blca/blca/.../model_best_s{0-4}.pth      ✅
└── full/blca/blca/.../model_best_s{0-4}.pth           ✅
```

### 预期输出
```
results/e4_intervention_audit/
├── direction_only/
│   ├── blca_fold0_results.csv
│   ├── blca_fold0_summary.txt
│   ├── ...
│   └── blca_fold4_summary.txt
├── ipcw_only/
│   └── ...
├── full/
│   └── ...
└── visualizations/
    ├── intervention_curves_direction_only.png
    ├── aggregated_trends.png
    └── monotonicity_comparison.png
```

---

## 🚀 总结

**当前状态**: 实验框架完备，只差模型加载这最后一步

**推荐路径**: 
1. 今天：启动subagent适配模型加载代码
2. 明天：运行完整E4实验
3. 后天：分析结果并撰写报告

**成功概率**: 
- 基于Direction Only的强劲性能（-1.2%），E4实验成功率预计 > 80%
- 模型加载适配的技术难度中等，预计1-2天可以解决

**最终目标**: 
直接证明"方向一致性"机制，为论文提供最强有力的实验证据！

---

**需要我现在启动模型加载适配任务吗？**
