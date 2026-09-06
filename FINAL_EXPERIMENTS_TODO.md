# 🎯 最终实验清单 - 明确的待办事项

**更新时间**: 2026-09-05 10:15  
**当前状态**: 核心消融和E4审计已完成

---

## ✅ 已完成的实验总结

### 1. DCT v3.10 消融实验 (BLCA数据集)
| 变体 | Mean C-index | 状态 |
|------|--------------|------|
| Full Model | **0.7175** 🥇 | ✅ 5/5 folds |
| Direction Only | **0.7087** 🥈 | ✅ 5/5 folds |
| NLL Only | **0.6824** 🥉 | ✅ 5/5 folds |
| IPCW Only | **0.6777** | ✅ 5/5 folds |

### 2. E4 方向一致性审计 (BLCA数据集)
| 变体 | Mean std_risk | 状态 |
|------|---------------|------|
| Direction Only | **0.453** 🥇 (最一致) | ✅ 5/5 folds |
| IPCW Only | **0.762** 🥈 | ✅ 5/5 folds |
| Full Model | **0.993** 🥉 (一致性最差) | ✅ 5/5 folds |

### 核心发现
- ✅ Full Model 预测性能最优
- ✅ Direction Only 方向一致性最优
- ✅ 存在**性能-可解释性权衡**

---

## ❌ 必须完成的实验（按优先级排序）

### 🔴 优先级1: Baseline对比实验 ⭐⭐⭐⭐⭐

**重要性**: 🚨 **发表必需** - 没有baseline对比无法发表

**需要实现的Baseline方法**:
```
1. Cox Proportional Hazards (CPH)
   - 库: lifelines 或 scikit-survival
   - 输入: 临床特征 + 病理特征
   
2. DeepSurv
   - 库: pycox 或自己实现
   - 简单的深度Cox模型
   
3. Random Survival Forest (可选但推荐)
   - 库: scikit-survival
   - 经典的机器学习baseline
```

**实验设置**:
- 数据集: BLCA
- 评估: 5-fold交叉验证（与DCT一致）
- 特征: 使用相同的WSI特征提取器
- 指标: C-index

**预期结果**:
```
Full Model (0.7175) > DeepSurv (0.65-0.69?) > Cox (0.60-0.65?)
```

**当前状态**: ❌ **未开始**  
**预计时间**: 2-3天  
**下一步行动**: 
1. 检查项目里是否有baseline实现
2. 如果没有，使用pycox或scikit-survival实现
3. 运行5-fold交叉验证

---

### 🟡 优先级2: 统计显著性检验 ⭐⭐⭐⭐⭐

**重要性**: 🚨 **发表必需** - 证明改进不是随机的

**需要做的分析**:
```python
1. 配对t检验
   - Full Model vs Direction Only (5 folds)
   - Full Model vs 每个Baseline (5 folds)
   
2. Bootstrap置信区间
   - 对每个方法计算95% CI
   
3. 效应量
   - Cohen's d (标准化差异)
```

**代码示例**:
```python
from scipy import stats

# Full Model的5个fold C-index
full_scores = [0.6950, 0.6300, 0.7166, 0.7884, 0.7573]
direction_scores = [0.7035, 0.6695, 0.6988, 0.6708, 0.8009]

# 配对t检验
t_stat, p_value = stats.ttest_rel(full_scores, direction_scores)
print(f"Full vs Direction: t={t_stat:.3f}, p={p_value:.3f}")

# 如果 p < 0.05，则显著
```

**当前状态**: ❌ **未开始**  
**预计时间**: 半天  
**下一步行动**: 创建统计分析脚本

---

### 🟢 优先级3: Kaplan-Meier生存曲线 ⭐⭐⭐⭐

**重要性**: ⭐⭐⭐⭐ 生存分析论文的标准展示

**需要制作的图**:
```
1. KM曲线 - 风险分层
   - 根据Full Model预测分为高/中/低风险组
   - 显示不同组的生存差异
   - Log-rank test p-value
   
2. (可选) 风险校准曲线
   - 预测风险 vs 观察到的生存率
```

**代码示例**:
```python
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

# 假设已有预测风险分数
risk_groups = pd.qcut(risk_scores, q=3, labels=['Low', 'Medium', 'High'])

# 绘制KM曲线
kmf = KaplanMeierFitter()
for group in ['Low', 'Medium', 'High']:
    mask = (risk_groups == group)
    kmf.fit(durations[mask], event_observed[mask], label=group)
    kmf.plot()

# Log-rank检验
results = multivariate_logrank_test(durations, risk_groups, event_observed)
print(f"Log-rank test p-value: {results.p_value:.4f}")
```

**当前状态**: ❌ **未开始**  
**预计时间**: 半天  
**下一步行动**: 提取预测结果并绘制KM曲线

---

## 📊 建议完成的实验（增强论文）

### 🔵 优先级4: 其他癌症类型验证 ⭐⭐⭐

**重要性**: ⭐⭐⭐ 证明方法泛化性

**当前发现**: 
- ⚠️ LUSC和UCEC的Full Model实验**部分完成**
- LUSC: 有3个folds的结果 (fold 1, 2, 4)
- UCEC: 结果格式不同，需要检查

**需要做的**:
1. 检查LUSC/UCEC现有结果的完整性
2. 补充缺失的folds（如果需要）
3. 汇总C-index结果

**预计时间**: 
- 如果结果完整: 1小时（提取数据）
- 如果需要补跑: 1-2天

---

### 🟣 优先级5: 超参数敏感性分析 ⭐⭐⭐

**重要性**: ⭐⭐⭐ 理解方法的行为

**实验设计**:
```python
# 当前Full Model的损失权重
λ_direction = 1.0
λ_ipcw = 1.0

# 测试不同配置
configs = [
    {"λ_direction": 0.5, "λ_ipcw": 1.0},  # 减弱direction
    {"λ_direction": 2.0, "λ_ipcw": 1.0},  # 增强direction
    {"λ_direction": 1.0, "λ_ipcw": 0.5},  # 减弱IPCW
    {"λ_direction": 1.0, "λ_ipcw": 2.0},  # 增强IPCW
]

# 每个配置跑3 folds（不需要5 folds）
# 分析C-index和E4一致性的权衡
```

**目标**: 找到性能和一致性都较好的配置

**当前状态**: ❌ **未开始**  
**预计时间**: 1-2天  
**优先级**: 可以延后到论文revise阶段

---

## 🚫 不需要做的实验

以下实验优先级低，可以跳过或延后：

❌ **Reader Ablation** - 理解OT计划的作用（次要）
❌ **Stage Jitter** - 已有的机制对照实验足够
❌ **嵌入空间可视化** - Nice to have，但不是必需
❌ **案例研究** - 时间充裕时可做
❌ **重复已完成的实验** - DCT消融和E4审计都已完成

---

## 📅 推荐时间规划

### 方案A: 最快发表 (3-4天)

```
Day 1 (今天):
  └─ 实现Baseline方法
     ├─ 检查现有代码
     ├─ 安装pycox/scikit-survival
     └─ 实现Cox和DeepSurv

Day 2-3:
  └─ 运行Baseline实验
     ├─ 5-fold交叉验证
     └─ 收集C-index结果

Day 4:
  └─ 统计分析 + 可视化
     ├─ t检验和置信区间
     ├─ KM曲线
     └─ 整合所有结果
```

### 方案B: 完整实验 (5-7天)

```
Day 1-3: 同方案A

Day 4:
  └─ 补充LUSC/UCEC结果
     ├─ 检查现有数据
     └─ 补跑缺失folds（如需要）

Day 5-6:
  └─ 超参数敏感性分析
     ├─ 3个配置 × 3 folds
     └─ 分析权衡

Day 7:
  └─ 论文撰写
     ├─ 整合所有结果
     └─ 初稿
```

---

## 🎬 立即开始的第一步

### Step 1: 检查Baseline实现

```bash
cd /data1/DCT-Reg

# 搜索现有的baseline代码
echo "=== 搜索Baseline实现 ==="
find . -name "*.py" -type f -exec grep -l "CoxPH\|DeepSurv\|RandomSurvival" {} \;

# 检查是否安装了相关库
echo -e "\n=== 检查Python库 ==="
python3 -c "import pycox; print('pycox:', pycox.__version__)" 2>&1
python3 -c "import sksurv; print('sksurv:', sksurv.__version__)" 2>&1
python3 -c "import lifelines; print('lifelines:', lifelines.__version__)" 2>&1
```

### Step 2: 如果没有Baseline，安装必要的库

```bash
# 安装生存分析库
pip install pycox
pip install scikit-survival
pip install lifelines

# 验证安装
python3 -c "from pycox.models import CoxPH; print('✅ pycox安装成功')"
python3 -c "from sksurv.ensemble import RandomSurvivalForest; print('✅ sksurv安装成功')"
python3 -c "from lifelines import CoxPHFitter; print('✅ lifelines安装成功')"
```

---

## 🎯 关键决策点

### 你现在需要决定：

**Q1: 你想要快速发表还是完整实验？**
- 快速（3-4天）→ 只做优先级1-3
- 完整（5-7天）→ 做优先级1-5

**Q2: 你的代码库里有没有baseline实现？**
- 有 → 直接运行，节省1-2天
- 没有 → 需要先实现

**Q3: 论文投稿目标是什么？**
- 会议 → 快速版本即可
- 期刊 → 建议完整版本

---

## 💡 我的最终建议

**建议**: 执行**方案A（最快发表）**

**理由**:
1. 核心实验（消融+E4）已经完成 ✅
2. Baseline对比是唯一的硬性要求
3. 统计检验和KM曲线半天可以完成
4. 其他实验可以在revise阶段补充

**立即执行**:
```bash
cd /data1/DCT-Reg
# 运行Step 1的检查命令
find . -name "*.py" -type f -exec grep -l "CoxPH\|DeepSurv\|RandomSurvival" {} \;
```

**告诉我结果，我会帮你下一步！**
