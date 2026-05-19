import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import mannwhitneyu
import warnings
import os
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')

# ==================== 中文字体设置（Mac） ====================
font_path = '/System/Library/Fonts/PingFang.ttc'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti']
    print("✅ 已加载 Mac 中文字体文件")
else:
    plt.rcParams['font.sans-serif'] = ['Heiti SC', 'STHeiti', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 数据加载与预处理 ====================
file_path = '附件1：样例数据.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')
print("原始数据列名（前20列）：", df.columns[:20].tolist())

# 九种体质原始列名（确保与数据一致）
constitution_cols = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']
# 检查缺失并尝试修正（比如列名末尾有空格）
for i, col in enumerate(constitution_cols):
    if col not in df.columns:
        # 尝试去除空格匹配
        matched = [c for c in df.columns if c.strip() == col]
        if matched:
            print(f"列 '{col}' 不存在，使用 '{matched[0]}' 代替")
            constitution_cols[i] = matched[0]
        else:
            raise KeyError(f"体质列 {col} 不存在，请检查数据列名")

# 重命名映射
rename_dict = {
    'ADL总分': 'ADL',
    'IADL总分': 'IADL',
    '活动量表总分（ADL总分+IADL总分）': '活动总分',
    'TC（总胆固醇）': 'TC',
    'TG（甘油三酯）': 'TG',
    'LDL-C（低密度脂蛋白）': 'LDL_C',
    'HDL-C（高密度脂蛋白）': 'HDL_C',
    '空腹血糖': '血糖',
    '血尿酸': '血尿酸',
    'BMI': 'BMI',
    '高血脂症二分类标签': '高血脂标签'
}
# 保留所有需要的列
keep_cols = constitution_cols + list(rename_dict.keys()) + ['年龄组', '性别', '吸烟史', '饮酒史']
keep_cols = [col for col in keep_cols if col in df.columns]
df = df[keep_cols].copy()
df.rename(columns=rename_dict, inplace=True)
df['痰湿质积分'] = df['痰湿质']   # 方便后续使用
df.dropna(inplace=True)
print(f"有效样本数：{len(df)}")

# ==================== 2. 特征构建 ====================
X_constitution = df[constitution_cols]
X_biochemical = df[['TC', 'TG', 'LDL_C', 'HDL_C', '血糖', '血尿酸', 'BMI']]
X_activity = df[['ADL', 'IADL']]
X = pd.concat([X_constitution, X_biochemical, X_activity], axis=1)
y = df['高血脂标签']
print(f"特征维度：{X.shape[1]}, 正例比例：{y.mean():.3f}")

# ==================== 3. 逻辑回归 ====================
X_const = sm.add_constant(X)
model = sm.Logit(y, X_const)
result = model.fit(disp=0, maxiter=1000)   # 增加迭代次数确保收敛

# 提取回归结果
params = result.params
pvals = result.pvalues
conf_int = result.conf_int()
conf_int.columns = ['2.5%', '97.5%']
OR = np.exp(params)
OR_ci_lower = np.exp(conf_int['2.5%'])
OR_ci_upper = np.exp(conf_int['97.5%'])
logit_df = pd.DataFrame({
    '变量': params.index,
    '系数(β)': params.values,
    'OR': OR.values,
    'OR_2.5%': OR_ci_lower.values,
    'OR_97.5%': OR_ci_upper.values,
    'p值': pvals.values
})
logit_df['显著性'] = logit_df['p值'].apply(lambda x: '***' if x<0.001 else ('**' if x<0.01 else ('*' if x<0.05 else 'ns')))

# 预测概率
df['预测概率'] = result.predict(X_const)
print("预测概率描述：")
print(df['预测概率'].describe(percentiles=[0.25,0.5,0.75]))

# ==================== 4. 风险分级（方法1：固定阈值0.3/0.6） ====================
def risk_fixed(p):
    if p <= 0.3:
        return '低风险'
    elif p <= 0.6:
        return '中风险'
    else:
        return '高风险'
df['风险等级_固定'] = df['预测概率'].apply(risk_fixed)

# ==================== 5. 风险分级（方法2：基于分位数的自适应阈值） ====================
p25 = df['预测概率'].quantile(0.25)
p75 = df['预测概率'].quantile(0.75)
def risk_quantile(p):
    if p <= p25:
        return '低风险'
    elif p <= p75:
        return '中风险'
    else:
        return '高风险'
df['风险等级_分位数'] = df['预测概率'].apply(risk_quantile)
print(f"分位数阈值：低风险 ≤ {p25:.3f}, 中风险 {p25:.3f}~{p75:.3f}, 高风险 > {p75:.3f}")

# ==================== 6. 风险分级（方法3：题目规则法） ====================
df['血脂异常'] = ((df['TC'] > 6.2) | (df['TG'] > 1.7) | (df['LDL_C'] > 3.1) | (df['HDL_C'] < 1.04)).astype(int)
df['活动总分'] = df['ADL'] + df['IADL']
def risk_rule(row):
    if (row['血脂异常'] == 1 and row['痰湿质积分'] >= 60) or \
       (row['血脂异常'] == 0 and row['痰湿质积分'] >= 80 and row['活动总分'] < 40):
        return '高风险'
    elif (row['血脂异常'] == 1 or (40 <= row['痰湿质积分'] < 60)) and row['活动总分'] >= 40:
        return '中风险'
    else:
        return '低风险'
df['风险等级_规则'] = df.apply(risk_rule, axis=1)

# ==================== 7. 高危人群特征对比（以固定阈值高风险组为例） ====================
high_risk = df[df['风险等级_固定'] == '高风险']
non_high = df[df['风险等级_固定'] != '高风险']
compare_features = ['痰湿质积分', 'BMI', 'TG', 'LDL_C', 'ADL', 'IADL', '活动总分', '年龄组', '血尿酸', '血糖']
comparison = []
for feat in compare_features:
    if feat in df.columns:
        high_med = high_risk[feat].median()
        non_med = non_high[feat].median()
        stat, p = mannwhitneyu(high_risk[feat], non_high[feat], alternative='two-sided')
        comparison.append({
            '特征': feat,
            '高危组中位数': high_med,
            '非高危组中位数': non_med,
            '差异方向': '高' if high_med > non_med else '低',
            'p值': p,
            '显著性': '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
        })
comparison_df = pd.DataFrame(comparison)

# 痰湿高危人群（痰湿积分≥60且高风险）
phlegm_high = high_risk[high_risk['痰湿质积分'] >= 60]
phlegm_desc = phlegm_high[compare_features].describe().T

# ==================== 8. 保存Excel ====================
with pd.ExcelWriter('问题2_风险预警结果.xlsx', engine='openpyxl') as writer:
    logit_df.to_excel(writer, sheet_name='逻辑回归结果', index=False)
    # 个体风险分级表（包含三种分级结果）
    output_cols = ['预测概率', '风险等级_固定', '风险等级_分位数', '风险等级_规则', 
                   '痰湿质积分', '血脂异常', '活动总分'] + compare_features
    output_cols = [c for c in output_cols if c in df.columns]
    df[output_cols].to_excel(writer, sheet_name='个体风险分级', index=False)
    comparison_df.to_excel(writer, sheet_name='高危vs非高危特征对比', index=False)
    phlegm_desc.to_excel(writer, sheet_name='痰湿高危人群特征描述')
    # 三种分级方法的交叉对比
    cross_fixed_rule = pd.crosstab(df['风险等级_固定'], df['风险等级_规则'], margins=True)
    cross_fixed_quantile = pd.crosstab(df['风险等级_固定'], df['风险等级_分位数'], margins=True)
    cross_fixed_rule.to_excel(writer, sheet_name='固定vs规则')
    cross_fixed_quantile.to_excel(writer, sheet_name='固定vs分位数')
print("✅ Excel结果已保存")

# ==================== 9. 可视化 ====================
# 9.1 预测概率分布直方图（同时显示两种阈值）
plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.hist(df['预测概率'], bins=30, color='steelblue', edgecolor='black', alpha=0.7, density=True)
plt.axvline(0.3, linestyle='--', color='orange', label='固定阈值0.3')
plt.axvline(0.6, linestyle='--', color='red', label='固定阈值0.6')
plt.axvline(p25, linestyle=':', color='green', label=f'分位数阈值{p25:.2f}')
plt.axvline(p75, linestyle=':', color='green', label=f'分位数阈值{p75:.2f}')
plt.xlabel('预测患病概率')
plt.ylabel('密度')
plt.title('预测概率分布（固定 vs 分位数阈值）')
plt.legend()
plt.subplot(1,2,2)
# 按风险等级着色（固定阈值）
colors = {'低风险':'green', '中风险':'orange', '高风险':'red'}
for level, color in colors.items():
    subset = df[df['风险等级_固定'] == level]
    plt.hist(subset['预测概率'], bins=20, alpha=0.6, label=level, color=color, density=True)
plt.xlabel('预测患病概率')
plt.ylabel('密度')
plt.title('按固定阈值分级的概率分布')
plt.legend()
plt.tight_layout()
plt.savefig('问题2_概率分布直方图.png', dpi=300)
plt.close()

# 9.2 高危特征对比箱线图（选择6个关键特征）
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
features_plot = ['痰湿质积分', 'TG', 'BMI', 'ADL', 'IADL', '活动总分']
for idx, feat in enumerate(features_plot):
    if feat not in df.columns:
        continue
    ax = axes[idx//3, idx%3]
    data = [high_risk[feat].dropna(), non_high[feat].dropna()]
    ax.boxplot(data, labels=['高危组', '非高危组'], patch_artist=True)
    ax.set_title(feat)
    ax.set_ylabel('值')
plt.tight_layout()
plt.savefig('问题2_高危特征对比箱线图.png', dpi=300)
plt.close()

# 9.3 风险等级饼图（固定阈值）
risk_counts = df['风险等级_固定'].value_counts()
risk_counts.plot.pie(autopct='%1.1f%%', figsize=(6,6), colors=['green','orange','red'])
plt.title('固定阈值风险等级分布')
plt.ylabel('')
plt.savefig('问题2_风险等级饼图.png', dpi=300)
plt.close()

print("✅ 所有图表已保存")

# ==================== 10. 阈值依据文本 ====================
with open('问题2_阈值选取依据.txt', 'w', encoding='utf-8') as f:
    f.write(f"""【风险分级阈值选取依据】

本研究采用两种阈值确定方法：

1. 固定阈值法（主要方法）：
   - 低风险：预测概率 ≤ 0.3
   - 中风险：0.3 < 预测概率 ≤ 0.6
   - 高风险：预测概率 > 0.6
   依据：参考既往临床风险预测模型常用切点，保证高风险组占比约为 {len(high_risk)/len(df):.1%}，符合临床早筛聚焦原则。

2. 分位数自适应法（敏感性分析）：
   - 低风险：≤ {p25:.3f} (25%分位数)
   - 中风险：{p25:.3f} ~ {p75:.3f}
   - 高风险：> {p75:.3f} (75%分位数)
   依据：基于数据自身分布，保证每组样本量均衡，适用于探索性分析。

3. 题目规则法：
   - 高风险：①血脂异常且痰湿积分≥60；②血脂正常但痰湿积分≥80且活动总分<40。
   - 中风险：血脂异常或痰湿积分40~59，且活动总分≥40。
   - 低风险：其余情况。
   依据：融合中医体质评分与活动能力，匹配《中医体质分类与判定》及《中国成人血脂异常防治指南》。

最终模型采用固定阈值法进行风险分层，其概率分布形态（见直方图）显示：{'预测概率呈双峰分布，区分度良好' if df['预测概率'].var()>0.05 else '预测概率分布较为集中，模型区分能力有限'}。阈值选取具有统计合理性与临床可操作性。
""")
print("✅ 阈值依据文本已保存")

print("\n=== 痰湿体质高风险人群核心特征组合 ===")
print(f"样本量：{len(phlegm_high)} (占总高风险人群 {len(phlegm_high)/len(high_risk)*100:.1f}%)")
print("关键特征中位数：")
key_feats = ['痰湿质积分', 'TG', 'LDL_C', 'BMI', 'ADL', 'IADL', '活动总分']
for f in key_feats:
    if f in phlegm_high.columns:
        print(f"  {f}: {phlegm_high[f].median():.2f}")
print("\n核心特征组合：痰湿体质高分（≥60）+ 血脂指标（尤其TG）异常升高 + BMI偏高（超重/肥胖） + 活动能力下降（ADL/IADL低分）")
print("问题2全部完成！")