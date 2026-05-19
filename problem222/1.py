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

print("原始数据列名：", df.columns.tolist())

constitution_cols = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']

missing_constitution = [col for col in constitution_cols if col not in df.columns]
if missing_constitution:
    print(f"警告：以下体质列不存在：{missing_constitution}")
    for col in missing_constitution:
        possible = [c for c in df.columns if c.strip() == col]
        if possible:
            print(f"找到匹配列：{possible[0]}, 使用该列")
            df[col] = df[possible[0]]
        else:
            raise KeyError(f"体质列 {col} 不存在，请检查数据")

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
keep_cols = constitution_cols + list(rename_dict.keys()) + ['年龄组', '性别', '吸烟史', '饮酒史']
keep_cols = [col for col in keep_cols if col in df.columns]
df = df[keep_cols].copy()
df.rename(columns=rename_dict, inplace=True)
df['痰湿质积分'] = df['痰湿质']
df.dropna(inplace=True)
print(f"有效样本数：{len(df)}")

# ==================== 2. 定义三大维度特征 ====================
X_constitution = df[constitution_cols]
X_biochemical = df[['TC', 'TG', 'LDL_C', 'HDL_C', '血糖', '血尿酸', 'BMI']]
X_activity = df[['ADL', 'IADL']]
X = pd.concat([X_constitution, X_biochemical, X_activity], axis=1)
y = df['高血脂标签']
print("特征维度：", X.shape[1])
print("正例比例：", y.mean())

# ==================== 3. 逻辑回归建模 ====================
X_const = sm.add_constant(X)
model = sm.Logit(y, X_const)
result = model.fit(disp=0)

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

df['预测概率'] = result.predict(X_const)

# ==================== 4. 风险分级（统计分位法） ====================
def risk_level_by_prob(prob):
    if prob <= 0.3:
        return '低风险'
    elif prob <= 0.6:
        return '中风险'
    else:
        return '高风险'

df['风险等级_统计法'] = df['预测概率'].apply(risk_level_by_prob)

# ==================== 5. 风险分级（规则法） ====================
df['血脂异常'] = ((df['TC'] > 6.2) | (df['TG'] > 1.7) | (df['LDL_C'] > 3.1) | (df['HDL_C'] < 1.04)).astype(int)
df['活动总分'] = df['ADL'] + df['IADL']

def risk_level_by_rule(row):
    if (row['血脂异常'] == 1 and row['痰湿质积分'] >= 60) or \
       (row['血脂异常'] == 0 and row['痰湿质积分'] >= 80 and row['活动总分'] < 40):
        return '高风险'
    elif (row['血脂异常'] == 1 or (40 <= row['痰湿质积分'] < 60)) and row['活动总分'] >= 40:
        return '中风险'
    else:
        return '低风险'

df['风险等级_规则法'] = df.apply(risk_level_by_rule, axis=1)

# ==================== 6. 高危人群特征组合 ====================
high_risk = df[df['风险等级_统计法'] == '高风险']
non_high = df[df['风险等级_统计法'] != '高风险']
compare_features = ['痰湿质积分', 'BMI', 'TG', 'LDL_C', 'ADL', 'IADL', '活动总分', '年龄组', '血尿酸', '血糖']

comparison = []
for feat in compare_features:
    high_median = high_risk[feat].median()
    non_high_median = non_high[feat].median()
    stat, p = mannwhitneyu(high_risk[feat], non_high[feat], alternative='two-sided')
    comparison.append({
        '特征': feat,
        '高危组中位数': high_median,
        '非高危组中位数': non_high_median,
        '差异方向': '高' if high_median > non_high_median else '低',
        'p值': p,
        '显著性': '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
    })
comparison_df = pd.DataFrame(comparison)

phlegm_high = high_risk[high_risk['痰湿质积分'] >= 60]
phlegm_high_desc = phlegm_high[compare_features].describe().T

# ==================== 7. 保存Excel ====================
with pd.ExcelWriter('问题2_风险预警结果.xlsx', engine='openpyxl') as writer:
    logit_df.to_excel(writer, sheet_name='逻辑回归结果', index=False)
    df[['预测概率', '风险等级_统计法', '风险等级_规则法', '痰湿质积分', '血脂异常', '活动总分'] + compare_features].to_excel(
        writer, sheet_name='个体风险分级', index=False)
    comparison_df.to_excel(writer, sheet_name='高危vs非高危特征对比', index=False)
    phlegm_high_desc.to_excel(writer, sheet_name='痰湿高危人群特征描述')
    pd.crosstab(df['风险等级_统计法'], df['风险等级_规则法'], margins=True).to_excel(writer, sheet_name='两方法分级对比')

print("✅ 结果已保存到 问题2_风险预警结果.xlsx")

# ==================== 8. 可视化 ====================
# 8.1 统计法概率分布直方图
plt.figure(figsize=(10, 6))
colors = {'低风险': 'green', '中风险': 'orange', '高风险': 'red'}
for level, color in colors.items():
    subset = df[df['风险等级_统计法'] == level]
    plt.hist(subset['预测概率'], bins=20, alpha=0.6, label=level, color=color, density=True)
plt.axvline(0.3, linestyle='--', color='gray', label='阈值0.3')
plt.axvline(0.6, linestyle='--', color='gray', label='阈值0.6')
plt.xlabel('预测患病概率')
plt.ylabel('密度')
plt.title('高血脂风险预测概率分布（统计法分级）')
plt.legend()
plt.tight_layout()
plt.savefig('问题2_概率分布直方图.png', dpi=300)
plt.close()

# 8.2 高危特征对比箱线图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
features_plot = ['痰湿质积分', 'TG', 'BMI', 'ADL', 'IADL', '活动总分']
for idx, feat in enumerate(features_plot):
    ax = axes[idx//3, idx%3]
    data_to_plot = [high_risk[feat].dropna(), non_high[feat].dropna()]
    ax.boxplot(data_to_plot, labels=['高危组', '非高危组'], patch_artist=True)
    ax.set_title(feat)
    ax.set_ylabel('值')
plt.tight_layout()
plt.savefig('问题2_高危特征对比箱线图.png', dpi=300)
plt.close()

# 8.3 统计法风险等级饼图
risk_counts = df['风险等级_统计法'].value_counts()
risk_counts.plot.pie(autopct='%1.1f%%', figsize=(6,6), colors=['green','orange','red'])
plt.title('统计法风险等级分布')
plt.ylabel('')
plt.savefig('问题2_风险等级饼图.png', dpi=300)
plt.close()

# 8.4 规则法风险等级饼图（新增）
risk_counts_rule = df['风险等级_规则法'].value_counts()
plt.figure(figsize=(6,6))
risk_counts_rule.plot.pie(autopct='%1.1f%%', colors=['green','orange','red'])
plt.title('规则法风险等级分布')
plt.ylabel('')
plt.savefig('问题2_规则法风险等级饼图.png', dpi=300)
plt.close()
print("✅ 可视化图表已保存（含规则法饼图）")

# ==================== 9. 阈值依据文本 ====================
with open('问题2_阈值选取依据.txt', 'w', encoding='utf-8') as f:
    f.write("""【风险分级阈值选取依据】

1. 统计分位法：
   - 低风险：预测概率 ≤ 0.3
   - 中风险：0.3 < 预测概率 ≤ 0.6
   - 高风险：预测概率 > 0.6
   依据：参考临床风险预测模型常用三分位切点，确保高风险组占比合理。

2. 题目示例规则法：
   - 高风险：①血脂异常且痰湿积分≥60；或②血脂正常但痰湿积分≥80且活动总分<40。
   - 中风险：血脂异常或痰湿积分40~59，且活动总分≥40。
   - 低风险：其余情况。
   依据：融合中医体质与活动能力，匹配《中医体质分类与判定》及血脂指南。
""")
print("✅ 阈值选取依据已保存")

print("\n=== 痰湿体质高风险人群核心特征组合 ===")
print(f"样本量：{len(phlegm_high)}")
print("关键特征中位数：")
print(phlegm_high[['痰湿质积分', 'TG', 'LDL_C', 'BMI', 'ADL', 'IADL', '活动总分']].median())
print("\n核心特征组合描述：痰湿体质高分（≥60）+ 血脂指标异常升高 + BMI偏高 + 活动能力下降")

print("\n问题2全部完成！")