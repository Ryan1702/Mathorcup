import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
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

# 九种体质列名（原数据中的列名）
constitution_cols = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']
# 混杂因素列名
confounders = ['年龄组', '性别', '吸烟史', '饮酒史', 'BMI']
# 因变量
target = '高血脂症二分类标签'

# 检查列是否存在
available_constitution = [col for col in constitution_cols if col in df.columns]
available_confounders = [col for col in confounders if col in df.columns]
print("可用体质列：", available_constitution)
print("可用混杂因素：", available_confounders)

# 剔除缺失值（体质得分和混杂因素及因变量）
df_clean = df[available_constitution + available_confounders + [target]].dropna()
print(f"有效样本数：{len(df_clean)}")

# 将年龄组、性别、吸烟史、饮酒史转为数值（已为数值，但确认类型）
for col in available_confounders:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
df_clean[target] = pd.to_numeric(df_clean[target], errors='coerce')
df_clean.dropna(inplace=True)

# ==================== 2. 单因素 logistic 回归 ====================
univariate_results = []
for col in available_constitution:
    X = sm.add_constant(df_clean[col])
    y = df_clean[target]
    model = sm.Logit(y, X)
    result = model.fit(disp=0)
    params = result.params
    pvals = result.pvalues
    conf_int = result.conf_int()
    or_val = np.exp(params[col])
    ci_lower = np.exp(conf_int.loc[col, 0])
    ci_upper = np.exp(conf_int.loc[col, 1])
    p_val = pvals[col]
    univariate_results.append({
        '体质类型': col,
        'OR (单因素)': or_val,
        'OR_2.5%': ci_lower,
        'OR_97.5%': ci_upper,
        'p值': p_val,
        '显著性': '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
    })
univariate_df = pd.DataFrame(univariate_results)
univariate_df = univariate_df.sort_values('OR (单因素)', ascending=False)

# ==================== 3. 多因素 logistic 回归（校正混杂） ====================
X_multi = df_clean[available_constitution + available_confounders]
X_multi = sm.add_constant(X_multi)
y_multi = df_clean[target]
model_multi = sm.Logit(y_multi, X_multi)
result_multi = model_multi.fit(disp=0)

multi_results = []
for col in available_constitution:
    params = result_multi.params
    pvals = result_multi.pvalues
    conf_int = result_multi.conf_int()
    or_val = np.exp(params[col])
    ci_lower = np.exp(conf_int.loc[col, 0])
    ci_upper = np.exp(conf_int.loc[col, 1])
    p_val = pvals[col]
    multi_results.append({
        '体质类型': col,
        'OR (多因素校正)': or_val,
        'OR_2.5%': ci_lower,
        'OR_97.5%': ci_upper,
        'p值': p_val,
        '显著性': '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
    })
multi_df = pd.DataFrame(multi_results)
multi_df = multi_df.sort_values('OR (多因素校正)', ascending=False)

# 合并单因素和多因素结果
combined_df = pd.merge(univariate_df, multi_df, on='体质类型', suffixes=('_单因素', '_多因素'))
combined_df = combined_df[['体质类型', 'OR (单因素)', 'OR_2.5%_单因素', 'OR_97.5%_单因素', 'p值_单因素', '显著性_单因素',
                           'OR (多因素校正)', 'OR_2.5%_多因素', 'OR_97.5%_多因素', 'p值_多因素', '显著性_多因素']]

# ==================== 4. 风险贡献度排序 ====================
multi_df['排序权重'] = multi_df['显著性'].map({'***': 4, '**': 3, '*': 2, 'ns': 1})
ranking_df = multi_df.sort_values(['排序权重', 'OR (多因素校正)'], ascending=[False, False])
ranking_df['风险等级'] = ranking_df['OR (多因素校正)'].apply(
    lambda x: '高风险' if x > 2 else ('中风险' if x > 1.5 else ('低风险' if x > 1 else '保护因素'))
)
ranking_result = ranking_df[['体质类型', 'OR (多因素校正)', 'OR_2.5%', 'OR_97.5%', 'p值', '显著性', '风险等级']]

# ==================== 5. 保存结果到 Excel ====================
with pd.ExcelWriter('step3_constitution_risk.xlsx', engine='openpyxl') as writer:
    univariate_df.to_excel(writer, sheet_name='单因素Logistic', index=False)
    multi_df.to_excel(writer, sheet_name='多因素Logistic(校正)', index=False)
    combined_df.to_excel(writer, sheet_name='单因素+多因素对比', index=False)
    ranking_result.to_excel(writer, sheet_name='风险贡献度排序', index=False)

print("✅ 体质风险分析结果已保存到 step3_constitution_risk.xlsx")

# ==================== 6. 森林图（修正颜色错误） ====================
plot_df = multi_df.sort_values('OR (多因素校正)', ascending=True).copy()
plot_df.reset_index(drop=True, inplace=True)

fig, ax = plt.subplots(figsize=(10, 8))
y_pos = np.arange(len(plot_df))

# 分别绘制显著和非显著的点
sig_mask = plot_df['p值'] < 0.05
non_sig_mask = ~sig_mask

if sig_mask.any():
    x_sig = plot_df.loc[sig_mask, 'OR (多因素校正)'].values
    y_sig = y_pos[sig_mask]
    ci_lower_sig = plot_df.loc[sig_mask, 'OR_2.5%'].values
    ci_upper_sig = plot_df.loc[sig_mask, 'OR_97.5%'].values
    ax.errorbar(x_sig, y_sig, 
                xerr=[x_sig - ci_lower_sig, ci_upper_sig - x_sig],
                fmt='o', capsize=5, color='red', ecolor='gray', elinewidth=2, markersize=8, label='p<0.05')

if non_sig_mask.any():
    x_nonsig = plot_df.loc[non_sig_mask, 'OR (多因素校正)'].values
    y_nonsig = y_pos[non_sig_mask]
    ci_lower_nonsig = plot_df.loc[non_sig_mask, 'OR_2.5%'].values
    ci_upper_nonsig = plot_df.loc[non_sig_mask, 'OR_97.5%'].values
    ax.errorbar(x_nonsig, y_nonsig,
                xerr=[x_nonsig - ci_lower_nonsig, ci_upper_nonsig - x_nonsig],
                fmt='o', capsize=5, color='gray', ecolor='lightgray', elinewidth=2, markersize=8, label='p≥0.05')

ax.axvline(x=1, linestyle='--', color='black', linewidth=1, label='OR=1 (无效应)')
ax.set_yticks(y_pos)
ax.set_yticklabels(plot_df['体质类型'])
ax.set_xlabel('优势比 (OR) 及 95% 置信区间', fontsize=12)
ax.set_title('九种体质多因素校正 Logistic 回归森林图', fontsize=14)
ax.set_xscale('log')
ax.grid(axis='x', linestyle='--', alpha=0.7)
ax.legend()
plt.tight_layout()
plt.savefig('step3_forest_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ 森林图已保存：step3_forest_plot.png")

# ==================== 7. 多因素模型摘要保存 ====================
with open('step3_multivariate_summary.txt', 'w', encoding='utf-8') as f:
    f.write(result_multi.summary().as_text())
print("✅ 多因素模型摘要已保存到 step3_multivariate_summary.txt")

print("第三步分析完成！")