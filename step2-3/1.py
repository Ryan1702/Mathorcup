import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
import os
import matplotlib.font_manager as fm
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

# 列名映射
rename_dict = {
    '痰湿质': '痰湿质积分',
    'ADL总分': 'ADL',
    'IADL总分': 'IADL',
    '活动量表总分（ADL总分+IADL总分）': '活动总分',  # 保留原列但不使用
    'TC（总胆固醇）': 'TC',
    'TG（甘油三酯）': 'TG',
    'LDL-C（低密度脂蛋白）': 'LDL_C',
    'HDL-C（高密度脂蛋白）': 'HDL_C',
    '空腹血糖': '血糖',
    '血尿酸': '血尿酸',
    'BMI': 'BMI',
    '高血脂症二分类标签': '高血脂标签'
}
keep_cols = list(rename_dict.keys()) + ['体质标签']
df = df[keep_cols].copy()
df.rename(columns=rename_dict, inplace=True)
df.dropna(inplace=True)
print(f"有效样本数：{len(df)}")

# ==================== 2. 定义候选自变量（策略二：只保留 ADL 和 IADL，删除活动总分） ====================
candidate_features = [
    'TC', 'TG', 'LDL_C', 'HDL_C', '血糖', '血尿酸', 'BMI',
    'ADL', 'IADL',           # 只保留 ADL 和 IADL
    '痰湿质积分'
]
X = df[candidate_features]
y = df['高血脂标签']

print("自变量列表：", candidate_features)

# ==================== 3. 二元逻辑回归 ====================
X_const = sm.add_constant(X)
model = sm.Logit(y, X_const)
result = model.fit(disp=0)

# 提取结果
params = result.params
pvalues = result.pvalues
conf_int = result.conf_int()
conf_int.columns = ['2.5%', '97.5%']

# 计算 OR 及其置信区间
OR = np.exp(params)
OR_CI_lower = np.exp(conf_int['2.5%'])
OR_CI_upper = np.exp(conf_int['97.5%'])

# 构建结果 DataFrame
logit_results = pd.DataFrame({
    '变量': params.index,
    '系数(β)': params.values,
    'OR (exp(β))': OR.values,
    'OR_2.5%': OR_CI_lower.values,
    'OR_97.5%': OR_CI_upper.values,
    'p值': pvalues.values
})
logit_results['显著性'] = logit_results['p值'].apply(
    lambda x: '***' if x < 0.001 else ('**' if x < 0.01 else ('*' if x < 0.05 else 'ns'))
)

# 筛选 p<0.05 的变量（不包括常数项）
sig_results = logit_results[(logit_results['变量'] != 'const') & (logit_results['p值'] < 0.05)].copy()

print("\n显著变量（p<0.05）：")
print(sig_results[['变量', 'OR (exp(β))', 'p值', '显著性']])

# ==================== 4. 保存结果到 Excel ====================
with pd.ExcelWriter('step2_logistic_results.xlsx', engine='openpyxl') as writer:
    logit_results.to_excel(writer, sheet_name='完整逻辑回归结果', index=False)
    sig_results.to_excel(writer, sheet_name='显著变量(p<0.05)', index=False)

# 保存模型摘要文本
with open('step2_model_summary.txt', 'w', encoding='utf-8') as f:
    f.write(result.summary().as_text())

print("✅ 结果已保存到 step2_logistic_results.xlsx 和 step2_model_summary.txt")

# ==================== 5. 森林图 ====================
plot_df = sig_results if len(sig_results) > 0 else logit_results[logit_results['变量'] != 'const']
plot_df = plot_df.sort_values('OR (exp(β))', ascending=True).copy()
plot_df.reset_index(drop=True, inplace=True)

if len(plot_df) > 0:
    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df)*0.5)))
    y_pos = np.arange(len(plot_df))
    x_vals = plot_df['OR (exp(β))'].values
    ci_low = plot_df['OR_2.5%'].values
    ci_high = plot_df['OR_97.5%'].values
    
    # 根据显著性标记颜色（显著红色，不显著灰色，这里plot_df已筛选显著，故全红）
    ax.errorbar(x_vals, y_pos,
                xerr=[x_vals - ci_low, ci_high - x_vals],
                fmt='o', capsize=5, color='red', ecolor='gray', elinewidth=2, markersize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['变量'])
    ax.axvline(x=1, linestyle='--', color='black', linewidth=1, label='OR=1 (无效应)')
    ax.set_xlabel('优势比 (OR) 及 95% 置信区间', fontsize=12)
    ax.set_title('核心预警指标逻辑回归森林图', fontsize=14)
    ax.set_xscale('log')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.savefig('step2_forest_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 森林图已保存：step2_forest_plot.png")
else:
    print("⚠️ 没有显著变量，跳过森林图绘制")

# ==================== 6. 多重共线性检验（VIF） ====================
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data['变量'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

vif_df = calculate_vif(X)
vif_df.to_excel('step2_vif.xlsx', index=False)
print("✅ 多重共线性检验结果（VIF）已保存到 step2_vif.xlsx")

print("第二步分析完成！")