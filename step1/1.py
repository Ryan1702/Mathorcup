import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, spearmanr, mannwhitneyu
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

# ==================== 1. 数据加载 ====================
file_path = '附件1：样例数据.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

print("数据列名预览：", df.columns.tolist()[:10], "...")

# 定义关键指标映射（原列名 -> 简化名）
rename_dict = {
    '痰湿质': '痰湿质积分',
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
# 只保留需要的列
keep_cols = list(rename_dict.keys()) + ['体质标签']
df = df[keep_cols].copy()
df.rename(columns=rename_dict, inplace=True)

# 剔除缺失值
df.dropna(inplace=True)
print(f"有效样本数：{len(df)}")

# ==================== 2. 痰湿严重程度分组 ====================
def classify_phlegm(score):
    if score < 20:
        return '无'
    elif score < 40:
        return '轻度'
    elif score < 60:
        return '中度'
    else:
        return '重度'

df['痰湿严重程度'] = df['痰湿质积分'].apply(classify_phlegm)
severity_order = ['无', '轻度', '中度', '重度']
df['痰湿严重程度'] = pd.Categorical(df['痰湿严重程度'], categories=severity_order, ordered=True)

# 待分析的连续指标（不包括体质标签和高血脂标签）
analysis_metrics = ['TC', 'TG', 'LDL_C', 'HDL_C', '血糖', '血尿酸', 'BMI', 'ADL', 'IADL', '活动总分']

# ==================== 3. 分组描述统计（中位数） ====================
group_median = df.groupby('痰湿严重程度')[analysis_metrics].median().T
group_median.columns = [f'中位数_{col}' for col in group_median.columns]

# ==================== 4. 非参数检验 ====================
# 4.1 Kruskal-Wallis 检验（多组差异）
kruskal_results = []
for metric in analysis_metrics:
    groups = [df[df['痰湿严重程度'] == level][metric].dropna().values 
              for level in severity_order if len(df[df['痰湿严重程度'] == level]) > 0]
    if len(groups) >= 2:
        stat, p = kruskal(*groups)
        kruskal_results.append({'指标': metric, 'H统计量': stat, 'p值': p})
kruskal_df = pd.DataFrame(kruskal_results)
kruskal_df['显著性'] = kruskal_df['p值'].apply(lambda x: '***' if x<0.001 else ('**' if x<0.01 else ('*' if x<0.05 else 'ns')))

# 4.2 两两 Mann-Whitney U 检验（仅对显著指标）
pairwise_results = []
sig_metrics = kruskal_df[kruskal_df['p值'] < 0.05]['指标'].tolist()
for metric in sig_metrics:
    for i in range(len(severity_order)):
        for j in range(i+1, len(severity_order)):
            g1 = df[df['痰湿严重程度'] == severity_order[i]][metric].dropna()
            g2 = df[df['痰湿严重程度'] == severity_order[j]][metric].dropna()
            if len(g1) > 0 and len(g2) > 0:
                stat, p = mannwhitneyu(g1, g2, alternative='two-sided')
                pairwise_results.append({
                    '指标': metric, '组1': severity_order[i], '组2': severity_order[j],
                    'U统计量': stat, 'p值': p
                })
pairwise_df = pd.DataFrame(pairwise_results)

# ==================== 5. Spearman 相关性分析 ====================
# 5.1 痰湿质积分与各指标的相关性
spearman_phlegm = []
for metric in analysis_metrics:
    corr, p = spearmanr(df['痰湿质积分'], df[metric])
    spearman_phlegm.append({'指标': metric, '与痰湿质积分的Spearman_r': corr, 'p值': p})
spearman_phlegm_df = pd.DataFrame(spearman_phlegm)
spearman_phlegm_df['显著性'] = spearman_phlegm_df['p值'].apply(lambda x: '***' if x<0.001 else ('**' if x<0.01 else ('*' if x<0.05 else 'ns')))

# 5.2 各指标与高血脂标签的相关性
spearman_hyper = []
for metric in analysis_metrics + ['痰湿质积分']:
    corr, p = spearmanr(df['高血脂标签'], df[metric])
    spearman_hyper.append({'指标': metric, '与高血脂标签的Spearman_r': corr, 'p值': p})
spearman_hyper_df = pd.DataFrame(spearman_hyper)
spearman_hyper_df['显著性'] = spearman_hyper_df['p值'].apply(lambda x: '***' if x<0.001 else ('**' if x<0.01 else ('*' if x<0.05 else 'ns')))

# 完整相关矩阵
all_corr_metrics = analysis_metrics + ['痰湿质积分', '高血脂标签']
corr_matrix = df[all_corr_metrics].corr(method='spearman')

# ==================== 6. 保存所有表格到 Excel ====================
with pd.ExcelWriter('step1_results.xlsx', engine='openpyxl') as writer:
    group_median.to_excel(writer, sheet_name='分组中位数')
    kruskal_df.to_excel(writer, sheet_name='Kruskal-Wallis检验', index=False)
    pairwise_df.to_excel(writer, sheet_name='两两比较(Mann-Whitney)', index=False)
    spearman_phlegm_df.to_excel(writer, sheet_name='Spearman_痰湿质与各指标', index=False)
    spearman_hyper_df.to_excel(writer, sheet_name='Spearman_各指标与高血脂', index=False)
    corr_matrix.to_excel(writer, sheet_name='Spearman相关矩阵')

print("✅ 表格结果已保存到 step1_results.xlsx")

# ==================== 7. 可视化 ====================
# 7.1 箱线图：各指标随痰湿严重程度的变化
fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.flatten()
for idx, metric in enumerate(analysis_metrics):
    ax = axes[idx]
    data_to_plot = [df[df['痰湿严重程度'] == level][metric].dropna() 
                    for level in severity_order]
    ax.boxplot(data_to_plot, labels=severity_order, patch_artist=True, showmeans=False)
    ax.set_title(metric, fontsize=12)
    ax.set_ylabel('值')
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('step1_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()

# 7.2 热力图：Spearman相关矩阵
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, fmt='.2f')
plt.title('Spearman 相关矩阵', fontsize=16)
plt.tight_layout()
plt.savefig('step1_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ 图表已保存：step1_boxplots.png 和 step1_heatmap.png")
print("第一步分析完成！")