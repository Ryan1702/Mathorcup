import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

# ==================== 中文字体设置 ====================
def set_chinese_font():
    font_candidates = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 
                       'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Arial Unicode MS', 'PingFang SC']
    available = [f.name for f in fm.fontManager.ttflist]
    for font in font_candidates:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return font
    return None

chinese_font = set_chinese_font()
if chinese_font is None:
    print("警告：未找到中文字体，图表中文将显示为方块。")

# 读取数据
df = pd.read_csv('清洗后_样例数据.csv', encoding='utf-8-sig')
print(f"数据维度: {df.shape}")

candidate_cols = [
    'HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）',
    'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI',
    'ADL总分', 'IADL总分', '活动量表总分（ADL总分+IADL总分）'
]

# 目标变量
sputum_score = df['痰湿质']
sputum_label = (df['体质标签'] == 5).astype(int)
hyper_label = df['高血脂症二分类标签']

# ==================== 1. 与痰湿质积分的Spearman相关 ====================
cont_results = []
for col in candidate_cols:
    corr, p = stats.spearmanr(df[col], sputum_score)
    cont_results.append((col, corr, p))
cont_df = pd.DataFrame(cont_results, columns=['指标', 'Spearman相关系数', 'p值(连续)'])
cont_df['排名_连续'] = cont_df['p值(连续)'].rank()

# ==================== 2. 痰湿分组差异 ====================
sputum_group = df[sputum_label == 1]
non_group = df[sputum_label == 0]
group_results = []
for col in candidate_cols:
    _, p = stats.mannwhitneyu(sputum_group[col], non_group[col], alternative='two-sided')
    mean_diff = sputum_group[col].mean() - non_group[col].mean()
    group_results.append((col, mean_diff, p))
group_df = pd.DataFrame(group_results, columns=['指标', '均值差(痰湿-非痰湿)', 'p值(分组)'])
group_df['排名_分组'] = group_df['p值(分组)'].rank()

# ==================== 3. 高血脂分组差异 ====================
hyper_group = df[hyper_label == 1]
non_hyper = df[hyper_label == 0]
hyper_results = []
for col in candidate_cols:
    _, p = stats.mannwhitneyu(hyper_group[col], non_hyper[col], alternative='two-sided')
    mean_diff = hyper_group[col].mean() - non_hyper[col].mean()
    hyper_results.append((col, mean_diff, p))
hyper_df = pd.DataFrame(hyper_results, columns=['指标', '均值差(高血脂-非高血脂)', 'p值(高血脂)'])
hyper_df['排名_高血脂'] = hyper_df['p值(高血脂)'].rank()

# ==================== 4. 综合排名 ====================
merged = cont_df[['指标', '排名_连续']].merge(group_df[['指标', '排名_分组']], on='指标')
merged = merged.merge(hyper_df[['指标', '排名_高血脂']], on='指标')
merged['痰湿综合排名'] = merged['排名_连续'] + merged['排名_分组']
merged['总排名'] = merged['痰湿综合排名'] + merged['排名_高血脂']
merged = merged.sort_values('总排名')

key_indicators = merged.head(3)['指标'].tolist()
print(f"关键指标: {key_indicators}")

# ==================== 5. 生成文本报告 ====================
with open("问题1_关键指标筛选报告_综合排名法.txt", "w", encoding="utf-8") as f:
    f.write("问题1子任务1：关键指标筛选报告\n")
    f.write("=" * 60 + "\n\n")
    f.write("一、方法说明\n")
    f.write("1. 分别计算每个候选指标与痰湿质积分的Spearman相关系数（p值），得到连续排名；\n")
    f.write("2. 分别计算每个候选指标在痰湿/非痰湿两组间的Mann-Whitney U检验（p值），得到分组排名；\n")
    f.write("3. 将连续排名与分组排名相加，得到痰湿综合排名；\n")
    f.write("4. 计算每个候选指标在高血脂/非高血脂两组间的Mann-Whitney U检验（p值），得到高血脂排名；\n")
    f.write("5. 将痰湿综合排名与高血脂排名相加，得到总排名，取总排名最小的3个指标作为关键指标。\n\n")
    
    f.write("二、各指标排名详情\n")
    f.write(f"{'指标':<25} {'连续排名':<8} {'分组排名':<8} {'痰湿综合':<8} {'高血脂排名':<8} {'总排名':<8}\n")
    for _, row in merged.iterrows():
        f.write(f"{row['指标']:<25} {row['排名_连续']:<8.0f} {row['排名_分组']:<8.0f} "
                f"{row['痰湿综合排名']:<8.0f} {row['排名_高血脂']:<8.0f} {row['总排名']:<8.0f}\n")
    
    f.write("\n三、最终关键指标\n")
    for idx, col in enumerate(key_indicators, 1):
        row = merged[merged['指标'] == col].iloc[0]
        f.write(f"{idx}. {col}\n")
        f.write(f"   - 与痰湿质积分Spearman相关系数：{cont_df[cont_df['指标']==col]['Spearman相关系数'].values[0]:.4f} "
                f"(p={cont_df[cont_df['指标']==col]['p值(连续)'].values[0]:.4f})\n")
        f.write(f"   - 痰湿/非痰湿分组均值差：{group_df[group_df['指标']==col]['均值差(痰湿-非痰湿)'].values[0]:.4f} "
                f"(p={group_df[group_df['指标']==col]['p值(分组)'].values[0]:.4f})\n")
        f.write(f"   - 高血脂/非高血脂分组均值差：{hyper_df[hyper_df['指标']==col]['均值差(高血脂-非高血脂)'].values[0]:.4f} "
                f"(p={hyper_df[hyper_df['指标']==col]['p值(高血脂)'].values[0]:.2e})\n\n")
    
    f.write("四、结论\n")
    f.write("基于综合排名法，筛选出的关键指标为：总胆固醇（TC）、ADL总分、甘油三酯（TG）。\n")
    f.write("其中，TC在痰湿表征和高血脂预警方面均表现优异，是最稳健的指标；\n")
    f.write("ADL总分是表征痰湿体质的最佳行为指标（痰湿积分越高，活动能力越差）；\n")
    f.write("TG是预警高血脂的最强指标，尽管与痰湿积分无显著相关，但其对高血脂的极强区分能力使其成为必要特征。\n")
    f.write("建议在后续风险预警模型中重点纳入这三个指标。\n")

print("文本报告已生成：问题1_关键指标筛选报告_综合排名法.txt")

# ==================== 6. 生成箱线图 ====================
for col in key_indicators:
    # 图1：按痰湿分组
    plt.figure(figsize=(6, 5))
    sns.boxplot(x=sputum_label, y=df[col], hue=sputum_label, palette='Set2', legend=False)
    plt.xticks([0, 1], ['非痰湿体质', '痰湿体质'])
    plt.title(f'{col} - 按痰湿体质分组')
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(f'{col}_痰湿分组箱线图.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 图2：按高血脂分组
    plt.figure(figsize=(6, 5))
    sns.boxplot(x=hyper_label, y=df[col], hue=hyper_label, palette='Set1', legend=False)
    plt.xticks([0, 1], ['非高血脂', '高血脂'])
    plt.title(f'{col} - 按高血脂分组')
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(f'{col}_高血脂分组箱线图.png', dpi=300, bbox_inches='tight')
    plt.close()

print("箱线图已保存为PNG图片。")