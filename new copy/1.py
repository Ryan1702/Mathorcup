import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import os

# ========== 1. 强制加载 Mac 系统自带的中文字体文件 ==========
font_path = '/System/Library/Fonts/PingFang.ttc'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti']
    print("✅ 已加载 Mac 中文字体文件 (PingFang.ttc)")
else:
    plt.rcParams['font.sans-serif'] = ['Heiti SC', 'STHeiti', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ========== 2. 读取数据 ==========
file_path = '附件1：样例数据.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

constitution_cols = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']
other_cols = ['ADL总分', 'IADL总分', '活动量表总分（ADL总分+IADL总分）', '空腹血糖', '血尿酸', 'BMI']
target_col = '高血脂症二分类标签'
all_vars = constitution_cols + other_cols

# 检查列名
missing = [c for c in all_vars + [target_col] if c not in df.columns]
if missing:
    raise KeyError(f"以下列缺失：{missing}")

# ========== 3. 数据清洗 ==========
df_clean = df[all_vars + [target_col]].copy()
for col in all_vars:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
df_clean = df_clean.dropna()   # 删除任何含有缺失值的行

group0 = df_clean[df_clean[target_col] == 0][all_vars]
group1 = df_clean[df_clean[target_col] == 1][all_vars]

print(f"\n总有效样本量: {len(df_clean)}")
print(f"无高血脂症组 (0): {len(group0)} 例")
print(f"有高血脂症组 (1): {len(group1)} 例\n")

if len(group0) == 0 or len(group1) == 0:
    raise ValueError("某一组样本数为0，无法进行检验。")

# ========== 4. Mann-Whitney U 检验（双侧） ==========
results = []
for var in all_vars:
    x = group0[var].values.astype(float)
    y = group1[var].values.astype(float)
    # 关键：使用 alternative='two-sided' 进行双侧检验
    u_stat, p_value = stats.mannwhitneyu(x, y, alternative='two-sided')
    median0 = np.median(x)
    median1 = np.median(y)
    results.append({
        '变量': var,
        'U统计量': u_stat,
        'P值': p_value,
        '无高血脂症组中位数': median0,
        '有高血脂症组中位数': median1
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('P值')

# ========== 5. 输出 Excel 文件（替代 txt） ==========
excel_path = 'mannwhitney_results.xlsx'
results_df.to_excel(excel_path, index=False, float_format='%.6f')
print(f"📊 统计结果已保存至 Excel: {excel_path}")
print(results_df.round(6).to_string(index=False))

# ========== 6. 绘制箱线图（全中文） ==========
os.makedirs('boxplots', exist_ok=True)
for var in all_vars:
    plt.figure(figsize=(6, 5))
    plot_data = pd.DataFrame({
        'Value': np.concatenate([group0[var].values, group1[var].values]),
        'Group': ['无高血脂症'] * len(group0) + ['有高血脂症'] * len(group1)
    })
    sns.boxplot(x='Group', y='Value', hue='Group', data=plot_data, palette='Set2', legend=False)
    p_val = results_df[results_df['变量'] == var]['P值'].values[0]
    plt.title(f'{var}\nMann-Whitney U 检验 P = {p_val:.4f}')
    plt.ylabel(var)
    plt.tight_layout()
    safe_name = var.replace('/', '_').replace('（', '_').replace('）', '_').replace(' ', '_')
    plt.savefig(f'boxplots/{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"🖼️ 已生成图片: boxplots/{safe_name}.png")

print("\n✨ 分析完成！")