import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import os

# ========== 1. 强制加载 Mac 中文字体 ==========
font_path = '/System/Library/Fonts/PingFang.ttc'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti']
    print("✅ 已加载 Mac 中文字体文件 (PingFang.ttc)")
else:
    plt.rcParams['font.sans-serif'] = ['Heiti SC', 'STHeiti', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ========== 2. 读取数据（使用 sheet2） ==========
file_path = '附件1：样例数据.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet2')   # 关键修改：读取 Sheet2

# 定义变量列名（与 Sheet2 中的列名完全一致）
constitution_cols = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']
other_cols = ['ADL总分', 'IADL总分', '活动量表总分（ADL总分+IADL总分）', '空腹血糖', '血尿酸', 'BMI']
target_col = '高血脂症二分类标签'   # Sheet2 中第一列就是该标签
all_vars = constitution_cols + other_cols

# 检查列名是否存在于 sheet2 中
missing = [c for c in all_vars + [target_col] if c not in df.columns]
if missing:
    raise KeyError(f"Sheet2 中缺失以下列：{missing}")

# ========== 3. 数据预处理（不删除行，只将非数值转为 NaN） ==========
# 为了保留所有原始数据，我们不对整行进行删除，只将无法转换为数值的单元格变为 NaN
df_clean = df[all_vars + [target_col]].copy()
for col in all_vars:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# 分组（不删除任何行，NaN 保留）
group0_mask = df_clean[target_col] == 0
group1_mask = df_clean[target_col] == 1

print(f"总样本量: {len(df_clean)}")
print(f"无高血脂症组 (0): {group0_mask.sum()} 例")
print(f"有高血脂症组 (1): {group1_mask.sum()} 例")
print("\n注意：每个变量实际用于检验的样本数可能少于总例数（因为存在缺失值），检验时会自动忽略 NaN。\n")

if group0_mask.sum() == 0 or group1_mask.sum() == 0:
    raise ValueError("某一组样本数为0，无法进行检验。")

# ========== 4. Mann-Whitney U 检验（逐变量忽略缺失值） ==========
results = []
for var in all_vars:
    # 提取两组中该变量的非缺失观测值
    x = df_clean.loc[group0_mask, var].dropna().values.astype(float)
    y = df_clean.loc[group1_mask, var].dropna().values.astype(float)
    
    # 如果某组缺失值过多导致无有效数据，则跳过该变量
    if len(x) == 0 or len(y) == 0:
        print(f"警告：变量 {var} 在某一组中无有效数值，跳过检验。")
        results.append({
            '变量': var,
            'U统计量': np.nan,
            'P值': np.nan,
            '无高血脂症组中位数': np.nan,
            '有高血脂症组中位数': np.nan,
            '有效样本数(组0/组1)': f"{len(x)}/{len(y)}"
        })
        continue
    
    # 执行双侧 Mann-Whitney U 检验
    u_stat, p_value = stats.mannwhitneyu(x, y, alternative='two-sided')
    median0 = np.median(x)
    median1 = np.median(y)
    results.append({
        '变量': var,
        'U统计量': u_stat,
        'P值': p_value,
        '无高血脂症组中位数': median0,
        '有高血脂症组中位数': median1,
        '有效样本数(组0/组1)': f"{len(x)}/{len(y)}"
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('P值')

# 为了显示精确 P 值（避免误以为 0），将 P 列转为科学计数法字符串
results_df['P值_科学计数'] = results_df['P值'].apply(lambda x: f'{x:.6e}' if pd.notnull(x) else 'NaN')
# 调整列顺序，方便查看
final_df = results_df[['变量', 'U统计量', 'P值_科学计数', '无高血脂症组中位数', '有高血脂症组中位数', '有效样本数(组0/组1)']]
final_df = final_df.rename(columns={'P值_科学计数': 'P值'})

# 保存为 Excel（保留科学计数法字符串）
excel_path = 'mannwhitney_results.xlsx'
final_df.to_excel(excel_path, index=False)
print(f"\n📊 统计结果已保存至 Excel: {excel_path}")
print(final_df.to_string(index=False))

# ========== 5. 绘制箱线图（仅针对有足够样本的变量） ==========
os.makedirs('boxplots', exist_ok=True)
for var in all_vars:
    # 获取该变量非缺失的观测值（用于绘图）
    x_vals = df_clean.loc[group0_mask, var].dropna()
    y_vals = df_clean.loc[group1_mask, var].dropna()
    if len(x_vals) == 0 or len(y_vals) == 0:
        print(f"跳过变量 {var}（某组无有效数据，无法绘图）")
        continue
    
    plt.figure(figsize=(6, 5))
    plot_data = pd.DataFrame({
        'Value': np.concatenate([x_vals.values, y_vals.values]),
        'Group': ['无高血脂症'] * len(x_vals) + ['有高血脂症'] * len(y_vals)
    })
    sns.boxplot(x='Group', y='Value', hue='Group', data=plot_data, palette='Set2', legend=False)
    # 获取对应的 P 值（原始数值）
    p_val_row = results_df[results_df['变量'] == var]['P值'].values
    p_val = p_val_row[0] if len(p_val_row) > 0 and not pd.isna(p_val_row[0]) else np.nan
    if pd.notna(p_val):
        title = f'{var}\nMann-Whitney U 检验 P = {p_val:.4e}'
    else:
        title = f'{var}\n（检验未进行或样本不足）'
    plt.title(title)
    plt.ylabel(var)
    plt.tight_layout()
    safe_name = var.replace('/', '_').replace('（', '_').replace('）', '_').replace(' ', '_')
    plt.savefig(f'boxplots/{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"🖼️ 已生成图片: boxplots/{safe_name}.png")

print("\n✨ 分析完成！")