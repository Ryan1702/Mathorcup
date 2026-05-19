import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
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

# ========== 2. 读取数据（Sheet2） ==========
file_path = '附件1：样例数据.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet2')

# 定义要检验的指标（按原顺序）
variables = [
    'ADL总分',
    'IADL总分',
    '活动量表总分（ADL总分+IADL总分）',
    'HDL-C（高密度脂蛋白）',
    'LDL-C（低密度脂蛋白）',
    'TG（甘油三酯）',
    'TC（总胆固醇）',
    '空腹血糖',
    '血尿酸',
    'BMI'
]
target_col = '高血脂症二分类标签'   # 0 = 无高血脂，1 = 有高血脂

# 检查列是否存在
missing = [v for v in variables if v not in df.columns] + ([target_col] if target_col not in df.columns else [])
if missing:
    raise KeyError(f"Sheet2 中缺失以下列：{missing}")

# 数据转换：转为数值，非数值变为 NaN
df_clean = df[variables + [target_col]].copy()
for col in variables:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# 分组
group0 = df_clean[target_col] == 0
group1 = df_clean[target_col] == 1

print(f"总样本量: {len(df_clean)}")
print(f"无高血脂症组 (0): {group0.sum()} 例")
print(f"有高血脂症组 (1): {group1.sum()} 例\n")

# ========== 3. Mann-Whitney U 检验 ==========
results = []
for var in variables:
    x = df_clean.loc[group0, var].dropna().values
    y = df_clean.loc[group1, var].dropna().values
    if len(x) == 0 or len(y) == 0:
        u, p = np.nan, np.nan
    else:
        u, p = stats.mannwhitneyu(x, y, alternative='two-sided')
    results.append({
        '变量': var,
        'P值': p,
        'U统计量': u,
        '无高血脂症组中位数': np.median(x) if len(x) > 0 else np.nan,
        '有高血脂症组中位数': np.median(y) if len(y) > 0 else np.nan,
        '有效样本数(0/1)': f"{len(x)}/{len(y)}"
    })
df_results = pd.DataFrame(results)
df_results = df_results.sort_values('P值')  # 按P值从小到大排序

# 保存 Excel
excel_path = 'mannwhitney_results.xlsx'
df_results.to_excel(excel_path, index=False)
print(f"📊 统计结果已保存至 {excel_path}")

# ========== 4. 绘制 P 值柱状图（无显著性标注，浅蓝色） ==========
# 保持原始顺序
order = variables
p_vals = []
for var in order:
    row = df_results[df_results['变量'] == var]
    p = row['P值'].values[0] if not row.empty else np.nan
    p_vals.append(p)

# 绘制柱状图
plt.figure(figsize=(12, 6))
bars = plt.bar(order, p_vals, color='lightblue', edgecolor='black')

# 在柱顶标注 P 值（不标注星号）
for bar, p in zip(bars, p_vals):
    if not np.isnan(p):
        # 格式化 P 值：小于 0.0001 用科学计数法，否则保留 4 位小数
        if p < 0.0001:
            label = f'{p:.2e}'
        else:
            label = f'{p:.4f}'
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 label, ha='center', va='bottom', fontsize=9)

plt.axhline(y=0.05, color='red', linestyle='--', linewidth=1, label='显著性阈值 α=0.05')
plt.ylabel('P 值', fontsize=12)
plt.title('各指标 Mann-Whitney U 检验的 P 值', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
# 设置纵坐标上限为最大 P 值的 1.2 倍（避免顶部空间不足）
max_p = max([p for p in p_vals if not np.isnan(p)])
plt.ylim(0, max_p * 1.2)
plt.legend()
plt.tight_layout()
plt.savefig('pvalue_barchart.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 柱状图已保存为 pvalue_barchart.png")