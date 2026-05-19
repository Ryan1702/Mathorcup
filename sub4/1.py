import pandas as pd
import numpy as np
from scipy.stats import kruskal
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
import os

# ========== 1. 自动检测并设置中文字体 ==========
font_path = '/System/Library/Fonts/PingFang.ttc'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti']
    print("✅ 已加载 Mac 中文字体文件 (PingFang.ttc)")
else:
    plt.rcParams['font.sans-serif'] = ['Heiti SC', 'STHeiti', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 检查 Matplotlib 版本以兼容 boxplot 参数
mpl_version = matplotlib.__version__
use_tick_labels = mpl_version >= '3.9.0'
print(f"Matplotlib 版本: {mpl_version}, 使用 tick_labels: {use_tick_labels}")

# ========== 2. 文件路径和参数设置 ==========
file_path = '附件1：样例数据 (1).xlsx'   # 请修改为实际路径
sheet_name = '体质类型'

# 体质标签映射
tizhi_labels = {
    1: '平和质', 2: '气虚质', 3: '阳虚质', 4: '阴虚质',
    5: '痰湿质', 6: '湿热质', 7: '血瘀质', 8: '气郁质', 9: '特禀质'
}

# 需要分析的指标及其列索引（0-based）
indicators = {
    'ADL总分': 17,
    'IADL总分': 23,
    'ADL+IADL': 24,
    'HDL-C': 25,
    'LDL-C': 26,
    'TG': 27,
    'TC': 28,
    '空腹血糖': 29,
    '血尿酸': 30,
    'BMI': 31
}

# ========== 3. 读取数据 ==========
df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
print(f"原始数据形状: {df.shape}")

# 体质标签列（第一列，索引0）
tizhi_col = df.iloc[:, 0].values

results = {}

# ========== 4. 对每个指标进行 Kruskal-Wallis 检验 ==========
for name, col_idx in indicators.items():
    # 提取指标数据并转换为数值类型
    data = pd.to_numeric(df.iloc[:, col_idx], errors='coerce').values
    # 去除缺失值
    mask = ~(pd.isna(tizhi_col) | pd.isna(data))
    tizhi_clean = tizhi_col[mask]
    data_clean = data[mask]
    
    # 按体质标签分组
    groups = []
    for label in range(1, 10):
        group_data = data_clean[tizhi_clean == label]
        group_data = group_data[~np.isnan(group_data)]   # 再次清理
        groups.append(group_data)
    
    # 只保留样本量 ≥2 的组进行检验
    non_empty_groups = [g for g in groups if len(g) >= 2]
    sample_sizes = [len(g) for g in groups]
    
    if len(non_empty_groups) >= 2:
        try:
            h_stat, p_value = kruskal(*non_empty_groups)
        except Exception as e:
            print(f"指标 {name} 检验出错: {e}")
            h_stat, p_value = np.nan, np.nan
    else:
        h_stat, p_value = np.nan, np.nan
        print(f"指标 {name} 有效组数不足2组，跳过检验")
    
    results[name] = {
        'H统计量': h_stat,
        'P值': p_value,
        '显著(p<0.05)': p_value < 0.05 if not np.isnan(p_value) else False,
        '各组样本量': sample_sizes,
        '各组数据': groups
    }
    h_str = f"{h_stat:.4f}" if not np.isnan(h_stat) else "NaN"
    p_str = f"{p_value:.6f}" if not np.isnan(p_value) else "NaN"
    print(f"{name}: H={h_str}, p={p_str}")

# ========== 5. 保存文本结果 ==========
output_txt = 'kruskal_results.txt'
with open(output_txt, 'w', encoding='utf-8') as f:
    f.write("Kruskal-Wallis H检验结果（9种体质分组）\n")
    f.write("="*60 + "\n")
    for name, res in results.items():
        f.write(f"\n指标: {name}\n")
        if not np.isnan(res['H统计量']):
            f.write(f"  H统计量: {res['H统计量']:.4f}\n")
            f.write(f"  P值: {res['P值']:.6f}\n")
        else:
            f.write("  H统计量: NaN\n")
            f.write("  P值: NaN\n")
        f.write(f"  是否显著(p<0.05): {'是' if res['显著(p<0.05)'] else '否'}\n")
        f.write(f"  各组样本量: {res['各组样本量']}\n")
    f.write("\n" + "="*60 + "\n")
    f.write("注：P值<0.05表示9种体质在该指标上存在显著差异。\n")
print(f"文本结果已保存至 {output_txt}")

# ========== 6. 绘制箱线图 ==========
output_dir = 'boxplots'
os.makedirs(output_dir, exist_ok=True)

for name, res in results.items():
    groups_data = res['各组数据']
    plot_data = []
    plot_labels = []
    for i, g in enumerate(groups_data):
        if len(g) > 0:
            plot_data.append(g)
            plot_labels.append(tizhi_labels[i+1])
    
    if len(plot_data) == 0:
        continue
    
    plt.figure(figsize=(10, 6))
    # 根据 Matplotlib 版本选择参数名称
    if use_tick_labels:
        bp = plt.boxplot(plot_data, tick_labels=plot_labels, patch_artist=True,
                         showmeans=True, meanline=True)
    else:
        bp = plt.boxplot(plot_data, labels=plot_labels, patch_artist=True,
                         showmeans=True, meanline=True)
    for box in bp['boxes']:
        box.set_facecolor('lightblue')
    plt.xticks(rotation=45)
    plt.title(f'{name} 在不同体质类型中的分布')
    plt.ylabel(name)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{name}.png'), dpi=300)
    plt.close()
    print(f"已保存箱线图: {name}.png")

# ========== 7. 绘制P值条形图 ==========
p_values = [res['P值'] for res in results.values()]
indicator_names = list(results.keys())
valid_p = [p for p in p_values if not np.isnan(p)]
valid_names = [name for name, p in zip(indicator_names, p_values) if not np.isnan(p)]

if len(valid_p) > 0:
    plt.figure(figsize=(12, 6))
    bars = plt.bar(valid_names, valid_p, color='skyblue')
    plt.axhline(y=0.05, color='red', linestyle='--', label='显著性阈值 (0.05)')
    plt.ylabel('P值')
    plt.title('各指标Kruskal-Wallis检验的P值')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    for bar, p in zip(bars, valid_p):
        color = 'salmon' if p < 0.05 else 'skyblue'
        bar.set_color(color)
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{p:.4f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig('pvalue_barchart.png', dpi=300)
    plt.close()
    print("已保存P值条形图: pvalue_barchart.png")
else:
    print("没有有效的P值可用于绘制条形图。")

print("\n分析完成！")