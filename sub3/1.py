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
print(f"当前 matplotlib 字体配置: {plt.rcParams['font.sans-serif']}")

# ========== 2. 读取数据（体质类型 sheet） ==========
file_path = '附件1：样例数据 (1).xlsx'   # 请修改为实际文件路径
df = pd.read_excel(file_path, sheet_name='体质类型')

# 定义需要分析的变量（与 Excel 列名完全一致）
target_vars = [
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
label_col = '体质标签'   # 第一列，值为 1-9

# 检查列是否存在
missing_cols = [c for c in [label_col] + target_vars if c not in df.columns]
if missing_cols:
    raise KeyError(f"以下列缺失：{missing_cols}")

# 体质类型映射
constitution_map = {
    1: '平和质', 2: '气虚质', 3: '阳虚质', 4: '阴虚质',
    5: '痰湿质', 6: '湿热质', 7: '血瘀质', 8: '气郁质', 9: '特禀质'
}

# ========== 3. 数据清洗（转为数值，保留原始行，检验时逐变量删除缺失） ==========
df_clean = df[[label_col] + target_vars].copy()
for col in target_vars:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# ========== 4. Kruskal-Wallis H 检验（每种体质 vs 其他） ==========
results = []
for constitution_id, constitution_name in constitution_map.items():
    print(f"\n正在处理体质类型：{constitution_name} (ID={constitution_id})")
    mask = df_clean[label_col] == constitution_id
    n_group = mask.sum()
    if n_group == 0:
        print(f"  警告：体质 {constitution_name} 无样本，跳过。")
        continue
    for var in target_vars:
        x = df_clean.loc[mask, var].dropna().values.astype(float)      # 该体质组
        y = df_clean.loc[~mask, var].dropna().values.astype(float)     # 其他组
        if len(x) == 0 or len(y) == 0:
            results.append({
                '体质类型': constitution_name,
                '指标': var,
                'H统计量': np.nan,
                'P值': np.nan,
                '该体质组中位数': np.nan,
                '其他组中位数': np.nan,
                '该体质组样本量': len(x),
                '其他组样本量': len(y)
            })
            continue
        # Kruskal-Wallis H 检验（两组情况）
        h_stat, p_value = stats.kruskal(x, y)
        median_x = np.median(x)
        median_y = np.median(y)
        results.append({
            '体质类型': constitution_name,
            '指标': var,
            'H统计量': h_stat,
            'P值': p_value,
            '该体质组中位数': median_x,
            '其他组中位数': median_y,
            '该体质组样本量': len(x),
            '其他组样本量': len(y)
        })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(['体质类型', 'P值'])
# 添加科学计数法 P 值列
results_df['P值_科学计数'] = results_df['P值'].apply(lambda x: f'{x:.6e}' if pd.notnull(x) else 'NaN')
final_df = results_df[['体质类型', '指标', 'H统计量', 'P值_科学计数', 
                       '该体质组中位数', '其他组中位数', '该体质组样本量', '其他组样本量']]
final_df = final_df.rename(columns={'P值_科学计数': 'P值'})

# 保存为 Excel
excel_path = 'constitution_kruskal_results.xlsx'
final_df.to_excel(excel_path, index=False)
print(f"\n📊 统计结果已保存至: {excel_path}")

# ========== 5. 绘制箱线图（每种体质一个子文件夹） ==========
output_dir = 'boxplots_constitution'
os.makedirs(output_dir, exist_ok=True)
use_english = (not os.path.exists(font_path))   # 若未加载中文字体则使用英文标签

for constitution_id, constitution_name in constitution_map.items():
    mask = df_clean[label_col] == constitution_id
    if mask.sum() == 0:
        continue
    sub_dir = os.path.join(output_dir, constitution_name)
    os.makedirs(sub_dir, exist_ok=True)
    for var in target_vars:
        x_vals = df_clean.loc[mask, var].dropna()
        y_vals = df_clean.loc[~mask, var].dropna()
        if len(x_vals) == 0 or len(y_vals) == 0:
            print(f"跳过绘图：{constitution_name} - {var}（某组无有效数据）")
            continue
        plt.figure(figsize=(6, 5))
        plot_data = pd.DataFrame({
            'Value': np.concatenate([x_vals.values, y_vals.values]),
            'Group': [f'{constitution_name}组'] * len(x_vals) + ['其他组'] * len(y_vals)
        })
        sns.boxplot(x='Group', y='Value', hue='Group', data=plot_data, palette='Set2', legend=False)
        p_row = results_df[(results_df['体质类型'] == constitution_name) & (results_df['指标'] == var)]
        if len(p_row) > 0:
            p_val = p_row['P值'].iloc[0] if 'P值' in p_row.columns else np.nan
            if pd.notna(p_val):
                if use_english:
                    plt.title(f'{constitution_name} vs Others\n{var}\nKruskal-Wallis H test P = {p_val}')
                else:
                    plt.title(f'{constitution_name} vs 其他组\n{var}\nKruskal-Wallis H 检验 P = {p_val}')
            else:
                plt.title(f'{constitution_name} vs Others\n{var}' if use_english else f'{constitution_name} vs 其他组\n{var}')
        else:
            plt.title(f'{constitution_name} vs Others\n{var}' if use_english else f'{constitution_name} vs 其他组\n{var}')
        plt.ylabel(var)
        plt.tight_layout()
        safe_name = var.replace('/', '_').replace('（', '_').replace('）', '_').replace(' ', '_')
        plt.savefig(os.path.join(sub_dir, f'{safe_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"🖼️ 已生成图片: {sub_dir}/{safe_name}.png")

print("\n✨ 分析完成！")