import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import os
import warnings

# ========== 1. 设置中文字体（Mac 系统） ==========
def get_chinese_font():
    available = {f.name for f in fm.fontManager.ttflist}
    candidates = ['PingFang SC', 'Heiti SC', 'STHeiti', 'SimHei', 'Microsoft YaHei']
    for font in candidates:
        if font in available:
            return font
    # 模糊匹配
    for font in available:
        if any(k in font.lower() for k in ['ping', 'hei', 'song', 'cjk']):
            return font
    return None

chinese_font = get_chinese_font()
if chinese_font:
    plt.rcParams['font.sans-serif'] = [chinese_font]
    print(f"✅ 使用中文字体: {chinese_font}")
else:
    print("⚠️ 未找到中文字体，图片将使用英文标签。")
plt.rcParams['axes.unicode_minus'] = False

# ========== 2. 读取数据（体质类型 sheet） ==========
file_path = '附件1：样例数据 (1).xlsx'   # 请根据实际文件名修改
df = pd.read_excel(file_path, sheet_name='体质类型')

# 定义需要分析的变量（列名必须与 Excel 完全一致）
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

# 体质标签列名（第一列）
label_col = '体质标签'

# 检查列是否存在
if label_col not in df.columns:
    raise KeyError(f"列 '{label_col}' 不存在，请检查 sheet 列名。")
for col in target_vars:
    if col not in df.columns:
        raise KeyError(f"列 '{col}' 不存在，请检查 sheet 列名。")

# 体质类型映射
constitution_map = {
    1: '平和质', 2: '气虚质', 3: '阳虚质', 4: '阴虚质',
    5: '痰湿质', 6: '湿热质', 7: '血瘀质', 8: '气郁质', 9: '特禀质'
}

# ========== 3. 数据清洗（转为数值，但保留原始行用于逐变量检验） ==========
df_clean = df[[label_col] + target_vars].copy()
for col in target_vars:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
# 不删除整行，检验时按变量删除缺失

# ========== 4. 循环体质类型，进行 Mann-Whitney U 检验 ==========
results = []   # 存储所有检验结果

for constitution_id, constitution_name in constitution_map.items():
    print(f"\n正在处理体质类型：{constitution_name} (ID={constitution_id})")
    # 分组：该体质 vs 其他体质
    mask = df_clean[label_col] == constitution_id
    # 如果该组样本太少，跳过
    if mask.sum() == 0:
        print(f"  警告：体质 {constitution_name} 无样本，跳过。")
        continue
    for var in target_vars:
        # 提取两组有效数据（去除 NaN）
        x = df_clean.loc[mask, var].dropna().values.astype(float)
        y = df_clean.loc[~mask, var].dropna().values.astype(float)
        if len(x) == 0 or len(y) == 0:
            print(f"  变量 {var} 在某组中无有效数据，跳过。")
            results.append({
                '体质类型': constitution_name,
                '指标': var,
                'U统计量': np.nan,
                'P值': np.nan,
                '该体质组中位数': np.nan,
                '其他组中位数': np.nan,
                '该体质组样本量': len(x),
                '其他组样本量': len(y)
            })
            continue
        # 执行双侧 Mann-Whitney U 检验
        u_stat, p_value = stats.mannwhitneyu(x, y, alternative='two-sided')
        median_x = np.median(x)
        median_y = np.median(y)
        results.append({
            '体质类型': constitution_name,
            '指标': var,
            'U统计量': u_stat,
            'P值': p_value,
            '该体质组中位数': median_x,
            '其他组中位数': median_y,
            '该体质组样本量': len(x),
            '其他组样本量': len(y)
        })

# 转换为 DataFrame 并保存为 Excel
results_df = pd.DataFrame(results)
# 按体质类型和 P 值排序
results_df = results_df.sort_values(['体质类型', 'P值'])
# 将 P 值显示为科学计数法（避免显示 0）
results_df['P值_科学计数'] = results_df['P值'].apply(lambda x: f'{x:.6e}' if pd.notnull(x) else 'NaN')
# 调整列顺序用于输出
output_cols = ['体质类型', '指标', 'U统计量', 'P值_科学计数', 
               '该体质组中位数', '其他组中位数', '该体质组样本量', '其他组样本量']
final_df = results_df[output_cols].rename(columns={'P值_科学计数': 'P值'})

excel_path = 'constitution_mannwhitney_results.xlsx'
final_df.to_excel(excel_path, index=False)
print(f"\n📊 统计结果已保存至: {excel_path}")

# ========== 5. 绘制箱线图（每个体质类型每个变量一张图） ==========
output_dir = 'boxplots_constitution'
os.makedirs(output_dir, exist_ok=True)

# 为避免重复警告，先对每个体质类型创建子文件夹
for constitution_id, constitution_name in constitution_map.items():
    mask = df_clean[label_col] == constitution_id
    if mask.sum() == 0:
        continue
    # 创建该体质的子文件夹
    sub_dir = os.path.join(output_dir, constitution_name)
    os.makedirs(sub_dir, exist_ok=True)
    for var in target_vars:
        # 提取两组有效数据（用于绘图）
        x_vals = df_clean.loc[mask, var].dropna()
        y_vals = df_clean.loc[~mask, var].dropna()
        if len(x_vals) == 0 or len(y_vals) == 0:
            print(f"跳过绘图：{constitution_name} - {var}（某组无数据）")
            continue
        plt.figure(figsize=(6, 5))
        plot_data = pd.DataFrame({
            'Value': np.concatenate([x_vals.values, y_vals.values]),
            'Group': [f'{constitution_name}组'] * len(x_vals) + ['其他组'] * len(y_vals)
        })
        sns.boxplot(x='Group', y='Value', hue='Group', data=plot_data, palette='Set2', legend=False)
        # 获取该变量对应的 P 值
        p_row = results_df[(results_df['体质类型'] == constitution_name) & (results_df['指标'] == var)]
        if len(p_row) > 0:
            p_val = p_row['P值'].iloc[0] if 'P值' in p_row.columns else np.nan
            if pd.notna(p_val):
                plt.title(f'{constitution_name} vs 其他组\n{var}\nMann-Whitney U 检验 P = {p_val}')
            else:
                plt.title(f'{constitution_name} vs 其他组\n{var}\n（检验未进行）')
        else:
            plt.title(f'{constitution_name} vs 其他组\n{var}')
        plt.ylabel(var)
        plt.tight_layout()
        safe_name = var.replace('/', '_').replace('（', '_').replace('）', '_').replace(' ', '_')
        plt.savefig(os.path.join(sub_dir, f'{safe_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"🖼️ 已生成图片: {sub_dir}/{safe_name}.png")

print("\n✨ 分析完成！")