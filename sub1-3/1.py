import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ========== 1. 强制加载 Mac 中文字体 ==========
font_path = '/System/Library/Fonts/PingFang.ttc'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti']
    print("✅ 已加载 Mac 中字体文件")
else:
    plt.rcParams['font.sans-serif'] = ['Heiti SC', 'STHeiti', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ========== 2. 输入数据 ==========
variables = ['TC', 'TG', 'LDL_C', 'HDL_C', '血糖', '血尿酸', 'BMI', 'ADL', 'IADL', '活动总分', '痰湿质积分']
p_values = [
    1.55237e-47,
    3.69054e-61,
    4.68045e-09,
    1.3894e-07,
    0.206272875,
    7.03198e-11,
    0.402746696,
    0.384171391,
    0.731505292,
    0.520019751,
    0.817155502
]

# 创建 DataFrame 并保存 Excel
df_results = pd.DataFrame({'变量': variables, 'P值': p_values})
excel_path = 'mannwhitney_results.xlsx'
df_results.to_excel(excel_path, index=False)
print(f"📊 统计结果已保存至 {excel_path}")

# ========== 3. 绘制柱状图 ==========
plt.figure(figsize=(12, 6))
bars = plt.bar(variables, p_values, color='lightblue', edgecolor='black')

# 在柱顶标注 P 值（科学计数法或小数）
for bar, p in zip(bars, p_values):
    if p < 0.0001:
        label = f'{p:.2e}'
    else:
        label = f'{p:.4f}'
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(p_values)*0.02,
             label, ha='center', va='bottom', fontsize=9, rotation=0)

plt.axhline(y=0.05, color='red', linestyle='--', linewidth=1, label='显著性阈值 α=0.05')
plt.ylabel('P 值', fontsize=12)
plt.title('各指标 Mann-Whitney U 检验的 P 值', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.ylim(0, max(p_values) * 1.2)
plt.legend()
plt.tight_layout()
plt.savefig('pvalue_barchart.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 柱状图已保存为 pvalue_barchart.png")