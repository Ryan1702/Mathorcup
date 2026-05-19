import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from scipy.stats import chi2_contingency

# ========== 1. 中文字体设置 ==========
font_path = '/System/Library/Fonts/PingFang.ttc'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti']
    print("✅ 已加载 Mac 中文字体文件")
else:
    plt.rcParams['font.sans-serif'] = ['Heiti SC', 'STHeiti', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ========== 2. 读取数据 ==========
file_path = '附件1：样例数据 (1).xlsx'   # 请修改为实际路径
sheet_name = '总'
df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
print("数据形状:", df.shape)
print("所有列名:\n", df.columns.tolist())

# ========== 3. 手动指定关键列名（请根据上面打印的列名修改） ==========
# 以下为默认列名，如果实际不同请修改引号内的字符串
tizhi_col = '体质标签'
target_col = '高血脂症二分类标签'      # 结局变量（1=高血脂，0=正常）
age_col = '年龄组'
gender_col = '性别'
smoking_col = '吸烟史'
alcohol_col = '饮酒史'

# 检查列名是否存在，若不存在则报错并提示
required_cols = [tizhi_col, target_col, age_col, gender_col, smoking_col, alcohol_col]
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"列 '{col}' 不存在于数据中，请根据打印的列名修改代码中的列名。")

print(f"使用的列: {required_cols}")

# ========== 4. 提取数据并清洗 ==========
data = df[required_cols].copy()

# 转换数据类型：体质标签和结局变量强制为数值
data[tizhi_col] = pd.to_numeric(data[tizhi_col], errors='coerce')
data[target_col] = pd.to_numeric(data[target_col], errors='coerce')

# 删除缺失值
data = data.dropna()
print(f"删除缺失值后样本数: {len(data)}")

# 过滤体质标签1-9
data = data[(data[tizhi_col] >= 1) & (data[tizhi_col] <= 9)]
print(f"过滤体质标签后样本数: {len(data)}")
if len(data) == 0:
    raise ValueError("没有有效样本，请检查数据中的体质标签是否为1-9。")

# 体质映射
tizhi_map = {1:'平和质',2:'气虚质',3:'阳虚质',4:'阴虚质',
             5:'痰湿质',6:'湿热质',7:'血瘀质',8:'气郁质',9:'特禀质'}
data['tizhi_cat'] = data[tizhi_col].map(tizhi_map)

# 查看各体质组样本量
print("\n各体质组样本量:")
print(data['tizhi_cat'].value_counts())

# ========== 5. 卡方检验（各体质患病率差异） ==========
contingency = pd.crosstab(data['tizhi_cat'], data[target_col])
print("\n列联表（患病/未患病）:\n", contingency)
if contingency.shape[0] > 1 and contingency.sum().sum() > 0:
    chi2, p_chi2, dof, expected = chi2_contingency(contingency)
    print(f"卡方检验: χ²={chi2:.4f}, p={p_chi2:.6f}")
else:
    p_chi2 = np.nan
    print("无法进行卡方检验（数据不足）")

# ========== 6. 多因素逻辑回归（调整年龄、性别、吸烟、饮酒） ==========
# 将分类变量转换为哑变量
data['age_cat'] = data[age_col].astype('category')
data['gender_cat'] = data[gender_col].astype('category')
data['smoking_cat'] = data[smoking_col].astype('category')
data['alcohol_cat'] = data[alcohol_col].astype('category')
# 设置体质参照组为平和质（按tizhi_map顺序，平和质第一个，drop_first=True会删除它）
data['tizhi_cat'] = pd.Categorical(data['tizhi_cat'], 
                                   categories=[tizhi_map[i] for i in range(1,10)], 
                                   ordered=False)

X = pd.get_dummies(data[['tizhi_cat', 'age_cat', 'gender_cat', 'smoking_cat', 'alcohol_cat']], 
                   drop_first=True)
X = sm.add_constant(X)
y = data[target_col]

# 强制转换为浮点数，避免 dtype object 错误
X = X.astype(float)
y = y.astype(float)

# 拟合模型
model = sm.Logit(y, X).fit(disp=0)
print("\n模型拟合结果摘要（仅显示系数部分）:")
print(model.summary2().tables[1])   # 显示系数表

# ========== 7. 提取体质相关的OR、CI、P值 ==========
tizhi_dummies = [col for col in X.columns if col.startswith('tizhi_cat_')]
results = []
for var in tizhi_dummies:
    coef = model.params[var]
    ci_low, ci_high = model.conf_int().loc[var]
    or_val = np.exp(coef)
    or_low = np.exp(ci_low)
    or_high = np.exp(ci_high)
    p_val = model.pvalues[var]
    tizhi_name = var.replace('tizhi_cat_', '')
    results.append({
        '体质': tizhi_name,
        'OR': or_val,
        '95% CI 下限': or_low,
        '95% CI 上限': or_high,
        'P值': p_val
    })

# ========== 8. 保存结果到文本文件 ==========
output_txt = 'logistic_regression_results.txt'
with open(output_txt, 'w', encoding='utf-8') as f:
    f.write("九种体质对高血脂症发病风险的贡献度分析\n")
    f.write("="*60 + "\n\n")
    if not np.isnan(p_chi2):
        f.write(f"卡方检验（各体质组患病率差异）: χ²={chi2:.4f}, P值={p_chi2:.6f}\n")
        f.write("（P<0.05表示不同体质间患病率有显著差异）\n\n")
    f.write("多因素逻辑回归（调整年龄、性别、吸烟、饮酒，参照组：平和质）:\n")
    f.write("体质\tOR\t95% CI\t\tP值\n")
    for res in results:
        f.write(f"{res['体质']}\t{res['OR']:.3f}\t({res['95% CI 下限']:.3f}-{res['95% CI 上限']:.3f})\t{res['P值']:.4f}\n")
    f.write("\n解释：OR>1表示该体质比平和质患病风险更高，OR<1表示风险更低。\n")
print(f"\n结果已保存至 {output_txt}")

# ========== 9. 绘制森林图 ==========
or_vals = [r['OR'] for r in results]
ci_low = [r['95% CI 下限'] for r in results]
ci_high = [r['95% CI 上限'] for r in results]
labels = [r['体质'] for r in results]

plt.figure(figsize=(10, 6))
plt.errorbar(or_vals, range(len(labels)), 
             xerr=[np.array(or_vals)-np.array(ci_low), np.array(ci_high)-np.array(or_vals)],
             fmt='o', capsize=5, color='green', ecolor='gray')
plt.axvline(x=1, linestyle='--', color='red')
plt.yticks(range(len(labels)), labels)
plt.xlabel('Odds Ratio (对数刻度)')
plt.title('多因素逻辑回归森林图（参照：平和质）')
plt.xscale('log')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('forest_plot.png', dpi=300)
plt.close()
print("森林图已保存: forest_plot.png")

# ========== 10. 输出各体质实际患病率 ==========
prevalence = data.groupby('tizhi_cat')[target_col].mean().sort_values(ascending=False)
print("\n各体质组高血脂症患病率（原始，未调整混杂）:")
for tizhi, rate in prevalence.items():
    print(f"  {tizhi}: {rate:.2%}")

print("\n分析完成！")