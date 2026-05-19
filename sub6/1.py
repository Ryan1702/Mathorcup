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
file_path = '附件1：样例数据 (1).xlsx'
sheet_name = '总'
df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
print("数据形状:", df.shape)
print("所有列名:", df.columns.tolist())

# 列名确认（根据实际输出调整）
tizhi_col = '体质标签'
target_col = '高血脂症二分类标签'
age_col = '年龄组'
gender_col = '性别'
smoking_col = '吸烟史'
alcohol_col = '饮酒史'

# 如果列名不存在，尝试模糊匹配
for col in [target_col, age_col, gender_col, smoking_col, alcohol_col]:
    if col not in df.columns:
        possible = [c for c in df.columns if col[:-1] in c or c in col]  # 简单匹配
        if possible:
            globals()[col + '_col'] = possible[0]  # 注意：这里需要动态修改变量，简单起见直接手动赋值
            print(f"列 {col} 匹配为 {possible[0]}")
        else:
            print(f"警告: 未找到列 {col}")

# 更安全的做法：手动指定（根据您打印的列名）
# 从您提供的列名看，没有显示 '高血脂症二分类标签'，可能实际列名不同
# 请根据打印的列名列表修改下面赋值
# 例如，如果实际列名是 '高血脂症二分类标签' 但没打印出来，可能在后半部分
# 建议您先运行到此处，查看打印的完整列名，然后手动修改。

# 为了快速解决，我们根据您之前的数据结构推断：第33列（0-based索引32）为高血脂标签
# 使用索引方式作为备用
target_idx = 32  # 根据之前错误信息中 '使用的列索引: 结局=31'，可能实际是31？需要确认
age_idx = 33
gender_idx = 34
smoking_idx = 35
alcohol_idx = 36

# 优先使用列名，如果失败则使用索引
if target_col not in df.columns:
    target_col = df.columns[target_idx]
    print(f"使用索引定位结局列: {target_col}")
if age_col not in df.columns:
    age_col = df.columns[age_idx]
if gender_col not in df.columns:
    gender_col = df.columns[gender_idx]
if smoking_col not in df.columns:
    smoking_col = df.columns[smoking_idx]
if alcohol_col not in df.columns:
    alcohol_col = df.columns[alcohol_idx]

print(f"最终使用的列: 体质={tizhi_col}, 结局={target_col}, 年龄={age_col}, 性别={gender_col}, 吸烟={smoking_col}, 饮酒={alcohol_col}")

# 提取数据
data = df[[tizhi_col, target_col, age_col, gender_col, smoking_col, alcohol_col]].copy()
# 转换数据类型
data[tizhi_col] = pd.to_numeric(data[tizhi_col], errors='coerce')
data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
data = data.dropna()
print(f"删除缺失值后样本数: {len(data)}")

# 过滤体质标签1-9
data = data[(data[tizhi_col] >= 1) & (data[tizhi_col] <= 9)]
print(f"过滤体质后样本数: {len(data)}")
if len(data) == 0:
    raise ValueError("没有有效样本，请检查数据")

# 体质映射
tizhi_map = {1:'平和质',2:'气虚质',3:'阳虚质',4:'阴虚质',
             5:'痰湿质',6:'湿热质',7:'血瘀质',8:'气郁质',9:'特禀质'}
data['tizhi_cat'] = data[tizhi_col].map(tizhi_map)

# ========== 3. 卡方检验 ==========
contingency = pd.crosstab(data['tizhi_cat'], data[target_col])
print("\n列联表:\n", contingency)
if contingency.shape[0] > 1 and contingency.sum().sum() > 0:
    chi2, p_chi2, dof, expected = chi2_contingency(contingency)
    print(f"卡方检验: χ²={chi2:.4f}, p={p_chi2:.6f}")
else:
    p_chi2 = np.nan

# ========== 4. 多因素逻辑回归 ==========
# 将分类变量转换为哑变量
data['age_cat'] = data[age_col].astype('category')
data['gender_cat'] = data[gender_col].astype('category')
data['smoking_cat'] = data[smoking_col].astype('category')
data['alcohol_cat'] = data[alcohol_col].astype('category')
data['tizhi_cat'] = pd.Categorical(data['tizhi_cat'], 
                                   categories=[tizhi_map[i] for i in range(1,10)], 
                                   ordered=False)

X = pd.get_dummies(data[['tizhi_cat', 'age_cat', 'gender_cat', 'smoking_cat', 'alcohol_cat']], 
                   drop_first=True)
X = sm.add_constant(X)
y = data[target_col]

# 关键：强制转换为 float
X = X.astype(float)
y = y.astype(float)

model = sm.Logit(y, X).fit(disp=0)
print(model.summary())

# 提取体质结果
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

# ========== 5. 保存结果 ==========
output_txt = 'logistic_regression_results.txt'
with open(output_txt, 'w', encoding='utf-8') as f:
    f.write("九种体质对高血脂症发病风险的贡献度分析\n")
    f.write("="*60 + "\n\n")
    if not np.isnan(p_chi2):
        f.write(f"卡方检验（各体质组患病率差异）: χ²={chi2:.4f}, P值={p_chi2:.6f}\n\n")
    f.write("多因素逻辑回归（调整年龄、性别、吸烟、饮酒，参照组：平和质）:\n")
    f.write("体质\tOR\t95% CI\t\tP值\n")
    for res in results:
        f.write(f"{res['体质']}\t{res['OR']:.3f}\t({res['95% CI 下限']:.3f}-{res['95% CI 上限']:.3f})\t{res['P值']:.4f}\n")
    f.write("\n解释：OR>1表示该体质比平和质患病风险更高，OR<1表示风险更低。\n")
print(f"结果已保存至 {output_txt}")

# ========== 6. 森林图 ==========
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

# ========== 7. 各体质患病率 ==========
prevalence = data.groupby('tizhi_cat')[target_col].mean().sort_values(ascending=False)
print("\n各体质组高血脂症患病率:")
for tizhi, rate in prevalence.items():
    print(f"  {tizhi}: {rate:.2%}")