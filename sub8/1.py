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
else:
    plt.rcParams['font.sans-serif'] = ['Heiti SC', 'STHeiti', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ========== 2. 读取数据 ==========
file_path = '附件1：样例数据 (1).xlsx'
sheet_name = '总'
df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
print("数据形状:", df.shape)

# ========== 3. 指定列名（请根据实际修改） ==========
tizhi_col = '体质标签'
target_col = '高血脂症二分类标签'
age_col = '年龄组'
gender_col = '性别'
smoking_col = '吸烟史'
alcohol_col = '饮酒史'

# ========== 4. 数据清洗 ==========
data = df[[tizhi_col, target_col, age_col, gender_col, smoking_col, alcohol_col]].copy()
data[tizhi_col] = pd.to_numeric(data[tizhi_col], errors='coerce')
data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
data = data.dropna()
data = data[(data[tizhi_col] >= 1) & (data[tizhi_col] <= 9)]
print(f"有效样本数: {len(data)}")

# 体质映射
tizhi_map = {1:'平和质',2:'气虚质',3:'阳虚质',4:'阴虚质',
             5:'痰湿质',6:'湿热质',7:'血瘀质',8:'气郁质',9:'特禀质'}
data['tizhi_cat'] = data[tizhi_col].map(tizhi_map)

# 将分类变量转换为 category 类型
data['age_cat'] = data[age_col].astype('category')
data['gender_cat'] = data[gender_col].astype('category')
data['smoking_cat'] = data[smoking_col].astype('category')
data['alcohol_cat'] = data[alcohol_col].astype('category')

# 设置参照组：平和质作为参照（drop_first=True）
data['tizhi_cat'] = pd.Categorical(data['tizhi_cat'], 
                                   categories=[tizhi_map[i] for i in range(1,10)], 
                                   ordered=False)

# 构建模型矩阵
X = pd.get_dummies(data[['tizhi_cat', 'age_cat', 'gender_cat', 'smoking_cat', 'alcohol_cat']], drop_first=True)
X = sm.add_constant(X)
y = data[target_col].astype(float)
X = X.astype(float)

# 拟合模型
model = sm.Logit(y, X).fit(disp=0)
print("\n模型拟合完成。")

# ========== 5. 提取体质相关的OR、CI、P值 ==========
tizhi_dummies = [col for col in X.columns if col.startswith('tizhi_cat_')]
results_OR = []
for var in tizhi_dummies:
    coef = model.params[var]
    ci_low, ci_high = model.conf_int().loc[var]
    or_val = np.exp(coef)
    or_low = np.exp(ci_low)
    or_high = np.exp(ci_high)
    p_val = model.pvalues[var]
    tizhi_name = var.replace('tizhi_cat_', '')
    results_OR.append({
        '体质': tizhi_name,
        'OR': or_val,
        '95% CI 下限': or_low,
        '95% CI 上限': or_high,
        'P值': p_val
    })
results_OR = pd.DataFrame(results_OR).sort_values('OR', ascending=False)

# ========== 6. 计算每种体质的调整后患病概率（控制其他变量为样本均值） ==========
# 方法：构建一个虚拟数据集，每种体质单独一个样本，其他变量取样本众数
# 获取各分类变量的众数
age_mode = data['age_cat'].mode()[0]
gender_mode = data['gender_cat'].mode()[0]
smoking_mode = data['smoking_cat'].mode()[0]
alcohol_mode = data['alcohol_cat'].mode()[0]

# 为每种体质创建一行数据
pred_data = []
for tizhi in tizhi_map.values():
    row = {'const': 1}
    # 体质哑变量：当前体质为1，其他为0（参照组平和质对应的所有哑变量为0）
    for other in tizhi_dummies:
        row[other] = 1 if other == f'tizhi_cat_{tizhi}' else 0
    # 其他变量取众数
    for col in X.columns:
        if col.startswith('age_cat_'):
            row[col] = 1 if col == f'age_cat_{age_mode}' else 0
        elif col.startswith('gender_cat_'):
            row[col] = 1 if col == f'gender_cat_{gender_mode}' else 0
        elif col.startswith('smoking_cat_'):
            row[col] = 1 if col == f'smoking_cat_{smoking_mode}' else 0
        elif col.startswith('alcohol_cat_'):
            row[col] = 1 if col == f'alcohol_cat_{alcohol_mode}' else 0
    pred_data.append(row)

pred_df = pd.DataFrame(pred_data).fillna(0)
# 确保列顺序与X一致
pred_df = pred_df[X.columns]
pred_prob = model.predict(pred_df)

prob_results = pd.DataFrame({'体质': list(tizhi_map.values()), '调整后患病概率': pred_prob})
prob_results = prob_results.sort_values('调整后患病概率', ascending=False)

# ========== 7. 保存结果 ==========
# 保存 OR 结果
results_OR.to_csv('OR_results.csv', index=False, encoding='utf-8-sig')
# 保存调整后患病概率
prob_results.to_csv('adjusted_probabilities.csv', index=False, encoding='utf-8-sig')

# 打印输出
print("\n=== 各体质相对于平和质的 OR（比值比）===")
print(results_OR.to_string(index=False))
print("\n=== 各体质调整后患病概率（控制混杂因素）===")
print(prob_results.to_string(index=False))

# ========== 8. 绘制条形图（调整后患病概率） ==========
plt.figure(figsize=(10,6))
plt.barh(prob_results['体质'], prob_results['调整后患病概率'], color='skyblue')
plt.xlabel('调整后患病概率')
plt.title('九种体质调整后高血脂症患病概率（控制年龄、性别、吸烟、饮酒）')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('adjusted_probabilities.png', dpi=300)
plt.close()
print("\n条形图已保存: adjusted_probabilities.png")

# ========== 9. 森林图（OR） ==========
or_vals = results_OR['OR']
ci_low = results_OR['95% CI 下限']
ci_high = results_OR['95% CI 上限']
labels = results_OR['体质']

plt.figure(figsize=(10,6))
plt.errorbar(or_vals, range(len(labels)), 
             xerr=[or_vals - ci_low, ci_high - or_vals],
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

print("\n分析完成！")