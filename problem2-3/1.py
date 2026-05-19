import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import warnings
import os
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')

# ==================== 中文字体设置（Mac） ====================
font_path = '/System/Library/Fonts/PingFang.ttc'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti']
    print("✅ 已加载 Mac 中文字体文件")
else:
    plt.rcParams['font.sans-serif'] = ['Heiti SC', 'STHeiti', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 数据加载（与之前一致） ====================
file_path = '附件1：样例数据.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 九种体质列名
constitution_cols = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']
# 检查列名存在性
for col in constitution_cols:
    if col not in df.columns:
        raise KeyError(f"列 '{col}' 不存在，请检查数据")

rename_dict = {
    'ADL总分': 'ADL',
    'IADL总分': 'IADL',
    '活动量表总分（ADL总分+IADL总分）': '活动总分',
    'TC（总胆固醇）': 'TC',
    'TG（甘油三酯）': 'TG',
    'LDL-C（低密度脂蛋白）': 'LDL_C',
    'HDL-C（高密度脂蛋白）': 'HDL_C',
    '空腹血糖': '血糖',
    '血尿酸': '血尿酸',
    'BMI': 'BMI',
    '高血脂症二分类标签': '高血脂标签'
}
keep_cols = constitution_cols + list(rename_dict.keys()) + ['年龄组', '性别', '吸烟史', '饮酒史']
keep_cols = [c for c in keep_cols if c in df.columns]
df = df[keep_cols].copy()
df.rename(columns=rename_dict, inplace=True)
df['痰湿质积分'] = df['痰湿质']
df.dropna(inplace=True)
print(f"有效样本数：{len(df)}")

# 构建特征矩阵 X 和标签 y
X_constitution = df[constitution_cols]
X_biochemical = df[['TC', 'TG', 'LDL_C', 'HDL_C', '血糖', '血尿酸', 'BMI']]
X_activity = df[['ADL', 'IADL']]
X = pd.concat([X_constitution, X_biochemical, X_activity], axis=1)
y = df['高血脂标签']
print(f"特征维度：{X.shape[1]}, 正例比例：{y.mean():.3f}")

# ==================== 2. 随机森林模型 ====================
# 采用默认参数，但可以调整树的数量和深度以优化
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X, y)

# 预测概率
y_pred_prob = rf.predict_proba(X)[:, 1]   # 正类的概率
df['RF预测概率'] = y_pred_prob

# 计算 AUC
auc_rf = roc_auc_score(y, y_pred_prob)
print(f"随机森林 AUC = {auc_rf:.4f}")

# 交叉验证 AUC（可选）
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='roc_auc')
print(f"5折交叉验证 AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ==================== 3. 特征重要性 ====================
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feat_importance = pd.DataFrame({
    '特征': X.columns[indices],
    '重要性': importances[indices]
})

# ==================== 4. 风险分级（基于随机森林预测概率，使用固定阈值0.3/0.6） ====================
def risk_level(p):
    if p <= 0.3:
        return '低风险'
    elif p <= 0.6:
        return '中风险'
    else:
        return '高风险'
df['RF风险等级'] = df['RF预测概率'].apply(risk_level)

# 对比逻辑回归的分级（如果之前运行过逻辑回归，可读取；这里重新计算逻辑回归以便对比）
# 为了独立运行，这里也快速拟合一个逻辑回归用于对比
import statsmodels.api as sm
X_const = sm.add_constant(X)
logit_model = sm.Logit(y, X_const)
logit_result = logit_model.fit(disp=0)
df['LR预测概率'] = logit_result.predict(X_const)
df['LR风险等级'] = df['LR预测概率'].apply(risk_level)
auc_lr = roc_auc_score(y, df['LR预测概率'])
print(f"逻辑回归 AUC = {auc_lr:.4f}")

# ==================== 5. 保存结果到 Excel ====================
with pd.ExcelWriter('问题2_随机森林结果.xlsx', engine='openpyxl') as writer:
    # 个体预测结果
    output_cols = ['RF预测概率', 'RF风险等级', 'LR预测概率', 'LR风险等级', '痰湿质积分'] + list(X.columns)
    output_cols = [c for c in output_cols if c in df.columns]
    df[output_cols].to_excel(writer, sheet_name='个体预测', index=False)
    # 特征重要性
    feat_importance.to_excel(writer, sheet_name='特征重要性', index=False)
    # 两种模型风险等级交叉表
    cross = pd.crosstab(df['RF风险等级'], df['LR风险等级'], margins=True)
    cross.to_excel(writer, sheet_name='RFvsLR分级对比')
    # 混淆矩阵（RF模型）
    y_pred_class = (y_pred_prob >= 0.5).astype(int)
    cm = pd.DataFrame(confusion_matrix(y, y_pred_class), 
                      index=['真负', '真正'], columns=['预测负', '预测正'])
    cm.to_excel(writer, sheet_name='RF混淆矩阵')
print("✅ 随机森林结果已保存到 问题2_随机森林结果.xlsx")

# ==================== 6. 可视化 ====================
# 6.1 两个模型的ROC曲线对比
fpr_lr, tpr_lr, _ = roc_curve(y, df['LR预测概率'])
fpr_rf, tpr_rf, _ = roc_curve(y, y_pred_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr_lr, tpr_lr, label=f'逻辑回归 (AUC={auc_lr:.3f})', lw=2)
plt.plot(fpr_rf, tpr_rf, label=f'随机森林 (AUC={auc_rf:.3f})', lw=2)
plt.plot([0,1], [0,1], 'k--', lw=1)
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('ROC曲线对比')
plt.legend()
plt.tight_layout()
plt.savefig('问题2_ROC对比.png', dpi=300)
plt.close()

# 6.2 随机森林预测概率分布直方图（按RF风险等级着色）
plt.figure(figsize=(10,5))
colors = {'低风险':'green', '中风险':'orange', '高风险':'red'}
for level, color in colors.items():
    subset = df[df['RF风险等级'] == level]
    plt.hist(subset['RF预测概率'], bins=20, alpha=0.6, label=level, color=color, density=True)
plt.axvline(0.3, linestyle='--', color='gray')
plt.axvline(0.6, linestyle='--', color='gray')
plt.xlabel('预测患病概率')
plt.ylabel('密度')
plt.title('随机森林预测概率分布')
plt.legend()
plt.savefig('问题2_RF概率分布.png', dpi=300)
plt.close()

# 6.3 特征重要性条形图
plt.figure(figsize=(10,8))
plt.barh(feat_importance['特征'][:15][::-1], feat_importance['重要性'][:15][::-1])
plt.xlabel('重要性')
plt.title('随机森林特征重要性（前15）')
plt.tight_layout()
plt.savefig('问题2_特征重要性.png', dpi=300)
plt.close()

print("✅ 图表已保存：问题2_ROC对比.png、问题2_RF概率分布.png、问题2_特征重要性.png")

# ==================== 7. 输出对比总结 ====================
with open('问题2_模型对比总结.txt', 'w', encoding='utf-8') as f:
    f.write(f"""【逻辑回归 vs 随机森林对比】

样本量：{len(df)}
正例比例：{y.mean():.2%}

模型性能：
- 逻辑回归 AUC = {auc_lr:.4f}
- 随机森林 AUC = {auc_rf:.4f}
- 随机森林交叉验证 AUC = {cv_scores.mean():.4f} (±{cv_scores.std():.4f})

分级一致性（RF vs LR）：
{cross.to_string()}

随机森林最重要的5个特征：
{feat_importance.head(5).to_string(index=False)}

结论：
随机森林的AUC{ '高于' if auc_rf > auc_lr else '略低于或等于' }逻辑回归，说明{'非线性模型能捕捉更多复杂模式' if auc_rf > auc_lr else '线性模型已经足够，数据中非线性关系不强'}。
建议在实际应用中，可优先使用逻辑回归（可解释性强），同时参考随机森林的特征重要性辅助判断。
""")
print("✅ 对比总结已保存到 问题2_模型对比总结.txt")
print("随机森林分析完成！")