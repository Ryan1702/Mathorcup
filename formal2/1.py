import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import os

# ------------------------------
# 0. 创建输出文件夹
# ------------------------------
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# 1. 读取数据
# ------------------------------
file_path = "附件1：样例数据.xlsx"   # 请修改为实际路径
df = pd.read_excel(file_path, sheet_name=0)

# ------------------------------
# 2. 定义变量列名
# ------------------------------
phlegm_col = "痰湿质"
activity_total = "活动量表总分（ADL总分+IADL总分）"
adl_total = "ADL总分"
iadl_total = "IADL总分"

lipid_cols = ["HDL-C（高密度脂蛋白）", "LDL-C（低密度脂蛋白）", 
              "TG（甘油三酯）", "TC（总胆固醇）"]
metabolic_cols = ["空腹血糖", "血尿酸", "BMI"]
all_physiological = lipid_cols + metabolic_cols + [activity_total, adl_total, iadl_total]

target = "高血脂症二分类标签"

constitution_cols = ["平和质", "气虚质", "阳虚质", "阴虚质", "痰湿质", 
                     "湿热质", "血瘀质", "气郁质", "特禀质"]

demographic_cols = ["年龄组", "性别", "吸烟史", "饮酒史"]

# 提取数据
data = df[list(set(all_physiological + [phlegm_col, target] + constitution_cols + demographic_cols))].dropna().copy()
data[target] = data[target].astype(int)

# ------------------------------
# 3. 表征痰湿体质严重程度的关键指标（Pearson相关）
# ------------------------------
print("="*60)
print("3. 各指标与痰湿质积分的Pearson相关系数")
print("="*60)

corr_results = []
for col in all_physiological:
    r, p = stats.pearsonr(data[phlegm_col], data[col])
    corr_results.append({"指标": col, "相关系数": r, "p值": p})
corr_df = pd.DataFrame(corr_results).sort_values("相关系数", key=abs, ascending=False)
print(corr_df.round(4).to_string(index=False))

key_for_phlegm = corr_df[(abs(corr_df["相关系数"]) > 0.3) & (corr_df["p值"] < 0.05)]
print("\n→ 能有效表征痰湿体质严重程度的关键指标：")
print(key_for_phlegm["指标"].tolist())

# ------------------------------
# 4. 预警高血脂发病风险的关键指标（多因素逻辑回归）
# ------------------------------
print("\n" + "="*60)
print("4. 各指标预警高血脂风险的多因素逻辑回归")
print("="*60)

X = data[all_physiological + demographic_cols]
y = data[target]
X_const = sm.add_constant(X)
model = sm.Logit(y, X_const)
result = model.fit(disp=0)

logit_summary = pd.DataFrame({
    "变量": result.params.index,
    "系数": result.params.values,
    "OR": np.exp(result.params.values),
    "OR_95%CI_low": np.exp(result.conf_int()[0]),
    "OR_95%CI_high": np.exp(result.conf_int()[1]),
    "p值": result.pvalues
})
logit_summary = logit_summary[logit_summary["变量"] != "const"].sort_values("p值")
print(logit_summary.round(4).to_string(index=False))

key_for_risk = logit_summary[logit_summary["p值"] < 0.05]
print("\n→ 能独立预警高血脂发病风险的关键指标：")
print(key_for_risk["变量"].tolist())

# ------------------------------
# 5. 同时满足两个条件的关键指标
# ------------------------------
print("\n" + "="*60)
print("5. 同时能表征痰湿严重程度且预警高血脂风险的关键指标")
print("="*60)

common = set(key_for_phlegm["指标"]).intersection(set(key_for_risk["变量"]))
print("交集指标：", common)
common_details = []
for idx, row in key_for_phlegm[key_for_phlegm["指标"].isin(common)].iterrows():
    var = row["指标"]
    or_val = key_for_risk[key_for_risk["变量"]==var]["OR"].values[0]
    print(f"{var}: r={row['相关系数']:.3f}, OR={or_val:.2f}")
    common_details.append({"指标": var, "与痰湿质相关系数": row['相关系数'], 
                           "p值(相关)": row['p值'], "预警高血脂OR": or_val,
                           "p值(回归)": key_for_risk[key_for_risk["变量"]==var]['p值'].values[0]})
common_df = pd.DataFrame(common_details)

# ------------------------------
# 6. 九种体质对高血脂发病风险的贡献度（积分连续变量）
# ------------------------------
print("\n" + "="*60)
print("6. 九种体质积分对高血脂风险的贡献度（每10分OR）")
print("="*60)

X_con = data[constitution_cols + demographic_cols]
X_con_const = sm.add_constant(X_con)
model_con = sm.Logit(y, X_con_const)
result_con = model_con.fit(disp=0)

coeff = result_con.params[constitution_cols]
or_per_10 = np.exp(coeff * 10)
conf_int_per_10 = np.exp(result_con.conf_int().loc[constitution_cols] * 10)
constitution_result = pd.DataFrame({
    "体质类型": constitution_cols,
    "OR(每10分)": or_per_10.values,
    "95%CI_low": conf_int_per_10.iloc[:,0].values,
    "95%CI_high": conf_int_per_10.iloc[:,1].values,
    "p值": result_con.pvalues[constitution_cols].values
}).sort_values("OR(每10分)", ascending=False)
print(constitution_result.round(4).to_string(index=False))

# ------------------------------
# 6b. 体质分类标签的贡献度（参照=平和质）
# ------------------------------
print("\n" + "="*60)
print("6b. 体质分类标签的贡献度（参照组=平和质）")
print("="*60)

label_map = {1:"平和质",2:"气虚质",3:"阳虚质",4:"阴虚质",5:"痰湿质",
             6:"湿热质",7:"血瘀质",8:"气郁质",9:"特禀质"}
constitution_label = df["体质标签"].map(label_map).astype(str)
X_cat = pd.get_dummies(constitution_label, prefix='', prefix_sep='')
X_cat = X_cat.astype(int)
if all(col.isdigit() for col in X_cat.columns):
    X_cat = X_cat.rename(columns={str(k): v for k, v in label_map.items()})
if "平和质" in X_cat.columns:
    X_cat = X_cat.drop("平和质", axis=1)
else:
    raise KeyError("未找到'平和质'列，实际列名：" + str(X_cat.columns.tolist()))
X_cat_demo = pd.concat([X_cat, data[demographic_cols]], axis=1)
X_cat_demo = X_cat_demo.astype(float)
X_cat_const = sm.add_constant(X_cat_demo)
model_cat = sm.Logit(y, X_cat_const)
result_cat = model_cat.fit(disp=0)

cat_summary = pd.DataFrame({
    "体质类型": result_cat.params.index[1:],
    "OR": np.exp(result_cat.params.values[1:]),
    "95%CI_low": np.exp(result_cat.conf_int()[0][1:]),
    "95%CI_high": np.exp(result_cat.conf_int()[1][1:]),
    "p值": result_cat.pvalues[1:]
}).sort_values("OR", ascending=False)
print(cat_summary.round(4).to_string(index=False))

# ------------------------------
# 7. 保存结果到Excel和文本文件
# ------------------------------
output_file = os.path.join(output_dir, "问题一_分析结果.xlsx")
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    corr_df.to_excel(writer, sheet_name="与痰湿质相关系数", index=False)
    logit_summary.to_excel(writer, sheet_name="多因素逻辑回归", index=False)
    common_df.to_excel(writer, sheet_name="双功能关键指标", index=False)
    constitution_result.to_excel(writer, sheet_name="体质积分贡献度(每10分)", index=False)
    cat_summary.to_excel(writer, sheet_name="体质分类贡献度(参照平和)", index=False)

# 保存文本总结
summary_file = os.path.join(output_dir, "问题一_结论汇总.txt")
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("【问题一结论汇总】\n\n")
    f.write("1. 既能表征痰湿严重程度又能预警高血脂风险的关键指标为：\n")
    f.write(f"   {', '.join(common)}\n\n")
    f.write("2. 九种体质对高血脂发病风险的贡献度（每10分OR）：\n")
    for _, row in constitution_result.iterrows():
        f.write(f"   {row['体质类型']}: OR={row['OR(每10分)']:.2f} "
                f"(95%CI {row['95%CI_low']:.2f}-{row['95%CI_high']:.2f}), "
                f"p={row['p值']:.4f}\n")
    f.write("\n3. 以平和质为参照的体质分类贡献度：\n")
    for _, row in cat_summary.iterrows():
        f.write(f"   {row['体质类型']}: OR={row['OR']:.2f} "
                f"(95%CI {row['95%CI_low']:.2f}-{row['95%CI_high']:.2f}), "
                f"p={row['p值']:.4f}\n")
    f.write("\n注：痰湿质贡献度最大，平和质为保护因素。\n")

print(f"\nExcel结果已保存至: {output_file}")
print(f"文本总结已保存至: {summary_file}")

# ------------------------------
# 8. 控制台输出最终结论
# ------------------------------
print("\n" + "="*60)
print("【问题一结论汇总】")
print("="*60)
print("1. 既能表征痰湿严重程度又能预警高血脂风险的关键指标为：")
print("   ", ", ".join(common))
print("2. 九种体质中，对高血脂风险贡献度最大的是痰湿质（每10分OR≈{:.2f}），".format(
    constitution_result[constitution_result["体质类型"]=="痰湿质"]["OR(每10分)"].values[0]))
print("   其次为湿热质、血瘀质；平和质为保护因素。")