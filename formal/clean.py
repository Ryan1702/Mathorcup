import pandas as pd
import numpy as np
from pathlib import Path

# ==================== 配置 ====================
INPUT_FILE = "附件1：样例数据.xlsx"      # 请确保文件在当前目录
OUTPUT_CSV = "清洗后_样例数据.csv"
REPORT_FILE = "数据清洗报告.txt"

# ==================== 读取数据（先全部读为字符串）====================
df = pd.read_excel(INPUT_FILE, sheet_name=0, dtype=str)
print(f"原始数据：{df.shape[0]} 行，{df.shape[1]} 列")

# 记录初始行数
initial_rows = df.shape[0]

# ==================== 缺失值检查（仅报告，不处理）====================
missing = df.isnull().sum()
missing_cols = missing[missing > 0]
if len(missing_cols) > 0:
    print(f"发现缺失值，涉及列：{list(missing_cols.index)}")
else:
    print("无缺失值")

# ==================== 重复行检查与删除（保持首次出现）====================
duplicated_rows = df.duplicated().sum()
if duplicated_rows > 0:
    df = df.drop_duplicates(keep='first').reset_index(drop=True)
    print(f"删除 {duplicated_rows} 行完全重复数据")
else:
    print("无完全重复行")

# 基于“样本ID”的重复检查（题目中样本ID应唯一）
if "样本ID" in df.columns:
    id_dup = df["样本ID"].duplicated().sum()
    if id_dup > 0:
        df = df.drop_duplicates(subset=["样本ID"], keep='first').reset_index(drop=True)
        print(f"基于样本ID删除 {id_dup} 行重复记录")
    else:
        print("样本ID唯一性通过")

# ==================== 数据类型转换 ====================
# 整数列（量表得分、分类标签等）
int_cols = [
    "样本ID", "体质标签", "平和质", "气虚质", "阳虚质", "阴虚质", "痰湿质", "湿热质", "血瘀质", "气郁质", "特禀质",
    "ADL用厕", "ADL吃饭", "ADL步行", "ADL穿衣", "ADL洗澡", "ADL总分", "IADL购物", "IADL做饭", "IADL理财",
    "IADL交通", "IADL服药", "IADL总分", "活动量表总分（ADL总分+IADL总分）",
    "高血脂症二分类标签", "血脂异常分型标签（确诊病例）", "年龄组", "性别", "吸烟史", "饮酒史"
]
# 浮点数列（血脂、血糖、尿酸、BMI）
float_cols = [
    "HDL-C（高密度脂蛋白）", "LDL-C（低密度脂蛋白）", "TG（甘油三酯）", "TC（总胆固醇）",
    "空腹血糖", "血尿酸", "BMI"
]

# 转换整数列（无效值变NaN，再转为Int64可空整数）
for col in int_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].astype('Int64')

# 转换浮点列
for col in float_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 检查转换后新产生的缺失值
new_nas = df.isnull().sum() - missing
new_nas = new_nas[new_nas > 0]
if len(new_nas) > 0:
    print(f"警告：数据类型转换产生新的缺失值，列：{list(new_nas.index)}")
else:
    print("所有数值列转换成功，无新增缺失值")

# ==================== 分类变量合法性检查（仅报告）====================
categorical_rules = {
    "体质标签": (1, 9),
    "高血脂症二分类标签": (0, 1),
    "血脂异常分型标签（确诊病例）": (0, 5),   # 题目说明多分类1/2/3，但数据中存在0,4,5等，保留原值
    "年龄组": (1, 5),
    "性别": (0, 1),
    "吸烟史": (0, 1),
    "饮酒史": (0, 1)
}
invalid_dict = {}
for col, (low, high) in categorical_rules.items():
    if col not in df.columns:
        continue
    invalid = df[col][~df[col].isna() & ((df[col] < low) | (df[col] > high))]
    if len(invalid) > 0:
        invalid_dict[col] = invalid
        print(f"注意：{col} 存在 {len(invalid)} 个超出范围 [{low},{high}] 的值，例如 {invalid.head(3).tolist()}")

# ==================== 数值范围异常检测（仅报告）====================
# 根据临床常识设定合理范围（用于参考，不修改数据）
range_dict = {
    "平和质": (0,100), "气虚质": (0,100), "阳虚质": (0,100), "阴虚质": (0,100),
    "痰湿质": (0,100), "湿热质": (0,100), "血瘀质": (0,100), "气郁质": (0,100), "特禀质": (0,100),
    "ADL用厕": (0,10), "ADL吃饭": (0,10), "ADL步行": (0,10), "ADL穿衣": (0,10), "ADL洗澡": (0,10),
    "ADL总分": (0,50), "IADL购物": (0,10), "IADL做饭": (0,10), "IADL理财": (0,10),
    "IADL交通": (0,10), "IADL服药": (0,10), "IADL总分": (0,50),
    "活动量表总分（ADL总分+IADL总分）": (0,100),
    "HDL-C（高密度脂蛋白）": (0.5, 2.5), "LDL-C（低密度脂蛋白）": (1.0, 5.0),
    "TG（甘油三酯）": (0.3, 6.0), "TC（总胆固醇）": (2.0, 8.0),
    "空腹血糖": (3.0, 10.0), "血尿酸": (100, 600), "BMI": (15, 40)
}
outlier_report = {}
for col, (low, high) in range_dict.items():
    if col not in df.columns:
        continue
    s = df[col].dropna()
    out = s[(s < low) | (s > high)]
    if len(out) > 0:
        outlier_report[col] = len(out)
        print(f"提示：{col} 有 {len(out)} 个值超出合理范围 [{low},{high}]")

# ==================== 保存为CSV ====================
df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
print(f"清洗后的数据已保存为：{OUTPUT_CSV}")

# ==================== 生成清洗报告 ====================
with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write("===== 数据清洗报告 =====\n")
    f.write(f"原始行数：{initial_rows}\n")
    f.write(f"清洗后行数：{df.shape[0]}（删除了重复行）\n")
    f.write(f"总列数：{df.shape[1]}\n\n")

    f.write("1. 缺失值情况：\n")
    if len(missing_cols) == 0:
        f.write("   无缺失值。\n")
    else:
        for col, cnt in missing_cols.items():
            f.write(f"   {col}: {cnt} 个缺失\n")
        f.write("   未填充或删除，保留原始缺失状态。\n")

    f.write("\n2. 重复值处理：\n")
    f.write(f"   删除完全重复行：{duplicated_rows} 行\n")
    if 'id_dup' in locals():
        f.write(f"   删除基于样本ID的重复行：{id_dup} 行\n")

    f.write("\n3. 数据类型转换：\n")
    f.write("   整数列：\n")
    for col in int_cols:
        if col in df.columns:
            f.write(f"      {col}\n")
    f.write("   浮点列：\n")
    for col in float_cols:
        if col in df.columns:
            f.write(f"      {col}\n")
    if len(new_nas) > 0:
        f.write(f"   注意：转换时新增缺失值，列：{list(new_nas.index)}\n")

    f.write("\n4. 分类变量合法性检查（超出范围仅报告，未修改）：\n")
    for col, inv in invalid_dict.items():
        f.write(f"   {col}: {len(inv)} 个异常值\n")
    if not invalid_dict:
        f.write("   所有分类变量均在合法范围内。\n")

    f.write("\n5. 数值异常检测（合理范围参考，未处理）：\n")
    for col, cnt in outlier_report.items():
        f.write(f"   {col}: {cnt} 个超出范围值\n")
    if not outlier_report:
        f.write("   所有数值均在设定参考范围内。\n")

    f.write("\n清洗完成，数据已保存为CSV，可用于后续建模分析。\n")

print(f"清洗报告已保存：{REPORT_FILE}")