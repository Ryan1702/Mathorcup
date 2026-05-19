import pandas as pd

# 创建Excel写入器
with pd.ExcelWriter('MathorCup_C题_结果汇总_框架.xlsx', engine='openpyxl') as writer:
    
    # ==================== 问题1相关表格 ====================
    # 表1：痰湿质积分与各指标Spearman相关分析
    df1 = pd.DataFrame(columns=['指标', '相关系数', 'p值', '显著性'])
    df1.to_excel(writer, sheet_name='问题1_痰湿相关分析', index=False)
    
    # 表2：各指标与高血脂患病状态Spearman相关分析
    df2 = pd.DataFrame(columns=['指标', '相关系数', 'p值', '显著性'])
    df2.to_excel(writer, sheet_name='问题1_高血脂相关分析', index=False)
    
    # 表3：逻辑回归核心预警指标结果（血脂四项+血尿酸+痰湿积分）
    df3 = pd.DataFrame(columns=['变量', '系数', 'OR', 'OR_2.5%', 'OR_97.5%', 'p值', '显著性'])
    df3.to_excel(writer, sheet_name='问题1_核心预警指标回归', index=False)
    
    # 表4：九种体质多因素Logistic回归贡献度
    df4 = pd.DataFrame(columns=['体质类型', 'OR', 'OR_2.5%', 'OR_97.5%', 'p值', '显著性'])
    df4.to_excel(writer, sheet_name='问题1_九种体质贡献度', index=False)
    
    # ==================== 问题2相关表格 ====================
    # 表5：多因素风险预警模型系数（全部自变量）
    df5 = pd.DataFrame(columns=['变量', '系数', 'OR', 'OR_2.5%', 'OR_97.5%', 'p值', '显著性'])
    df5.to_excel(writer, sheet_name='问题2_预警模型系数', index=False)
    
    # 表6：个体风险分级结果
    df6 = pd.DataFrame(columns=['样本ID', '预测概率', '风险等级', '高血脂标签'])
    df6.to_excel(writer, sheet_name='问题2_个体风险分级', index=False)
    
    # 表7：高危组 vs 非高危组特征对比
    df7 = pd.DataFrame(columns=['特征', '高危组中位数', '非高危组中位数', '差异方向', 'p值', '显著性'])
    df7.to_excel(writer, sheet_name='问题2_高危vs非高危对比', index=False)
    
    # 表8：痰湿体质高危人群（痰湿积分≥60且高风险）特征描述
    df8 = pd.DataFrame(columns=['统计量', '痰湿质积分', 'BMI', 'TG', 'LDL_C', 'ADL', 'IADL', '活动总分', '年龄组', '血尿酸', '血糖'])
    # 添加常见的统计量行名
    stat_names = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    df8['统计量'] = stat_names
    df8.to_excel(writer, sheet_name='问题2_痰湿高危人群描述', index=False)
    
    # ==================== 问题3相关表格 ====================
    # 表9：所有痰湿体质患者最优干预方案汇总
    df9 = pd.DataFrame(columns=['样本ID', '痰湿积分', '年龄组', '活动总分', '中医调理等级', '活动强度等级', 
                                '每周活动次数', '6个月总成本(元)', '痰湿积分下降值', '干预后痰湿积分'])
    df9.to_excel(writer, sheet_name='问题3_痰湿体质干预方案', index=False)
    
    # 表10：样本ID 1,2,3专属方案（含体检信息）
    df10 = pd.DataFrame(columns=['样本ID', '痰湿积分', '年龄组', '活动总分', '中医调理等级', '活动强度等级',
                                 '每周活动次数', '6个月总成本(元)', '痰湿积分下降值', '干预后痰湿积分',
                                 '性别', 'BMI', 'TC', 'TG', 'LDL_C', 'HDL_C', '空腹血糖', '血尿酸'])
    df10.to_excel(writer, sheet_name='问题3_ID123专属方案', index=False)

print("Excel框架已生成：MathorCup_C题_结果汇总_框架.xlsx")
print("请手动将您的计算结果填入对应工作表中。")