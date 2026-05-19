# ==================== 第一部分：导入必需的库 ====================

# 1. 数据处理核心库
import numpy as np
import pandas as pd

# 2. 数据可视化库
import matplotlib.pyplot as plt
import seaborn as sns

# 3. 统计分析与假设检验
from scipy import stats
from scipy.stats import norm, skew, kurtosis, pearsonr, spearmanr

# 4. 数据预处理与特征工程
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# 5. 异常值检测
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest

# 6. 设置中文显示（解决绘图中文乱码问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 7. 设置绘图风格
sns.set_style("whitegrid")
sns.set_palette("Set2")

# 8. 忽略警告信息（可选）
import warnings
warnings.filterwarnings('ignore')

print("所有库导入成功！")