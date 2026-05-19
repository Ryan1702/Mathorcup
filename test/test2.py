from test1 import *
# ==================== 第二部分：数据读取 ====================

def load_data(file_path, file_type='csv', **kwargs):
    """
    通用数据加载函数
    
    参数:
        file_path: 文件路径
        file_type: 文件类型 ('csv', 'excel', 'txt')
        **kwargs: 传递给读取函数的额外参数
    
    返回:
        df: 加载的DataFrame对象
    """
    if file_type == 'csv':
        df = pd.read_csv(file_path, **kwargs)
    elif file_type == 'excel':
        df = pd.read_excel(file_path, **kwargs)
    elif file_type == 'txt':
        df = pd.read_csv(file_path, sep='\t', **kwargs)
    else:
        raise ValueError("不支持的file_type,请使用 'csv', 'excel' 或 'txt'")
    
    print(f"✅ 数据加载成功！")
    print(f"   文件路径: {file_path}")
    print(f"   数据形状: {df.shape[0]}行 × {df.shape[1]}列")
    return df

# 使用示例
# df = load_data('data/stroke.csv', file_type='csv')
# df = load_data('data/dataset.xlsx', file_type='excel')

# 如果数据不在当前目录，需要指定完整路径
# df = load_data('/path/to/your/data/stroke.csv')