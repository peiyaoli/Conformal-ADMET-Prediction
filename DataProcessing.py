import os
import pandas as pd

# 读取IVIT.xlsx表，如果不存在则创建一个新的表

# 遍历文件夹中的csv文件
folder_path = '3fold/'
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)

        # 从文件名中提取相关信息
        parts = filename.split('_')
        if parts[0]=='Permeability':
            Dataset = parts[0]+'_'+parts[1]+'_'+parts[2]
            method = parts[4]
            fold = int(parts[6])
            split = parts[3]
        else:
            split = parts[2]
            method = parts[3]
            Dataset = parts[0]+'_'+parts[1]
            fold = int(parts[5])

        # 读取文件的第二行数据
        with open(file_path, 'r') as file:
            data = file.readlines()[1].strip()[2:]
        data_values = data.split(',')
        data = [float(i) for i in data_values]

        if split == 'IVIT':
            excel_file = '汇总\IVIT.xlsx'
        elif split == 'OVOT':
            excel_file = '汇总\OVOT.xlsx'
        else:
            excel_file = '汇总\IVOT.xlsx'
        # 读取表
        df = pd.read_excel(excel_file)

        # 找到对应的行，并在后面写入数据
        condition = (df['数据集']==Dataset) & (df['方法']==method)
        row_index = df.index[condition].tolist()[fold]  

        for i, value in enumerate(data):
            df.at[row_index, df.columns[i + 2]] = value

        # 保存更新后的表
        df.to_excel(excel_file, index=False)
