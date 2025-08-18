import pandas as pd
import os

def get_diff(xlsx_paths, output_path=r'C:\Users\oft\Documents\ShenZhenCup\output'):
    for xlsx_path in xlsx_paths:
        xlsx_name = xlsx_path.split('\\')[-1].split('_')[0]
        output_dataf = {}
        excel_data = pd.read_excel(io=xlsx_path, sheet_name=None)
        excel_data = sort_dataf(excel_data)

        cte, ela1, ela2, tem_strs = [], [], [], []
        for sheet_name, df in excel_data.items():
            if 'CTE' in sheet_name:
                cte.append(get_value(df, question=xlsx_name, duty='CTE'))
                tem_strs.append(sheet_name[4:])
            elif 'ela' in sheet_name:
                if xlsx_name == 'Q3':
                    ela1.append(get_value(df, question=xlsx_name, duty='ela', row=3))
                    ela2.append(get_value(df, question=xlsx_name, duty='ela', row=-4))
                else:
                    ela1.append(get_value(df, question=xlsx_name, duty='ela'))

        output_dataf['温度'] = tem_strs
        output_dataf['热膨胀系数'] = cte
        if xlsx_name == 'Q3':
            output_dataf['焊球端拉伸模量'] = ela1
            output_dataf['无焊球端拉伸模量'] = ela2
        else:
            output_dataf['拉伸模量'] = ela1
        output_dataf = pd.DataFrame(output_dataf)

        try:
            with pd.ExcelWriter(path=os.path.join(output_path, '三问随温度变化仿真结果.xlsx'), engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                output_dataf.to_excel(excel_writer=writer, sheet_name=xlsx_name, index=False)
        except FileNotFoundError:
            with pd.ExcelWriter(path=os.path.join(output_path, '三问随温度变化仿真结果.xlsx'), engine='openpyxl', mode='w') as writer:
                output_dataf.to_excel(excel_writer=writer, sheet_name=xlsx_name, index=False)

def get_value(input_dataframe, duty, question, row=0):
    if duty == 'ela' and question == 'Q1':
        row = 2

    return input_dataframe.iloc[row, 0]

def sort_dataf(input_dataf):
    output_dataf = {}
    
    dataf_list = [(float(tem_str[4:8]), tem_str, fr) for tem_str, fr in input_dataf.items()]
    dataf_list = [x for x in dataf_list if 25 < x[0] <= 100]
    dataf_list.sort(key=lambda x: x[0])
    for _, tem_str, fr in dataf_list:
        output_dataf[tem_str] = fr

    return output_dataf

# 调用函数
get_diff([r"C:\Users\oft\Documents\ShenZhenCup\output\Q1\Q1-m0.2\Q1_CTE_ela.xlsx",
          r"C:\Users\oft\Documents\ShenZhenCup\output\Q2\Q2v0-0.07\Q2_CTE_ela.xlsx",
          r"C:\Users\oft\Documents\ShenZhenCup\output\Q3\Q3-0.2\Q3_CTE_ela.xlsx"])