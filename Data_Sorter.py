import numpy as np
import pandas as pd
import csv 
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Data_Sorter():
    def __init__(self, question=1):
        self.question = question
        if self.question == 1:
            self.initpath = os.path.join("C:\\Users\\oft\\Documents\\ShenZhenCup\\output", 'Q1')
            self.fontsize = 5
            self.yscale = (-200000, 500000)
            self.name_converter = {
                'Q1v0-b2_1' : 'BGA0.5mm分割',
                'Q1v1-b3_2' : 'BGA0.5mm分割',
                'Q1v2-b4_3' : 'BGA0.5mm分割',
                'Q1v3-m0.7' : 'BGA0.7mm分割',
                'Q1v4-m0.3' : 'BGA0.3mm分割',
                'Q1v5-m1' : 'BGA1mm分割',
                'Q1v6-b1_1': 'PCB1mm分割-BGA0.5mm分割',
                'Q1v7-m0.4': 'BGA0.4mm分割',
                'Q1v8-s0.2': 'PCB4mm分割-BGA0.3mm分割-焊球0.2分割',
                'Q1v9-s0.1': 'PCB4mm分割-BGA0.3mm分割-焊球0.1分割',
                'Q1v10-s0.3': 'PCB4mm分割-BGA0.3mm分割-焊球0.3分割',
                'Q1-m1': 'BGA1mm分割',
                'Q1-m0.7': 'BGA0.7mm分割',
                'Q1-m0.5': 'BGA0.5mm分割',
                'Q1-m0.2': 'BGA0.2mm分割',
                'Q1-m2.5': 'BGA2.5mm分割',
                'Q1-m2': 'BGA2mm分割',
                'Q1-m1.5': 'BGA1.5mm分割',
                'Q1-m0.3': 'BGA0.3mm分割'
            }
            self.path_names = ['BGA2mm高度处对角线线路', 'BGA1.5mm高度处对角线线路', 
                'BGA1mm高度处对角线线路', 'BGA0.5mm高度处对角线线路', 'BGA角点处2mm至0mm竖直线路']
            self.exchange_char = {'h0':'-height-2mm',
                    'h1':'-height-1.5mm',
                    'h2':'-height-1mm',
                    'h3':'-height-0.5mm',
                    'v0':'-height-0mm_to_2mm'}
            self.CTE_title = 'Q1CTE热膨胀系数在不同BGA高度上以及角点处竖直线路上的变化'
            self.ela_title = 'Q1拉伸模量在不同BGA高度上以及角点处竖直线路上的变化'
            self.height_strs_cte = ['CTE-Path-height-2mm', 'CTE-Path-height-1.5mm', 
                'CTE-Path-height-1mm', 'CTE-Path-height-0.5mm', 'CTE-Path-height-0mm_to_2mm']
            self.height_strs_ela = ['ela-Path-height-2mm', 'ela-Path-height-1.5mm', 
                    'ela-Path-height-1mm', 'ela-Path-height-0.5mm', 'ela-Path-height-0mm_to_2mm']
            
        elif self.question ==2:
            self.initpath = os.path.join("C:\\Users\\oft\\Documents\\ShenZhenCup\\output", 'Q2')
            self.fontsize = 4
            self.yscale = (-200000, 300000)
            self.name_converter = {
                'Q2v0-4' : '芯片4mm精度分割',
                'Q2v0-3.5' : '芯片3.5mm精度分割',
                'Q2v0-3' : '芯片3mm精度分割',
                'Q2v0-2.5' : '芯片2.5mm精度分割',
                'Q2v0-2' : '芯片2mm精度分割',
                'Q2v0-1' : '芯片1mm精度分割',
                'Q2v0-1.5' : '芯片1.5mm精度分割',
                'Q2v0-0.5' : '芯片0.5mm精度分割',
                'Q2v0-0.2' : '芯片0.2mm精度分割',
                'Q2v0-0.1' : '芯片0.1mm精度分割',
                'Q2v0-0.09' : '芯片0.09mm精度分割',
                'Q2v0-0.08' : '芯片0.08mm精度分割',
                'Q2v0-0.07' : '芯片0.07mm精度分割',
                'Q2v0-tmp' : '芯片0.1mm精度分割'
            }
            self.path_names = ['芯片3.57mm高度处对角线线路', '芯片2.67mm高度处对角线线路', 
                '芯片1.77mm高度处对角线线路', '芯片0.87mm高度处对角线线路', '芯片角点处3.57mm至0mm竖直线路']
            self.exchange_char = {
                    'h0':'-height-3.57mm',
                    'h1':'-height-2.67mm',
                    'h2':'-height-1.77mm',
                    'h3':'-height-0.87mm',
                    'v0':'-height-3.57mm_to_0mm'}
            self.CTE_title = f'Q{self.question}' + 'CTE热膨胀系数在不同芯片高度上以及角点处竖直线路上的变化'
            self.ela_title = f'Q{self.question}' + '拉伸模量在不同芯片高度上以及角点处竖直线路上的变化'
            self.height_strs_cte = ['CTE-Path-height-3.57mm', 'CTE-Path-height-2.67mm', 
                'CTE-Path-height-1.77mm', 'CTE-Path-height-0.87mm', 'CTE-Path-height-3.57mm_to_0mm']
            self.height_strs_ela = ['ela-Path-height-3.57mm', 'ela-Path-height-2.67mm', 
                    'ela-Path-height-1.77mm', 'ela-Path-height-0.87mm', 'ela-Path-height-3.57mm_to_0mm']

        elif self.question == 3:
            self.initpath = os.path.join("C:\\Users\\oft\\Documents\\ShenZhenCup\\output", 'Q3')
            self.fontsize = 4
            self.yscale = (-200000, 300000)
            self.name_converter = {
                'Q3-4': '芯片4mm精度分割',
                'Q3-5': '芯片5mm精度分割',
                'Q3-3': '芯片3mm精度分割',
                'Q3-2': '芯片2mm精度分割',
                'Q3-1.5': '芯片1.5mm精度分割',
                'Q3-0.3': '芯片0.3mm精度分割',
                'Q3-1': '芯片1mm精度分割',
                'Q3-0.5': '芯片0.5mm精度分割',
                'Q3-0.4': '芯片0.4mm精度分割',
                'Q3-0.2': '芯片0.2mm精度分割',
                'Q3-1.3': '芯片1.3mm精度分割'
            }
            self.path_names = ['芯片1.97mm高度处对角线线路', '芯片1.5mm高度处对角线线路', 
                '芯片1.3mm高度处对角线线路', '芯片焊球角点处1.97mm至0mm竖直线路',
                '芯片无焊球角点处1.97mm至0mm竖直线路']
            self.exchange_char = {
                    'h0':'-height-1.97mm',
                    'h1':'-height-1.5mm',
                    'h2':'-height-1.3mm',
                    'v0':'-height-1.97mm_to_0mm_焊球端',
                    'v1':'-height-1.97mm_to_0mm_无焊球端'}
            self.CTE_title = f'Q{self.question}' + 'CTE热膨胀系数在不同芯片高度上以及角点处竖直线路上的变化'
            self.ela_title = f'Q{self.question}' + '拉伸模量在不同芯片高度上以及角点处竖直线路上的变化'
            self.height_strs_cte = ['CTE-Path-height-1.97mm', 'CTE-Path-height-1.5mm', 
                'CTE-Path-height-1.3mm', 'CTE-Path-height-1.97mm_to_0mm_焊球端', 'CTE-Path-height-1.97mm_to_0mm_无焊球端']
            self.height_strs_ela = ['ela-Path-height-1.97mm', 'ela-Path-height-1.5mm', 
                'ela-Path-height-1.3mm', 'ela-Path-height-1.97mm_to_0mm_焊球端', 'ela-Path-height-1.97mm_to_0mm_无焊球端']
            
    def process_folder(self, folder_Path, duty):
        if not os.path.exists(folder_Path):
            print(f"Not find {folder_Path}, skip it")
            return

        data = {}
        if os.path.isdir(folder_Path):
            names = [x for x in os.listdir(folder_Path) if x.startswith(duty)]
        else:
            print(f'{folder_Path} is not a folder')
            return None
        
        for name in names:
            name_dict = name.split('.')[0]
            name_dict = name_dict[:-2] + self.exchange_char[name_dict[-2:]]
            data[name_dict] = []
            path = os.path.join(folder_Path, name)
            with open(path, 'r') as file:
                csv_reader = csv.reader(file, delimiter='\t')
                next(csv_reader)
                for row in csv_reader:
                    data[name_dict].append(row[-1])
        
        return pd.DataFrame(data, dtype=np.float64)

    def to_xlsx(self, input_folder, output_file):
        for folder_name in os.listdir(input_folder):
            folder_Path = os.path.join(input_folder, folder_name)
            tem_str = folder_name.split('-')[-1].split()[0][:4]
            cte_output = self.process_folder(folder_Path, 'CTE')
            ela_output = self.process_folder(folder_Path, 'ela')

            if cte_output is None or ela_output is None:
                continue
            try:
                with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                    cte_output.to_excel(writer, sheet_name='CTE-'+tem_str+'摄氏度', index=False)
                    ela_output.to_excel(writer, sheet_name='ela-'+tem_str+'摄氏度', index=False)
            except FileNotFoundError:
                with pd.ExcelWriter(output_file, engine='openpyxl', mode='w') as writer:
                    cte_output.to_excel(writer, sheet_name='CTE-'+tem_str+'摄氏度', index=False)
                    ela_output.to_excel(writer, sheet_name='ela-'+tem_str+'摄氏度', index=False)

    def to_total_dataframe(self, input_folder):    
        CTE_LIST, ELA_LIST = {}, {}
        for folder_name in os.listdir(input_folder):
            folder_Path = os.path.join(input_folder, folder_name)
            tem_str = folder_name.split('-')[-1].split()[0][:4]
            cte_output = self.process_folder(folder_Path, 'CTE')
            ela_output = self.process_folder(folder_Path, 'ela')
            if cte_output is None or ela_output is None:
                continue

            CTE_LIST[tem_str] = cte_output
            ELA_LIST[tem_str] = ela_output
        return CTE_LIST, ELA_LIST

    def pathdata2variance(self, path_data):
        variance = np.var(path_data)
        squr_str = f"方差: {variance:.2e}"
        if np.isnan(variance):
            pass 
        return squr_str

    def draw_result(self, results, duty):
        if self.question ==2:
            dir_names = ['4mm网格', '3.5mm网格', '3mm网格', '2.5mm网格', '2mm网格',
                         '1.5mm网格', '1mm网格', '0.5mm网格', '0.2mm网格', '0.1mm网格',
                         '0.09mm网格', '0.08mm网格', '0.07mm网格']
        elif self.question == 3:
            dir_names = ['5mm网格', '4mm网格', '3mm网格', '2mm网格', '1mm网格', 
                         '0.5mm网格', '0.4mm网格', '0.3mm网格', '0.2mm网格']
            labels = ['焊球端', '无焊球端']
        elif self.question == 1:
            dir_names = ['2.5mm网格', '2mm网格', '1.5mm网格', 
                         '1mm网格', '0.7mm网格', '0.5mm网格', '0.3mm网格', '0.2mm网格']
            
        plt.rcParams.update({'font.size':7,
                            'font.family':'SimHei',
                            'axes.unicode_minus':False})
        FIG = plt.figure(dpi=800, figsize=(8, 6))
        axe = FIG.subplots(1, 1)
        for i, result in enumerate(results):
            if self.question == 3:
                axe.plot(dir_names, result, label=labels[i])
                axe.legend(loc='upper right', fontsize=8)
            else:
                axe.plot(dir_names, result)
        FIG.suptitle(f'Q{self.question}网格细分程度稳定性检验', fontsize=12)
        if duty == 'CTE':
            FIG.supylabel('热膨胀系数', fontsize=12)
        else:
            FIG.supylabel('拉伸模量', fontsize=12)
        FIG.supxlabel('不同网格大小', fontsize=12)
        FIG.savefig(os.path.join(self.initpath, duty+'不同网格细化下的最终角点结果.jpg'))

    def avertem_diffmiss(self, dir_names, duty, step, shareY=False, scale=False):
        plt.rcParams.update({'font.size':7,
                            'font.family':'SimHei',
                            'axes.unicode_minus':False})
        colors = cm.coolwarm(np.linspace(0, 1, len(dir_names)))

        def draw(axes, result_list, name, name_id, global_handles, global_labels, local_handles, local_labels):
            for i, (_, path_data) in enumerate(result_list.items()):
                if i == 0:
                    line, = axes.flat[i].plot(path_data, color=colors[name_id])
                    global_labels.append(self.name_converter[name])
                    global_handles.append(line)
                else:
                    line, = axes.flat[i].plot(path_data, color=colors[name_id])
                local_handles[i].append(line)
                local_labels[i].append(self.pathdata2variance(path_data))
                axes.flat[i].set_title(self.path_names[i])
                axes.flat[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                if scale:
                    axes.flat[i].set_ylim(*self.yscale)
            for i in range(len(result_list)):
                axes.flat[i].legend(local_handles[i], local_labels[i], fontsize=self.fontsize, loc='lower left')

            return global_handles, global_labels, local_handles, local_labels
        
        FIG = plt.figure(dpi=800, figsize=(8, 6))
        axes = FIG.subplots(2, 3, sharey=shareY, sharex=True)
        axes[1, 2].remove()

        dir_list = [(x, os.path.join(self.initpath, x)) for x in dir_names if os.path.isdir(os.path.join(self.initpath, x))]
        global_handles, global_labels = [], []
        local_handles, local_labels = [[] for _ in range(5)], [[] for _ in range(5)]
        result1, result2 = [], []
        for name_id, (name, dir) in enumerate(dir_list):
            if duty =='CTE':
                LIST, _ = self.to_total_dataframe(dir)
            else:
                _, LIST = self.to_total_dataframe(dir)

            result_list = {}
            for i, (_, output) in enumerate(LIST.items()):
                for path_name, path_data in output.items():
                    if i == 0:
                        result_list[path_name] = path_data / len(LIST)
                    else:
                        result_list[path_name] += path_data / len(LIST)
            global_handles, global_labels, local_handles, local_labels= draw(axes, result_list, name, name_id, global_handles, global_labels, local_handles, local_labels)

            if duty == 'CTE':
                result_1 = result_list[self.height_strs_cte[0]]
            else:
                result_1 = result_list[self.height_strs_ela[0]]                  
            if self.question == 3:
                if duty == 'ela':
                    result1.append(result_1.iloc[3]); result2.append(result_1.iloc[-4])
                else:
                    result1.append(result_1.iloc[0]); result2.append(result_1.iloc[-1])
            else:
                if self.question == 1 and duty == 'ela':
                    result1.append(result_1.iloc[2])
                else:
                    result1.append(result_1.iloc[0])

        if self.question == 3:
            self.draw_result([result1, result2], duty)
        else:
            self.draw_result([result1], duty)

        FIG.legend(global_handles, global_labels, bbox_to_anchor=(0.9, 0.4), fontsize=8, ncol=1)
        if duty == 'CTE':
            FIG.supylabel('CTE热膨胀系数', fontsize=12)
            FIG.suptitle('不同网格细化程度下,' + self.CTE_title, fontsize=12)
        else:
            FIG.supylabel('拉伸模量', fontsize=12)
            FIG.suptitle('不同网格细化程度下,' + self.ela_title, fontsize=12)
        FIG.supxlabel('节点序号', fontsize=12)

        output_name = f'Q{self.question}-' + duty
        if shareY:
            output_name += '-shareY'
        if scale:
            output_name += '-scale'
        output_name += f'-STEP{step}'
        FIG.savefig(os.path.join(self.initpath, output_name+'.jpg'))
            
    def draw_from_dataframe(self, LIST, name, output_path, scale=False, shareY=False):
        color_len = len(LIST)
        list_ordered = [(float(tem), tem, array) for tem, array in LIST.items()]
        list_ordered.sort(key=lambda x: x[0])

        colors = cm.coolwarm(np.linspace(0, 1, color_len))
        plt.rcParams.update({'font.size':7,
                            'font.family':'SimHei',
                            'axes.unicode_minus':False})
        FIG = plt.figure(figsize=(8,6), dpi=800)
        axes = FIG.subplots(2, 3, sharex=True, sharey=shareY)

        if name == 'ela':
            height_strs = self.height_strs_ela
        else:
            height_strs = self.height_strs_cte

        local_handles, local_labels = [[] for _ in range(5)], [[] for _ in range(5)]    
        for j, (_, tem_str, output) in enumerate(list_ordered):
            for i in range(len(height_strs)):
                if i == 0 :
                    line, = axes.flat[i].plot(output[height_strs[i]], label=f'Tem:{tem_str}℃', color=colors[j])
                else:
                    line, = axes.flat[i].plot(output[height_strs[i]], color=colors[j])
                axes.flat[i].set_title(self.path_names[i])
                axes.flat[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                local_handles[i].append(line)
                local_labels[i].append(self.pathdata2variance(output[height_strs[i]]))
                if scale:
                    axes.flat[i].set_ylim(self.yscale)
        for i in range(len(height_strs)):
            axes.flat[i].legend(local_handles[i], local_labels[i], fontsize=3, loc='upper right', ncol=2)
        axes[1, 2].remove()

        FIG.legend(bbox_to_anchor=(0.95, 0.4), fontsize=8, ncol=2)
        FIG.supxlabel('节点序号', fontsize=12)
        if name == 'CTE':
            FIG.supylabel('CTE热膨胀系数', fontsize=12)
            FIG.suptitle(self.CTE_title, fontsize=12)
        else:
            FIG.supylabel('拉伸模量', fontsize=12)
            FIG.suptitle(self.ela_title, fontsize=12)
        
        output_name = name
        if scale:
            output_name += '-scale'
        if shareY:
            output_name += '-sharey'
        FIG.savefig(os.path.join(output_path, output_name + '.jpg'))

    def add_xlsx_photo(self, dir_names):
        output_xlsx_name = f'Q{self.question}_CTE_ela.xlsx'
        for dir_name in dir_names:
            dir_path = os.path.join(self.initpath, dir_name)
            self.to_xlsx(dir_path, os.path.join(dir_path, output_xlsx_name))
            CTE_LIST, ELA_LIST = self.to_total_dataframe(dir_path)
            self.draw_from_dataframe(CTE_LIST, 'CTE', dir_path, shareY=True)
            self.draw_from_dataframe(ELA_LIST, 'ela', dir_path, scale=True)
            self.draw_from_dataframe(CTE_LIST, 'CTE', dir_path)
            self.draw_from_dataframe(ELA_LIST, 'ela', dir_path)

    def get_from_difftems(self, folder_name, duty):
        folder_path = os.path.join(self.initpath, folder_name)
        if duty == 'CTE':
            LIST, _ = self.to_total_dataframe(folder_path)
        else:
            _, LIST = self.to_total_dataframe(folder_path)

        output_dataf = {}
        tems = []
        for tem_str, output in LIST.items():
            tems.append(float(tem_str))
            if duty == 'CTE':
                output_dataf[tem_str+'摄氏度'] = [output[self.height_strs_cte[0]].iloc[0]]
            else:
                if self.question == 1:
                    output_dataf[tem_str+'摄氏度'] = [output[self.height_strs_ela[0]].iloc[1]] 
                else:
                    output_dataf[tem_str+'摄氏度'] = [output[self.height_strs_ela[0]].iloc[0]] 
        
        output_dataf = pd.DataFrame(output_dataf)
        output_dataf.to_excel(os.path.join(self.initpath, f'Q{self.question}不同温度拉伸模量.xlsx'),
                                index=None, engine='openpyxl', 
                                sheet_name='Q'+str(self.question))

def Q1():
    Data_Sorter1 = Data_Sorter(question=1)
    # dir_names_list = [['Q1-m2.5', 'Q1-m2', 'Q1-m1.5', 'Q1-m1', 'Q1-m0.7', 'Q1-m0.5', 'Q1-m0.3', 'Q1-m0.2']]
    # for step, dir_names in enumerate(dir_names_list):
    #     Data_Sorter1.avertem_diffmiss(dir_names, 'CTE', step, shareY=True)
    #     Data_Sorter1.avertem_diffmiss(dir_names, 'ela', step, scale=True)
    dir_names = ['Q1-m0.2']
    Data_Sorter1.add_xlsx_photo(dir_names)
    # Data_Sorter1.get_from_difftems(folder_name='Q1-m0.3', duty='ela')

def Q2():
    Data_Sorter1 = Data_Sorter(question=2)
    # dir_names_list = [['Q2v0-4', 'Q2v0-3.5', 'Q2v0-3', 'Q2v0-2.5', 'Q2v0-2', 'Q2v0-1.5', 
    #                    'Q2v0-1', 'Q2v0-0.5', 'Q2v0-0.2', 'Q2v0-0.1', 'Q2v0-0.09', 'Q2v0-0.08',
    #                    'Q2v0-0.07']]
    # for step, dir_names in enumerate(dir_names_list):
    #     Data_Sorter1.avertem_diffmiss(dir_names, 'CTE', step, shareY=True)
    #     Data_Sorter1.avertem_diffmiss(dir_names, 'ela', step, scale=True)
    dir_names = ['Q2v0-0.07']
    Data_Sorter1.add_xlsx_photo(dir_names)
    # Data_Sorter1.get_from_difftems(folder_name='Q2v0-0.2', duty='ela')

def Q3():
    Data_Sorter1 = Data_Sorter(question=3)
    # dir_names_list = [['Q3-5', 'Q3-4', 'Q3-3', 'Q3-2', 'Q3-1', 
    #                    'Q3-0.5', 'Q3-0.4', 'Q3-0.3', 'Q3-0.2']]
    # for step, dir_names in enumerate(dir_names_list):
    #     Data_Sorter1.avertem_diffmiss(dir_names, 'CTE', step, shareY=True)
    #     Data_Sorter1.avertem_diffmiss(dir_names, 'ela', step, scale=True)
    dir_names = ['Q3-0.2']
    Data_Sorter1.add_xlsx_photo(dir_names)
    # Data_Sorter1.get_from_difftems(folder_name='Q3-0.3', duty='ela')
Q1()