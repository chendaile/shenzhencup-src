import numpy as np
import pandas as pd
import csv 
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path

@dataclass
class QuestionConfig:
    """Configuration for each question type"""
    question_id: int
    output_folder: str
    fontsize: int
    yscale: Tuple[float, float]
    name_converter: Dict[str, str]
    path_names: List[str]
    exchange_char: Dict[str, str]
    height_strs_cte: List[str]
    height_strs_ela: List[str]
    grid_names: List[str]  # 添加网格名称配置
    duty_names: dict = field(default_factory=lambda: {
            'CTE': '热膨胀系数',
            'ela': '拉伸模量',
            'Ther': '热应变',
            'stress': '对角线方向应力',
            'Temp': '温度',
            'strain': '对角线方向应变'
        })

    @property
    def cte_title(self) -> str:
        return f'Q{self.question_id}CTE热膨胀系数在不同高度上以及角点处竖直线路上的变化'
    
    @property
    def ela_title(self) -> str:
        return f'Q{self.question_id}拉伸模量在不同高度上以及角点处竖直线路上的变化'

class ConfigManager:
    """Manages configurations for different questions"""
    
    @staticmethod
    def get_config(question: int, base_path: str = "C:\\Users\\oft\\Documents\\ShenZhenCup\\output") -> QuestionConfig:
        """Get configuration for specified question"""
        configs = {
            1: ConfigManager._get_q1_config(base_path),
            2: ConfigManager._get_q2_config(base_path),
            3: ConfigManager._get_q3_config(base_path)
        }
        
        if question not in configs:
            raise ValueError(f"Question {question} not supported. Available: {list(configs.keys())}")
        
        return configs[question]
    
    @staticmethod
    def _get_q1_config(base_path: str) -> QuestionConfig:
        return QuestionConfig(
            question_id=1,
            output_folder=os.path.join(base_path, 'Q1'),
            fontsize=5,
            yscale=(-200000, 500000),
            name_converter={
                'Q1-1': 'BGA1mm分割',
                'Q1-2': 'BGA2mm分割',
                'Q1-3': 'BGA3mm分割',
                'Q1-2.5': 'BGA2.5mm分割'               
            },
            path_names=[
                'BGA2mm高度处对角线线路', 
                'BGA1.5mm高度处对角线线路',
                'BGA1mm高度处对角线线路', 
                'BGA0.5mm高度处对角线线路', 
                'BGA角点处2mm至0mm竖直线路'
            ],
            exchange_char={
                'h0': '-height-2mm',
                'h1': '-height-1.5mm',
                'h2': '-height-1mm',
                'h3': '-height-0.5mm',
                'v0': '-height-2mm_to_0mm'
            },
            height_strs_cte=[
                'CTE-Path-height-2mm', 
                'CTE-Path-height-1.5mm',
                'CTE-Path-height-1mm', 
                'CTE-Path-height-0.5mm', 
                'CTE-Path-height-2mm_to_0mm'
            ],
            height_strs_ela=[
                'ela-Path-height-2mm', 
                'ela-Path-height-1.5mm',
                'ela-Path-height-1mm', 
                'ela-Path-height-0.5mm', 
                'ela-Path-height-2mm_to_0mm'
            ],
            grid_names=['3mm网格', '2.5mm网格']
        )
    
    @staticmethod
    def _get_q2_config(base_path: str) -> QuestionConfig:
        return QuestionConfig(
            question_id=2,
            output_folder=os.path.join(base_path, 'Q2'),
            fontsize=5,
            yscale=(-200000, 300000),
            name_converter={
                'Q2-3': '芯片4mm精度分割',
                'Q2-2.5': '芯片3.5mm精度分割',
                'Q2-2': '芯片3mm精度分割'
            },
            path_names=[
                '芯片3.57mm高度处对角线线路', 
                '芯片2.67mm高度处对角线线路',
                '芯片1.77mm高度处对角线线路', 
                '芯片0.87mm高度处对角线线路', 
                '芯片角点处3.57mm至0mm竖直线路'
            ],
            exchange_char={
                'h0': '-height-3.57mm',
                'h1': '-height-2.67mm',
                'h2': '-height-1.77mm',
                'h3': '-height-0.87mm',
                'v0': '-height-3.57mm_to_0mm'
            },
            height_strs_cte=[
                'CTE-Path-height-3.57mm', 
                'CTE-Path-height-2.67mm',
                'CTE-Path-height-1.77mm', 
                'CTE-Path-height-0.87mm', 
                'CTE-Path-height-3.57mm_to_0mm'
            ],
            height_strs_ela=[
                'ela-Path-height-3.57mm', 
                'ela-Path-height-2.67mm',
                'ela-Path-height-1.77mm', 
                'ela-Path-height-0.87mm', 
                'ela-Path-height-3.57mm_to_0mm'
            ],
            grid_names=['4mm网格', '3.5mm网格', '3mm网格', '2.5mm网格', '2mm网格',
                       '1.5mm网格', '1mm网格', '0.5mm网格', '0.2mm网格', '0.1mm网格',
                       '0.09mm网格', '0.08mm网格', '0.07mm网格']
        )
    
    @staticmethod
    def _get_q3_config(base_path: str) -> QuestionConfig:
        return QuestionConfig(
            question_id=3,
            output_folder=os.path.join(base_path, 'Q3'),
            fontsize=5,
            yscale=(-200000, 300000),
            name_converter={
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
            },
            path_names=[
                '芯片1.97mm高度处对角线线路', 
                '芯片1.5mm高度处对角线线路',
                '芯片1.3mm高度处对角线线路', 
                '芯片焊球角点处1.97mm至0mm竖直线路',
                '芯片无焊球角点处1.97mm至0mm竖直线路'
            ],
            exchange_char={
                'h0': '-height-1.97mm',
                'h1': '-height-1.5mm',
                'h2': '-height-1.3mm',
                'v0': '-height-1.97mm_to_0mm_焊球端',
                'v1': '-height-1.97mm_to_0mm_无焊球端'
            },
            height_strs_cte=[
                'CTE-Path-height-1.97mm', 
                'CTE-Path-height-1.5mm',
                'CTE-Path-height-1.3mm', 
                'CTE-Path-height-1.97mm_to_0mm_焊球端',
                'CTE-Path-height-1.97mm_to_0mm_无焊球端'
            ],
            height_strs_ela=[
                'ela-Path-height-1.97mm', 
                'ela-Path-height-1.5mm',
                'ela-Path-height-1.3mm', 
                'ela-Path-height-1.97mm_to_0mm_焊球端',
                'ela-Path-height-1.97mm_to_0mm_无焊球端'
            ],
            grid_names=['5mm网格', '4mm网格', '3mm网格', '2mm网格', '1mm网格',
                       '0.5mm网格', '0.4mm网格', '0.3mm网格', '0.2mm网格']
        )

class DataProcessor:
    """Handles data processing operations"""
    
    def __init__(self, config: QuestionConfig):
        self.config = config
    
    def process_folder(self, folder_path: str, duty: str) -> Optional[pd.DataFrame]:
        """Process a folder containing data files"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            print(f"Directory not found: {folder_path}")
            return None
        
        if not folder_path.is_dir():
            print(f"Path is not a directory: {folder_path}")
            return None
        
        # Find files starting with duty prefix
        data_files = [f for f in folder_path.iterdir() if f.name.startswith(duty)]
        
        if not data_files:
            print(f"No files found starting with '{duty}' in {folder_path}")
            return None
        
        data = {}
        for file_path in data_files:
            # Parse filename to get key
            name_parts = file_path.stem.split('.')
            if len(name_parts) < 1:
                continue
                
            name_key = name_parts[0]
            if len(name_key) >= 2:
                suffix = name_key[-2:]
                if suffix in self.config.exchange_char:
                    name_key = name_key[:-2] + self.config.exchange_char[suffix]
            
            # Read file data
            try:
                with open(file_path, 'r') as file:
                    csv_reader = csv.reader(file, delimiter='\t')
                    next(csv_reader)  # Skip header
                    values = [row[-1] for row in csv_reader if row]
                    data[name_key] = values
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue
        
        if not data:
            return None
        # converted_data = {key: [float(value) for value in values] for key, values in data.items()}

        return pd.DataFrame(data, dtype=np.float64)
    
    def to_excel(self, input_folder: str, output_file: str) -> None:
        """Convert processed data to Excel format"""
        input_path = Path(input_folder)
        
        for folder_name in input_path.iterdir():
            if not folder_name.is_dir():
                continue
                
            # Extract temperature from folder name
            try:
                temp_str = folder_name.name.split('-')[-1].split()[0][:4]
            except (IndexError, ValueError):
                continue
            
            # Process CTE and elasticity data
            dutys = ['CTE', 'ela', 'Ther', 'stress', 'strain']
            results = {}
            for duty in dutys:
                results[duty] = self.process_folder(folder_name, duty)
                if results[duty] is None:
                    raise ValueError(f"Data for duty '{duty}' is None ")
                
            # Write to Excel
            try:
                if Path(output_file).exists():
                    with pd.ExcelWriter(output_file, engine='openpyxl', 
                                    mode='a', if_sheet_exists='replace') as writer:
                        for duty_name, result in results.items():
                            result.to_excel(writer, sheet_name=f'{duty_name}-{temp_str}摄氏度', index=False)
                else:
                    with pd.ExcelWriter(output_file, engine='openpyxl', 
                                    mode='w') as writer:
                        for duty_name, result in results.items():
                            result.to_excel(writer, sheet_name=f'{duty_name}-{temp_str}摄氏度', index=False)
            except Exception as e:
                print(f"Error writing Excel file: {e}")
    
    def get_total_dataframes(self, input_folder: str) -> Tuple[Dict, Dict]:
        """Get all dataframes organized by temperature"""
        input_path = Path(input_folder)
        result_list = {}

        dutys = ['CTE', 'ela', 'Ther', 'stress', 'strain', 'Temp']
        for duty in dutys:
            result_data = {}
            for folder_name in input_path.iterdir():
                if not folder_name.is_dir():
                    continue
                    
                try:
                    temp_str = folder_name.name.split('-')[-1].split()[0][:4]
                except (IndexError, ValueError):
                    continue

                result_data[temp_str] = self.process_folder(folder_name, duty)
                if result_data[temp_str] is None:
                    raise ValueError(f"duty {duty} can't find result")

            result_list[duty] = result_data

        return result_list

    def get_aver_scatter(self, dir_names: List[str], duty: str):
        if duty not in ['thermal', 'modulus']:
            raise ValueError(f"{duty} not in ['thermal', 'modulus']")
        
        mapping = {
            'thermal': ['Ther', 'Temp'],
            'modulus': ['stress', 'strain']
        }
        duty_targeted = mapping[duty]
        
        all_processing_units = []
        all_colors = []
        
        for dir_name in dir_names:
            dir_path = Path(self.config.output_folder) / dir_name        
            result_list = self.get_total_dataframes(dir_path)
            
            # 为每个targeted duty创建处理单元
            for duty_name in duty_targeted:
                temp_data = result_list[duty_name]  # 温度字典
                
                processing_unit = []
                colors = []
                
                # 获取所有温度值用于归一化
                all_temps = [float(temp) for temp in temp_data.keys()]
                min_temp, max_temp = min(all_temps), max(all_temps)
                temp_range = max_temp - min_temp if max_temp != min_temp else 1
                
                # 处理每个温度的数据
                for temp_str, route_data in temp_data.items():
                    temp_val = float(temp_str)
                    
                    # 获取第一条线路数据
                    first_route_name = list(route_data.columns)[0]
                    route_values = route_data[first_route_name].values
                    
                    # 温度归一化 (0-1, 0=蓝色, 1=红色)
                    temp_normalized = (temp_val - min_temp) / temp_range
                    
                    # 为这条线路的每个点生成颜色
                    route_length = len(route_values)
                    for i, value in enumerate(route_values):
                        processing_unit.append(value)
                        
                        # 位置归一化 (0-1, 0和1为端点=黄色, 0.5为中心=绿色)
                        if route_length > 1:
                            pos_normalized = i / (route_length - 1)
                            # 计算距离中心的距离，用于黄绿色插值
                            center_distance = abs(pos_normalized - 0.5) * 2  # 0-1之间
                        else:
                            center_distance = 0
                        
                        # 生成颜色 (RGB格式)
                        # 红蓝分量基于温度
                        red = temp_normalized
                        blue = 1 - temp_normalized
                        
                        # 绿黄分量基于位置
                        # 靠近端点时，增加黄色(红+绿)，减少蓝色
                        # 靠近中心时，增加绿色
                        green = 0.3 + 0.7 * (1 - center_distance)  # 基础绿色 + 位置调节
                        yellow_boost = center_distance * 0.5  # 端点黄色增强
                        
                        # 调整颜色
                        red = min(1.0, red + yellow_boost)
                        green = min(1.0, green + yellow_boost)
                        
                        colors.append((red, green, blue))
                
                all_processing_units.append(processing_unit)
                all_colors.append(colors)
        
        return all_processing_units, all_colors

    @staticmethod
    def calculate_variance_string(data: pd.Series) -> str:
        """Calculate variance and return formatted string"""
        variance = np.var(data)
        return f"方差: {variance:.2e}" if not np.isnan(variance) else "方差: NaN"

class Visualizer:
    """Handles all visualization operations"""
    
    def __init__(self, config: QuestionConfig):
        self.config = config
        self._setup_matplotlib()
    
    def _setup_matplotlib(self) -> None:
        """Setup matplotlib with Chinese font support"""
        # plt.rcParams.update({
        #     'font.size': 7,
        #     'font.family': 'simhei',
        #     'axes.unicode_minus': False
        # })
        pass
    
    def plot_stability_results(self, results: List[List], duty: str) -> None:
        """Plot stability test results"""
        if duty not in ['CTE', 'ela']:
            raise ValueError(f"{duty} not in ['CTE', 'ela'] ")
        
        fig = plt.figure(dpi=800, figsize=(8, 6))
        ax = fig.subplots(1, 1)
        
        grid_names = self.config.grid_names
        
        if self.config.question_id == 3:
            labels = ['焊球端', '无焊球端']
            for i, result in enumerate(results):
                ax.plot(grid_names, result, label=labels[i])
            ax.legend(loc='upper right', fontsize=8)
        else:
            for result in results:
                ax.plot(grid_names, result)
        
        fig.suptitle(f'Q{self.config.question_id}网格细分程度稳定性检验', fontsize=12)
        
        ylabel = self.config.duty_names[duty]
        fig.supylabel(ylabel, fontsize=12)
        fig.supxlabel('不同网格大小', fontsize=12)
        
        output_path = Path(self.config.output_folder) / f'{ylabel}不同网格细化下的最终角点结果.jpg'
        fig.savefig(output_path)
        plt.close(fig)
    
    def plot_multi_mesh_comparison(self, dir_names: List[str], duty: str, step: int,
                                 share_y: bool = False, scale: bool = False) -> None:
        """Plot comparison across different mesh sizes"""
        if duty not in ['CTE', 'ela']:
            raise ValueError(f"{duty} not in ['CTE', 'ela'] ")

        colors = cm.coolwarm(np.linspace(0, 1, len(dir_names)))
        
        fig = plt.figure(dpi=800, figsize=(8, 6))
        axes = fig.subplots(2, 3, sharey=share_y, sharex=True)
        axes[1, 2].remove()
        
        # Process data for each directory
        processor = DataProcessor(self.config)
        global_handles, global_labels = [], []
        local_handles = [[] for _ in range(5)]
        local_labels = [[] for _ in range(5)]
        
        results = [[] for _ in range(2)]  # For stability plot
        
        for name_id, dir_name in enumerate(dir_names):
            dir_path = Path(self.config.output_folder) / dir_name
            if not dir_path.exists():
                continue
            
            # Get data
            data_list = processor.get_total_dataframes(dir_path)[duty]
            
            if not data_list:
                continue
            
            # Average across temperatures
            result_data = {}
            for i, (_, output) in enumerate(data_list.items()):
                for path_name, path_data in output.items():
                    if i == 0:
                        result_data[path_name] = path_data / len(data_list)
                    else:
                        result_data[path_name] += path_data / len(data_list)
            
            # Plot data
            height_strs = (self.config.height_strs_cte if duty == 'CTE' 
                          else self.config.height_strs_ela)
            
            for i, height_str in enumerate(height_strs):
                if height_str not in result_data:
                    continue
                    
                path_data = result_data[height_str]
                line, = axes.flat[i].plot(path_data, color=colors[name_id])
                
                if i == 0:
                    global_labels.append(self.config.name_converter.get(dir_name, dir_name))
                    global_handles.append(line)
                
                local_handles[i].append(line)
                local_labels[i].append(DataProcessor.calculate_variance_string(path_data))
                
                axes.flat[i].set_title(self.config.path_names[i])
                axes.flat[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                
                if scale:
                    axes.flat[i].set_ylim(*self.config.yscale)
            
            # Collect data for stability plot
            first_height_data = result_data.get(height_strs[0])
            if first_height_data is not None:
                if self.config.question_id == 3:
                    if duty == 'ela':
                        results[0].append(first_height_data.iloc[3])
                        results[1].append(first_height_data.iloc[-4])
                    else:
                        results[0].append(first_height_data.iloc[0])
                        results[1].append(first_height_data.iloc[-1])
                else:
                    if self.config.question_id == 1 and duty == 'ela':
                        results[0].append(first_height_data.iloc[0])
                    else:
                        results[0].append(first_height_data.iloc[0])
        
        # Add legends
        for i in range(5):
            if local_handles[i]:
                axes.flat[i].legend(local_handles[i], local_labels[i], 
                                  fontsize=self.config.fontsize, loc='lower left')
        
        fig.legend(global_handles, global_labels, bbox_to_anchor=(0.9, 0.4), 
                  fontsize=8, ncol=1)
        
        # Set labels and title
        ylabel = 'CTE热膨胀系数' if duty == 'CTE' else '拉伸模量'
        title = (self.config.cte_title if duty == 'CTE' else self.config.ela_title)
        
        fig.supylabel(ylabel, fontsize=12)
        fig.suptitle(f'不同网格细化程度下,{title}', fontsize=12)
        fig.supxlabel('节点序号', fontsize=12)
        
        # Save figure
        output_name = f'Q{self.config.question_id}-{duty}'
        if share_y:
            output_name += '-shareY'
        if scale:
            output_name += '-scale'
        output_name += f'-STEP{step}'
        
        output_path = Path(self.config.output_folder) / f'{output_name}.jpg'
        fig.savefig(output_path)
        plt.close(fig)
        
        # Generate stability plot
        if self.config.question_id == 3:
            self.plot_stability_results(results, duty)
        else:
            self.plot_stability_results([results[0]], duty)

    def plot_temperature_comparison(self, data_dict: Dict, duty: str, output_path: str,
                                   scale: bool = False, share_y: bool = False) -> None:
        """Plot temperature comparison for given data"""
        if duty not in ['CTE', 'ela']:
            raise ValueError(f"{duty} not in ['CTE', 'ela'] ")

        # Sort by temperature
        sorted_data = sorted([(float(temp), temp, data) for temp, data in data_dict.items()])
        
        colors = cm.coolwarm(np.linspace(0, 1, len(sorted_data)))
        
        fig = plt.figure(figsize=(8, 6), dpi=800)
        axes = fig.subplots(2, 3, sharex=True, sharey=share_y)
        axes[1, 2].remove()
        
        height_strs = (self.config.height_strs_cte if duty == 'CTE' 
                      else self.config.height_strs_ela)
        
        local_handles = [[] for _ in range(5)]
        local_labels = [[] for _ in range(5)]
        
        for j, (_, temp_str, output) in enumerate(sorted_data):
            for i, height_str in enumerate(height_strs):
                if height_str not in output.columns:
                    continue
                
                if i == 1:
                    line, = axes.flat[i].plot(output[height_str], 
                                            label=f'Temp:{temp_str}℃', 
                                            color=colors[j])
                else:
                    line, = axes.flat[i].plot(output[height_str], 
                                            color=colors[j])                    
                axes.flat[i].set_title(self.config.path_names[i])
                axes.flat[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                
                local_handles[i].append(line)
                local_labels[i].append(DataProcessor.calculate_variance_string(output[height_str]))
                
                if scale:
                    axes.flat[i].set_ylim(self.config.yscale)
        
        # Add legends
        for i in range(5):
            if local_handles[i]:
                axes.flat[i].legend(local_handles[i], local_labels[i], 
                                  fontsize=3, loc='upper right', ncol=2)
        
        fig.legend(bbox_to_anchor=(0.95, 0.4), fontsize=8, ncol=2)
        fig.supxlabel('节点序号', fontsize=12)
        
        if duty == 'CTE':
            fig.supylabel('CTE热膨胀系数', fontsize=12)
            fig.suptitle(self.config.cte_title, fontsize=12)
        else:
            fig.supylabel('拉伸模量', fontsize=12)
            fig.suptitle(self.config.ela_title, fontsize=12)
        
        suffix = '-shareY' if share_y else ''
        suffix += '-scale' if scale else ''
        output_file = Path(output_path) / f'{duty}{suffix}.jpg'
        fig.savefig(output_file)
        
        plt.close(fig)

    def plot_scatter_analysis(self, all_processing_units: List[List], all_colors: List[List], 
                            duty: str, output_path: str = None, title_suffix: str = "") -> None:
        if len(all_processing_units) != 2 or len(all_colors) != 2:
            raise ValueError("Expected exactly 2 processing units and 2 color lists")
        
        # Extract data
        x_data = all_processing_units[1]  # Second list for x-axis
        y_data = all_processing_units[0]  # First list for y-axis
        colors = all_colors[0]  # Use first color list (should be consistent)
        
        # Verify data consistency
        if len(x_data) != len(y_data) or len(colors) != len(x_data):
            raise ValueError("Data lengths don't match")
        
        # Create figure with academic style
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 8), dpi=600)
        
        # Create scatter plot
        scatter = ax.scatter(x_data, y_data, c=colors, s=50, alpha=0.7, 
                            edgecolors='black', linewidth=0.5)
        
        # Set labels based on duty type
        if duty == 'thermal':
            ax.set_xlabel('温度 (Temperature)', fontsize=14, fontweight='bold')
            ax.set_ylabel('热应变 (Thermal Strain)', fontsize=14, fontweight='bold')
            title = f'Q{self.config.question_id}热应变-温度关系图{title_suffix}'
        elif duty == 'modulus':
            ax.set_xlabel('应变 (Strain)', fontsize=14, fontweight='bold')
            ax.set_ylabel('应力 (Stress)', fontsize=14, fontweight='bold')
            title = f'Q{self.config.question_id}应力-应变关系图{title_suffix}'
        else:
            ax.set_xlabel('X 值', fontsize=14, fontweight='bold')
            ax.set_ylabel('Y 值', fontsize=14, fontweight='bold')
            title = f'Q{self.config.question_id}数据关系图{title_suffix}'
        
        # Set title
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Customize grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Customize spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('black')
        
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=12, 
                    direction='in', length=6, width=1.2)
        ax.tick_params(axis='both', which='minor', direction='in', 
                    length=3, width=1)
        
        # Enable minor ticks
        ax.minorticks_on()
        
        # Format numbers in scientific notation if needed
        ax.ticklabel_format(style='sci', axis='both', scilimits=(-3, 3))
        
        # Add color explanation text box
        textstr = '颜色说明:\n温度: 蓝色(低) → 红色(高)\n位置: 绿色(中心) → 黄色(端点)'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save plot
        output_path = self.config.output_folder
        output_file = Path(output_path) / f'{duty}_scatter_analysis.png'
        fig.savefig(output_file, dpi=600, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        
        plt.close(fig)
        
        print(f"Scatter plot saved: {output_file}")

class EnhancedDataSorter:
    """Main class that orchestrates data processing and visualization"""
    
    def __init__(self, question: int, base_path: str = "C:\\Users\\oft\\Documents\\ShenZhenCup\\output"):
        """Initialize with question number and base path"""
        self.config = ConfigManager.get_config(question, base_path)
        self.processor = DataProcessor(self.config)
        self.visualizer = Visualizer(self.config)
    
    def update_config(self, **kwargs) -> None:
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
    
    def process_directories(self, duty: str, dir_names: List[str],
                          share_y: bool = False, scale: bool = False, step: int = 0) -> None:
        """Process multiple directories and generate comparisons"""
        self.visualizer.plot_multi_mesh_comparison(
            dir_names, duty, step, share_y, scale
        )
    
    def generate_excel_and_plots(self, dir_names: List[str]) -> None:
        """Generate Excel files and plots for specified directories"""
        for dir_name in dir_names:
            dir_path = Path(self.config.output_folder) / dir_name
            if not dir_path.exists():
                print(f"Directory not found: {dir_path}")
                continue
            
            # Generate Excel file
            output_xlsx = dir_path / f'Q{self.config.question_id}_CTE_ela.xlsx'
            self.processor.to_excel(str(dir_path), str(output_xlsx))
            
            # Generate plots
            cte_data, ela_data = [self.processor.get_total_dataframes(str(dir_path))[key] for key in ['CTE', 'ela']]
            
            if cte_data:
                self.visualizer.plot_temperature_comparison(cte_data, 'CTE', str(dir_path), share_y=True)
            if ela_data:
                self.visualizer.plot_temperature_comparison(ela_data, 'ela', str(dir_path), scale=True)

# Convenience functions for backward compatibility and easy usage
def run_question_analysis(question: int, directories: List[str] = None, 
                         base_path: str = None) -> EnhancedDataSorter:
    """Run analysis for a specific question"""
    if base_path is None:
        base_path = "C:\\Users\\oft\\Documents\\ShenZhenCup\\output"
    
    sorter = EnhancedDataSorter(question, base_path)
    all_processing_units, all_colors = sorter.processor.get_aver_scatter(['Q2-0.5'], 'thermal')
    sorter.visualizer.plot_scatter_analysis(all_processing_units, all_colors, 'thermal')
    if directories:
        sorter.generate_excel_and_plots(directories)
    
    return sorter

# Example usage functions
def analyze_q1(base_path: str = None):
    """Analyze Question 1"""
    directories = ['Q1-3']
    return run_question_analysis(1, directories, base_path)

def analyze_q2(base_path: str = None):
    """Analyze Question 2"""
    directories = ['Q2-0.5']
    return run_question_analysis(2, directories, base_path)

def analyze_q3(base_path: str = None):
    """Analyze Question 3"""
    directories = ['Q3-2']
    return run_question_analysis(3, directories, base_path)

def run_mesh_convergence_study(question: int, dir_names: List[str]) -> None:
    """Run mesh convergence study for specified question and directories"""
    sorter = EnhancedDataSorter(question)
    
    # Run with different visualization options
    sorter.process_directories(dir_names, 'CTE', share_y=True, step=0)
    sorter.process_directories(dir_names, 'ela', scale=True, step=0)

# Main execution function
def main():
    """Main execution function with examples"""
    
    # Example 1: Basic analysis for each question
    # print("Running Q1 analysis...")
    # q1_sorter = analyze_q1()
    
    print("Running Q2 analysis...")
    q2_sorter = analyze_q2()
    
    # print("Running Q3 analysis...")
    # q3_sorter = analyze_q3()
    
    # # Example 2: Mesh convergence studies
    # print("Running mesh convergence studies...")
    
    # Q1 mesh convergence
    # q1_dirs = ['Q1-3', 'Q1-2.5']
    # run_mesh_convergence_study(1, q1_dirs)
    
    # # Q2 mesh convergence
    # q2_dirs = ['Q2v0-4', 'Q2v0-3.5', 'Q2v0-3', 'Q2v0-2.5', 'Q2v0-2', 
    #            'Q2v0-1.5', 'Q2v0-1', 'Q2v0-0.5', 'Q2v0-0.2', 'Q2v0-0.1', 
    #            'Q2v0-0.09', 'Q2v0-0.08', 'Q2v0-0.07']
    # run_mesh_convergence_study(2, q2_dirs)
    
    # # Q3 mesh convergence
    # q3_dirs = ['Q3-5', 'Q3-4', 'Q3-3', 'Q3-2', 'Q3-1', 
    #            'Q3-0.5', 'Q3-0.4', 'Q3-0.3', 'Q3-0.2']
    # run_mesh_convergence_study(3, q3_dirs)
    
    # # Example 3: Custom configuration
    # print("Running custom analysis...")
    # custom_sorter = EnhancedDataSorter(1)
    
    # # Update some parameters
    # custom_sorter.update_config(
    #     fontsize=8,
    #     yscale=(-100000, 400000)
    # )
    
    # # Run custom analysis
    # custom_sorter.generate_excel_and_plots(['Q1-m0.3'])

if __name__ == "__main__":
    main()