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
    grid_names: List[str]
    duty_names: dict = field(default_factory=lambda: {
            'CTE': 'Thermal Expansion Coefficient',
            'ela': 'Elastic Modulus',
            'Ther': 'Thermal Strain',
            'stress': 'Diagonal Stress',
            'Temp': 'Temperature',
            'strain': 'Diagonal Strain'
        })

    @property
    def cte_title(self) -> str:
        return f'Q{self.question_id} CTE Variation at Different Heights and Vertical Path at Corner Points'
    
    @property
    def ela_title(self) -> str:
        return f'Q{self.question_id} Elastic Modulus Variation at Different Heights and Vertical Path at Corner Points'

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
            fontsize=7,
            yscale=(-200000, 500000),
            name_converter={
                'Q1-1': 'BGA 1mm Grid',
                'Q1-2': 'BGA 2mm Grid',
                'Q1-3': 'BGA 3mm Grid',
                'Q1-2.5': 'BGA 2.5mm Grid'               
            },
            path_names=[
                'BGA Diagonal Path at 2mm Height', 
                'BGA Diagonal Path at 1.5mm Height',
                'BGA Diagonal Path at 1mm Height', 
                'BGA Diagonal Path at 0.5mm Height', 
                'BGA Vertical Path from 2mm to 0mm at Corner'
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
            grid_names=['3mm Grid', '2.5mm Grid']
        )
    
    @staticmethod
    def _get_q2_config(base_path: str) -> QuestionConfig:
        return QuestionConfig(
            question_id=2,
            output_folder=os.path.join(base_path, 'Q2'),
            fontsize=7,
            yscale=(-200000, 300000),
            name_converter={
                'Q2-3': 'Chip 4mm Grid',
                'Q2-2.5': 'Chip 3.5mm Grid',
                'Q2-0.5': 'Chip 0.5mm Grid',
                'Q2-2': 'Chip 3mm Grid'
            },
            path_names=[
                'Chip Diagonal Path at 3.57mm Height', 
                'Chip Diagonal Path at 2.67mm Height',
                'Chip Diagonal Path at 1.77mm Height', 
                'Chip Diagonal Path at 0.87mm Height', 
                'Chip Vertical Path from 3.57mm to 0mm at Corner'
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
            grid_names=['4mm Grid', '3.5mm Grid', '3mm Grid', '2.5mm Grid', '2mm Grid',
                       '1.5mm Grid', '1mm Grid', '0.5mm Grid', '0.2mm Grid', '0.1mm Grid',
                       '0.09mm Grid', '0.08mm Grid', '0.07mm Grid']
        )
    
    @staticmethod
    def _get_q3_config(base_path: str) -> QuestionConfig:
        return QuestionConfig(
            question_id=3,
            output_folder=os.path.join(base_path, 'Q3'),
            fontsize=7,
            yscale=(-200000, 300000),
            name_converter={
                'Q3-4': 'Chip 4mm Precision Grid',
                'Q3-5': 'Chip 5mm Precision Grid',
                'Q3-3': 'Chip 3mm Precision Grid',
                'Q3-2': 'Chip 2mm Precision Grid',
                'Q3-1.5': 'Chip 1.5mm Precision Grid',
                'Q3-0.3': 'Chip 0.3mm Precision Grid',
                'Q3-1': 'Chip 1mm Precision Grid',
                'Q3-0.5': 'Chip 0.5mm Precision Grid',
                'Q3-0.4': 'Chip 0.4mm Precision Grid',
                'Q3-0.2': 'Chip 0.2mm Precision Grid',
                'Q3-1.3': 'Chip 1.3mm Precision Grid'
            },
            path_names=[
                'Chip Diagonal Path at 1.97mm Height', 
                'Chip Diagonal Path at 1.5mm Height',
                'Chip Diagonal Path at 1.3mm Height', 
                'Chip Vertical Path from 1.97mm to 0mm (Solder Ball End)',
                'Chip Vertical Path from 1.97mm to 0mm (No Solder Ball End)'
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
            grid_names=['5mm Grid', '4mm Grid', '3mm Grid', '2mm Grid', '1mm Grid',
                       '0.5mm Grid', '0.4mm Grid', '0.3mm Grid', '0.2mm Grid']
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
                
            # Write to Excel with English sheet names
            try:
                if Path(output_file).exists():
                    with pd.ExcelWriter(output_file, engine='openpyxl', 
                                    mode='a', if_sheet_exists='replace') as writer:
                        for duty_name, result in results.items():
                            sheet_name = f'{duty_name}-{temp_str}C'  # Changed from 摄氏度 to C
                            result.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    with pd.ExcelWriter(output_file, engine='openpyxl', 
                                    mode='w') as writer:
                        for duty_name, result in results.items():
                            sheet_name = f'{duty_name}-{temp_str}C'  # Changed from 摄氏度 to C
                            result.to_excel(writer, sheet_name=sheet_name, index=False)
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
        all_shapes = []  # Changed from all_colors to all_shapes for academic style
        
        for dir_name in dir_names:
            dir_path = Path(self.config.output_folder) / dir_name        
            result_list = self.get_total_dataframes(dir_path)
            
            # Create processing units for each targeted duty
            for duty_name in duty_targeted:
                temp_data = result_list[duty_name]  # Temperature dictionary
                
                processing_unit = []
                shapes = []
                
                # Get all temperature values for categorization
                all_temps = sorted([float(temp) for temp in temp_data.keys()])
                temp_categories = self._categorize_temperatures(all_temps)
                
                # Process data for each temperature
                for temp_str, route_data in temp_data.items():
                    temp_val = float(temp_str)
                    temp_category = temp_categories[temp_val]
                    
                    # Get first route data
                    first_route_name = list(route_data.columns)[0]
                    route_values = route_data[first_route_name].values
                    
                    # Store values and corresponding categories
                    route_length = len(route_values)
                    for i, value in enumerate(route_values):
                        processing_unit.append(value)
                        
                        # Create shape category based on temperature and position
                        # Shape combines temperature category and position information
                        position_category = self._categorize_position(i, route_length)
                        shapes.append((temp_category, position_category))
                
                all_processing_units.append(processing_unit)
                all_shapes.append(shapes)
        
        return all_processing_units, all_shapes
    
    def _categorize_temperatures(self, temps: List[float]) -> Dict[float, str]:
        """Categorize temperatures into 3 groups for academic visualization"""
        if not temps:
            return {}
        
        categories = {}
        sorted_temps = sorted(temps)
        
        if len(sorted_temps) <= 3:
            # For small number of temps, use simple categories
            for i, temp in enumerate(sorted_temps):
                if i == 0:
                    categories[temp] = 'low'
                elif i == len(sorted_temps) - 1:
                    categories[temp] = 'high'
                else:
                    categories[temp] = 'medium'
        else:
            # For larger number, divide into three groups using tertiles
            tertile1 = np.percentile(sorted_temps, 33.33)
            tertile2 = np.percentile(sorted_temps, 66.67)
            
            for temp in temps:
                if temp <= tertile1:
                    categories[temp] = 'low'
                elif temp <= tertile2:
                    categories[temp] = 'medium'
                else:
                    categories[temp] = 'high'
        
        return categories
    
    def _categorize_position(self, index: int, total_length: int) -> str:
        """Categorize position along the path"""
        if total_length <= 1:
            return 'single'
        
        relative_pos = index / (total_length - 1)
        
        if relative_pos <= 0.1 or relative_pos >= 0.9:
            return 'edge'
        elif 0.4 <= relative_pos <= 0.6:
            return 'center'
        else:
            return 'intermediate'

    @staticmethod
    def calculate_variance_string(data: pd.Series) -> str:
        """Calculate variance and return formatted string"""
        variance = np.var(data)
        return f"Var: {variance:.2e}" if not np.isnan(variance) else "Var: NaN"

class Visualizer:
    """Handles all visualization operations with enhanced aesthetics"""
    
    def __init__(self, config: QuestionConfig):
        self.config = config
        self._setup_matplotlib()
    
    def _setup_matplotlib(self) -> None:
        """Setup matplotlib with enhanced style for academic papers"""
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        
        # Use seaborn-paper style as base for academic look
        plt.style.use('seaborn-v0_8-paper')
        
        # Custom settings for better aesthetics
        mpl.rcParams.update({
            'font.size': 10,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans'],
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'axes.linewidth': 1.2,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'lines.linewidth': 1.5,
            'lines.markersize': 6,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.edgecolor': '#333333',
            'axes.labelcolor': '#333333',
            'text.color': '#333333',
            'xtick.color': '#333333',
            'ytick.color': '#333333',
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
    
    def plot_stability_results(self, results: List[List], duty: str) -> None:
        """Plot stability test results with enhanced aesthetics"""
        if duty not in ['CTE', 'ela']:
            raise ValueError(f"{duty} not in ['CTE', 'ela']")
        
        # Create figure with golden ratio proportions
        fig, ax = plt.subplots(figsize=(10, 6.18), dpi=300)
        
        grid_names = self.config.grid_names
        
        # Define academic color palette
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        markers = ['o', 's', '^', 'D']
        
        if self.config.question_id == 3:
            labels = ['Solder Ball End', 'No Solder Ball End']
            for i, result in enumerate(results):
                ax.plot(grid_names, result, 
                       color=colors[i % len(colors)],
                       marker=markers[i % len(markers)],
                       label=labels[i],
                       linewidth=2,
                       markersize=8,
                       markeredgewidth=1.5,
                       markeredgecolor='white',
                       alpha=0.9)
            
            ax.legend(loc='best', frameon=True, fancybox=True, 
                     shadow=True, framealpha=0.95, edgecolor='#CCCCCC')
        else:
            for i, result in enumerate(results):
                ax.plot(grid_names, result,
                       color=colors[i % len(colors)],
                       marker=markers[i % len(markers)],
                       linewidth=2,
                       markersize=8,
                       markeredgewidth=1.5,
                       markeredgecolor='white',
                       alpha=0.9)
        
        # Enhanced title and labels
        ax.set_title(f'Q{self.config.question_id} Grid Refinement Stability Test', 
                    fontsize=14, fontweight='bold', pad=20)
        
        ylabel = self.config.duty_names[duty]
        ax.set_ylabel(ylabel, fontsize=12, fontweight='semibold')
        ax.set_xlabel('Grid Size', fontsize=12, fontweight='semibold')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add subtle grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Enhance spines
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(1.2)
            ax.spines[spine].set_color('#333333')
        
        # Add minor ticks
        ax.minorticks_on()
        ax.tick_params(which='minor', length=3, width=0.5)
        ax.tick_params(which='major', length=5, width=1.2)
        
        plt.tight_layout()
        
        output_path = Path(self.config.output_folder) / f'{ylabel.replace(" ", "_")}_grid_refinement_results.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def plot_multi_mesh_comparison(self, dir_names: List[str], duty: str, step: int,
                                 share_y: bool = False, scale: bool = False) -> None:
        """Plot comparison across different mesh sizes with enhanced aesthetics"""
        if duty not in ['CTE', 'ela']:
            raise ValueError(f"{duty} not in ['CTE', 'ela']")

        # Use colorblind-friendly palette
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 0.9, len(dir_names)))
        
        # Create figure with better proportions
        fig = plt.figure(figsize=(14, 8), dpi=300)
        
        # Create subplots with better spacing
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
        axes = [fig.add_subplot(gs[i//3, i%3]) for i in range(5)]
        
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
            
            # Plot data with enhanced style
            height_strs = (self.config.height_strs_cte if duty == 'CTE' 
                          else self.config.height_strs_ela)
            
            for i, height_str in enumerate(height_strs):
                if height_str not in result_data:
                    continue
                    
                path_data = result_data[height_str]
                line, = axes[i].plot(path_data, 
                                    color=colors[name_id],
                                    linewidth=1.8,
                                    alpha=0.85)
                
                if i == 0:
                    global_labels.append(self.config.name_converter.get(dir_name, dir_name))
                    global_handles.append(line)
                
                local_handles[i].append(line)
                local_labels[i].append(DataProcessor.calculate_variance_string(path_data))
                
                # Enhance subplot appearance
                axes[i].set_title(self.config.path_names[i], fontsize=10, fontweight='semibold')
                axes[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                axes[i].grid(True, alpha=0.25, linestyle='--')
                axes[i].set_axisbelow(True)
                
                # Enhance spines
                for spine in ['top', 'right']:
                    axes[i].spines[spine].set_visible(False)
                for spine in ['left', 'bottom']:
                    axes[i].spines[spine].set_linewidth(1)
                    axes[i].spines[spine].set_color('#666666')
                
                if scale:
                    axes[i].set_ylim(*self.config.yscale)
            
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
                    results[0].append(first_height_data.iloc[0])
        
        # Add legends with better styling
        for i in range(5):
            if local_handles[i]:
                legend = axes[i].legend(local_handles[i], local_labels[i], 
                                      fontsize=7, loc='best',
                                      frameon=True, fancybox=True,
                                      framealpha=0.9, edgecolor='#CCCCCC')
                legend.get_frame().set_linewidth(0.5)
        
        # Add global legend with better positioning
        if global_handles:
            global_legend = fig.legend(global_handles, global_labels, 
                                      bbox_to_anchor=(0.98, 0.5), 
                                      loc='center left',
                                      fontsize=9, 
                                      frameon=True, 
                                      fancybox=True,
                                      shadow=True,
                                      framealpha=0.95,
                                      edgecolor='#CCCCCC',
                                      title='Mesh Configuration')
            global_legend.get_frame().set_linewidth(0.8)
        
        # Set labels and title with enhanced styling
        ylabel = self.config.duty_names[duty]
        title = (self.config.cte_title if duty == 'CTE' else self.config.ela_title)
        
        fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical', 
                fontsize=12, fontweight='semibold')
        fig.suptitle(f'Grid Refinement Study: {title}', 
                    fontsize=14, fontweight='bold', y=0.98)
        fig.text(0.5, 0.02, 'Node Index', ha='center', 
                fontsize=12, fontweight='semibold')
        
        # Save figure with high quality
        output_name = f'Q{self.config.question_id}-{duty}'
        if share_y:
            output_name += '-shareY'
        if scale:
            output_name += '-scale'
        output_name += f'-STEP{step}'
        
        output_path = Path(self.config.output_folder) / f'{output_name}.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        # Generate stability plot
        if self.config.question_id == 3:
            self.plot_stability_results(results, duty)
        else:
            self.plot_stability_results([results[0]], duty)

    def plot_temperature_comparison(self, data_dict: Dict, duty: str, output_path: str,
                                   scale: bool = False, share_y: bool = False) -> None:
        """Plot temperature comparison with enhanced aesthetics"""
        if duty not in ['CTE', 'ela']:
            raise ValueError(f"{duty} not in ['CTE', 'ela']")

        # Sort by temperature
        sorted_data = sorted([(float(temp), temp, data) for temp, data in data_dict.items()])
        
        # Use temperature-based colormap for intuitive visualization
        temps = [item[0] for item in sorted_data]
        norm = plt.Normalize(vmin=min(temps), vmax=max(temps))
        colors = plt.cm.coolwarm(norm(temps))
        
        # Create figure with optimal layout
        fig = plt.figure(figsize=(14, 8), dpi=600)
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
        axes = [fig.add_subplot(gs[i//3, i%3]) for i in range(5)]
        
        height_strs = (self.config.height_strs_cte if duty == 'CTE' 
                      else self.config.height_strs_ela)
        
        local_handles = [[] for _ in range(5)]
        local_labels = [[] for _ in range(5)]
        
        for j, (temp_val, temp_str, output) in enumerate(sorted_data):
            for i, height_str in enumerate(height_strs):
                if height_str not in output.columns:
                    continue
                
                # Plot with temperature-based color
                line, = axes[i].plot(output[height_str], 
                                    color=colors[j],
                                    linewidth=1.8,
                                    alpha=0.85,
                                    label=f'{temp_str}°C' if i == 1 else None)
                
                # Enhance subplot appearance
                axes[i].set_title(self.config.path_names[i], 
                                fontsize=10, fontweight='semibold', pad=8)
                axes[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                axes[i].grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
                axes[i].set_axisbelow(True)
                
                # Style spines
                for spine in ['top', 'right']:
                    axes[i].spines[spine].set_visible(False)
                for spine in ['left', 'bottom']:
                    axes[i].spines[spine].set_linewidth(1)
                    axes[i].spines[spine].set_color('#666666')
                
                # Add minor ticks
                axes[i].minorticks_on()
                axes[i].tick_params(which='minor', length=2, width=0.5)
                axes[i].tick_params(which='major', length=4, width=1)
                
                local_handles[i].append(line)
                local_labels[i].append(DataProcessor.calculate_variance_string(output[height_str]))
                
                if scale:
                    axes[i].set_ylim(self.config.yscale)
        
        # Add variance legends to each subplot
        for i in range(5):
            if local_handles[i]:
                var_legend = axes[i].legend(local_handles[i], local_labels[i], 
                                          fontsize=6, loc='upper right',
                                          frameon=True, fancybox=True,
                                          framealpha=0.9, edgecolor='#CCCCCC',
                                          ncol=2 if len(local_handles[i]) > 4 else 1)
                var_legend.get_frame().set_linewidth(0.5)
        
        # Add temperature colorbar instead of regular legend
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.92, 0.35, 0.02, 0.3])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Temperature (°C)', fontsize=10, fontweight='semibold')
        cbar.ax.tick_params(labelsize=8)
        
        # Set main labels and title
        fig.text(0.04, 0.5, self.config.duty_names[duty], 
                va='center', rotation='vertical', 
                fontsize=12, fontweight='semibold')
        fig.text(0.5, 0.02, 'Node Index', ha='center', 
                fontsize=12, fontweight='semibold')
        
        title = self.config.cte_title if duty == 'CTE' else self.config.ela_title
        fig.suptitle(f'Temperature-Dependent Analysis: {title}', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        # Save with high quality
        suffix = '-shareY' if share_y else ''
        suffix += '-scale' if scale else ''
        output_file = Path(output_path) / f'{duty}_temperature_analysis{suffix}.png'
        fig.savefig(output_file, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        
        plt.close(fig)

    def plot_scatter_analysis(self, all_processing_units: List[List], all_shapes: List[List], 
                            duty: str, output_path: str = None, title_suffix: str = "") -> None:
        """Create academic-style scatter plot with shape-based categorization"""
        if len(all_processing_units) != 2 or len(all_shapes) != 2:
            raise ValueError("Expected exactly 2 processing units and 2 shape lists")
        
        # Extract data
        x_data = all_processing_units[1]  # Second list for x-axis
        y_data = all_processing_units[0]  # First list for y-axis
        shapes_data = all_shapes[0]  # Shape categories
        
        # Verify data consistency
        if len(x_data) != len(y_data) or len(shapes_data) != len(x_data):
            raise ValueError("Data lengths don't match")
        
        # Create figure with publication-quality style
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)  # Increased width for better legend placement
        
        # Define marker styles for different categories - simplified to 3 temperature categories
        temp_markers = {
            'low': 'o',      # Circle
            'medium': 's',   # Square
            'high': '^',     # Triangle
            'Q1': 'o',       # Map quartiles to 3 categories
            'Q2': 's', 
            'Q3': 's',       # Q2 and Q3 both use square
            'Q4': '^'
        }
        
        # Simplify temperature categories to just 3
        simplified_temp_map = {
            'Q1': 'low',
            'Q2': 'medium',
            'Q3': 'medium',
            'Q4': 'high'
        }
        
        pos_colors = {
            'edge': '#E74C3C',       # Red for edges
            'center': '#3498DB',     # Blue for center
            'intermediate': "#C3F449", # Gray for intermediate
            'single': '#2ECC71'      # Green for single point
        }
        
        # Group data by categories for efficient plotting
        from collections import defaultdict
        grouped_data = defaultdict(list)
        
        # Simplify temperature categories and regroup data
        for i, (x, y, (temp_cat, pos_cat)) in enumerate(zip(x_data, y_data, shapes_data)):
            # Simplify temperature category
            if temp_cat in simplified_temp_map:
                temp_cat = simplified_temp_map[temp_cat]
            grouped_data[(temp_cat, pos_cat)].append((x, y))
        
        # Plot each group with appropriate style
        plotted_temp_cats = set()
        plotted_pos_cats = set()
        
        for (temp_cat, pos_cat), points in grouped_data.items():
            if not points:
                continue
                
            xs, ys = zip(*points)
            marker = temp_markers.get(temp_cat, 'o')
            color = pos_colors.get(pos_cat, "#7FD5DC")
            
            # Plot with enhanced style
            ax.scatter(xs, ys, 
                      marker=marker,
                      c=color,
                      s=80,  # Slightly larger markers
                      alpha=0.7,
                      edgecolors='black',
                      linewidth=0.8,
                      zorder=5)
            
            # Track categories for legend
            plotted_temp_cats.add(temp_cat)
            plotted_pos_cats.add(pos_cat)
        
        # Create custom legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        
        # Temperature category legend elements (simplified to 3)
        temp_legend = []
        temp_labels = {
            'low': 'Low Temp.',
            'medium': 'Med. Temp.',
            'high': 'High Temp.'
        }
        for cat in ['low', 'medium', 'high']:  # Fixed order
            if cat in plotted_temp_cats:
                temp_legend.append(Line2D([0], [0], marker=temp_markers[cat], 
                                         color='w', markerfacecolor='gray',
                                         markersize=9, markeredgecolor='black',
                                         markeredgewidth=0.8, label=temp_labels[cat]))
        
        # Position category legend elements  
        pos_legend = []
        pos_labels = {
            'edge': 'Edge Points',
            'center': 'Center Points',
            'intermediate': 'Intermediate',
            'single': 'Single Point'
        }
        for cat in ['edge', 'center', 'intermediate', 'single']:  # Fixed order
            if cat in plotted_pos_cats:
                pos_legend.append(Patch(facecolor=pos_colors[cat], 
                                       edgecolor='black', linewidth=0.8,
                                       label=pos_labels.get(cat, cat.capitalize())))
        
        # Calculate data range for better legend positioning and axis limits
        x_range = max(x_data) - min(x_data)
        y_range = max(y_data) - min(y_data)
        x_margin = x_range * 0.15  # Increased margin to show extended regression lines
        y_margin = y_range * 0.15
        
        # Set axis limits with larger margins, ensuring origin and regression lines are visible
        x_min_limit = min(x_data) - x_range*0.5  # Match regression line extent
        x_max_limit = max(x_data) + x_range*0.5
        
        # Calculate y limits based on regression lines
        y_values_at_limits = []
        if 'slope_all' in locals():
            y_values_at_limits.extend([slope_all * x_min_limit, slope_all * x_max_limit])
        
        y_min_limit = min(min(y_data) - y_margin, min(y_values_at_limits) if y_values_at_limits else 0)
        y_max_limit = max(max(y_data) + y_margin, max(y_values_at_limits) if y_values_at_limits else max(y_data))
        
        ax.set_xlim(x_min_limit, x_max_limit)
        ax.set_ylim(y_min_limit, y_max_limit)
        
        # Add both legends in upper left corner
        # Temperature legend
        legend1 = ax.legend(handles=temp_legend, 
                          loc='upper left',
                          bbox_to_anchor=(0.02, 0.98),
                          title='Temperature', 
                          frameon=True,
                          fancybox=True, 
                          shadow=True, 
                          framealpha=0.95,
                          edgecolor='#666666',
                          facecolor='white',
                          title_fontsize=10,
                          fontsize=9,
                          borderpad=1,
                          columnspacing=1.2,
                          handletextpad=0.8)
        ax.add_artist(legend1)
        
        # Position legend below temperature legend
        legend2 = ax.legend(handles=pos_legend, 
                          loc='upper left',
                          bbox_to_anchor=(0.02, 0.78),
                          title='Position', 
                          frameon=True,
                          fancybox=True, 
                          shadow=True, 
                          framealpha=0.95,
                          edgecolor='#666666',
                          facecolor='white',
                          title_fontsize=10,
                          fontsize=9,
                          borderpad=1,
                          columnspacing=1.2,
                          handletextpad=0.8)
        
        # Perform comprehensive regression analysis with zero intercept
        from scipy import stats
        import numpy as np
        
        # Function to perform regression with zero intercept
        def zero_intercept_regression(x, y):
            """Perform linear regression forcing intercept to 0"""
            x = np.array(x)
            y = np.array(y)
            # For y = ax (no intercept), a = sum(xy) / sum(x^2)
            slope = np.sum(x * y) / np.sum(x * x)
            # Calculate R-squared
            y_pred = slope * x
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            # Calculate p-value (simplified)
            n = len(x)
            if n > 2:
                se = np.sqrt(ss_res / (n - 1))
                t_stat = slope / (se / np.sqrt(np.sum(x * x)))
                from scipy.stats import t
                p_value = 2 * (1 - t.cdf(abs(t_stat), n - 1))
            else:
                p_value = np.nan
            return slope, r_squared, p_value
        
        # Prepare data by categories for regression
        position_data = defaultdict(list)
        temp_data = defaultdict(list)
        combined_data = defaultdict(list)
        
        # Reorganize with simplified temperature categories
        for i, (x, y, (temp_cat, pos_cat)) in enumerate(zip(x_data, y_data, shapes_data)):
            # Simplify temperature category
            if temp_cat in simplified_temp_map:
                temp_cat = simplified_temp_map[temp_cat]
            
            position_data[pos_cat].append((x, y))
            temp_data[temp_cat].append((x, y))
            combined_data[(temp_cat, pos_cat)].append((x, y))
        
        # Regression line styles
        position_styles = {
            'edge': {'color': '#E74C3C', 'linestyle': '-', 'alpha': 0.7, 'linewidth': 2.2},
            'center': {'color': '#3498DB', 'linestyle': '-', 'alpha': 0.7, 'linewidth': 2.2},
            'intermediate': {'color': "#C3F449", 'linestyle': '-', 'alpha': 0.7, 'linewidth': 2.2}
        }
        
        temp_styles = {
            'low': {'color': '#2ECC71', 'linestyle': '--', 'alpha': 0.7, 'linewidth': 2.2},
            'medium': {'color': '#F39C12', 'linestyle': '--', 'alpha': 0.7, 'linewidth': 2.2},
            'high': {'color': '#8E44AD', 'linestyle': '--', 'alpha': 0.7, 'linewidth': 2.2}
        }
        
        # Store regression results
        position_regressions = []
        temp_regressions = []
        
        # Perform regression for each position category
        for pos_cat in ['edge', 'center', 'intermediate']:
            if pos_cat not in position_data or len(position_data[pos_cat]) < 2:
                continue
            
            cat_points = position_data[pos_cat]
            cat_x, cat_y = zip(*cat_points)
            
            if len(set(cat_x)) < 2:
                continue
            
            try:
                slope, r_squared, p_value = zero_intercept_regression(cat_x, cat_y)
                
                # Plot regression line with extended range
                # Extend range significantly to ensure visibility
                length = max(x_data) - min(x_data)
                x_min = min(x_data) - length*2
                x_max = max(x_data) + length*2
                x_range = np.array([x_min, x_max])
                y_range = slope * x_range
                
                style = position_styles.get(pos_cat, position_styles['intermediate'])
                ax.plot(x_range, y_range, 
                       color=style['color'],
                       linestyle=style['linestyle'],
                       alpha=style['alpha'],
                       linewidth=style['linewidth'],
                       zorder=4)
                
                position_regressions.append({
                    'category': pos_cat,
                    'slope': slope,
                    'r_squared': r_squared,
                    'p_value': p_value,
                    'n_points': len(cat_points)
                })
            except:
                continue
        
        # Perform regression for each temperature category
        for temp_cat in ['low', 'medium', 'high']:
            if temp_cat not in temp_data or len(temp_data[temp_cat]) < 2:
                continue
            
            cat_points = temp_data[temp_cat]
            cat_x, cat_y = zip(*cat_points)
            
            if len(set(cat_x)) < 2:
                continue
            
            try:
                slope, r_squared, p_value = zero_intercept_regression(cat_x, cat_y)
                
                # Plot regression line with extended range
                # Extend range significantly to ensure visibility
                length = max(x_data) - min(x_data)
                x_min = min(x_data) - length*2
                x_max = max(x_data) + length*2

                x_range = np.array([x_min, x_max])
                y_range = slope * x_range
                
                style = temp_styles.get(temp_cat, temp_styles['medium'])
                ax.plot(x_range, y_range, 
                       color=style['color'],
                       linestyle=style['linestyle'],
                       alpha=style['alpha'],
                       linewidth=style['linewidth'],
                       zorder=4)
                
                temp_regressions.append({
                    'category': temp_cat,
                    'slope': slope,
                    'r_squared': r_squared,
                    'p_value': p_value,
                    'n_points': len(cat_points)
                })
            except:
                continue
        
        # Overall regression with all data (zero intercept)
        slope_all, r_squared_all, p_value_all = zero_intercept_regression(x_data, y_data)
        
        # Plot overall regression line with extended range
        length = max(x_data) - min(x_data)
        x_min = min(x_data) - length*0.5
        x_max = max(x_data) + length*0.5
        x_line = np.array([x_min, x_max])
        y_line = slope_all * x_line
        ax.plot(x_line, y_line, 'k-', alpha=0.9, linewidth=3.0, 
               zorder=3)
        
        # Create comprehensive regression summary text in lower right
        regression_text = "━━━ REGRESSION ANALYSIS ━━━\n"
        regression_text += "(All fits: y = ax, b = 0)\n\n"
        
        # Overall regression
        regression_text += f"▶ Overall (black solid):\n"
        regression_text += f"  R² = {r_squared_all:.3f}, n = {len(x_data)}\n"
        regression_text += f"  y = {slope_all:.3e}x\n\n"
        
        # Position-based regressions
        if position_regressions:
            regression_text += "▶ By Position (solid lines):\n"
            for result in position_regressions:
                cat_name = result['category'].capitalize()[:6]
                regression_text += f"  {cat_name}: R² = {result['r_squared']:.3f}, n = {result['n_points']}\n"
                if result['r_squared'] > 0.7:  # Show slope for good fits
                    regression_text += f"    a = {result['slope']:.3e}\n"
            regression_text += "\n"
        
        # Temperature-based regressions
        if temp_regressions:
            regression_text += "▶ By Temperature (dashed):\n"
            for result in temp_regressions:
                cat_name = result['category'].capitalize()
                regression_text += f"  {cat_name}: R² = {result['r_squared']:.3f}, n = {result['n_points']}\n"
                if result['r_squared'] > 0.7:  # Show slope for good fits
                    regression_text += f"    a = {result['slope']:.3e}\n"
        
        # Add the regression analysis box in lower right
        ax.text(0.98, 0.02, regression_text, 
               transform=ax.transAxes,
               fontsize=8, 
               verticalalignment='bottom',
               horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.8', 
                        facecolor='#FFFFF5',  # Very light yellow
                        alpha=0.95, 
                        edgecolor='#444444',
                        linewidth=1.2),
               family='monospace',  # Use monospace for better alignment
               zorder=10)
        
        # Set labels and title based on duty type
        if duty == 'thermal':
            ax.set_xlabel('Temperature (°C)', fontsize=13, fontweight='semibold')
            ax.set_ylabel('Thermal Strain', fontsize=13, fontweight='semibold')
            title = f'Q{self.config.question_id} Thermal Strain-Temperature Relationship{title_suffix}'
        elif duty == 'modulus':
            ax.set_xlabel('Strain', fontsize=13, fontweight='semibold')
            ax.set_ylabel('Stress (Pa)', fontsize=13, fontweight='semibold')
            title = f'Q{self.config.question_id} Stress-Strain Relationship{title_suffix}'
        else:
            ax.set_xlabel('X Values', fontsize=13, fontweight='semibold')
            ax.set_ylabel('Y Values', fontsize=13, fontweight='semibold')
            title = f'Q{self.config.question_id} Data Relationship{title_suffix}'
        
        ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
        
        # Enhance grid and appearance
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        
        # Style spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_edgecolor('#333333')
        
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=11, 
                      direction='in', length=5, width=1.2)
        ax.tick_params(axis='both', which='minor', direction='in', 
                      length=3, width=0.8)
        
        # Enable minor ticks
        ax.minorticks_on()
        
        # Format numbers in scientific notation if needed
        ax.ticklabel_format(style='sci', axis='both', scilimits=(-3, 3))
        
        # Add subtle background gradient for professional look
        ax.set_facecolor('#FAFAFA')
        
        # Adjust layout with more padding to ensure legends don't get cut off
        plt.tight_layout(pad=1.5)
        
        # Save plot with high quality
        output_path = output_path or self.config.output_folder
        output_file = Path(output_path) / f'{duty}_scatter_analysis_academic.png'
        fig.savefig(output_file, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0.15)
        
        plt.close(fig)
        
        print(f"Academic scatter plot saved: {output_file}")

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
    
    def process_directories(self, dir_names: List[str], duty: str,
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
            output_xlsx = dir_path / f'Q{self.config.question_id}_CTE_ela_analysis.xlsx'
            print(f"Generating Excel file: {output_xlsx}")
            self.processor.to_excel(str(dir_path), str(output_xlsx))
            
            # Generate plots
            print(f"Generating visualization plots for {dir_name}...")
            result_dict = self.processor.get_total_dataframes(str(dir_path))
            cte_data = result_dict.get('CTE', {})
            ela_data = result_dict.get('ela', {})
            
            if cte_data:
                print("  - Creating CTE temperature comparison plots...")
                self.visualizer.plot_temperature_comparison(cte_data, 'CTE', str(dir_path), share_y=True)
            if ela_data:
                print("  - Creating Elastic Modulus temperature comparison plots...")
                self.visualizer.plot_temperature_comparison(ela_data, 'ela', str(dir_path), scale=True)
            
            print(f"Completed processing for {dir_name}\n")
    
    def run_scatter_analysis(self, dir_names: List[str], duty_type: str = 'thermal') -> None:
        """Run scatter plot analysis for specified directories"""
        print(f"Running scatter analysis for {duty_type} data...")
        all_processing_units, all_shapes = self.processor.get_aver_scatter(dir_names, duty_type)
        self.visualizer.plot_scatter_analysis(all_processing_units, all_shapes, duty_type)
        print(f"Scatter analysis completed for {duty_type}\n")

# Convenience functions for backward compatibility and easy usage
def run_question_analysis(question: int, directories: List[str] = None, 
                         base_path: str = None, include_scatter: bool = True) -> EnhancedDataSorter:
    """
    Run comprehensive analysis for a specific question
    
    Parameters:
    -----------
    question : int
        Question number (1, 2, or 3)
    directories : List[str]
        List of directory names to process
    base_path : str
        Base path for output directory
    include_scatter : bool
        Whether to include scatter plot analysis
    
    Returns:
    --------
    EnhancedDataSorter : The sorter object for further processing
    """
    if base_path is None:
        base_path = "C:\\Users\\oft\\Documents\\ShenZhenCup\\output"
    
    print(f"=" * 60)
    print(f"Starting Analysis for Question {question}")
    print(f"=" * 60)
    
    sorter = EnhancedDataSorter(question, base_path)
    
    if directories:
        # Generate Excel and basic plots
        sorter.generate_excel_and_plots(directories)
        
        # Generate scatter plots if requested
        if include_scatter:
            print("Generating scatter plot analyses...")
            # Thermal scatter plot
            sorter.run_scatter_analysis(directories[:1], 'thermal')
            # Modulus scatter plot
            sorter.run_scatter_analysis(directories[:1], 'modulus')
    
    print(f"Analysis completed for Question {question}")
    print(f"=" * 60 + "\n")
    
    return sorter

# Specific analysis functions for each question
def analyze_q1(base_path: str = None, detailed: bool = True):
    """
    Analyze Question 1 with BGA grid refinement
    
    Parameters:
    -----------
    base_path : str
        Base path for output directory
    detailed : bool
        Whether to include detailed analysis
    """
    print("\n" + "="*60)
    print("QUESTION 1: BGA GRID REFINEMENT ANALYSIS")
    print("="*60)
    
    directories = ['Q1-3', 'Q1-2.5', 'Q1-2', 'Q1-1'] if detailed else ['Q1-3']
    sorter = run_question_analysis(1, directories, base_path)
    
    if detailed:
        print("\nRunning mesh convergence study for Q1...")
        sorter.process_directories(directories[:2], 'CTE', share_y=True, step=1)
        sorter.process_directories(directories[:2], 'ela', scale=True, step=1)
    
    return sorter

def analyze_q2(base_path: str = None, detailed: bool = True):
    """
    Analyze Question 2 with chip precision grid
    
    Parameters:
    -----------
    base_path : str
        Base path for output directory
    detailed : bool
        Whether to include detailed analysis
    """
    print("\n" + "="*60)
    print("QUESTION 2: CHIP PRECISION GRID ANALYSIS")
    print("="*60)
    
    directories = ['Q2-0.5']
    sorter = run_question_analysis(2, directories, base_path)
    
    if detailed :
        print("\nRunning extensive mesh convergence study for Q2...")
        extended_dirs = ['Q2v0-4', 'Q2v0-3.5', 'Q2v0-3', 'Q2v0-2.5', 'Q2v0-2', 
                        'Q2v0-1.5', 'Q2v0-1', 'Q2v0-0.5', 'Q2v0-0.2', 'Q2v0-0.1', 
                        'Q2v0-0.09', 'Q2v0-0.08', 'Q2v0-0.07']
        
        # Check which directories exist
        available_dirs = []
        for dir_name in extended_dirs:
            dir_path = Path(sorter.config.output_folder) / dir_name
            if dir_path.exists():
                available_dirs.append(dir_name)
        
        if available_dirs:
            print(f"Found {len(available_dirs)} directories for convergence study")
            sorter.process_directories(available_dirs[:5], 'CTE', share_y=True, step=2)
            sorter.process_directories(available_dirs[:5], 'ela', scale=True, step=2)
    
    return sorter

def analyze_q3(base_path: str = None, detailed: bool = True):
    """
    Analyze Question 3 with solder ball comparison
    
    Parameters:
    -----------
    base_path : str
        Base path for output directory
    detailed : bool
        Whether to include detailed analysis
    """
    print("\n" + "="*60)
    print("QUESTION 3: SOLDER BALL CONFIGURATION ANALYSIS")
    print("="*60)
    
    directories = ['Q3-2', 'Q3-1', 'Q3-0.5'] if detailed else ['Q3-2']
    sorter = run_question_analysis(3, directories, base_path)
    
    if detailed:
        print("\nRunning mesh convergence study for Q3 with solder ball comparison...")
        convergence_dirs = ['Q3-5', 'Q3-4', 'Q3-3', 'Q3-2', 'Q3-1', 
                           'Q3-0.5', 'Q3-0.4', 'Q3-0.3', 'Q3-0.2']
        
        # Check which directories exist
        available_dirs = []
        for dir_name in convergence_dirs:
            dir_path = Path(sorter.config.output_folder) / dir_name
            if dir_path.exists():
                available_dirs.append(dir_name)
        
        if available_dirs:
            print(f"Found {len(available_dirs)} directories for convergence study")
            sorter.process_directories(available_dirs[:5], 'CTE', share_y=True, step=3)
            sorter.process_directories(available_dirs[:5], 'ela', scale=True, step=3)
    
    return sorter

def run_mesh_convergence_study(question: int, dir_names: List[str], 
                               output_suffix: str = "") -> None:
    """
    Run detailed mesh convergence study for specified question and directories
    
    Parameters:
    -----------
    question : int
        Question number
    dir_names : List[str]
        Directory names to compare
    output_suffix : str
        Additional suffix for output files
    """
    print(f"\n{'='*60}")
    print(f"MESH CONVERGENCE STUDY - QUESTION {question}")
    print(f"{'='*60}")
    print(f"Comparing {len(dir_names)} mesh configurations...")
    
    sorter = EnhancedDataSorter(question)
    
    # Run with different visualization options
    print("\nGenerating CTE convergence plots...")
    sorter.process_directories(dir_names, 'CTE', share_y=True, step=0)
    
    print("Generating Elastic Modulus convergence plots...")
    sorter.process_directories(dir_names, 'ela', scale=True, step=0)
    
    print(f"Mesh convergence study completed for Question {question}\n")

# Main execution function
def main():
    """
    Main execution function with comprehensive analysis pipeline
    """
    print("\n" + "="*70)
    print(" "*20 + "ENHANCED DATA ANALYSIS SYSTEM")
    print(" "*15 + "Academic Publication Quality Output")
    print("="*70)
    
    # Configuration
    base_path = None  # Use default or specify custom path
    run_detailed = True  # Set to False for quick analysis
    
    try:
        # Example 1: Basic analysis for each question
        if True:  # Set to False to skip
            print("\n[1] Running Basic Analysis for All Questions")
            print("-"*50)
            
            # Uncomment the analyses you want to run
            # q1_sorter = analyze_q1(base_path, detailed=False)
            q2_sorter = analyze_q2(base_path, detailed=False)
            # q3_sorter = analyze_q3(base_path, detailed=False)
        
        # Example 2: Detailed analysis with mesh convergence
        if run_detailed and False:  # Set to True to enable
            print("\n[2] Running Detailed Analysis with Mesh Convergence")
            print("-"*50)
            
            # Q1 detailed analysis
            # q1_sorter = analyze_q1(base_path, detailed=True)
            
            # Q2 detailed analysis
            q2_sorter = analyze_q2(base_path, detailed=True)
            
            # Q3 detailed analysis
            # q3_sorter = analyze_q3(base_path, detailed=True)
        
        # Example 3: Custom mesh convergence studies
        if False:  # Set to True to enable
            print("\n[3] Running Custom Mesh Convergence Studies")
            print("-"*50)
            
            # Custom Q1 convergence
            q1_custom_dirs = ['Q1-3', 'Q1-2.5', 'Q1-2', 'Q1-1.5', 'Q1-1']
            # run_mesh_convergence_study(1, q1_custom_dirs)
            
            # Custom Q2 convergence with fine meshes
            q2_fine_dirs = ['Q2-0.1', 'Q2-0.09', 'Q2-0.08', 'Q2-0.07', 'Q2-0.05']
            # run_mesh_convergence_study(2, q2_fine_dirs)
        
        # Example 4: Custom configuration example
        if False:  # Set to True to enable
            print("\n[4] Running Custom Configuration Analysis")
            print("-"*50)
            
            custom_sorter = EnhancedDataSorter(1, base_path)
            
            # Update visualization parameters
            custom_sorter.update_config(
                fontsize=9,
                yscale=(-150000, 450000)
            )
            
            # Run custom analysis
            custom_sorter.generate_excel_and_plots(['Q1-m0.3'])
        
        print("\n" + "="*70)
        print(" "*20 + "ANALYSIS PIPELINE COMPLETED")
        print(" "*15 + "All outputs saved to configured directories")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {str(e)}")
        print("Please check your data directories and configuration.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()