import numpy as np
import pandas as pd
import csv 
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from collections import defaultdict
from scipy.stats import t

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
    Grid_stability_ther = defaultdict(list)
    Grid_stability_modu = defaultdict(list)

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
                'Q1-1': 'FQN 1mm Grid',
                'Q1-2': 'FQN 2mm Grid',
                'Q1-3': 'FQN 3mm Grid',
                'Q1-2.5': 'FQN 2.5mm Grid',
                'Q1-0.5': 'FQN 0.5mm Grid',
                'Q1-0.3': 'FQN 0.3mm Grid',
                'Q1-1.5': 'FQN 1.5mm Grid'
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
            grid_names=['3mm Grid', '2.5mm Grid', '2mm Grid', '1.5mm Grid',
                        '1mm Grid', '0.5mm Grid', '0.3mm Grid']
        )
    
    @staticmethod
    def _get_q2_config(base_path: str) -> QuestionConfig:
        return QuestionConfig(
            question_id=2,
            output_folder=os.path.join(base_path, 'Q2'),
            fontsize=7,
            yscale=(-200000, 300000),
            name_converter={
                'Q2-1': 'BGA 1mm Grid',
                'Q2-2': 'BGA 2mm Grid',
                'Q2-3': 'BGA 3mm Grid',
                'Q2-2.5': 'BGA 2.5mm Grid',
                'Q2-0.5': 'BGA 0.5mm Grid',
                'Q2-1.5': 'BGA 1.5mm Grid'
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
            grid_names=['3mm Grid', '2.5mm Grid', '2mm Grid', '1.5mm Grid',
                        '1mm Grid', '0.5mm Grid']
        )
    
    @staticmethod
    def _get_q3_config(base_path: str) -> QuestionConfig:
        return QuestionConfig(
            question_id=3,
            output_folder=os.path.join(base_path, 'Q3'),
            fontsize=7,
            yscale=(-200000, 300000),
            name_converter={
                'Q3-1': 'BGA 1mm Grid',
                'Q3-2': 'BGA 2mm Grid',
                'Q3-3': 'BGA 3mm Grid',
                'Q3-2.5': 'BGA 2.5mm Grid',
                'Q3-0.5': 'BGA 0.5mm Grid',
                'Q3-0.3': 'BGA 0.3mm Grid',
                'Q3-1.5': 'BGA 1.5mm Grid'
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
            grid_names=['3mm Grid', '2.5mm Grid', '2mm Grid', '1.5mm Grid',
                        '1mm Grid', '0.5mm Grid', '0.3mm Grid']
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

    def get_aver_scatter(self, dir_name: str, duty: str):
        if duty not in ['thermal', 'modulus']:
            raise ValueError(f"{duty} not in ['thermal', 'modulus']")
        
        mapping = {
            'thermal': ['Ther', 'Temp'],
            'modulus': ['stress', 'strain']
        }
        duty_targeted = mapping[duty]
        
        all_processing_units = []
        all_shapes = []  # Changed from all_colors to all_shapes for academic style
        
        # for dir_name in dir_names:
        dir_path = Path(self.config.output_folder) / dir_name        
        result_list = self.get_total_dataframes(dir_path)
            
        # Create processing units for each targeted duty
        for duty_name in duty_targeted:
            temp_data = result_list[duty_name]  # Temperature dictionary
            
            processing_unit = []
            shapes = []

            all_temps = sorted([float(temp) for temp in temp_data.keys()])
            temp_categories, temp_ranges = self._categorize_temperatures(all_temps)

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
        
        return all_processing_units, all_shapes, temp_ranges
    
    def _categorize_temperatures(self, temps: List[float]) -> Tuple[Dict[float, str], Dict[str, Tuple[float, float]]]:
        """Categorize temperatures into 3 groups for academic visualization"""
        if not temps:
            return {}, {}
        
        categories = {}
        temp_ranges = {}
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
        
        # Calculate actual temperature ranges for each category
        low_temps = [t for t, cat in categories.items() if cat == 'low']
        medium_temps = [t for t, cat in categories.items() if cat == 'medium']
        high_temps = [t for t, cat in categories.items() if cat == 'high']
        
        if low_temps:
            temp_ranges['low'] = f"low:{min(low_temps)}-{max(low_temps)}degrees Celsius"
        if medium_temps:
            temp_ranges['medium'] = f"medium:{min(medium_temps)}-{max(medium_temps)}degrees Celsius"
        if high_temps:
            temp_ranges['high'] = f"high:{min(high_temps)}-{max(high_temps)}degrees Celsius"
        
        return categories, temp_ranges
    
    def _categorize_position(self, index: int, total_length: int) -> str:
        """Categorize position along the path"""
        if total_length <= 1:
            return 'single'
        
        relative_pos = index / (total_length - 1)
        
        if self.config.question_id == 3:
            center_index = total_length // 2
            
            if index < center_index:
                side = 'solder'  # Solder ball side (near side)
            else:
                side = 'no_solder'  # No solder ball side (far side)
            
            if relative_pos <= 0.1 or relative_pos >= 0.9:
                return f'edge_{side}'
            elif 0.4 <= relative_pos <= 0.6:
                return f'center_{side}'
            else:
                return f'intermediate_{side}'
        
        # Original logic for Q1 and Q2
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
            'savefig.dpi': 600,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })

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
        axes = []
        for i in range(5):
            if i == 0:
                ax = fig.add_subplot(gs[i//3, i%3])
            else:
                ax = fig.add_subplot(gs[i//3, i%3], sharey=axes[0])
            axes.append(ax)
        
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
        fig.savefig(output_file, dpi=600, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        
        plt.close(fig)

    def _forced_point_regression(self, x, y, fixed_x, fixed_y):
        """Perform linear regression forcing through a fixed point"""
        x = np.array(x)
        y = np.array(y)
        dx = x - fixed_x
        dy = y - fixed_y
        slope = np.sum(dx * dy) / np.sum(dx ** 2)
        y_pred = slope * dx + fixed_y
        # R-squared
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        # p-value
        n = len(x)
        if n > 2:
            se = np.sqrt(ss_res / (n - 1))
            t_stat = slope / (se / np.sqrt(np.sum(dx ** 2)))

            p_value = 2 * (1 - t.cdf(abs(t_stat), n - 1))
        else:
            p_value = np.nan
        return slope, r_squared, p_value

    def plot_scatter_analysis(self, all_processing_units: List[List], all_shapes: List[List], temp_map: dict,
                            duty: str, output_name: str = None, title_suffix: str = "") -> None:
        """Create academic-style scatter plot with shape-based categorization"""
        if len(all_processing_units) != 2 or len(all_shapes) != 2:
            raise ValueError("Expected exactly 2 processing units and 2 shape lists")

        x_data = all_processing_units[1]
        y_data = all_processing_units[0] 
        shapes_data = all_shapes[0] 
        
        if len(x_data) != len(y_data) or len(shapes_data) != len(x_data):
            raise ValueError("Data lengths don't match")
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8), dpi=600) 
        
        temp_markers = {
            'low': 'o',      # Circle
            'medium': 's',   # Square
            'high': '^'     # Triangle
        }
        
        temp_styles = {
            'low': {'color': '#2ECC71', 'linestyle': '--', 'alpha': 0.7, 'linewidth': 2.2},
            'medium': {'color': '#F39C12', 'linestyle': '--', 'alpha': 0.7, 'linewidth': 2.2},
            'high': {'color': '#8E44AD', 'linestyle': '--', 'alpha': 0.7, 'linewidth': 2.2}
        }
        
        pos_colors = {
            'edge': '#E74C3C',       # Red for edges
            'center': '#3498DB',     # Blue for center
            'intermediate': "#C3F449", # Light green for intermediate
            'single': '#2ECC71',      # Green for single point
            # Q3 specific - solder side
            'edge_solder': '#E74C3C',       # Red 
            'center_solder': '#3498DB',     # Blue
            'intermediate_solder': "#C3F449", # Light green
            # Q3 specific - no solder side  
            'edge_no_solder': "#A31600EB",       # Dark red
            'center_no_solder': "#000DC9",     # Dark blue  
            'intermediate_no_solder': "#869800", # Dark green
        }

        position_styles = {
            'edge': {'color': '#E74C3C', 'linestyle': '-', 'alpha': 0.7, 'linewidth': 2.2},
            'center': {'color': '#3498DB', 'linestyle': '-', 'alpha': 0.7, 'linewidth': 2.2},
            'intermediate': {'color': "#C3F449", 'linestyle': '-', 'alpha': 0.7, 'linewidth': 2.2},
            # Q3 specific styles
            'edge_solder': {'color': '#E74C3C', 'linestyle': '-', 'alpha': 0.7, 'linewidth': 2.2},
            'center_solder': {'color': '#3498DB', 'linestyle': '-', 'alpha': 0.7, 'linewidth': 2.2},
            'intermediate_solder': {'color': "#C3F449", 'linestyle': '-', 'alpha': 0.7, 'linewidth': 2.2},
            'edge_no_solder': {'color': "#A31600EB", 'linestyle': '-', 'alpha': 0.7, 'linewidth': 2.2},
            'center_no_solder': {'color': "#000DC9", 'linestyle': '-', 'alpha': 0.7, 'linewidth': 2.2},
            'intermediate_no_solder': {'color': "#869800", 'linestyle': '-', 'alpha': 0.7, 'linewidth': 2.2},
        }

        grouped_data = defaultdict(list)
        for i, (x, y, (temp_cat, pos_cat)) in enumerate(zip(x_data, y_data, shapes_data)):
            grouped_data[(temp_cat, pos_cat)].append((x, y))

        plotted_temp_cats = set()
        plotted_pos_cats = set()
        
        for (temp_cat, pos_cat), points in grouped_data.items():
            xs, ys = zip(*points)
            marker = temp_markers[temp_cat]
            color = pos_colors[pos_cat]
            
            ax.scatter(xs, ys, 
                      marker=marker,
                      c=color,
                      s=80,
                      alpha=0.7,
                      edgecolors='black',
                      linewidth=0.8,
                      zorder=5)
            plotted_temp_cats.add(temp_cat)
            plotted_pos_cats.add(pos_cat)
        
        # Temperature category legend elements
        temp_legend = []
        temp_labels = {
            'low': 'Low Temp.',
            'medium': 'Med. Temp.',
            'high': 'High Temp.'
        }
        
        for cat in ['low', 'medium', 'high']:
            if cat in plotted_temp_cats:
                temp_legend.append(Line2D([0], [0], 
                          marker=temp_markers[cat],
                          color=temp_styles[cat]['color'],
                          markerfacecolor='gray',
                          markeredgecolor='black',
                          markeredgewidth=0.8,
                          markersize=9,
                          label=temp_labels[cat]))
        
        # Position category legend elements  
        pos_legend = []
        pos_labels = {
            'edge': 'Edge Points',
            'center': 'Center Points',
            'intermediate': 'Intermediate',
            'single': 'Single Point',
            # Q3 specific
            'edge_solder': 'Edge (Solder Side)',
            'center_solder': 'Center (Solder Side)',
            'intermediate_solder': 'Intermediate (Solder Side)',
            'edge_no_solder': 'Edge (No Solder Side)',
            'center_no_solder': 'Center (No Solder Side)',
            'intermediate_no_solder': 'Intermediate (No Solder Side)',
        }

        if self.config.question_id == 3:
            categories_to_show = ['edge_solder', 'center_solder', 'intermediate_solder',
                                'edge_no_solder', 'center_no_solder', 'intermediate_no_solder']
        else:
            categories_to_show = ['edge', 'center', 'intermediate', 'single']

        for cat in categories_to_show:
            if cat in plotted_pos_cats:
                pos_legend.append(Patch(facecolor=pos_colors[cat], 
                                    edgecolor='black', linewidth=0.8,
                                    label=pos_labels[cat]))
        
        x_range = max(x_data) - min(x_data)
        y_range = max(y_data) - min(y_data)

        x_min_limit = min(x_data) - x_range*0.5 
        x_max_limit = max(x_data) + x_range*0.5
        y_min_limit = min(y_data) - y_range*0.2 
        y_max_limit = max(y_data) + y_range*0.2
        
        ax.set_xlim(x_min_limit, x_max_limit)
        ax.set_ylim(y_min_limit, y_max_limit)
        
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
        
        if duty == 'thermal':
            fixed_x = 25.0  # For temperature, force through (25, 0)
            fixed_y = 0.0
        else:
            fixed_x = 0.0   # For modulus or others, force through (0, 0)
            fixed_y = 0.0
          
        # Perform regression for each position category
        if self.config.question_id == 3:
            position_categories = ['edge_solder', 'center_solder', 'intermediate_solder',
                                'edge_no_solder', 'center_no_solder', 'intermediate_no_solder']
        else:
            position_categories = ['edge', 'center', 'intermediate']

        position_regressions = []
        for pos_cat in position_categories:
            cat_points = [point for key in grouped_data if key[1] == pos_cat for point in grouped_data[key]]
            cat_x, cat_y = zip(*cat_points)

            slope, r_squared, p_value = self._forced_point_regression(cat_x, cat_y, fixed_x, fixed_y)
        
            x_range = np.array([x_min_limit, x_max_limit])
            y_range = slope * (x_range - fixed_x) + fixed_y
            
            style = position_styles[pos_cat]
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
        self._write_data(position_regressions, duty)

        temp_solder_regressions = []
        temp_regressions = []
        temp_no_solder_regressions = []
        if self.config.question_id != 3:
            # Perform regression for each temperature category
            for temp_cat in ['low', 'medium', 'high']:
                cat_points = [point for key in grouped_data if key[0] == temp_cat and 'edge' in key[1] for point in grouped_data[key]]
                cat_x, cat_y = zip(*cat_points)

                slope, r_squared, p_value = self._forced_point_regression(cat_x, cat_y, fixed_x, fixed_y)
            
                x_range = np.array([x_min_limit, x_max_limit])
                y_range = slope * (x_range - fixed_x) + fixed_y
                
                style = temp_styles[temp_cat]
                ax.plot(x_range, y_range, 
                        color=style['color'],
                        linestyle=style['linestyle'],
                        alpha=style['alpha'],
                        linewidth=style['linewidth'],
                        zorder=4)
                
                temp_regressions.append({
                    'category': temp_map[temp_cat],
                    'slope': slope,
                    'r_squared': r_squared,
                    'p_value': p_value,
                    'n_points': len(cat_points)
                })
        else:
            # Perform regression for each temperature category
            for temp_cat in ['low', 'medium', 'high']:
                cat_points = [point for key in grouped_data if key[0] == temp_cat and key[1]=='edge_solder' for point in grouped_data[key]]
                cat_x, cat_y = zip(*cat_points)

                slope, r_squared, p_value = self._forced_point_regression(cat_x, cat_y, fixed_x, fixed_y)
            
                x_range = np.array([x_min_limit, x_max_limit])
                y_range = slope * (x_range - fixed_x) + fixed_y
                
                style = temp_styles[temp_cat]
                ax.plot(x_range, y_range, 
                        color=style['color'],
                        linestyle=style['linestyle'],
                        alpha=style['alpha'],
                        linewidth=style['linewidth'],
                        zorder=4)
                
                temp_solder_regressions.append({
                    'category': temp_map[temp_cat],
                    'slope': slope,
                    'r_squared': r_squared,
                    'p_value': p_value,
                    'n_points': len(cat_points)
                })

            for temp_cat in ['low', 'medium', 'high']:
                cat_points = [point for key in grouped_data if key[0] == temp_cat and key[1]=='edge_no_solder' for point in grouped_data[key]]
                cat_x, cat_y = zip(*cat_points)

                slope, r_squared, p_value = self._forced_point_regression(cat_x, cat_y, fixed_x, fixed_y)
            
                x_range = np.array([x_min_limit, x_max_limit])
                y_range = slope * (x_range - fixed_x) + fixed_y
                
                style = temp_styles[temp_cat]
                ax.plot(x_range, y_range, 
                        color=style['color'],
                        linestyle=style['linestyle'],
                        alpha=style['alpha'],
                        linewidth=style['linewidth'],
                        zorder=4)
                
                temp_no_solder_regressions.append({
                    'category': temp_map[temp_cat],
                    'slope': slope,
                    'r_squared': r_squared,
                    'p_value': p_value,
                    'n_points': len(cat_points)
                })

        # Overall regression with all data (duty-dependent fixed point)
        slope_all, r_squared_all, p_value_all = self._forced_point_regression(x_data, y_data, fixed_x, fixed_y)
        x_line = np.array([x_min_limit, x_max_limit])
        y_line = slope_all * (x_line - fixed_x) + fixed_y
        ax.plot(x_line, y_line, 'k-', alpha=0.9, linewidth=3.0, 
               zorder=3)
        
        # Create comprehensive regression summary text in lower right
        regression_text = "━━━ REGRESSION ANALYSIS ━━━\n"
        regression_text += f"(All fits: y = a(x - {fixed_x:.1f}) + {fixed_y:.1f})\n\n"
        
        # Overall regression
        regression_text += f"▶ Overall (black solid):\n"
        regression_text += f"  R² = {r_squared_all:.3f}, n = {len(x_data)}\n"
        regression_text += f"  y = {slope_all:.3e}(x - {fixed_x:.1f})\n\n"
        
        # Position-based regressions
        if position_regressions:
            regression_text += "▶ By Position(solid lines):\n"
            for result in position_regressions:
                cat_name = result['category'].capitalize()
                regression_text += f"  {cat_name}: R² = {result['r_squared']:.3f}, n = {result['n_points']}\n"
                if result['r_squared'] > 0.7:  # Show slope for good fits
                    regression_text += f"    a = {result['slope']:.3e}\n"
            regression_text += "\n"
        
        # Temperature-based regressions
        if temp_regressions:
            regression_text += "▶ By Temperature in edge(dashed):\n"
            for result in temp_regressions:
                cat_name = result['category'].capitalize()
                regression_text += f"  {cat_name}: R² = {result['r_squared']:.3f}, n = {result['n_points']}\n"
                if result['r_squared'] > 0.7:  # Show slope for good fits
                    regression_text += f"    a = {result['slope']:.3e}\n"

        if temp_solder_regressions:
            regression_text += "▶ By Temperature in solder edge(dashed):\n"
            for result in temp_solder_regressions:
                cat_name = result['category'].capitalize()
                regression_text += f"  {cat_name}: R² = {result['r_squared']:.3f}, n = {result['n_points']}\n"
                if result['r_squared'] > 0.7:  # Show slope for good fits
                    regression_text += f"    a = {result['slope']:.3e}\n"

        if temp_no_solder_regressions:
            regression_text += "▶ By Temperature in no solder edge(dashed):\n"
            for result in temp_no_solder_regressions:
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
        ax.minorticks_on()
        ax.ticklabel_format(style='sci', axis='both', scilimits=(-3, 3))
        ax.set_facecolor('#FAFAFA')
        plt.tight_layout(pad=1.5)
        output_file = os.path.join(self.config.output_folder, output_name, f'{duty}_scatter_analysis_academic.png')
        fig.savefig(output_file, dpi=600, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0.15)
        
        plt.close(fig)
        
        print(f"Academic scatter plot saved: {output_file}")

    def _write_data(self, pos_regressions, duty):
        pos_regressions = [x for x in pos_regressions if 'edge' in x['category']]
        for x in pos_regressions:
            if duty == 'thermal':
                self.config.Grid_stability_ther[x['category']].append(x['slope'])
            elif duty == 'modulus':
                self.config.Grid_stability_modu[x['category']].append(x['slope'])

    def draw_stability(self, output_name: str = None):
        """Plot grid refinement stability analysis with enhanced aesthetics"""
        
        # Check if there's data to plot
        if not self.config.Grid_stability_ther and not self.config.Grid_stability_modu:
            print("No stability data available to plot")
            return
        
        # Create figure with subplots for thermal and modulus
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=600)
        
        # Define color schemes based on question type
        if self.config.question_id == 3:
            colors = {
                'edge_solder': '#E74C3C',       # Red
                'edge_no_solder': '#8B0000',    # Dark red
            }
            markers = {
                'edge_solder': 'o',
                'edge_no_solder': 's',
            }
            linestyles = {
                'edge_solder': '-',
                'edge_no_solder': '-.',
            }
            labels = {
                'edge_solder': 'Solder Ball Side',
                'edge_no_solder': 'No Solder Ball Side',
            }
        else:
            colors = {'edge': '#E74C3C'}
            markers = {'edge': 'o'}
            linestyles = {'edge': '-'}
            labels = {'edge': 'Edge Points'}
        
        # Plot thermal expansion stability
        if self.config.Grid_stability_ther:
            for name, data in self.config.Grid_stability_ther.items():
                if len(data) == len(self.config.grid_names):
                    ax1.plot(self.config.grid_names, data, 
                            color=colors.get(name, '#333333'),
                            marker=markers.get(name, 'o'),
                            linestyle=linestyles.get(name, '-'),
                            label=labels.get(name, name),
                            linewidth=2.5,
                            markersize=10,
                            markeredgewidth=2,
                            markeredgecolor='white',
                            alpha=0.9)
                    
                    # Add data point annotations for the last point
                    if len(data) > 0:
                        ax1.annotate(f'{data[-1]:.2e}', 
                                xy=(self.config.grid_names[-1], data[-1]),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=7, alpha=0.7)
            
            # Enhance ax1 appearance
            ax1.set_title('Thermal Expansion Coefficient Convergence', 
                        fontsize=14, fontweight='bold', pad=15)
            ax1.set_xlabel('Grid Size', fontsize=12, fontweight='semibold')
            ax1.set_ylabel('CTE Slope (∂ε/∂T)', fontsize=12, fontweight='semibold')
            ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax1.set_axisbelow(True)
            
            # Add legend
            ax1.legend(loc='center right', frameon=True, fancybox=True,
                    shadow=True, framealpha=0.95, edgecolor='#CCCCCC',
                    fontsize=10, title='Position', title_fontsize=11)
            
            # Rotate x-axis labels
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add minor ticks
            ax1.minorticks_on()
            ax1.tick_params(which='minor', length=3, width=0.5)
            ax1.tick_params(which='major', length=5, width=1.2)
            
            # Style spines
            for spine in ['top', 'right']:
                ax1.spines[spine].set_visible(False)
            for spine in ['left', 'bottom']:
                ax1.spines[spine].set_linewidth(1.2)
                ax1.spines[spine].set_color('#333333')
            
            # Format y-axis in scientific notation
            ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            
            # Add horizontal line at y=0 for reference
            ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        
        # Plot modulus stability
        if self.config.Grid_stability_modu:
            for name, data in self.config.Grid_stability_modu.items():
                if len(data) == len(self.config.grid_names):
                    ax2.plot(self.config.grid_names, data,
                            color=colors.get(name, '#333333'),
                            marker=markers.get(name, 'o'),
                            linestyle=linestyles.get(name, '-'),
                            label=labels.get(name, name),
                            linewidth=2.5,
                            markersize=10,
                            markeredgewidth=2,
                            markeredgecolor='white',
                            alpha=0.9)
                    
                    # Add data point annotations for the last point
                    if len(data) > 0:
                        ax2.annotate(f'{data[-1]:.2e}', 
                                xy=(self.config.grid_names[-1], data[-1]),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=7, alpha=0.7)
            
            # Enhance ax2 appearance
            ax2.set_title('Elastic Modulus Convergence', 
                        fontsize=14, fontweight='bold', pad=15)
            ax2.set_xlabel('Grid Size', fontsize=12, fontweight='semibold')
            ax2.set_ylabel('Modulus Slope (∂σ/∂ε)', fontsize=12, fontweight='semibold')
            ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax2.set_axisbelow(True)
            
            # Add legend
            ax2.legend(loc='center right', frameon=True, fancybox=True,
                    shadow=True, framealpha=0.95, edgecolor='#CCCCCC',
                    fontsize=10, title='Position', title_fontsize=11)
            
            # Rotate x-axis labels
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add minor ticks
            ax2.minorticks_on()
            ax2.tick_params(which='minor', length=3, width=0.5)
            ax2.tick_params(which='major', length=5, width=1.2)
            
            # Style spines
            for spine in ['top', 'right']:
                ax2.spines[spine].set_visible(False)
            for spine in ['left', 'bottom']:
                ax2.spines[spine].set_linewidth(1.2)
                ax2.spines[spine].set_color('#333333')
            
            # Format y-axis in scientific notation
            ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            
            # Add horizontal line at y=0 for reference
            ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        
        # Overall title
        fig.suptitle(f'Q{self.config.question_id} Grid Refinement Convergence Analysis',
                    fontsize=16, fontweight='bold', y=1.02)
        
        # Add convergence analysis text box
        convergence_info = self._calculate_convergence_metrics()
        if convergence_info:
            fig.text(0.5, -0.08, convergence_info, ha='center', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F0F0',
                            alpha=0.9, edgecolor='#666666', linewidth=1))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        if output_name:
            output_file = Path(self.config.output_folder) / f'{output_name}_stability_analysis.png'
        else:
            output_file = Path(self.config.output_folder) / f'Q{self.config.question_id}_grid_stability_analysis.png'
        
        fig.savefig(output_file, dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
        
        plt.close(fig)
        print(f"Stability analysis plot saved: {output_file}")
        
        # Print convergence summary
        self._print_convergence_summary()

    def _calculate_convergence_metrics(self) -> str:
        """Calculate convergence metrics for display"""
        metrics = []
        
        # Analyze thermal data
        if self.config.Grid_stability_ther:
            for name, data in self.config.Grid_stability_ther.items():
                if len(data) >= 2:
                    # Calculate relative change for last two points
                    rel_change = abs((data[-1] - data[-2]) / data[-2] * 100) if data[-2] != 0 else 0
                    converged = "[CONVERGED]" if rel_change < 5 else "[NOT CONVERGED]"
                    label_name = name.replace('_', ' ').title()
                    metrics.append(f"Thermal {label_name}: {rel_change:.1f}% {converged}")
        
        # Analyze modulus data
        if self.config.Grid_stability_modu:
            for name, data in self.config.Grid_stability_modu.items():
                if len(data) >= 2:
                    rel_change = abs((data[-1] - data[-2]) / data[-2] * 100) if data[-2] != 0 else 0
                    converged = "[CONVERGED]" if rel_change < 5 else "[NOT CONVERGED]"
                    label_name = name.replace('_', ' ').title()
                    metrics.append(f"Modulus {label_name}: {rel_change:.1f}% {converged}")
        
        if metrics:
            header = "Convergence Criteria: < 5% relative change between last two grids\n"
            return header + "\n".join(metrics)
        return ""

    def _print_convergence_summary(self):
        """Print detailed convergence summary to console"""
        print("\n" + "="*70)
        print(" "*25 + "CONVERGENCE SUMMARY")
        print("="*70)
        
        # Thermal expansion analysis
        if self.config.Grid_stability_ther:
            print("\n>>> Thermal Expansion Coefficient (dε/dT):")
            print("-"*50)
            for name, data in self.config.Grid_stability_ther.items():
                if len(data) > 0:
                    label_name = name.replace('_', ' ').upper()
                    print(f"\n  {label_name}:")
                    print(f"    Final value: {data[-1]:.3e}")
                    
                    if len(data) >= 2:
                        abs_change = data[-1] - data[-2]
                        rel_change = abs((data[-1] - data[-2]) / data[-2] * 100) if data[-2] != 0 else 0
                        print(f"    Absolute change: {abs_change:.3e}")
                        print(f"    Relative change: {rel_change:.2f}%")
                        
                        if rel_change < 5:
                            print(f"    Status: [CONVERGED]")
                        else:
                            print(f"    Status: [NOT CONVERGED]")
        
        # Elastic modulus analysis
        if self.config.Grid_stability_modu:
            print("\n>>> Elastic Modulus (dσ/dε):")
            print("-"*50)
            for name, data in self.config.Grid_stability_modu.items():
                if len(data) > 0:
                    label_name = name.replace('_', ' ').upper()
                    print(f"\n  {label_name}:")
                    print(f"    Final value: {data[-1]:.3e}")
                    
                    if len(data) >= 2:
                        abs_change = data[-1] - data[-2]
                        rel_change = abs((data[-1] - data[-2]) / data[-2] * 100) if data[-2] != 0 else 0
                        print(f"    Absolute change: {abs_change:.3e}")
                        print(f"    Relative change: {rel_change:.2f}%")
                        
                        if rel_change < 5:
                            print(f"    Status: [CONVERGED]")
                        else:
                            print(f"    Status: [NOT CONVERGED]")
        
        print("\n" + "="*70 + "\n")

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
        for dir_name in dir_names:
            all_processing_units, all_shapes, temp_map = self.processor.get_aver_scatter(dir_name, duty_type)
            self.visualizer.plot_scatter_analysis(all_processing_units, all_shapes, temp_map, duty_type, dir_name)
        print(f"Scatter analysis completed for {duty_type}\n")

def run_question_analysis(question: int, directories: List[str] = None) -> EnhancedDataSorter:
    base_path = "C:\\Users\\oft\\Documents\\ShenZhenCup\\output"
    print(f"=" * 60)
    print(f"Starting Analysis for Question {question}")
    print(f"=" * 60)
    
    sorter = EnhancedDataSorter(question, base_path)
    # sorter.generate_excel_and_plots(directories)
    print("Generating scatter plot analyses...")

    # Thermal scatter plot
    sorter.run_scatter_analysis(directories, 'thermal')
    # Modulus scatter plot
    sorter.run_scatter_analysis(directories, 'modulus')

    # sorter.visualizer.draw_stability()
    
    print(f"Analysis completed for Question {question}")
    print(f"=" * 60 + "\n")
    
    return sorter

def analyze_q1():

    print("\n" + "="*60)
    print("QUESTION 1: BGA GRID REFINEMENT ANALYSIS")
    print("="*60)
    
    directories = ['Q1-3', 'Q1-2.5', 'Q1-2', 'Q1-1.5', 'Q1-1','Q1-0.5', 'Q1-0.3']
    sorter = run_question_analysis(1, directories)
    
    return sorter

def analyze_q2():

    print("\n" + "="*60)
    print("QUESTION 2: CHIP PRECISION GRID ANALYSIS")
    print("="*60)
    
    directories = ['Q2-3', 'Q2-2.5', 'Q2-2', 'Q2-1.5', 'Q2-1','Q2-0.5']
    sorter = run_question_analysis(2, directories)
    
    return sorter

def analyze_q3():

    print("\n" + "="*60)
    print("QUESTION 3: SOLDER BALL CONFIGURATION ANALYSIS")
    print("="*60)
    
    directories = ['Q3-3', 'Q3-2.5', 'Q3-2', 'Q3-1.5', 'Q3-1','Q3-0.5', 'Q3-0.3']
    sorter = run_question_analysis(3, directories)
    
    return sorter

def main():
    print("\n" + "="*70)
    print(" "*20 + "ENHANCED DATA ANALYSIS SYSTEM")
    print(" "*15 + "Academic Publication Quality Output")
    print("="*70)

    if True: 
        print("\n Running Basic Analysis for All Questions")
        print("-"*50)
        # q1_sorter = analyze_q1()
        # q2_sorter = analyze_q2()
        q3_sorter = analyze_q3()
    
    print("\n" + "="*70)
    print(" "*20 + "ANALYSIS PIPELINE COMPLETED")
    print(" "*15 + "All outputs saved to configured directories")
    print("="*70 + "\n")
    
if __name__ == "__main__":
    main()