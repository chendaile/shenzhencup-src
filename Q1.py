import os

system_path = "C:\\Users\\oft\\Documents\\ShenZhenCup\\output\\Q1\\Q1-3\\"
def solve_heat(mag, heat_system_id = 9, inter_heat_id = 3,
                        solution_id = 4, tem_id=1):
    global heat_system, heat_solution, tem
    
    heat_system = Model.Children[heat_system_id]
    inter_heat = heat_system.Children[inter_heat_id]
    inter_heat.Magnitude.Output.SetDiscreteValue(0, Quantity(mag, "W mm^-1 mm^-1 mm^-1"))
    
    heat_solution = heat_system.Children[solution_id]
    tem = heat_solution.Children[tem_id]
    
    tem.Identifier = r''
    heat_system.Solve()
    
    tem.Identifier = r'tem'
    
    return tem.Average
    
# solve_heat(2e-4) 

def solve_static(static_system_id=10, solution_id=3):
    global static_system, static_solution
    static_system = Model.Children[static_system_id]
    static_solution = static_system.Children[solution_id]
    
    static_system.Solve()
    
# solve_static()

def photo_mesh(path=system_path):
    mesh_result = Model.Mesh
    mesh_result.Activate()
    
    Graphics.Camera.FocalPoint = Point([0.00029416287495454175, 0.0023687992055113683, 0.00019064117765789295], 'm')
    Graphics.Camera.ViewVector = Vector3D(-0.75671963101900463, 0.5424853546892312, -0.36480822356158649)
    Graphics.Camera.UpVector = Vector3D(0.62880546470334509, 0.75663703100266444, -0.17917614482593175)
    Graphics.Camera.SceneHeight = Quantity(0.0028335438343695118, 'm')
    Graphics.Camera.SceneWidth = Quantity(0.0047682016942846538, 'm')
    
    image_settings = Ansys.Mechanical.Graphics.GraphicsImageExportSettings()
    image_settings.CurrentGraphicsDisplay = False
    image_settings.Resolution = GraphicsResolutionType.EnhancedResolution
    image_settings.Capture = GraphicsCaptureType.ImageAndLegend
    image_settings.Background = GraphicsBackgroundType.GraphicsAppearanceSetting
    image_settings.FontMagnification =0.8
    Graphics.ExportImage(path + mesh_result.Name + ".png", GraphicsImageExportFormat.PNG, image_settings)

# photo_mesh()

def photo_tem(path=system_path):
    tem_result = tem
    tem_result.Activate()
    
    Graphics.Camera.FocalPoint = Point([-5.7228633038315549e-05, 0.0013197039236377891, -0.00019106176157549745], 'm')
    Graphics.Camera.ViewVector = Vector3D(-0.61040465245721842, 0.65158655681870192, -0.45037886188378229)
    Graphics.Camera.UpVector = Vector3D(0.75986738291073153, 0.64222945770018025, -0.10071188634182195)
    Graphics.Camera.SceneHeight = Quantity(0.0050952141637961744, 'm')
    Graphics.Camera.SceneWidth = Quantity(0.008574071985006702, 'm')
    
    image_settings = Ansys.Mechanical.Graphics.GraphicsImageExportSettings()
    image_settings.CurrentGraphicsDisplay = False
    image_settings.Resolution = GraphicsResolutionType.EnhancedResolution
    image_settings.Capture = GraphicsCaptureType.ImageAndLegend
    image_settings.Background = GraphicsBackgroundType.GraphicsAppearanceSetting
    image_settings.FontMagnification =0.8
    Graphics.ExportImage(path + tem_result.Name + ".png", GraphicsImageExportFormat.PNG, image_settings)

# photo_tem()

def export_path(project_id, tem_flag, path_ids = list(range(10, 20))+[5,6,7], save_path=system_path):
    target_path = save_path + 'id-' + str(project_id) + '-' + str(tem_flag) + '\\'
    os.makedirs(target_path)
    for path_id in path_ids:
        path = static_solution.Children[path_id]
        path.ExportToTextFile(target_path + path.Name + '.csv')
    
    path = heat_solution.Children[2]
    path.ExportToTextFile(target_path + path.Name + '.csv')

# export_path(1, 30)

def main(start=0, end=10):
    for project_id in range(start, end+1):
        mag = project_id * 1.5e-4 + 1e-5
        tem_flag = solve_heat(mag)
        solve_static()
        export_path(project_id, tem_flag)
        
main()