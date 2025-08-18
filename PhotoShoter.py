import os 

system_path = "C:\\Users\\oft\\Documents\\ShenZhenCup\\output\\Q3\\photos\\"
def shot_heat(heat_system_id, shot_ids, path):
    heat_system = Model.Children[heat_system_id]
    heat_solution = heat_system.Solution
    for shot_id in shot_ids:
        target = heat_solution.Children[shot_id]
        target.Activate()

        Graphics.Camera.FocalPoint = Point([0.010853337799413117, -0.0043197794684404428, 0.0097874877838722597], 'm')
        Graphics.Camera.ViewVector = Vector3D(-0.35786676134616696, 0.64196530692648701, 0.67809433401731567)
        Graphics.Camera.UpVector = Vector3D(0.14990526079355287, 0.75626631900403485, -0.63685922112073112)
        Graphics.Camera.SceneHeight = Quantity(0.039273547659927008, 'm')
        Graphics.Camera.SceneWidth = Quantity(0.041633160672864156, 'm')
        
        image_settings = Ansys.Mechanical.Graphics.GraphicsImageExportSettings()
        image_settings.CurrentGraphicsDisplay = False
        image_settings.Resolution = GraphicsResolutionType.EnhancedResolution
        image_settings.Capture = GraphicsCaptureType.ImageAndLegend
        image_settings.Background = GraphicsBackgroundType.GraphicsAppearanceSetting
        image_settings.FontMagnification =0.8
        Graphics.ExportImage(path + target.Name + ".png", GraphicsImageExportFormat.PNG, image_settings)

def shot_mecha(mecha_system_id, shot_ids, path):
    mecha_system = Model.Children[mecha_system_id]
    mecha_solution = mecha_system.Solution
    for shot_id in shot_ids:
        target = mecha_solution.Children[shot_id]
        target.Activate()
        
        Graphics.Camera.FocalPoint = Point([0.010853337799413117, -0.0043197794684404428, 0.0097874877838722597], 'm')
        Graphics.Camera.ViewVector = Vector3D(-0.35786676134616696, 0.64196530692648701, 0.67809433401731567)
        Graphics.Camera.UpVector = Vector3D(0.14990526079355287, 0.75626631900403485, -0.63685922112073112)
        Graphics.Camera.SceneHeight = Quantity(0.039273547659927008, 'm')
        Graphics.Camera.SceneWidth = Quantity(0.041633160672864156, 'm')
        
        image_settings = Ansys.Mechanical.Graphics.GraphicsImageExportSettings()
        image_settings.CurrentGraphicsDisplay = False
        image_settings.Resolution = GraphicsResolutionType.EnhancedResolution
        image_settings.Capture = GraphicsCaptureType.ImageAndLegend
        image_settings.Background = GraphicsBackgroundType.GraphicsAppearanceSetting
        image_settings.FontMagnification =0.8
        Graphics.ExportImage(path + target.Name + ".png", GraphicsImageExportFormat.PNG, image_settings)

shot_heat(9,[1],system_path)
shot_mecha(10,[1,2,3,4,5,16],system_path)


