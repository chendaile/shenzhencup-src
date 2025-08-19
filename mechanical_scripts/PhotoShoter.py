import os 

system_path = "C:\\Users\\oft\\Documents\\ShenZhenCup\\output\\Q1\\photos\\"
def shot_heat(heat_system_id, shot_ids, path):
    heat_system = Model.Children[heat_system_id]
    heat_solution = heat_system.Solution
    for shot_id in shot_ids:
        target = heat_solution.Children[shot_id]
        target.Activate()

        Graphics.Camera.FocalPoint = Point([0.012629139225755164, 0.0056962164435974631, -0.00081074499112088303], 'm')
        Graphics.Camera.ViewVector = Vector3D(-0.88375763363099136, 0.28407254606849802, -0.37185377982361772)
        Graphics.Camera.UpVector = Vector3D(0.32842939376355912, 0.94259052394059684, -0.060475098094111016)
        Graphics.Camera.SceneHeight = Quantity(0.017943329180931288, 'm')
        Graphics.Camera.SceneWidth = Quantity(0.030194490575319189, 'm')
        
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
        
        Graphics.Camera.FocalPoint = Point([0.012629139225755164, 0.0056962164435974631, -0.00081074499112088303], 'm')
        Graphics.Camera.ViewVector = Vector3D(-0.88375763363099136, 0.28407254606849802, -0.37185377982361772)
        Graphics.Camera.UpVector = Vector3D(0.32842939376355912, 0.94259052394059684, -0.060475098094111016)
        Graphics.Camera.SceneHeight = Quantity(0.017943329180931288, 'm')
        Graphics.Camera.SceneWidth = Quantity(0.030194490575319189, 'm')
        
        image_settings = Ansys.Mechanical.Graphics.GraphicsImageExportSettings()
        image_settings.CurrentGraphicsDisplay = False
        image_settings.Resolution = GraphicsResolutionType.EnhancedResolution
        image_settings.Capture = GraphicsCaptureType.ImageAndLegend
        image_settings.Background = GraphicsBackgroundType.GraphicsAppearanceSetting
        image_settings.FontMagnification =0.8
        Graphics.ExportImage(path + target.Name + ".png", GraphicsImageExportFormat.PNG, image_settings)

shot_heat(9,[1],system_path)
shot_mecha(10,[1,2,3,4,8,9],system_path)

