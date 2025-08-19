import os 

system_path = "C:\\Users\\oft\\Documents\\ShenZhenCup\\output\\Q3\\photos\\"
def shot_heat(heat_system_id, shot_ids, path):
    heat_system = Model.Children[heat_system_id]
    heat_solution = heat_system.Solution
    for shot_id in shot_ids:
        target = heat_solution.Children[shot_id]
        target.Activate()

        Graphics.Camera.FocalPoint = Point([0.02120487928552528, 0.01012141299805879, 0.019735954116051418], 'm')
        Graphics.Camera.ViewVector = Vector3D(0.71665822652084554, 0.56961959284548658, 0.40241086690910616)
        Graphics.Camera.UpVector = Vector3D(-0.60928776626031811, 0.79212350647765362, -0.0361769038346108)
        Graphics.Camera.SceneHeight = Quantity(0.042569138053587925, 'm')
        Graphics.Camera.SceneWidth = Quantity(0.045126755026990414, 'm')
        
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
        
        Graphics.Camera.FocalPoint = Point([0.02120487928552528, 0.01012141299805879, 0.019735954116051418], 'm')
        Graphics.Camera.ViewVector = Vector3D(0.71665822652084554, 0.56961959284548658, 0.40241086690910616)
        Graphics.Camera.UpVector = Vector3D(-0.60928776626031811, 0.79212350647765362, -0.0361769038346108)
        Graphics.Camera.SceneHeight = Quantity(0.042569138053587925, 'm')
        Graphics.Camera.SceneWidth = Quantity(0.045126755026990414, 'm')

        image_settings = Ansys.Mechanical.Graphics.GraphicsImageExportSettings()
        image_settings.CurrentGraphicsDisplay = False
        image_settings.Resolution = GraphicsResolutionType.EnhancedResolution
        image_settings.Capture = GraphicsCaptureType.ImageAndLegend
        image_settings.Background = GraphicsBackgroundType.GraphicsAppearanceSetting
        image_settings.FontMagnification =0.8
        Graphics.ExportImage(path + target.Name + ".png", GraphicsImageExportFormat.PNG, image_settings)

# shot_heat(9,[1],system_path)
shot_mecha(10,[1,2,3,4,8,19],system_path)
