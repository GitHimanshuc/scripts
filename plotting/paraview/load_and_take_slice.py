from pathlib import Path

pvd_file_path = Path('/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/74_master_1_1/Ev_JustLogGamma_1_1/Lev1_AA/Run/GaugeVis_test.pvd')
save_path = Path("/panfs/ds09/sxs/himanshu/scripts/plotting/test")
var = "H"
component = "0"
component = "magnitude"
# trace generated using paraview version 5.10.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'PVD Reader'
gaugeVis_testpvd = PVDReader(registrationName='GaugeVis_test.pvd', FileName=str(pvd_file_path))
gaugeVis_testpvd.PointArrays = ['Lapse', 'Shift', 'H', 'GhCe', 'Detg', 'TrK', 'GaugeF', 'Theta', 'H_master']

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
gaugeVis_testpvdDisplay = Show(gaugeVis_testpvd, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
gaugeVis_testpvdDisplay.Representation = 'Surface'
gaugeVis_testpvdDisplay.ColorArrayName = [None, '']
gaugeVis_testpvdDisplay.SelectTCoordArray = 'None'
gaugeVis_testpvdDisplay.SelectNormalArray = 'None'
gaugeVis_testpvdDisplay.SelectTangentArray = 'None'
gaugeVis_testpvdDisplay.OSPRayScaleArray = 'Detg'
gaugeVis_testpvdDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
gaugeVis_testpvdDisplay.SelectOrientationVectors = 'GaugeF'
gaugeVis_testpvdDisplay.ScaleFactor = 5.9562531211049805
gaugeVis_testpvdDisplay.SelectScaleArray = 'Detg'
gaugeVis_testpvdDisplay.GlyphType = 'Arrow'
gaugeVis_testpvdDisplay.GlyphTableIndexArray = 'Detg'
gaugeVis_testpvdDisplay.GaussianRadius = 0.29781265605524904
gaugeVis_testpvdDisplay.SetScaleArray = ['POINTS', 'Detg']
gaugeVis_testpvdDisplay.ScaleTransferFunction = 'PiecewiseFunction'
gaugeVis_testpvdDisplay.OpacityArray = ['POINTS', 'Detg']
gaugeVis_testpvdDisplay.OpacityTransferFunction = 'PiecewiseFunction'
gaugeVis_testpvdDisplay.DataAxesGrid = 'GridAxesRepresentation'
gaugeVis_testpvdDisplay.PolarAxes = 'PolarAxesRepresentation'
gaugeVis_testpvdDisplay.ScalarOpacityUnitDistance = 2.4361840742060976
gaugeVis_testpvdDisplay.OpacityArrayName = ['POINTS', 'Detg']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
gaugeVis_testpvdDisplay.ScaleTransferFunction.Points = [1.21363500678979, 0.0, 0.5, 0.0, 9.62030679629937, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
gaugeVis_testpvdDisplay.OpacityTransferFunction.Points = [1.21363500678979, 0.0, 0.5, 0.0, 9.62030679629937, 1.0, 0.5, 0.0]

# reset view to fit data
renderView1.ResetCamera(False)

# get the material library
materialLibrary1 = GetMaterialLibrary()

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(gaugeVis_testpvdDisplay, ('FIELD', 'vtkBlockColors'))

# show color bar/color legend
gaugeVis_testpvdDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'vtkBlockColors'
vtkBlockColorsLUT = GetColorTransferFunction('vtkBlockColors')

# get opacity transfer function/opacity map for 'vtkBlockColors'
vtkBlockColorsPWF = GetOpacityTransferFunction('vtkBlockColors')

# create a new 'Slice'
slice1 = Slice(registrationName='Slice1', Input=gaugeVis_testpvd)
slice1.SliceType = 'Plane'
slice1.HyperTreeGridSlicer = 'Plane'
slice1.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice1.SliceType.Origin = [1.1302514479893944e-09, 0.0, -0.43587272957284995]

# init the 'Plane' selected for 'HyperTreeGridSlicer'
slice1.HyperTreeGridSlicer.Origin = [1.1302514479893944e-09, 0.0, -0.43587272957284995]

# Properties modified on slice1.SliceType
slice1.SliceType.Normal = [0.0, 0.0, 1.0]

# show data in view
slice1Display = Show(slice1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
slice1Display.Representation = 'Surface'
slice1Display.ColorArrayName = [None, '']
slice1Display.SelectTCoordArray = 'None'
slice1Display.SelectNormalArray = 'None'
slice1Display.SelectTangentArray = 'None'
slice1Display.OSPRayScaleArray = 'Detg'
slice1Display.OSPRayScaleFunction = 'PiecewiseFunction'
slice1Display.SelectOrientationVectors = 'GaugeF'
slice1Display.ScaleFactor = 5.95506786688013
slice1Display.SelectScaleArray = 'Detg'
slice1Display.GlyphType = 'Arrow'
slice1Display.GlyphTableIndexArray = 'Detg'
slice1Display.GaussianRadius = 0.29775339334400647
slice1Display.SetScaleArray = ['POINTS', 'Detg']
slice1Display.ScaleTransferFunction = 'PiecewiseFunction'
slice1Display.OpacityArray = ['POINTS', 'Detg']
slice1Display.OpacityTransferFunction = 'PiecewiseFunction'
slice1Display.DataAxesGrid = 'GridAxesRepresentation'
slice1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
slice1Display.ScaleTransferFunction.Points = [1.2136356195075588, 0.0, 0.5, 0.0, 9.504245735926302, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
slice1Display.OpacityTransferFunction.Points = [1.2136356195075588, 0.0, 0.5, 0.0, 9.504245735926302, 1.0, 0.5, 0.0]

# hide data in view
Hide(gaugeVis_testpvd, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(slice1Display, ('FIELD', 'vtkBlockColors'))

# show color bar/color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# set scalar coloring
ColorBy(slice1Display, ('POINTS', var, component))
# ColorBy(slice1Display, ('POINTS', 'Lapse'))
# ColorBy(slice1Display, ('POINTS', 'H', 'Magnitude'))
# ColorBy(slice1Display, ('POINTS', 'H', '0'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(vtkBlockColorsLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
slice1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'H'
hLUT = GetColorTransferFunction(var)

# get opacity transfer function/opacity map for 'H'
hPWF = GetOpacityTransferFunction(var)

# Properties modified on animationScene1
animationScene1.AnimationTime = 30.305047332645

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# Rescale transfer function
hLUT.RescaleTransferFunction(6.1002417265091616e-05, 1.8521081105362096)

# Rescale transfer function
hPWF.RescaleTransferFunction(6.1002417265091616e-05, 1.8521081105362096)

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(2105, 1121)

# current camera placement for renderView1
renderView1.CameraPosition = [1.1302514479893944e-09, 0.0, 29.115025858860385]
renderView1.CameraFocalPoint = [1.1302514479893944e-09, 0.0, -0.43587272957284995]
renderView1.CameraParallelScale = 51.45417571036033


# create a new 'Annotate Time Filter'
annotateTimeFilter1 = AnnotateTimeFilter(registrationName='AnnotateTimeFilter1', Input=slice1)

# show data in view
annotateTimeFilter1Display = Show(annotateTimeFilter1, renderView1, 'TextSourceRepresentation')

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on annotateTimeFilter1Display
annotateTimeFilter1Display.WindowLocation = 'Upper Center'

# Properties modified on annotateTimeFilter1Display
annotateTimeFilter1Display.FontSize = 25

# Properties modified on annotateTimeFilter1Display
annotateTimeFilter1Display.Color = [0.0, 0.0, 0.0]

# save animation
SaveAnimation(f'{save_path}/{var}.png', renderView1, ImageResolution=[2105, 1121],
    FrameRate=5,
    FrameWindow=[0, 13])
