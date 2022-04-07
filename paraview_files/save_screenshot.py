# trace generated using paraview version 5.10.0
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10

save_folder = "/home/himanshu/Desktop/temp/report_figs/temp/"

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Unstructured Grid Reader'
sphereA0_00000000000000000e00vtu = XMLUnstructuredGridReader(registrationName='SphereA0_0.0000000000000000e+00.vtu', FileName=['/home/himanshu/Desktop/spec/Tests/BlackBoxTests/GeneralizedHarmonicExamples/BBHLong/Run/Lev0_AA/GaugeVis/SphereA0_0.0000000000000000e+00.vtu'])
sphereA0_00000000000000000e00vtu.PointArrayStatus = ['Lapse', 'Shift', 'SecondaryWeightZ', 'H', 'H3', 'dxHNumericalFlattened', 'SpacetimeDerivOfHFlattened', 'gTargetShift', 'gTargetShift3', 'dxgTargetShiftNumericalFlattened', 'SpacetimeDerivOfgTargetShiftFlattened', 'TargetShift', 'TargetShift3', 'dxTargetShiftNumericalFlattened', 'SpacetimeDerivOfTargetShiftFlattened']

# create a new 'XML Unstructured Grid Reader'
sphereA0_12999999999999999e04vtu = XMLUnstructuredGridReader(registrationName='SphereA0_1.2999999999999999e-04.vtu', FileName=['/home/himanshu/Desktop/spec/Tests/BlackBoxTests/GeneralizedHarmonicExamples/BBHLong/Run/Lev0_AA/GaugeVis/SphereA0_1.2999999999999999e-04.vtu'])
sphereA0_12999999999999999e04vtu.PointArrayStatus = ['Lapse', 'Shift', 'SecondaryWeightZ', 'H', 'H3', 'dxHNumericalFlattened', 'SpacetimeDerivOfHFlattened', 'gTargetShift', 'gTargetShift3', 'dxgTargetShiftNumericalFlattened', 'SpacetimeDerivOfgTargetShiftFlattened', 'TargetShift', 'TargetShift3', 'dxTargetShiftNumericalFlattened', 'SpacetimeDerivOfTargetShiftFlattened']

# Properties modified on sphereA0_12999999999999999e04vtu
sphereA0_12999999999999999e04vtu.TimeArray = 'None'

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
sphereA0_12999999999999999e04vtuDisplay = Show(sphereA0_12999999999999999e04vtu, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
sphereA0_12999999999999999e04vtuDisplay.Representation = 'Surface'
sphereA0_12999999999999999e04vtuDisplay.ColorArrayName = [None, '']
sphereA0_12999999999999999e04vtuDisplay.SelectTCoordArray = 'None'
sphereA0_12999999999999999e04vtuDisplay.SelectNormalArray = 'None'
sphereA0_12999999999999999e04vtuDisplay.SelectTangentArray = 'None'
sphereA0_12999999999999999e04vtuDisplay.OSPRayScaleArray = 'H'
sphereA0_12999999999999999e04vtuDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
sphereA0_12999999999999999e04vtuDisplay.SelectOrientationVectors = 'H'
sphereA0_12999999999999999e04vtuDisplay.ScaleFactor = 0.28759257575172015
sphereA0_12999999999999999e04vtuDisplay.SelectScaleArray = 'H3'
sphereA0_12999999999999999e04vtuDisplay.GlyphType = 'Arrow'
sphereA0_12999999999999999e04vtuDisplay.GlyphTableIndexArray = 'H'
sphereA0_12999999999999999e04vtuDisplay.GaussianRadius = 0.014379628787586007
sphereA0_12999999999999999e04vtuDisplay.SetScaleArray = ['POINTS', 'H']
sphereA0_12999999999999999e04vtuDisplay.ScaleTransferFunction = 'PiecewiseFunction'
sphereA0_12999999999999999e04vtuDisplay.OpacityArray = ['POINTS', 'H']
sphereA0_12999999999999999e04vtuDisplay.OpacityTransferFunction = 'PiecewiseFunction'
sphereA0_12999999999999999e04vtuDisplay.DataAxesGrid = 'GridAxesRepresentation'
sphereA0_12999999999999999e04vtuDisplay.PolarAxes = 'PolarAxesRepresentation'
sphereA0_12999999999999999e04vtuDisplay.ScalarOpacityUnitDistance = 0.38422267757342804
sphereA0_12999999999999999e04vtuDisplay.OpacityArrayName = ['POINTS', 'H']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
sphereA0_12999999999999999e04vtuDisplay.ScaleTransferFunction.Points = [-1.61173511324884, 0.0, 0.5, 0.0, -0.26557743119686394, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
sphereA0_12999999999999999e04vtuDisplay.OpacityTransferFunction.Points = [-1.61173511324884, 0.0, 0.5, 0.0, -0.26557743119686394, 1.0, 0.5, 0.0]

# reset view to fit data
renderView1.ResetCamera(False)

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Properties modified on sphereA0_00000000000000000e00vtu
sphereA0_00000000000000000e00vtu.TimeArray = 'None'

# show data in view
sphereA0_00000000000000000e00vtuDisplay = Show(sphereA0_00000000000000000e00vtu, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
sphereA0_00000000000000000e00vtuDisplay.Representation = 'Surface'
sphereA0_00000000000000000e00vtuDisplay.ColorArrayName = [None, '']
sphereA0_00000000000000000e00vtuDisplay.SelectTCoordArray = 'None'
sphereA0_00000000000000000e00vtuDisplay.SelectNormalArray = 'None'
sphereA0_00000000000000000e00vtuDisplay.SelectTangentArray = 'None'
sphereA0_00000000000000000e00vtuDisplay.OSPRayScaleArray = 'H'
sphereA0_00000000000000000e00vtuDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
sphereA0_00000000000000000e00vtuDisplay.SelectOrientationVectors = 'H'
sphereA0_00000000000000000e00vtuDisplay.ScaleFactor = 0.2875925757517798
sphereA0_00000000000000000e00vtuDisplay.SelectScaleArray = 'H3'
sphereA0_00000000000000000e00vtuDisplay.GlyphType = 'Arrow'
sphereA0_00000000000000000e00vtuDisplay.GlyphTableIndexArray = 'H'
sphereA0_00000000000000000e00vtuDisplay.GaussianRadius = 0.014379628787588992
sphereA0_00000000000000000e00vtuDisplay.SetScaleArray = ['POINTS', 'H']
sphereA0_00000000000000000e00vtuDisplay.ScaleTransferFunction = 'PiecewiseFunction'
sphereA0_00000000000000000e00vtuDisplay.OpacityArray = ['POINTS', 'H']
sphereA0_00000000000000000e00vtuDisplay.OpacityTransferFunction = 'PiecewiseFunction'
sphereA0_00000000000000000e00vtuDisplay.DataAxesGrid = 'GridAxesRepresentation'
sphereA0_00000000000000000e00vtuDisplay.PolarAxes = 'PolarAxesRepresentation'
sphereA0_00000000000000000e00vtuDisplay.ScalarOpacityUnitDistance = 0.3842226775734826
sphereA0_00000000000000000e00vtuDisplay.OpacityArrayName = ['POINTS', 'H']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
sphereA0_00000000000000000e00vtuDisplay.ScaleTransferFunction.Points = [-1.61173511320675, 0.0, 0.5, 0.0, -0.26557743059290395, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
sphereA0_00000000000000000e00vtuDisplay.OpacityTransferFunction.Points = [-1.61173511320675, 0.0, 0.5, 0.0, -0.26557743059290395, 1.0, 0.5, 0.0]

# update the view to ensure updated data information
renderView1.Update()

# set active source
SetActiveSource(sphereA0_00000000000000000e00vtu)

# set active source
SetActiveSource(sphereA0_12999999999999999e04vtu)

# create a new 'Programmable Filter'
programmableFilter1 = ProgrammableFilter(registrationName='ProgrammableFilter1', Input=[sphereA0_00000000000000000e00vtu, sphereA0_12999999999999999e04vtu])
programmableFilter1.Script = ''
programmableFilter1.RequestInformationScript = ''
programmableFilter1.RequestUpdateExtentScript = ''
programmableFilter1.PythonPath = ''

# Properties modified on programmableFilter1
programmableFilter1.Script = """from paraview.vtk.numpy_interface import dataset_adapter as dsa
from paraview.vtk.numpy_interface import algorithms as algs
import numpy as np

#####################################################################################
# Load data
#####################################################################################

dt = 1.299999999999999999e-04
t1 = inputs[0]
t2 = inputs[1]

# Write data with another name
#for var_name in t1.PointData.keys():
#  output.PointData.append(t1.PointData[var_name],var_name+"_0")
#  output.PointData.append(data1.PointData[var_name],var_name+"_1")


#####################################################################################
# Helper functions
#####################################################################################

dxx_indices=[5,6,7,9,10,11,13,14,15] # spatial derivatives of spatial indices
dx_indices=[1,2,3,5,6,7,9,10,11,13,14,15] # spatial derivatives of spacetime indices

def find_time_derivative(t1_data,t2_data,dt):
  return (t2_data-t1_data)/dt

def find_rel_and_abs_diff(data1,data2,indices):
  rel_diff = np.zeros((data1.shape[0],len(indices)))
  abs_diff = np.zeros((data1.shape[0],len(indices)))

  for i,j in enumerate(indices):
    abs_diff[:, i] = (data1[:, j] - data2[:, j])

    numerator = np.abs(data1[:, j])+1e-25
    rel_diff[:, i] = (data1[:, j] - data2[:, j])/numerator
    rel_diff[abs_diff<1e-10] = 1e-10

  return rel_diff,abs_diff


#####################################################################################
# Space derivs and their diff for target shift
#####################################################################################
dxnum = t1.PointData["dxTargetShiftNumericalFlattened"]
dxana = t1.PointData["SpacetimeDerivOfTargetShiftFlattened"]
rel_dx_diff,abs_dx_diff = find_rel_and_abs_diff(dxana,dxnum,dxx_indices)
output.PointData.append(abs_dx_diff,"pv_dx_diff_abs_TargetShift")
output.PointData.append(rel_dx_diff,"pv_dx_diff_rel_TargetShift")


# Time derivative
fddTargetShift = find_time_derivative(t1.PointData["TargetShift"],
                                      t2.PointData["TargetShift"],dt)
output.PointData.append(fddTargetShift,"pv_dtTargetShiftNumerical")
                               
rel_dt_diff,abs_dt_diff = find_rel_and_abs_diff(dxana[:,np.array([0,4,8,12])],
                                                fddTargetShift,[0,1,2,3])
output.PointData.append(abs_dt_diff,"pv_dt_diff_abs_TargetShift")
output.PointData.append(rel_dt_diff,"pv_dt_diff_rel_TargetShift")


#####################################################################################
# Space derivs and their diff for gtargetshift
#####################################################################################
dxnum = t1.PointData["dxgTargetShiftNumericalFlattened"]
dxana = t1.PointData["SpacetimeDerivOfgTargetShiftFlattened"]
rel_dx_diff,abs_dx_diff = find_rel_and_abs_diff(dxana,dxnum,dxx_indices)
output.PointData.append(abs_dx_diff,"pv_dx_diff_abs_gTargetShift")
output.PointData.append(rel_dx_diff,"pv_dx_diff_rel_gTargetShift")


# Time derivative
fddgTargetShift = find_time_derivative(t1.PointData["gTargetShift"],
                                      t2.PointData["gTargetShift"],dt)
output.PointData.append(fddgTargetShift,"pv_dtgTargetShiftNumerical")
                               
rel_dt_diff,abs_dt_diff = find_rel_and_abs_diff(dxana[:,np.array([0,4,8,12])],
                                                fddgTargetShift,[0,1,2,3])
output.PointData.append(abs_dt_diff,"pv_dt_diff_abs_gTargetShift")
output.PointData.append(rel_dt_diff,"pv_dt_diff_rel_gTargetShift")



#####################################################################################
# Space derivs and their diff for gtargetshift
#####################################################################################
dxnum = t1.PointData["dxHNumericalFlattened"]
dxana = t1.PointData["SpacetimeDerivOfHFlattened"]
rel_dx_diff,abs_dx_diff = find_rel_and_abs_diff(dxana,dxnum,dxx_indices)
output.PointData.append(abs_dx_diff,"pv_dx_diff_abs_H")
output.PointData.append(rel_dx_diff,"pv_dx_diff_rel_H")


# Time derivative
fddH = find_time_derivative(t1.PointData["H"],
                                      t2.PointData["H"],dt)
output.PointData.append(fddH,"pv_dtHNumerical")
                               
rel_dt_diff,abs_dt_diff = find_rel_and_abs_diff(dxana[:,np.array([0,4,8,12])],
                                                fddH,[0,1,2,3])
output.PointData.append(abs_dt_diff,"pv_dt_diff_abs_H")
output.PointData.append(rel_dt_diff,"pv_dt_diff_rel_H")
"""
programmableFilter1.RequestInformationScript = ''
programmableFilter1.RequestUpdateExtentScript = ''
programmableFilter1.PythonPath = ''

# show data in view
programmableFilter1Display = Show(programmableFilter1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
programmableFilter1Display.Representation = 'Surface'
programmableFilter1Display.ColorArrayName = [None, '']
programmableFilter1Display.SelectTCoordArray = 'None'
programmableFilter1Display.SelectNormalArray = 'None'
programmableFilter1Display.SelectTangentArray = 'None'
programmableFilter1Display.OSPRayScaleArray = 'pv_dtHNumerical'
programmableFilter1Display.OSPRayScaleFunction = 'PiecewiseFunction'
programmableFilter1Display.SelectOrientationVectors = 'None'
programmableFilter1Display.ScaleFactor = 0.2875925757517798
programmableFilter1Display.SelectScaleArray = 'None'
programmableFilter1Display.GlyphType = 'Arrow'
programmableFilter1Display.GlyphTableIndexArray = 'None'
programmableFilter1Display.GaussianRadius = 0.014379628787588992
programmableFilter1Display.SetScaleArray = ['POINTS', 'pv_dtHNumerical']
programmableFilter1Display.ScaleTransferFunction = 'PiecewiseFunction'
programmableFilter1Display.OpacityArray = ['POINTS', 'pv_dtHNumerical']
programmableFilter1Display.OpacityTransferFunction = 'PiecewiseFunction'
programmableFilter1Display.DataAxesGrid = 'GridAxesRepresentation'
programmableFilter1Display.PolarAxes = 'PolarAxesRepresentation'
programmableFilter1Display.ScalarOpacityUnitDistance = 0.3842226775734826
programmableFilter1Display.OpacityArrayName = ['POINTS', 'pv_dtHNumerical']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
programmableFilter1Display.ScaleTransferFunction.Points = [-1.9180084870767518e-05, 0.0, 0.5, 0.0, -3.06537701205274e-07, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
programmableFilter1Display.OpacityTransferFunction.Points = [-1.9180084870767518e-05, 0.0, 0.5, 0.0, -3.06537701205274e-07, 1.0, 0.5, 0.0]

# hide data in view
Hide(sphereA0_00000000000000000e00vtu, renderView1)

# hide data in view
Hide(sphereA0_12999999999999999e04vtu, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Slice'
slice1 = Slice(registrationName='Slice1', Input=programmableFilter1)
slice1.SliceType = 'Plane'
slice1.HyperTreeGridSlicer = 'Plane'
slice1.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice1.SliceType.Origin = [20.0, 0.0, 0.0]

# init the 'Plane' selected for 'HyperTreeGridSlicer'
slice1.HyperTreeGridSlicer.Origin = [20.0, 0.0, 0.0]

# reset view to fit data
renderView1.ResetCamera(False)

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
slice1Display.OSPRayScaleArray = 'pv_dtHNumerical'
slice1Display.OSPRayScaleFunction = 'PiecewiseFunction'
slice1Display.SelectOrientationVectors = 'None'
slice1Display.ScaleFactor = 0.2875925757517798
slice1Display.SelectScaleArray = 'None'
slice1Display.GlyphType = 'Arrow'
slice1Display.GlyphTableIndexArray = 'None'
slice1Display.GaussianRadius = 0.014379628787588992
slice1Display.SetScaleArray = ['POINTS', 'pv_dtHNumerical']
slice1Display.ScaleTransferFunction = 'PiecewiseFunction'
slice1Display.OpacityArray = ['POINTS', 'pv_dtHNumerical']
slice1Display.OpacityTransferFunction = 'PiecewiseFunction'
slice1Display.DataAxesGrid = 'GridAxesRepresentation'
slice1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
slice1Display.ScaleTransferFunction.Points = [-1.9180061812289312e-05, 0.0, 0.5, 0.0, -3.065385552229837e-07, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
slice1Display.OpacityTransferFunction.Points = [-1.9180061812289312e-05, 0.0, 0.5, 0.0, -3.065385552229837e-07, 1.0, 0.5, 0.0]

# hide data in view
Hide(programmableFilter1, renderView1)

#############################################################################################
# pv_dx_diff_abs_H
#############################################################################################

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
renderView1.ResetCamera(True)
ColorBy(slice1Display, ('POINTS', 'pv_dx_diff_abs_H', 'Magnitude'))
# rescale color and/or opacity maps used to include current data range
slice1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'pv_dx_diff_abs_H'
pv_dx_diff_abs_HLUT = GetColorTransferFunction('pv_dx_diff_abs_H')

# get opacity transfer function/opacity map for 'pv_dx_diff_abs_H'
pv_dx_diff_abs_HPWF = GetOpacityTransferFunction('pv_dx_diff_abs_H')

# rescale color and/or opacity maps used to exactly fit the current data range
slice1Display.RescaleTransferFunctionToDataRange(False, True)

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(901, 797)

# current camera placement for renderView1
renderView1.CameraPosition = [20.0, 0.0, 11.761937128780712]
renderView1.CameraFocalPoint = [20.0, 0.0, 0.0]
renderView1.CameraParallelScale = 3.044213336226908

renderView1.ResetCamera(True)
# save screenshot
figure_path=save_folder+'pv_dx_diff_abs_H'+'.png'
SaveScreenshot(figure_path, renderView1, ImageResolution=[901, 797])



#############################################################################################
# pv_dx_diff_rel_H
#############################################################################################

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
renderView1.ResetCamera(True)
ColorBy(slice1Display, ('POINTS', 'pv_dx_diff_rel_H', 'Magnitude'))
HideScalarBarIfNotNeeded(pv_dx_diff_abs_HLUT,renderView1)
# rescale color and/or opacity maps used to include current data range
slice1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'pv_dx_diff_rel_H'
pv_dx_diff_absrelUT = GetColorTransferFunction('pv_dx_diff_rel_H')

# get opacity transfer function/opacity map for 'pv_dx_diff_rel_H'
pv_dx_diff_absrelWF = GetOpacityTransferFunction('pv_dx_diff_rel_H')

# rescale color and/or opacity maps used to exactly fit the current data range
slice1Display.RescaleTransferFunctionToDataRange(False, True)

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(901, 797)

# current camera placement for renderView1
renderView1.CameraPosition = [20.0, 0.0, 11.761937128780712]
renderView1.CameraFocalPoint = [20.0, 0.0, 0.0]
renderView1.CameraParallelScale = 3.044213336226908

renderView1.ResetCamera(True)
# save screenshot
figure_path=save_folder+'pv_dx_diff_rel_H'+'.png'
SaveScreenshot(figure_path, renderView1, ImageResolution=[901, 797])



#############################################################################################
# pv_dt_diff_abs_H
#############################################################################################

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
renderView1.ResetCamera(True)
ColorBy(slice1Display, ('POINTS', 'pv_dt_diff_abs_H', 'Magnitude'))
HideScalarBarIfNotNeeded(pv_dx_diff_absrelUT,renderView1)
# rescale color and/or opacity maps used to include current data range
slice1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'pv_dt_diff_abs_H'
pv_dt_diff_abs_HLUT = GetColorTransferFunction('pv_dt_diff_abs_H')

# get opacity transfer function/opacity map for 'pv_dt_diff_abs_H'
pv_dt_diff_abs_HPWF = GetOpacityTransferFunction('pv_dt_diff_abs_H')

# rescale color and/or opacity maps used to exactly fit the current data range
slice1Display.RescaleTransferFunctionToDataRange(False, True)

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(901, 797)

# current camera placement for renderView1
renderView1.CameraPosition = [20.0, 0.0, 11.761937128780712]
renderView1.CameraFocalPoint = [20.0, 0.0, 0.0]
renderView1.CameraParallelScale = 3.044213336226908

renderView1.ResetCamera(True)
# save screenshot
figure_path=save_folder+'pv_dt_diff_abs_H'+'.png'
SaveScreenshot(figure_path, renderView1, ImageResolution=[901, 797])



#############################################################################################
# pv_dt_diff_rel_H
#############################################################################################

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
renderView1.ResetCamera(True)
ColorBy(slice1Display, ('POINTS', 'pv_dt_diff_rel_H', 'Magnitude'))
HideScalarBarIfNotNeeded(pv_dt_diff_abs_HLUT,renderView1)
# rescale color and/or opacity maps used to include current data range
slice1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'pv_dt_diff_rel_H'
pv_dt_diff_absrelUT = GetColorTransferFunction('pv_dt_diff_rel_H')

# get opacity transfer function/opacity map for 'pv_dt_diff_rel_H'
pv_dt_diff_absrelWF = GetOpacityTransferFunction('pv_dt_diff_rel_H')

# rescale color and/or opacity maps used to exactly fit the current data range
slice1Display.RescaleTransferFunctionToDataRange(False, True)

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(901, 797)

# current camera placement for renderView1
renderView1.CameraPosition = [20.0, 0.0, 11.761937128780712]
renderView1.CameraFocalPoint = [20.0, 0.0, 0.0]
renderView1.CameraParallelScale = 3.044213336226908

renderView1.ResetCamera(True)
# save screenshot
figure_path=save_folder+'pv_dt_diff_rel_H'+'.png'
SaveScreenshot(figure_path, renderView1, ImageResolution=[901, 797])



#############################################################################################
# pv_dx_diff_abs_TargetShift
#############################################################################################

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
renderView1.ResetCamera(True)
ColorBy(slice1Display, ('POINTS', 'pv_dx_diff_abs_TargetShift', 'Magnitude'))
HideScalarBarIfNotNeeded(pv_dt_diff_absrelUT,renderView1)
# rescale color and/or opacity maps used to include current data range
slice1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'pv_dx_diff_abs_TargetShift'
pv_dx_diff_abs_TargetShiftLUT = GetColorTransferFunction('pv_dx_diff_abs_TargetShift')

# get opacity transfer function/opacity map for 'pv_dx_diff_abs_TargetShift'
pv_dx_diff_abs_TargetShiftPWF = GetOpacityTransferFunction('pv_dx_diff_abs_TargetShift')

# rescale color and/or opacity maps used to exactly fit the current data range
slice1Display.RescaleTransferFunctionToDataRange(False, True)

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(901, 797)

# current camera placement for renderView1
renderView1.CameraPosition = [20.0, 0.0, 11.761937128780712]
renderView1.CameraFocalPoint = [20.0, 0.0, 0.0]
renderView1.CameraParallelScale = 3.044213336226908

renderView1.ResetCamera(True)
# save screenshot
figure_path=save_folder+'pv_dx_diff_abs_TargetShift'+'.png'
SaveScreenshot(figure_path, renderView1, ImageResolution=[901, 797])



#############################################################################################
# pv_dx_diff_rel_TargetShift
#############################################################################################

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
renderView1.ResetCamera(True)
ColorBy(slice1Display, ('POINTS', 'pv_dx_diff_rel_TargetShift', 'Magnitude'))
HideScalarBarIfNotNeeded(pv_dx_diff_abs_TargetShiftLUT,renderView1)
# rescale color and/or opacity maps used to include current data range
slice1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'pv_dx_diff_rel_TargetShift'
pv_dx_diff_absrelUT = GetColorTransferFunction('pv_dx_diff_rel_TargetShift')

# get opacity transfer function/opacity map for 'pv_dx_diff_rel_TargetShift'
pv_dx_diff_absrelWF = GetOpacityTransferFunction('pv_dx_diff_rel_TargetShift')

# rescale color and/or opacity maps used to exactly fit the current data range
slice1Display.RescaleTransferFunctionToDataRange(False, True)

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(901, 797)

# current camera placement for renderView1
renderView1.CameraPosition = [20.0, 0.0, 11.761937128780712]
renderView1.CameraFocalPoint = [20.0, 0.0, 0.0]
renderView1.CameraParallelScale = 3.044213336226908

renderView1.ResetCamera(True)
# save screenshot
figure_path=save_folder+'pv_dx_diff_rel_TargetShift'+'.png'
SaveScreenshot(figure_path, renderView1, ImageResolution=[901, 797])



#############################################################################################
# pv_dt_diff_abs_TargetShift
#############################################################################################

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
renderView1.ResetCamera(True)
ColorBy(slice1Display, ('POINTS', 'pv_dt_diff_abs_TargetShift', 'Magnitude'))
HideScalarBarIfNotNeeded(pv_dx_diff_absrelUT,renderView1)
# rescale color and/or opacity maps used to include current data range
slice1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'pv_dt_diff_abs_TargetShift'
pv_dt_diff_abs_TargetShiftLUT = GetColorTransferFunction('pv_dt_diff_abs_TargetShift')

# get opacity transfer function/opacity map for 'pv_dt_diff_abs_TargetShift'
pv_dt_diff_abs_TargetShiftPWF = GetOpacityTransferFunction('pv_dt_diff_abs_TargetShift')

# rescale color and/or opacity maps used to exactly fit the current data range
slice1Display.RescaleTransferFunctionToDataRange(False, True)

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(901, 797)

# current camera placement for renderView1
renderView1.CameraPosition = [20.0, 0.0, 11.761937128780712]
renderView1.CameraFocalPoint = [20.0, 0.0, 0.0]
renderView1.CameraParallelScale = 3.044213336226908

renderView1.ResetCamera(True)
# save screenshot
figure_path=save_folder+'pv_dt_diff_abs_TargetShift'+'.png'
SaveScreenshot(figure_path, renderView1, ImageResolution=[901, 797])



#############################################################################################
# pv_dt_diff_rel_TargetShift
#############################################################################################

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
renderView1.ResetCamera(True)
ColorBy(slice1Display, ('POINTS', 'pv_dt_diff_rel_TargetShift', 'Magnitude'))
HideScalarBarIfNotNeeded(pv_dt_diff_abs_TargetShiftLUT,renderView1)
# rescale color and/or opacity maps used to include current data range
slice1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'pv_dt_diff_rel_TargetShift'
pv_dt_diff_absrelUT = GetColorTransferFunction('pv_dt_diff_rel_TargetShift')

# get opacity transfer function/opacity map for 'pv_dt_diff_rel_TargetShift'
pv_dt_diff_absrelWF = GetOpacityTransferFunction('pv_dt_diff_rel_TargetShift')

# rescale color and/or opacity maps used to exactly fit the current data range
slice1Display.RescaleTransferFunctionToDataRange(False, True)

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(901, 797)

# current camera placement for renderView1
renderView1.CameraPosition = [20.0, 0.0, 11.761937128780712]
renderView1.CameraFocalPoint = [20.0, 0.0, 0.0]
renderView1.CameraParallelScale = 3.044213336226908

renderView1.ResetCamera(True)
# save screenshot
figure_path=save_folder+'pv_dt_diff_rel_TargetShift'+'.png'
SaveScreenshot(figure_path, renderView1, ImageResolution=[901, 797])



#############################################################################################
# pv_dx_diff_abs_gTargetShift
#############################################################################################

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
renderView1.ResetCamera(True)
ColorBy(slice1Display, ('POINTS', 'pv_dx_diff_abs_gTargetShift', 'Magnitude'))
HideScalarBarIfNotNeeded(pv_dt_diff_absrelUT,renderView1)
# rescale color and/or opacity maps used to include current data range
slice1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'pv_dx_diff_abs_gTargetShift'
pv_dx_diff_abs_TarggetShiftLUT = GetColorTransferFunction('pv_dx_diff_abs_gTargetShift')

# get opacity transfer function/opacity map for 'pv_dx_diff_abs_gTargetShift'
pv_dx_diff_abs_TarggetShiftPWF = GetOpacityTransferFunction('pv_dx_diff_abs_gTargetShift')

# rescale color and/or opacity maps used to exactly fit the current data range
slice1Display.RescaleTransferFunctionToDataRange(False, True)

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(901, 797)

# current camera placement for renderView1
renderView1.CameraPosition = [20.0, 0.0, 11.761937128780712]
renderView1.CameraFocalPoint = [20.0, 0.0, 0.0]
renderView1.CameraParallelScale = 3.044213336226908

renderView1.ResetCamera(True)
# save screenshot
figure_path=save_folder+'pv_dx_diff_abs_gTargetShift'+'.png'
SaveScreenshot(figure_path, renderView1, ImageResolution=[901, 797])



#############################################################################################
# pv_dx_diff_rel_gTargetShift
#############################################################################################

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
renderView1.ResetCamera(True)
ColorBy(slice1Display, ('POINTS', 'pv_dx_diff_rel_gTargetShift', 'Magnitude'))
HideScalarBarIfNotNeeded(pv_dx_diff_abs_TarggetShiftLUT,renderView1)
# rescale color and/or opacity maps used to include current data range
slice1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'pv_dx_diff_rel_gTargetShift'
pv_dx_diff_absrelUT = GetColorTransferFunction('pv_dx_diff_rel_gTargetShift')

# get opacity transfer function/opacity map for 'pv_dx_diff_rel_gTargetShift'
pv_dx_diff_absrelWF = GetOpacityTransferFunction('pv_dx_diff_rel_gTargetShift')

# rescale color and/or opacity maps used to exactly fit the current data range
slice1Display.RescaleTransferFunctionToDataRange(False, True)

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(901, 797)

# current camera placement for renderView1
renderView1.CameraPosition = [20.0, 0.0, 11.761937128780712]
renderView1.CameraFocalPoint = [20.0, 0.0, 0.0]
renderView1.CameraParallelScale = 3.044213336226908

renderView1.ResetCamera(True)
# save screenshot
figure_path=save_folder+'pv_dx_diff_rel_gTargetShift'+'.png'
SaveScreenshot(figure_path, renderView1, ImageResolution=[901, 797])



#############################################################################################
# pv_dt_diff_abs_gTargetShift
#############################################################################################

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
renderView1.ResetCamera(True)
ColorBy(slice1Display, ('POINTS', 'pv_dt_diff_abs_gTargetShift', 'Magnitude'))
HideScalarBarIfNotNeeded(pv_dx_diff_absrelUT,renderView1)
# rescale color and/or opacity maps used to include current data range
slice1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'pv_dt_diff_abs_gTargetShift'
pv_dt_diff_abs_TarggetShiftLUT = GetColorTransferFunction('pv_dt_diff_abs_gTargetShift')

# get opacity transfer function/opacity map for 'pv_dt_diff_abs_gTargetShift'
pv_dt_diff_abs_TarggetShiftPWF = GetOpacityTransferFunction('pv_dt_diff_abs_gTargetShift')

# rescale color and/or opacity maps used to exactly fit the current data range
slice1Display.RescaleTransferFunctionToDataRange(False, True)

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(901, 797)

# current camera placement for renderView1
renderView1.CameraPosition = [20.0, 0.0, 11.761937128780712]
renderView1.CameraFocalPoint = [20.0, 0.0, 0.0]
renderView1.CameraParallelScale = 3.044213336226908

renderView1.ResetCamera(True)
# save screenshot
figure_path=save_folder+'pv_dt_diff_abs_gTargetShift'+'.png'
SaveScreenshot(figure_path, renderView1, ImageResolution=[901, 797])



#############################################################################################
# pv_dt_diff_rel_gTargetShift
#############################################################################################

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
renderView1.ResetCamera(True)
ColorBy(slice1Display, ('POINTS', 'pv_dt_diff_rel_gTargetShift', 'Magnitude'))
HideScalarBarIfNotNeeded(pv_dt_diff_abs_TarggetShiftLUT,renderView1)
# rescale color and/or opacity maps used to include current data range
slice1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'pv_dt_diff_rel_gTargetShift'
pv_dt_diff_absrelUT = GetColorTransferFunction('pv_dt_diff_rel_gTargetShift')

# get opacity transfer function/opacity map for 'pv_dt_diff_rel_gTargetShift'
pv_dt_diff_absrelWF = GetOpacityTransferFunction('pv_dt_diff_rel_gTargetShift')

# rescale color and/or opacity maps used to exactly fit the current data range
slice1Display.RescaleTransferFunctionToDataRange(False, True)

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(901, 797)

# current camera placement for renderView1
renderView1.CameraPosition = [20.0, 0.0, 11.761937128780712]
renderView1.CameraFocalPoint = [20.0, 0.0, 0.0]
renderView1.CameraParallelScale = 3.044213336226908

renderView1.ResetCamera(True)
# save screenshot
figure_path=save_folder+'pv_dt_diff_rel_gTargetShift'+'.png'
SaveScreenshot(figure_path, renderView1, ImageResolution=[901, 797])


