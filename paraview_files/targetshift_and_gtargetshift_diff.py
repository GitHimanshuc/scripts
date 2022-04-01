from paraview.vtk.numpy_interface import dataset_adapter as dsa
from paraview.vtk.numpy_interface import algorithms as algs
import numpy as np


data = inputs[0]
print(data.PointData.keys())
data1 = inputs[1]
print(data1.PointData.keys())

#####################################################################################
# Copy the rest of the data
#####################################################################################


for var_name in data.PointData.keys():
  output.PointData.append(data.PointData[var_name],var_name+"_0")
#  output.PointData.append(data1.PointData[var_name],var_name+"_1")

#####################################################################################
# Space derivs and their diff for target shift
#####################################################################################


dxnum = data.PointData["dxTargetShiftNumericalFlattened"]
dxana = data.PointData["SpacetimeDerivOfTargetShiftFlattened"]

print(algs.shape(dxnum))
print(algs.shape(dxana))

dx_diff = dxnum*0.0

dx_diff[:,0,0] = (dxana[:,5] - dxnum[:,0,0])
dx_diff[:,1,0] = (dxana[:,6] - dxnum[:,1,0])
dx_diff[:,2,0] = (dxana[:,7] - dxnum[:,2,0])

dx_diff[:,0,1] = (dxana[:,9] - dxnum[:,0,1])
dx_diff[:,1,1] = (dxana[:,10] - dxnum[:,1,1])
dx_diff[:,2,1] = (dxana[:,11] - dxnum[:,2,1])

dx_diff[:,0,2] = (dxana[:,13] - dxnum[:,0,2])
dx_diff[:,1,2] = (dxana[:,14] - dxnum[:,1,2])
dx_diff[:,2,2] = (dxana[:,15] - dxnum[:,2,2])
output.PointData.append(dx_diff,"pv_dx_diff")


dx_analytical= dxnum*0.0
dx_analytical[:,0,0] = dxana[:,5]
dx_analytical[:,1,0] = dxana[:,6]
dx_analytical[:,2,0] = dxana[:,7]

dx_analytical[:,0,1] = dxana[:,9]
dx_analytical[:,1,1] = dxana[:,10]
dx_analytical[:,2,1] = dxana[:,11]

dx_analytical[:,0,2] = dxana[:,13]
dx_analytical[:,1,2] = dxana[:,14]
dx_analytical[:,2,2] = dxana[:,15]
output.PointData.append(dx_analytical,"dx_TargetShiftanalytical")



#####################################################################################
# Time derivatives and their diff for target shift
#####################################################################################
tshift0 =  data.PointData["TargetShift"]
tshift1 =  data1.PointData["TargetShift"]
print(algs.shape(tshift0))
print(algs.shape(tshift1))


time_step = 1.299999999999999999e-04
fddTargetShift = (tshift1 - tshift0)/time_step
output.PointData.append(fddTargetShift,"pv_dtTargetShiftNumerical")

copy_dt_shift = tshift0*0.0
copy_dt_shift[:,0] = dxana[:,4]
copy_dt_shift[:,1] = dxana[:,8]
copy_dt_shift[:,2] = dxana[:,12]

output.PointData.append(copy_dt_shift,"pv_dtTargetShiftAnalytical")
output.PointData.append(copy_dt_shift-fddTargetShift,"pv_dtDiff")



#####################################################################################
# Space derivs and their diff for gtargetshift
#####################################################################################

gdxnum = data.PointData["dxgTargetShiftNumericalFlattened"]
gdxana = data.PointData["SpacetimeDerivOfgTargetShiftFlattened"]


print(algs.shape(gdxnum))
print(algs.shape(gdxana))

gdx_diff = gdxnum*0.0

gdx_diff[:,1] = (gdxana[:,1] - gdxnum[:,1])
gdx_diff[:,2] = (gdxana[:,2] - gdxnum[:,2])
gdx_diff[:,3] = (gdxana[:,3] - gdxnum[:,3])

gdx_diff[:,5] = (gdxana[:,5] - gdxnum[:,5])
gdx_diff[:,6] = (gdxana[:,6] - gdxnum[:,6])
gdx_diff[:,7] = (gdxana[:,7] - gdxnum[:,7])

gdx_diff[:,9] = (gdxana[:,9] - gdxnum[:,9])
gdx_diff[:,10] = (gdxana[:,10] - gdxnum[:,10])
gdx_diff[:,11] = (gdxana[:,11] - gdxnum[:,11])

gdx_diff[:,13] = (gdxana[:,13] - gdxnum[:,13])
gdx_diff[:,14] = (gdxana[:,14] - gdxnum[:,14])
gdx_diff[:,15] = (gdxana[:,15] - gdxnum[:,15])
output.PointData.append(gdx_diff,"pv_gdx_diff")


gdx_analytical= gdxnum*0.0
gdx_analytical[:,1] = gdxana[:,1]
gdx_analytical[:,2] = gdxana[:,2]
gdx_analytical[:,3] = gdxana[:,3]

gdx_analytical[:,5] = gdxana[:,5]
gdx_analytical[:,6] = gdxana[:,6]
gdx_analytical[:,7] = gdxana[:,7]

gdx_analytical[:,9] = gdxana[:,9]
gdx_analytical[:,10] = gdxana[:,10]
gdx_analytical[:,11] = gdxana[:,11]

gdx_analytical[:,13] = gdxana[:,13]
gdx_analytical[:,14] = gdxana[:,14]
gdx_analytical[:,15] = gdxana[:,15]
output.PointData.append(gdx_analytical,"dx_gTargetShiftanalytical")



#####################################################################################
# Time derivatives and their diff for gtargetshift
#####################################################################################
gtshift0 =  data.PointData["gTargetShift"]
gtshift1 =  data1.PointData["gTargetShift"]
print(algs.shape(gtshift0))
print(algs.shape(gtshift1))


time_step = 1.299999999999999999e-04
gfddTargetShift = (gtshift1 - gtshift0)/time_step
output.PointData.append(gfddTargetShift,"pv_dtgTargetShiftNumerical")

gcopy_dt_shift = gtshift0*0.0
gcopy_dt_shift[:,0] = gdxana[:,0]
gcopy_dt_shift[:,1] = gdxana[:,4]
gcopy_dt_shift[:,2] = gdxana[:,8]
gcopy_dt_shift[:,3] = gdxana[:,12]

output.PointData.append(gcopy_dt_shift,"pv_dtgTargetShiftAnalytical")
output.PointData.append(gcopy_dt_shift-gfddTargetShift,"pv_dtDiffg")
