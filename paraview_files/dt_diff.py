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
# Space derivs and their diff
#####################################################################################


dxnum = data.PointData["dxTargetShiftNumericalFlattened"]
dxana = data.PointData["dTargetShiftAnalyticFlattened"]

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
output.PointData.append(dx_analytical,"dx_analytical")



#####################################################################################
# Time derivatives and their diff
#####################################################################################
tshift0 =  data.PointData["TargetShift"]
tshift1 =  data1.PointData["TargetShift"]
print(algs.shape(tshift0))
print(algs.shape(tshift1))



time_step = 1.2999999999999999e-04
output.PointData.append((tshift1 - tshift0)/time_step,"pv_dtTargetShiftNumerical")

copy_dt_shift = tshift0*0.0
copy_dt_shift[:,0] = dxana[:,4]
copy_dt_shift[:,1] = dxana[:,8]
copy_dt_shift[:,2] = dxana[:,12]

output.PointData.append(copy_dt_shift,"pv_dtTargetShiftAnalytical")
 
