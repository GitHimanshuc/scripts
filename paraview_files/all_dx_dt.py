from paraview.vtk.numpy_interface import dataset_adapter as dsa
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
# Space derivs and their diff for H
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
