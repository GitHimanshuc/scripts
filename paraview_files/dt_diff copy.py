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
# Time derivatives and their diff
#####################################################################################
tshift0 =  data.PointData["HNoz"]
tshift1 =  data1.PointData["HNoz"]
print(algs.shape(tshift0))
print(algs.shape(tshift1))


time_step = 1.299999999999999999e-04
fddTargetShift = (tshift1 - tshift0)/time_step
output.PointData.append(fddTargetShift,"pv_dtHNoz")