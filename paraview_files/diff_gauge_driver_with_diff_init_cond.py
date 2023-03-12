from paraview.vtk.numpy_interface import dataset_adapter as dsa

from paraview.vtk.numpy_interface import algorithms as algs
import numpy as np

#####################################################################################
# Load data
#####################################################################################

t1 = inputs[0] # dtH = 0
t2 = inputs[1] # dtH != 0



print("\nt1: ", t1.PointData.keys())
print("\nt2:",t2.PointData.keys())

# Write data with another name
for var_name in t2.PointData.keys():
    output.PointData.append(t1.PointData[var_name]-t2.PointData[var_name],"diff_"+var_name)


output.PointData.append(t1.PointData["H"]-t1.PointData["GaugeF"],"diff_Hgd_GaugeF_dtH0")
output.PointData.append(t2.PointData["H"]-t2.PointData["GaugeF"],"diff_Hgd_GaugeF")


