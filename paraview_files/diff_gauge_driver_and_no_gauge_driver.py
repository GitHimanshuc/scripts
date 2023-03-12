from paraview.vtk.numpy_interface import dataset_adapter as dsa

from paraview.vtk.numpy_interface import algorithms as algs
import numpy as np

#####################################################################################
# Load data
#####################################################################################

t1 = inputs[0]
t2 = inputs[1]


# Write data with another name

for var_name in t1.PointData.keys():
  if var_name == "f_SpacetimeDerivOfH":
    output.PointData.append(t1.PointData[var_name]-t2.PointData["f_SpacetimeDerivOfGaugeF"],"diff_dH_dGaugeF")
  else:
    output.PointData.append(t1.PointData[var_name]-t2.PointData[var_name],"diff_"+var_name)


output.PointData.append(t2.PointData["H"]-t2.PointData["GaugeF"],"diff_Hgd_GaugeF")
output.PointData.append(t1.PointData["H"]-t2.PointData["GaugeF"],"diff_Hngd_GaugeF")
output.PointData.append(t2.PointData["Theta"],"Theta")

#for var_name in t2.PointData.keys():

#  output.PointData.append(t2.PointData[var_name],"ngd_"+var_name)

print("\nt1: ", t1.PointData.keys())
print("\nt2:",t2.PointData.keys())

ngd_dtH = t1.PointData["f_SpacetimeDerivOfH"]
gd_dtH = t2.PointData["dtH"]

print(algs.shape(gd_dtH))
print(algs.shape(ngd_dtH))

output.PointData.append(gd_dtH[:,0] - ngd_dtH[:,0], "pv_diff_dtH_0")
output.PointData.append(gd_dtH[:,1] - ngd_dtH[:,4], "pv_diff_dtH_1")
output.PointData.append(gd_dtH[:,2] - ngd_dtH[:,8], "pv_diff_dtH_2")
output.PointData.append(gd_dtH[:,3] - ngd_dtH[:,12], "pv_diff_dtH_3")

