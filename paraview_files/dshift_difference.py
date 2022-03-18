from paraview.vtk.numpy_interface import dataset_adapter as dsa
from paraview.vtk.numpy_interface import algorithms as algs

data = inputs[0]
print(data.PointData.keys())

dxnum = data.PointData["dxTargetShiftNumericalFlattened"]
dxana = data.PointData["dTargetShiftAnalyticFlattened"]

print(algs.shape(dxnum))
print(algs.shape(dxana))


output.PointData.append(dxana[:,5] - dxnum[:,0,0],"xx")
output.PointData.append(dxana[:,6] - dxnum[:,0,1],"yx")
output.PointData.append(dxana[:,7] - dxnum[:,0,2],"zx")

output.PointData.append(dxana[:,9] - dxnum[:,1,0],"xy")
output.PointData.append(dxana[:,10] - dxnum[:,1,1],"yy")
output.PointData.append(dxana[:,11] - dxnum[:,1,2],"zy")

output.PointData.append(dxana[:,13] - dxnum[:,2,0],"xz")
output.PointData.append(dxana[:,14] - dxnum[:,2,1],"yz")
output.PointData.append(dxana[:,15] - dxnum[:,2,2],"zz")



dxnum = data.PointData["dxTargetShiftFromGridNumericalFlattened"]
dxana = data.PointData["dTargetShiftFromGridAnalyticFlattened"]

output.PointData.append(dxana[:,5] - dxnum[:,0,0],"xx_grid")
output.PointData.append(dxana[:,6] - dxnum[:,0,1],"yx_grid")
output.PointData.append(dxana[:,7] - dxnum[:,0,2],"zx_grid")

output.PointData.append(dxana[:,9] - dxnum[:,1,0],"xy_grid")
output.PointData.append(dxana[:,10] - dxnum[:,1,1],"yy_grid")
output.PointData.append(dxana[:,11] - dxnum[:,1,2],"zy_grid")

output.PointData.append(dxana[:,13] - dxnum[:,2,0],"xz_grid")
output.PointData.append(dxana[:,14] - dxnum[:,2,1],"yz_grid")
output.PointData.append(dxana[:,15] - dxnum[:,2,2],"zz_grid")