# tanh = FindSource('GaugeVis2.pvd')
# old = FindSource('old_gauge.pvd')


# tanh = inputs[0]
# old = inputs[1]

# tanh_evals = tanh.PointData["EvalsK_obs"]
# old_evals = old.PointData["EvalsK_obs"]
# diff_evals = tanh_evals-old_evals


# output.PointData.append(diff_evals,"diff_evals")



################################################################################
################################################################################
################################################################################

tanh = inputs[0]

tanh_evals = tanh.PointData["g"]
shift = tanh.PointData["Shift"]
temp = shift*0.0


# temp[:,0] = tanh_evals[:,2,0]
# temp[:,1] = tanh_evals[:,2,1]
# temp[:,2] = tanh_evals[:,2,2]

output.PointData.append(tanh_evals[:,2,:],"largets_evecK")



