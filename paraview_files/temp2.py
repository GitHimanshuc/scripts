folder_name="gauge_driver_mr3"
# create a new 'PVD Reader'
combinedpvd = PVDReader(registrationName=folder_name, FileName=f'/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/data_from_checkpoints/{folder_name}/GaugeVis_except_C.pvd')
# create a new 'Slice'
slice2 = Slice(registrationName=f'Slice_{folder_name}', Input=combinedpvd)

# Properties modified on slice2.SliceType
slice2.SliceType.Normal = [0.0, 0.0, 1.0]

# create a new 'Calculator'
calculator_nada_alpha = Calculator(registrationName=f'nada_alpha_{folder_name}', Input=slice2)
calculator_nada_alpha.ResultArrayName = 'nada_alpha'
calculator_nada_alpha.Function = '"partial_t_alpha_1"-"beta^i_partial_i_alpha_1"'

# create a new 'Calculator'
calculator_nada_logfac = Calculator(registrationName=f'nada_logfac_{folder_name}', Input=calculator_nada_alpha)
calculator_nada_logfac.ResultArrayName = 'nada_logfac'
calculator_nada_logfac.Function = '"partial_t_logfac_1"-"beta^i_partial_i_logfac_1"'

# create a new 'Calculator'
calculator_nada_shift1 = Calculator(registrationName=f'nada_shift1_{folder_name}', Input=calculator_nada_logfac)
calculator_nada_shift1.ResultArrayName = 'nada_shift1'
calculator_nada_shift1.Function = '"partial_t_beta^i_1"-"beta^j_partial_j_beta^i_1"'

# create a new 'Calculator'
calculator_nada_shift2 = Calculator(registrationName=f'nada_shift2_{folder_name}', Input=calculator_nada_shift1)
calculator_nada_shift2.ResultArrayName = 'nada_shift2'
calculator_nada_shift2.Function = '"partial_t_beta^i_2"-"beta^j_partial_j_beta^i_2"'

# create a new 'Calculator'
calculator_nada_shift3 = Calculator(registrationName=f'nada_shift3_{folder_name}', Input=calculator_nada_shift2)
calculator_nada_shift3.ResultArrayName = 'nada_shift3'
calculator_nada_shift3.Function = '"partial_t_beta^i_3"-"beta^j_partial_j_beta^i_3"'

# create a new 'Plot Over Line'
plotOverLine2 = PlotOverLine(registrationName=f'PlotOverLine_{folder_name}', Input=calculator_nada_shift3)

# Properties modified on plotOverLine2
plotOverLine2.Point1 = [-100.0, 15.0, 0.0]
plotOverLine2.Point2 = [100.0, 15.0, 0.0]