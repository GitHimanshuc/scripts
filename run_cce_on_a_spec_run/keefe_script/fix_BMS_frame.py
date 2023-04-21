#!/usr/local/python/anaconda3-2019.10/bin/python
"""
This will fix the BMS frame of CCE waveforms
to be either the PN BMS frame or the superrest frame.
"""

import os
import sys
import scri
import h5py
import json
import scipy
import argparse
import numpy as np

from utils import *

cwd = os.getcwd() + '/'
input_command = 'python ' + ' '.join(sys.argv)

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawTextHelpFormatter)

p.add_argument(
    '--dir',
    '-d',
    dest='directory',
    type=str,
    default='.',
    help='Directory containing the CCE directory and its metadata file.')
p.add_argument(
    '--cce_prefix',
    '-p',
    dest='cce_prefix',
    type=str,
    default='bondi_cce/',
    help='CCE directory.')
p.add_argument(
    '--radius',
    '-r',
    dest='radius',
    type=str,
    default=None,
    help='Worldtube radius of CCE files to use.')
p.add_argument(
    '--target_frame',
    '-f',
    dest='target_frame',
    type=str,
    default='superrest',
    help='Target BMS frame (superrest or PN BMS).')
p.add_argument(
    '--time',
    '-t',
    dest='time',
    type=float,
    default=0.0,
    help='Time at which to map to the superrest frame, or the start of the matching window for the PN BMS frame.')
p.add_argument(
    '--n_orbits',
    '-c',
    dest='n_orbits',
    type=float,
    default=3,
    help='The size of the matching window for the PN BMS frame in orbits.')
p.add_argument(
    '--PN_strain',
    '-i',
    dest='PN_strain_name',
    type=str,
    default='rhOverM_Inertial_PN_Dict_T4.h5',
    help='PN strain filename.')
p.add_argument(
    '--PN_supermomentum',
    '-j',
    dest='PN_supermomentum_name',
    type=str,
    default='supermomentum_PN_Dict_T4.h5',
    help='PN supermomentum filename.')
p.add_argument(
    '--EXT_strain',
    '-k',
    dest='EXT_strain_name',
    type=str,
    default='rhOverM_Asymptotic_GeometricUnits_CoM.h5/Extrapolated_N2.dir',
    help='EXT strain filename.')
p.add_argument(
    '--save_suffix',
    '-s',
    dest='save_suffix',
    type=str,
    default='',
    help='A suffix to append to the saved files.')
p.add_argument(
    '--json_name',
    '-o',
    dest='json_name',
    type=str,
    default='PN_BMS_superrest_errors.json',
    help='JSON file to write to.')

args = p.parse_args()

def radius_via_news_decay(hs, radii, t1=200, t2=None):
    violations = []
    for h in hs:
        news = h.copy()
        news.data = h.data_dot
        news.dataType = scri.hdot
        news.t -= news.t[np.argmax(news.norm())]

        if t1 is None:
            idx1 = 0
        else:
            idx1 = np.argmin(abs(news.t - t1))
        if t2 is None:
            idx2 = -1
        else:
            idx2 = np.argmin(abs(news.t - t2))

        if news.t[idx1] == news.t[idx2]:
            violations.append(np.inf)
        else:
            violations.append(scipy.integrate.trapezoid(news.norm()[idx1:idx2], news.t[idx1:idx2])/
                              (news.t[idx2] - news.t[idx1]))

    return radii[np.argmin(violations)]

def read_in_abd_object(directory, cce_prefix, radius, time, n_orbits, is_superrest=True):
    abd = scri.SpEC.file_io.create_abd_from_h5(h=f'{directory}{cce_prefix}rhOverM_BondiCce_R{radius}.h5',\
                                               Psi4=f'{directory}{cce_prefix}rMPsi4_BondiCce_R{radius}.h5',\
                                               Psi3=f'{directory}{cce_prefix}r2Psi3_BondiCce_R{radius}.h5',\
                                               Psi2=f'{directory}{cce_prefix}r3Psi2OverM_BondiCce_R{radius}.h5',\
                                               Psi1=f'{directory}{cce_prefix}r4Psi1OverM2_BondiCce_R{radius}.h5',\
                                               Psi0=f'{directory}{cce_prefix}r5Psi0OverM3_BondiCce_R{radius}.h5',\
                                               file_format='SXS')

    peak_time = abd.t[np.argmax(MT_to_WM(2.0*abd.sigma.bar.dot, dataType=scri.hdot).norm())]
    abd.t = abd.t - peak_time

    if is_superrest:
        if time > 0:
            abd = abd.interpolate(np.arange(-100, abd.t[-1], 0.1))
        t_0 = time

        return abd, t_0
    else:
        t1 = time - peak_time
        abd = abd.interpolate(np.arange(abd.t[np.argmin(abs(abd.t - (t1 - 500)))], abd.t[-1], 0.1))
        t2 = compute_number_of_orbits_past_time(MT_to_WM(2.0*abd.sigma.bar, dataType=scri.h), n_orbits, t1)

        return abd, [t1, t2]
    
def read_in_PN_objects(abd, t1, t2, directory, PN_strain_name, PN_supermomentum_name):
    try:
        # strain
        h_PN = scri.SpEC.file_io.read_from_h5(f'{directory}{PN_strain_name}')
        
        # supermomentum
        PsiM_PN = scri.SpEC.file_io.read_from_h5(f'{directory}{PN_supermomentum_name}')
    except:
        # strain
        with h5py.File(f'{directory}{PN_strain_name}', 'r') as f:
            t_PN = np.array(f['t'])
            h_PN_DataGrp = f['h']
            h_PN_Dict = {}
            for modeKey in h_PN_DataGrp.keys():
                h_PN_Dict[modeKey] = np.array(h_PN_DataGrp[modeKey])
        ell_min = min([int(key.split('L')[1].split('_')[0]) for key in h_PN_Dict])
        ell_max = max([int(key.split('L')[1].split('_')[0]) for key in h_PN_Dict])
        
        h_PN = scri.WaveformModes(t=t_PN,\
                                  data=np.zeros((len(t_PN), (ell_max + 1)**2 - (ell_min)**2), dtype=complex),\
                                  ell_min=ell_min,\
                                  ell_max=ell_max,\
                                  frameType=scri.Inertial,\
                                  dataType=scri.h,\
                                  r_is_scaled_out=True,\
                                  m_is_scaled_out=True)
        for key in h_PN_Dict:
            L = int(key.split('L')[1].split('_')[0])
            M = int(key.split('L')[1].split('M')[1])
            h_PN.data[:,LM_index(L, M, h_PN.ell_min)] = h_PN_Dict[key]
        
        h_PN = h_PN[:np.argmin(abs(h_PN.t - (h_PN.t[-1] - 200))) + 1]
        
        # supermomentum
        with h5py.File(f'{directory}{PN_supermomentum_name}', 'r') as f:
            t_PN = np.array(f['t'])
            PsiM_PN_DataGrp = f['h']
            PsiM_PN_Dict = {}
            for modeKey in PsiM_PN_DataGrp.keys():
                PsiM_PN_Dict[modeKey] = np.array(PsiM_PN_DataGrp[modeKey])
        ell_min = min([int(key.split('L')[1].split('_')[0]) for key in PsiM_PN_Dict])
        ell_max = max([int(key.split('L')[1].split('_')[0]) for key in PsiM_PN_Dict])
        
        PsiM_PN = scri.WaveformModes(t=t_PN,\
                                     data=np.zeros((len(t_PN), (ell_max + 1)**2 - (ell_min)**2), dtype=complex),\
                                     ell_min=ell_min,\
                                     ell_max=ell_max,\
                                     frameType=scri.Inertial,\
                                     dataType=scri.psi2,\
                                     r_is_scaled_out=True,\
                                     m_is_scaled_out=True)
        for key in h_PN_Dict:
            L = int(key.split('L')[1].split('_')[0])
            M = int(key.split('L')[1].split('M')[1])
            PsiM_PN.data[:,LM_index(L, M, PsiM_PN.ell_min)] = PsiM_PN_Dict[key]

        PsiM_PN = PsiM_PN[:np.argmin(abs(PsiM_PN.t - (PsiM_PN.t[-1] - 200))) + 1]
    
    # rough align waveforms using the angular velocity computed from the news waveforms
    NR_news = MT_to_WM(2.0*abd.sigma.bar.dot, dataType=scri.hdot)
    NR_omega = np.linalg.norm(NR_news.angular_velocity(), axis=1)
    
    PN_news = MT_to_WM(WM_to_MT(h_PN).dot, dataType=scri.hdot)
    PN_omega = np.linalg.norm(PN_news.angular_velocity(), axis=1)

    t_delta = h_PN.t[np.argmin(abs(PN_omega - (NR_omega[np.argmin(abs(abd.t - (t1 + t2)/2))])))] - (t1 + t2)/2
    h_PN.t = h_PN.t - t_delta
    PsiM_PN.t = PsiM_PN.t - t_delta

    return h_PN, PsiM_PN

def read_in_EXT_objects(directory, EXT_strain_name):
    h_EXT = scri.SpEC.file_io.read_from_h5(f'{directory}{EXT_strain_name}')
    
    peak_time = h_EXT.t[np.argmax(MT_to_WM(WM_to_MT(h_EXT).dot, dataType=scri.hdot).norm())]
    h_EXT.t = h_EXT.t - peak_time
 
    return h_EXT

if __name__ == '__main__':
    is_superrest = args.target_frame == 'superrest'
    
    if args.radius is None:
        radii = [x.split('_R')[1][:4] for x in os.listdir(f'{args.directory}{args.cce_prefix}') if 'rhOverM' in x]
        hs = []
        for radius in radii:
            h = scri.SpEC.file_io.read_from_h5(f'{args.directory}{args.cce_prefix}rhOverM_BondiCce_R{radius}.h5')
            hs.append(h)

        radius = radius_via_news_decay(hs, radii)
        del hs

    # ADDED FOR LONG RUNS
    #abd = scri.SpEC.file_io.create_abd_from_h5(h=f'{args.directory}{args.cce_prefix}rhOverM_BondiCce_R{radius}.h5',\
    #                                           Psi4=f'{args.directory}{args.cce_prefix}rMPsi4_BondiCce_R{radius}.h5',\
    #                                           Psi3=f'{args.directory}{args.cce_prefix}r2Psi3_BondiCce_R{radius}.h5',\
    #                                           Psi2=f'{args.directory}{args.cce_prefix}r3Psi2OverM_BondiCce_R{radius}.h5',\
    #                                           Psi1=f'{args.directory}{args.cce_prefix}r4Psi1OverM2_BondiCce_R{radius}.h5',\
    ##                                           Psi0=f'{args.directory}{args.cce_prefix}r5Psi0OverM3_BondiCce_R{radius}.h5',\
    #                                           file_format='SXS')
    #peak_time = abd.t[np.argmax(MT_to_WM(2.0*abd.sigma.bar.dot, dataType=scri.hdot).norm())] 
    # 
    #print(args.directory)
    #run_name = args.directory.split('q8_3dAl_long/')[1].split('/')[0]
    #with open(f'/panfs/ds09/sxs/kmitman/AnnexToLoopOver/CCEAnnex/Public/q8_3dAl/{run_name}/Lev3/PN_BMS_errors_EOB_Lev3.json') as input_file:
    #    errors = json.load(input_file)
    #time = errors['times']['t1'] + peak_time
    #

    abd, times = read_in_abd_object(args.directory, args.cce_prefix, radius, args.time, args.n_orbits, is_superrest)

    if is_superrest:
        t_0 = min(times, abd.t[-1])
        abd, transformations = abd.map_to_superrest_frame(t_0=t_0, padding_time=50)

        peak_time = abd.t[np.argmax(MT_to_WM(2.0*abd.sigma.bar, dataType=scri.h).norm())]
        abd.t = abd.t - peak_time

        save_dir = f'{args.directory}{args.cce_prefix[:-1]}_superrest_iplus/'
        try:
            os.mkdir(save_dir)
        except:
            pass

        scri.SpEC.file_io.rotating_paired_xor_multishuffle_bzip2.save(MT_to_WM(2.0*abd.sigma.bar, dataType=scri.h),\
                                                                      f'{save_dir}rhOverM_BondiCce_R{radius}_superrest.h5')
        scri.SpEC.file_io.rotating_paired_xor_multishuffle_bzip2.save(MT_to_WM(2.0*abd.psi4, dataType=scri.psi4),\
                                                                      f'{save_dir}rMPsi4_BondiCce_R{radius}_superrest.h5')
        scri.SpEC.file_io.rotating_paired_xor_multishuffle_bzip2.save(MT_to_WM(-np.sqrt(2)*abd.psi3, dataType=scri.psi3),\
                                                                      f'{save_dir}r2Psi3_BondiCce_R{radius}_superrest.h5')
        scri.SpEC.file_io.rotating_paired_xor_multishuffle_bzip2.save(MT_to_WM(abd.psi2, dataType=scri.psi2),\
                                                                      f'{save_dir}r3Psi2OverM_BondiCce_R{radius}_superrest.h5')
        scri.SpEC.file_io.rotating_paired_xor_multishuffle_bzip2.save(MT_to_WM(-(1/np.sqrt(2))*abd.psi1, dataType=scri.psi1),\
                                                                      f'{save_dir}r4Psi1OverM2_BondiCce_R{radius}_superrest.h5')
        scri.SpEC.file_io.rotating_paired_xor_multishuffle_bzip2.save(MT_to_WM(0.5*abd.psi0, dataType=scri.psi0),\
                                                                      f'{save_dir}r5Psi0OverM3_BondiCce_R{radius}_superrest.h5')
    else:
        t1, t2 = times
        h_PN, PsiM_PN = read_in_PN_objects(abd, t1, t2, args.directory, args.PN_strain_name, args.PN_supermomentum_name)
        h_EXT = read_in_EXT_objects(args.directory, args.EXT_strain_name)
    
        # save initial waveforms
        t_cut = h_PN.t[-1] - 10000
        h_PN_cut = h_PN.copy()[np.argmin(abs(h_PN.t - t_cut)):]
        PsiM_PN_cut = PsiM_PN.copy()[np.argmin(abs(h_PN.t - t_cut)):]
        
        #save_dir = f'{args.directory}{args.cce_prefix[:-1]}_PN_BMS/'
        #try:
        #    os.mkdir(save_dir)
        #except:
        #    pass

        #scri.SpEC.file_io.write_to_h5(h_PN_cut,
        #                              f'{save_dir}Inertial_PN_Dict_T4_for_hybrid{args.save_suffix}.h5')

        CCE_sur_modes = [(L,M) for L in range(2, 4 + 1) for M in range(0, L + 1)] + [(5, 5)]
        EXT_sur_modes = [(2,2), (2,1), (3,3), (3,2), (3,1), (4,4), (4,3), (4,2), (5,5)]

        # CCE sur modes
        error_BMS_cce_sur_modes, h_CCE_prime, _, _, abd_prime = PN_BMS_w_time_phase(abd, h_PN_cut, PsiM_PN_cut, t1, t2, include_modes=CCE_sur_modes, N=4)
        
        #scri.SpEC.file_io.write_to_h5(h_CCE_prime,
        #                              f'{save_dir}BondiCce_R0247_PN_BMS{args.save_suffix}.h5')
        
        #scri.SpEC.file_io.rotating_paired_xor_multishuffle_bzip2.save(MT_to_WM(2.0*abd_prime.sigma.bar, dataType=scri.h),\
        #                                                              f'{save_dir}rhOverM_BondiCce_R{radius}_PN_BMS{args.save_suffix}.h5')
        #scri.SpEC.file_io.rotating_paired_xor_multishuffle_bzip2.save(MT_to_WM(2.0*abd_prime.psi4, dataType=scri.psi4),\
        #                                                              f'{save_dir}rMPsi4_BondiCce_R{radius}_PN_BMS{args.save_suffix}.h5')
        #scri.SpEC.file_io.rotating_paired_xor_multishuffle_bzip2.save(MT_to_WM(-np.sqrt(2)*abd_prime.psi3, dataType=scri.psi3),\
        #                                                              f'{save_dir}r2Psi3_BondiCce_R{radius}_PN_BMS{args.save_suffix}.h5')
        #scri.SpEC.file_io.rotating_paired_xor_multishuffle_bzip2.save(MT_to_WM(abd_prime.psi2, dataType=scri.psi2),\
        #                                                              f'{save_dir}r3Psi2OverM_BondiCce_R{radius}_PN_BMS{args.save_suffix}.h5')
        #scri.SpEC.file_io.rotating_paired_xor_multishuffle_bzip2.save(MT_to_WM(-(1/np.sqrt(2))*abd_prime.psi1, dataType=scri.psi1),\
        #                                                              f'{save_dir}r4Psi1OverM2_BondiCce_R{radius}_PN_BMS{args.save_suffix}.h5')
        #scri.SpEC.file_io.rotating_paired_xor_multishuffle_bzip2.save(MT_to_WM(0.5*abd_prime.psi0, dataType=scri.psi0),\
        #                                                              f'{save_dir}r5Psi0OverM3_BondiCce_R{radius}_PN_BMS{args.save_suffix}.h5')
        
        #h_hybrid = hybridize(h_CCE_prime, h_PN, t1, t2)
        #scri.SpEC.file_io.write_to_h5(h_hybrid,
        #                              f'{save_dir}BondiCce_R{radius}_PN_BMS_hybrid{args.save_suffix}.h5')

        # EXT sur modes
        error_BMS_ext_sur_modes, h_CCE_prime, _, _, _ = PN_BMS_w_time_phase(abd, h_PN_cut, PsiM_PN_cut, t1, t2, include_modes=EXT_sur_modes, N=4) 
        
        error_time_phase_ext_sur_modes_ext = align2d(h_EXT, h_PN_cut, t1, t2, n_brute_force_δt=None, n_brute_force_δϕ=None, include_modes=EXT_sur_modes, nprocs=None)[0]

        errors = {
            'CCE sur modes': {
                'CCE': {
                    'PN_BMS_time_phase': error_BMS_cce_sur_modes
                }
            },
            'EXT sur modes': {
                'CCE': {
                    'PN_BMS_time_phase': error_BMS_ext_sur_modes
                },
                'EXT': {
                    'time_phase':        error_time_phase_ext_sur_modes_ext
                }
            },
            'times': {
                't1':                    t1,
                't2':                    t2,
            }
        }

        # extra thing
        #try:
        #    os.system(f'cp {args.directory}PN_parameters_Lev3.json {save_dir}PN_parameters_Lev3.json')
        #except:
        #    pass
        #try:
        #    os.system(f'cp {args.directory}PN_parameters_Lev4.json {save_dir}PN_parameters_Lev4.json')
        #except:
        #    pass

        with open(f'{args.directory}{args.json_name}', 'w') as outfile:
            json.dump(errors, outfile, indent=2, separators=(",", ": "), ensure_ascii=True)
