# Size of variable arrays:
sizeAlgebraic = 78
sizeStates = 27
sizeConstants = 61
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component Environment (millisecond)"
    legend_constants[0] = "R in component Environment (millijoule_per_mole_kelvin)"
    legend_constants[1] = "T in component Environment (kelvin)"
    legend_constants[2] = "F in component Environment (coulomb_per_mole)"
    legend_constants[3] = "K_o in component Environment (millimolar)"
    legend_constants[4] = "Ca_o in component Environment (millimolar)"
    legend_constants[5] = "Na_o in component Environment (millimolar)"
    legend_states[0] = "V in component cell (millivolt)"
    legend_constants[6] = "Cm in component cell (nanoF)"
    legend_constants[7] = "Vol_c in component cell (nanolitre)"
    legend_algebraic[54] = "i_K1 in component IK1 (nanoA_per_nanoF)"
    legend_algebraic[57] = "i_to in component Ito (nanoA_per_nanoF)"
    legend_algebraic[55] = "i_Kr in component IKr (nanoA_per_nanoF)"
    legend_algebraic[56] = "i_Ks in component IKs (nanoA_per_nanoF)"
    legend_algebraic[60] = "i_CaL in component ICaL (nanoA_per_nanoF)"
    legend_algebraic[64] = "i_NaK in component INaK (nanoA_per_nanoF)"
    legend_algebraic[58] = "i_Na in component INa (nanoA_per_nanoF)"
    legend_algebraic[59] = "i_b_Na in component INab (nanoA_per_nanoF)"
    legend_algebraic[65] = "i_NaCa in component INaCa (nanoA_per_nanoF)"
    legend_algebraic[61] = "i_b_Ca in component ICab (nanoA_per_nanoF)"
    legend_algebraic[63] = "i_p_K in component IpK (nanoA_per_nanoF)"
    legend_algebraic[62] = "i_p_Ca in component IpCa (nanoA_per_nanoF)"
    legend_algebraic[0] = "i_Stim in component cell (nanoA_per_nanoF)"
    legend_algebraic[66] = "i_tot in component cell (nanoA_per_nanoF)"
    legend_constants[8] = "stim_Period in component cell (millisecond)"
    legend_algebraic[12] = "E_Na in component reversal_potentials (millivolt)"
    legend_algebraic[24] = "E_K in component reversal_potentials (millivolt)"
    legend_algebraic[31] = "E_Ks in component reversal_potentials (millivolt)"
    legend_algebraic[38] = "E_Ca in component reversal_potentials (millivolt)"
    legend_constants[9] = "P_kna in component reversal_potentials (dimensionless)"
    legend_states[1] = "K_i in component K (millimolar)"
    legend_states[2] = "Na_i in component Na (millimolar)"
    legend_states[3] = "Ca_i in component Ca (millimolar)"
    legend_constants[10] = "g_K1_0 in component IK1 (microS_per_nanoF)"
    legend_algebraic[53] = "xK1_inf in component iK1_rectification (dimensionless)"
    legend_constants[11] = "Mg_Buf in component iK1_rectification (millimolar)"
    legend_constants[12] = "SPM in component iK1_rectification (millimolar)"
    legend_constants[13] = "fac in component iK1_rectification (dimensionless)"
    legend_constants[14] = "phi in component iK1_rectification (dimensionless)"
    legend_algebraic[49] = "temp in component iK1_rectification (dimensionless)"
    legend_algebraic[51] = "rec1 in component iK1_rectification (dimensionless)"
    legend_algebraic[52] = "rec2 in component iK1_rectification (dimensionless)"
    legend_algebraic[41] = "KiMg in component iK1_rectification (millimolar)"
    legend_algebraic[43] = "KbMg in component iK1_rectification (millimolar)"
    legend_algebraic[45] = "Kd1SPM in component iK1_rectification (millimolar)"
    legend_algebraic[47] = "Kd2SPM in component iK1_rectification (millimolar)"
    legend_constants[15] = "g_Kr_0 in component IKr (microS_per_nanoF)"
    legend_states[4] = "Or4 in component iKr_Markov (dimensionless)"
    legend_states[5] = "Cr1 in component iKr_Markov (dimensionless)"
    legend_states[6] = "Cr2 in component iKr_Markov (dimensionless)"
    legend_states[7] = "Cr3 in component iKr_Markov (dimensionless)"
    legend_states[8] = "Ir5 in component iKr_Markov (dimensionless)"
    legend_constants[16] = "T_Base in component iKr_Markov (kelvin)"
    legend_algebraic[1] = "alpha_xr1 in component iKr_Markov (per_millisecond)"
    legend_algebraic[13] = "beta_xr1 in component iKr_Markov (per_millisecond)"
    legend_algebraic[25] = "alpha_xr2 in component iKr_Markov (per_millisecond)"
    legend_algebraic[32] = "beta_xr2 in component iKr_Markov (per_millisecond)"
    legend_algebraic[39] = "alpha_xr3 in component iKr_Markov (per_millisecond)"
    legend_algebraic[42] = "beta_xr3 in component iKr_Markov (per_millisecond)"
    legend_algebraic[44] = "alpha_xr4 in component iKr_Markov (per_millisecond)"
    legend_algebraic[46] = "beta_xr4 in component iKr_Markov (per_millisecond)"
    legend_algebraic[48] = "OtoB in component iKr_Markov_Sotalol_block (per_millisecond)"
    legend_algebraic[50] = "BtoO in component iKr_Markov_Sotalol_block (per_millisecond)"
    legend_constants[17] = "Sotalol_mM in component iKr_Markov_Sotalol_block (millimolar)"
    legend_states[9] = "BCr1 in component iKr_Markov_Sotalol_block (dimensionless)"
    legend_states[10] = "BCr2 in component iKr_Markov_Sotalol_block (dimensionless)"
    legend_states[11] = "BCr3 in component iKr_Markov_Sotalol_block (dimensionless)"
    legend_states[12] = "BOr4 in component iKr_Markov_Sotalol_block (dimensionless)"
    legend_states[13] = "BIr5 in component iKr_Markov_Sotalol_block (dimensionless)"
    legend_constants[18] = "kBinding in component iKr_Markov_Sotalol_block (per_millimolar_per_millisecond)"
    legend_constants[19] = "kDiss in component iKr_Markov_Sotalol_block (per_millisecond)"
    legend_constants[20] = "g_Ks in component IKs (microS_per_nanoF)"
    legend_states[14] = "Xs in component iKs_Xs_gate (dimensionless)"
    legend_algebraic[2] = "xs_inf in component iKs_Xs_gate (dimensionless)"
    legend_algebraic[14] = "alpha_xs in component iKs_Xs_gate (dimensionless)"
    legend_algebraic[26] = "beta_xs in component iKs_Xs_gate (dimensionless)"
    legend_algebraic[33] = "tau_xs in component iKs_Xs_gate (millisecond)"
    legend_constants[21] = "g_to in component Ito (microS_per_nanoF)"
    legend_states[15] = "s in component ito_s_gate (dimensionless)"
    legend_states[16] = "r in component ito_r_gate (dimensionless)"
    legend_algebraic[3] = "s_inf in component ito_s_gate (dimensionless)"
    legend_algebraic[15] = "tau_s in component ito_s_gate (millisecond)"
    legend_algebraic[4] = "r_inf in component ito_r_gate (dimensionless)"
    legend_algebraic[16] = "tau_r in component ito_r_gate (millisecond)"
    legend_constants[22] = "g_Na in component INa (microS_per_nanoF)"
    legend_constants[23] = "shift_INa_inact in component INa (millivolt)"
    legend_states[17] = "m in component iNa_m_gate (dimensionless)"
    legend_states[18] = "h in component iNa_h_gate (dimensionless)"
    legend_states[19] = "j in component iNa_j_gate (dimensionless)"
    legend_algebraic[5] = "m_inf in component iNa_m_gate (dimensionless)"
    legend_algebraic[17] = "alpha_m in component iNa_m_gate (dimensionless)"
    legend_algebraic[27] = "beta_m in component iNa_m_gate (dimensionless)"
    legend_algebraic[34] = "tau_m in component iNa_m_gate (millisecond)"
    legend_algebraic[6] = "h_inf in component iNa_h_gate (dimensionless)"
    legend_algebraic[18] = "alpha_h in component iNa_h_gate (per_millisecond)"
    legend_algebraic[28] = "beta_h in component iNa_h_gate (per_millisecond)"
    legend_algebraic[35] = "tau_h in component iNa_h_gate (millisecond)"
    legend_algebraic[7] = "j_inf in component iNa_j_gate (dimensionless)"
    legend_algebraic[19] = "alpha_j in component iNa_j_gate (per_millisecond)"
    legend_algebraic[29] = "beta_j in component iNa_j_gate (per_millisecond)"
    legend_algebraic[36] = "tau_j in component iNa_j_gate (millisecond)"
    legend_constants[24] = "g_bna in component INab (microS_per_nanoF)"
    legend_constants[25] = "g_CaL in component ICaL (litre_per_farad_millisecond)"
    legend_states[20] = "Ca_ss in component Ca (millimolar)"
    legend_states[21] = "d in component iCaL_d_gate (dimensionless)"
    legend_states[22] = "f in component iCaL_f_gate (dimensionless)"
    legend_states[23] = "f2 in component iCaL_f2_gate (dimensionless)"
    legend_states[24] = "fCass in component iCaL_fCass_gate (dimensionless)"
    legend_constants[26] = "z in component ICaL (dimensionless)"
    legend_algebraic[8] = "d_inf in component iCaL_d_gate (dimensionless)"
    legend_algebraic[20] = "alpha_d in component iCaL_d_gate (dimensionless)"
    legend_algebraic[30] = "beta_d in component iCaL_d_gate (dimensionless)"
    legend_algebraic[37] = "gamma_d in component iCaL_d_gate (millisecond)"
    legend_algebraic[40] = "tau_d in component iCaL_d_gate (millisecond)"
    legend_constants[27] = "d_inf_shift in component iCaL_d_gate (millivolt)"
    legend_algebraic[9] = "f_inf in component iCaL_f_gate (dimensionless)"
    legend_algebraic[21] = "tau_f in component iCaL_f_gate (millisecond)"
    legend_algebraic[10] = "f2_inf in component iCaL_f2_gate (dimensionless)"
    legend_algebraic[22] = "tau_f2 in component iCaL_f2_gate (millisecond)"
    legend_algebraic[11] = "fCass_inf in component iCaL_fCass_gate (dimensionless)"
    legend_algebraic[23] = "tau_fCass in component iCaL_fCass_gate (millisecond)"
    legend_constants[28] = "g_bca in component ICab (microS_per_nanoF)"
    legend_constants[29] = "g_pCa in component IpCa (nanoA_per_nanoF)"
    legend_constants[30] = "K_pCa in component IpCa (millimolar)"
    legend_constants[31] = "g_pK in component IpK (microS_per_nanoF)"
    legend_constants[32] = "P_NaK in component INaK (nanoA_per_nanoF)"
    legend_constants[33] = "K_mk in component INaK (millimolar)"
    legend_constants[34] = "K_mNa in component INaK (millimolar)"
    legend_constants[35] = "K_NaCa in component INaCa (nanoA_per_nanoF)"
    legend_constants[36] = "K_sat in component INaCa (dimensionless)"
    legend_constants[37] = "alpha in component INaCa (dimensionless)"
    legend_constants[38] = "gamma in component INaCa (dimensionless)"
    legend_constants[39] = "Km_Ca in component INaCa (millimolar)"
    legend_constants[40] = "Km_Nai in component INaCa (millimolar)"
    legend_states[25] = "Ca_SR in component Ca (millimolar)"
    legend_algebraic[70] = "Ca_i_bufc in component Ca_buffer (dimensionless)"
    legend_algebraic[76] = "Ca_sr_bufsr in component Ca_buffer (dimensionless)"
    legend_algebraic[77] = "Ca_ss_bufss in component Ca_buffer (dimensionless)"
    legend_constants[41] = "V_sr in component Ca (nanolitre)"
    legend_constants[42] = "V_ss in component Ca (nanolitre)"
    legend_algebraic[75] = "i_rel in component Irel (millimolar_per_millisecond)"
    legend_algebraic[67] = "i_up in component Ileak_Iup_Ixfer (millimolar_per_millisecond)"
    legend_algebraic[68] = "i_leak in component Ileak_Iup_Ixfer (millimolar_per_millisecond)"
    legend_algebraic[69] = "i_xfer in component Ileak_Iup_Ixfer (millimolar_per_millisecond)"
    legend_constants[43] = "Vol_xfer in component Ileak_Iup_Ixfer (per_millisecond)"
    legend_constants[44] = "K_up in component Ileak_Iup_Ixfer (millimolar)"
    legend_constants[45] = "Vol_leak in component Ileak_Iup_Ixfer (per_millisecond)"
    legend_constants[46] = "Vmax_up in component Ileak_Iup_Ixfer (millimolar_per_millisecond)"
    legend_states[26] = "R_prime in component Irel (dimensionless)"
    legend_constants[47] = "k1_prime in component Irel (per_millimolar2_per_millisecond)"
    legend_constants[48] = "k2_prime in component Irel (per_millimolar_per_millisecond)"
    legend_constants[49] = "EC in component Irel (millimolar)"
    legend_constants[50] = "max_sr in component Irel (dimensionless)"
    legend_constants[51] = "min_sr in component Irel (dimensionless)"
    legend_constants[52] = "k3 in component Irel (per_millisecond)"
    legend_constants[53] = "k4 in component Irel (per_millisecond)"
    legend_constants[54] = "Vol_rel in component Irel (per_millisecond)"
    legend_algebraic[74] = "O in component Irel (dimensionless)"
    legend_algebraic[71] = "kcasr in component Irel (dimensionless)"
    legend_algebraic[72] = "k1 in component Irel (per_millimolar2_per_millisecond)"
    legend_algebraic[73] = "k2 in component Irel (per_millimolar_per_millisecond)"
    legend_constants[55] = "Buf_c in component Ca_buffer (millimolar)"
    legend_constants[56] = "K_buf_c in component Ca_buffer (millimolar)"
    legend_constants[57] = "Buf_sr in component Ca_buffer (millimolar)"
    legend_constants[58] = "K_buf_sr in component Ca_buffer (millimolar)"
    legend_constants[59] = "Buf_ss in component Ca_buffer (millimolar)"
    legend_constants[60] = "K_buf_ss in component Ca_buffer (millimolar)"
    legend_rates[0] = "d/dt V in component cell (millivolt)"
    legend_rates[5] = "d/dt Cr1 in component iKr_Markov (dimensionless)"
    legend_rates[6] = "d/dt Cr2 in component iKr_Markov (dimensionless)"
    legend_rates[7] = "d/dt Cr3 in component iKr_Markov (dimensionless)"
    legend_rates[4] = "d/dt Or4 in component iKr_Markov (dimensionless)"
    legend_rates[8] = "d/dt Ir5 in component iKr_Markov (dimensionless)"
    legend_rates[9] = "d/dt BCr1 in component iKr_Markov_Sotalol_block (dimensionless)"
    legend_rates[10] = "d/dt BCr2 in component iKr_Markov_Sotalol_block (dimensionless)"
    legend_rates[11] = "d/dt BCr3 in component iKr_Markov_Sotalol_block (dimensionless)"
    legend_rates[12] = "d/dt BOr4 in component iKr_Markov_Sotalol_block (dimensionless)"
    legend_rates[13] = "d/dt BIr5 in component iKr_Markov_Sotalol_block (dimensionless)"
    legend_rates[14] = "d/dt Xs in component iKs_Xs_gate (dimensionless)"
    legend_rates[15] = "d/dt s in component ito_s_gate (dimensionless)"
    legend_rates[16] = "d/dt r in component ito_r_gate (dimensionless)"
    legend_rates[17] = "d/dt m in component iNa_m_gate (dimensionless)"
    legend_rates[18] = "d/dt h in component iNa_h_gate (dimensionless)"
    legend_rates[19] = "d/dt j in component iNa_j_gate (dimensionless)"
    legend_rates[21] = "d/dt d in component iCaL_d_gate (dimensionless)"
    legend_rates[22] = "d/dt f in component iCaL_f_gate (dimensionless)"
    legend_rates[23] = "d/dt f2 in component iCaL_f2_gate (dimensionless)"
    legend_rates[24] = "d/dt fCass in component iCaL_fCass_gate (dimensionless)"
    legend_rates[3] = "d/dt Ca_i in component Ca (millimolar)"
    legend_rates[25] = "d/dt Ca_SR in component Ca (millimolar)"
    legend_rates[20] = "d/dt Ca_ss in component Ca (millimolar)"
    legend_rates[26] = "d/dt R_prime in component Irel (dimensionless)"
    legend_rates[2] = "d/dt Na_i in component Na (millimolar)"
    legend_rates[1] = "d/dt K_i in component K (millimolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 8314.472
    constants[1] = 310
    constants[2] = 96485.3415
    constants[3] = 5.4
    constants[4] = 2
    constants[5] = 140
    states[0] = -86.45
    constants[6] = 0.115
    constants[7] = 0.016404
    constants[8] = 1000
    constants[9] = 0.03
    states[1] = 141.0167
    states[2] = 7.940167
    states[3] = 1.092e-4
    constants[10] = 0.6821
    constants[11] = 0.0356
    constants[12] = 1.4613e-3
    constants[13] = 1.0648
    constants[14] = 0.8838
    constants[15] = 0.024
    states[4] = 0.014
    states[5] = 0.9786
    states[6] = 0.0031
    states[7] = 0.0029
    states[8] = 0.0014
    constants[16] = 310
    constants[17] = 0
    states[9] = 0
    states[10] = 0
    states[11] = 0
    states[12] = 0
    states[13] = 0
    constants[18] = 5e-3
    constants[19] = 0.00125
    constants[20] = 0.0392
    states[14] = 0.00303
    constants[21] = 0.2
    states[15] = 1
    states[16] = 2.11e-8
    constants[22] = 11
    constants[23] = 0
    states[17] = 0.00132
    states[18] = 0.7768
    states[19] = 0.7766
    constants[24] = 0.00029
    constants[25] = 2e-5
    states[20] = 1.893e-4
    states[21] = 5.06e-6
    states[22] = 0.9999
    states[23] = 0.9995
    states[24] = 1
    constants[26] = 2
    constants[27] = 5
    constants[28] = 0.0004736
    constants[29] = 0.0619
    constants[30] = 0.0005
    constants[31] = 0.00973
    constants[32] = 1.297
    constants[33] = 1
    constants[34] = 40
    constants[35] = 200
    constants[36] = 0.1
    constants[37] = 2.5
    constants[38] = 0.35
    constants[39] = 1.38
    constants[40] = 87.5
    states[25] = 2.7656
    constants[41] = 0.001094
    constants[42] = 0.00005468
    constants[43] = 0.0038
    constants[44] = 0.00025
    constants[45] = 0.00036
    constants[46] = 0.006375
    states[26] = 0.9864
    constants[47] = 0.15
    constants[48] = 0.045
    constants[49] = 1.5
    constants[50] = 2.5
    constants[51] = 1
    constants[52] = 0.06
    constants[53] = 0.005
    constants[54] = 0.306
    constants[55] = 0.2
    constants[56] = 0.001
    constants[57] = 10
    constants[58] = 0.3
    constants[59] = 0.4
    constants[60] = 0.00025
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[1] = ((1.00000*constants[1])/constants[16])*exp(24.3350+(constants[16]/constants[1])*(0.0112000*states[0]-25.9140))
    algebraic[13] = ((1.00000*constants[1])/constants[16])*exp(13.6880+(constants[16]/constants[1])*(-0.0603000*states[0]-15.7070))
    rates[5] = algebraic[13]*states[6]-algebraic[1]*states[5]
    rates[9] = algebraic[13]*states[10]-algebraic[1]*states[9]
    algebraic[3] = 1.00000/(1.00000+exp((states[0]+20.0000)/5.00000))
    algebraic[15] = 85.0000*exp(-(power(states[0]+45.0000, 2.00000))/320.000)+5.00000/(1.00000+exp((states[0]-20.0000)/5.00000))+3.00000
    rates[15] = (algebraic[3]-states[15])/algebraic[15]
    algebraic[4] = 1.00000/(1.00000+exp((20.0000-states[0])/6.00000))
    algebraic[16] = 9.50000*exp(-(power(states[0]+40.0000, 2.00000))/1800.00)+0.800000
    rates[16] = (algebraic[4]-states[16])/algebraic[16]
    algebraic[9] = 1.00000/(1.00000+exp((states[0]+20.0000)/7.00000))
    algebraic[21] = (1102.50*exp(-(power(states[0]+27.0000, 2.00000))/225.000)+200.000/(1.00000+exp((13.0000-states[0])/10.0000))+180.000/(1.00000+exp((states[0]+30.0000)/10.0000))+20.0000)/4.00000
    rates[22] = (algebraic[9]-states[22])/algebraic[21]
    algebraic[10] = 0.750000/(1.00000+exp((states[0]+35.0000)/7.00000))+0.250000
    algebraic[22] = (562.000*exp(-(power(states[0]+27.0000, 2.00000))/240.000)+31.0000/(1.00000+exp((25.0000-states[0])/10.0000))+80.0000/(1.00000+exp((states[0]+30.0000)/10.0000)))/2.00000
    rates[23] = (algebraic[10]-states[23])/algebraic[22]
    algebraic[11] = 0.400000/(1.00000+power(states[20]/0.0500000, 2.00000))+0.600000
    algebraic[23] = 80.0000/(1.00000+power(states[20]/0.0500000, 2.00000))+2.00000
    rates[24] = (algebraic[11]-states[24])/algebraic[23]
    algebraic[25] = ((1.00000*constants[1])/constants[16])*exp(22.7460+(constants[16]/constants[1])*(0.00000*states[0]-25.9140))
    algebraic[32] = ((1.00000*constants[1])/constants[16])*exp(13.1930+(constants[16]/constants[1])*(0.00000*states[0]-15.7070))
    rates[6] = (algebraic[1]*states[5]+algebraic[32]*states[7])-(algebraic[25]+algebraic[13])*states[6]
    rates[10] = (algebraic[1]*states[9]+algebraic[32]*states[11])-(algebraic[25]+algebraic[13])*states[10]
    algebraic[2] = 1.00000/(1.00000+exp((-5.00000-states[0])/14.0000))
    algebraic[14] = 1400.00/(power(1.00000+exp((5.00000-states[0])/6.00000), 1.0/2))
    algebraic[26] = 1.00000/(1.00000+exp((states[0]-35.0000)/15.0000))
    algebraic[33] = 1.00000*algebraic[14]*algebraic[26]+80.0000
    rates[14] = (algebraic[2]-states[14])/algebraic[33]
    algebraic[5] = 1.00000/(power(1.00000+exp((-56.8600-states[0])/9.03000), 2.00000))
    algebraic[17] = 1.00000/(1.00000+exp((-60.0000-states[0])/5.00000))
    algebraic[27] = 0.100000/(1.00000+exp((states[0]+35.0000)/5.00000))+0.100000/(1.00000+exp((states[0]-50.0000)/200.000))
    algebraic[34] = 1.00000*algebraic[17]*algebraic[27]
    rates[17] = (algebraic[5]-states[17])/algebraic[34]
    algebraic[6] = 1.00000/(power(1.00000+exp(((states[0]+71.5500)-constants[23])/7.43000), 2.00000))
    algebraic[18] = custom_piecewise([less(states[0] , -40.0000+constants[23]), 0.0570000*exp(-((states[0]+80.0000)-constants[23])/6.80000) , True, 0.00000])
    algebraic[28] = custom_piecewise([less(states[0] , -40.0000+constants[23]), 2.70000*exp(0.0790000*(states[0]-constants[23]))+310000.*exp(0.348500*(states[0]-constants[23])) , True, 0.770000/(0.130000*(1.00000+exp(((states[0]+10.6600)-constants[23])/-11.1000)))])
    algebraic[35] = 1.00000/(algebraic[18]+algebraic[28])
    rates[18] = (algebraic[6]-states[18])/algebraic[35]
    algebraic[7] = 1.00000/(power(1.00000+exp(((states[0]+71.5500)-constants[23])/7.43000), 2.00000))
    algebraic[19] = custom_piecewise([less(states[0] , -40.0000+constants[23]), (((-25428.0*exp(0.244400*(states[0]-constants[23]))-6.94800e-06*exp(-0.0439100*(states[0]-constants[23])))*(states[0]+37.7800))/1.00000)/(1.00000+exp(0.311000*((states[0]+79.2300)-constants[23]))) , True, 0.00000])
    algebraic[29] = custom_piecewise([less(states[0] , -40.0000+constants[23]), (0.0242400*exp(-0.0105200*(states[0]-constants[23])))/(1.00000+exp(-0.137800*((states[0]+40.1400)-constants[23]))) , True, (0.600000*exp(0.0570000*(states[0]-constants[23])))/(1.00000+exp(-0.100000*((states[0]+32.0000)-constants[23])))])
    algebraic[36] = 1.00000/(algebraic[19]+algebraic[29])
    rates[19] = (algebraic[7]-states[19])/algebraic[36]
    algebraic[8] = 1.00000/(1.00000+exp((constants[27]-states[0])/7.50000))
    algebraic[20] = 1.40000/(1.00000+exp((-35.0000-states[0])/13.0000))+0.250000
    algebraic[30] = 1.40000/(1.00000+exp((states[0]+5.00000)/5.00000))
    algebraic[37] = 1.00000/(1.00000+exp((50.0000-states[0])/20.0000))
    algebraic[40] = 1.00000*algebraic[20]*algebraic[30]+algebraic[37]
    rates[21] = (algebraic[8]-states[21])/algebraic[40]
    algebraic[39] = ((1.00000*constants[1])/constants[16])*exp(22.0980+(constants[16]/constants[1])*(0.0365000*states[0]-25.9140))
    algebraic[42] = ((1.00000*constants[1])/constants[16])*exp(7.31300+(constants[16]/constants[1])*(-0.0399000*states[0]-15.7070))
    rates[7] = (algebraic[25]*states[6]+algebraic[42]*states[4])-(algebraic[39]+algebraic[32])*states[7]
    rates[11] = (algebraic[25]*states[10]+algebraic[42]*states[12])-(algebraic[39]+algebraic[32])*states[11]
    algebraic[44] = ((1.00000*constants[1])/constants[16])*exp(30.0160+(constants[16]/constants[1])*(0.0223000*states[0]-30.8880))*(power(5.40000/constants[3], 0.400000))
    algebraic[46] = ((1.00000*constants[1])/constants[16])*exp(30.0610+(constants[16]/constants[1])*(-0.0312000*states[0]-33.2430))
    rates[8] = algebraic[44]*states[4]-algebraic[46]*states[8]
    rates[13] = algebraic[44]*states[12]-algebraic[46]*states[13]
    algebraic[48] = states[4]*constants[17]*constants[18]
    algebraic[50] = states[12]*constants[19]
    rates[4] = (((algebraic[39]*states[7]+algebraic[46]*states[8])-(algebraic[44]+algebraic[42])*states[4])-algebraic[48])+algebraic[50]
    rates[12] = (((algebraic[39]*states[11]+algebraic[46]*states[13])-(algebraic[44]+algebraic[42])*states[12])+algebraic[48])-algebraic[50]
    algebraic[24] = ((constants[0]*constants[1])/constants[2])*log(constants[3]/states[1])
    algebraic[43] = 0.450000*exp(-(states[0]-constants[13]*algebraic[24])/20.0000)
    algebraic[49] = 1.00000+constants[11]/algebraic[43]
    algebraic[41] = 2.80000*exp(-(states[0]-constants[13]*algebraic[24])/180.000)
    algebraic[45] = 0.000700000*exp(-((states[0]-constants[13]*algebraic[24])+8.00000*constants[11])/4.80000)
    algebraic[51] = (algebraic[49]*algebraic[49])/(constants[12]/algebraic[45]+constants[11]/algebraic[41]+algebraic[49]*algebraic[49]*algebraic[49])
    algebraic[47] = 0.0400000*exp(-(states[0]-constants[13]*algebraic[24])/9.10000)
    algebraic[52] = 1.00000/(1.00000+constants[12]/algebraic[47])
    algebraic[53] = constants[14]*algebraic[51]+(1.00000-constants[14])*algebraic[52]
    algebraic[54] = constants[10]*(constants[1]/35.0000-55.0000/7.00000)*(power(constants[3]/5.40000, 1.0/2))*algebraic[53]*(states[0]-algebraic[24])
    algebraic[57] = constants[21]*states[16]*states[15]*(states[0]-algebraic[24])
    algebraic[55] = constants[15]*(constants[1]/35.0000-55.0000/7.00000)*(power(constants[3]/5.40000, 1.0/2))*states[4]*(states[0]-algebraic[24])
    algebraic[31] = ((constants[0]*constants[1])/constants[2])*log((constants[3]+constants[9]*constants[5])/(states[1]+constants[9]*states[2]))
    algebraic[56] = constants[20]*(power(states[14], 2.00000))*(states[0]-algebraic[31])
    algebraic[64] = ((((constants[32]*constants[3])/(constants[3]+constants[33]))*states[2])/(states[2]+constants[34]))/(1.00000+0.124500*exp((-0.100000*states[0]*constants[2])/(constants[0]*constants[1]))+0.0353000*exp((-states[0]*constants[2])/(constants[0]*constants[1])))
    algebraic[63] = (constants[31]*(states[0]-algebraic[24]))/(1.00000+exp((25.0000-states[0])/5.98000))
    algebraic[0] = custom_piecewise([greater_equal(voi-floor(voi/constants[8])*constants[8] , 100.000) & less_equal(voi-floor(voi/constants[8])*constants[8] , 103.000), -12.0000 , True, 0.00000])
    rates[1] = (-((algebraic[54]+algebraic[57]+algebraic[55]+algebraic[56]+algebraic[63]+algebraic[0])-2.00000*algebraic[64])/(constants[7]*constants[2]))*constants[6]
    algebraic[12] = ((constants[0]*constants[1])/constants[2])*log(constants[5]/states[2])
    algebraic[58] = constants[22]*(power(states[17], 3.00000))*states[18]*states[19]*(states[0]-algebraic[12])
    algebraic[59] = constants[24]*(states[0]-algebraic[12])
    algebraic[65] = (constants[35]*(exp((constants[38]*states[0]*constants[2])/(constants[0]*constants[1]))*(power(states[2], 3.00000))*constants[4]-exp(((constants[38]-1.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(constants[5], 3.00000))*states[3]*constants[37]))/((power(constants[40], 3.00000)+power(constants[5], 3.00000))*(constants[39]+constants[4])*(1.00000+constants[36]*exp(((constants[38]-1.00000)*states[0]*constants[2])/(constants[0]*constants[1]))))
    rates[2] = (-(algebraic[58]+algebraic[59]+3.00000*algebraic[64]+3.00000*algebraic[65])*constants[6])/(constants[7]*constants[2])
    algebraic[60] = (((constants[25]*states[21]*states[22]*states[23]*states[24]*(power(constants[26], 2.00000))*(states[0]-15.0000)*(power(constants[2], 2.00000)))/(constants[0]*constants[1]))*(0.250000*states[20]*exp((2.00000*(states[0]-15.0000)*constants[2])/(constants[0]*constants[1]))-constants[4]))/(exp((2.00000*(states[0]-15.0000)*constants[2])/(constants[0]*constants[1]))-1.00000)
    algebraic[38] = ((0.500000*constants[0]*constants[1])/constants[2])*log(constants[4]/states[3])
    algebraic[61] = constants[28]*(states[0]-algebraic[38])
    algebraic[62] = (constants[29]*states[3])/(states[3]+constants[30])
    algebraic[66] = algebraic[54]+algebraic[57]+algebraic[55]+algebraic[56]+algebraic[60]+algebraic[64]+algebraic[58]+algebraic[59]+algebraic[65]+algebraic[61]+algebraic[63]+algebraic[62]+algebraic[0]
    rates[0] = -algebraic[66]
    algebraic[70] = 1.00000/(1.00000+(constants[55]*constants[56])/(power(states[3]+constants[56], 2.00000)))
    algebraic[67] = constants[46]/(1.00000+(power(constants[44], 2.00000))/(power(states[3], 2.00000)))
    algebraic[68] = constants[45]*(states[25]-states[3])
    algebraic[69] = constants[43]*(states[20]-states[3])
    rates[3] = algebraic[70]*((((algebraic[68]-algebraic[67])*constants[41])/constants[7]+algebraic[69])-(((algebraic[61]+algebraic[62])-2.00000*algebraic[65])*constants[6])/(2.00000*constants[7]*constants[2]))
    algebraic[71] = constants[50]-(constants[50]-constants[51])/(1.00000+power(constants[49]/states[25], 2.00000))
    algebraic[73] = constants[48]*algebraic[71]
    rates[26] = -algebraic[73]*states[20]*states[26]+constants[53]*(1.00000-states[26])
    algebraic[76] = 1.00000/(1.00000+(constants[57]*constants[58])/(power(states[25]+constants[58], 2.00000)))
    algebraic[72] = constants[47]/algebraic[71]
    algebraic[74] = (algebraic[72]*(power(states[20], 2.00000))*states[26])/(constants[52]+algebraic[72]*(power(states[20], 2.00000)))
    algebraic[75] = constants[54]*algebraic[74]*(states[25]-states[20])
    rates[25] = algebraic[76]*(algebraic[67]-(algebraic[75]+algebraic[68]))
    algebraic[77] = 1.00000/(1.00000+(constants[59]*constants[60])/(power(states[20]+constants[60], 2.00000)))
    rates[20] = algebraic[77]*(((-algebraic[60]*constants[6])/(2.00000*constants[42]*constants[2])+(algebraic[75]*constants[41])/constants[42])-(algebraic[69]*constants[7])/constants[42])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[1] = ((1.00000*constants[1])/constants[16])*exp(24.3350+(constants[16]/constants[1])*(0.0112000*states[0]-25.9140))
    algebraic[13] = ((1.00000*constants[1])/constants[16])*exp(13.6880+(constants[16]/constants[1])*(-0.0603000*states[0]-15.7070))
    algebraic[3] = 1.00000/(1.00000+exp((states[0]+20.0000)/5.00000))
    algebraic[15] = 85.0000*exp(-(power(states[0]+45.0000, 2.00000))/320.000)+5.00000/(1.00000+exp((states[0]-20.0000)/5.00000))+3.00000
    algebraic[4] = 1.00000/(1.00000+exp((20.0000-states[0])/6.00000))
    algebraic[16] = 9.50000*exp(-(power(states[0]+40.0000, 2.00000))/1800.00)+0.800000
    algebraic[9] = 1.00000/(1.00000+exp((states[0]+20.0000)/7.00000))
    algebraic[21] = (1102.50*exp(-(power(states[0]+27.0000, 2.00000))/225.000)+200.000/(1.00000+exp((13.0000-states[0])/10.0000))+180.000/(1.00000+exp((states[0]+30.0000)/10.0000))+20.0000)/4.00000
    algebraic[10] = 0.750000/(1.00000+exp((states[0]+35.0000)/7.00000))+0.250000
    algebraic[22] = (562.000*exp(-(power(states[0]+27.0000, 2.00000))/240.000)+31.0000/(1.00000+exp((25.0000-states[0])/10.0000))+80.0000/(1.00000+exp((states[0]+30.0000)/10.0000)))/2.00000
    algebraic[11] = 0.400000/(1.00000+power(states[20]/0.0500000, 2.00000))+0.600000
    algebraic[23] = 80.0000/(1.00000+power(states[20]/0.0500000, 2.00000))+2.00000
    algebraic[25] = ((1.00000*constants[1])/constants[16])*exp(22.7460+(constants[16]/constants[1])*(0.00000*states[0]-25.9140))
    algebraic[32] = ((1.00000*constants[1])/constants[16])*exp(13.1930+(constants[16]/constants[1])*(0.00000*states[0]-15.7070))
    algebraic[2] = 1.00000/(1.00000+exp((-5.00000-states[0])/14.0000))
    algebraic[14] = 1400.00/(power(1.00000+exp((5.00000-states[0])/6.00000), 1.0/2))
    algebraic[26] = 1.00000/(1.00000+exp((states[0]-35.0000)/15.0000))
    algebraic[33] = 1.00000*algebraic[14]*algebraic[26]+80.0000
    algebraic[5] = 1.00000/(power(1.00000+exp((-56.8600-states[0])/9.03000), 2.00000))
    algebraic[17] = 1.00000/(1.00000+exp((-60.0000-states[0])/5.00000))
    algebraic[27] = 0.100000/(1.00000+exp((states[0]+35.0000)/5.00000))+0.100000/(1.00000+exp((states[0]-50.0000)/200.000))
    algebraic[34] = 1.00000*algebraic[17]*algebraic[27]
    algebraic[6] = 1.00000/(power(1.00000+exp(((states[0]+71.5500)-constants[23])/7.43000), 2.00000))
    algebraic[18] = custom_piecewise([less(states[0] , -40.0000+constants[23]), 0.0570000*exp(-((states[0]+80.0000)-constants[23])/6.80000) , True, 0.00000])
    algebraic[28] = custom_piecewise([less(states[0] , -40.0000+constants[23]), 2.70000*exp(0.0790000*(states[0]-constants[23]))+310000.*exp(0.348500*(states[0]-constants[23])) , True, 0.770000/(0.130000*(1.00000+exp(((states[0]+10.6600)-constants[23])/-11.1000)))])
    algebraic[35] = 1.00000/(algebraic[18]+algebraic[28])
    algebraic[7] = 1.00000/(power(1.00000+exp(((states[0]+71.5500)-constants[23])/7.43000), 2.00000))
    algebraic[19] = custom_piecewise([less(states[0] , -40.0000+constants[23]), (((-25428.0*exp(0.244400*(states[0]-constants[23]))-6.94800e-06*exp(-0.0439100*(states[0]-constants[23])))*(states[0]+37.7800))/1.00000)/(1.00000+exp(0.311000*((states[0]+79.2300)-constants[23]))) , True, 0.00000])
    algebraic[29] = custom_piecewise([less(states[0] , -40.0000+constants[23]), (0.0242400*exp(-0.0105200*(states[0]-constants[23])))/(1.00000+exp(-0.137800*((states[0]+40.1400)-constants[23]))) , True, (0.600000*exp(0.0570000*(states[0]-constants[23])))/(1.00000+exp(-0.100000*((states[0]+32.0000)-constants[23])))])
    algebraic[36] = 1.00000/(algebraic[19]+algebraic[29])
    algebraic[8] = 1.00000/(1.00000+exp((constants[27]-states[0])/7.50000))
    algebraic[20] = 1.40000/(1.00000+exp((-35.0000-states[0])/13.0000))+0.250000
    algebraic[30] = 1.40000/(1.00000+exp((states[0]+5.00000)/5.00000))
    algebraic[37] = 1.00000/(1.00000+exp((50.0000-states[0])/20.0000))
    algebraic[40] = 1.00000*algebraic[20]*algebraic[30]+algebraic[37]
    algebraic[39] = ((1.00000*constants[1])/constants[16])*exp(22.0980+(constants[16]/constants[1])*(0.0365000*states[0]-25.9140))
    algebraic[42] = ((1.00000*constants[1])/constants[16])*exp(7.31300+(constants[16]/constants[1])*(-0.0399000*states[0]-15.7070))
    algebraic[44] = ((1.00000*constants[1])/constants[16])*exp(30.0160+(constants[16]/constants[1])*(0.0223000*states[0]-30.8880))*(power(5.40000/constants[3], 0.400000))
    algebraic[46] = ((1.00000*constants[1])/constants[16])*exp(30.0610+(constants[16]/constants[1])*(-0.0312000*states[0]-33.2430))
    algebraic[48] = states[4]*constants[17]*constants[18]
    algebraic[50] = states[12]*constants[19]
    algebraic[24] = ((constants[0]*constants[1])/constants[2])*log(constants[3]/states[1])
    algebraic[43] = 0.450000*exp(-(states[0]-constants[13]*algebraic[24])/20.0000)
    algebraic[49] = 1.00000+constants[11]/algebraic[43]
    algebraic[41] = 2.80000*exp(-(states[0]-constants[13]*algebraic[24])/180.000)
    algebraic[45] = 0.000700000*exp(-((states[0]-constants[13]*algebraic[24])+8.00000*constants[11])/4.80000)
    algebraic[51] = (algebraic[49]*algebraic[49])/(constants[12]/algebraic[45]+constants[11]/algebraic[41]+algebraic[49]*algebraic[49]*algebraic[49])
    algebraic[47] = 0.0400000*exp(-(states[0]-constants[13]*algebraic[24])/9.10000)
    algebraic[52] = 1.00000/(1.00000+constants[12]/algebraic[47])
    algebraic[53] = constants[14]*algebraic[51]+(1.00000-constants[14])*algebraic[52]
    algebraic[54] = constants[10]*(constants[1]/35.0000-55.0000/7.00000)*(power(constants[3]/5.40000, 1.0/2))*algebraic[53]*(states[0]-algebraic[24])
    algebraic[57] = constants[21]*states[16]*states[15]*(states[0]-algebraic[24])
    algebraic[55] = constants[15]*(constants[1]/35.0000-55.0000/7.00000)*(power(constants[3]/5.40000, 1.0/2))*states[4]*(states[0]-algebraic[24])
    algebraic[31] = ((constants[0]*constants[1])/constants[2])*log((constants[3]+constants[9]*constants[5])/(states[1]+constants[9]*states[2]))
    algebraic[56] = constants[20]*(power(states[14], 2.00000))*(states[0]-algebraic[31])
    algebraic[64] = ((((constants[32]*constants[3])/(constants[3]+constants[33]))*states[2])/(states[2]+constants[34]))/(1.00000+0.124500*exp((-0.100000*states[0]*constants[2])/(constants[0]*constants[1]))+0.0353000*exp((-states[0]*constants[2])/(constants[0]*constants[1])))
    algebraic[63] = (constants[31]*(states[0]-algebraic[24]))/(1.00000+exp((25.0000-states[0])/5.98000))
    algebraic[0] = custom_piecewise([greater_equal(voi-floor(voi/constants[8])*constants[8] , 100.000) & less_equal(voi-floor(voi/constants[8])*constants[8] , 103.000), -12.0000 , True, 0.00000])
    algebraic[12] = ((constants[0]*constants[1])/constants[2])*log(constants[5]/states[2])
    algebraic[58] = constants[22]*(power(states[17], 3.00000))*states[18]*states[19]*(states[0]-algebraic[12])
    algebraic[59] = constants[24]*(states[0]-algebraic[12])
    algebraic[65] = (constants[35]*(exp((constants[38]*states[0]*constants[2])/(constants[0]*constants[1]))*(power(states[2], 3.00000))*constants[4]-exp(((constants[38]-1.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(constants[5], 3.00000))*states[3]*constants[37]))/((power(constants[40], 3.00000)+power(constants[5], 3.00000))*(constants[39]+constants[4])*(1.00000+constants[36]*exp(((constants[38]-1.00000)*states[0]*constants[2])/(constants[0]*constants[1]))))
    algebraic[60] = (((constants[25]*states[21]*states[22]*states[23]*states[24]*(power(constants[26], 2.00000))*(states[0]-15.0000)*(power(constants[2], 2.00000)))/(constants[0]*constants[1]))*(0.250000*states[20]*exp((2.00000*(states[0]-15.0000)*constants[2])/(constants[0]*constants[1]))-constants[4]))/(exp((2.00000*(states[0]-15.0000)*constants[2])/(constants[0]*constants[1]))-1.00000)
    algebraic[38] = ((0.500000*constants[0]*constants[1])/constants[2])*log(constants[4]/states[3])
    algebraic[61] = constants[28]*(states[0]-algebraic[38])
    algebraic[62] = (constants[29]*states[3])/(states[3]+constants[30])
    algebraic[66] = algebraic[54]+algebraic[57]+algebraic[55]+algebraic[56]+algebraic[60]+algebraic[64]+algebraic[58]+algebraic[59]+algebraic[65]+algebraic[61]+algebraic[63]+algebraic[62]+algebraic[0]
    algebraic[70] = 1.00000/(1.00000+(constants[55]*constants[56])/(power(states[3]+constants[56], 2.00000)))
    algebraic[67] = constants[46]/(1.00000+(power(constants[44], 2.00000))/(power(states[3], 2.00000)))
    algebraic[68] = constants[45]*(states[25]-states[3])
    algebraic[69] = constants[43]*(states[20]-states[3])
    algebraic[71] = constants[50]-(constants[50]-constants[51])/(1.00000+power(constants[49]/states[25], 2.00000))
    algebraic[73] = constants[48]*algebraic[71]
    algebraic[76] = 1.00000/(1.00000+(constants[57]*constants[58])/(power(states[25]+constants[58], 2.00000)))
    algebraic[72] = constants[47]/algebraic[71]
    algebraic[74] = (algebraic[72]*(power(states[20], 2.00000))*states[26])/(constants[52]+algebraic[72]*(power(states[20], 2.00000)))
    algebraic[75] = constants[54]*algebraic[74]*(states[25]-states[20])
    algebraic[77] = 1.00000/(1.00000+(constants[59]*constants[60])/(power(states[20]+constants[60], 2.00000)))
    return algebraic

def custom_piecewise(cases):
    """Compute result of a piecewise function"""
    return select(cases[0::2],cases[1::2])

def solve_model():
    """Solve model with ODE solver"""
    from scipy.integrate import ode
    # Initialise constants and state variables
    (init_states, constants) = initConsts()

    # Set timespan to solve over
    voi = linspace(0, 10, 500)

    # Construct ODE object to solve
    r = ode(computeRates)
    r.set_integrator('vode', method='bdf', atol=1e-06, rtol=1e-06, max_step=1)
    r.set_initial_value(init_states, voi[0])
    r.set_f_params(constants)

    # Solve model
    states = array([[0.0] * len(voi)] * sizeStates)
    states[:,0] = init_states
    for (i,t) in enumerate(voi[1:]):
        if r.successful():
            r.integrate(t)
            states[:,i+1] = r.y
        else:
            break

    # Compute algebraic variables
    algebraic = computeAlgebraic(constants, states, voi)
    return (voi, states, algebraic)

def plot_model(voi, states, algebraic):
    """Plot variables against variable of integration"""
    import pylab
    (legend_states, legend_algebraic, legend_voi, legend_constants) = createLegends()
    pylab.figure(1)
    pylab.plot(voi,vstack((states,algebraic)).T)
    pylab.xlabel(legend_voi)
    pylab.legend(legend_states + legend_algebraic, loc='best')
    pylab.show()

if __name__ == "__main__":
    (voi, states, algebraic) = solve_model()
    plot_model(voi, states, algebraic)