# Size of variable arrays:
sizeAlgebraic = 51
sizeStates = 23
sizeConstants = 69
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_constants[0] = "R in component cell_parameters (millijoule_per_mole_kelvin)"
    legend_constants[1] = "T in component cell_parameters (kelvin)"
    legend_constants[2] = "F in component cell_parameters (coulomb_per_mole)"
    legend_constants[3] = "Cm in component cell_parameters (microF)"
    legend_constants[4] = "v_i in component cell_parameters (microlitre)"
    legend_constants[5] = "v_SR in component cell_parameters (microlitre)"
    legend_constants[6] = "Na_o in component cell_parameters (millimolar)"
    legend_constants[7] = "K_o in component cell_parameters (millimolar)"
    legend_constants[8] = "Ca_o in component cell_parameters (millimolar)"
    legend_states[0] = "V in component membrane_potential (millivolt)"
    legend_algebraic[21] = "i_Na in component fast_sodium_current (nanoA)"
    legend_algebraic[23] = "i_K1 in component time_independent_potassium_current (nanoA)"
    legend_algebraic[24] = "i_to in component transient_outward_current (nanoA)"
    legend_algebraic[22] = "i_K in component time_dependent_rectifier_potassium_current (nanoA)"
    legend_algebraic[28] = "i_Ca_L in component L_type_Ca_channel (nanoA)"
    legend_algebraic[32] = "i_NaK in component sodium_potassium_pump (nanoA)"
    legend_algebraic[30] = "i_NaCa in component sodium_calcium_exchanger (nanoA)"
    legend_algebraic[39] = "i_b_Na in component sodium_background_current (nanoA)"
    legend_algebraic[36] = "i_b_K in component potassium_background_current (nanoA)"
    legend_algebraic[35] = "i_b_Ca in component calcium_background_current (nanoA)"
    legend_algebraic[0] = "i_Stim in component membrane_potential (nanoA)"
    legend_constants[9] = "stim_start in component membrane_potential (second)"
    legend_constants[10] = "stim_end in component membrane_potential (second)"
    legend_constants[11] = "stim_period in component membrane_potential (second)"
    legend_constants[12] = "stim_duration in component membrane_potential (second)"
    legend_constants[13] = "stim_amplitude in component membrane_potential (nanoA)"
    legend_algebraic[8] = "E_Na in component reversal_potentials (millivolt)"
    legend_algebraic[15] = "E_K in component reversal_potentials (millivolt)"
    legend_algebraic[19] = "E_Ca in component reversal_potentials (millivolt)"
    legend_algebraic[20] = "E_mh in component reversal_potentials (millivolt)"
    legend_states[1] = "K_i in component intracellular_potassium_concentration (millimolar)"
    legend_states[2] = "Na_i in component intracellular_sodium_concentration (millimolar)"
    legend_states[3] = "Ca_i in component intracellular_calcium_concentration (millimolar)"
    legend_constants[14] = "g_Na in component fast_sodium_current (microS)"
    legend_states[4] = "m in component fast_sodium_current_m_gate (dimensionless)"
    legend_states[5] = "h in component fast_sodium_current_h_gate (dimensionless)"
    legend_algebraic[9] = "alpha_m in component fast_sodium_current_m_gate (per_second)"
    legend_algebraic[16] = "beta_m in component fast_sodium_current_m_gate (per_second)"
    legend_constants[15] = "delta_m in component fast_sodium_current_m_gate (millivolt)"
    legend_algebraic[1] = "E0_m in component fast_sodium_current_m_gate (millivolt)"
    legend_algebraic[2] = "alpha_h in component fast_sodium_current_h_gate (per_second)"
    legend_algebraic[10] = "beta_h in component fast_sodium_current_h_gate (per_second)"
    legend_constants[16] = "shift_h in component fast_sodium_current_h_gate (millivolt)"
    legend_constants[17] = "i_Kmax in component time_dependent_rectifier_potassium_current (nanoA)"
    legend_states[6] = "x in component time_dependent_rectifier_potassium_current_x_gate (dimensionless)"
    legend_algebraic[3] = "alpha_x in component time_dependent_rectifier_potassium_current_x_gate (per_second)"
    legend_algebraic[11] = "beta_x in component time_dependent_rectifier_potassium_current_x_gate (per_second)"
    legend_constants[18] = "K_mk1 in component time_independent_potassium_current (millimolar)"
    legend_constants[19] = "g_K1 in component time_independent_potassium_current (microS)"
    legend_constants[20] = "g_to in component transient_outward_current (microS)"
    legend_states[7] = "s in component transient_outward_current_s_gate (dimensionless)"
    legend_states[8] = "r in component transient_outward_current_r_gate (dimensionless)"
    legend_algebraic[4] = "alpha_s in component transient_outward_current_s_gate (per_second)"
    legend_algebraic[12] = "beta_s in component transient_outward_current_s_gate (per_second)"
    legend_algebraic[27] = "i_Ca_L_Na in component L_type_Ca_channel (nanoA)"
    legend_algebraic[26] = "i_Ca_L_K in component L_type_Ca_channel (nanoA)"
    legend_algebraic[25] = "i_Ca_L_Ca in component L_type_Ca_channel (nanoA)"
    legend_constants[21] = "P_Ca_L_Ca in component L_type_Ca_channel (nanoA_per_millimolar)"
    legend_states[9] = "d in component L_type_Ca_channel_d_gate (dimensionless)"
    legend_states[10] = "f in component L_type_Ca_channel_f_gate (dimensionless)"
    legend_algebraic[13] = "alpha_d in component L_type_Ca_channel_d_gate (per_second)"
    legend_algebraic[17] = "beta_d in component L_type_Ca_channel_d_gate (per_second)"
    legend_algebraic[5] = "E0_d in component L_type_Ca_channel_d_gate (millivolt)"
    legend_constants[22] = "speed_d in component L_type_Ca_channel_d_gate (dimensionless)"
    legend_algebraic[14] = "alpha_f in component L_type_Ca_channel_f_gate (per_second)"
    legend_algebraic[18] = "beta_f in component L_type_Ca_channel_f_gate (per_second)"
    legend_constants[23] = "speed_f in component L_type_Ca_channel_f_gate (dimensionless)"
    legend_algebraic[6] = "E0_f in component L_type_Ca_channel_f_gate (millivolt)"
    legend_constants[24] = "i_NaCa_max in component sodium_calcium_exchanger (nanoA_per_millimolar4)"
    legend_constants[25] = "gamma in component sodium_calcium_exchanger (dimensionless)"
    legend_constants[26] = "i_NaK_max in component sodium_potassium_pump (nanoA)"
    legend_constants[27] = "K_mK in component sodium_potassium_pump (millimolar)"
    legend_constants[28] = "K_mNa in component sodium_potassium_pump (millimolar)"
    legend_constants[29] = "g_b_Ca in component calcium_background_current (microS)"
    legend_constants[30] = "g_b_K in component potassium_background_current (microS)"
    legend_constants[31] = "g_b_Na in component sodium_background_current (microS)"
    legend_states[11] = "F_CaMK in component CaMKII_factor (dimensionless)"
    legend_algebraic[7] = "Inf_CaMK in component CaMKII_factor (dimensionless)"
    legend_constants[32] = "Tau_CaMK in component CaMKII_factor (second)"
    legend_states[12] = "Cmdn_Ca in component calmodulin (millimolar)"
    legend_algebraic[41] = "j_rel in component RyR (millimolar_per_second)"
    legend_algebraic[40] = "K_rel in component RyR (per_second)"
    legend_algebraic[37] = "F_rel in component RyR (dimensionless)"
    legend_states[13] = "Ca_SR in component SR_calcium_concentration (millimolar)"
    legend_constants[33] = "K_rel_max in component RyR (per_second)"
    legend_states[14] = "F_SRCa_RyR in component RyR (millimolar)"
    legend_constants[34] = "Tau_SRCa_RyR in component RyR (second)"
    legend_algebraic[29] = "N_CaMK in component RyR (dimensionless)"
    legend_constants[35] = "gain_k1 in component RyR (dimensionless)"
    legend_constants[36] = "gain_k2 in component RyR (dimensionless)"
    legend_constants[37] = "gain_k3 in component RyR (dimensionless)"
    legend_constants[38] = "gain_k4 in component RyR (dimensionless)"
    legend_algebraic[31] = "k_1 in component RyR (per_second)"
    legend_algebraic[33] = "k_2 in component RyR (per_second)"
    legend_algebraic[34] = "k_3 in component RyR (per_second)"
    legend_constants[53] = "k_4 in component RyR (per_second)"
    legend_states[15] = "F_1 in component RyR (dimensionless)"
    legend_states[16] = "F_2 in component RyR (dimensionless)"
    legend_algebraic[38] = "F_3 in component RyR (dimensionless)"
    legend_constants[39] = "K_leak_rate in component RyR (per_second)"
    legend_algebraic[44] = "j_up in component SERCA (millimolar_per_second)"
    legend_constants[40] = "V_max_f in component SERCA (millimolar_per_second)"
    legend_constants[41] = "V_max_r in component SERCA (millimolar_per_second)"
    legend_algebraic[42] = "f_b in component SERCA (dimensionless)"
    legend_algebraic[43] = "r_b in component SERCA (dimensionless)"
    legend_constants[42] = "Cmdn_tot in component calmodulin (millimolar)"
    legend_constants[43] = "alpha_cmdn in component calmodulin (per_millimolar_per_second)"
    legend_constants[44] = "beta_cmdn in component calmodulin (per_second)"
    legend_algebraic[45] = "dCmdn_Ca_dtime in component calmodulin (millimolar_per_second)"
    legend_states[17] = "Trpn_Ca in component troponin (millimolar)"
    legend_constants[45] = "Trpn_tot in component troponin (millimolar)"
    legend_constants[46] = "alpha_trpn in component troponin (per_millimolar_per_second)"
    legend_constants[47] = "beta_trpn in component troponin (per_second)"
    legend_algebraic[48] = "Force_norm in component Force (dimensionless)"
    legend_algebraic[49] = "dTrpn_Ca_dtime in component troponin (millimolar_per_second)"
    legend_algebraic[50] = "Force in component Force (N_per_mm2)"
    legend_constants[48] = "zeta in component Force (N_per_mm2)"
    legend_constants[68] = "Force_max in component Force (dimensionless)"
    legend_constants[54] = "phi_SL in component Force (dimensionless)"
    legend_constants[65] = "P_1_max in component Force (dimensionless)"
    legend_constants[66] = "P_2_max in component Force (dimensionless)"
    legend_constants[67] = "P_3_max in component Force (dimensionless)"
    legend_constants[64] = "sigma_paths in component Force (dimensionless)"
    legend_states[18] = "N_0 in component Force (dimensionless)"
    legend_states[19] = "P_0 in component Force (dimensionless)"
    legend_states[20] = "P_1 in component Force (dimensionless)"
    legend_states[21] = "P_2 in component Force (dimensionless)"
    legend_states[22] = "P_3 in component Force (dimensionless)"
    legend_algebraic[47] = "N_1 in component Force (dimensionless)"
    legend_algebraic[46] = "alpha_tm in component Force (per_second)"
    legend_constants[49] = "beta_tm in component Force (per_second)"
    legend_constants[57] = "K_tm in component Force (dimensionless)"
    legend_constants[58] = "N_tm in component Force (dimensionless)"
    legend_constants[50] = "SL in component Force (micrometre)"
    legend_constants[55] = "SL_norm in component Force (dimensionless)"
    legend_constants[56] = "f_01 in component Force (per_second)"
    legend_constants[59] = "f_12 in component Force (per_second)"
    legend_constants[63] = "f_23 in component Force (per_second)"
    legend_constants[60] = "g_01 in component Force (per_second)"
    legend_constants[61] = "g_12 in component Force (per_second)"
    legend_constants[62] = "g_23 in component Force (per_second)"
    legend_constants[51] = "f_XB in component Force (per_second)"
    legend_constants[52] = "g_XB in component Force (per_second)"
    legend_rates[0] = "d/dt V in component membrane_potential (millivolt)"
    legend_rates[4] = "d/dt m in component fast_sodium_current_m_gate (dimensionless)"
    legend_rates[5] = "d/dt h in component fast_sodium_current_h_gate (dimensionless)"
    legend_rates[6] = "d/dt x in component time_dependent_rectifier_potassium_current_x_gate (dimensionless)"
    legend_rates[7] = "d/dt s in component transient_outward_current_s_gate (dimensionless)"
    legend_rates[8] = "d/dt r in component transient_outward_current_r_gate (dimensionless)"
    legend_rates[9] = "d/dt d in component L_type_Ca_channel_d_gate (dimensionless)"
    legend_rates[10] = "d/dt f in component L_type_Ca_channel_f_gate (dimensionless)"
    legend_rates[11] = "d/dt F_CaMK in component CaMKII_factor (dimensionless)"
    legend_rates[15] = "d/dt F_1 in component RyR (dimensionless)"
    legend_rates[16] = "d/dt F_2 in component RyR (dimensionless)"
    legend_rates[14] = "d/dt F_SRCa_RyR in component RyR (millimolar)"
    legend_rates[12] = "d/dt Cmdn_Ca in component calmodulin (millimolar)"
    legend_rates[17] = "d/dt Trpn_Ca in component troponin (millimolar)"
    legend_rates[3] = "d/dt Ca_i in component intracellular_calcium_concentration (millimolar)"
    legend_rates[13] = "d/dt Ca_SR in component SR_calcium_concentration (millimolar)"
    legend_rates[2] = "d/dt Na_i in component intracellular_sodium_concentration (millimolar)"
    legend_rates[1] = "d/dt K_i in component intracellular_potassium_concentration (millimolar)"
    legend_rates[18] = "d/dt N_0 in component Force (dimensionless)"
    legend_rates[19] = "d/dt P_0 in component Force (dimensionless)"
    legend_rates[20] = "d/dt P_1 in component Force (dimensionless)"
    legend_rates[21] = "d/dt P_2 in component Force (dimensionless)"
    legend_rates[22] = "d/dt P_3 in component Force (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 8314.472
    constants[1] = 310
    constants[2] = 96485.3415
    constants[3] = 9.5e-5
    constants[4] = 1.6404e-5
    constants[5] = 3.3477e-6
    constants[6] = 140
    constants[7] = 4
    constants[8] = 2
    states[0] = -92.849333
    constants[9] = 0
    constants[10] = 1000
    constants[11] = 0.5
    constants[12] = 0.002
    constants[13] = -4
    states[1] = 138.22
    states[2] = 5.8041
    states[3] = 9.91e-6
    constants[14] = 2.5
    states[4] = 0.0013809
    states[5] = 0.99569
    constants[15] = 1e-5
    constants[16] = 0
    constants[17] = 1
    states[6] = 5.1127e-2
    constants[18] = 10
    constants[19] = 1
    constants[20] = 0.005
    states[7] = 0.95854
    states[8] = 1.5185e-8
    constants[21] = 0.25
    states[9] = 1.7908e-8
    states[10] = 1
    constants[22] = 3
    constants[23] = 0.5
    constants[24] = 0.0005
    constants[25] = 0.5
    constants[26] = 1.36
    constants[27] = 1
    constants[28] = 21.7
    constants[29] = 0.00025
    constants[30] = 0.0006
    constants[31] = 0.0006
    states[11] = 1.028
    constants[32] = 0.8
    states[12] = 3.9636e-6
    states[13] = 0.24886
    constants[33] = 500
    states[14] = 0.25089
    constants[34] = 0.05
    constants[35] = 1
    constants[36] = 1
    constants[37] = 1
    constants[38] = 1
    states[15] = 0.5268
    states[16] = 8.7508e-6
    constants[39] = 0
    constants[40] = 0.292
    constants[41] = 0.391
    constants[42] = 0.02
    constants[43] = 10000
    constants[44] = 500
    states[17] = 2.7661e-4
    constants[45] = 0.07
    constants[46] = 80000
    constants[47] = 200
    constants[48] = 0.1
    states[18] = 0.99917
    states[19] = 9.8593e-5
    states[20] = 1.3331e-4
    states[21] = 2.3505e-4
    states[22] = 1.5349e-4
    constants[49] = 40
    constants[50] = 2.15
    constants[51] = 10
    constants[52] = 30
    constants[53] = constants[38]*1.80000
    constants[54] = custom_piecewise([greater_equal(constants[50] , 1.70000) & less_equal(constants[50] , 2.00000), (constants[50]-0.600000)/1.40000 , greater(constants[50] , 2.00000) & less_equal(constants[50] , 2.20000), 1.00000 , greater(constants[50] , 2.20000) & less_equal(constants[50] , 2.30000), (3.60000-constants[50])/1.40000 , True, float('nan')])
    constants[55] = (constants[50]-1.70000)/0.700000
    constants[56] = 3.00000*constants[51]
    constants[57] = 1.00000/(1.00000+(constants[47]/constants[46])/(0.00170000-0.000900000*constants[55]))
    constants[58] = 3.50000+2.50000*constants[55]
    constants[59] = 10.0000*constants[51]
    constants[60] = constants[52]*(2.00000-constants[55])
    constants[61] = 2.00000*constants[52]*(2.00000-constants[55])
    constants[62] = 3.00000*constants[52]*(2.00000-constants[55])
    constants[63] = 7.00000*constants[51]
    constants[64] = 1.00000*constants[52]*2.00000*constants[52]*3.00000*constants[52]+1.00000*constants[56]*2.00000*constants[52]*3.00000*constants[52]+1.00000*constants[56]*1.00000*constants[59]*3.00000*constants[52]+1.00000*constants[56]*1.00000*constants[59]*1.00000*constants[63]
    constants[65] = (1.00000*constants[56]*2.00000*constants[52]*3.00000*constants[52])/constants[64]
    constants[66] = (1.00000*constants[56]*1.00000*constants[59]*3.00000*constants[52])/constants[64]
    constants[67] = (1.00000*constants[56]*1.00000*constants[59]*1.00000*constants[63])/constants[64]
    constants[68] = constants[65]+2.00000*constants[66]+3.00000*constants[67]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[8] = 333.000*(1.00000/(1.00000+exp(-(states[0]+4.00000)/5.00000))-states[8])
    rates[14] = (states[13]-states[14])/constants[34]
    rates[12] = constants[43]*(constants[42]-states[12])*states[3]-constants[44]*states[12]
    rates[21] = -(constants[63]+constants[61])*states[21]+constants[59]*states[20]+constants[62]*states[22]
    rates[22] = -constants[62]*states[22]+constants[63]*states[21]
    algebraic[7] = states[12]/5.00000e-05
    rates[11] = (algebraic[7]-states[11])/constants[32]
    algebraic[2] = 20.0000*exp(-0.125000*((states[0]+75.0000)-constants[16]))
    algebraic[10] = 2000.00/(1.00000+320.000*exp(-0.100000*((states[0]+75.0000)-constants[16])))
    rates[5] = algebraic[2]*(1.00000-states[5])-algebraic[10]*states[5]
    algebraic[3] = (0.500000*exp(0.0826000*(states[0]+50.0000)))/(1.00000+exp(0.0570000*(states[0]+50.0000)))
    algebraic[11] = (1.30000*exp(-0.0600000*(states[0]+20.0000)))/(1.00000+exp(-0.0400000*(states[0]+20.0000)))
    rates[6] = algebraic[3]*(1.00000-states[6])-algebraic[11]*states[6]
    algebraic[4] = 0.0330000*exp(-states[0]/17.0000)
    algebraic[12] = 33.0000/(1.00000+exp(-0.125000*(states[0]+10.0000)))
    rates[7] = algebraic[4]*(1.00000-states[7])-algebraic[12]*states[7]
    algebraic[1] = states[0]+41.0000
    algebraic[9] = custom_piecewise([less(fabs(algebraic[1]) , constants[15]), 2000.00 , True, (200.000*algebraic[1])/(1.00000-exp(-0.100000*algebraic[1]))])
    algebraic[16] = 8000.00*exp(-0.0560000*(states[0]+66.0000))
    rates[4] = algebraic[9]*(1.00000-states[4])-algebraic[16]*states[4]
    algebraic[5] = (states[0]+24.0000)-5.00000
    algebraic[13] = custom_piecewise([less(fabs(algebraic[5]) , 1.00000e-05), constants[22]*120.000 , True, (constants[22]*30.0000*algebraic[5])/(1.00000-exp(-algebraic[5]/4.00000))])
    algebraic[17] = custom_piecewise([less(fabs(algebraic[5]) , 1.00000e-05), constants[22]*120.000 , True, (constants[22]*-12.0000*algebraic[5])/(1.00000-exp(algebraic[5]/10.0000))])
    rates[9] = algebraic[13]*(1.00000-states[9])-algebraic[17]*states[9]
    algebraic[6] = states[0]+34.0000
    algebraic[14] = custom_piecewise([less(fabs(algebraic[6]) , 1.00000e-05), constants[23]*25.0000 , True, (constants[23]*6.25000*algebraic[6])/(-1.00000+exp(algebraic[6]/4.00000))])
    algebraic[18] = (constants[23]*50.0000)/(1.00000+exp(-algebraic[6]/4.00000))
    rates[10] = algebraic[14]*(1.00000-states[10])-algebraic[18]*states[10]
    algebraic[27] = ((0.0100000*states[9]*states[10]*constants[21]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]*(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))))*(states[2]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-constants[6]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[26] = ((0.00200000*states[9]*states[10]*constants[21]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]*(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))))*(states[1]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-constants[7]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[25] = ((4.00000*states[9]*states[10]*constants[21]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]*(1.00000-exp((-2.00000*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))))*(states[3]*exp((100.000*constants[2])/(constants[0]*constants[1]))-constants[8]*exp((-2.00000*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[28] = algebraic[25]+algebraic[26]+algebraic[27]
    algebraic[31] = constants[35]*(3.06250e+07*(power(states[3], 2.00000))-245.000*algebraic[28])
    algebraic[33] = (constants[36]*450.000)/(1.00000+0.360000/states[13])
    rates[16] = algebraic[31]*states[15]-algebraic[33]*states[16]
    algebraic[29] = power(states[11]/0.700000, 2.00000)
    algebraic[34] = constants[37]*1.88500*(power(states[14]/0.220000, algebraic[29]))
    algebraic[38] = 1.00000-(states[15]+states[16])
    rates[15] = (algebraic[34]*algebraic[38]-constants[53]*states[15])-algebraic[31]*states[15]
    algebraic[15] = ((constants[0]*constants[1])/constants[2])*log(constants[7]/states[1])
    algebraic[23] = (((constants[19]*constants[7])/(constants[7]+constants[18]))*(states[0]-algebraic[15]))/(1.00000+exp((2.00000*constants[2]*((states[0]-algebraic[15])-10.0000))/(constants[0]*constants[1])))
    algebraic[24] = constants[20]*states[7]*states[8]*(states[0]-algebraic[15])
    algebraic[22] = (constants[17]*states[6]*(states[1]-constants[7]*exp((-states[0]*constants[2])/(constants[0]*constants[1]))))/140.000
    algebraic[32] = (((((constants[26]*constants[7])/(constants[27]+constants[7]))*states[2])/(constants[28]+states[2]))*1.00000)/(1.00000+0.124500*exp((-0.100000*states[0]*constants[2])/(constants[0]*constants[1]))+0.0353000*exp((-states[0]*constants[2])/(constants[0]*constants[1])))
    algebraic[36] = constants[30]*(states[0]-algebraic[15])
    rates[1] = -((algebraic[23]+algebraic[22]+algebraic[24]+algebraic[36]+algebraic[26])-2.00000*algebraic[32])/(constants[4]*constants[2])
    algebraic[20] = ((constants[0]*constants[1])/constants[2])*log((constants[6]+0.120000*constants[7])/(states[2]+0.120000*states[1]))
    algebraic[21] = constants[14]*(power(states[4], 3.00000))*states[5]*(states[0]-algebraic[20])
    algebraic[30] = (constants[24]*(exp((constants[25]*states[0]*constants[2])/(constants[0]*constants[1]))*(power(states[2], 3.00000))*constants[8]-exp(((constants[25]-1.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(constants[6], 3.00000))*states[3]))/(1.00000+states[3]/0.00690000)
    algebraic[8] = ((constants[0]*constants[1])/constants[2])*log(constants[6]/states[2])
    algebraic[39] = constants[31]*(states[0]-algebraic[8])
    algebraic[19] = ((0.500000*constants[0]*constants[1])/constants[2])*log(constants[8]/states[3])
    algebraic[35] = constants[29]*(states[0]-algebraic[19])
    algebraic[0] = custom_piecewise([greater_equal(voi , constants[9]) & less_equal(voi , constants[10]) & less_equal((voi-constants[9])-floor((voi-constants[9])/constants[11])*constants[11] , constants[12]), constants[13] , True, 0.00000])
    rates[0] = (-1.00000/constants[3])*(algebraic[21]+algebraic[39]+algebraic[23]+algebraic[22]+algebraic[24]+algebraic[36]+algebraic[28]+algebraic[35]+algebraic[30]+algebraic[32]+algebraic[0])
    rates[2] = -(algebraic[21]+algebraic[39]+algebraic[27]+3.00000*algebraic[30]+3.00000*algebraic[32])/(constants[4]*constants[2])
    algebraic[40] = (constants[33]*states[14])/(states[14]+0.200000)
    algebraic[37] = power(states[16]/(states[16]+0.250000), 2.00000)
    algebraic[41] = (algebraic[40]*algebraic[37]+constants[39])*(states[13]-states[3])
    algebraic[42] = power(states[3]/0.000240000, 2.00000)
    algebraic[43] = power(states[13]/1.64000, 2.00000)
    algebraic[44] = (states[11]*constants[40]*algebraic[42]-constants[41]*algebraic[43])/(1.00000+algebraic[42]+algebraic[43])
    rates[13] = (algebraic[44]*constants[4])/constants[5]-algebraic[41]
    algebraic[46] = constants[49]*(power(states[17]/(constants[45]*constants[57]), constants[58]))
    rates[19] = -(constants[49]+constants[56])*states[19]+algebraic[46]*states[18]+constants[60]*states[20]
    algebraic[47] = 1.00000-(states[18]+states[19]+states[20]+states[21]+states[22])
    rates[18] = (constants[49]*states[19]-algebraic[46]*states[18])+constants[60]*algebraic[47]
    rates[20] = -(constants[49]+constants[59]+constants[60])*states[20]+algebraic[46]*algebraic[47]+constants[56]*states[19]+constants[61]*states[21]
    algebraic[48] = (constants[54]*(states[20]+algebraic[47]+2.00000*states[21]+3.00000*states[22]))/constants[68]
    rates[17] = constants[46]*(constants[45]-states[17])*states[3]-((constants[47]*(1.00000+2.00000*(1.00000-algebraic[48])))/3.00000)*states[17]
    algebraic[45] = constants[43]*(constants[42]-states[12])*states[3]-constants[44]*states[12]
    algebraic[49] = constants[46]*(constants[45]-states[17])*states[3]-((constants[47]*(1.00000+2.00000*(1.00000-algebraic[48])))/3.00000)*states[17]
    rates[3] = (((-((algebraic[25]+algebraic[35])-2.00000*algebraic[30])/(2.00000*constants[4]*constants[2])-algebraic[44])+(algebraic[41]*constants[5])/constants[4])-algebraic[45])-algebraic[49]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[7] = states[12]/5.00000e-05
    algebraic[2] = 20.0000*exp(-0.125000*((states[0]+75.0000)-constants[16]))
    algebraic[10] = 2000.00/(1.00000+320.000*exp(-0.100000*((states[0]+75.0000)-constants[16])))
    algebraic[3] = (0.500000*exp(0.0826000*(states[0]+50.0000)))/(1.00000+exp(0.0570000*(states[0]+50.0000)))
    algebraic[11] = (1.30000*exp(-0.0600000*(states[0]+20.0000)))/(1.00000+exp(-0.0400000*(states[0]+20.0000)))
    algebraic[4] = 0.0330000*exp(-states[0]/17.0000)
    algebraic[12] = 33.0000/(1.00000+exp(-0.125000*(states[0]+10.0000)))
    algebraic[1] = states[0]+41.0000
    algebraic[9] = custom_piecewise([less(fabs(algebraic[1]) , constants[15]), 2000.00 , True, (200.000*algebraic[1])/(1.00000-exp(-0.100000*algebraic[1]))])
    algebraic[16] = 8000.00*exp(-0.0560000*(states[0]+66.0000))
    algebraic[5] = (states[0]+24.0000)-5.00000
    algebraic[13] = custom_piecewise([less(fabs(algebraic[5]) , 1.00000e-05), constants[22]*120.000 , True, (constants[22]*30.0000*algebraic[5])/(1.00000-exp(-algebraic[5]/4.00000))])
    algebraic[17] = custom_piecewise([less(fabs(algebraic[5]) , 1.00000e-05), constants[22]*120.000 , True, (constants[22]*-12.0000*algebraic[5])/(1.00000-exp(algebraic[5]/10.0000))])
    algebraic[6] = states[0]+34.0000
    algebraic[14] = custom_piecewise([less(fabs(algebraic[6]) , 1.00000e-05), constants[23]*25.0000 , True, (constants[23]*6.25000*algebraic[6])/(-1.00000+exp(algebraic[6]/4.00000))])
    algebraic[18] = (constants[23]*50.0000)/(1.00000+exp(-algebraic[6]/4.00000))
    algebraic[27] = ((0.0100000*states[9]*states[10]*constants[21]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]*(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))))*(states[2]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-constants[6]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[26] = ((0.00200000*states[9]*states[10]*constants[21]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]*(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))))*(states[1]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-constants[7]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[25] = ((4.00000*states[9]*states[10]*constants[21]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]*(1.00000-exp((-2.00000*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))))*(states[3]*exp((100.000*constants[2])/(constants[0]*constants[1]))-constants[8]*exp((-2.00000*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[28] = algebraic[25]+algebraic[26]+algebraic[27]
    algebraic[31] = constants[35]*(3.06250e+07*(power(states[3], 2.00000))-245.000*algebraic[28])
    algebraic[33] = (constants[36]*450.000)/(1.00000+0.360000/states[13])
    algebraic[29] = power(states[11]/0.700000, 2.00000)
    algebraic[34] = constants[37]*1.88500*(power(states[14]/0.220000, algebraic[29]))
    algebraic[38] = 1.00000-(states[15]+states[16])
    algebraic[15] = ((constants[0]*constants[1])/constants[2])*log(constants[7]/states[1])
    algebraic[23] = (((constants[19]*constants[7])/(constants[7]+constants[18]))*(states[0]-algebraic[15]))/(1.00000+exp((2.00000*constants[2]*((states[0]-algebraic[15])-10.0000))/(constants[0]*constants[1])))
    algebraic[24] = constants[20]*states[7]*states[8]*(states[0]-algebraic[15])
    algebraic[22] = (constants[17]*states[6]*(states[1]-constants[7]*exp((-states[0]*constants[2])/(constants[0]*constants[1]))))/140.000
    algebraic[32] = (((((constants[26]*constants[7])/(constants[27]+constants[7]))*states[2])/(constants[28]+states[2]))*1.00000)/(1.00000+0.124500*exp((-0.100000*states[0]*constants[2])/(constants[0]*constants[1]))+0.0353000*exp((-states[0]*constants[2])/(constants[0]*constants[1])))
    algebraic[36] = constants[30]*(states[0]-algebraic[15])
    algebraic[20] = ((constants[0]*constants[1])/constants[2])*log((constants[6]+0.120000*constants[7])/(states[2]+0.120000*states[1]))
    algebraic[21] = constants[14]*(power(states[4], 3.00000))*states[5]*(states[0]-algebraic[20])
    algebraic[30] = (constants[24]*(exp((constants[25]*states[0]*constants[2])/(constants[0]*constants[1]))*(power(states[2], 3.00000))*constants[8]-exp(((constants[25]-1.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(constants[6], 3.00000))*states[3]))/(1.00000+states[3]/0.00690000)
    algebraic[8] = ((constants[0]*constants[1])/constants[2])*log(constants[6]/states[2])
    algebraic[39] = constants[31]*(states[0]-algebraic[8])
    algebraic[19] = ((0.500000*constants[0]*constants[1])/constants[2])*log(constants[8]/states[3])
    algebraic[35] = constants[29]*(states[0]-algebraic[19])
    algebraic[0] = custom_piecewise([greater_equal(voi , constants[9]) & less_equal(voi , constants[10]) & less_equal((voi-constants[9])-floor((voi-constants[9])/constants[11])*constants[11] , constants[12]), constants[13] , True, 0.00000])
    algebraic[40] = (constants[33]*states[14])/(states[14]+0.200000)
    algebraic[37] = power(states[16]/(states[16]+0.250000), 2.00000)
    algebraic[41] = (algebraic[40]*algebraic[37]+constants[39])*(states[13]-states[3])
    algebraic[42] = power(states[3]/0.000240000, 2.00000)
    algebraic[43] = power(states[13]/1.64000, 2.00000)
    algebraic[44] = (states[11]*constants[40]*algebraic[42]-constants[41]*algebraic[43])/(1.00000+algebraic[42]+algebraic[43])
    algebraic[46] = constants[49]*(power(states[17]/(constants[45]*constants[57]), constants[58]))
    algebraic[47] = 1.00000-(states[18]+states[19]+states[20]+states[21]+states[22])
    algebraic[48] = (constants[54]*(states[20]+algebraic[47]+2.00000*states[21]+3.00000*states[22]))/constants[68]
    algebraic[45] = constants[43]*(constants[42]-states[12])*states[3]-constants[44]*states[12]
    algebraic[49] = constants[46]*(constants[45]-states[17])*states[3]-((constants[47]*(1.00000+2.00000*(1.00000-algebraic[48])))/3.00000)*states[17]
    algebraic[50] = constants[48]*algebraic[48]
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