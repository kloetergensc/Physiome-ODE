# Size of variable arrays:
sizeAlgebraic = 52
sizeStates = 21
sizeConstants = 70
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_states[0] = "V in component membrane (millivolt)"
    legend_constants[0] = "R in component membrane (millijoule_per_mole_kelvin)"
    legend_constants[1] = "T in component membrane (kelvin)"
    legend_constants[2] = "F in component membrane (coulomb_per_mole)"
    legend_constants[3] = "Cm in component membrane (microF)"
    legend_algebraic[26] = "i_K1 in component time_independent_potassium_current (nanoA)"
    legend_algebraic[43] = "i_to in component transient_outward_current (nanoA)"
    legend_algebraic[30] = "i_K in component time_dependent_potassium_current (nanoA)"
    legend_algebraic[36] = "i_Ca_L_K_cyt in component L_type_Ca_channel (nanoA)"
    legend_algebraic[39] = "i_Ca_L_K_ds in component L_type_Ca_channel (nanoA)"
    legend_algebraic[44] = "i_NaK in component sodium_potassium_pump (nanoA)"
    legend_algebraic[32] = "i_Na in component fast_sodium_current (nanoA)"
    legend_algebraic[33] = "i_b_Na in component sodium_background_current (nanoA)"
    legend_algebraic[37] = "i_Ca_L_Na_cyt in component L_type_Ca_channel (nanoA)"
    legend_algebraic[40] = "i_Ca_L_Na_ds in component L_type_Ca_channel (nanoA)"
    legend_algebraic[45] = "i_NaCa_cyt in component sodium_calcium_exchanger (nanoA)"
    legend_algebraic[46] = "i_NaCa_ds in component sodium_calcium_exchanger (nanoA)"
    legend_algebraic[35] = "i_Ca_L_Ca_cyt in component L_type_Ca_channel (nanoA)"
    legend_algebraic[38] = "i_Ca_L_Ca_ds in component L_type_Ca_channel (nanoA)"
    legend_algebraic[42] = "i_b_Ca in component calcium_background_current (nanoA)"
    legend_algebraic[34] = "i_b_K in component potassium_background_current (nanoA)"
    legend_algebraic[6] = "i_Stim in component membrane (nanoA)"
    legend_constants[4] = "stim_start in component membrane (second)"
    legend_constants[5] = "stim_end in component membrane (second)"
    legend_constants[6] = "stim_period in component membrane (second)"
    legend_constants[7] = "stim_duration in component membrane (second)"
    legend_constants[8] = "stim_amplitude in component membrane (nanoA)"
    legend_algebraic[14] = "E_Na in component reversal_potentials (millivolt)"
    legend_algebraic[20] = "E_K in component reversal_potentials (millivolt)"
    legend_algebraic[22] = "E_Ca in component reversal_potentials (millivolt)"
    legend_algebraic[24] = "E_mh in component reversal_potentials (millivolt)"
    legend_states[1] = "K_o in component extracellular_potassium_concentration (millimolar)"
    legend_constants[9] = "Na_o in component extracellular_sodium_concentration (millimolar)"
    legend_states[2] = "K_i in component intracellular_potassium_concentration (millimolar)"
    legend_states[3] = "Na_i in component intracellular_sodium_concentration (millimolar)"
    legend_constants[10] = "Ca_o in component extracellular_calcium_concentration (millimolar)"
    legend_states[4] = "Ca_i in component intracellular_calcium_concentration (millimolar)"
    legend_constants[11] = "K_mk1 in component time_independent_potassium_current (millimolar)"
    legend_constants[12] = "g_K1 in component time_independent_potassium_current (microS)"
    legend_algebraic[28] = "I_K in component time_dependent_potassium_current (nanoA)"
    legend_constants[13] = "i_K_max in component time_dependent_potassium_current (nanoA)"
    legend_states[5] = "x in component time_dependent_potassium_current_x_gate (dimensionless)"
    legend_algebraic[8] = "alpha_x in component time_dependent_potassium_current_x_gate (per_second)"
    legend_algebraic[16] = "beta_x in component time_dependent_potassium_current_x_gate (per_second)"
    legend_constants[14] = "delta_x in component time_dependent_potassium_current_x_gate (millivolt)"
    legend_algebraic[0] = "E0_x in component time_dependent_potassium_current_x_gate (millivolt)"
    legend_constants[15] = "g_Na in component fast_sodium_current (microS)"
    legend_states[6] = "m in component fast_sodium_current_m_gate (dimensionless)"
    legend_states[7] = "h in component fast_sodium_current_h_gate (dimensionless)"
    legend_algebraic[9] = "alpha_m in component fast_sodium_current_m_gate (per_second)"
    legend_algebraic[17] = "beta_m in component fast_sodium_current_m_gate (per_second)"
    legend_constants[16] = "delta_m in component fast_sodium_current_m_gate (millivolt)"
    legend_algebraic[1] = "E0_m in component fast_sodium_current_m_gate (millivolt)"
    legend_algebraic[2] = "alpha_h in component fast_sodium_current_h_gate (per_second)"
    legend_algebraic[10] = "beta_h in component fast_sodium_current_h_gate (per_second)"
    legend_constants[17] = "shift_h in component fast_sodium_current_h_gate (millivolt)"
    legend_constants[18] = "g_bna in component sodium_background_current (microS)"
    legend_constants[19] = "g_bk in component potassium_background_current (microS)"
    legend_algebraic[41] = "i_Ca_L in component L_type_Ca_channel (nanoA)"
    legend_constants[20] = "P_Ca_L in component L_type_Ca_channel (nanoA_per_millimolar)"
    legend_constants[21] = "P_CaK in component L_type_Ca_channel (dimensionless)"
    legend_constants[22] = "P_CaNa in component L_type_Ca_channel (dimensionless)"
    legend_states[8] = "Ca_ds in component intracellular_calcium_concentration (millimolar)"
    legend_states[9] = "d in component L_type_Ca_channel_d_gate (dimensionless)"
    legend_states[10] = "f in component L_type_Ca_channel_f_gate (dimensionless)"
    legend_states[11] = "f2 in component L_type_Ca_channel_f2_gate (dimensionless)"
    legend_states[12] = "f2ds in component L_type_Ca_channel_f2ds_gate (dimensionless)"
    legend_constants[23] = "Km_f2 in component L_type_Ca_channel (millimolar)"
    legend_constants[24] = "Km_f2ds in component L_type_Ca_channel (millimolar)"
    legend_constants[25] = "R_decay in component L_type_Ca_channel (per_second)"
    legend_constants[26] = "FrICa in component L_type_Ca_channel (dimensionless)"
    legend_algebraic[11] = "alpha_d in component L_type_Ca_channel_d_gate (per_second)"
    legend_algebraic[18] = "beta_d in component L_type_Ca_channel_d_gate (per_second)"
    legend_algebraic[3] = "E0_d in component L_type_Ca_channel_d_gate (millivolt)"
    legend_constants[27] = "speed_d in component L_type_Ca_channel_d_gate (dimensionless)"
    legend_algebraic[12] = "alpha_f in component L_type_Ca_channel_f_gate (per_second)"
    legend_algebraic[19] = "beta_f in component L_type_Ca_channel_f_gate (per_second)"
    legend_constants[28] = "speed_f in component L_type_Ca_channel_f_gate (dimensionless)"
    legend_constants[29] = "delta_f in component L_type_Ca_channel_f_gate (millivolt)"
    legend_algebraic[4] = "E0_f in component L_type_Ca_channel_f_gate (millivolt)"
    legend_constants[30] = "g_bca in component calcium_background_current (microS)"
    legend_constants[31] = "g_to in component transient_outward_current (microS)"
    legend_constants[32] = "g_tos in component transient_outward_current (dimensionless)"
    legend_states[13] = "s in component transient_outward_current_s_gate (dimensionless)"
    legend_states[14] = "r in component transient_outward_current_r_gate (dimensionless)"
    legend_algebraic[5] = "alpha_s in component transient_outward_current_s_gate (per_second)"
    legend_algebraic[13] = "beta_s in component transient_outward_current_s_gate (per_second)"
    legend_constants[33] = "i_NaK_max in component sodium_potassium_pump (nanoA)"
    legend_constants[34] = "K_mK in component sodium_potassium_pump (millimolar)"
    legend_constants[35] = "K_mNa in component sodium_potassium_pump (millimolar)"
    legend_algebraic[48] = "i_NaCa in component sodium_calcium_exchanger (nanoA)"
    legend_constants[36] = "k_NaCa in component sodium_calcium_exchanger (nanoA)"
    legend_constants[37] = "n_NaCa in component sodium_calcium_exchanger (dimensionless)"
    legend_constants[38] = "d_NaCa in component sodium_calcium_exchanger (dimensionless)"
    legend_constants[39] = "gamma in component sodium_calcium_exchanger (dimensionless)"
    legend_constants[40] = "FRiNaCa in component sodium_calcium_exchanger (dimensionless)"
    legend_algebraic[49] = "i_up in component sarcoplasmic_reticulum_calcium_pump (millimolar_per_second)"
    legend_constants[66] = "K_1 in component sarcoplasmic_reticulum_calcium_pump (dimensionless)"
    legend_algebraic[47] = "K_2 in component sarcoplasmic_reticulum_calcium_pump (millimolar)"
    legend_constants[41] = "K_cyca in component sarcoplasmic_reticulum_calcium_pump (millimolar)"
    legend_constants[42] = "K_xcs in component sarcoplasmic_reticulum_calcium_pump (dimensionless)"
    legend_constants[43] = "K_srca in component sarcoplasmic_reticulum_calcium_pump (millimolar)"
    legend_constants[44] = "alpha_up in component sarcoplasmic_reticulum_calcium_pump (millimolar_per_second)"
    legend_constants[45] = "beta_up in component sarcoplasmic_reticulum_calcium_pump (millimolar_per_second)"
    legend_states[15] = "Ca_up in component intracellular_calcium_concentration (millimolar)"
    legend_algebraic[50] = "i_trans in component calcium_translocation (millimolar_per_second)"
    legend_states[16] = "Ca_rel in component intracellular_calcium_concentration (millimolar)"
    legend_algebraic[51] = "i_rel in component calcium_release (millimolar_per_second)"
    legend_algebraic[7] = "VoltDep in component calcium_release (dimensionless)"
    legend_algebraic[23] = "RegBindSite in component calcium_release (dimensionless)"
    legend_algebraic[15] = "CaiReg in component calcium_release (dimensionless)"
    legend_algebraic[21] = "CadsReg in component calcium_release (dimensionless)"
    legend_algebraic[25] = "ActRate in component calcium_release (per_second)"
    legend_algebraic[27] = "InactRate in component calcium_release (per_second)"
    legend_constants[46] = "K_leak_rate in component calcium_release (per_second)"
    legend_constants[47] = "K_m_rel in component calcium_release (per_second)"
    legend_constants[48] = "K_m_Ca_cyt in component calcium_release (millimolar)"
    legend_constants[49] = "K_m_Ca_ds in component calcium_release (millimolar)"
    legend_algebraic[31] = "PrecFrac in component calcium_release (dimensionless)"
    legend_states[17] = "ActFrac in component calcium_release (dimensionless)"
    legend_states[18] = "ProdFrac in component calcium_release (dimensionless)"
    legend_algebraic[29] = "SpeedRel in component calcium_release (dimensionless)"
    legend_constants[69] = "V_i in component intracellular_calcium_concentration (micrometre3)"
    legend_constants[50] = "K_b in component extracellular_potassium_concentration (millimolar)"
    legend_constants[51] = "pf in component extracellular_potassium_concentration (per_second)"
    legend_constants[68] = "V_e in component intracellular_calcium_concentration (micrometre3)"
    legend_states[19] = "Ca_Calmod in component intracellular_calcium_concentration (millimolar)"
    legend_states[20] = "Ca_Trop in component intracellular_calcium_concentration (millimolar)"
    legend_constants[52] = "Calmod in component intracellular_calcium_concentration (millimolar)"
    legend_constants[53] = "Trop in component intracellular_calcium_concentration (millimolar)"
    legend_constants[54] = "alpha_Calmod in component intracellular_calcium_concentration (per_millimolar_second)"
    legend_constants[55] = "beta_Calmod in component intracellular_calcium_concentration (per_second)"
    legend_constants[56] = "alpha_Trop in component intracellular_calcium_concentration (per_millimolar_second)"
    legend_constants[57] = "beta_Trop in component intracellular_calcium_concentration (per_second)"
    legend_constants[58] = "radius in component intracellular_calcium_concentration (micrometre)"
    legend_constants[59] = "length in component intracellular_calcium_concentration (micrometre)"
    legend_constants[65] = "V_Cell in component intracellular_calcium_concentration (micrometre3)"
    legend_constants[67] = "V_i_ratio in component intracellular_calcium_concentration (dimensionless)"
    legend_constants[60] = "V_ds_ratio in component intracellular_calcium_concentration (dimensionless)"
    legend_constants[61] = "V_rel_ratio in component intracellular_calcium_concentration (dimensionless)"
    legend_constants[62] = "V_e_ratio in component intracellular_calcium_concentration (dimensionless)"
    legend_constants[63] = "V_up_ratio in component intracellular_calcium_concentration (dimensionless)"
    legend_constants[64] = "Kdecay in component intracellular_calcium_concentration (per_second)"
    legend_rates[0] = "d/dt V in component membrane (millivolt)"
    legend_rates[5] = "d/dt x in component time_dependent_potassium_current_x_gate (dimensionless)"
    legend_rates[6] = "d/dt m in component fast_sodium_current_m_gate (dimensionless)"
    legend_rates[7] = "d/dt h in component fast_sodium_current_h_gate (dimensionless)"
    legend_rates[9] = "d/dt d in component L_type_Ca_channel_d_gate (dimensionless)"
    legend_rates[10] = "d/dt f in component L_type_Ca_channel_f_gate (dimensionless)"
    legend_rates[11] = "d/dt f2 in component L_type_Ca_channel_f2_gate (dimensionless)"
    legend_rates[12] = "d/dt f2ds in component L_type_Ca_channel_f2ds_gate (dimensionless)"
    legend_rates[13] = "d/dt s in component transient_outward_current_s_gate (dimensionless)"
    legend_rates[14] = "d/dt r in component transient_outward_current_r_gate (dimensionless)"
    legend_rates[17] = "d/dt ActFrac in component calcium_release (dimensionless)"
    legend_rates[18] = "d/dt ProdFrac in component calcium_release (dimensionless)"
    legend_rates[3] = "d/dt Na_i in component intracellular_sodium_concentration (millimolar)"
    legend_rates[1] = "d/dt K_o in component extracellular_potassium_concentration (millimolar)"
    legend_rates[2] = "d/dt K_i in component intracellular_potassium_concentration (millimolar)"
    legend_rates[4] = "d/dt Ca_i in component intracellular_calcium_concentration (millimolar)"
    legend_rates[19] = "d/dt Ca_Calmod in component intracellular_calcium_concentration (millimolar)"
    legend_rates[20] = "d/dt Ca_Trop in component intracellular_calcium_concentration (millimolar)"
    legend_rates[8] = "d/dt Ca_ds in component intracellular_calcium_concentration (millimolar)"
    legend_rates[15] = "d/dt Ca_up in component intracellular_calcium_concentration (millimolar)"
    legend_rates[16] = "d/dt Ca_rel in component intracellular_calcium_concentration (millimolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = -85.35765
    constants[0] = 8314.472
    constants[1] = 310
    constants[2] = 96485.3415
    constants[3] = 9.5e-5
    constants[4] = 0.02
    constants[5] = 9
    constants[6] = 0.5
    constants[7] = 0.002
    constants[8] = -1.8
    states[1] = 5.3367
    constants[9] = 148.5
    states[2] = 140.096
    states[3] = 6.9366
    constants[10] = 2.5
    states[4] = 3.792e-5
    constants[11] = 10
    constants[12] = 0.12
    constants[13] = 1.5
    states[5] = 3.5095e-6
    constants[14] = 0.0001
    constants[15] = 0.6
    states[6] = 0.00448
    states[7] = 0.9704
    constants[16] = 1e-5
    constants[17] = 0
    constants[18] = 0.0001
    constants[19] = 0.004
    constants[20] = 0.045
    constants[21] = 0.003
    constants[22] = 0.01
    states[8] = 0.00077
    states[9] = 4.171e-12
    states[10] = 0.999997
    states[11] = 0.99279
    states[12] = 0.459
    constants[23] = 100000
    constants[24] = 0.001
    constants[25] = 20
    constants[26] = 1
    constants[27] = 10
    constants[28] = 2
    constants[29] = 0.0001
    constants[30] = 0.00025
    constants[31] = 0.048
    constants[32] = 0.15
    states[13] = 0.9379
    states[14] = 2.6578e-5
    constants[33] = 0.7
    constants[34] = 1
    constants[35] = 40
    constants[36] = 0.0002
    constants[37] = 3
    constants[38] = 0.001
    constants[39] = 0.5
    constants[40] = 0.001
    constants[41] = 0.0003
    constants[42] = 0.4
    constants[43] = 0.5
    constants[44] = 0.4
    constants[45] = 0.03
    states[15] = 0.3342
    states[16] = 0.31007
    constants[46] = 0.05
    constants[47] = 250
    constants[48] = 0.0005
    constants[49] = 0.01
    states[17] = 0.0112
    states[18] = 0.9059
    constants[50] = 5.4
    constants[51] = 0.7
    states[19] = 0.001419
    states[20] = 0.000932
    constants[52] = 0.02
    constants[53] = 0.05
    constants[54] = 100000
    constants[55] = 50
    constants[56] = 100000
    constants[57] = 200
    constants[58] = 12
    constants[59] = 74
    constants[60] = 0.1
    constants[61] = 0.1
    constants[62] = 0.4
    constants[63] = 0.01
    constants[64] = 10
    constants[65] = (3.14159*(power(constants[58]/1000.00, 2.00000))*constants[59])/1000.00
    constants[66] = (constants[41]*constants[42])/constants[43]
    constants[67] = ((1.00000-constants[62])-constants[63])-constants[61]
    constants[68] = constants[65]*constants[62]
    constants[69] = constants[65]*constants[67]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[11] = 1.00000-1.00000*(states[4]/(constants[23]+states[4])+states[11])
    rates[12] = constants[25]*(1.00000-(states[8]/(constants[24]+states[8])+states[12]))
    rates[14] = 333.000*(1.00000/(1.00000+exp(-((states[0]+4.00000)-24.0000)/10.0000))-states[14])
    algebraic[2] = 20.0000*exp(-0.125000*((states[0]+75.0000)-constants[17]))
    algebraic[10] = 2000.00/(1.00000+320.000*exp(-0.100000*((states[0]+75.0000)-constants[17])))
    rates[7] = algebraic[2]*(1.00000-states[7])-algebraic[10]*states[7]
    algebraic[5] = 0.260000*0.0330000*exp(-states[0]/14.8750)
    algebraic[13] = (0.260000*33.0000)/(1.00000+exp(-(states[0]+10.0000)/7.00000))
    rates[13] = algebraic[5]*(1.00000-states[13])-algebraic[13]*states[13]
    algebraic[0] = (states[0]+22.0000)-40.0000
    algebraic[8] = custom_piecewise([less(fabs(algebraic[0]) , constants[14]), 2.50000 , True, (3.00000*0.500000*algebraic[0])/(1.00000-exp(-algebraic[0]/5.00000))])
    algebraic[16] = custom_piecewise([less(fabs(algebraic[0]) , constants[14]), 2.50000 , True, (3.00000*0.178000*algebraic[0])/(exp(algebraic[0]/15.0000)-1.00000)])
    rates[5] = algebraic[8]*(1.00000-states[5])-algebraic[16]*states[5]
    algebraic[1] = states[0]+41.0000
    algebraic[9] = custom_piecewise([less(fabs(algebraic[1]) , constants[16]), 2000.00 , True, (200.000*algebraic[1])/(1.00000-exp(-0.100000*algebraic[1]))])
    algebraic[17] = 8000.00*exp(-0.0560000*(states[0]+66.0000))
    rates[6] = algebraic[9]*(1.00000-states[6])-algebraic[17]*states[6]
    algebraic[3] = (states[0]+24.0000)-20.0000
    algebraic[11] = custom_piecewise([less(fabs(algebraic[3]) , 0.000100000), 120.000 , True, (30.0000*algebraic[3])/(1.00000-exp(-algebraic[3]/3.00000))])
    algebraic[18] = custom_piecewise([less(fabs(algebraic[3]) , 0.000100000), 120.000 , True, (12.0000*algebraic[3])/(exp(algebraic[3]/7.50000)-1.00000)])
    rates[9] = constants[27]*(algebraic[11]*(1.00000-states[9])-algebraic[18]*states[9])
    algebraic[4] = (states[0]+34.0000)-10.0000
    algebraic[12] = custom_piecewise([less(fabs(algebraic[4]) , constants[29]), 25.0000 , True, (6.25000*algebraic[4])/(exp(algebraic[4]/5.50000)-1.00000)])
    algebraic[19] = 12.0000/(1.00000+exp((-1.00000*(states[0]+34.0000))/5.50000))
    rates[10] = constants[28]*(algebraic[12]*(1.00000-states[10])-algebraic[19]*states[10])
    algebraic[15] = states[4]/(states[4]+constants[48])
    algebraic[21] = states[8]/(states[8]+constants[49])
    algebraic[23] = algebraic[15]+(1.00000-algebraic[15])*algebraic[21]
    algebraic[27] = 60.0000+500.000*(power(algebraic[23], 2.00000))
    algebraic[29] = custom_piecewise([less(states[0] , -50.0000), 5.00000 , True, 1.00000])
    rates[18] = states[17]*algebraic[29]*algebraic[27]-algebraic[29]*1.00000*states[18]
    algebraic[7] = exp(0.0800000*(states[0]-40.0000))
    algebraic[25] = 0.00000*algebraic[7]+500.000*(power(algebraic[23], 2.00000))
    algebraic[31] = (1.00000-states[17])-states[18]
    rates[17] = algebraic[31]*algebraic[29]*algebraic[25]-states[17]*algebraic[29]*algebraic[27]
    algebraic[38] = (((constants[26]*4.00000*constants[20]*states[9]*states[10]*states[12]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2]*2.00000)/(constants[0]*constants[1]))))*(states[4]*exp((100.000*constants[2])/(constants[0]*constants[1]))-constants[10]*exp((-(states[0]-50.0000)*constants[2]*2.00000)/(constants[0]*constants[1])))
    rates[8] = (-1.00000*algebraic[38])/(2.00000*constants[60]*1.00000*constants[69]*constants[2])-states[8]*constants[64]
    algebraic[20] = ((constants[0]*constants[1])/constants[2])*log(states[1]/states[2])
    algebraic[26] = (((constants[12]*states[1])/(states[1]+constants[11]))*(states[0]-algebraic[20]))/(1.00000+exp((((states[0]-algebraic[20])+10.0000)*constants[2]*1.67000)/(constants[0]*constants[1])))
    algebraic[43] = constants[31]*(constants[32]+states[13]*(1.00000-constants[32]))*states[14]*(states[0]-algebraic[20])
    algebraic[28] = (constants[13]*(states[2]-states[1]*exp((-states[0]*constants[2])/(constants[0]*constants[1]))))/140.000
    algebraic[30] = states[5]*algebraic[28]
    algebraic[36] = ((((1.00000-constants[26])*constants[21]*constants[20]*states[9]*states[10]*states[11]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))))*(states[2]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-states[1]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[39] = (((constants[26]*constants[21]*constants[20]*states[9]*states[10]*states[12]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))))*(states[2]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-states[1]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[44] = (((constants[33]*states[1])/(constants[34]+states[1]))*states[3])/(constants[35]+states[3])
    algebraic[34] = constants[19]*(states[0]-algebraic[20])
    rates[1] = (1.00000*(algebraic[26]+algebraic[43]+algebraic[36]+algebraic[39]+-2.00000*algebraic[44]+algebraic[30]+algebraic[34]))/(1.00000*constants[68]*constants[2])-constants[51]*(states[1]-constants[50])
    rates[2] = (-1.00000/(1.00000*constants[69]*constants[2]))*((algebraic[26]+algebraic[30]+algebraic[36]+algebraic[39]+algebraic[43]+algebraic[34])-2.00000*algebraic[44])
    algebraic[24] = ((constants[0]*constants[1])/constants[2])*log((constants[9]+0.120000*states[1])/(states[3]+0.120000*states[2]))
    algebraic[32] = constants[15]*(power(states[6], 3.00000))*states[7]*(states[0]-algebraic[24])
    algebraic[14] = ((constants[0]*constants[1])/constants[2])*log(constants[9]/states[3])
    algebraic[33] = constants[18]*(states[0]-algebraic[14])
    algebraic[37] = ((((1.00000-constants[26])*constants[22]*constants[20]*states[9]*states[10]*states[11]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))))*(states[3]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-constants[9]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[40] = (((constants[26]*constants[22]*constants[20]*states[9]*states[10]*states[12]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))))*(states[3]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-constants[9]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[45] = ((1.00000-constants[40])*constants[36]*(exp((constants[39]*(constants[37]-2.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(states[3], constants[37]))*constants[10]-exp(((constants[39]-1.00000)*(constants[37]-2.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(constants[9], constants[37]))*states[4]))/((1.00000+constants[38]*(states[4]*(power(constants[9], constants[37]))+constants[10]*(power(states[3], constants[37]))))*(1.00000+states[4]/0.00690000))
    rates[3] = (-1.00000/(1.00000*constants[69]*constants[2]))*(algebraic[32]+algebraic[33]+3.00000*algebraic[44]+3.00000*algebraic[45]+algebraic[37]+algebraic[40])
    algebraic[46] = (constants[40]*constants[36]*(exp((constants[39]*(constants[37]-2.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(states[3], constants[37]))*constants[10]-exp(((constants[39]-1.00000)*(constants[37]-2.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(constants[9], constants[37]))*states[8]))/((1.00000+constants[38]*(states[8]*(power(constants[9], constants[37]))+constants[10]*(power(states[3], constants[37]))))*(1.00000+states[8]/0.00690000))
    algebraic[35] = ((((1.00000-constants[26])*4.00000*constants[20]*states[9]*states[10]*states[11]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2]*2.00000)/(constants[0]*constants[1]))))*(states[4]*exp((100.000*constants[2])/(constants[0]*constants[1]))-constants[10]*exp((-(states[0]-50.0000)*constants[2]*2.00000)/(constants[0]*constants[1])))
    algebraic[22] = ((0.500000*constants[0]*constants[1])/constants[2])*log(constants[10]/states[4])
    algebraic[42] = constants[30]*(states[0]-algebraic[22])
    algebraic[6] = custom_piecewise([greater_equal(voi , constants[4]) & less_equal(voi , constants[5]) & less_equal((voi-constants[4])-floor((voi-constants[4])/constants[6])*constants[6] , constants[7]), constants[8] , True, 0.00000])
    rates[0] = (-1.00000/constants[3])*(algebraic[6]+algebraic[26]+algebraic[43]+algebraic[30]+algebraic[44]+algebraic[32]+algebraic[33]+algebraic[37]+algebraic[40]+algebraic[45]+algebraic[46]+algebraic[35]+algebraic[38]+algebraic[36]+algebraic[39]+algebraic[42]+algebraic[34])
    algebraic[47] = states[4]+states[15]*constants[66]+constants[41]*constants[42]+constants[41]
    algebraic[49] = (states[4]/algebraic[47])*constants[44]-((states[15]*constants[66])/algebraic[47])*constants[45]
    algebraic[50] = 50.0000*(states[15]-states[16])
    rates[15] = (constants[67]/constants[63])*algebraic[49]-algebraic[50]
    rates[19] = constants[54]*states[4]*(constants[52]-states[19])-constants[55]*states[19]
    algebraic[51] = ((power(states[17]/(states[17]+0.250000), 2.00000))*constants[47]+constants[46])*states[16]
    rates[16] = (constants[63]/constants[61])*algebraic[50]-algebraic[51]
    rates[20] = constants[56]*states[4]*(constants[53]-states[20])-constants[57]*states[20]
    rates[4] = ((((-1.00000/(2.00000*1.00000*constants[69]*constants[2]))*((algebraic[35]+algebraic[42])-2.00000*algebraic[45])+states[8]*constants[60]*constants[64]+(algebraic[51]*constants[61])/constants[67])-rates[19])-rates[20])-algebraic[49]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[2] = 20.0000*exp(-0.125000*((states[0]+75.0000)-constants[17]))
    algebraic[10] = 2000.00/(1.00000+320.000*exp(-0.100000*((states[0]+75.0000)-constants[17])))
    algebraic[5] = 0.260000*0.0330000*exp(-states[0]/14.8750)
    algebraic[13] = (0.260000*33.0000)/(1.00000+exp(-(states[0]+10.0000)/7.00000))
    algebraic[0] = (states[0]+22.0000)-40.0000
    algebraic[8] = custom_piecewise([less(fabs(algebraic[0]) , constants[14]), 2.50000 , True, (3.00000*0.500000*algebraic[0])/(1.00000-exp(-algebraic[0]/5.00000))])
    algebraic[16] = custom_piecewise([less(fabs(algebraic[0]) , constants[14]), 2.50000 , True, (3.00000*0.178000*algebraic[0])/(exp(algebraic[0]/15.0000)-1.00000)])
    algebraic[1] = states[0]+41.0000
    algebraic[9] = custom_piecewise([less(fabs(algebraic[1]) , constants[16]), 2000.00 , True, (200.000*algebraic[1])/(1.00000-exp(-0.100000*algebraic[1]))])
    algebraic[17] = 8000.00*exp(-0.0560000*(states[0]+66.0000))
    algebraic[3] = (states[0]+24.0000)-20.0000
    algebraic[11] = custom_piecewise([less(fabs(algebraic[3]) , 0.000100000), 120.000 , True, (30.0000*algebraic[3])/(1.00000-exp(-algebraic[3]/3.00000))])
    algebraic[18] = custom_piecewise([less(fabs(algebraic[3]) , 0.000100000), 120.000 , True, (12.0000*algebraic[3])/(exp(algebraic[3]/7.50000)-1.00000)])
    algebraic[4] = (states[0]+34.0000)-10.0000
    algebraic[12] = custom_piecewise([less(fabs(algebraic[4]) , constants[29]), 25.0000 , True, (6.25000*algebraic[4])/(exp(algebraic[4]/5.50000)-1.00000)])
    algebraic[19] = 12.0000/(1.00000+exp((-1.00000*(states[0]+34.0000))/5.50000))
    algebraic[15] = states[4]/(states[4]+constants[48])
    algebraic[21] = states[8]/(states[8]+constants[49])
    algebraic[23] = algebraic[15]+(1.00000-algebraic[15])*algebraic[21]
    algebraic[27] = 60.0000+500.000*(power(algebraic[23], 2.00000))
    algebraic[29] = custom_piecewise([less(states[0] , -50.0000), 5.00000 , True, 1.00000])
    algebraic[7] = exp(0.0800000*(states[0]-40.0000))
    algebraic[25] = 0.00000*algebraic[7]+500.000*(power(algebraic[23], 2.00000))
    algebraic[31] = (1.00000-states[17])-states[18]
    algebraic[38] = (((constants[26]*4.00000*constants[20]*states[9]*states[10]*states[12]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2]*2.00000)/(constants[0]*constants[1]))))*(states[4]*exp((100.000*constants[2])/(constants[0]*constants[1]))-constants[10]*exp((-(states[0]-50.0000)*constants[2]*2.00000)/(constants[0]*constants[1])))
    algebraic[20] = ((constants[0]*constants[1])/constants[2])*log(states[1]/states[2])
    algebraic[26] = (((constants[12]*states[1])/(states[1]+constants[11]))*(states[0]-algebraic[20]))/(1.00000+exp((((states[0]-algebraic[20])+10.0000)*constants[2]*1.67000)/(constants[0]*constants[1])))
    algebraic[43] = constants[31]*(constants[32]+states[13]*(1.00000-constants[32]))*states[14]*(states[0]-algebraic[20])
    algebraic[28] = (constants[13]*(states[2]-states[1]*exp((-states[0]*constants[2])/(constants[0]*constants[1]))))/140.000
    algebraic[30] = states[5]*algebraic[28]
    algebraic[36] = ((((1.00000-constants[26])*constants[21]*constants[20]*states[9]*states[10]*states[11]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))))*(states[2]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-states[1]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[39] = (((constants[26]*constants[21]*constants[20]*states[9]*states[10]*states[12]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))))*(states[2]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-states[1]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[44] = (((constants[33]*states[1])/(constants[34]+states[1]))*states[3])/(constants[35]+states[3])
    algebraic[34] = constants[19]*(states[0]-algebraic[20])
    algebraic[24] = ((constants[0]*constants[1])/constants[2])*log((constants[9]+0.120000*states[1])/(states[3]+0.120000*states[2]))
    algebraic[32] = constants[15]*(power(states[6], 3.00000))*states[7]*(states[0]-algebraic[24])
    algebraic[14] = ((constants[0]*constants[1])/constants[2])*log(constants[9]/states[3])
    algebraic[33] = constants[18]*(states[0]-algebraic[14])
    algebraic[37] = ((((1.00000-constants[26])*constants[22]*constants[20]*states[9]*states[10]*states[11]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))))*(states[3]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-constants[9]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[40] = (((constants[26]*constants[22]*constants[20]*states[9]*states[10]*states[12]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))))*(states[3]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-constants[9]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[45] = ((1.00000-constants[40])*constants[36]*(exp((constants[39]*(constants[37]-2.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(states[3], constants[37]))*constants[10]-exp(((constants[39]-1.00000)*(constants[37]-2.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(constants[9], constants[37]))*states[4]))/((1.00000+constants[38]*(states[4]*(power(constants[9], constants[37]))+constants[10]*(power(states[3], constants[37]))))*(1.00000+states[4]/0.00690000))
    algebraic[46] = (constants[40]*constants[36]*(exp((constants[39]*(constants[37]-2.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(states[3], constants[37]))*constants[10]-exp(((constants[39]-1.00000)*(constants[37]-2.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(constants[9], constants[37]))*states[8]))/((1.00000+constants[38]*(states[8]*(power(constants[9], constants[37]))+constants[10]*(power(states[3], constants[37]))))*(1.00000+states[8]/0.00690000))
    algebraic[35] = ((((1.00000-constants[26])*4.00000*constants[20]*states[9]*states[10]*states[11]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2]*2.00000)/(constants[0]*constants[1]))))*(states[4]*exp((100.000*constants[2])/(constants[0]*constants[1]))-constants[10]*exp((-(states[0]-50.0000)*constants[2]*2.00000)/(constants[0]*constants[1])))
    algebraic[22] = ((0.500000*constants[0]*constants[1])/constants[2])*log(constants[10]/states[4])
    algebraic[42] = constants[30]*(states[0]-algebraic[22])
    algebraic[6] = custom_piecewise([greater_equal(voi , constants[4]) & less_equal(voi , constants[5]) & less_equal((voi-constants[4])-floor((voi-constants[4])/constants[6])*constants[6] , constants[7]), constants[8] , True, 0.00000])
    algebraic[47] = states[4]+states[15]*constants[66]+constants[41]*constants[42]+constants[41]
    algebraic[49] = (states[4]/algebraic[47])*constants[44]-((states[15]*constants[66])/algebraic[47])*constants[45]
    algebraic[50] = 50.0000*(states[15]-states[16])
    algebraic[51] = ((power(states[17]/(states[17]+0.250000), 2.00000))*constants[47]+constants[46])*states[16]
    algebraic[41] = algebraic[35]+algebraic[36]+algebraic[37]+algebraic[38]+algebraic[39]+algebraic[40]
    algebraic[48] = algebraic[45]+algebraic[46]
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