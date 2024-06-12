# Size of variable arrays:
sizeAlgebraic = 56
sizeStates = 22
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
    legend_constants[0] = "R in component membrane (joule_per_kilomole_kelvin)"
    legend_constants[1] = "T in component membrane (kelvin)"
    legend_constants[2] = "F in component membrane (coulomb_per_mole)"
    legend_constants[3] = "Cm in component membrane (microF)"
    legend_algebraic[31] = "i_K1 in component time_independent_potassium_current (nanoA)"
    legend_algebraic[47] = "i_to in component transient_outward_current (nanoA)"
    legend_algebraic[33] = "i_Kr in component rapid_delayed_rectifier_potassium_current (nanoA)"
    legend_algebraic[35] = "i_Ks in component slow_delayed_rectifier_potassium_current (nanoA)"
    legend_algebraic[40] = "i_Ca_L_K_cyt in component L_type_Ca_channel (nanoA)"
    legend_algebraic[43] = "i_Ca_L_K_ds in component L_type_Ca_channel (nanoA)"
    legend_algebraic[48] = "i_NaK in component sodium_potassium_pump (nanoA)"
    legend_algebraic[36] = "i_Na in component fast_sodium_current (nanoA)"
    legend_algebraic[38] = "i_b_Na in component sodium_background_current (nanoA)"
    legend_algebraic[37] = "i_p_Na in component persistent_sodium_current (nanoA)"
    legend_algebraic[41] = "i_Ca_L_Na_cyt in component L_type_Ca_channel (nanoA)"
    legend_algebraic[44] = "i_Ca_L_Na_ds in component L_type_Ca_channel (nanoA)"
    legend_algebraic[49] = "i_NaCa_cyt in component sodium_calcium_exchanger (nanoA)"
    legend_algebraic[50] = "i_NaCa_ds in component sodium_calcium_exchanger (nanoA)"
    legend_algebraic[39] = "i_Ca_L_Ca_cyt in component L_type_Ca_channel (nanoA)"
    legend_algebraic[42] = "i_Ca_L_Ca_ds in component L_type_Ca_channel (nanoA)"
    legend_algebraic[46] = "i_b_Ca in component calcium_background_current (nanoA)"
    legend_algebraic[8] = "i_Stim in component membrane (nanoA)"
    legend_constants[4] = "stim_start in component membrane (second)"
    legend_constants[5] = "stim_end in component membrane (second)"
    legend_constants[6] = "stim_period in component membrane (second)"
    legend_constants[7] = "stim_duration in component membrane (second)"
    legend_constants[8] = "stim_amplitude in component membrane (nanoA)"
    legend_algebraic[18] = "E_Na in component reversal_potentials (millivolt)"
    legend_algebraic[23] = "E_K in component reversal_potentials (millivolt)"
    legend_algebraic[25] = "E_Ks in component reversal_potentials (millivolt)"
    legend_algebraic[27] = "E_Ca in component reversal_potentials (millivolt)"
    legend_algebraic[29] = "E_mh in component reversal_potentials (millivolt)"
    legend_constants[9] = "P_kna in component reversal_potentials (dimensionless)"
    legend_constants[10] = "K_o in component extracellular_potassium_concentration (millimolar)"
    legend_constants[11] = "Na_o in component extracellular_sodium_concentration (millimolar)"
    legend_states[1] = "K_i in component intracellular_potassium_concentration (millimolar)"
    legend_states[2] = "Na_i in component intracellular_sodium_concentration (millimolar)"
    legend_constants[12] = "Ca_o in component extracellular_calcium_concentration (millimolar)"
    legend_states[3] = "Ca_i in component intracellular_calcium_concentration (millimolar)"
    legend_constants[13] = "K_mk1 in component time_independent_potassium_current (millimolar)"
    legend_constants[14] = "g_K1 in component time_independent_potassium_current (microS)"
    legend_constants[15] = "g_Kr1 in component rapid_delayed_rectifier_potassium_current (microS)"
    legend_constants[16] = "g_Kr2 in component rapid_delayed_rectifier_potassium_current (microS)"
    legend_states[4] = "xr1 in component rapid_delayed_rectifier_potassium_current_xr1_gate (dimensionless)"
    legend_states[5] = "xr2 in component rapid_delayed_rectifier_potassium_current_xr2_gate (dimensionless)"
    legend_algebraic[0] = "alpha_xr1 in component rapid_delayed_rectifier_potassium_current_xr1_gate (per_second)"
    legend_algebraic[10] = "beta_xr1 in component rapid_delayed_rectifier_potassium_current_xr1_gate (per_second)"
    legend_algebraic[1] = "alpha_xr2 in component rapid_delayed_rectifier_potassium_current_xr2_gate (per_second)"
    legend_algebraic[11] = "beta_xr2 in component rapid_delayed_rectifier_potassium_current_xr2_gate (per_second)"
    legend_constants[17] = "g_Ks in component slow_delayed_rectifier_potassium_current (microS)"
    legend_states[6] = "xs in component slow_delayed_rectifier_potassium_current_xs_gate (dimensionless)"
    legend_algebraic[2] = "alpha_xs in component slow_delayed_rectifier_potassium_current_xs_gate (per_second)"
    legend_algebraic[12] = "beta_xs in component slow_delayed_rectifier_potassium_current_xs_gate (per_second)"
    legend_constants[18] = "g_Na in component fast_sodium_current (microS)"
    legend_states[7] = "m in component fast_sodium_current_m_gate (dimensionless)"
    legend_states[8] = "h in component fast_sodium_current_h_gate (dimensionless)"
    legend_algebraic[13] = "alpha_m in component fast_sodium_current_m_gate (per_second)"
    legend_algebraic[20] = "beta_m in component fast_sodium_current_m_gate (per_second)"
    legend_constants[19] = "delta_m in component fast_sodium_current_m_gate (millivolt)"
    legend_algebraic[3] = "E0_m in component fast_sodium_current_m_gate (millivolt)"
    legend_algebraic[4] = "alpha_h in component fast_sodium_current_h_gate (per_second)"
    legend_algebraic[14] = "beta_h in component fast_sodium_current_h_gate (per_second)"
    legend_constants[20] = "shift_h in component fast_sodium_current_h_gate (millivolt)"
    legend_constants[21] = "g_pna in component persistent_sodium_current (microS)"
    legend_constants[22] = "g_bna in component sodium_background_current (microS)"
    legend_algebraic[45] = "i_Ca_L in component L_type_Ca_channel (nanoA)"
    legend_constants[23] = "P_Ca_L in component L_type_Ca_channel (nanoA_per_millimolar)"
    legend_constants[24] = "P_CaK in component L_type_Ca_channel (dimensionless)"
    legend_constants[25] = "P_CaNa in component L_type_Ca_channel (dimensionless)"
    legend_states[9] = "Ca_ds in component intracellular_calcium_concentration (millimolar)"
    legend_states[10] = "d in component L_type_Ca_channel_d_gate (dimensionless)"
    legend_states[11] = "f in component L_type_Ca_channel_f_gate (dimensionless)"
    legend_states[12] = "f2 in component L_type_Ca_channel_f2_gate (dimensionless)"
    legend_states[13] = "f2ds in component L_type_Ca_channel_f2ds_gate (dimensionless)"
    legend_constants[26] = "Km_f2 in component L_type_Ca_channel (millimolar)"
    legend_constants[27] = "Km_f2ds in component L_type_Ca_channel (millimolar)"
    legend_constants[28] = "R_decay in component L_type_Ca_channel (per_second)"
    legend_constants[29] = "FrICa in component L_type_Ca_channel (dimensionless)"
    legend_algebraic[15] = "alpha_d in component L_type_Ca_channel_d_gate (per_second)"
    legend_algebraic[21] = "beta_d in component L_type_Ca_channel_d_gate (per_second)"
    legend_algebraic[5] = "E0_d in component L_type_Ca_channel_d_gate (millivolt)"
    legend_constants[30] = "speed_d in component L_type_Ca_channel_d_gate (dimensionless)"
    legend_algebraic[16] = "alpha_f in component L_type_Ca_channel_f_gate (per_second)"
    legend_algebraic[22] = "beta_f in component L_type_Ca_channel_f_gate (per_second)"
    legend_constants[31] = "speed_f in component L_type_Ca_channel_f_gate (dimensionless)"
    legend_constants[32] = "delta_f in component L_type_Ca_channel_f_gate (millivolt)"
    legend_algebraic[6] = "E0_f in component L_type_Ca_channel_f_gate (millivolt)"
    legend_constants[33] = "g_bca in component calcium_background_current (microS)"
    legend_constants[34] = "g_to in component transient_outward_current (microS)"
    legend_constants[35] = "g_tos in component transient_outward_current (dimensionless)"
    legend_states[14] = "s in component transient_outward_current_s_gate (dimensionless)"
    legend_states[15] = "r in component transient_outward_current_r_gate (dimensionless)"
    legend_algebraic[7] = "alpha_s in component transient_outward_current_s_gate (per_second)"
    legend_algebraic[17] = "beta_s in component transient_outward_current_s_gate (per_second)"
    legend_constants[36] = "i_NaK_max in component sodium_potassium_pump (nanoA)"
    legend_constants[37] = "K_mK in component sodium_potassium_pump (millimolar)"
    legend_constants[38] = "K_mNa in component sodium_potassium_pump (millimolar)"
    legend_algebraic[51] = "i_NaCa in component sodium_calcium_exchanger (nanoA)"
    legend_constants[39] = "k_NaCa in component sodium_calcium_exchanger (nanoA)"
    legend_constants[40] = "n_NaCa in component sodium_calcium_exchanger (dimensionless)"
    legend_constants[41] = "d_NaCa in component sodium_calcium_exchanger (dimensionless)"
    legend_constants[42] = "gamma in component sodium_calcium_exchanger (dimensionless)"
    legend_constants[43] = "FRiNaCa in component sodium_calcium_exchanger (dimensionless)"
    legend_algebraic[53] = "i_up in component sarcoplasmic_reticulum_calcium_pump (millimolar_per_second)"
    legend_constants[67] = "K_1 in component sarcoplasmic_reticulum_calcium_pump (dimensionless)"
    legend_algebraic[52] = "K_2 in component sarcoplasmic_reticulum_calcium_pump (millimolar)"
    legend_constants[44] = "K_cyca in component sarcoplasmic_reticulum_calcium_pump (millimolar)"
    legend_constants[45] = "K_xcs in component sarcoplasmic_reticulum_calcium_pump (dimensionless)"
    legend_constants[46] = "K_srca in component sarcoplasmic_reticulum_calcium_pump (millimolar)"
    legend_constants[47] = "alpha_up in component sarcoplasmic_reticulum_calcium_pump (millimolar_per_second)"
    legend_constants[48] = "beta_up in component sarcoplasmic_reticulum_calcium_pump (millimolar_per_second)"
    legend_states[16] = "Ca_up in component intracellular_calcium_concentration (millimolar)"
    legend_algebraic[54] = "i_trans in component calcium_translocation (millimolar_per_second)"
    legend_states[17] = "Ca_rel in component intracellular_calcium_concentration (millimolar)"
    legend_algebraic[55] = "i_rel in component calcium_release (millimolar_per_second)"
    legend_algebraic[9] = "VoltDep in component calcium_release (dimensionless)"
    legend_algebraic[26] = "RegBindSite in component calcium_release (dimensionless)"
    legend_algebraic[19] = "CaiReg in component calcium_release (dimensionless)"
    legend_algebraic[24] = "CadsReg in component calcium_release (dimensionless)"
    legend_algebraic[28] = "ActRate in component calcium_release (per_second)"
    legend_algebraic[30] = "InactRate in component calcium_release (per_second)"
    legend_constants[49] = "K_leak_rate in component calcium_release (per_second)"
    legend_constants[50] = "K_m_rel in component calcium_release (per_second)"
    legend_constants[51] = "K_m_Ca_cyt in component calcium_release (millimolar)"
    legend_constants[52] = "K_m_Ca_ds in component calcium_release (millimolar)"
    legend_algebraic[34] = "PrecFrac in component calcium_release (dimensionless)"
    legend_states[18] = "ActFrac in component calcium_release (dimensionless)"
    legend_states[19] = "ProdFrac in component calcium_release (dimensionless)"
    legend_algebraic[32] = "SpeedRel in component calcium_release (dimensionless)"
    legend_constants[69] = "V_i in component intracellular_calcium_concentration (micrometre3)"
    legend_states[20] = "Ca_Calmod in component intracellular_calcium_concentration (millimolar)"
    legend_states[21] = "Ca_Trop in component intracellular_calcium_concentration (millimolar)"
    legend_constants[53] = "Calmod in component intracellular_calcium_concentration (millimolar)"
    legend_constants[54] = "Trop in component intracellular_calcium_concentration (millimolar)"
    legend_constants[55] = "alpha_Calmod in component intracellular_calcium_concentration (per_millimolar_second)"
    legend_constants[56] = "beta_Calmod in component intracellular_calcium_concentration (per_second)"
    legend_constants[57] = "alpha_Trop in component intracellular_calcium_concentration (per_millimolar_second)"
    legend_constants[58] = "beta_Trop in component intracellular_calcium_concentration (per_second)"
    legend_constants[59] = "radius in component intracellular_calcium_concentration (micrometre)"
    legend_constants[60] = "length in component intracellular_calcium_concentration (micrometre)"
    legend_constants[66] = "V_Cell in component intracellular_calcium_concentration (micrometre3)"
    legend_constants[68] = "V_i_ratio in component intracellular_calcium_concentration (dimensionless)"
    legend_constants[61] = "V_ds_ratio in component intracellular_calcium_concentration (dimensionless)"
    legend_constants[62] = "V_rel_ratio in component intracellular_calcium_concentration (dimensionless)"
    legend_constants[63] = "V_e_ratio in component intracellular_calcium_concentration (dimensionless)"
    legend_constants[64] = "V_up_ratio in component intracellular_calcium_concentration (dimensionless)"
    legend_constants[65] = "Kdecay in component intracellular_calcium_concentration (per_second)"
    legend_rates[0] = "d/dt V in component membrane (millivolt)"
    legend_rates[4] = "d/dt xr1 in component rapid_delayed_rectifier_potassium_current_xr1_gate (dimensionless)"
    legend_rates[5] = "d/dt xr2 in component rapid_delayed_rectifier_potassium_current_xr2_gate (dimensionless)"
    legend_rates[6] = "d/dt xs in component slow_delayed_rectifier_potassium_current_xs_gate (dimensionless)"
    legend_rates[7] = "d/dt m in component fast_sodium_current_m_gate (dimensionless)"
    legend_rates[8] = "d/dt h in component fast_sodium_current_h_gate (dimensionless)"
    legend_rates[10] = "d/dt d in component L_type_Ca_channel_d_gate (dimensionless)"
    legend_rates[11] = "d/dt f in component L_type_Ca_channel_f_gate (dimensionless)"
    legend_rates[12] = "d/dt f2 in component L_type_Ca_channel_f2_gate (dimensionless)"
    legend_rates[13] = "d/dt f2ds in component L_type_Ca_channel_f2ds_gate (dimensionless)"
    legend_rates[14] = "d/dt s in component transient_outward_current_s_gate (dimensionless)"
    legend_rates[15] = "d/dt r in component transient_outward_current_r_gate (dimensionless)"
    legend_rates[18] = "d/dt ActFrac in component calcium_release (dimensionless)"
    legend_rates[19] = "d/dt ProdFrac in component calcium_release (dimensionless)"
    legend_rates[2] = "d/dt Na_i in component intracellular_sodium_concentration (millimolar)"
    legend_rates[1] = "d/dt K_i in component intracellular_potassium_concentration (millimolar)"
    legend_rates[3] = "d/dt Ca_i in component intracellular_calcium_concentration (millimolar)"
    legend_rates[20] = "d/dt Ca_Calmod in component intracellular_calcium_concentration (millimolar)"
    legend_rates[21] = "d/dt Ca_Trop in component intracellular_calcium_concentration (millimolar)"
    legend_rates[9] = "d/dt Ca_ds in component intracellular_calcium_concentration (millimolar)"
    legend_rates[16] = "d/dt Ca_up in component intracellular_calcium_concentration (millimolar)"
    legend_rates[17] = "d/dt Ca_rel in component intracellular_calcium_concentration (millimolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = -92.849333
    constants[0] = 8314.472
    constants[1] = 310
    constants[2] = 96485.3415
    constants[3] = 9.5e-5
    constants[4] = 0.1
    constants[5] = 100000
    constants[6] = 1
    constants[7] = 0.003
    constants[8] = -3
    constants[9] = 0.03
    constants[10] = 4
    constants[11] = 140
    states[1] = 136.5644281
    states[2] = 7.3321223
    constants[12] = 2
    states[3] = 1.4e-5
    constants[13] = 10
    constants[14] = 0.5
    constants[15] = 0.0021
    constants[16] = 0.0013
    states[4] = 1.03e-5
    states[5] = 2e-7
    constants[17] = 0.0026
    states[6] = 0.001302
    constants[18] = 2.5
    states[7] = 0.0016203
    states[8] = 0.9944036
    constants[19] = 1e-5
    constants[20] = 0
    constants[21] = 0.004
    constants[22] = 0.0006
    constants[23] = 0.1
    constants[24] = 0.002
    constants[25] = 0.01
    states[9] = 1.88e-5
    states[10] = 0
    states[11] = 1
    states[12] = 0.9349197
    states[13] = 0.9651958
    constants[26] = 100000
    constants[27] = 0.001
    constants[28] = 20
    constants[29] = 1
    constants[30] = 3
    constants[31] = 0.3
    constants[32] = 0.0001
    constants[33] = 0.00025
    constants[34] = 0.005
    constants[35] = 0
    states[14] = 0.9948645
    states[15] = 0
    constants[36] = 0.7
    constants[37] = 1
    constants[38] = 40
    constants[39] = 0.0005
    constants[40] = 3
    constants[41] = 0
    constants[42] = 0.5
    constants[43] = 0.001
    constants[44] = 0.0003
    constants[45] = 0.4
    constants[46] = 0.5
    constants[47] = 0.4
    constants[48] = 0.03
    states[16] = 0.4531889
    states[17] = 0.4481927
    constants[49] = 0.05
    constants[50] = 250
    constants[51] = 0.0005
    constants[52] = 0.01
    states[18] = 0.0042614
    states[19] = 0.4068154
    states[20] = 0.0005555
    states[21] = 0.0003542
    constants[53] = 0.02
    constants[54] = 0.05
    constants[55] = 100000
    constants[56] = 50
    constants[57] = 100000
    constants[58] = 200
    constants[59] = 0.012
    constants[60] = 0.074
    constants[61] = 0.1
    constants[62] = 0.1
    constants[63] = 0.4
    constants[64] = 0.01
    constants[65] = 10
    constants[66] = 3.14159*(power(constants[59], 2.00000))*constants[60]
    constants[67] = (constants[44]*constants[45])/constants[46]
    constants[68] = ((1.00000-constants[63])-constants[64])-constants[62]
    constants[69] = constants[66]*constants[68]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[12] = 1.00000-1.00000*(states[3]/(constants[26]+states[3])+states[12])
    rates[13] = constants[28]*(1.00000-(states[9]/(constants[27]+states[9])+states[13]))
    rates[15] = 333.000*(1.00000/(1.00000+exp(-(states[0]+4.00000)/5.00000))-states[15])
    algebraic[0] = 50.0000/(1.00000+exp(-(states[0]-5.00000)/9.00000))
    algebraic[10] = 0.0500000*exp(-(states[0]-20.0000)/15.0000)
    rates[4] = algebraic[0]*(1.00000-states[4])-algebraic[10]*states[4]
    algebraic[1] = 50.0000/(1.00000+exp(-(states[0]-5.00000)/9.00000))
    algebraic[11] = 0.400000*exp(-(power((states[0]+30.0000)/30.0000, 3.00000)))
    rates[5] = algebraic[1]*(1.00000-states[5])-algebraic[11]*states[5]
    algebraic[2] = 14.0000/(1.00000+exp(-(states[0]-40.0000)/9.00000))
    algebraic[12] = 1.00000*exp(-states[0]/45.0000)
    rates[6] = algebraic[2]*(1.00000-states[6])-algebraic[12]*states[6]
    algebraic[4] = 20.0000*exp(-0.125000*((states[0]+75.0000)-constants[20]))
    algebraic[14] = 2000.00/(1.00000+320.000*exp(-0.100000*((states[0]+75.0000)-constants[20])))
    rates[8] = algebraic[4]*(1.00000-states[8])-algebraic[14]*states[8]
    algebraic[7] = 0.0330000*exp(-states[0]/17.0000)
    algebraic[17] = 33.0000/(1.00000+exp(-0.125000*(states[0]+10.0000)))
    rates[14] = algebraic[7]*(1.00000-states[14])-algebraic[17]*states[14]
    algebraic[3] = states[0]+41.0000
    algebraic[13] = custom_piecewise([less(fabs(algebraic[3]) , constants[19]), 2000.00 , True, (200.000*algebraic[3])/(1.00000-exp(-0.100000*algebraic[3]))])
    algebraic[20] = 8000.00*exp(-0.0560000*(states[0]+66.0000))
    rates[7] = algebraic[13]*(1.00000-states[7])-algebraic[20]*states[7]
    algebraic[5] = (states[0]+24.0000)-5.00000
    algebraic[15] = custom_piecewise([less(fabs(algebraic[5]) , 0.000100000), 120.000 , True, (30.0000*algebraic[5])/(1.00000-exp(-algebraic[5]/4.00000))])
    algebraic[21] = custom_piecewise([less(fabs(algebraic[5]) , 0.000100000), 120.000 , True, (12.0000*algebraic[5])/(exp(algebraic[5]/10.0000)-1.00000)])
    rates[10] = constants[30]*(algebraic[15]*(1.00000-states[10])-algebraic[21]*states[10])
    algebraic[6] = states[0]+34.0000
    algebraic[16] = custom_piecewise([less(fabs(algebraic[6]) , constants[32]), 25.0000 , True, (6.25000*algebraic[6])/(exp(algebraic[6]/4.00000)-1.00000)])
    algebraic[22] = 12.0000/(1.00000+exp((-1.00000*(states[0]+34.0000))/4.00000))
    rates[11] = constants[31]*(algebraic[16]*(1.00000-states[11])-algebraic[22]*states[11])
    algebraic[19] = states[3]/(states[3]+constants[51])
    algebraic[24] = states[9]/(states[9]+constants[52])
    algebraic[26] = algebraic[19]+(1.00000-algebraic[19])*algebraic[24]
    algebraic[30] = 60.0000+500.000*(power(algebraic[26], 2.00000))
    algebraic[32] = custom_piecewise([less(states[0] , -50.0000), 5.00000 , True, 1.00000])
    rates[19] = states[18]*algebraic[32]*algebraic[30]-algebraic[32]*1.00000*states[19]
    algebraic[9] = exp(0.0800000*(states[0]-40.0000))
    algebraic[28] = 0.00000*algebraic[9]+500.000*(power(algebraic[26], 2.00000))
    algebraic[34] = (1.00000-states[18])-states[19]
    rates[18] = algebraic[34]*algebraic[32]*algebraic[28]-states[18]*algebraic[32]*algebraic[30]
    algebraic[42] = (((constants[29]*4.00000*constants[23]*states[10]*states[11]*states[13]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2]*2.00000)/(constants[0]*constants[1]))))*(states[3]*exp((100.000*constants[2])/(constants[0]*constants[1]))-constants[12]*exp((-(states[0]-50.0000)*constants[2]*2.00000)/(constants[0]*constants[1])))
    rates[9] = (-1.00000*algebraic[42])/(2.00000*1.00000*constants[61]*constants[69]*constants[2])-states[9]*constants[65]
    algebraic[23] = ((constants[0]*constants[1])/constants[2])*log(constants[10]/states[1])
    algebraic[31] = (((constants[14]*constants[10])/(constants[10]+constants[13]))*(states[0]-algebraic[23]))/(1.00000+exp((((states[0]-algebraic[23])-10.0000)*constants[2]*1.25000)/(constants[0]*constants[1])))
    algebraic[47] = constants[34]*(constants[35]+states[14]*(1.00000-constants[35]))*states[15]*(states[0]-algebraic[23])
    algebraic[33] = (((constants[15]*states[4]+constants[16]*states[5])*1.00000)/(1.00000+exp((states[0]+9.00000)/22.4000)))*(states[0]-algebraic[23])
    algebraic[25] = ((constants[0]*constants[1])/constants[2])*log((constants[10]+constants[9]*constants[11])/(states[1]+constants[9]*states[2]))
    algebraic[35] = constants[17]*(power(states[6], 2.00000))*(states[0]-algebraic[25])
    algebraic[40] = ((((1.00000-constants[29])*constants[24]*constants[23]*states[10]*states[11]*states[12]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))))*(states[1]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-constants[10]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[43] = (((constants[29]*constants[24]*constants[23]*states[10]*states[11]*states[13]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))))*(states[1]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-constants[10]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[48] = (((constants[36]*constants[10])/(constants[37]+constants[10]))*states[2])/(constants[38]+states[2])
    rates[1] = (-1.00000/(1.00000*constants[69]*constants[2]))*((algebraic[31]+algebraic[33]+algebraic[35]+algebraic[40]+algebraic[43]+algebraic[47])-2.00000*algebraic[48])
    algebraic[29] = ((constants[0]*constants[1])/constants[2])*log((constants[11]+0.120000*constants[10])/(states[2]+0.120000*states[1]))
    algebraic[36] = constants[18]*(power(states[7], 3.00000))*states[8]*(states[0]-algebraic[29])
    algebraic[18] = ((constants[0]*constants[1])/constants[2])*log(constants[11]/states[2])
    algebraic[38] = constants[22]*(states[0]-algebraic[18])
    algebraic[37] = ((constants[21]*1.00000)/(1.00000+exp(-(states[0]+52.0000)/8.00000)))*(states[0]-algebraic[18])
    algebraic[41] = ((((1.00000-constants[29])*constants[25]*constants[23]*states[10]*states[11]*states[12]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))))*(states[2]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-constants[11]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[44] = (((constants[29]*constants[25]*constants[23]*states[10]*states[11]*states[13]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))))*(states[2]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-constants[11]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[49] = ((1.00000-constants[43])*constants[39]*(exp((constants[42]*(constants[40]-2.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(states[2], constants[40]))*constants[12]-exp(((constants[42]-1.00000)*(constants[40]-2.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(constants[11], constants[40]))*states[3]))/((1.00000+constants[41]*(states[3]*(power(constants[11], constants[40]))+constants[12]*(power(states[2], constants[40]))))*(1.00000+states[3]/0.00690000))
    rates[2] = (-1.00000/(1.00000*constants[69]*constants[2]))*(algebraic[36]+algebraic[37]+algebraic[38]+3.00000*algebraic[48]+3.00000*algebraic[49]+algebraic[41]+algebraic[44])
    algebraic[50] = (constants[43]*constants[39]*(exp((constants[42]*(constants[40]-2.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(states[2], constants[40]))*constants[12]-exp(((constants[42]-1.00000)*(constants[40]-2.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(constants[11], constants[40]))*states[9]))/((1.00000+constants[41]*(states[9]*(power(constants[11], constants[40]))+constants[12]*(power(states[2], constants[40]))))*(1.00000+states[9]/0.00690000))
    algebraic[39] = ((((1.00000-constants[29])*4.00000*constants[23]*states[10]*states[11]*states[12]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2]*2.00000)/(constants[0]*constants[1]))))*(states[3]*exp((100.000*constants[2])/(constants[0]*constants[1]))-constants[12]*exp((-(states[0]-50.0000)*constants[2]*2.00000)/(constants[0]*constants[1])))
    algebraic[27] = ((0.500000*constants[0]*constants[1])/constants[2])*log(constants[12]/states[3])
    algebraic[46] = constants[33]*(states[0]-algebraic[27])
    algebraic[8] = custom_piecewise([greater_equal(voi , constants[4]) & less_equal(voi , constants[5]) & less_equal((voi-constants[4])-floor((voi-constants[4])/constants[6])*constants[6] , constants[7]), constants[8] , True, 0.00000])
    rates[0] = (-1.00000/constants[3])*(algebraic[8]+algebraic[31]+algebraic[47]+algebraic[33]+algebraic[35]+algebraic[48]+algebraic[36]+algebraic[38]+algebraic[37]+algebraic[41]+algebraic[44]+algebraic[49]+algebraic[50]+algebraic[39]+algebraic[42]+algebraic[40]+algebraic[43]+algebraic[46])
    algebraic[52] = states[3]+states[16]*constants[67]+constants[44]*constants[45]+constants[44]
    algebraic[53] = (states[3]/algebraic[52])*constants[47]-((states[16]*constants[67])/algebraic[52])*constants[48]
    algebraic[54] = 50.0000*(states[16]-states[17])
    rates[16] = (constants[68]/constants[64])*algebraic[53]-algebraic[54]
    rates[20] = constants[55]*states[3]*(constants[53]-states[20])-constants[56]*states[20]
    algebraic[55] = ((power(states[18]/(states[18]+0.250000), 2.00000))*constants[50]+constants[49])*states[17]
    rates[17] = (constants[64]/constants[62])*algebraic[54]-algebraic[55]
    rates[21] = constants[57]*states[3]*(constants[54]-states[21])-constants[58]*states[21]
    rates[3] = ((((-1.00000/(2.00000*1.00000*constants[69]*constants[2]))*(((algebraic[39]+algebraic[46])-2.00000*algebraic[49])-2.00000*algebraic[50])+states[9]*constants[61]*constants[65]+(algebraic[55]*constants[62])/constants[68])-rates[20])-rates[21])-algebraic[53]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = 50.0000/(1.00000+exp(-(states[0]-5.00000)/9.00000))
    algebraic[10] = 0.0500000*exp(-(states[0]-20.0000)/15.0000)
    algebraic[1] = 50.0000/(1.00000+exp(-(states[0]-5.00000)/9.00000))
    algebraic[11] = 0.400000*exp(-(power((states[0]+30.0000)/30.0000, 3.00000)))
    algebraic[2] = 14.0000/(1.00000+exp(-(states[0]-40.0000)/9.00000))
    algebraic[12] = 1.00000*exp(-states[0]/45.0000)
    algebraic[4] = 20.0000*exp(-0.125000*((states[0]+75.0000)-constants[20]))
    algebraic[14] = 2000.00/(1.00000+320.000*exp(-0.100000*((states[0]+75.0000)-constants[20])))
    algebraic[7] = 0.0330000*exp(-states[0]/17.0000)
    algebraic[17] = 33.0000/(1.00000+exp(-0.125000*(states[0]+10.0000)))
    algebraic[3] = states[0]+41.0000
    algebraic[13] = custom_piecewise([less(fabs(algebraic[3]) , constants[19]), 2000.00 , True, (200.000*algebraic[3])/(1.00000-exp(-0.100000*algebraic[3]))])
    algebraic[20] = 8000.00*exp(-0.0560000*(states[0]+66.0000))
    algebraic[5] = (states[0]+24.0000)-5.00000
    algebraic[15] = custom_piecewise([less(fabs(algebraic[5]) , 0.000100000), 120.000 , True, (30.0000*algebraic[5])/(1.00000-exp(-algebraic[5]/4.00000))])
    algebraic[21] = custom_piecewise([less(fabs(algebraic[5]) , 0.000100000), 120.000 , True, (12.0000*algebraic[5])/(exp(algebraic[5]/10.0000)-1.00000)])
    algebraic[6] = states[0]+34.0000
    algebraic[16] = custom_piecewise([less(fabs(algebraic[6]) , constants[32]), 25.0000 , True, (6.25000*algebraic[6])/(exp(algebraic[6]/4.00000)-1.00000)])
    algebraic[22] = 12.0000/(1.00000+exp((-1.00000*(states[0]+34.0000))/4.00000))
    algebraic[19] = states[3]/(states[3]+constants[51])
    algebraic[24] = states[9]/(states[9]+constants[52])
    algebraic[26] = algebraic[19]+(1.00000-algebraic[19])*algebraic[24]
    algebraic[30] = 60.0000+500.000*(power(algebraic[26], 2.00000))
    algebraic[32] = custom_piecewise([less(states[0] , -50.0000), 5.00000 , True, 1.00000])
    algebraic[9] = exp(0.0800000*(states[0]-40.0000))
    algebraic[28] = 0.00000*algebraic[9]+500.000*(power(algebraic[26], 2.00000))
    algebraic[34] = (1.00000-states[18])-states[19]
    algebraic[42] = (((constants[29]*4.00000*constants[23]*states[10]*states[11]*states[13]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2]*2.00000)/(constants[0]*constants[1]))))*(states[3]*exp((100.000*constants[2])/(constants[0]*constants[1]))-constants[12]*exp((-(states[0]-50.0000)*constants[2]*2.00000)/(constants[0]*constants[1])))
    algebraic[23] = ((constants[0]*constants[1])/constants[2])*log(constants[10]/states[1])
    algebraic[31] = (((constants[14]*constants[10])/(constants[10]+constants[13]))*(states[0]-algebraic[23]))/(1.00000+exp((((states[0]-algebraic[23])-10.0000)*constants[2]*1.25000)/(constants[0]*constants[1])))
    algebraic[47] = constants[34]*(constants[35]+states[14]*(1.00000-constants[35]))*states[15]*(states[0]-algebraic[23])
    algebraic[33] = (((constants[15]*states[4]+constants[16]*states[5])*1.00000)/(1.00000+exp((states[0]+9.00000)/22.4000)))*(states[0]-algebraic[23])
    algebraic[25] = ((constants[0]*constants[1])/constants[2])*log((constants[10]+constants[9]*constants[11])/(states[1]+constants[9]*states[2]))
    algebraic[35] = constants[17]*(power(states[6], 2.00000))*(states[0]-algebraic[25])
    algebraic[40] = ((((1.00000-constants[29])*constants[24]*constants[23]*states[10]*states[11]*states[12]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))))*(states[1]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-constants[10]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[43] = (((constants[29]*constants[24]*constants[23]*states[10]*states[11]*states[13]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))))*(states[1]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-constants[10]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[48] = (((constants[36]*constants[10])/(constants[37]+constants[10]))*states[2])/(constants[38]+states[2])
    algebraic[29] = ((constants[0]*constants[1])/constants[2])*log((constants[11]+0.120000*constants[10])/(states[2]+0.120000*states[1]))
    algebraic[36] = constants[18]*(power(states[7], 3.00000))*states[8]*(states[0]-algebraic[29])
    algebraic[18] = ((constants[0]*constants[1])/constants[2])*log(constants[11]/states[2])
    algebraic[38] = constants[22]*(states[0]-algebraic[18])
    algebraic[37] = ((constants[21]*1.00000)/(1.00000+exp(-(states[0]+52.0000)/8.00000)))*(states[0]-algebraic[18])
    algebraic[41] = ((((1.00000-constants[29])*constants[25]*constants[23]*states[10]*states[11]*states[12]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))))*(states[2]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-constants[11]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[44] = (((constants[29]*constants[25]*constants[23]*states[10]*states[11]*states[13]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))))*(states[2]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-constants[11]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[49] = ((1.00000-constants[43])*constants[39]*(exp((constants[42]*(constants[40]-2.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(states[2], constants[40]))*constants[12]-exp(((constants[42]-1.00000)*(constants[40]-2.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(constants[11], constants[40]))*states[3]))/((1.00000+constants[41]*(states[3]*(power(constants[11], constants[40]))+constants[12]*(power(states[2], constants[40]))))*(1.00000+states[3]/0.00690000))
    algebraic[50] = (constants[43]*constants[39]*(exp((constants[42]*(constants[40]-2.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(states[2], constants[40]))*constants[12]-exp(((constants[42]-1.00000)*(constants[40]-2.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(constants[11], constants[40]))*states[9]))/((1.00000+constants[41]*(states[9]*(power(constants[11], constants[40]))+constants[12]*(power(states[2], constants[40]))))*(1.00000+states[9]/0.00690000))
    algebraic[39] = ((((1.00000-constants[29])*4.00000*constants[23]*states[10]*states[11]*states[12]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2]*2.00000)/(constants[0]*constants[1]))))*(states[3]*exp((100.000*constants[2])/(constants[0]*constants[1]))-constants[12]*exp((-(states[0]-50.0000)*constants[2]*2.00000)/(constants[0]*constants[1])))
    algebraic[27] = ((0.500000*constants[0]*constants[1])/constants[2])*log(constants[12]/states[3])
    algebraic[46] = constants[33]*(states[0]-algebraic[27])
    algebraic[8] = custom_piecewise([greater_equal(voi , constants[4]) & less_equal(voi , constants[5]) & less_equal((voi-constants[4])-floor((voi-constants[4])/constants[6])*constants[6] , constants[7]), constants[8] , True, 0.00000])
    algebraic[52] = states[3]+states[16]*constants[67]+constants[44]*constants[45]+constants[44]
    algebraic[53] = (states[3]/algebraic[52])*constants[47]-((states[16]*constants[67])/algebraic[52])*constants[48]
    algebraic[54] = 50.0000*(states[16]-states[17])
    algebraic[55] = ((power(states[18]/(states[18]+0.250000), 2.00000))*constants[50]+constants[49])*states[17]
    algebraic[45] = algebraic[39]+algebraic[40]+algebraic[41]+algebraic[42]+algebraic[43]+algebraic[44]
    algebraic[51] = algebraic[49]+algebraic[50]
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