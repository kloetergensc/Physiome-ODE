# Size of variable arrays:
sizeAlgebraic = 60
sizeStates = 20
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
    legend_states[0] = "V in component membrane (millivolt)"
    legend_constants[0] = "R in component membrane (joule_per_kilomole_kelvin)"
    legend_constants[1] = "T in component membrane (kelvin)"
    legend_constants[2] = "F in component membrane (coulomb_per_mole)"
    legend_constants[3] = "Cm in component membrane (microF)"
    legend_algebraic[28] = "i_K1 in component time_independent_potassium_current (nanoA)"
    legend_algebraic[51] = "i_to in component transient_outward_current (nanoA)"
    legend_algebraic[30] = "i_Kr in component rapid_delayed_rectifier_potassium_current (nanoA)"
    legend_algebraic[32] = "i_Ks in component slow_delayed_rectifier_potassium_current (nanoA)"
    legend_algebraic[42] = "i_Ca_L_K_cyt in component L_type_Ca_channel (nanoA)"
    legend_algebraic[47] = "i_Ca_L_K_ds in component L_type_Ca_channel (nanoA)"
    legend_algebraic[52] = "i_NaK in component sodium_potassium_pump (nanoA)"
    legend_algebraic[33] = "i_Na in component fast_sodium_current (nanoA)"
    legend_algebraic[35] = "i_b_Na in component sodium_background_current (nanoA)"
    legend_algebraic[34] = "i_p_Na in component persistent_sodium_current (nanoA)"
    legend_algebraic[43] = "i_Ca_L_Na_cyt in component L_type_Ca_channel (nanoA)"
    legend_algebraic[48] = "i_Ca_L_Na_ds in component L_type_Ca_channel (nanoA)"
    legend_algebraic[53] = "i_NaCa_cyt in component sodium_calcium_exchanger (nanoA)"
    legend_algebraic[54] = "i_NaCa_ds in component sodium_calcium_exchanger (nanoA)"
    legend_algebraic[41] = "i_Ca_L_Ca_cyt in component L_type_Ca_channel (nanoA)"
    legend_algebraic[46] = "i_Ca_L_Ca_ds in component L_type_Ca_channel (nanoA)"
    legend_algebraic[50] = "i_b_Ca in component calcium_background_current (nanoA)"
    legend_algebraic[6] = "i_Stim in component membrane (nanoA)"
    legend_constants[4] = "stim_start in component membrane (second)"
    legend_constants[5] = "stim_end in component membrane (second)"
    legend_constants[6] = "stim_period in component membrane (second)"
    legend_constants[7] = "stim_duration in component membrane (second)"
    legend_constants[8] = "stim_amplitude in component membrane (nanoA)"
    legend_algebraic[15] = "E_Na in component reversal_potentials (millivolt)"
    legend_algebraic[20] = "E_K in component reversal_potentials (millivolt)"
    legend_algebraic[22] = "E_Ks in component reversal_potentials (millivolt)"
    legend_algebraic[24] = "E_Ca in component reversal_potentials (millivolt)"
    legend_algebraic[26] = "E_mh in component reversal_potentials (millivolt)"
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
    legend_algebraic[9] = "beta_xr1 in component rapid_delayed_rectifier_potassium_current_xr1_gate (per_second)"
    legend_algebraic[1] = "alpha_xr2 in component rapid_delayed_rectifier_potassium_current_xr2_gate (per_second)"
    legend_algebraic[10] = "beta_xr2 in component rapid_delayed_rectifier_potassium_current_xr2_gate (per_second)"
    legend_constants[17] = "g_Ks in component slow_delayed_rectifier_potassium_current (microS)"
    legend_states[6] = "xs in component slow_delayed_rectifier_potassium_current_xs_gate (dimensionless)"
    legend_algebraic[2] = "alpha_xs in component slow_delayed_rectifier_potassium_current_xs_gate (per_second)"
    legend_algebraic[11] = "beta_xs in component slow_delayed_rectifier_potassium_current_xs_gate (per_second)"
    legend_constants[18] = "g_Na in component fast_sodium_current (microS)"
    legend_states[7] = "m in component fast_sodium_current_m_gate (dimensionless)"
    legend_states[8] = "h in component fast_sodium_current_h_gate (dimensionless)"
    legend_algebraic[12] = "alpha_m in component fast_sodium_current_m_gate (per_second)"
    legend_algebraic[18] = "beta_m in component fast_sodium_current_m_gate (per_second)"
    legend_constants[19] = "delta_m in component fast_sodium_current_m_gate (millivolt)"
    legend_algebraic[3] = "E0_m in component fast_sodium_current_m_gate (millivolt)"
    legend_algebraic[4] = "alpha_h in component fast_sodium_current_h_gate (per_second)"
    legend_algebraic[13] = "beta_h in component fast_sodium_current_h_gate (per_second)"
    legend_constants[20] = "shift_h in component fast_sodium_current_h_gate (millivolt)"
    legend_constants[21] = "g_pna in component persistent_sodium_current (microS)"
    legend_constants[22] = "g_bna in component sodium_background_current (microS)"
    legend_algebraic[49] = "i_Ca_L in component L_type_Ca_channel (nanoA)"
    legend_constants[23] = "P_Ca_L in component L_type_Ca_channel (nanoA_per_millimolar)"
    legend_constants[24] = "P_CaK in component L_type_Ca_channel (dimensionless)"
    legend_constants[25] = "P_CaNa in component L_type_Ca_channel (dimensionless)"
    legend_states[9] = "Ca_ds in component intracellular_calcium_concentration (millimolar)"
    legend_states[10] = "d in component L_type_Ca_channel_d_gate (dimensionless)"
    legend_algebraic[40] = "CaChoncyt in component L_type_Ca_channel_f_gate (dimensionless)"
    legend_algebraic[45] = "CaChonds in component L_type_Ca_channel_f_gate (dimensionless)"
    legend_constants[26] = "KCaChoff in component L_type_Ca_channel (millimolar)"
    legend_constants[27] = "Kmdsinact in component L_type_Ca_channel (millimolar)"
    legend_constants[28] = "FrICa in component L_type_Ca_channel (dimensionless)"
    legend_algebraic[14] = "alpha_d in component L_type_Ca_channel_d_gate (per_second)"
    legend_algebraic[19] = "beta_d in component L_type_Ca_channel_d_gate (per_second)"
    legend_algebraic[5] = "E0_d in component L_type_Ca_channel_d_gate (millivolt)"
    legend_constants[29] = "speed_d in component L_type_Ca_channel_d_gate (dimensionless)"
    legend_states[11] = "f in component L_type_Ca_channel_f_gate (dimensionless)"
    legend_algebraic[39] = "CaChoffcyt in component L_type_Ca_channel_f_gate (dimensionless)"
    legend_algebraic[44] = "CaChoffds in component L_type_Ca_channel_f_gate (dimensionless)"
    legend_algebraic[37] = "alpha_f in component L_type_Ca_channel_f_gate (per_second)"
    legend_algebraic[38] = "beta_f in component L_type_Ca_channel_f_gate (per_second)"
    legend_constants[30] = "speed_f in component L_type_Ca_channel_f_gate (dimensionless)"
    legend_constants[31] = "delta_f in component L_type_Ca_channel_f_gate (millivolt)"
    legend_algebraic[36] = "E0_f in component L_type_Ca_channel_f_gate (millivolt)"
    legend_constants[32] = "g_bca in component calcium_background_current (microS)"
    legend_constants[33] = "g_to in component transient_outward_current (microS)"
    legend_constants[34] = "g_tos in component transient_outward_current (dimensionless)"
    legend_states[12] = "s in component transient_outward_current_s_gate (dimensionless)"
    legend_states[13] = "r in component transient_outward_current_r_gate (dimensionless)"
    legend_algebraic[7] = "alpha_s in component transient_outward_current_s_gate (per_second)"
    legend_algebraic[16] = "beta_s in component transient_outward_current_s_gate (per_second)"
    legend_constants[35] = "i_NaK_max in component sodium_potassium_pump (nanoA)"
    legend_constants[36] = "K_mK in component sodium_potassium_pump (millimolar)"
    legend_constants[37] = "K_mNa in component sodium_potassium_pump (millimolar)"
    legend_algebraic[55] = "i_NaCa in component sodium_calcium_exchanger (nanoA)"
    legend_constants[38] = "k_NaCa in component sodium_calcium_exchanger (nanoA)"
    legend_constants[39] = "n_NaCa in component sodium_calcium_exchanger (dimensionless)"
    legend_constants[40] = "d_NaCa in component sodium_calcium_exchanger (dimensionless)"
    legend_constants[41] = "gamma in component sodium_calcium_exchanger (dimensionless)"
    legend_constants[42] = "FRiNaCa in component sodium_calcium_exchanger (dimensionless)"
    legend_algebraic[57] = "i_up in component sarcoplasmic_reticulum_calcium_pump (millimolar_per_second)"
    legend_constants[66] = "K_1 in component sarcoplasmic_reticulum_calcium_pump (dimensionless)"
    legend_algebraic[56] = "K_2 in component sarcoplasmic_reticulum_calcium_pump (millimolar)"
    legend_constants[43] = "K_cyca in component sarcoplasmic_reticulum_calcium_pump (millimolar)"
    legend_constants[44] = "K_xcs in component sarcoplasmic_reticulum_calcium_pump (dimensionless)"
    legend_constants[45] = "K_srca in component sarcoplasmic_reticulum_calcium_pump (millimolar)"
    legend_constants[46] = "alpha_up in component sarcoplasmic_reticulum_calcium_pump (millimolar_per_second)"
    legend_constants[47] = "beta_up in component sarcoplasmic_reticulum_calcium_pump (millimolar_per_second)"
    legend_states[14] = "Ca_up in component intracellular_calcium_concentration (millimolar)"
    legend_algebraic[58] = "i_trans in component calcium_translocation (millimolar_per_second)"
    legend_states[15] = "Ca_rel in component intracellular_calcium_concentration (millimolar)"
    legend_algebraic[59] = "i_rel in component calcium_release (millimolar_per_second)"
    legend_algebraic[8] = "VoltDep in component calcium_release (dimensionless)"
    legend_algebraic[23] = "RegBindSite in component calcium_release (dimensionless)"
    legend_algebraic[17] = "CaiReg in component calcium_release (dimensionless)"
    legend_algebraic[21] = "CadsReg in component calcium_release (dimensionless)"
    legend_algebraic[25] = "ActRate in component calcium_release (per_second)"
    legend_algebraic[27] = "InactRate in component calcium_release (per_second)"
    legend_constants[48] = "K_leak_rate in component calcium_release (per_second)"
    legend_constants[49] = "K_m_rel in component calcium_release (per_second)"
    legend_constants[50] = "K_m_Ca_cyt in component calcium_release (millimolar)"
    legend_constants[51] = "K_m_Ca_ds in component calcium_release (millimolar)"
    legend_algebraic[31] = "PrecFrac in component calcium_release (dimensionless)"
    legend_states[16] = "ActFrac in component calcium_release (dimensionless)"
    legend_states[17] = "ProdFrac in component calcium_release (dimensionless)"
    legend_algebraic[29] = "SpeedRel in component calcium_release (dimensionless)"
    legend_constants[68] = "V_i in component intracellular_calcium_concentration (micrometre3)"
    legend_states[18] = "Ca_Calmod in component intracellular_calcium_concentration (millimolar)"
    legend_states[19] = "Ca_Trop in component intracellular_calcium_concentration (millimolar)"
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
    legend_rates[4] = "d/dt xr1 in component rapid_delayed_rectifier_potassium_current_xr1_gate (dimensionless)"
    legend_rates[5] = "d/dt xr2 in component rapid_delayed_rectifier_potassium_current_xr2_gate (dimensionless)"
    legend_rates[6] = "d/dt xs in component slow_delayed_rectifier_potassium_current_xs_gate (dimensionless)"
    legend_rates[7] = "d/dt m in component fast_sodium_current_m_gate (dimensionless)"
    legend_rates[8] = "d/dt h in component fast_sodium_current_h_gate (dimensionless)"
    legend_rates[10] = "d/dt d in component L_type_Ca_channel_d_gate (dimensionless)"
    legend_rates[11] = "d/dt f in component L_type_Ca_channel_f_gate (dimensionless)"
    legend_rates[12] = "d/dt s in component transient_outward_current_s_gate (dimensionless)"
    legend_rates[13] = "d/dt r in component transient_outward_current_r_gate (dimensionless)"
    legend_rates[16] = "d/dt ActFrac in component calcium_release (dimensionless)"
    legend_rates[17] = "d/dt ProdFrac in component calcium_release (dimensionless)"
    legend_rates[2] = "d/dt Na_i in component intracellular_sodium_concentration (millimolar)"
    legend_rates[1] = "d/dt K_i in component intracellular_potassium_concentration (millimolar)"
    legend_rates[3] = "d/dt Ca_i in component intracellular_calcium_concentration (millimolar)"
    legend_rates[18] = "d/dt Ca_Calmod in component intracellular_calcium_concentration (millimolar)"
    legend_rates[19] = "d/dt Ca_Trop in component intracellular_calcium_concentration (millimolar)"
    legend_rates[9] = "d/dt Ca_ds in component intracellular_calcium_concentration (millimolar)"
    legend_rates[14] = "d/dt Ca_up in component intracellular_calcium_concentration (millimolar)"
    legend_rates[15] = "d/dt Ca_rel in component intracellular_calcium_concentration (millimolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = -93.0550060340878
    constants[0] = 8314.472
    constants[1] = 310
    constants[2] = 96485.3415
    constants[3] = 9.5e-5
    constants[4] = 0.1
    constants[5] = 600
    constants[6] = 1
    constants[7] = 0.002
    constants[8] = -6
    constants[9] = 0.03
    constants[10] = 4
    constants[11] = 140
    states[1] = 138.49601832864
    states[2] = 5.68929064089788
    constants[12] = 2
    states[3] = 5.60867201979414e-6
    constants[13] = 10
    constants[14] = 0.5
    constants[15] = 0.0028
    constants[16] = 0.0017
    states[4] = 9.88715446781661e-6
    states[5] = 2.15138346929839e-7
    constants[17] = 0.0016
    states[6] = 6.72192033116639e-7
    constants[18] = 0.5
    states[7] = 0.00157541794600382
    states[8] = 0.99465397024286
    constants[19] = 1e-5
    constants[20] = 0
    constants[21] = 0.0027
    constants[22] = 0.0006
    constants[23] = 1
    constants[24] = 0.002
    constants[25] = 0.01
    states[9] = 2.15623822848619e-7
    states[10] = 2.27643784381792e-8
    constants[26] = 0.01
    constants[27] = 0.001
    constants[28] = 0.7
    constants[29] = 3
    states[11] = 4.03082807230434e-9
    constants[30] = 0.3
    constants[31] = 0.0001
    constants[32] = 0.00025
    constants[33] = 0.005
    constants[34] = 0
    states[12] = 0.999870016703514
    states[13] = 1.8398417043009e-8
    constants[35] = 0.7
    constants[36] = 1
    constants[37] = 40
    constants[38] = 0.0005
    constants[39] = 3
    constants[40] = 0
    constants[41] = 0.2
    constants[42] = 0.001
    constants[43] = 0.0003
    constants[44] = 0.4
    constants[45] = 0.5
    constants[46] = 0.4
    constants[47] = 0.03
    states[14] = 0.187470810108547
    states[15] = 0.185609773872992
    constants[48] = 0.05
    constants[49] = 250
    constants[50] = 0.0005
    constants[51] = 0.01
    states[16] = 0.000183030061767109
    states[17] = 0.0109931117325847
    states[18] = 0.000221858224298955
    states[19] = 0.00013982468565579
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
    constants[66] = (constants[43]*constants[44])/constants[45]
    constants[67] = ((1.00000-constants[62])-constants[63])-constants[61]
    constants[68] = constants[65]*constants[67]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[13] = 333.000*(1.00000/(1.00000+exp(-(states[0]+4.00000)/5.00000))-states[13])
    algebraic[0] = 50.0000/(1.00000+exp(-(states[0]-5.00000)/9.00000))
    algebraic[9] = 0.0500000*exp(-(states[0]-20.0000)/15.0000)
    rates[4] = algebraic[0]*(1.00000-states[4])-algebraic[9]*states[4]
    algebraic[1] = 50.0000/(1.00000+exp(-(states[0]-5.00000)/9.00000))
    algebraic[10] = 0.400000*exp(-(power((states[0]+30.0000)/30.0000, 3.00000)))
    rates[5] = algebraic[1]*(1.00000-states[5])-algebraic[10]*states[5]
    algebraic[2] = 14.0000/(1.00000+exp(-(states[0]-40.0000)/9.00000))
    algebraic[11] = 1.00000*exp(-states[0]/45.0000)
    rates[6] = algebraic[2]*(1.00000-states[6])-algebraic[11]*states[6]
    algebraic[4] = 20.0000*exp(-0.125000*((states[0]+75.0000)-constants[20]))
    algebraic[13] = 2000.00/(1.00000+320.000*exp(-0.100000*((states[0]+75.0000)-constants[20])))
    rates[8] = algebraic[4]*(1.00000-states[8])-algebraic[13]*states[8]
    algebraic[7] = 0.0330000*exp(-states[0]/17.0000)
    algebraic[16] = 33.0000/(1.00000+exp(-0.125000*(states[0]+10.0000)))
    rates[12] = algebraic[7]*(1.00000-states[12])-algebraic[16]*states[12]
    algebraic[3] = states[0]+41.0000
    algebraic[12] = custom_piecewise([less(fabs(algebraic[3]) , constants[19]), 2000.00 , True, (200.000*algebraic[3])/(1.00000-exp(-0.100000*algebraic[3]))])
    algebraic[18] = 8000.00*exp(-0.0560000*(states[0]+66.0000))
    rates[7] = algebraic[12]*(1.00000-states[7])-algebraic[18]*states[7]
    algebraic[5] = (states[0]+24.0000)-5.00000
    algebraic[14] = custom_piecewise([less(fabs(algebraic[5]) , 0.000100000), 120.000 , True, (30.0000*algebraic[5])/(1.00000-exp(-algebraic[5]/4.00000))])
    algebraic[19] = custom_piecewise([less(fabs(algebraic[5]) , 0.000100000), 120.000 , True, (12.0000*algebraic[5])/(exp(algebraic[5]/10.0000)-1.00000)])
    rates[10] = constants[29]*(algebraic[14]*(1.00000-states[10])-algebraic[19]*states[10])
    algebraic[17] = states[3]/(states[3]+constants[50])
    algebraic[21] = states[9]/(states[9]+constants[51])
    algebraic[23] = algebraic[17]+(1.00000-algebraic[17])*algebraic[21]
    algebraic[27] = 60.0000+500.000*(power(algebraic[23], 2.00000))
    algebraic[29] = custom_piecewise([less(states[0] , -50.0000), 5.00000 , True, 1.00000])
    rates[17] = states[16]*algebraic[29]*algebraic[27]-algebraic[29]*1.00000*states[17]
    algebraic[8] = exp(0.0800000*(states[0]-40.0000))
    algebraic[25] = 0.00000*algebraic[8]+90.0000*(power(algebraic[23], 2.00000))
    algebraic[31] = (1.00000-states[16])-states[17]
    rates[16] = algebraic[31]*algebraic[29]*algebraic[25]-states[16]*algebraic[29]*algebraic[27]
    algebraic[39] = states[3]/(constants[26]+states[3])
    algebraic[36] = states[0]+34.0000
    algebraic[37] = custom_piecewise([less(fabs(algebraic[36]) , constants[31]), 25.0000 , True, (6.25000*algebraic[36])/(exp(algebraic[36]/4.00000)-1.00000)])
    algebraic[38] = 12.0000/(1.00000+exp((-1.00000*(states[0]+34.0000))/4.00000))
    rates[11] = (120.000*(1.00000-states[11])*algebraic[39]+(1.00000-states[11])*(1.00000-algebraic[39]))*constants[30]*algebraic[38]-algebraic[37]*states[11]
    algebraic[44] = states[9]/(constants[27]+states[9])
    algebraic[45] = (1.00000-states[11])*(1.00000-algebraic[44])
    algebraic[46] = (((constants[28]*4.00000*constants[23]*states[10]*algebraic[45]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2]*2.00000)/(constants[0]*constants[1]))))*(states[3]*exp((100.000*constants[2])/(constants[0]*constants[1]))-constants[12]*exp((-(states[0]-50.0000)*constants[2]*2.00000)/(constants[0]*constants[1])))
    rates[9] = (-1.00000*algebraic[46])/(2.00000*1.00000*constants[60]*constants[68]*constants[2])-states[9]*constants[64]
    algebraic[20] = ((constants[0]*constants[1])/constants[2])*log(constants[10]/states[1])
    algebraic[28] = (((constants[14]*constants[10])/(constants[10]+constants[13]))*(states[0]-algebraic[20]))/(1.00000+exp((((states[0]-algebraic[20])-10.0000)*constants[2]*1.25000)/(constants[0]*constants[1])))
    algebraic[51] = constants[33]*(constants[34]+states[12]*(1.00000-constants[34]))*states[13]*(states[0]-algebraic[20])
    algebraic[30] = (((constants[15]*states[4]+constants[16]*states[5])*1.00000)/(1.00000+exp((states[0]+9.00000)/22.4000)))*(states[0]-algebraic[20])
    algebraic[22] = ((constants[0]*constants[1])/constants[2])*log((constants[10]+constants[9]*constants[11])/(states[1]+constants[9]*states[2]))
    algebraic[32] = constants[17]*(power(states[6], 2.00000))*(states[0]-algebraic[22])
    algebraic[40] = (1.00000-states[11])*(1.00000-algebraic[39])
    algebraic[42] = ((((1.00000-constants[28])*constants[24]*constants[23]*states[10]*algebraic[40]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))))*(states[1]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-constants[10]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[47] = (((constants[28]*constants[24]*constants[23]*states[10]*algebraic[45]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))))*(states[1]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-constants[10]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[52] = (((constants[35]*constants[10])/(constants[36]+constants[10]))*states[2])/(constants[37]+states[2])
    rates[1] = (-1.00000/(1.00000*constants[68]*constants[2]))*((algebraic[28]+algebraic[30]+algebraic[32]+algebraic[42]+algebraic[47]+algebraic[51])-2.00000*algebraic[52])
    algebraic[26] = ((constants[0]*constants[1])/constants[2])*log((constants[11]+0.120000*constants[10])/(states[2]+0.120000*states[1]))
    algebraic[33] = constants[18]*(power(states[7], 3.00000))*states[8]*(states[0]-algebraic[26])
    algebraic[15] = ((constants[0]*constants[1])/constants[2])*log(constants[11]/states[2])
    algebraic[35] = constants[22]*(states[0]-algebraic[15])
    algebraic[34] = ((constants[21]*1.00000)/(1.00000+exp(-(states[0]+52.0000)/8.00000)))*(states[0]-algebraic[15])
    algebraic[43] = ((((1.00000-constants[28])*constants[25]*constants[23]*states[10]*algebraic[40]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))))*(states[2]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-constants[11]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[48] = (((constants[28]*constants[25]*constants[23]*states[10]*algebraic[45]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))))*(states[2]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-constants[11]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[53] = ((1.00000-constants[42])*constants[38]*(exp((constants[41]*(constants[39]-2.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(states[2], constants[39]))*constants[12]-exp(((constants[41]-1.00000)*(constants[39]-2.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(constants[11], constants[39]))*states[3]))/((1.00000+constants[40]*(states[3]*(power(constants[11], constants[39]))+constants[12]*(power(states[2], constants[39]))))*(1.00000+states[3]/0.00690000))
    rates[2] = (-1.00000/(1.00000*constants[68]*constants[2]))*(algebraic[33]+algebraic[34]+algebraic[35]+3.00000*algebraic[52]+3.00000*algebraic[53]+algebraic[43]+algebraic[48])
    algebraic[54] = (constants[42]*constants[38]*(exp((constants[41]*(constants[39]-2.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(states[2], constants[39]))*constants[12]-exp(((constants[41]-1.00000)*(constants[39]-2.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(constants[11], constants[39]))*states[9]))/((1.00000+constants[40]*(states[9]*(power(constants[11], constants[39]))+constants[12]*(power(states[2], constants[39]))))*(1.00000+states[9]/0.00690000))
    algebraic[41] = ((((1.00000-constants[28])*4.00000*constants[23]*states[10]*algebraic[40]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2]*2.00000)/(constants[0]*constants[1]))))*(states[3]*exp((100.000*constants[2])/(constants[0]*constants[1]))-constants[12]*exp((-(states[0]-50.0000)*constants[2]*2.00000)/(constants[0]*constants[1])))
    algebraic[24] = ((0.500000*constants[0]*constants[1])/constants[2])*log(constants[12]/states[3])
    algebraic[50] = constants[32]*(states[0]-algebraic[24])
    algebraic[6] = custom_piecewise([greater_equal(voi , constants[4]) & less_equal(voi , constants[5]) & less_equal((voi-constants[4])-floor((voi-constants[4])/constants[6])*constants[6] , constants[7]), constants[8] , True, 0.00000])
    rates[0] = (-1.00000/constants[3])*(algebraic[6]+algebraic[28]+algebraic[51]+algebraic[30]+algebraic[32]+algebraic[52]+algebraic[33]+algebraic[35]+algebraic[34]+algebraic[43]+algebraic[48]+algebraic[53]+algebraic[54]+algebraic[41]+algebraic[46]+algebraic[42]+algebraic[47]+algebraic[50])
    algebraic[56] = states[3]+states[14]*constants[66]+constants[43]*constants[44]+constants[43]
    algebraic[57] = (states[3]/algebraic[56])*constants[46]-((states[14]*constants[66])/algebraic[56])*constants[47]
    algebraic[58] = 50.0000*(states[14]-states[15])
    rates[14] = (constants[67]/constants[63])*algebraic[57]-algebraic[58]
    rates[18] = constants[54]*states[3]*(constants[52]-states[18])-constants[55]*states[18]
    algebraic[59] = ((power(states[16]/(states[16]+0.250000), 2.00000))*constants[49]+constants[48])*states[15]
    rates[15] = (constants[63]/constants[61])*algebraic[58]-algebraic[59]
    rates[19] = constants[56]*states[3]*(constants[53]-states[19])-constants[57]*states[19]
    rates[3] = ((((-1.00000/(2.00000*1.00000*constants[68]*constants[2]))*(((algebraic[41]+algebraic[50])-2.00000*algebraic[53])-2.00000*algebraic[54])+states[9]*constants[60]*constants[64]+(algebraic[59]*constants[61])/constants[67])-rates[18])-rates[19])-algebraic[57]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = 50.0000/(1.00000+exp(-(states[0]-5.00000)/9.00000))
    algebraic[9] = 0.0500000*exp(-(states[0]-20.0000)/15.0000)
    algebraic[1] = 50.0000/(1.00000+exp(-(states[0]-5.00000)/9.00000))
    algebraic[10] = 0.400000*exp(-(power((states[0]+30.0000)/30.0000, 3.00000)))
    algebraic[2] = 14.0000/(1.00000+exp(-(states[0]-40.0000)/9.00000))
    algebraic[11] = 1.00000*exp(-states[0]/45.0000)
    algebraic[4] = 20.0000*exp(-0.125000*((states[0]+75.0000)-constants[20]))
    algebraic[13] = 2000.00/(1.00000+320.000*exp(-0.100000*((states[0]+75.0000)-constants[20])))
    algebraic[7] = 0.0330000*exp(-states[0]/17.0000)
    algebraic[16] = 33.0000/(1.00000+exp(-0.125000*(states[0]+10.0000)))
    algebraic[3] = states[0]+41.0000
    algebraic[12] = custom_piecewise([less(fabs(algebraic[3]) , constants[19]), 2000.00 , True, (200.000*algebraic[3])/(1.00000-exp(-0.100000*algebraic[3]))])
    algebraic[18] = 8000.00*exp(-0.0560000*(states[0]+66.0000))
    algebraic[5] = (states[0]+24.0000)-5.00000
    algebraic[14] = custom_piecewise([less(fabs(algebraic[5]) , 0.000100000), 120.000 , True, (30.0000*algebraic[5])/(1.00000-exp(-algebraic[5]/4.00000))])
    algebraic[19] = custom_piecewise([less(fabs(algebraic[5]) , 0.000100000), 120.000 , True, (12.0000*algebraic[5])/(exp(algebraic[5]/10.0000)-1.00000)])
    algebraic[17] = states[3]/(states[3]+constants[50])
    algebraic[21] = states[9]/(states[9]+constants[51])
    algebraic[23] = algebraic[17]+(1.00000-algebraic[17])*algebraic[21]
    algebraic[27] = 60.0000+500.000*(power(algebraic[23], 2.00000))
    algebraic[29] = custom_piecewise([less(states[0] , -50.0000), 5.00000 , True, 1.00000])
    algebraic[8] = exp(0.0800000*(states[0]-40.0000))
    algebraic[25] = 0.00000*algebraic[8]+90.0000*(power(algebraic[23], 2.00000))
    algebraic[31] = (1.00000-states[16])-states[17]
    algebraic[39] = states[3]/(constants[26]+states[3])
    algebraic[36] = states[0]+34.0000
    algebraic[37] = custom_piecewise([less(fabs(algebraic[36]) , constants[31]), 25.0000 , True, (6.25000*algebraic[36])/(exp(algebraic[36]/4.00000)-1.00000)])
    algebraic[38] = 12.0000/(1.00000+exp((-1.00000*(states[0]+34.0000))/4.00000))
    algebraic[44] = states[9]/(constants[27]+states[9])
    algebraic[45] = (1.00000-states[11])*(1.00000-algebraic[44])
    algebraic[46] = (((constants[28]*4.00000*constants[23]*states[10]*algebraic[45]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2]*2.00000)/(constants[0]*constants[1]))))*(states[3]*exp((100.000*constants[2])/(constants[0]*constants[1]))-constants[12]*exp((-(states[0]-50.0000)*constants[2]*2.00000)/(constants[0]*constants[1])))
    algebraic[20] = ((constants[0]*constants[1])/constants[2])*log(constants[10]/states[1])
    algebraic[28] = (((constants[14]*constants[10])/(constants[10]+constants[13]))*(states[0]-algebraic[20]))/(1.00000+exp((((states[0]-algebraic[20])-10.0000)*constants[2]*1.25000)/(constants[0]*constants[1])))
    algebraic[51] = constants[33]*(constants[34]+states[12]*(1.00000-constants[34]))*states[13]*(states[0]-algebraic[20])
    algebraic[30] = (((constants[15]*states[4]+constants[16]*states[5])*1.00000)/(1.00000+exp((states[0]+9.00000)/22.4000)))*(states[0]-algebraic[20])
    algebraic[22] = ((constants[0]*constants[1])/constants[2])*log((constants[10]+constants[9]*constants[11])/(states[1]+constants[9]*states[2]))
    algebraic[32] = constants[17]*(power(states[6], 2.00000))*(states[0]-algebraic[22])
    algebraic[40] = (1.00000-states[11])*(1.00000-algebraic[39])
    algebraic[42] = ((((1.00000-constants[28])*constants[24]*constants[23]*states[10]*algebraic[40]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))))*(states[1]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-constants[10]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[47] = (((constants[28]*constants[24]*constants[23]*states[10]*algebraic[45]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))))*(states[1]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-constants[10]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[52] = (((constants[35]*constants[10])/(constants[36]+constants[10]))*states[2])/(constants[37]+states[2])
    algebraic[26] = ((constants[0]*constants[1])/constants[2])*log((constants[11]+0.120000*constants[10])/(states[2]+0.120000*states[1]))
    algebraic[33] = constants[18]*(power(states[7], 3.00000))*states[8]*(states[0]-algebraic[26])
    algebraic[15] = ((constants[0]*constants[1])/constants[2])*log(constants[11]/states[2])
    algebraic[35] = constants[22]*(states[0]-algebraic[15])
    algebraic[34] = ((constants[21]*1.00000)/(1.00000+exp(-(states[0]+52.0000)/8.00000)))*(states[0]-algebraic[15])
    algebraic[43] = ((((1.00000-constants[28])*constants[25]*constants[23]*states[10]*algebraic[40]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))))*(states[2]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-constants[11]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[48] = (((constants[28]*constants[25]*constants[23]*states[10]*algebraic[45]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))))*(states[2]*exp((50.0000*constants[2])/(constants[0]*constants[1]))-constants[11]*exp((-(states[0]-50.0000)*constants[2])/(constants[0]*constants[1])))
    algebraic[53] = ((1.00000-constants[42])*constants[38]*(exp((constants[41]*(constants[39]-2.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(states[2], constants[39]))*constants[12]-exp(((constants[41]-1.00000)*(constants[39]-2.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(constants[11], constants[39]))*states[3]))/((1.00000+constants[40]*(states[3]*(power(constants[11], constants[39]))+constants[12]*(power(states[2], constants[39]))))*(1.00000+states[3]/0.00690000))
    algebraic[54] = (constants[42]*constants[38]*(exp((constants[41]*(constants[39]-2.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(states[2], constants[39]))*constants[12]-exp(((constants[41]-1.00000)*(constants[39]-2.00000)*states[0]*constants[2])/(constants[0]*constants[1]))*(power(constants[11], constants[39]))*states[9]))/((1.00000+constants[40]*(states[9]*(power(constants[11], constants[39]))+constants[12]*(power(states[2], constants[39]))))*(1.00000+states[9]/0.00690000))
    algebraic[41] = ((((1.00000-constants[28])*4.00000*constants[23]*states[10]*algebraic[40]*(states[0]-50.0000)*constants[2])/(constants[0]*constants[1]))/(1.00000-exp((-(states[0]-50.0000)*constants[2]*2.00000)/(constants[0]*constants[1]))))*(states[3]*exp((100.000*constants[2])/(constants[0]*constants[1]))-constants[12]*exp((-(states[0]-50.0000)*constants[2]*2.00000)/(constants[0]*constants[1])))
    algebraic[24] = ((0.500000*constants[0]*constants[1])/constants[2])*log(constants[12]/states[3])
    algebraic[50] = constants[32]*(states[0]-algebraic[24])
    algebraic[6] = custom_piecewise([greater_equal(voi , constants[4]) & less_equal(voi , constants[5]) & less_equal((voi-constants[4])-floor((voi-constants[4])/constants[6])*constants[6] , constants[7]), constants[8] , True, 0.00000])
    algebraic[56] = states[3]+states[14]*constants[66]+constants[43]*constants[44]+constants[43]
    algebraic[57] = (states[3]/algebraic[56])*constants[46]-((states[14]*constants[66])/algebraic[56])*constants[47]
    algebraic[58] = 50.0000*(states[14]-states[15])
    algebraic[59] = ((power(states[16]/(states[16]+0.250000), 2.00000))*constants[49]+constants[48])*states[15]
    algebraic[49] = algebraic[41]+algebraic[42]+algebraic[43]+algebraic[46]+algebraic[47]+algebraic[48]
    algebraic[55] = algebraic[53]+algebraic[54]
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