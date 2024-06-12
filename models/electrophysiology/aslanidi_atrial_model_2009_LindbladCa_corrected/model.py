# Size of variable arrays:
sizeAlgebraic = 78
sizeStates = 29
sizeConstants = 49
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_constants[0] = "CT in component environment (dimensionless)"
    legend_constants[1] = "PM in component environment (dimensionless)"
    legend_states[0] = "V in component membrane (millivolt)"
    legend_constants[2] = "R in component membrane (millijoule_per_mole_kelvin)"
    legend_constants[3] = "T in component membrane (kelvin)"
    legend_constants[4] = "F in component membrane (coulomb_per_mole)"
    legend_constants[5] = "Cm in component membrane (nanoF)"
    legend_algebraic[30] = "i_Na in component sodium_current (picoA)"
    legend_algebraic[52] = "i_Ca_L in component L_type_Ca_channel (picoA)"
    legend_algebraic[57] = "i_Ca_T in component T_type_Ca_channel (picoA)"
    legend_algebraic[61] = "i_to in component Ca_independent_transient_outward_K_current (picoA)"
    legend_algebraic[62] = "i_sus in component Ca_independent_transient_outward_K_current (picoA)"
    legend_algebraic[65] = "i_K1 in component inward_rectifier (picoA)"
    legend_algebraic[64] = "i_Kr in component delayed_rectifier_K_current (picoA)"
    legend_algebraic[63] = "i_Ks in component delayed_rectifier_K_current (picoA)"
    legend_algebraic[67] = "i_B_Na in component background_currents (picoA)"
    legend_algebraic[68] = "i_B_Ca in component background_currents (picoA)"
    legend_algebraic[69] = "i_p in component sodium_potassium_pump (picoA)"
    legend_algebraic[70] = "i_CaP in component sarcolemmal_calcium_pump_current (picoA)"
    legend_algebraic[71] = "i_NaCa in component Na_Ca_ion_exchanger_current (picoA)"
    legend_algebraic[0] = "i_Stim in component membrane (picoA)"
    legend_constants[6] = "stim_start in component membrane (second)"
    legend_constants[7] = "stim_end in component membrane (second)"
    legend_constants[8] = "stim_period in component membrane (second)"
    legend_constants[9] = "stim_duration in component membrane (second)"
    legend_constants[10] = "stim_amplitude in component membrane (picoA)"
    legend_algebraic[15] = "E_Na in component sodium_current (millivolt)"
    legend_constants[11] = "P_Na in component sodium_current (nanolitre_per_second)"
    legend_constants[12] = "Na_c in component cleft_space_ion_concentrations (millimolar)"
    legend_states[1] = "Na_i in component intracellular_ion_concentrations (millimolar)"
    legend_states[2] = "m in component sodium_current_m_gate (dimensionless)"
    legend_states[3] = "h1 in component sodium_current_h1_gate (dimensionless)"
    legend_states[4] = "h2 in component sodium_current_h2_gate (dimensionless)"
    legend_algebraic[1] = "E0_m in component sodium_current_m_gate (millivolt)"
    legend_algebraic[16] = "alpha_m in component sodium_current_m_gate (per_second)"
    legend_algebraic[31] = "beta_m in component sodium_current_m_gate (per_second)"
    legend_algebraic[2] = "alpha_h in component sodium_current_h1_gate (per_second)"
    legend_algebraic[17] = "beta_h in component sodium_current_h1_gate (per_second)"
    legend_algebraic[32] = "h_infinity in component sodium_current_h1_gate (dimensionless)"
    legend_algebraic[42] = "tau_h1 in component sodium_current_h1_gate (second)"
    legend_algebraic[43] = "tau_h2 in component sodium_current_h2_gate (second)"
    legend_constants[13] = "g_Ca_L in component L_type_Ca_channel (nanoS)"
    legend_constants[14] = "E_Ca_app in component L_type_Ca_channel (millivolt)"
    legend_algebraic[41] = "d_prime in component L_type_Ca_channel (dimensionless)"
    legend_states[5] = "d_L in component L_type_Ca_channel_d_L_gate (dimensionless)"
    legend_states[6] = "f_L in component L_type_Ca_channel_f_L_gate (dimensionless)"
    legend_algebraic[3] = "E0_alpha_d_L in component L_type_Ca_channel_d_L_gate (millivolt)"
    legend_algebraic[18] = "E0_beta_d_L in component L_type_Ca_channel_d_L_gate (millivolt)"
    legend_algebraic[33] = "E10 in component L_type_Ca_channel_d_L_gate (millivolt)"
    legend_algebraic[44] = "alpha_d_L in component L_type_Ca_channel_d_L_gate (per_second)"
    legend_algebraic[53] = "beta_d_L in component L_type_Ca_channel_d_L_gate (per_second)"
    legend_algebraic[58] = "d_L_infinity in component L_type_Ca_channel_d_L_gate (dimensionless)"
    legend_algebraic[60] = "tau_d_L in component L_type_Ca_channel_d_L_gate (second)"
    legend_algebraic[4] = "E0_f_L in component L_type_Ca_channel_f_L_gate (millivolt)"
    legend_algebraic[19] = "alpha_f_L in component L_type_Ca_channel_f_L_gate (per_second)"
    legend_algebraic[34] = "beta_f_L in component L_type_Ca_channel_f_L_gate (per_second)"
    legend_algebraic[45] = "f_L_infinity in component L_type_Ca_channel_f_L_gate (dimensionless)"
    legend_algebraic[54] = "tau_f_L in component L_type_Ca_channel_f_L_gate (second)"
    legend_constants[15] = "g_Ca_T in component T_type_Ca_channel (nanoS)"
    legend_constants[16] = "E_Ca_T in component T_type_Ca_channel (millivolt)"
    legend_states[7] = "d_T in component T_type_Ca_channel_d_T_gate (dimensionless)"
    legend_states[8] = "f_T in component T_type_Ca_channel_f_T_gate (dimensionless)"
    legend_algebraic[5] = "E0_d_T in component T_type_Ca_channel_d_T_gate (millivolt)"
    legend_algebraic[20] = "alpha_d_T in component T_type_Ca_channel_d_T_gate (per_second)"
    legend_algebraic[35] = "beta_d_T in component T_type_Ca_channel_d_T_gate (per_second)"
    legend_algebraic[46] = "d_T_infinity in component T_type_Ca_channel_d_T_gate (dimensionless)"
    legend_algebraic[55] = "tau_d_T in component T_type_Ca_channel_d_T_gate (second)"
    legend_algebraic[6] = "E0_f_T in component T_type_Ca_channel_f_T_gate (millivolt)"
    legend_algebraic[21] = "alpha_f_T in component T_type_Ca_channel_f_T_gate (per_second)"
    legend_algebraic[36] = "beta_f_T in component T_type_Ca_channel_f_T_gate (per_second)"
    legend_algebraic[47] = "f_T_infinity in component T_type_Ca_channel_f_T_gate (dimensionless)"
    legend_algebraic[56] = "tau_f_T in component T_type_Ca_channel_f_T_gate (second)"
    legend_algebraic[59] = "E_K in component Ca_independent_transient_outward_K_current (millivolt)"
    legend_constants[17] = "g_to in component Ca_independent_transient_outward_K_current (nanoS)"
    legend_states[9] = "K_c in component cleft_space_ion_concentrations (millimolar)"
    legend_states[10] = "K_i in component intracellular_ion_concentrations (millimolar)"
    legend_states[11] = "r in component Ca_independent_transient_outward_K_current_r_gate (dimensionless)"
    legend_states[12] = "s1 in component Ca_independent_transient_outward_K_current_s1_gate (dimensionless)"
    legend_states[13] = "s2 in component Ca_independent_transient_outward_K_current_s2_gate (dimensionless)"
    legend_states[14] = "s3 in component Ca_independent_transient_outward_K_current_s3_gate (dimensionless)"
    legend_algebraic[7] = "alpha_r in component Ca_independent_transient_outward_K_current_r_gate (per_second)"
    legend_algebraic[22] = "beta_r in component Ca_independent_transient_outward_K_current_r_gate (per_second)"
    legend_algebraic[48] = "tau_r in component Ca_independent_transient_outward_K_current_r_gate (second)"
    legend_algebraic[37] = "r_infinity in component Ca_independent_transient_outward_K_current_r_gate (dimensionless)"
    legend_algebraic[23] = "tau_s1 in component Ca_independent_transient_outward_K_current_s1_gate (second)"
    legend_algebraic[8] = "s1_infinity in component Ca_independent_transient_outward_K_current_s1_gate (dimensionless)"
    legend_algebraic[24] = "tau_s2 in component Ca_independent_transient_outward_K_current_s2_gate (second)"
    legend_algebraic[9] = "s2_infinity in component Ca_independent_transient_outward_K_current_s2_gate (dimensionless)"
    legend_algebraic[25] = "tau_s3 in component Ca_independent_transient_outward_K_current_s3_gate (second)"
    legend_algebraic[10] = "s3_infinity in component Ca_independent_transient_outward_K_current_s3_gate (dimensionless)"
    legend_constants[18] = "g_Ks in component delayed_rectifier_K_current (nanoS)"
    legend_constants[19] = "g_Kr in component delayed_rectifier_K_current (nanoS)"
    legend_states[15] = "z in component delayed_rectifier_K_current_z_gate (dimensionless)"
    legend_states[16] = "p_a in component delayed_rectifier_K_current_pa_gate (dimensionless)"
    legend_states[17] = "p_i in component delayed_rectifier_K_current_pi_gate (dimensionless)"
    legend_algebraic[11] = "alpha_z in component delayed_rectifier_K_current_z_gate (per_second)"
    legend_algebraic[26] = "beta_z in component delayed_rectifier_K_current_z_gate (per_second)"
    legend_algebraic[49] = "tau_z in component delayed_rectifier_K_current_z_gate (second)"
    legend_algebraic[38] = "z_infinity in component delayed_rectifier_K_current_z_gate (dimensionless)"
    legend_algebraic[12] = "alpha_p_a in component delayed_rectifier_K_current_pa_gate (per_second)"
    legend_algebraic[27] = "beta_p_a in component delayed_rectifier_K_current_pa_gate (per_second)"
    legend_algebraic[50] = "tau_p_a in component delayed_rectifier_K_current_pa_gate (second)"
    legend_algebraic[39] = "p_a_infinity in component delayed_rectifier_K_current_pa_gate (dimensionless)"
    legend_algebraic[13] = "alpha_p_i in component delayed_rectifier_K_current_pi_gate (per_second)"
    legend_algebraic[28] = "beta_p_i in component delayed_rectifier_K_current_pi_gate (per_second)"
    legend_algebraic[51] = "tau_p_i in component delayed_rectifier_K_current_pi_gate (second)"
    legend_algebraic[40] = "p_i_infinity in component delayed_rectifier_K_current_pi_gate (dimensionless)"
    legend_constants[20] = "g_K1 in component inward_rectifier (nanoS)"
    legend_constants[21] = "KmK1 in component inward_rectifier (millimolar)"
    legend_constants[22] = "steepK1 in component inward_rectifier (dimensionless)"
    legend_constants[23] = "shiftK1 in component inward_rectifier (millivolt)"
    legend_constants[24] = "g_B_Na in component background_currents (nanoS)"
    legend_constants[25] = "g_B_Ca in component background_currents (nanoS)"
    legend_algebraic[66] = "E_Ca in component background_currents (millivolt)"
    legend_constants[26] = "Ca_c in component cleft_space_ion_concentrations (millimolar)"
    legend_states[18] = "Ca_i in component intracellular_ion_concentrations (millimolar)"
    legend_constants[27] = "k_NaK_K in component sodium_potassium_pump (millimolar)"
    legend_constants[28] = "k_NaK_Na in component sodium_potassium_pump (millimolar)"
    legend_constants[29] = "i_NaK_max in component sodium_potassium_pump (picoA)"
    legend_constants[30] = "i_CaP_max in component sarcolemmal_calcium_pump_current (picoA)"
    legend_constants[31] = "k_CaP in component sarcolemmal_calcium_pump_current (millimolar)"
    legend_constants[32] = "k_NaCa in component Na_Ca_ion_exchanger_current (picoA_per_millimolar_4)"
    legend_constants[33] = "d_NaCa in component Na_Ca_ion_exchanger_current (per_millimolar_4)"
    legend_constants[34] = "gamma in component Na_Ca_ion_exchanger_current (dimensionless)"
    legend_constants[35] = "Vol_i in component intracellular_ion_concentrations (nanolitre)"
    legend_constants[36] = "Vol_Ca in component intracellular_ion_concentrations (nanolitre)"
    legend_algebraic[75] = "i_up in component Ca_handling_by_the_SR (picoA)"
    legend_algebraic[77] = "i_rel in component Ca_handling_by_the_SR (picoA)"
    legend_algebraic[72] = "dOCdt in component intracellular_Ca_buffering (per_second)"
    legend_algebraic[73] = "dOTCdt in component intracellular_Ca_buffering (per_second)"
    legend_algebraic[74] = "dOTMgCdt in component intracellular_Ca_buffering (per_second)"
    legend_states[19] = "O_C in component intracellular_Ca_buffering (dimensionless)"
    legend_states[20] = "O_TC in component intracellular_Ca_buffering (dimensionless)"
    legend_states[21] = "O_TMgC in component intracellular_Ca_buffering (dimensionless)"
    legend_states[22] = "O_TMgMg in component intracellular_Ca_buffering (dimensionless)"
    legend_constants[37] = "Mg_i in component intracellular_Ca_buffering (millimolar)"
    legend_constants[38] = "Vol_c in component cleft_space_ion_concentrations (nanolitre)"
    legend_algebraic[76] = "i_tr in component Ca_handling_by_the_SR (picoA)"
    legend_constants[39] = "I_up_max in component Ca_handling_by_the_SR (picoA)"
    legend_constants[40] = "k_cyca in component Ca_handling_by_the_SR (millimolar)"
    legend_constants[41] = "k_srca in component Ca_handling_by_the_SR (millimolar)"
    legend_constants[42] = "k_xcs in component Ca_handling_by_the_SR (dimensionless)"
    legend_constants[43] = "alpha_rel in component Ca_handling_by_the_SR (picoA_per_millimolar)"
    legend_states[23] = "Ca_rel in component Ca_handling_by_the_SR (millimolar)"
    legend_states[24] = "Ca_up in component Ca_handling_by_the_SR (millimolar)"
    legend_constants[44] = "Vol_up in component Ca_handling_by_the_SR (nanolitre)"
    legend_constants[45] = "Vol_rel in component Ca_handling_by_the_SR (nanolitre)"
    legend_algebraic[14] = "r_act in component Ca_handling_by_the_SR (per_second)"
    legend_algebraic[29] = "r_inact in component Ca_handling_by_the_SR (per_second)"
    legend_states[25] = "O_Calse in component Ca_handling_by_the_SR (dimensionless)"
    legend_states[26] = "F1 in component Ca_handling_by_the_SR (dimensionless)"
    legend_states[27] = "F2 in component Ca_handling_by_the_SR (dimensionless)"
    legend_states[28] = "F3 in component Ca_handling_by_the_SR (dimensionless)"
    legend_constants[46] = "tau_tr in component Ca_handling_by_the_SR (second)"
    legend_constants[47] = "k_rel in component Ca_handling_by_the_SR (millimolar)"
    legend_constants[48] = "k_F3 in component Ca_handling_by_the_SR (per_second)"
    legend_rates[0] = "d/dt V in component membrane (millivolt)"
    legend_rates[2] = "d/dt m in component sodium_current_m_gate (dimensionless)"
    legend_rates[3] = "d/dt h1 in component sodium_current_h1_gate (dimensionless)"
    legend_rates[4] = "d/dt h2 in component sodium_current_h2_gate (dimensionless)"
    legend_rates[5] = "d/dt d_L in component L_type_Ca_channel_d_L_gate (dimensionless)"
    legend_rates[6] = "d/dt f_L in component L_type_Ca_channel_f_L_gate (dimensionless)"
    legend_rates[7] = "d/dt d_T in component T_type_Ca_channel_d_T_gate (dimensionless)"
    legend_rates[8] = "d/dt f_T in component T_type_Ca_channel_f_T_gate (dimensionless)"
    legend_rates[11] = "d/dt r in component Ca_independent_transient_outward_K_current_r_gate (dimensionless)"
    legend_rates[12] = "d/dt s1 in component Ca_independent_transient_outward_K_current_s1_gate (dimensionless)"
    legend_rates[13] = "d/dt s2 in component Ca_independent_transient_outward_K_current_s2_gate (dimensionless)"
    legend_rates[14] = "d/dt s3 in component Ca_independent_transient_outward_K_current_s3_gate (dimensionless)"
    legend_rates[15] = "d/dt z in component delayed_rectifier_K_current_z_gate (dimensionless)"
    legend_rates[16] = "d/dt p_a in component delayed_rectifier_K_current_pa_gate (dimensionless)"
    legend_rates[17] = "d/dt p_i in component delayed_rectifier_K_current_pi_gate (dimensionless)"
    legend_rates[1] = "d/dt Na_i in component intracellular_ion_concentrations (millimolar)"
    legend_rates[10] = "d/dt K_i in component intracellular_ion_concentrations (millimolar)"
    legend_rates[18] = "d/dt Ca_i in component intracellular_ion_concentrations (millimolar)"
    legend_rates[19] = "d/dt O_C in component intracellular_Ca_buffering (dimensionless)"
    legend_rates[20] = "d/dt O_TC in component intracellular_Ca_buffering (dimensionless)"
    legend_rates[21] = "d/dt O_TMgC in component intracellular_Ca_buffering (dimensionless)"
    legend_rates[22] = "d/dt O_TMgMg in component intracellular_Ca_buffering (dimensionless)"
    legend_rates[9] = "d/dt K_c in component cleft_space_ion_concentrations (millimolar)"
    legend_rates[25] = "d/dt O_Calse in component Ca_handling_by_the_SR (dimensionless)"
    legend_rates[23] = "d/dt Ca_rel in component Ca_handling_by_the_SR (millimolar)"
    legend_rates[24] = "d/dt Ca_up in component Ca_handling_by_the_SR (millimolar)"
    legend_rates[26] = "d/dt F1 in component Ca_handling_by_the_SR (dimensionless)"
    legend_rates[27] = "d/dt F2 in component Ca_handling_by_the_SR (dimensionless)"
    legend_rates[28] = "d/dt F3 in component Ca_handling_by_the_SR (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 1
    constants[1] = 0
    states[0] = -80
    constants[2] = 8314
    constants[3] = 308
    constants[4] = 96487
    constants[5] = 0.00005
    constants[6] = 0.01
    constants[7] = 100
    constants[8] = 0.5
    constants[9] = 0.0002
    constants[10] = -20
    constants[11] = 0.0000014
    constants[12] = 140
    states[1] = 8.4
    states[2] = 0.01309
    states[3] = 0.706
    states[4] = 0.61493
    constants[13] = 0.004
    constants[14] = 50
    states[5] = 0.00003
    states[6] = 0.99981
    constants[15] = 0.006
    constants[16] = 38
    states[7] = 0.00046
    states[8] = 0.30752
    constants[17] = 0.050002
    states[9] = 5
    states[10] = 100
    states[11] = 0.00006
    states[12] = 0.5753
    states[13] = 0.39871
    states[14] = 0.57363
    constants[18] = 0.0025
    constants[19] = 0.0035
    states[15] = 0.02032
    states[16] = 0.00016
    states[17] = 0.76898
    constants[20] = 0.00508
    constants[21] = 0.59
    constants[22] = 1.393
    constants[23] = -3.6
    constants[24] = 6.4e-5
    constants[25] = 3.1e-5
    constants[26] = 2.5
    states[18] = 0.000071
    constants[27] = 1
    constants[28] = 11
    constants[29] = 0.06441
    constants[30] = 0.009509
    constants[31] = 2e-4
    constants[32] = 2e-5
    constants[33] = 3e-4
    constants[34] = 0.45
    constants[35] = 1.26e-5
    constants[36] = 5.884e-6
    states[19] = 0.029108
    states[20] = 0.014071
    states[21] = 0.214036
    states[22] = 0.693565
    constants[37] = 2.5
    constants[38] = 0.0000025
    constants[39] = 2.8
    constants[40] = 0.0003
    constants[41] = 0.5
    constants[42] = 0.4
    constants[43] = 200
    states[23] = 0.726776
    states[24] = 0.730866
    constants[44] = 3.969e-7
    constants[45] = 4.4e-8
    states[25] = 0.465921
    states[26] = 0.288039
    states[27] = 0.002262
    states[28] = 0.612697
    constants[46] = 0.01
    constants[47] = 0.0003
    constants[48] = 0.815
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[22] = 2000.00*constants[37]*((1.00000-states[21])-states[22])-666.000*states[22]
    algebraic[14] = 240.000*exp(0.0800000*(states[0]-20.0000))+203.800*(power(states[18]/(states[18]+constants[47]), 4.00000))
    rates[26] = constants[48]*states[28]-algebraic[14]*states[26]
    algebraic[23] = 0.546600/(1.00000+exp((states[0]+32.8000)/0.100000))+0.0204000
    algebraic[8] = 1.00000/(1.00000+exp((states[0]+28.2900)/7.06000))
    rates[12] = (algebraic[8]-states[12])/algebraic[23]
    algebraic[24] = 5.75000/(1.00000+exp((states[0]+32.8000)/0.100000))+0.450000/(1.00000+exp((states[0]-13.5400)/-13.9700))
    algebraic[9] = 1.00000/(1.00000+exp((states[0]+28.2900)/7.06000))
    rates[13] = (algebraic[9]-states[13])/algebraic[24]
    algebraic[25] = 7.50000/(1.00000+exp((states[0]+23.0000)/0.500000))+0.500000
    algebraic[10] = (1.00000/(1.00000+exp((states[0]+50.6700)/27.3800))+0.666000)/1.66600
    rates[14] = (algebraic[10]-states[14])/algebraic[25]
    algebraic[29] = 33.9600+339.600*(power(states[18]/(states[18]+constants[47]), 4.00000))
    rates[27] = algebraic[14]*states[26]-algebraic[29]*states[27]
    rates[28] = states[27]*algebraic[29]-constants[48]*states[28]
    algebraic[1] = states[0]+44.4000
    algebraic[16] = (-460.000*algebraic[1])/(exp(algebraic[1]/-12.6730)-1.00000)
    algebraic[31] = 18400.0*exp(algebraic[1]/-12.6730)
    rates[2] = algebraic[16]*(1.00000-states[2])-algebraic[31]*states[2]
    algebraic[2] = 44.9000*exp((states[0]+66.9000)/-5.57000)
    algebraic[17] = 1491.00/(1.00000+323.300*exp((states[0]+94.6000)/-12.9000))
    algebraic[32] = algebraic[2]/(algebraic[2]+algebraic[17])
    algebraic[42] = 0.0300000/(1.00000+exp((states[0]+40.0000)/6.00000))+0.000150000
    rates[3] = (algebraic[32]-states[3])/algebraic[42]
    algebraic[43] = 0.120000/(1.00000+exp((states[0]+60.0000)/2.00000))+0.000450000
    rates[4] = (algebraic[32]-states[4])/algebraic[43]
    algebraic[7] = 386.600*exp(states[0]/12.0000)
    algebraic[22] = 8.01100*exp(states[0]/-7.20000)
    algebraic[48] = 1.00000/(algebraic[7]+algebraic[22])+0.000400000
    algebraic[37] = 1.00000/(1.00000+exp((states[0]+15.0000)/-5.63300))
    rates[11] = (algebraic[37]-states[11])/algebraic[48]
    algebraic[11] = 1.66000*exp(states[0]/69.4520)
    algebraic[26] = 0.300000*exp(states[0]/-21.8260)
    algebraic[49] = 1.00000/(algebraic[11]+algebraic[26])+0.0600000
    algebraic[38] = 1.00000/(1.00000+exp((states[0]-0.900000)/-13.8000))
    rates[15] = (algebraic[38]-states[15])/algebraic[49]
    algebraic[12] = 9.00000*exp(states[0]/25.3710)
    algebraic[27] = 1.30000*exp(states[0]/-13.0260)
    algebraic[50] = 1.00000/(algebraic[12]+algebraic[27])
    algebraic[39] = 1.00000/(1.00000+exp((states[0]+5.10000)/-7.40000))
    rates[16] = (algebraic[39]-states[16])/algebraic[50]
    algebraic[13] = 100.000*exp(states[0]/-54.6450)
    algebraic[28] = 656.000*exp(states[0]/106.157)
    algebraic[51] = 1.00000/(algebraic[13]+algebraic[28])
    algebraic[40] = 1.00000/(1.00000+exp((states[0]+47.3921)/18.6603))
    rates[17] = (algebraic[40]-states[17])/algebraic[51]
    algebraic[4] = states[0]+18.0000
    algebraic[19] = (8.49000*algebraic[4])/(exp(algebraic[4]/4.00000)-1.00000)
    algebraic[34] = 67.9220/(1.00000+exp(algebraic[4]/-4.00000))
    algebraic[45] = algebraic[19]/(algebraic[19]+algebraic[34])
    algebraic[54] = 1.00000/(algebraic[19]+algebraic[34])
    rates[6] = (algebraic[45]-states[6])/algebraic[54]
    algebraic[5] = states[0]+23.3000
    algebraic[46] = 1.00000/(1.00000+exp((algebraic[5]-0.300000)/-6.10000))
    algebraic[20] = 674.173*exp(algebraic[5]/30.0000)
    algebraic[35] = 674.173*exp(algebraic[5]/-30.0000)
    algebraic[55] = 1.00000/(algebraic[20]+algebraic[35])
    rates[7] = (algebraic[46]-states[7])/algebraic[55]
    algebraic[6] = states[0]+75.0000
    algebraic[21] = 9.63700*exp(algebraic[6]/-83.3000)
    algebraic[36] = 9.63700*exp(algebraic[6]/15.3800)
    algebraic[47] = algebraic[21]/(algebraic[21]+algebraic[36])
    algebraic[56] = 1.00000/(algebraic[21]+algebraic[36])
    rates[8] = (algebraic[47]-states[8])/algebraic[56]
    algebraic[33] = states[0]+10.0000
    algebraic[58] = 1.00000/(1.00000+exp((algebraic[33]+0.950000)/-6.60000))
    algebraic[3] = states[0]+45.0000
    algebraic[44] = (-16.7200*algebraic[3])/(exp(algebraic[3]/-2.50000)-1.00000)+(-50.0000*algebraic[33])/(exp(algebraic[33]/-4.80800)-1.00000)
    algebraic[18] = states[0]+5.00000
    algebraic[53] = (4.48000*algebraic[18])/(exp(algebraic[18]/2.50000)-1.00000)
    algebraic[60] = 1.00000/(algebraic[44]+algebraic[53])
    rates[5] = (algebraic[58]-states[5])/algebraic[60]
    algebraic[59] = ((constants[2]*constants[3])/constants[4])*log(states[9]/states[10])
    algebraic[61] = custom_piecewise([equal(constants[0] , 1.00000) & equal(constants[1] , 0.00000), 0.200000*constants[17]*states[11]*(0.590000*(power(states[12], 3.00000))+0.410000*(power(states[13], 3.00000)))*(0.600000*(power(states[14], 6.00000))+0.400000)*(states[0]-algebraic[59]) , equal(constants[0] , 0.00000) & equal(constants[1] , 1.00000), 0.350000*constants[17]*states[11]*(0.590000*(power(states[12], 3.00000))+0.410000*(power(states[13], 3.00000)))*(0.600000*(power(states[14], 6.00000))+0.400000)*(states[0]-algebraic[59]) , True, constants[17]*states[11]*(0.590000*(power(states[12], 3.00000))+0.410000*(power(states[13], 3.00000)))*(0.600000*(power(states[14], 6.00000))+0.400000)*(states[0]-algebraic[59])])
    algebraic[62] = custom_piecewise([equal(constants[0] , 1.00000) & equal(constants[1] , 0.00000), 0.00140000*(states[0]+70.0000) , equal(constants[0] , 0.00000) & equal(constants[1] , 1.00000), 0.00240000*(states[0]+70.0000) , True, 0.00100000*(states[0]+70.0000)])
    algebraic[65] = custom_piecewise([equal(constants[0] , 1.00000) & equal(constants[1] , 0.00000), (2.00000*constants[20]*(states[0]-algebraic[59])*(power(states[9]/(states[9]+constants[21]), 3.00000))*1.00000)/(1.00000+exp((constants[22]*constants[4]*((states[0]-algebraic[59])-constants[23]))/(constants[2]*constants[3]))) , equal(constants[0] , 0.00000) & equal(constants[1] , 1.00000), (2.50000*constants[20]*(states[0]-algebraic[59])*(power(states[9]/(states[9]+constants[21]), 3.00000))*1.00000)/(1.00000+exp((constants[22]*constants[4]*((states[0]-algebraic[59])-constants[23]))/(constants[2]*constants[3]))) , True, (constants[20]*(states[0]-algebraic[59])*(power(states[9]/(states[9]+constants[21]), 3.00000))*1.00000)/(1.00000+exp((constants[22]*constants[4]*((states[0]-algebraic[59])-constants[23]))/(constants[2]*constants[3])))])
    algebraic[64] = constants[19]*states[16]*states[17]*(states[0]-algebraic[59])
    algebraic[63] = constants[18]*states[15]*(states[0]-algebraic[59])
    algebraic[69] = (((((constants[29]*states[9])/(states[9]+constants[27]))*(power(states[1], 1.50000)))/(power(states[1], 1.50000)+power(constants[28], 1.50000)))*1.60000)/(1.50000+exp((states[0]+60.0000)/-40.0000))
    rates[10] = -((algebraic[61]+algebraic[62]+algebraic[65]+algebraic[64]+algebraic[63])-2.00000*algebraic[69])/(constants[35]*constants[4])
    rates[9] = ((algebraic[61]+algebraic[62]+algebraic[65]+algebraic[64]+algebraic[63])-2.00000*algebraic[69])/(constants[38]*constants[4])
    algebraic[15] = ((constants[2]*constants[3])/constants[4])*log(constants[12]/states[1])
    algebraic[30] = (((constants[11]*(power(states[2], 3.00000))*(0.635000*states[3]+0.365000*states[4])*constants[12]*states[0]*(power(constants[4], 2.00000)))/(constants[2]*constants[3]))*(exp(((states[0]-algebraic[15])*constants[4])/(constants[2]*constants[3]))-1.00000))/(exp((states[0]*constants[4])/(constants[2]*constants[3]))-1.00000)
    algebraic[41] = 1.00000/(1.00000+exp((states[0]-23.0000)/-12.0000))
    algebraic[52] = custom_piecewise([equal(constants[0] , 1.00000) & equal(constants[1] , 0.00000), 1.80000*constants[13]*(states[5]*states[6]+algebraic[41])*(states[0]-constants[14]) , equal(constants[0] , 0.00000) & equal(constants[1] , 1.00000), 2.10000*constants[13]*(states[5]*states[6]+algebraic[41])*(states[0]-constants[14]) , True, constants[13]*(states[5]*states[6]+algebraic[41])*(states[0]-constants[14])])
    algebraic[57] = constants[15]*states[7]*states[8]*(states[0]-constants[16])
    algebraic[67] = custom_piecewise([equal(constants[0] , 1.00000) & equal(constants[1] , 0.00000), 2.00000e-05*(states[0]-algebraic[15]) , equal(constants[0] , 0.00000) & equal(constants[1] , 1.00000), 3.00000e-05*(states[0]-algebraic[15]) , True, constants[24]*(states[0]-algebraic[15])])
    algebraic[66] = ((constants[2]*constants[3])/(2.00000*constants[4]))*log(constants[26]/states[18])
    algebraic[68] = custom_piecewise([equal(constants[0] , 1.00000) & equal(constants[1] , 0.00000), 2.00000e-05*(states[0]-algebraic[66]) , equal(constants[0] , 0.00000) & equal(constants[1] , 1.00000), 3.00000e-05*(states[0]-algebraic[66]) , True, constants[25]*(states[0]-algebraic[66])])
    algebraic[70] = (constants[30]*states[18])/(states[18]+constants[31])
    algebraic[71] = (constants[32]*((power(states[1], 3.00000))*constants[26]*exp((constants[34]*constants[4]*states[0])/(constants[2]*constants[3]))-(power(constants[12], 3.00000))*states[18]*exp(((constants[34]-1.00000)*states[0]*constants[4])/(constants[2]*constants[3]))))/(1.00000+constants[33]*((power(constants[12], 3.00000))*states[18]+(power(states[1], 3.00000))*constants[26]))
    algebraic[0] = custom_piecewise([greater_equal(voi , constants[6]) & less_equal(voi , constants[7]) & less_equal((voi-constants[6])-floor((voi-constants[6])/constants[8])*constants[8] , constants[9]), constants[10] , True, 0.00000])
    rates[0] = (-1.00000/constants[5])*(algebraic[64]+algebraic[63]+algebraic[30]+algebraic[52]+algebraic[57]+algebraic[61]+algebraic[62]+algebraic[65]+algebraic[67]+algebraic[68]+algebraic[69]+algebraic[70]+algebraic[71]+algebraic[0])
    rates[1] = -(algebraic[30]+algebraic[67]+3.00000*algebraic[69]+3.00000*algebraic[71])/(constants[35]*constants[4])
    algebraic[72] = 200000.*states[18]*(1.00000-states[19])-476.000*states[19]
    rates[19] = algebraic[72]
    algebraic[73] = 78400.0*states[18]*(1.00000-states[20])-392.000*states[20]
    rates[20] = algebraic[73]
    rates[25] = 480.000*states[23]*(1.00000-states[25])-400.000*states[25]
    algebraic[74] = 200000.*states[18]*((1.00000-states[21])-states[22])-6.60000*states[21]
    rates[21] = algebraic[74]
    algebraic[75] = (constants[39]*(states[18]/constants[40]-((power(constants[42], 2.00000))*states[24])/constants[41]))/((states[18]+constants[40])/constants[40]+(constants[42]*(states[24]+constants[41]))/constants[41])
    algebraic[76] = ((states[24]-states[23])*2.00000*constants[4]*constants[45])/constants[46]
    rates[24] = (algebraic[75]-algebraic[76])/(2.00000*constants[44]*constants[4])
    algebraic[77] = constants[43]*(power(states[27]/(states[27]+0.250000), 2.00000))*(states[23]-states[18])
    rates[18] = -((((algebraic[52]+algebraic[57]+algebraic[68]+algebraic[70])-2.00000*algebraic[71])+algebraic[75])-algebraic[77])/(2.00000*constants[36]*constants[4])-(0.0800000*algebraic[73]+0.160000*algebraic[74]+0.0450000*algebraic[72])
    rates[23] = (algebraic[76]-algebraic[77])/(2.00000*constants[45]*constants[4])-31.0000*rates[25]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[14] = 240.000*exp(0.0800000*(states[0]-20.0000))+203.800*(power(states[18]/(states[18]+constants[47]), 4.00000))
    algebraic[23] = 0.546600/(1.00000+exp((states[0]+32.8000)/0.100000))+0.0204000
    algebraic[8] = 1.00000/(1.00000+exp((states[0]+28.2900)/7.06000))
    algebraic[24] = 5.75000/(1.00000+exp((states[0]+32.8000)/0.100000))+0.450000/(1.00000+exp((states[0]-13.5400)/-13.9700))
    algebraic[9] = 1.00000/(1.00000+exp((states[0]+28.2900)/7.06000))
    algebraic[25] = 7.50000/(1.00000+exp((states[0]+23.0000)/0.500000))+0.500000
    algebraic[10] = (1.00000/(1.00000+exp((states[0]+50.6700)/27.3800))+0.666000)/1.66600
    algebraic[29] = 33.9600+339.600*(power(states[18]/(states[18]+constants[47]), 4.00000))
    algebraic[1] = states[0]+44.4000
    algebraic[16] = (-460.000*algebraic[1])/(exp(algebraic[1]/-12.6730)-1.00000)
    algebraic[31] = 18400.0*exp(algebraic[1]/-12.6730)
    algebraic[2] = 44.9000*exp((states[0]+66.9000)/-5.57000)
    algebraic[17] = 1491.00/(1.00000+323.300*exp((states[0]+94.6000)/-12.9000))
    algebraic[32] = algebraic[2]/(algebraic[2]+algebraic[17])
    algebraic[42] = 0.0300000/(1.00000+exp((states[0]+40.0000)/6.00000))+0.000150000
    algebraic[43] = 0.120000/(1.00000+exp((states[0]+60.0000)/2.00000))+0.000450000
    algebraic[7] = 386.600*exp(states[0]/12.0000)
    algebraic[22] = 8.01100*exp(states[0]/-7.20000)
    algebraic[48] = 1.00000/(algebraic[7]+algebraic[22])+0.000400000
    algebraic[37] = 1.00000/(1.00000+exp((states[0]+15.0000)/-5.63300))
    algebraic[11] = 1.66000*exp(states[0]/69.4520)
    algebraic[26] = 0.300000*exp(states[0]/-21.8260)
    algebraic[49] = 1.00000/(algebraic[11]+algebraic[26])+0.0600000
    algebraic[38] = 1.00000/(1.00000+exp((states[0]-0.900000)/-13.8000))
    algebraic[12] = 9.00000*exp(states[0]/25.3710)
    algebraic[27] = 1.30000*exp(states[0]/-13.0260)
    algebraic[50] = 1.00000/(algebraic[12]+algebraic[27])
    algebraic[39] = 1.00000/(1.00000+exp((states[0]+5.10000)/-7.40000))
    algebraic[13] = 100.000*exp(states[0]/-54.6450)
    algebraic[28] = 656.000*exp(states[0]/106.157)
    algebraic[51] = 1.00000/(algebraic[13]+algebraic[28])
    algebraic[40] = 1.00000/(1.00000+exp((states[0]+47.3921)/18.6603))
    algebraic[4] = states[0]+18.0000
    algebraic[19] = (8.49000*algebraic[4])/(exp(algebraic[4]/4.00000)-1.00000)
    algebraic[34] = 67.9220/(1.00000+exp(algebraic[4]/-4.00000))
    algebraic[45] = algebraic[19]/(algebraic[19]+algebraic[34])
    algebraic[54] = 1.00000/(algebraic[19]+algebraic[34])
    algebraic[5] = states[0]+23.3000
    algebraic[46] = 1.00000/(1.00000+exp((algebraic[5]-0.300000)/-6.10000))
    algebraic[20] = 674.173*exp(algebraic[5]/30.0000)
    algebraic[35] = 674.173*exp(algebraic[5]/-30.0000)
    algebraic[55] = 1.00000/(algebraic[20]+algebraic[35])
    algebraic[6] = states[0]+75.0000
    algebraic[21] = 9.63700*exp(algebraic[6]/-83.3000)
    algebraic[36] = 9.63700*exp(algebraic[6]/15.3800)
    algebraic[47] = algebraic[21]/(algebraic[21]+algebraic[36])
    algebraic[56] = 1.00000/(algebraic[21]+algebraic[36])
    algebraic[33] = states[0]+10.0000
    algebraic[58] = 1.00000/(1.00000+exp((algebraic[33]+0.950000)/-6.60000))
    algebraic[3] = states[0]+45.0000
    algebraic[44] = (-16.7200*algebraic[3])/(exp(algebraic[3]/-2.50000)-1.00000)+(-50.0000*algebraic[33])/(exp(algebraic[33]/-4.80800)-1.00000)
    algebraic[18] = states[0]+5.00000
    algebraic[53] = (4.48000*algebraic[18])/(exp(algebraic[18]/2.50000)-1.00000)
    algebraic[60] = 1.00000/(algebraic[44]+algebraic[53])
    algebraic[59] = ((constants[2]*constants[3])/constants[4])*log(states[9]/states[10])
    algebraic[61] = custom_piecewise([equal(constants[0] , 1.00000) & equal(constants[1] , 0.00000), 0.200000*constants[17]*states[11]*(0.590000*(power(states[12], 3.00000))+0.410000*(power(states[13], 3.00000)))*(0.600000*(power(states[14], 6.00000))+0.400000)*(states[0]-algebraic[59]) , equal(constants[0] , 0.00000) & equal(constants[1] , 1.00000), 0.350000*constants[17]*states[11]*(0.590000*(power(states[12], 3.00000))+0.410000*(power(states[13], 3.00000)))*(0.600000*(power(states[14], 6.00000))+0.400000)*(states[0]-algebraic[59]) , True, constants[17]*states[11]*(0.590000*(power(states[12], 3.00000))+0.410000*(power(states[13], 3.00000)))*(0.600000*(power(states[14], 6.00000))+0.400000)*(states[0]-algebraic[59])])
    algebraic[62] = custom_piecewise([equal(constants[0] , 1.00000) & equal(constants[1] , 0.00000), 0.00140000*(states[0]+70.0000) , equal(constants[0] , 0.00000) & equal(constants[1] , 1.00000), 0.00240000*(states[0]+70.0000) , True, 0.00100000*(states[0]+70.0000)])
    algebraic[65] = custom_piecewise([equal(constants[0] , 1.00000) & equal(constants[1] , 0.00000), (2.00000*constants[20]*(states[0]-algebraic[59])*(power(states[9]/(states[9]+constants[21]), 3.00000))*1.00000)/(1.00000+exp((constants[22]*constants[4]*((states[0]-algebraic[59])-constants[23]))/(constants[2]*constants[3]))) , equal(constants[0] , 0.00000) & equal(constants[1] , 1.00000), (2.50000*constants[20]*(states[0]-algebraic[59])*(power(states[9]/(states[9]+constants[21]), 3.00000))*1.00000)/(1.00000+exp((constants[22]*constants[4]*((states[0]-algebraic[59])-constants[23]))/(constants[2]*constants[3]))) , True, (constants[20]*(states[0]-algebraic[59])*(power(states[9]/(states[9]+constants[21]), 3.00000))*1.00000)/(1.00000+exp((constants[22]*constants[4]*((states[0]-algebraic[59])-constants[23]))/(constants[2]*constants[3])))])
    algebraic[64] = constants[19]*states[16]*states[17]*(states[0]-algebraic[59])
    algebraic[63] = constants[18]*states[15]*(states[0]-algebraic[59])
    algebraic[69] = (((((constants[29]*states[9])/(states[9]+constants[27]))*(power(states[1], 1.50000)))/(power(states[1], 1.50000)+power(constants[28], 1.50000)))*1.60000)/(1.50000+exp((states[0]+60.0000)/-40.0000))
    algebraic[15] = ((constants[2]*constants[3])/constants[4])*log(constants[12]/states[1])
    algebraic[30] = (((constants[11]*(power(states[2], 3.00000))*(0.635000*states[3]+0.365000*states[4])*constants[12]*states[0]*(power(constants[4], 2.00000)))/(constants[2]*constants[3]))*(exp(((states[0]-algebraic[15])*constants[4])/(constants[2]*constants[3]))-1.00000))/(exp((states[0]*constants[4])/(constants[2]*constants[3]))-1.00000)
    algebraic[41] = 1.00000/(1.00000+exp((states[0]-23.0000)/-12.0000))
    algebraic[52] = custom_piecewise([equal(constants[0] , 1.00000) & equal(constants[1] , 0.00000), 1.80000*constants[13]*(states[5]*states[6]+algebraic[41])*(states[0]-constants[14]) , equal(constants[0] , 0.00000) & equal(constants[1] , 1.00000), 2.10000*constants[13]*(states[5]*states[6]+algebraic[41])*(states[0]-constants[14]) , True, constants[13]*(states[5]*states[6]+algebraic[41])*(states[0]-constants[14])])
    algebraic[57] = constants[15]*states[7]*states[8]*(states[0]-constants[16])
    algebraic[67] = custom_piecewise([equal(constants[0] , 1.00000) & equal(constants[1] , 0.00000), 2.00000e-05*(states[0]-algebraic[15]) , equal(constants[0] , 0.00000) & equal(constants[1] , 1.00000), 3.00000e-05*(states[0]-algebraic[15]) , True, constants[24]*(states[0]-algebraic[15])])
    algebraic[66] = ((constants[2]*constants[3])/(2.00000*constants[4]))*log(constants[26]/states[18])
    algebraic[68] = custom_piecewise([equal(constants[0] , 1.00000) & equal(constants[1] , 0.00000), 2.00000e-05*(states[0]-algebraic[66]) , equal(constants[0] , 0.00000) & equal(constants[1] , 1.00000), 3.00000e-05*(states[0]-algebraic[66]) , True, constants[25]*(states[0]-algebraic[66])])
    algebraic[70] = (constants[30]*states[18])/(states[18]+constants[31])
    algebraic[71] = (constants[32]*((power(states[1], 3.00000))*constants[26]*exp((constants[34]*constants[4]*states[0])/(constants[2]*constants[3]))-(power(constants[12], 3.00000))*states[18]*exp(((constants[34]-1.00000)*states[0]*constants[4])/(constants[2]*constants[3]))))/(1.00000+constants[33]*((power(constants[12], 3.00000))*states[18]+(power(states[1], 3.00000))*constants[26]))
    algebraic[0] = custom_piecewise([greater_equal(voi , constants[6]) & less_equal(voi , constants[7]) & less_equal((voi-constants[6])-floor((voi-constants[6])/constants[8])*constants[8] , constants[9]), constants[10] , True, 0.00000])
    algebraic[72] = 200000.*states[18]*(1.00000-states[19])-476.000*states[19]
    algebraic[73] = 78400.0*states[18]*(1.00000-states[20])-392.000*states[20]
    algebraic[74] = 200000.*states[18]*((1.00000-states[21])-states[22])-6.60000*states[21]
    algebraic[75] = (constants[39]*(states[18]/constants[40]-((power(constants[42], 2.00000))*states[24])/constants[41]))/((states[18]+constants[40])/constants[40]+(constants[42]*(states[24]+constants[41]))/constants[41])
    algebraic[76] = ((states[24]-states[23])*2.00000*constants[4]*constants[45])/constants[46]
    algebraic[77] = constants[43]*(power(states[27]/(states[27]+0.250000), 2.00000))*(states[23]-states[18])
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