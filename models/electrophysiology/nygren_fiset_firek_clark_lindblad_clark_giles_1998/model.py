# Size of variable arrays:
sizeAlgebraic = 49
sizeStates = 29
sizeConstants = 51
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
    legend_constants[3] = "Cm in component membrane (nanoF)"
    legend_algebraic[26] = "i_Na in component sodium_current (picoA)"
    legend_algebraic[28] = "i_Ca_L in component L_type_Ca_channel (picoA)"
    legend_algebraic[30] = "i_t in component Ca_independent_transient_outward_K_current (picoA)"
    legend_algebraic[31] = "i_sus in component sustained_outward_K_current (picoA)"
    legend_algebraic[35] = "i_K1 in component inward_rectifier (picoA)"
    legend_algebraic[34] = "i_Kr in component delayed_rectifier_K_currents (picoA)"
    legend_algebraic[32] = "i_Ks in component delayed_rectifier_K_currents (picoA)"
    legend_algebraic[36] = "i_B_Na in component background_currents (picoA)"
    legend_algebraic[38] = "i_B_Ca in component background_currents (picoA)"
    legend_algebraic[39] = "i_NaK in component sodium_potassium_pump (picoA)"
    legend_algebraic[40] = "i_CaP in component sarcolemmal_calcium_pump_current (picoA)"
    legend_algebraic[41] = "i_NaCa in component Na_Ca_ion_exchanger_current (picoA)"
    legend_algebraic[0] = "i_Stim in component membrane (picoA)"
    legend_constants[4] = "stim_start in component membrane (second)"
    legend_constants[5] = "stim_end in component membrane (second)"
    legend_constants[6] = "stim_period in component membrane (second)"
    legend_constants[7] = "stim_duration in component membrane (second)"
    legend_constants[8] = "stim_amplitude in component membrane (picoA)"
    legend_algebraic[12] = "E_Na in component sodium_current (millivolt)"
    legend_constants[9] = "P_Na in component sodium_current (nanolitre_per_second)"
    legend_states[1] = "Na_c in component cleft_space_ion_concentrations (millimolar)"
    legend_states[2] = "Na_i in component intracellular_ion_concentrations (millimolar)"
    legend_states[3] = "m in component sodium_current_m_gate (dimensionless)"
    legend_states[4] = "h1 in component sodium_current_h1_gate (dimensionless)"
    legend_states[5] = "h2 in component sodium_current_h2_gate (dimensionless)"
    legend_algebraic[1] = "m_infinity in component sodium_current_m_gate (dimensionless)"
    legend_algebraic[13] = "tau_m in component sodium_current_m_gate (second)"
    legend_algebraic[2] = "h_infinity in component sodium_current_h1_gate (dimensionless)"
    legend_algebraic[14] = "tau_h1 in component sodium_current_h1_gate (second)"
    legend_algebraic[15] = "tau_h2 in component sodium_current_h2_gate (second)"
    legend_constants[10] = "g_Ca_L in component L_type_Ca_channel (nanoS)"
    legend_constants[11] = "E_Ca_app in component L_type_Ca_channel (millivolt)"
    legend_algebraic[27] = "f_Ca in component L_type_Ca_channel (dimensionless)"
    legend_constants[12] = "k_Ca in component L_type_Ca_channel (millimolar)"
    legend_states[6] = "Ca_d in component intracellular_ion_concentrations (millimolar)"
    legend_states[7] = "d_L in component L_type_Ca_channel_d_L_gate (dimensionless)"
    legend_states[8] = "f_L_1 in component L_type_Ca_channel_f_L1_gate (dimensionless)"
    legend_states[9] = "f_L_2 in component L_type_Ca_channel_f_L2_gate (dimensionless)"
    legend_algebraic[3] = "d_L_infinity in component L_type_Ca_channel_d_L_gate (dimensionless)"
    legend_algebraic[16] = "tau_d_L in component L_type_Ca_channel_d_L_gate (second)"
    legend_algebraic[4] = "f_L_infinity in component L_type_Ca_channel_f_L1_gate (dimensionless)"
    legend_algebraic[17] = "tau_f_L1 in component L_type_Ca_channel_f_L1_gate (second)"
    legend_algebraic[18] = "tau_f_L2 in component L_type_Ca_channel_f_L2_gate (second)"
    legend_algebraic[29] = "E_K in component Ca_independent_transient_outward_K_current (millivolt)"
    legend_constants[13] = "g_t in component Ca_independent_transient_outward_K_current (nanoS)"
    legend_states[10] = "K_c in component cleft_space_ion_concentrations (millimolar)"
    legend_states[11] = "K_i in component intracellular_ion_concentrations (millimolar)"
    legend_states[12] = "r in component Ca_independent_transient_outward_K_current_r_gate (dimensionless)"
    legend_states[13] = "s in component Ca_independent_transient_outward_K_current_s_gate (dimensionless)"
    legend_algebraic[19] = "tau_r in component Ca_independent_transient_outward_K_current_r_gate (second)"
    legend_algebraic[5] = "r_infinity in component Ca_independent_transient_outward_K_current_r_gate (dimensionless)"
    legend_algebraic[20] = "tau_s in component Ca_independent_transient_outward_K_current_s_gate (second)"
    legend_algebraic[6] = "s_infinity in component Ca_independent_transient_outward_K_current_s_gate (dimensionless)"
    legend_constants[14] = "g_sus in component sustained_outward_K_current (nanoS)"
    legend_states[14] = "r_sus in component sustained_outward_K_current_r_sus_gate (dimensionless)"
    legend_states[15] = "s_sus in component sustained_outward_K_current_s_sus_gate (dimensionless)"
    legend_algebraic[21] = "tau_r_sus in component sustained_outward_K_current_r_sus_gate (second)"
    legend_algebraic[7] = "r_sus_infinity in component sustained_outward_K_current_r_sus_gate (dimensionless)"
    legend_algebraic[22] = "tau_s_sus in component sustained_outward_K_current_s_sus_gate (second)"
    legend_algebraic[8] = "s_sus_infinity in component sustained_outward_K_current_s_sus_gate (dimensionless)"
    legend_constants[15] = "g_Ks in component delayed_rectifier_K_currents (nanoS)"
    legend_constants[16] = "g_Kr in component delayed_rectifier_K_currents (nanoS)"
    legend_states[16] = "n in component delayed_rectifier_K_currents_n_gate (dimensionless)"
    legend_states[17] = "p_a in component delayed_rectifier_K_currents_pa_gate (dimensionless)"
    legend_algebraic[33] = "p_i in component delayed_rectifier_K_currents_pi_gate (dimensionless)"
    legend_algebraic[23] = "tau_n in component delayed_rectifier_K_currents_n_gate (second)"
    legend_algebraic[9] = "n_infinity in component delayed_rectifier_K_currents_n_gate (dimensionless)"
    legend_algebraic[24] = "tau_p_a in component delayed_rectifier_K_currents_pa_gate (second)"
    legend_algebraic[10] = "p_a_infinity in component delayed_rectifier_K_currents_pa_gate (dimensionless)"
    legend_constants[17] = "g_K1 in component inward_rectifier (nanoS)"
    legend_constants[18] = "g_B_Na in component background_currents (nanoS)"
    legend_constants[19] = "g_B_Ca in component background_currents (nanoS)"
    legend_algebraic[37] = "E_Ca in component background_currents (millivolt)"
    legend_states[18] = "Ca_c in component cleft_space_ion_concentrations (millimolar)"
    legend_states[19] = "Ca_i in component intracellular_ion_concentrations (millimolar)"
    legend_constants[20] = "k_NaK_K in component sodium_potassium_pump (millimolar)"
    legend_constants[21] = "k_NaK_Na in component sodium_potassium_pump (millimolar)"
    legend_constants[22] = "i_NaK_max in component sodium_potassium_pump (picoA)"
    legend_constants[23] = "i_CaP_max in component sarcolemmal_calcium_pump_current (picoA)"
    legend_constants[24] = "k_CaP in component sarcolemmal_calcium_pump_current (millimolar)"
    legend_constants[25] = "k_NaCa in component Na_Ca_ion_exchanger_current (picoA_per_millimolar_4)"
    legend_constants[26] = "d_NaCa in component Na_Ca_ion_exchanger_current (per_millimolar_4)"
    legend_constants[27] = "gamma in component Na_Ca_ion_exchanger_current (dimensionless)"
    legend_constants[28] = "phi_Na_en in component intracellular_ion_concentrations (picoA)"
    legend_constants[29] = "Vol_i in component intracellular_ion_concentrations (nanolitre)"
    legend_constants[49] = "Vol_d in component intracellular_ion_concentrations (nanolitre)"
    legend_algebraic[42] = "i_di in component intracellular_ion_concentrations (picoA)"
    legend_constants[30] = "tau_di in component intracellular_ion_concentrations (second)"
    legend_algebraic[46] = "i_up in component Ca_handling_by_the_SR (picoA)"
    legend_algebraic[48] = "i_rel in component Ca_handling_by_the_SR (picoA)"
    legend_algebraic[43] = "dOCdt in component intracellular_Ca_buffering (per_second)"
    legend_algebraic[44] = "dOTCdt in component intracellular_Ca_buffering (per_second)"
    legend_algebraic[45] = "dOTMgCdt in component intracellular_Ca_buffering (per_second)"
    legend_states[20] = "O_C in component intracellular_Ca_buffering (dimensionless)"
    legend_states[21] = "O_TC in component intracellular_Ca_buffering (dimensionless)"
    legend_states[22] = "O_TMgC in component intracellular_Ca_buffering (dimensionless)"
    legend_states[23] = "O_TMgMg in component intracellular_Ca_buffering (dimensionless)"
    legend_constants[31] = "Mg_i in component intracellular_Ca_buffering (millimolar)"
    legend_constants[50] = "Vol_c in component cleft_space_ion_concentrations (nanolitre)"
    legend_constants[32] = "tau_Na in component cleft_space_ion_concentrations (second)"
    legend_constants[33] = "tau_K in component cleft_space_ion_concentrations (second)"
    legend_constants[34] = "tau_Ca in component cleft_space_ion_concentrations (second)"
    legend_constants[35] = "Na_b in component cleft_space_ion_concentrations (millimolar)"
    legend_constants[36] = "Ca_b in component cleft_space_ion_concentrations (millimolar)"
    legend_constants[37] = "K_b in component cleft_space_ion_concentrations (millimolar)"
    legend_algebraic[47] = "i_tr in component Ca_handling_by_the_SR (picoA)"
    legend_constants[38] = "I_up_max in component Ca_handling_by_the_SR (picoA)"
    legend_constants[39] = "k_cyca in component Ca_handling_by_the_SR (millimolar)"
    legend_constants[40] = "k_srca in component Ca_handling_by_the_SR (millimolar)"
    legend_constants[41] = "k_xcs in component Ca_handling_by_the_SR (dimensionless)"
    legend_constants[42] = "alpha_rel in component Ca_handling_by_the_SR (picoA_per_millimolar)"
    legend_states[24] = "Ca_rel in component Ca_handling_by_the_SR (millimolar)"
    legend_states[25] = "Ca_up in component Ca_handling_by_the_SR (millimolar)"
    legend_constants[43] = "Vol_up in component Ca_handling_by_the_SR (nanolitre)"
    legend_constants[44] = "Vol_rel in component Ca_handling_by_the_SR (nanolitre)"
    legend_algebraic[11] = "r_act in component Ca_handling_by_the_SR (per_second)"
    legend_algebraic[25] = "r_inact in component Ca_handling_by_the_SR (per_second)"
    legend_constants[45] = "r_recov in component Ca_handling_by_the_SR (per_second)"
    legend_states[26] = "O_Calse in component Ca_handling_by_the_SR (dimensionless)"
    legend_states[27] = "F1 in component Ca_handling_by_the_SR (dimensionless)"
    legend_states[28] = "F2 in component Ca_handling_by_the_SR (dimensionless)"
    legend_constants[46] = "tau_tr in component Ca_handling_by_the_SR (second)"
    legend_constants[47] = "k_rel_i in component Ca_handling_by_the_SR (millimolar)"
    legend_constants[48] = "k_rel_d in component Ca_handling_by_the_SR (millimolar)"
    legend_rates[0] = "d/dt V in component membrane (millivolt)"
    legend_rates[3] = "d/dt m in component sodium_current_m_gate (dimensionless)"
    legend_rates[4] = "d/dt h1 in component sodium_current_h1_gate (dimensionless)"
    legend_rates[5] = "d/dt h2 in component sodium_current_h2_gate (dimensionless)"
    legend_rates[7] = "d/dt d_L in component L_type_Ca_channel_d_L_gate (dimensionless)"
    legend_rates[8] = "d/dt f_L_1 in component L_type_Ca_channel_f_L1_gate (dimensionless)"
    legend_rates[9] = "d/dt f_L_2 in component L_type_Ca_channel_f_L2_gate (dimensionless)"
    legend_rates[12] = "d/dt r in component Ca_independent_transient_outward_K_current_r_gate (dimensionless)"
    legend_rates[13] = "d/dt s in component Ca_independent_transient_outward_K_current_s_gate (dimensionless)"
    legend_rates[14] = "d/dt r_sus in component sustained_outward_K_current_r_sus_gate (dimensionless)"
    legend_rates[15] = "d/dt s_sus in component sustained_outward_K_current_s_sus_gate (dimensionless)"
    legend_rates[16] = "d/dt n in component delayed_rectifier_K_currents_n_gate (dimensionless)"
    legend_rates[17] = "d/dt p_a in component delayed_rectifier_K_currents_pa_gate (dimensionless)"
    legend_rates[2] = "d/dt Na_i in component intracellular_ion_concentrations (millimolar)"
    legend_rates[11] = "d/dt K_i in component intracellular_ion_concentrations (millimolar)"
    legend_rates[19] = "d/dt Ca_i in component intracellular_ion_concentrations (millimolar)"
    legend_rates[6] = "d/dt Ca_d in component intracellular_ion_concentrations (millimolar)"
    legend_rates[20] = "d/dt O_C in component intracellular_Ca_buffering (dimensionless)"
    legend_rates[21] = "d/dt O_TC in component intracellular_Ca_buffering (dimensionless)"
    legend_rates[22] = "d/dt O_TMgC in component intracellular_Ca_buffering (dimensionless)"
    legend_rates[23] = "d/dt O_TMgMg in component intracellular_Ca_buffering (dimensionless)"
    legend_rates[1] = "d/dt Na_c in component cleft_space_ion_concentrations (millimolar)"
    legend_rates[10] = "d/dt K_c in component cleft_space_ion_concentrations (millimolar)"
    legend_rates[18] = "d/dt Ca_c in component cleft_space_ion_concentrations (millimolar)"
    legend_rates[26] = "d/dt O_Calse in component Ca_handling_by_the_SR (dimensionless)"
    legend_rates[24] = "d/dt Ca_rel in component Ca_handling_by_the_SR (millimolar)"
    legend_rates[25] = "d/dt Ca_up in component Ca_handling_by_the_SR (millimolar)"
    legend_rates[27] = "d/dt F1 in component Ca_handling_by_the_SR (dimensionless)"
    legend_rates[28] = "d/dt F2 in component Ca_handling_by_the_SR (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = -74.2525
    constants[0] = 8314
    constants[1] = 306.15
    constants[2] = 96487
    constants[3] = 0.05
    constants[4] = 0.1
    constants[5] = 100000000
    constants[6] = 1
    constants[7] = 0.006
    constants[8] = -280
    constants[9] = 0.0016
    states[1] = 130.011
    states[2] = 8.5547
    states[3] = 0.0032017
    states[4] = 0.8814
    states[5] = 0.8742
    constants[10] = 6.75
    constants[11] = 60
    constants[12] = 0.025
    states[6] = 7.2495e-5
    states[7] = 1.3005e-5
    states[8] = 0.9986
    states[9] = 0.9986
    constants[13] = 7.5
    states[10] = 5.3581
    states[11] = 129.435
    states[12] = 0.0010678
    states[13] = 0.949
    constants[14] = 2.75
    states[14] = 0.00015949
    states[15] = 0.9912
    constants[15] = 1
    constants[16] = 0.5
    states[16] = 0.0048357
    states[17] = 0.0001
    constants[17] = 3
    constants[18] = 0.060599
    constants[19] = 0.078681
    states[18] = 1.8147
    states[19] = 6.729e-5
    constants[20] = 1
    constants[21] = 11
    constants[22] = 70.8253
    constants[23] = 4
    constants[24] = 0.0002
    constants[25] = 0.0374842
    constants[26] = 0.0003
    constants[27] = 0.45
    constants[28] = -1.68
    constants[29] = 0.005884
    constants[30] = 0.01
    states[20] = 0.0275
    states[21] = 0.0133
    states[22] = 0.1961
    states[23] = 0.7094
    constants[31] = 2.5
    constants[32] = 14.3
    constants[33] = 10
    constants[34] = 24.7
    constants[35] = 130
    constants[36] = 1.8
    constants[37] = 5.4
    constants[38] = 2800
    constants[39] = 0.0003
    constants[40] = 0.5
    constants[41] = 0.4
    constants[42] = 200000
    states[24] = 0.6465
    states[25] = 0.6646
    constants[43] = 0.0003969
    constants[44] = 4.41e-5
    constants[45] = 0.815
    states[26] = 0.4369
    states[27] = 0.4284
    states[28] = 0.0028
    constants[46] = 0.01
    constants[47] = 0.0003
    constants[48] = 0.003
    constants[49] = 0.0200000*constants[29]
    constants[50] = 0.136000*constants[29]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[23] = 2000.00*constants[31]*((1.00000-states[22])-states[23])-666.000*states[23]
    algebraic[11] = 203.800*(power(states[19]/(states[19]+constants[47]), 4.00000)+power(states[6]/(states[6]+constants[48]), 4.00000))
    rates[27] = constants[45]*((1.00000-states[27])-states[28])-algebraic[11]*states[27]
    algebraic[1] = 1.00000/(1.00000+exp((states[0]+27.1200)/-8.21000))
    algebraic[13] = 4.20000e-05*exp(-(power((states[0]+25.5700)/28.8000, 2.00000)))+2.40000e-05
    rates[3] = (algebraic[1]-states[3])/algebraic[13]
    algebraic[2] = 1.00000/(1.00000+exp((states[0]+63.6000)/5.30000))
    algebraic[14] = 0.0300000/(1.00000+exp((states[0]+35.1000)/3.20000))+0.000300000
    rates[4] = (algebraic[2]-states[4])/algebraic[14]
    algebraic[15] = 0.120000/(1.00000+exp((states[0]+35.1000)/3.20000))+0.00300000
    rates[5] = (algebraic[2]-states[5])/algebraic[15]
    algebraic[3] = 1.00000/(1.00000+exp((states[0]+9.00000)/-5.80000))
    algebraic[16] = 0.00270000*exp(-(power((states[0]+35.0000)/30.0000, 2.00000)))+0.00200000
    rates[7] = (algebraic[3]-states[7])/algebraic[16]
    algebraic[4] = 1.00000/(1.00000+exp((states[0]+27.4000)/7.10000))
    algebraic[17] = 0.161000*exp(-(power((states[0]+40.0000)/14.4000, 2.00000)))+0.0100000
    rates[8] = (algebraic[4]-states[8])/algebraic[17]
    algebraic[18] = 1.33230*exp(-(power((states[0]+40.0000)/14.2000, 2.00000)))+0.0626000
    rates[9] = (algebraic[4]-states[9])/algebraic[18]
    algebraic[19] = 0.00350000*exp(-(power(states[0]/30.0000, 2.00000)))+0.00150000
    algebraic[5] = 1.00000/(1.00000+exp((states[0]-1.00000)/-11.0000))
    rates[12] = (algebraic[5]-states[12])/algebraic[19]
    algebraic[20] = 0.481200*exp(-(power((states[0]+52.4500)/14.9700, 2.00000)))+0.0141400
    algebraic[6] = 1.00000/(1.00000+exp((states[0]+40.5000)/11.5000))
    rates[13] = (algebraic[6]-states[13])/algebraic[20]
    algebraic[21] = 0.00900000/(1.00000+exp((states[0]+5.00000)/12.0000))+0.000500000
    algebraic[7] = 1.00000/(1.00000+exp((states[0]+4.30000)/-8.00000))
    rates[14] = (algebraic[7]-states[14])/algebraic[21]
    algebraic[22] = 0.0470000/(1.00000+exp((states[0]+60.0000)/10.0000))+0.300000
    algebraic[8] = 0.400000/(1.00000+exp((states[0]+20.0000)/10.0000))+0.600000
    rates[15] = (algebraic[8]-states[15])/algebraic[22]
    algebraic[23] = 0.700000+0.400000*exp(-(power((states[0]-20.0000)/20.0000, 2.00000)))
    algebraic[9] = 1.00000/(1.00000+exp((states[0]-19.9000)/-12.7000))
    rates[16] = (algebraic[9]-states[16])/algebraic[23]
    algebraic[24] = 0.0311800+0.217180*exp(-(power((states[0]+20.1376)/22.1996, 2.00000)))
    algebraic[10] = 1.00000/(1.00000+exp((states[0]+15.0000)/-6.00000))
    rates[17] = (algebraic[10]-states[17])/algebraic[24]
    algebraic[25] = 33.9600+339.600*(power(states[19]/(states[19]+constants[47]), 4.00000))
    rates[28] = algebraic[11]*states[27]-algebraic[25]*states[28]
    algebraic[29] = ((constants[0]*constants[1])/constants[2])*log(states[10]/states[11])
    algebraic[30] = constants[13]*states[12]*states[13]*(states[0]-algebraic[29])
    algebraic[31] = constants[14]*states[14]*states[15]*(states[0]-algebraic[29])
    algebraic[35] = (constants[17]*(power(states[10]/1.00000, 0.445700))*(states[0]-algebraic[29]))/(1.00000+exp((1.50000*((states[0]-algebraic[29])+3.60000)*constants[2])/(constants[0]*constants[1])))
    algebraic[33] = 1.00000/(1.00000+exp((states[0]+55.0000)/24.0000))
    algebraic[34] = constants[16]*states[17]*algebraic[33]*(states[0]-algebraic[29])
    algebraic[32] = constants[15]*states[16]*(states[0]-algebraic[29])
    algebraic[39] = (((((constants[22]*states[10])/(states[10]+constants[20]))*(power(states[2], 1.50000)))/(power(states[2], 1.50000)+power(constants[21], 1.50000)))*(states[0]+150.000))/(states[0]+200.000)
    rates[11] = -((algebraic[30]+algebraic[31]+algebraic[35]+algebraic[34]+algebraic[32])-2.00000*algebraic[39])/(constants[29]*constants[2])
    rates[10] = (constants[37]-states[10])/constants[33]+((algebraic[30]+algebraic[31]+algebraic[35]+algebraic[34]+algebraic[32])-2.00000*algebraic[39])/(constants[50]*constants[2])
    algebraic[12] = ((constants[0]*constants[1])/constants[2])*log(states[1]/states[2])
    algebraic[26] = (((constants[9]*(power(states[3], 3.00000))*(0.900000*states[4]+0.100000*states[5])*states[1]*states[0]*(power(constants[2], 2.00000)))/(constants[0]*constants[1]))*(exp(((states[0]-algebraic[12])*constants[2])/(constants[0]*constants[1]))-1.00000))/(exp((states[0]*constants[2])/(constants[0]*constants[1]))-1.00000)
    algebraic[27] = states[6]/(states[6]+constants[12])
    algebraic[28] = constants[10]*states[7]*(algebraic[27]*states[8]+(1.00000-algebraic[27])*states[9])*(states[0]-constants[11])
    algebraic[36] = constants[18]*(states[0]-algebraic[12])
    algebraic[37] = ((constants[0]*constants[1])/(2.00000*constants[2]))*log(states[18]/states[19])
    algebraic[38] = constants[19]*(states[0]-algebraic[37])
    algebraic[40] = (constants[23]*states[19])/(states[19]+constants[24])
    algebraic[41] = (constants[25]*((power(states[2], 3.00000))*states[18]*exp((constants[27]*constants[2]*states[0])/(constants[0]*constants[1]))-(power(states[1], 3.00000))*states[19]*exp(((constants[27]-1.00000)*states[0]*constants[2])/(constants[0]*constants[1]))))/(1.00000+constants[26]*((power(states[1], 3.00000))*states[19]+(power(states[2], 3.00000))*states[18]))
    algebraic[0] = custom_piecewise([greater_equal(voi , constants[4]) & less_equal(voi , constants[5]) & less_equal((voi-constants[4])-floor((voi-constants[4])/constants[6])*constants[6] , constants[7]), constants[8] , True, 0.00000])
    rates[0] = (-1.00000/constants[3])*(algebraic[0]+algebraic[26]+algebraic[28]+algebraic[30]+algebraic[31]+algebraic[35]+algebraic[34]+algebraic[32]+algebraic[36]+algebraic[38]+algebraic[39]+algebraic[40]+algebraic[41])
    rates[2] = -(algebraic[26]+algebraic[36]+3.00000*algebraic[39]+3.00000*algebraic[41]+constants[28])/(constants[29]*constants[2])
    rates[1] = (constants[35]-states[1])/constants[32]+(algebraic[26]+algebraic[36]+3.00000*algebraic[39]+3.00000*algebraic[41]+constants[28])/(constants[50]*constants[2])
    rates[18] = (constants[36]-states[18])/constants[34]+((algebraic[28]+algebraic[38]+algebraic[40])-2.00000*algebraic[41])/(2.00000*constants[50]*constants[2])
    rates[20] = 200000.*states[19]*(1.00000-states[20])-476.000*states[20]
    algebraic[42] = ((states[6]-states[19])*2.00000*constants[2]*constants[49])/constants[30]
    rates[6] = -(algebraic[28]+algebraic[42])/(2.00000*constants[49]*constants[2])
    rates[21] = 78400.0*states[19]*(1.00000-states[21])-392.000*states[21]
    rates[22] = 200000.*states[19]*((1.00000-states[22])-states[23])-6.60000*states[22]
    algebraic[46] = (constants[38]*(states[19]/constants[39]-((power(constants[41], 2.00000))*states[25])/constants[40]))/((states[19]+constants[39])/constants[39]+(constants[41]*(states[25]+constants[40]))/constants[40])
    algebraic[47] = ((states[25]-states[24])*2.00000*constants[2]*constants[44])/constants[46]
    rates[25] = (algebraic[46]-algebraic[47])/(2.00000*constants[43]*constants[2])
    algebraic[48] = constants[42]*(power(states[28]/(states[28]+0.250000), 2.00000))*(states[24]-states[19])
    algebraic[43] = rates[20]
    algebraic[44] = rates[21]
    algebraic[45] = rates[22]
    rates[19] = -((((-algebraic[42]+algebraic[38]+algebraic[40])-2.00000*algebraic[41])+algebraic[46])-algebraic[48])/(2.00000*constants[29]*constants[2])-(0.0800000*algebraic[44]+0.160000*algebraic[45]+0.0450000*algebraic[43])
    rates[26] = 480.000*states[24]*(1.00000-states[26])-400.000*states[26]
    rates[24] = (algebraic[47]-algebraic[48])/(2.00000*constants[44]*constants[2])-31.0000*rates[26]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[11] = 203.800*(power(states[19]/(states[19]+constants[47]), 4.00000)+power(states[6]/(states[6]+constants[48]), 4.00000))
    algebraic[1] = 1.00000/(1.00000+exp((states[0]+27.1200)/-8.21000))
    algebraic[13] = 4.20000e-05*exp(-(power((states[0]+25.5700)/28.8000, 2.00000)))+2.40000e-05
    algebraic[2] = 1.00000/(1.00000+exp((states[0]+63.6000)/5.30000))
    algebraic[14] = 0.0300000/(1.00000+exp((states[0]+35.1000)/3.20000))+0.000300000
    algebraic[15] = 0.120000/(1.00000+exp((states[0]+35.1000)/3.20000))+0.00300000
    algebraic[3] = 1.00000/(1.00000+exp((states[0]+9.00000)/-5.80000))
    algebraic[16] = 0.00270000*exp(-(power((states[0]+35.0000)/30.0000, 2.00000)))+0.00200000
    algebraic[4] = 1.00000/(1.00000+exp((states[0]+27.4000)/7.10000))
    algebraic[17] = 0.161000*exp(-(power((states[0]+40.0000)/14.4000, 2.00000)))+0.0100000
    algebraic[18] = 1.33230*exp(-(power((states[0]+40.0000)/14.2000, 2.00000)))+0.0626000
    algebraic[19] = 0.00350000*exp(-(power(states[0]/30.0000, 2.00000)))+0.00150000
    algebraic[5] = 1.00000/(1.00000+exp((states[0]-1.00000)/-11.0000))
    algebraic[20] = 0.481200*exp(-(power((states[0]+52.4500)/14.9700, 2.00000)))+0.0141400
    algebraic[6] = 1.00000/(1.00000+exp((states[0]+40.5000)/11.5000))
    algebraic[21] = 0.00900000/(1.00000+exp((states[0]+5.00000)/12.0000))+0.000500000
    algebraic[7] = 1.00000/(1.00000+exp((states[0]+4.30000)/-8.00000))
    algebraic[22] = 0.0470000/(1.00000+exp((states[0]+60.0000)/10.0000))+0.300000
    algebraic[8] = 0.400000/(1.00000+exp((states[0]+20.0000)/10.0000))+0.600000
    algebraic[23] = 0.700000+0.400000*exp(-(power((states[0]-20.0000)/20.0000, 2.00000)))
    algebraic[9] = 1.00000/(1.00000+exp((states[0]-19.9000)/-12.7000))
    algebraic[24] = 0.0311800+0.217180*exp(-(power((states[0]+20.1376)/22.1996, 2.00000)))
    algebraic[10] = 1.00000/(1.00000+exp((states[0]+15.0000)/-6.00000))
    algebraic[25] = 33.9600+339.600*(power(states[19]/(states[19]+constants[47]), 4.00000))
    algebraic[29] = ((constants[0]*constants[1])/constants[2])*log(states[10]/states[11])
    algebraic[30] = constants[13]*states[12]*states[13]*(states[0]-algebraic[29])
    algebraic[31] = constants[14]*states[14]*states[15]*(states[0]-algebraic[29])
    algebraic[35] = (constants[17]*(power(states[10]/1.00000, 0.445700))*(states[0]-algebraic[29]))/(1.00000+exp((1.50000*((states[0]-algebraic[29])+3.60000)*constants[2])/(constants[0]*constants[1])))
    algebraic[33] = 1.00000/(1.00000+exp((states[0]+55.0000)/24.0000))
    algebraic[34] = constants[16]*states[17]*algebraic[33]*(states[0]-algebraic[29])
    algebraic[32] = constants[15]*states[16]*(states[0]-algebraic[29])
    algebraic[39] = (((((constants[22]*states[10])/(states[10]+constants[20]))*(power(states[2], 1.50000)))/(power(states[2], 1.50000)+power(constants[21], 1.50000)))*(states[0]+150.000))/(states[0]+200.000)
    algebraic[12] = ((constants[0]*constants[1])/constants[2])*log(states[1]/states[2])
    algebraic[26] = (((constants[9]*(power(states[3], 3.00000))*(0.900000*states[4]+0.100000*states[5])*states[1]*states[0]*(power(constants[2], 2.00000)))/(constants[0]*constants[1]))*(exp(((states[0]-algebraic[12])*constants[2])/(constants[0]*constants[1]))-1.00000))/(exp((states[0]*constants[2])/(constants[0]*constants[1]))-1.00000)
    algebraic[27] = states[6]/(states[6]+constants[12])
    algebraic[28] = constants[10]*states[7]*(algebraic[27]*states[8]+(1.00000-algebraic[27])*states[9])*(states[0]-constants[11])
    algebraic[36] = constants[18]*(states[0]-algebraic[12])
    algebraic[37] = ((constants[0]*constants[1])/(2.00000*constants[2]))*log(states[18]/states[19])
    algebraic[38] = constants[19]*(states[0]-algebraic[37])
    algebraic[40] = (constants[23]*states[19])/(states[19]+constants[24])
    algebraic[41] = (constants[25]*((power(states[2], 3.00000))*states[18]*exp((constants[27]*constants[2]*states[0])/(constants[0]*constants[1]))-(power(states[1], 3.00000))*states[19]*exp(((constants[27]-1.00000)*states[0]*constants[2])/(constants[0]*constants[1]))))/(1.00000+constants[26]*((power(states[1], 3.00000))*states[19]+(power(states[2], 3.00000))*states[18]))
    algebraic[0] = custom_piecewise([greater_equal(voi , constants[4]) & less_equal(voi , constants[5]) & less_equal((voi-constants[4])-floor((voi-constants[4])/constants[6])*constants[6] , constants[7]), constants[8] , True, 0.00000])
    algebraic[42] = ((states[6]-states[19])*2.00000*constants[2]*constants[49])/constants[30]
    algebraic[46] = (constants[38]*(states[19]/constants[39]-((power(constants[41], 2.00000))*states[25])/constants[40]))/((states[19]+constants[39])/constants[39]+(constants[41]*(states[25]+constants[40]))/constants[40])
    algebraic[47] = ((states[25]-states[24])*2.00000*constants[2]*constants[44])/constants[46]
    algebraic[48] = constants[42]*(power(states[28]/(states[28]+0.250000), 2.00000))*(states[24]-states[19])
    algebraic[43] = rates[20]
    algebraic[44] = rates[21]
    algebraic[45] = rates[22]
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