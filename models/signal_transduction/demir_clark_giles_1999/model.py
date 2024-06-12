# Size of variable arrays:
sizeAlgebraic = 73
sizeStates = 29
sizeConstants = 55
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
    legend_algebraic[63] = "i_Na in component sodium_current (nanoA)"
    legend_algebraic[40] = "i_Ca_T in component T_type_Ca_channel (nanoA)"
    legend_algebraic[39] = "i_Ca_L in component L_type_Ca_channel (nanoA)"
    legend_algebraic[66] = "i_K in component delayed_rectifying_potassium_current (nanoA)"
    legend_algebraic[45] = "i_f in component hyperpolarisation_activated_current (nanoA)"
    legend_algebraic[72] = "i_B in component linear_background_current (nanoA)"
    legend_algebraic[47] = "i_NaK in component sodium_potassium_pump (nanoA)"
    legend_algebraic[49] = "i_NaCa in component sodium_calcium_pump (nanoA)"
    legend_algebraic[48] = "i_Ca_P in component calcium_pump_current (nanoA)"
    legend_algebraic[69] = "i_K_ACh in component muscarinic_potassium_current (nanoA)"
    legend_constants[4] = "P_Na in component sodium_current (mul_per_second)"
    legend_algebraic[62] = "E_Na in component reversal_potentials (millivolt)"
    legend_states[1] = "Na_c in component cleft_space_equations (millimolar)"
    legend_constants[45] = "F_ACh_Na in component sodium_current (dimensionless)"
    legend_constants[5] = "ACh in component cAMP_balance (millimolar)"
    legend_states[2] = "m in component sodium_current_m_gate (dimensionless)"
    legend_states[3] = "h1 in component sodium_current_h_gate (dimensionless)"
    legend_states[4] = "h2 in component sodium_current_h_gate (dimensionless)"
    legend_algebraic[22] = "m_infinity in component sodium_current_m_gate (dimensionless)"
    legend_algebraic[29] = "tau_m in component sodium_current_m_gate (second)"
    legend_algebraic[0] = "alpha_m in component sodium_current_m_gate (per_second)"
    legend_algebraic[12] = "beta_m in component sodium_current_m_gate (per_second)"
    legend_algebraic[23] = "h1_infinity in component sodium_current_h_gate (dimensionless)"
    legend_algebraic[35] = "h2_infinity in component sodium_current_h_gate (dimensionless)"
    legend_algebraic[30] = "tau_h1 in component sodium_current_h_gate (second)"
    legend_algebraic[37] = "tau_h2 in component sodium_current_h_gate (second)"
    legend_algebraic[1] = "alpha_h1 in component sodium_current_h_gate (per_second)"
    legend_algebraic[13] = "beta_h1 in component sodium_current_h_gate (per_second)"
    legend_algebraic[20] = "g_Ca_L in component L_type_Ca_channel (microS)"
    legend_constants[6] = "g_Ca_L_cont in component L_type_Ca_channel (microS)"
    legend_algebraic[9] = "F_cAMP_CaL in component L_type_Ca_channel (dimensionless)"
    legend_states[5] = "cAMP in component cAMP_balance (millimolar)"
    legend_constants[7] = "E_Ca_L in component L_type_Ca_channel (millivolt)"
    legend_states[6] = "d_L in component L_type_Ca_channel_d_gate (dimensionless)"
    legend_algebraic[38] = "d_L_infinity in component L_type_Ca_channel_d_gate (dimensionless)"
    legend_states[7] = "f_L in component L_type_Ca_channel_f_gate (dimensionless)"
    legend_algebraic[28] = "alpha_d_L in component L_type_Ca_channel_d_gate (per_second)"
    legend_algebraic[34] = "beta_d_L in component L_type_Ca_channel_d_gate (per_second)"
    legend_algebraic[36] = "tau_d_L in component L_type_Ca_channel_d_gate (second)"
    legend_algebraic[2] = "alpha_f_L in component L_type_Ca_channel_f_gate (per_second)"
    legend_algebraic[14] = "beta_f_L in component L_type_Ca_channel_f_gate (per_second)"
    legend_algebraic[31] = "f_L_infinity in component L_type_Ca_channel_f_gate (dimensionless)"
    legend_algebraic[24] = "tau_f_L in component L_type_Ca_channel_f_gate (second)"
    legend_constants[8] = "g_Ca_T in component T_type_Ca_channel (microS)"
    legend_constants[9] = "E_Ca_T in component T_type_Ca_channel (millivolt)"
    legend_states[8] = "d_T in component T_type_Ca_channel_d_gate (dimensionless)"
    legend_states[9] = "f_T in component T_type_Ca_channel_f_gate (dimensionless)"
    legend_algebraic[3] = "alpha_d_T in component T_type_Ca_channel_d_gate (per_second)"
    legend_algebraic[15] = "beta_d_T in component T_type_Ca_channel_d_gate (per_second)"
    legend_algebraic[32] = "d_T_infinity in component T_type_Ca_channel_d_gate (dimensionless)"
    legend_algebraic[25] = "tau_d_T in component T_type_Ca_channel_d_gate (second)"
    legend_algebraic[4] = "alpha_f_T in component T_type_Ca_channel_f_gate (per_second)"
    legend_algebraic[16] = "beta_f_T in component T_type_Ca_channel_f_gate (per_second)"
    legend_algebraic[33] = "f_T_infinity in component T_type_Ca_channel_f_gate (dimensionless)"
    legend_algebraic[26] = "tau_f_T in component T_type_Ca_channel_f_gate (second)"
    legend_algebraic[42] = "g_K in component delayed_rectifying_potassium_current (microS)"
    legend_algebraic[41] = "F_cAMP_K in component delayed_rectifying_potassium_current (dimensionless)"
    legend_algebraic[65] = "E_K in component reversal_potentials (millivolt)"
    legend_constants[10] = "K_b in component cleft_space_equations (millimolar)"
    legend_states[10] = "P_a in component delayed_rectifying_potassium_current_P_a_gate (dimensionless)"
    legend_states[11] = "P_i in component delayed_rectifying_potassium_current_P_i_gate (dimensionless)"
    legend_algebraic[17] = "tau_P_a in component delayed_rectifying_potassium_current_P_a_gate (second)"
    legend_algebraic[5] = "P_a_infinity in component delayed_rectifying_potassium_current_P_a_gate (dimensionless)"
    legend_algebraic[6] = "alpha_P_i in component delayed_rectifying_potassium_current_P_i_gate (per_second)"
    legend_algebraic[18] = "beta_P_i in component delayed_rectifying_potassium_current_P_i_gate (per_second)"
    legend_algebraic[64] = "i_B_Na in component linear_background_current (nanoA)"
    legend_algebraic[71] = "i_B_Ca in component linear_background_current (nanoA)"
    legend_algebraic[67] = "i_B_K in component linear_background_current (nanoA)"
    legend_constants[11] = "g_B_Na in component linear_background_current (microS)"
    legend_constants[12] = "g_B_Ca in component linear_background_current (microS)"
    legend_constants[13] = "g_B_K in component linear_background_current (microS)"
    legend_algebraic[70] = "E_Ca in component reversal_potentials (millivolt)"
    legend_constants[46] = "F_ACh_bNa in component linear_background_current (dimensionless)"
    legend_algebraic[43] = "i_f_Na in component hyperpolarisation_activated_current (nanoA)"
    legend_algebraic[44] = "i_f_K in component hyperpolarisation_activated_current (nanoA)"
    legend_constants[14] = "g_f_Na in component hyperpolarisation_activated_current (microS)"
    legend_constants[15] = "g_f_K in component hyperpolarisation_activated_current (microS)"
    legend_states[12] = "y in component hyperpolarisation_activated_current_y_gate (dimensionless)"
    legend_algebraic[19] = "y_infinity in component hyperpolarisation_activated_current_y_gate (dimensionless)"
    legend_algebraic[7] = "V_half in component hyperpolarisation_activated_current_y_gate (millivolt)"
    legend_algebraic[27] = "tau_y in component hyperpolarisation_activated_current_y_gate (second)"
    legend_constants[16] = "K_m_Na in component sodium_potassium_pump (millimolar)"
    legend_constants[17] = "K_m_K in component sodium_potassium_pump (millimolar)"
    legend_constants[18] = "i_NaK_max in component sodium_potassium_pump (nanoA)"
    legend_algebraic[46] = "F_cAMP_NaK in component sodium_potassium_pump (dimensionless)"
    legend_states[13] = "Na_i in component intracellular_concentrations_and_buffer_equations (millimolar)"
    legend_states[14] = "K_c in component cleft_space_equations (millimolar)"
    legend_states[15] = "Ca_i in component intracellular_concentrations_and_buffer_equations (millimolar)"
    legend_constants[19] = "i_Ca_P_max in component calcium_pump_current (nanoA)"
    legend_constants[20] = "K_NaCa in component sodium_calcium_pump (nanoA)"
    legend_constants[21] = "d_NaCa in component sodium_calcium_pump (dimensionless)"
    legend_constants[22] = "gamma in component sodium_calcium_pump (dimensionless)"
    legend_states[16] = "Ca_c in component cleft_space_equations (millimolar)"
    legend_algebraic[68] = "I_K_ACh in component muscarinic_potassium_current (nanoA)"
    legend_constants[52] = "g_K_ACh in component muscarinic_potassium_current (microS)"
    legend_constants[23] = "g_K_ACh_base in component muscarinic_potassium_current (microS)"
    legend_constants[47] = "P_M2_KACh in component muscarinic_potassium_current (dimensionless)"
    legend_states[17] = "a in component muscarinic_potassium_current (dimensionless)"
    legend_algebraic[8] = "alpha_a in component muscarinic_potassium_current (per_second)"
    legend_constants[48] = "beta_a in component muscarinic_potassium_current (per_second)"
    legend_constants[24] = "f_Vagal in component cAMP_balance (per_second)"
    legend_states[18] = "K_i in component intracellular_concentrations_and_buffer_equations (millimolar)"
    legend_states[19] = "Ca_Calmod in component intracellular_concentrations_and_buffer_equations (dimensionless)"
    legend_states[20] = "Ca_Trop in component intracellular_concentrations_and_buffer_equations (dimensionless)"
    legend_states[21] = "Ca_Mg_Trop in component intracellular_concentrations_and_buffer_equations (dimensionless)"
    legend_states[22] = "Mg_Mg_Trop in component intracellular_concentrations_and_buffer_equations (dimensionless)"
    legend_algebraic[50] = "phi_C in component intracellular_concentrations_and_buffer_equations (per_second)"
    legend_algebraic[51] = "phi_TC in component intracellular_concentrations_and_buffer_equations (per_second)"
    legend_algebraic[52] = "phi_TMgC in component intracellular_concentrations_and_buffer_equations (per_second)"
    legend_algebraic[10] = "phi_TMgM in component intracellular_concentrations_and_buffer_equations (per_second)"
    legend_algebraic[56] = "phi_B in component intracellular_concentrations_and_buffer_equations (millimolar_per_second)"
    legend_constants[25] = "Mg_i in component intracellular_concentrations_and_buffer_equations (millimolar)"
    legend_algebraic[53] = "F_C in component intracellular_concentrations_and_buffer_equations (millimolar_per_second)"
    legend_algebraic[54] = "F_TC in component intracellular_concentrations_and_buffer_equations (millimolar_per_second)"
    legend_algebraic[55] = "F_TMgC in component intracellular_concentrations_and_buffer_equations (millimolar_per_second)"
    legend_constants[26] = "Vol in component cleft_space_equations (mm_cubed)"
    legend_constants[49] = "V_i in component intracellular_concentrations_and_buffer_equations (mm_cubed)"
    legend_algebraic[61] = "i_up in component SR_Ca_uptake_and_release (nanoA)"
    legend_algebraic[59] = "i_rel in component SR_Ca_uptake_and_release (nanoA)"
    legend_constants[27] = "Na_b in component cleft_space_equations (millimolar)"
    legend_constants[28] = "Ca_b in component cleft_space_equations (millimolar)"
    legend_constants[50] = "V_c in component cleft_space_equations (mm_cubed)"
    legend_constants[29] = "tau_p in component cleft_space_equations (second)"
    legend_states[23] = "Ca_up in component SR_Ca_uptake_and_release (millimolar)"
    legend_constants[30] = "alpha_up in component SR_Ca_uptake_and_release (nanoA)"
    legend_constants[31] = "beta_up in component SR_Ca_uptake_and_release (nanoA)"
    legend_states[24] = "Ca_rel in component SR_Ca_uptake_and_release (millimolar)"
    legend_constants[32] = "alpha_rel in component SR_Ca_uptake_and_release (nanoA_per_millimolar)"
    legend_algebraic[60] = "i_tr in component SR_Ca_uptake_and_release (nanoA)"
    legend_constants[51] = "K1 in component SR_Ca_uptake_and_release (dimensionless)"
    legend_algebraic[57] = "K2 in component SR_Ca_uptake_and_release (millimolar)"
    legend_constants[33] = "k_cyca in component SR_Ca_uptake_and_release (millimolar)"
    legend_constants[34] = "k_xcs in component SR_Ca_uptake_and_release (dimensionless)"
    legend_constants[35] = "k_SRCa in component SR_Ca_uptake_and_release (millimolar)"
    legend_constants[36] = "k_rel in component SR_Ca_uptake_and_release (millimolar)"
    legend_algebraic[11] = "r_act in component SR_Ca_uptake_and_release (per_second)"
    legend_algebraic[21] = "r_inact in component SR_Ca_uptake_and_release (per_second)"
    legend_states[25] = "Ca_Calse in component SR_Ca_uptake_and_release (dimensionless)"
    legend_algebraic[58] = "phi_Calse in component SR_Ca_uptake_and_release (per_second)"
    legend_states[26] = "F1 in component SR_Ca_uptake_and_release (dimensionless)"
    legend_states[27] = "F2 in component SR_Ca_uptake_and_release (dimensionless)"
    legend_states[28] = "F3 in component SR_Ca_uptake_and_release (dimensionless)"
    legend_constants[53] = "V_up in component SR_Ca_uptake_and_release (mm_cubed)"
    legend_constants[54] = "V_rel in component SR_Ca_uptake_and_release (mm_cubed)"
    legend_constants[37] = "cGMP in component cAMP_balance (millimolar)"
    legend_constants[38] = "Iso in component cAMP_balance (millimolar)"
    legend_constants[39] = "Km_Iso in component cAMP_balance (millimolar)"
    legend_constants[40] = "Km_ACh in component cAMP_balance (millimolar)"
    legend_constants[41] = "K_PDE in component cAMP_balance (dimensionless)"
    legend_constants[42] = "K_ADC in component cAMP_balance (millimolar_per_second)"
    legend_constants[43] = "V_PDE in component cAMP_balance (per_second)"
    legend_constants[44] = "P_M2_ADC in component cAMP_balance (dimensionless)"
    legend_rates[0] = "d/dt V in component membrane (millivolt)"
    legend_rates[2] = "d/dt m in component sodium_current_m_gate (dimensionless)"
    legend_rates[3] = "d/dt h1 in component sodium_current_h_gate (dimensionless)"
    legend_rates[4] = "d/dt h2 in component sodium_current_h_gate (dimensionless)"
    legend_rates[6] = "d/dt d_L in component L_type_Ca_channel_d_gate (dimensionless)"
    legend_rates[7] = "d/dt f_L in component L_type_Ca_channel_f_gate (dimensionless)"
    legend_rates[8] = "d/dt d_T in component T_type_Ca_channel_d_gate (dimensionless)"
    legend_rates[9] = "d/dt f_T in component T_type_Ca_channel_f_gate (dimensionless)"
    legend_rates[10] = "d/dt P_a in component delayed_rectifying_potassium_current_P_a_gate (dimensionless)"
    legend_rates[11] = "d/dt P_i in component delayed_rectifying_potassium_current_P_i_gate (dimensionless)"
    legend_rates[12] = "d/dt y in component hyperpolarisation_activated_current_y_gate (dimensionless)"
    legend_rates[17] = "d/dt a in component muscarinic_potassium_current (dimensionless)"
    legend_rates[19] = "d/dt Ca_Calmod in component intracellular_concentrations_and_buffer_equations (dimensionless)"
    legend_rates[20] = "d/dt Ca_Trop in component intracellular_concentrations_and_buffer_equations (dimensionless)"
    legend_rates[21] = "d/dt Ca_Mg_Trop in component intracellular_concentrations_and_buffer_equations (dimensionless)"
    legend_rates[22] = "d/dt Mg_Mg_Trop in component intracellular_concentrations_and_buffer_equations (dimensionless)"
    legend_rates[13] = "d/dt Na_i in component intracellular_concentrations_and_buffer_equations (millimolar)"
    legend_rates[18] = "d/dt K_i in component intracellular_concentrations_and_buffer_equations (millimolar)"
    legend_rates[15] = "d/dt Ca_i in component intracellular_concentrations_and_buffer_equations (millimolar)"
    legend_rates[1] = "d/dt Na_c in component cleft_space_equations (millimolar)"
    legend_rates[14] = "d/dt K_c in component cleft_space_equations (millimolar)"
    legend_rates[16] = "d/dt Ca_c in component cleft_space_equations (millimolar)"
    legend_rates[25] = "d/dt Ca_Calse in component SR_Ca_uptake_and_release (dimensionless)"
    legend_rates[26] = "d/dt F1 in component SR_Ca_uptake_and_release (dimensionless)"
    legend_rates[27] = "d/dt F2 in component SR_Ca_uptake_and_release (dimensionless)"
    legend_rates[28] = "d/dt F3 in component SR_Ca_uptake_and_release (dimensionless)"
    legend_rates[23] = "d/dt Ca_up in component SR_Ca_uptake_and_release (millimolar)"
    legend_rates[24] = "d/dt Ca_rel in component SR_Ca_uptake_and_release (millimolar)"
    legend_rates[5] = "d/dt cAMP in component cAMP_balance (millimolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = -49.54105
    constants[0] = 8314.472
    constants[1] = 310
    constants[2] = 96485.3415
    constants[3] = 5.5e-5
    constants[4] = 0.00344
    states[1] = 139.9988
    constants[5] = 0
    states[2] = 0.250113
    states[3] = 0.001386897
    states[4] = 0.002065463
    constants[6] = 0.02115
    states[5] = 3e-3
    constants[7] = 46.4
    states[6] = 0.002572773
    states[7] = 0.98651
    constants[8] = 0.02521
    constants[9] = 45
    states[8] = 0.02012114
    states[9] = 0.1945111
    constants[10] = 5.4
    states[10] = 0.02302278
    states[11] = 0.3777728
    constants[11] = 0.00016
    constants[12] = 0.0000364
    constants[13] = 0.0000694
    constants[14] = 0.0067478
    constants[15] = 0.0128821
    states[12] = 0.09227776
    constants[16] = 5.46
    constants[17] = 0.621
    constants[18] = 0.2192
    states[13] = 9.701621
    states[14] = 5.389014
    states[15] = 3.787018e-4
    constants[19] = 0.02869
    constants[20] = 0.00001248
    constants[21] = 0.0001
    constants[22] = 0.5
    states[16] = 2.00474
    constants[23] = 7.833e-3
    states[17] = 0
    constants[24] = 200
    states[18] = 1.407347e2
    states[19] = 0.1411678
    states[20] = 0.07331396
    states[21] = 0.7618549
    states[22] = 0.2097049
    constants[25] = 2.5
    constants[26] = 3.497e-6
    constants[27] = 140
    constants[28] = 2
    constants[29] = 0.01
    states[23] = 16.95311
    constants[30] = 0.08
    constants[31] = 0.072
    states[24] = 16.85024
    constants[32] = 0.5
    constants[33] = 0.00005
    constants[34] = 0.9
    constants[35] = 22
    constants[36] = 0.004
    states[25] = 0.9528726
    states[26] = 0.1133251
    states[27] = 0.0007594214
    states[28] = 0.8859153
    constants[37] = 2e-3
    constants[38] = 0
    constants[39] = 0.14e-3
    constants[40] = 0.14e-3
    constants[41] = 6
    constants[42] = 8e-3
    constants[43] = 20
    constants[44] = 0.02
    constants[45] = 1.00000-constants[5]/(constants[5]+0.00100000)
    constants[46] = 1.00000-constants[5]/(constants[5]+0.500000)
    constants[47] = custom_piecewise([less(constants[24] , 100.000) & greater(constants[24] , 25.0000), 1.02600/(1.00000+exp((constants[24]+11.0500)/-7.50950))-0.990000 , True, 0.000600000])
    constants[48] = 12.3200/(1.00000+0.00420000/constants[5])
    constants[49] = 0.465000*constants[26]
    constants[50] = 0.136000*constants[26]
    constants[51] = (constants[33]*constants[34])/constants[35]
    constants[52] = constants[47]*constants[23]
    constants[53] = 0.0116600*constants[49]
    constants[54] = 0.00129600*constants[49]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[5] = 1.00000*(constants[42]*((1.00000+constants[38]/(constants[38]+constants[39]))-(constants[44]*constants[5])/(constants[44]*constants[5]+constants[40]))-(constants[43]*constants[37]*states[5])/(states[5]+constants[41]*constants[37]))
    algebraic[8] = 17.0000*exp(0.0133000*(states[0]+40.0000))
    rates[17] = constants[48]*(1.00000-states[17])-algebraic[8]*states[17]
    algebraic[10] = 1290.00*constants[25]*(1.00000-(states[21]+states[22]))-429.000*states[22]
    rates[22] = algebraic[10]
    algebraic[11] = 240.000*exp((states[0]-40.0000)*0.0800000)+240.000*(power(states[15]/(states[15]+constants[36]), 4.00000))
    rates[26] = 0.960000*states[28]-algebraic[11]*states[26]
    algebraic[17] = 1.00000/(17.0000*exp(0.0398000*states[0])+2.11000*exp(-0.0510000*states[0]))
    algebraic[5] = 1.00000/(1.00000+exp((states[0]+5.10000)/-7.40000))
    rates[10] = (algebraic[5]-states[10])/algebraic[17]
    algebraic[6] = 100.000*exp(-0.0183000*states[0])
    algebraic[18] = 656.000*exp(0.00942000*states[0])
    rates[11] = algebraic[6]*(1.00000-states[11])-algebraic[18]*states[11]
    algebraic[21] = 40.0000+240.000*(power(states[15]/(states[15]+constants[36]), 4.00000))
    rates[27] = algebraic[11]*states[26]-algebraic[21]*states[27]
    rates[28] = algebraic[21]*states[27]-0.960000*states[28]
    algebraic[7] = 20.5000/(1.00000+exp((states[5]-0.00340000)/-0.000500000))-78.5600
    algebraic[19] = 1.00000/(1.00000+exp((states[0]-algebraic[7])/9.00000))
    algebraic[27] = 1.00000/(1.64830*exp((states[0]+54.0600)/-24.3300)+14.0106/(0.700000+exp((states[0]+60.0000)/-5.50000)))
    rates[12] = (algebraic[19]-states[12])/algebraic[27]
    algebraic[0] = (-824.000*(states[0]+51.9000))/(exp((states[0]+51.9000)/-8.90000)-1.00000)
    algebraic[12] = 32960.0*exp((states[0]+51.9000)/-8.90000)
    algebraic[22] = algebraic[0]/(algebraic[0]+algebraic[12])
    algebraic[29] = 1.00000/(algebraic[0]+algebraic[12])+1.50000e-05
    rates[2] = (algebraic[22]-states[2])/algebraic[29]
    algebraic[1] = 165.000*exp((states[0]+101.300)/-12.6000)
    algebraic[13] = 12360.0/(320.000*exp((states[0]+101.300)/-12.6000)+1.00000)
    algebraic[23] = algebraic[1]/(algebraic[1]+algebraic[13])
    algebraic[30] = 1.00000/(algebraic[1]+algebraic[13])
    rates[3] = (algebraic[23]-states[3])/algebraic[30]
    algebraic[31] = 1.00000/(1.00000+exp((states[0]+30.0000)/5.00000))
    algebraic[2] = (3.75000*(states[0]+28.0000))/(exp((states[0]+28.0000)/4.00000)-1.00000)
    algebraic[14] = 30.0000/(1.00000+exp((states[0]+28.0000)/-4.00000))
    algebraic[24] = 1.00000/(algebraic[2]+algebraic[14])
    rates[7] = (algebraic[31]-states[7])/algebraic[24]
    algebraic[32] = 1.00000/(1.00000+exp((states[0]+26.3000)/-6.00000))
    algebraic[3] = 1068.00*exp((states[0]+26.3000)/30.0000)
    algebraic[15] = 1068.00*exp((states[0]+26.3000)/-30.0000)
    algebraic[25] = 1.00000/(algebraic[3]+algebraic[15])
    rates[8] = (algebraic[32]-states[8])/algebraic[25]
    algebraic[33] = 1.00000/(1.00000+exp((states[0]+61.7000)/5.60000))
    algebraic[4] = 15.3000*exp((states[0]+61.7000)/-83.3000)
    algebraic[16] = 15.0000*exp((states[0]+61.7000)/15.3800)
    algebraic[26] = 1.00000/(algebraic[4]+algebraic[16])
    rates[9] = (algebraic[33]-states[9])/algebraic[26]
    algebraic[35] = algebraic[23]
    algebraic[37] = 20.0000*algebraic[30]
    rates[4] = (algebraic[35]-states[4])/algebraic[37]
    algebraic[38] = 1.00000/(1.00000+exp((states[0]+14.1000)/-6.00000))
    algebraic[28] = (-28.3900*(states[0]+35.0000))/(exp((states[0]+35.0000)/-2.50000)-1.00000)+(-84.9000*states[0])/(exp(-0.208000*states[0])-1.00000)
    algebraic[34] = (11.4300*(states[0]-5.00000))/(exp(0.400000*(states[0]-5.00000))-1.00000)
    algebraic[36] = 1.00000/(algebraic[28]+algebraic[34])
    rates[6] = (algebraic[38]-states[6])/algebraic[36]
    algebraic[50] = 129000.*states[15]*(1.00000-states[19])-307.000*states[19]
    rates[19] = algebraic[50]
    algebraic[51] = 50500.0*states[15]*(1.00000-states[20])-252.000*states[20]
    rates[20] = algebraic[51]
    algebraic[52] = 129000.*states[15]*(1.00000-(states[21]+states[22]))-4.25000*states[21]
    rates[21] = algebraic[52]
    algebraic[58] = 770.000*states[24]*(1.00000-states[25])-641.000*states[25]
    rates[25] = algebraic[58]
    algebraic[59] = constants[32]*(power(states[27]/(states[27]+0.250000), 2.00000))*states[24]
    algebraic[60] = ((states[23]-states[24])*2.00000*constants[2]*constants[53])/0.0641800
    rates[24] = (algebraic[60]-algebraic[59])/(2.00000*constants[54]*constants[2])-11.4800*algebraic[58]
    algebraic[57] = states[15]+states[23]*constants[51]+constants[33]*constants[34]+constants[33]
    algebraic[61] = (constants[30]*states[15]-constants[31]*states[23]*constants[51])/algebraic[57]
    rates[23] = (algebraic[61]-algebraic[60])/(2.00000*constants[53]*constants[2])
    algebraic[62] = ((constants[0]*constants[1])/constants[2])*log(states[1]/states[13])
    algebraic[63] = (((constants[45]*constants[4]*(power(states[2], 3.00000))*states[3]*states[4]*states[1]*states[0]*(power(constants[2], 2.00000)))/(constants[0]*constants[1]))*(exp(((states[0]-algebraic[62])*constants[2])/(constants[0]*constants[1]))-1.00000))/(exp((states[0]*constants[2])/(constants[0]*constants[1]))-1.00000)
    algebraic[46] = 1.60000/(1.00000+exp((states[5]-0.00375000)/-0.000150000))+0.990000
    algebraic[47] = custom_piecewise([less(states[15] , 0.000150000), ((2.00000-algebraic[46])*constants[18]*(power(states[13]/(constants[16]+states[13]), 3.00000))*(power(states[14]/(constants[17]+states[14]), 2.00000))*1.60000)/(1.50000+exp((states[0]+60.0000)/-40.0000)) , True, (algebraic[46]*constants[18]*(power(states[13]/(constants[16]+states[13]), 3.00000))*(power(states[14]/(constants[17]+states[14]), 2.00000))*1.60000)/(1.50000+exp((states[0]+60.0000)/-40.0000))])
    algebraic[49] = (constants[20]*((power(states[13], 3.00000))*states[16]*exp(0.0374300*states[0]*constants[22])-(power(states[1], 3.00000))*states[15]*exp(0.0374300*states[0]*(constants[22]-1.00000))))/(1.00000+constants[21]*(states[15]*(power(states[1], 3.00000))+states[16]*(power(states[13], 3.00000))))
    algebraic[64] = constants[46]*constants[11]*(states[0]-algebraic[62])
    algebraic[43] = constants[14]*(power(states[12], 2.00000))*(states[0]-75.0000)
    rates[13] = -(3.00000*algebraic[47]+3.00000*algebraic[49]+algebraic[64]+algebraic[43]+algebraic[63])/(constants[2]*constants[49])
    rates[1] = (constants[27]-states[1])/constants[29]+(algebraic[63]+3.00000*algebraic[49]+3.00000*algebraic[47]+algebraic[64]+algebraic[43])/(constants[2]*constants[50])
    algebraic[41] = 0.620000*(1.00000+(2.61290*states[5])/(states[5]+0.00900000))-0.0250000
    algebraic[42] = algebraic[41]*0.00693000*(power(constants[10]/1.00000, 0.590000))
    algebraic[65] = ((constants[0]*constants[1])/constants[2])*log(states[14]/states[18])
    algebraic[66] = algebraic[42]*states[10]*states[11]*(states[0]-algebraic[65])
    algebraic[67] = constants[13]*(states[0]-algebraic[65])
    algebraic[44] = constants[15]*(power(states[12], 2.00000))*(states[0]+85.0000)
    rates[18] = (2.00000*algebraic[47]-(algebraic[66]+algebraic[44]+algebraic[67]))/(constants[2]*constants[49])
    rates[14] = (constants[10]-states[14])/constants[29]+(-2.00000*algebraic[47]+algebraic[66]+algebraic[67]+algebraic[44])/(constants[2]*constants[50])
    algebraic[40] = constants[8]*states[8]*states[9]*(states[0]-constants[9])
    algebraic[9] = 0.400000*(1.00000+(4.50000*states[5])/(states[5]+0.00650000))+0.0315700
    algebraic[20] = constants[6]*algebraic[9]
    algebraic[39] = algebraic[20]*(states[7]*states[6]+0.0950000*algebraic[38])*(states[0]-constants[7])
    algebraic[48] = (constants[19]*states[15])/(states[15]+0.000400000)
    algebraic[70] = ((0.500000*constants[0]*constants[1])/constants[2])*log(states[16]/states[15])
    algebraic[71] = constants[12]*(states[0]-algebraic[70])
    algebraic[53] = 0.0900000*algebraic[50]
    algebraic[54] = 0.0310000*algebraic[51]
    algebraic[55] = 0.0620000*algebraic[52]
    algebraic[56] = algebraic[53]+algebraic[54]+algebraic[55]
    rates[15] = ((2.00000*algebraic[49]+algebraic[59])-(algebraic[39]+algebraic[40]+algebraic[48]+algebraic[71]+algebraic[61]))/(2.00000*constants[49]*constants[2])-algebraic[56]
    rates[16] = (constants[28]-states[16])/constants[29]+(-2.00000*algebraic[49]+algebraic[39]+algebraic[40]+algebraic[48]+algebraic[71])/(2.00000*constants[2]*constants[50])
    algebraic[45] = algebraic[43]+algebraic[44]
    algebraic[72] = algebraic[64]+algebraic[71]+algebraic[67]
    algebraic[68] = constants[52]*(states[0]-algebraic[65])
    algebraic[69] = states[17]*algebraic[68]
    rates[0] = -(algebraic[63]+algebraic[40]+algebraic[39]+algebraic[66]+algebraic[45]+algebraic[72]+algebraic[47]+algebraic[49]+algebraic[48]+algebraic[69])/constants[3]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[8] = 17.0000*exp(0.0133000*(states[0]+40.0000))
    algebraic[10] = 1290.00*constants[25]*(1.00000-(states[21]+states[22]))-429.000*states[22]
    algebraic[11] = 240.000*exp((states[0]-40.0000)*0.0800000)+240.000*(power(states[15]/(states[15]+constants[36]), 4.00000))
    algebraic[17] = 1.00000/(17.0000*exp(0.0398000*states[0])+2.11000*exp(-0.0510000*states[0]))
    algebraic[5] = 1.00000/(1.00000+exp((states[0]+5.10000)/-7.40000))
    algebraic[6] = 100.000*exp(-0.0183000*states[0])
    algebraic[18] = 656.000*exp(0.00942000*states[0])
    algebraic[21] = 40.0000+240.000*(power(states[15]/(states[15]+constants[36]), 4.00000))
    algebraic[7] = 20.5000/(1.00000+exp((states[5]-0.00340000)/-0.000500000))-78.5600
    algebraic[19] = 1.00000/(1.00000+exp((states[0]-algebraic[7])/9.00000))
    algebraic[27] = 1.00000/(1.64830*exp((states[0]+54.0600)/-24.3300)+14.0106/(0.700000+exp((states[0]+60.0000)/-5.50000)))
    algebraic[0] = (-824.000*(states[0]+51.9000))/(exp((states[0]+51.9000)/-8.90000)-1.00000)
    algebraic[12] = 32960.0*exp((states[0]+51.9000)/-8.90000)
    algebraic[22] = algebraic[0]/(algebraic[0]+algebraic[12])
    algebraic[29] = 1.00000/(algebraic[0]+algebraic[12])+1.50000e-05
    algebraic[1] = 165.000*exp((states[0]+101.300)/-12.6000)
    algebraic[13] = 12360.0/(320.000*exp((states[0]+101.300)/-12.6000)+1.00000)
    algebraic[23] = algebraic[1]/(algebraic[1]+algebraic[13])
    algebraic[30] = 1.00000/(algebraic[1]+algebraic[13])
    algebraic[31] = 1.00000/(1.00000+exp((states[0]+30.0000)/5.00000))
    algebraic[2] = (3.75000*(states[0]+28.0000))/(exp((states[0]+28.0000)/4.00000)-1.00000)
    algebraic[14] = 30.0000/(1.00000+exp((states[0]+28.0000)/-4.00000))
    algebraic[24] = 1.00000/(algebraic[2]+algebraic[14])
    algebraic[32] = 1.00000/(1.00000+exp((states[0]+26.3000)/-6.00000))
    algebraic[3] = 1068.00*exp((states[0]+26.3000)/30.0000)
    algebraic[15] = 1068.00*exp((states[0]+26.3000)/-30.0000)
    algebraic[25] = 1.00000/(algebraic[3]+algebraic[15])
    algebraic[33] = 1.00000/(1.00000+exp((states[0]+61.7000)/5.60000))
    algebraic[4] = 15.3000*exp((states[0]+61.7000)/-83.3000)
    algebraic[16] = 15.0000*exp((states[0]+61.7000)/15.3800)
    algebraic[26] = 1.00000/(algebraic[4]+algebraic[16])
    algebraic[35] = algebraic[23]
    algebraic[37] = 20.0000*algebraic[30]
    algebraic[38] = 1.00000/(1.00000+exp((states[0]+14.1000)/-6.00000))
    algebraic[28] = (-28.3900*(states[0]+35.0000))/(exp((states[0]+35.0000)/-2.50000)-1.00000)+(-84.9000*states[0])/(exp(-0.208000*states[0])-1.00000)
    algebraic[34] = (11.4300*(states[0]-5.00000))/(exp(0.400000*(states[0]-5.00000))-1.00000)
    algebraic[36] = 1.00000/(algebraic[28]+algebraic[34])
    algebraic[50] = 129000.*states[15]*(1.00000-states[19])-307.000*states[19]
    algebraic[51] = 50500.0*states[15]*(1.00000-states[20])-252.000*states[20]
    algebraic[52] = 129000.*states[15]*(1.00000-(states[21]+states[22]))-4.25000*states[21]
    algebraic[58] = 770.000*states[24]*(1.00000-states[25])-641.000*states[25]
    algebraic[59] = constants[32]*(power(states[27]/(states[27]+0.250000), 2.00000))*states[24]
    algebraic[60] = ((states[23]-states[24])*2.00000*constants[2]*constants[53])/0.0641800
    algebraic[57] = states[15]+states[23]*constants[51]+constants[33]*constants[34]+constants[33]
    algebraic[61] = (constants[30]*states[15]-constants[31]*states[23]*constants[51])/algebraic[57]
    algebraic[62] = ((constants[0]*constants[1])/constants[2])*log(states[1]/states[13])
    algebraic[63] = (((constants[45]*constants[4]*(power(states[2], 3.00000))*states[3]*states[4]*states[1]*states[0]*(power(constants[2], 2.00000)))/(constants[0]*constants[1]))*(exp(((states[0]-algebraic[62])*constants[2])/(constants[0]*constants[1]))-1.00000))/(exp((states[0]*constants[2])/(constants[0]*constants[1]))-1.00000)
    algebraic[46] = 1.60000/(1.00000+exp((states[5]-0.00375000)/-0.000150000))+0.990000
    algebraic[47] = custom_piecewise([less(states[15] , 0.000150000), ((2.00000-algebraic[46])*constants[18]*(power(states[13]/(constants[16]+states[13]), 3.00000))*(power(states[14]/(constants[17]+states[14]), 2.00000))*1.60000)/(1.50000+exp((states[0]+60.0000)/-40.0000)) , True, (algebraic[46]*constants[18]*(power(states[13]/(constants[16]+states[13]), 3.00000))*(power(states[14]/(constants[17]+states[14]), 2.00000))*1.60000)/(1.50000+exp((states[0]+60.0000)/-40.0000))])
    algebraic[49] = (constants[20]*((power(states[13], 3.00000))*states[16]*exp(0.0374300*states[0]*constants[22])-(power(states[1], 3.00000))*states[15]*exp(0.0374300*states[0]*(constants[22]-1.00000))))/(1.00000+constants[21]*(states[15]*(power(states[1], 3.00000))+states[16]*(power(states[13], 3.00000))))
    algebraic[64] = constants[46]*constants[11]*(states[0]-algebraic[62])
    algebraic[43] = constants[14]*(power(states[12], 2.00000))*(states[0]-75.0000)
    algebraic[41] = 0.620000*(1.00000+(2.61290*states[5])/(states[5]+0.00900000))-0.0250000
    algebraic[42] = algebraic[41]*0.00693000*(power(constants[10]/1.00000, 0.590000))
    algebraic[65] = ((constants[0]*constants[1])/constants[2])*log(states[14]/states[18])
    algebraic[66] = algebraic[42]*states[10]*states[11]*(states[0]-algebraic[65])
    algebraic[67] = constants[13]*(states[0]-algebraic[65])
    algebraic[44] = constants[15]*(power(states[12], 2.00000))*(states[0]+85.0000)
    algebraic[40] = constants[8]*states[8]*states[9]*(states[0]-constants[9])
    algebraic[9] = 0.400000*(1.00000+(4.50000*states[5])/(states[5]+0.00650000))+0.0315700
    algebraic[20] = constants[6]*algebraic[9]
    algebraic[39] = algebraic[20]*(states[7]*states[6]+0.0950000*algebraic[38])*(states[0]-constants[7])
    algebraic[48] = (constants[19]*states[15])/(states[15]+0.000400000)
    algebraic[70] = ((0.500000*constants[0]*constants[1])/constants[2])*log(states[16]/states[15])
    algebraic[71] = constants[12]*(states[0]-algebraic[70])
    algebraic[53] = 0.0900000*algebraic[50]
    algebraic[54] = 0.0310000*algebraic[51]
    algebraic[55] = 0.0620000*algebraic[52]
    algebraic[56] = algebraic[53]+algebraic[54]+algebraic[55]
    algebraic[45] = algebraic[43]+algebraic[44]
    algebraic[72] = algebraic[64]+algebraic[71]+algebraic[67]
    algebraic[68] = constants[52]*(states[0]-algebraic[65])
    algebraic[69] = states[17]*algebraic[68]
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