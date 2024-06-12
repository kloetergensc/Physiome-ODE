# Size of variable arrays:
sizeAlgebraic = 75
sizeStates = 21
sizeConstants = 49
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (millisecond)"
    legend_states[0] = "V in component membrane (millivolt)"
    legend_constants[0] = "R in component membrane (joule_per_mole_kelvin)"
    legend_constants[1] = "T in component membrane (kelvin)"
    legend_constants[2] = "F in component membrane (coulomb_per_millimole)"
    legend_constants[3] = "Cm in component membrane (picoF)"
    legend_algebraic[0] = "i_st in component membrane (picoA)"
    legend_algebraic[30] = "i_Na in component fast_sodium_current (picoA)"
    legend_algebraic[50] = "i_K1 in component time_independent_potassium_current (picoA)"
    legend_algebraic[51] = "i_to in component transient_outward_K_current (picoA)"
    legend_algebraic[53] = "i_Kur in component ultrarapid_delayed_rectifier_K_current (picoA)"
    legend_algebraic[54] = "i_Kr in component rapid_delayed_rectifier_K_current (picoA)"
    legend_algebraic[55] = "i_Ks in component slow_delayed_rectifier_K_current (picoA)"
    legend_algebraic[56] = "i_Ca_L in component L_type_Ca_channel (picoA)"
    legend_algebraic[64] = "i_CaP in component sarcolemmal_calcium_pump_current (picoA)"
    legend_algebraic[58] = "i_NaK in component sodium_potassium_pump (picoA)"
    legend_algebraic[63] = "i_NaCa in component Na_Ca_exchanger_current (picoA)"
    legend_algebraic[61] = "i_B_Na in component background_currents (picoA)"
    legend_algebraic[62] = "i_B_Ca in component background_currents (picoA)"
    legend_constants[4] = "stim_start in component membrane (millisecond)"
    legend_constants[5] = "stim_end in component membrane (millisecond)"
    legend_constants[6] = "stim_period in component membrane (millisecond)"
    legend_constants[7] = "stim_duration in component membrane (millisecond)"
    legend_constants[8] = "stim_amplitude in component membrane (picoA)"
    legend_algebraic[17] = "E_Na in component fast_sodium_current (millivolt)"
    legend_constants[9] = "g_Na in component fast_sodium_current (nanoS_per_picoF)"
    legend_states[1] = "Na_i in component intracellular_ion_concentrations (millimolar)"
    legend_constants[10] = "Na_o in component standard_ionic_concentrations (millimolar)"
    legend_states[2] = "m in component fast_sodium_current_m_gate (dimensionless)"
    legend_states[3] = "h in component fast_sodium_current_h_gate (dimensionless)"
    legend_states[4] = "j in component fast_sodium_current_j_gate (dimensionless)"
    legend_algebraic[1] = "alpha_m in component fast_sodium_current_m_gate (per_millisecond)"
    legend_algebraic[18] = "beta_m in component fast_sodium_current_m_gate (per_millisecond)"
    legend_algebraic[31] = "m_inf in component fast_sodium_current_m_gate (dimensionless)"
    legend_algebraic[41] = "tau_m in component fast_sodium_current_m_gate (millisecond)"
    legend_algebraic[2] = "alpha_h in component fast_sodium_current_h_gate (per_millisecond)"
    legend_algebraic[19] = "beta_h in component fast_sodium_current_h_gate (per_millisecond)"
    legend_algebraic[32] = "h_inf in component fast_sodium_current_h_gate (dimensionless)"
    legend_algebraic[42] = "tau_h in component fast_sodium_current_h_gate (millisecond)"
    legend_algebraic[3] = "alpha_j in component fast_sodium_current_j_gate (per_millisecond)"
    legend_algebraic[20] = "beta_j in component fast_sodium_current_j_gate (per_millisecond)"
    legend_algebraic[33] = "j_inf in component fast_sodium_current_j_gate (dimensionless)"
    legend_algebraic[43] = "tau_j in component fast_sodium_current_j_gate (millisecond)"
    legend_algebraic[40] = "E_K in component time_independent_potassium_current (millivolt)"
    legend_constants[11] = "g_K1 in component time_independent_potassium_current (nanoS_per_picoF)"
    legend_constants[12] = "K_o in component standard_ionic_concentrations (millimolar)"
    legend_states[5] = "K_i in component intracellular_ion_concentrations (millimolar)"
    legend_constants[13] = "K_Q10 in component transient_outward_K_current (dimensionless)"
    legend_constants[14] = "g_to in component transient_outward_K_current (nanoS_per_picoF)"
    legend_states[6] = "oa in component transient_outward_K_current_oa_gate (dimensionless)"
    legend_states[7] = "oi in component transient_outward_K_current_oi_gate (dimensionless)"
    legend_algebraic[4] = "alpha_oa in component transient_outward_K_current_oa_gate (per_millisecond)"
    legend_algebraic[21] = "beta_oa in component transient_outward_K_current_oa_gate (per_millisecond)"
    legend_algebraic[34] = "tau_oa in component transient_outward_K_current_oa_gate (millisecond)"
    legend_algebraic[44] = "oa_infinity in component transient_outward_K_current_oa_gate (dimensionless)"
    legend_algebraic[5] = "alpha_oi in component transient_outward_K_current_oi_gate (per_millisecond)"
    legend_algebraic[22] = "beta_oi in component transient_outward_K_current_oi_gate (per_millisecond)"
    legend_algebraic[35] = "tau_oi in component transient_outward_K_current_oi_gate (millisecond)"
    legend_algebraic[45] = "oi_infinity in component transient_outward_K_current_oi_gate (dimensionless)"
    legend_algebraic[52] = "g_Kur in component ultrarapid_delayed_rectifier_K_current (nanoS_per_picoF)"
    legend_states[8] = "ua in component ultrarapid_delayed_rectifier_K_current_ua_gate (dimensionless)"
    legend_states[9] = "ui in component ultrarapid_delayed_rectifier_K_current_ui_gate (dimensionless)"
    legend_algebraic[6] = "alpha_ua in component ultrarapid_delayed_rectifier_K_current_ua_gate (per_millisecond)"
    legend_algebraic[23] = "beta_ua in component ultrarapid_delayed_rectifier_K_current_ua_gate (per_millisecond)"
    legend_algebraic[36] = "tau_ua in component ultrarapid_delayed_rectifier_K_current_ua_gate (millisecond)"
    legend_algebraic[46] = "ua_infinity in component ultrarapid_delayed_rectifier_K_current_ua_gate (dimensionless)"
    legend_algebraic[7] = "alpha_ui in component ultrarapid_delayed_rectifier_K_current_ui_gate (per_millisecond)"
    legend_algebraic[24] = "beta_ui in component ultrarapid_delayed_rectifier_K_current_ui_gate (per_millisecond)"
    legend_algebraic[37] = "tau_ui in component ultrarapid_delayed_rectifier_K_current_ui_gate (millisecond)"
    legend_algebraic[47] = "ui_infinity in component ultrarapid_delayed_rectifier_K_current_ui_gate (dimensionless)"
    legend_constants[15] = "g_Kr in component rapid_delayed_rectifier_K_current (nanoS_per_picoF)"
    legend_states[10] = "xr in component rapid_delayed_rectifier_K_current_xr_gate (dimensionless)"
    legend_algebraic[8] = "alpha_xr in component rapid_delayed_rectifier_K_current_xr_gate (per_millisecond)"
    legend_algebraic[25] = "beta_xr in component rapid_delayed_rectifier_K_current_xr_gate (per_millisecond)"
    legend_algebraic[38] = "tau_xr in component rapid_delayed_rectifier_K_current_xr_gate (millisecond)"
    legend_algebraic[48] = "xr_infinity in component rapid_delayed_rectifier_K_current_xr_gate (dimensionless)"
    legend_constants[16] = "g_Ks in component slow_delayed_rectifier_K_current (nanoS_per_picoF)"
    legend_states[11] = "xs in component slow_delayed_rectifier_K_current_xs_gate (dimensionless)"
    legend_algebraic[9] = "alpha_xs in component slow_delayed_rectifier_K_current_xs_gate (per_millisecond)"
    legend_algebraic[26] = "beta_xs in component slow_delayed_rectifier_K_current_xs_gate (per_millisecond)"
    legend_algebraic[39] = "tau_xs in component slow_delayed_rectifier_K_current_xs_gate (millisecond)"
    legend_algebraic[49] = "xs_infinity in component slow_delayed_rectifier_K_current_xs_gate (dimensionless)"
    legend_constants[17] = "g_Ca_L in component L_type_Ca_channel (nanoS_per_picoF)"
    legend_states[12] = "Ca_i in component intracellular_ion_concentrations (millimolar)"
    legend_states[13] = "d in component L_type_Ca_channel_d_gate (dimensionless)"
    legend_states[14] = "f in component L_type_Ca_channel_f_gate (dimensionless)"
    legend_states[15] = "f_Ca in component L_type_Ca_channel_f_Ca_gate (dimensionless)"
    legend_algebraic[10] = "d_infinity in component L_type_Ca_channel_d_gate (dimensionless)"
    legend_algebraic[27] = "tau_d in component L_type_Ca_channel_d_gate (millisecond)"
    legend_algebraic[11] = "f_infinity in component L_type_Ca_channel_f_gate (dimensionless)"
    legend_algebraic[28] = "tau_f in component L_type_Ca_channel_f_gate (millisecond)"
    legend_algebraic[12] = "f_Ca_infinity in component L_type_Ca_channel_f_Ca_gate (dimensionless)"
    legend_constants[44] = "tau_f_Ca in component L_type_Ca_channel_f_Ca_gate (millisecond)"
    legend_constants[18] = "Km_Na_i in component sodium_potassium_pump (millimolar)"
    legend_constants[19] = "Km_K_o in component sodium_potassium_pump (millimolar)"
    legend_constants[20] = "i_NaK_max in component sodium_potassium_pump (picoA_per_picoF)"
    legend_algebraic[57] = "f_NaK in component sodium_potassium_pump (dimensionless)"
    legend_constants[45] = "sigma in component sodium_potassium_pump (dimensionless)"
    legend_algebraic[60] = "i_B_K in component background_currents (picoA)"
    legend_constants[21] = "g_B_Na in component background_currents (nanoS_per_picoF)"
    legend_constants[22] = "g_B_Ca in component background_currents (nanoS_per_picoF)"
    legend_constants[23] = "g_B_K in component background_currents (nanoS_per_picoF)"
    legend_algebraic[59] = "E_Ca in component background_currents (millivolt)"
    legend_constants[24] = "Ca_o in component standard_ionic_concentrations (millimolar)"
    legend_constants[25] = "I_NaCa_max in component Na_Ca_exchanger_current (picoA_per_picoF)"
    legend_constants[26] = "K_mNa in component Na_Ca_exchanger_current (millimolar)"
    legend_constants[27] = "K_mCa in component Na_Ca_exchanger_current (millimolar)"
    legend_constants[28] = "K_sat in component Na_Ca_exchanger_current (dimensionless)"
    legend_constants[29] = "gamma in component Na_Ca_exchanger_current (dimensionless)"
    legend_constants[30] = "i_CaP_max in component sarcolemmal_calcium_pump_current (picoA_per_picoF)"
    legend_algebraic[65] = "i_rel in component Ca_release_current_from_JSR (millimolar_per_millisecond)"
    legend_algebraic[66] = "Fn in component Ca_release_current_from_JSR (dimensionless)"
    legend_constants[31] = "K_rel in component Ca_release_current_from_JSR (per_millisecond)"
    legend_constants[47] = "V_rel in component intracellular_ion_concentrations (micrometre_3)"
    legend_states[16] = "Ca_rel in component intracellular_ion_concentrations (millimolar)"
    legend_states[17] = "u in component Ca_release_current_from_JSR_u_gate (dimensionless)"
    legend_states[18] = "v in component Ca_release_current_from_JSR_v_gate (dimensionless)"
    legend_states[19] = "w in component Ca_release_current_from_JSR_w_gate (dimensionless)"
    legend_constants[46] = "tau_u in component Ca_release_current_from_JSR_u_gate (millisecond)"
    legend_algebraic[68] = "u_infinity in component Ca_release_current_from_JSR_u_gate (dimensionless)"
    legend_algebraic[69] = "tau_v in component Ca_release_current_from_JSR_v_gate (millisecond)"
    legend_algebraic[71] = "v_infinity in component Ca_release_current_from_JSR_v_gate (dimensionless)"
    legend_algebraic[13] = "tau_w in component Ca_release_current_from_JSR_w_gate (millisecond)"
    legend_algebraic[29] = "w_infinity in component Ca_release_current_from_JSR_w_gate (dimensionless)"
    legend_algebraic[67] = "i_tr in component transfer_current_from_NSR_to_JSR (millimolar_per_millisecond)"
    legend_constants[32] = "tau_tr in component transfer_current_from_NSR_to_JSR (millisecond)"
    legend_states[20] = "Ca_up in component intracellular_ion_concentrations (millimolar)"
    legend_constants[33] = "I_up_max in component Ca_uptake_current_by_the_NSR (millimolar_per_millisecond)"
    legend_algebraic[70] = "i_up in component Ca_uptake_current_by_the_NSR (millimolar_per_millisecond)"
    legend_constants[34] = "K_up in component Ca_uptake_current_by_the_NSR (millimolar)"
    legend_algebraic[72] = "i_up_leak in component Ca_leak_current_by_the_NSR (millimolar_per_millisecond)"
    legend_constants[35] = "Ca_up_max in component Ca_leak_current_by_the_NSR (millimolar)"
    legend_constants[36] = "CMDN_max in component Ca_buffers (millimolar)"
    legend_constants[37] = "TRPN_max in component Ca_buffers (millimolar)"
    legend_constants[38] = "CSQN_max in component Ca_buffers (millimolar)"
    legend_constants[39] = "Km_CMDN in component Ca_buffers (millimolar)"
    legend_constants[40] = "Km_TRPN in component Ca_buffers (millimolar)"
    legend_constants[41] = "Km_CSQN in component Ca_buffers (millimolar)"
    legend_algebraic[14] = "Ca_CMDN in component Ca_buffers (millimolar)"
    legend_algebraic[15] = "Ca_TRPN in component Ca_buffers (millimolar)"
    legend_algebraic[16] = "Ca_CSQN in component Ca_buffers (millimolar)"
    legend_constants[42] = "V_cell in component intracellular_ion_concentrations (micrometre_3)"
    legend_constants[43] = "V_i in component intracellular_ion_concentrations (micrometre_3)"
    legend_constants[48] = "V_up in component intracellular_ion_concentrations (micrometre_3)"
    legend_algebraic[73] = "B1 in component intracellular_ion_concentrations (millimolar_per_millisecond)"
    legend_algebraic[74] = "B2 in component intracellular_ion_concentrations (dimensionless)"
    legend_rates[0] = "d/dt V in component membrane (millivolt)"
    legend_rates[2] = "d/dt m in component fast_sodium_current_m_gate (dimensionless)"
    legend_rates[3] = "d/dt h in component fast_sodium_current_h_gate (dimensionless)"
    legend_rates[4] = "d/dt j in component fast_sodium_current_j_gate (dimensionless)"
    legend_rates[6] = "d/dt oa in component transient_outward_K_current_oa_gate (dimensionless)"
    legend_rates[7] = "d/dt oi in component transient_outward_K_current_oi_gate (dimensionless)"
    legend_rates[8] = "d/dt ua in component ultrarapid_delayed_rectifier_K_current_ua_gate (dimensionless)"
    legend_rates[9] = "d/dt ui in component ultrarapid_delayed_rectifier_K_current_ui_gate (dimensionless)"
    legend_rates[10] = "d/dt xr in component rapid_delayed_rectifier_K_current_xr_gate (dimensionless)"
    legend_rates[11] = "d/dt xs in component slow_delayed_rectifier_K_current_xs_gate (dimensionless)"
    legend_rates[13] = "d/dt d in component L_type_Ca_channel_d_gate (dimensionless)"
    legend_rates[14] = "d/dt f in component L_type_Ca_channel_f_gate (dimensionless)"
    legend_rates[15] = "d/dt f_Ca in component L_type_Ca_channel_f_Ca_gate (dimensionless)"
    legend_rates[17] = "d/dt u in component Ca_release_current_from_JSR_u_gate (dimensionless)"
    legend_rates[18] = "d/dt v in component Ca_release_current_from_JSR_v_gate (dimensionless)"
    legend_rates[19] = "d/dt w in component Ca_release_current_from_JSR_w_gate (dimensionless)"
    legend_rates[1] = "d/dt Na_i in component intracellular_ion_concentrations (millimolar)"
    legend_rates[5] = "d/dt K_i in component intracellular_ion_concentrations (millimolar)"
    legend_rates[12] = "d/dt Ca_i in component intracellular_ion_concentrations (millimolar)"
    legend_rates[20] = "d/dt Ca_up in component intracellular_ion_concentrations (millimolar)"
    legend_rates[16] = "d/dt Ca_rel in component intracellular_ion_concentrations (millimolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = -81.18
    constants[0] = 8.3143
    constants[1] = 310
    constants[2] = 96.4867
    constants[3] = 100
    constants[4] = 50
    constants[5] = 50000
    constants[6] = 1000
    constants[7] = 2
    constants[8] = -2000
    constants[9] = 7.8
    states[1] = 1.117e+01
    constants[10] = 140
    states[2] = 2.908e-3
    states[3] = 9.649e-1
    states[4] = 9.775e-1
    constants[11] = 0.09
    constants[12] = 5.4
    states[5] = 1.39e+02
    constants[13] = 3
    constants[14] = 0.1652
    states[6] = 3.043e-2
    states[7] = 9.992e-1
    states[8] = 4.966e-3
    states[9] = 9.986e-1
    constants[15] = 0.029411765
    states[10] = 3.296e-5
    constants[16] = 0.12941176
    states[11] = 1.869e-2
    constants[17] = 0.12375
    states[12] = 1.013e-4
    states[13] = 1.367e-4
    states[14] = 9.996e-1
    states[15] = 7.755e-1
    constants[18] = 10
    constants[19] = 1.5
    constants[20] = 0.59933874
    constants[21] = 0.0006744375
    constants[22] = 0.001131
    constants[23] = 0
    constants[24] = 1.8
    constants[25] = 1600
    constants[26] = 87.5
    constants[27] = 1.38
    constants[28] = 0.1
    constants[29] = 0.35
    constants[30] = 0.275
    constants[31] = 30
    states[16] = 1.488
    states[17] = 2.35e-112
    states[18] = 1
    states[19] = 0.9992
    constants[32] = 180
    states[20] = 1.488
    constants[33] = 0.005
    constants[34] = 0.00092
    constants[35] = 15
    constants[36] = 0.05
    constants[37] = 0.07
    constants[38] = 10
    constants[39] = 0.00238
    constants[40] = 0.0005
    constants[41] = 0.8
    constants[42] = 20100
    constants[43] = constants[42]*0.680000
    constants[44] = 2.00000
    constants[45] = (1.00000/7.00000)*(exp(constants[10]/67.3000)-1.00000)
    constants[46] = 8.00000
    constants[47] = 0.00480000*constants[42]
    constants[48] = 0.0552000*constants[42]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[12] = power(1.00000+states[12]/0.000350000, -1.00000)
    rates[15] = (algebraic[12]-states[15])/constants[44]
    algebraic[10] = power(1.00000+exp((states[0]+10.0000)/-8.00000), -1.00000)
    algebraic[27] = custom_piecewise([less(fabs(states[0]+10.0000) , 1.00000e-10), 4.57900/(1.00000+exp((states[0]+10.0000)/-6.24000)) , True, (1.00000-exp((states[0]+10.0000)/-6.24000))/(0.0350000*(states[0]+10.0000)*(1.00000+exp((states[0]+10.0000)/-6.24000)))])
    rates[13] = (algebraic[10]-states[13])/algebraic[27]
    algebraic[11] = exp(-(states[0]+28.0000)/6.90000)/(1.00000+exp(-(states[0]+28.0000)/6.90000))
    algebraic[28] = 9.00000*(power(0.0197000*exp(-(power(0.0337000, 2.00000))*(power(states[0]+10.0000, 2.00000)))+0.0200000, -1.00000))
    rates[14] = (algebraic[11]-states[14])/algebraic[28]
    algebraic[13] = custom_piecewise([less(fabs(states[0]-7.90000) , 1.00000e-10), (6.00000*0.200000)/1.30000 , True, (6.00000*(1.00000-exp(-(states[0]-7.90000)/5.00000)))/((1.00000+0.300000*exp(-(states[0]-7.90000)/5.00000))*1.00000*(states[0]-7.90000))])
    algebraic[29] = 1.00000-power(1.00000+exp(-(states[0]-40.0000)/17.0000), -1.00000)
    rates[19] = (algebraic[29]-states[19])/algebraic[13]
    algebraic[1] = custom_piecewise([equal(states[0] , -47.1300), 3.20000 , True, (0.320000*(states[0]+47.1300))/(1.00000-exp(-0.100000*(states[0]+47.1300)))])
    algebraic[18] = 0.0800000*exp(-states[0]/11.0000)
    algebraic[31] = algebraic[1]/(algebraic[1]+algebraic[18])
    algebraic[41] = 1.00000/(algebraic[1]+algebraic[18])
    rates[2] = (algebraic[31]-states[2])/algebraic[41]
    algebraic[2] = custom_piecewise([less(states[0] , -40.0000), 0.135000*exp((states[0]+80.0000)/-6.80000) , True, 0.00000])
    algebraic[19] = custom_piecewise([less(states[0] , -40.0000), 3.56000*exp(0.0790000*states[0])+310000.*exp(0.350000*states[0]) , True, 1.00000/(0.130000*(1.00000+exp((states[0]+10.6600)/-11.1000)))])
    algebraic[32] = algebraic[2]/(algebraic[2]+algebraic[19])
    algebraic[42] = 1.00000/(algebraic[2]+algebraic[19])
    rates[3] = (algebraic[32]-states[3])/algebraic[42]
    algebraic[3] = custom_piecewise([less(states[0] , -40.0000), ((-127140.*exp(0.244400*states[0])-3.47400e-05*exp(-0.0439100*states[0]))*(states[0]+37.7800))/(1.00000+exp(0.311000*(states[0]+79.2300))) , True, 0.00000])
    algebraic[20] = custom_piecewise([less(states[0] , -40.0000), (0.121200*exp(-0.0105200*states[0]))/(1.00000+exp(-0.137800*(states[0]+40.1400))) , True, (0.300000*exp(-2.53500e-07*states[0]))/(1.00000+exp(-0.100000*(states[0]+32.0000)))])
    algebraic[33] = algebraic[3]/(algebraic[3]+algebraic[20])
    algebraic[43] = 1.00000/(algebraic[3]+algebraic[20])
    rates[4] = (algebraic[33]-states[4])/algebraic[43]
    algebraic[4] = 0.650000*(power(exp((states[0]--10.0000)/-8.50000)+exp(((states[0]--10.0000)-40.0000)/-59.0000), -1.00000))
    algebraic[21] = 0.650000*(power(2.50000+exp(((states[0]--10.0000)+72.0000)/17.0000), -1.00000))
    algebraic[34] = (power(algebraic[4]+algebraic[21], -1.00000))/constants[13]
    algebraic[44] = power(1.00000+exp(((states[0]--10.0000)+10.4700)/-17.5400), -1.00000)
    rates[6] = (algebraic[44]-states[6])/algebraic[34]
    algebraic[5] = power(18.5300+1.00000*exp(((states[0]--10.0000)+103.700)/10.9500), -1.00000)
    algebraic[22] = power(35.5600+1.00000*exp(((states[0]--10.0000)-8.74000)/-7.44000), -1.00000)
    algebraic[35] = (power(algebraic[5]+algebraic[22], -1.00000))/constants[13]
    algebraic[45] = power(1.00000+exp(((states[0]--10.0000)+33.1000)/5.30000), -1.00000)
    rates[7] = (algebraic[45]-states[7])/algebraic[35]
    algebraic[6] = 0.650000*(power(exp((states[0]--10.0000)/-8.50000)+exp(((states[0]--10.0000)-40.0000)/-59.0000), -1.00000))
    algebraic[23] = 0.650000*(power(2.50000+exp(((states[0]--10.0000)+72.0000)/17.0000), -1.00000))
    algebraic[36] = (power(algebraic[6]+algebraic[23], -1.00000))/constants[13]
    algebraic[46] = power(1.00000+exp(((states[0]--10.0000)+20.3000)/-9.60000), -1.00000)
    rates[8] = (algebraic[46]-states[8])/algebraic[36]
    algebraic[7] = power(21.0000+1.00000*exp(((states[0]--10.0000)-195.000)/-28.0000), -1.00000)
    algebraic[24] = 1.00000/exp(((states[0]--10.0000)-168.000)/-16.0000)
    algebraic[37] = (power(algebraic[7]+algebraic[24], -1.00000))/constants[13]
    algebraic[47] = power(1.00000+exp(((states[0]--10.0000)-109.450)/27.4800), -1.00000)
    rates[9] = (algebraic[47]-states[9])/algebraic[37]
    algebraic[8] = custom_piecewise([less(fabs(states[0]+14.1000) , 1.00000e-10), 0.00150000 , True, (0.000300000*(states[0]+14.1000))/(1.00000-exp((states[0]+14.1000)/-5.00000))])
    algebraic[25] = custom_piecewise([less(fabs(states[0]-3.33280) , 1.00000e-10), 0.000378361 , True, (7.38980e-05*(states[0]-3.33280))/(exp((states[0]-3.33280)/5.12370)-1.00000)])
    algebraic[38] = power(algebraic[8]+algebraic[25], -1.00000)
    algebraic[48] = power(1.00000+exp((states[0]+14.1000)/-6.50000), -1.00000)
    rates[10] = (algebraic[48]-states[10])/algebraic[38]
    algebraic[9] = custom_piecewise([less(fabs(states[0]-19.9000) , 1.00000e-10), 0.000680000 , True, (4.00000e-05*(states[0]-19.9000))/(1.00000-exp((states[0]-19.9000)/-17.0000))])
    algebraic[26] = custom_piecewise([less(fabs(states[0]-19.9000) , 1.00000e-10), 0.000315000 , True, (3.50000e-05*(states[0]-19.9000))/(exp((states[0]-19.9000)/9.00000)-1.00000)])
    algebraic[39] = 0.500000*(power(algebraic[9]+algebraic[26], -1.00000))
    algebraic[49] = power(1.00000+exp((states[0]-19.9000)/-12.7000), -0.500000)
    rates[11] = (algebraic[49]-states[11])/algebraic[39]
    algebraic[40] = ((constants[0]*constants[1])/constants[2])*log(constants[12]/states[5])
    algebraic[50] = (constants[3]*constants[11]*(states[0]-algebraic[40]))/(1.00000+exp(0.0700000*(states[0]+80.0000)))
    algebraic[51] = constants[3]*constants[14]*(power(states[6], 3.00000))*states[7]*(states[0]-algebraic[40])
    algebraic[52] = 0.00500000+0.0500000/(1.00000+exp((states[0]-15.0000)/-13.0000))
    algebraic[53] = constants[3]*algebraic[52]*(power(states[8], 3.00000))*states[9]*(states[0]-algebraic[40])
    algebraic[54] = (constants[3]*constants[15]*states[10]*(states[0]-algebraic[40]))/(1.00000+exp((states[0]+15.0000)/22.4000))
    algebraic[55] = constants[3]*constants[16]*(power(states[11], 2.00000))*(states[0]-algebraic[40])
    algebraic[57] = power(1.00000+0.124500*exp((-0.100000*constants[2]*states[0])/(constants[0]*constants[1]))+0.0365000*constants[45]*exp((-constants[2]*states[0])/(constants[0]*constants[1])), -1.00000)
    algebraic[58] = (((constants[3]*constants[20]*algebraic[57]*1.00000)/(1.00000+power(constants[18]/states[1], 1.50000)))*constants[12])/(constants[12]+constants[19])
    algebraic[60] = constants[3]*constants[23]*(states[0]-algebraic[40])
    rates[5] = (2.00000*algebraic[58]-(algebraic[50]+algebraic[51]+algebraic[53]+algebraic[54]+algebraic[55]+algebraic[60]))/(constants[43]*constants[2])
    algebraic[17] = ((constants[0]*constants[1])/constants[2])*log(constants[10]/states[1])
    algebraic[30] = constants[3]*constants[9]*(power(states[2], 3.00000))*states[3]*states[4]*(states[0]-algebraic[17])
    algebraic[63] = (constants[3]*constants[25]*(exp((constants[29]*constants[2]*states[0])/(constants[0]*constants[1]))*(power(states[1], 3.00000))*constants[24]-exp(((constants[29]-1.00000)*constants[2]*states[0])/(constants[0]*constants[1]))*(power(constants[10], 3.00000))*states[12]))/((power(constants[26], 3.00000)+power(constants[10], 3.00000))*(constants[27]+constants[24])*(1.00000+constants[28]*exp(((constants[29]-1.00000)*states[0]*constants[2])/(constants[0]*constants[1]))))
    algebraic[61] = constants[3]*constants[21]*(states[0]-algebraic[17])
    rates[1] = (-3.00000*algebraic[58]-(3.00000*algebraic[63]+algebraic[61]+algebraic[30]))/(constants[43]*constants[2])
    algebraic[0] = custom_piecewise([greater_equal(voi , constants[4]) & less_equal(voi , constants[5]) & less_equal((voi-constants[4])-floor((voi-constants[4])/constants[6])*constants[6] , constants[7]), constants[8] , True, 0.00000])
    algebraic[56] = constants[3]*constants[17]*states[13]*states[14]*states[15]*(states[0]-65.0000)
    algebraic[64] = (constants[3]*constants[30]*states[12])/(0.000500000+states[12])
    algebraic[59] = ((constants[0]*constants[1])/(2.00000*constants[2]))*log(constants[24]/states[12])
    algebraic[62] = constants[3]*constants[22]*(states[0]-algebraic[59])
    rates[0] = -(algebraic[30]+algebraic[50]+algebraic[51]+algebraic[53]+algebraic[54]+algebraic[55]+algebraic[61]+algebraic[62]+algebraic[58]+algebraic[64]+algebraic[63]+algebraic[56]+algebraic[0])/constants[3]
    algebraic[65] = constants[31]*(power(states[17], 2.00000))*states[18]*states[19]*(states[16]-states[12])
    algebraic[67] = (states[20]-states[16])/constants[32]
    rates[16] = (algebraic[67]-algebraic[65])*(power(1.00000+(constants[38]*constants[41])/(power(states[16]+constants[41], 2.00000)), -1.00000))
    algebraic[66] = 1000.00*(1.00000e-15*constants[47]*algebraic[65]-(1.00000e-15/(2.00000*constants[2]))*(0.500000*algebraic[56]-0.200000*algebraic[63]))
    algebraic[68] = power(1.00000+exp(-(algebraic[66]-3.41750e-13)/1.36700e-15), -1.00000)
    rates[17] = (algebraic[68]-states[17])/constants[46]
    algebraic[69] = 1.91000+2.09000*(power(1.00000+exp(-(algebraic[66]-3.41750e-13)/1.36700e-15), -1.00000))
    algebraic[71] = 1.00000-power(1.00000+exp(-(algebraic[66]-6.83500e-14)/1.36700e-15), -1.00000)
    rates[18] = (algebraic[71]-states[18])/algebraic[69]
    algebraic[70] = constants[33]/(1.00000+constants[34]/states[12])
    algebraic[72] = (constants[33]*states[20])/constants[35]
    rates[20] = algebraic[70]-(algebraic[72]+(algebraic[67]*constants[47])/constants[48])
    algebraic[73] = (2.00000*algebraic[63]-(algebraic[64]+algebraic[56]+algebraic[62]))/(2.00000*constants[43]*constants[2])+(constants[48]*(algebraic[72]-algebraic[70])+algebraic[65]*constants[47])/constants[43]
    algebraic[74] = 1.00000+(constants[37]*constants[40])/(power(states[12]+constants[40], 2.00000))+(constants[36]*constants[39])/(power(states[12]+constants[39], 2.00000))
    rates[12] = algebraic[73]/algebraic[74]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[12] = power(1.00000+states[12]/0.000350000, -1.00000)
    algebraic[10] = power(1.00000+exp((states[0]+10.0000)/-8.00000), -1.00000)
    algebraic[27] = custom_piecewise([less(fabs(states[0]+10.0000) , 1.00000e-10), 4.57900/(1.00000+exp((states[0]+10.0000)/-6.24000)) , True, (1.00000-exp((states[0]+10.0000)/-6.24000))/(0.0350000*(states[0]+10.0000)*(1.00000+exp((states[0]+10.0000)/-6.24000)))])
    algebraic[11] = exp(-(states[0]+28.0000)/6.90000)/(1.00000+exp(-(states[0]+28.0000)/6.90000))
    algebraic[28] = 9.00000*(power(0.0197000*exp(-(power(0.0337000, 2.00000))*(power(states[0]+10.0000, 2.00000)))+0.0200000, -1.00000))
    algebraic[13] = custom_piecewise([less(fabs(states[0]-7.90000) , 1.00000e-10), (6.00000*0.200000)/1.30000 , True, (6.00000*(1.00000-exp(-(states[0]-7.90000)/5.00000)))/((1.00000+0.300000*exp(-(states[0]-7.90000)/5.00000))*1.00000*(states[0]-7.90000))])
    algebraic[29] = 1.00000-power(1.00000+exp(-(states[0]-40.0000)/17.0000), -1.00000)
    algebraic[1] = custom_piecewise([equal(states[0] , -47.1300), 3.20000 , True, (0.320000*(states[0]+47.1300))/(1.00000-exp(-0.100000*(states[0]+47.1300)))])
    algebraic[18] = 0.0800000*exp(-states[0]/11.0000)
    algebraic[31] = algebraic[1]/(algebraic[1]+algebraic[18])
    algebraic[41] = 1.00000/(algebraic[1]+algebraic[18])
    algebraic[2] = custom_piecewise([less(states[0] , -40.0000), 0.135000*exp((states[0]+80.0000)/-6.80000) , True, 0.00000])
    algebraic[19] = custom_piecewise([less(states[0] , -40.0000), 3.56000*exp(0.0790000*states[0])+310000.*exp(0.350000*states[0]) , True, 1.00000/(0.130000*(1.00000+exp((states[0]+10.6600)/-11.1000)))])
    algebraic[32] = algebraic[2]/(algebraic[2]+algebraic[19])
    algebraic[42] = 1.00000/(algebraic[2]+algebraic[19])
    algebraic[3] = custom_piecewise([less(states[0] , -40.0000), ((-127140.*exp(0.244400*states[0])-3.47400e-05*exp(-0.0439100*states[0]))*(states[0]+37.7800))/(1.00000+exp(0.311000*(states[0]+79.2300))) , True, 0.00000])
    algebraic[20] = custom_piecewise([less(states[0] , -40.0000), (0.121200*exp(-0.0105200*states[0]))/(1.00000+exp(-0.137800*(states[0]+40.1400))) , True, (0.300000*exp(-2.53500e-07*states[0]))/(1.00000+exp(-0.100000*(states[0]+32.0000)))])
    algebraic[33] = algebraic[3]/(algebraic[3]+algebraic[20])
    algebraic[43] = 1.00000/(algebraic[3]+algebraic[20])
    algebraic[4] = 0.650000*(power(exp((states[0]--10.0000)/-8.50000)+exp(((states[0]--10.0000)-40.0000)/-59.0000), -1.00000))
    algebraic[21] = 0.650000*(power(2.50000+exp(((states[0]--10.0000)+72.0000)/17.0000), -1.00000))
    algebraic[34] = (power(algebraic[4]+algebraic[21], -1.00000))/constants[13]
    algebraic[44] = power(1.00000+exp(((states[0]--10.0000)+10.4700)/-17.5400), -1.00000)
    algebraic[5] = power(18.5300+1.00000*exp(((states[0]--10.0000)+103.700)/10.9500), -1.00000)
    algebraic[22] = power(35.5600+1.00000*exp(((states[0]--10.0000)-8.74000)/-7.44000), -1.00000)
    algebraic[35] = (power(algebraic[5]+algebraic[22], -1.00000))/constants[13]
    algebraic[45] = power(1.00000+exp(((states[0]--10.0000)+33.1000)/5.30000), -1.00000)
    algebraic[6] = 0.650000*(power(exp((states[0]--10.0000)/-8.50000)+exp(((states[0]--10.0000)-40.0000)/-59.0000), -1.00000))
    algebraic[23] = 0.650000*(power(2.50000+exp(((states[0]--10.0000)+72.0000)/17.0000), -1.00000))
    algebraic[36] = (power(algebraic[6]+algebraic[23], -1.00000))/constants[13]
    algebraic[46] = power(1.00000+exp(((states[0]--10.0000)+20.3000)/-9.60000), -1.00000)
    algebraic[7] = power(21.0000+1.00000*exp(((states[0]--10.0000)-195.000)/-28.0000), -1.00000)
    algebraic[24] = 1.00000/exp(((states[0]--10.0000)-168.000)/-16.0000)
    algebraic[37] = (power(algebraic[7]+algebraic[24], -1.00000))/constants[13]
    algebraic[47] = power(1.00000+exp(((states[0]--10.0000)-109.450)/27.4800), -1.00000)
    algebraic[8] = custom_piecewise([less(fabs(states[0]+14.1000) , 1.00000e-10), 0.00150000 , True, (0.000300000*(states[0]+14.1000))/(1.00000-exp((states[0]+14.1000)/-5.00000))])
    algebraic[25] = custom_piecewise([less(fabs(states[0]-3.33280) , 1.00000e-10), 0.000378361 , True, (7.38980e-05*(states[0]-3.33280))/(exp((states[0]-3.33280)/5.12370)-1.00000)])
    algebraic[38] = power(algebraic[8]+algebraic[25], -1.00000)
    algebraic[48] = power(1.00000+exp((states[0]+14.1000)/-6.50000), -1.00000)
    algebraic[9] = custom_piecewise([less(fabs(states[0]-19.9000) , 1.00000e-10), 0.000680000 , True, (4.00000e-05*(states[0]-19.9000))/(1.00000-exp((states[0]-19.9000)/-17.0000))])
    algebraic[26] = custom_piecewise([less(fabs(states[0]-19.9000) , 1.00000e-10), 0.000315000 , True, (3.50000e-05*(states[0]-19.9000))/(exp((states[0]-19.9000)/9.00000)-1.00000)])
    algebraic[39] = 0.500000*(power(algebraic[9]+algebraic[26], -1.00000))
    algebraic[49] = power(1.00000+exp((states[0]-19.9000)/-12.7000), -0.500000)
    algebraic[40] = ((constants[0]*constants[1])/constants[2])*log(constants[12]/states[5])
    algebraic[50] = (constants[3]*constants[11]*(states[0]-algebraic[40]))/(1.00000+exp(0.0700000*(states[0]+80.0000)))
    algebraic[51] = constants[3]*constants[14]*(power(states[6], 3.00000))*states[7]*(states[0]-algebraic[40])
    algebraic[52] = 0.00500000+0.0500000/(1.00000+exp((states[0]-15.0000)/-13.0000))
    algebraic[53] = constants[3]*algebraic[52]*(power(states[8], 3.00000))*states[9]*(states[0]-algebraic[40])
    algebraic[54] = (constants[3]*constants[15]*states[10]*(states[0]-algebraic[40]))/(1.00000+exp((states[0]+15.0000)/22.4000))
    algebraic[55] = constants[3]*constants[16]*(power(states[11], 2.00000))*(states[0]-algebraic[40])
    algebraic[57] = power(1.00000+0.124500*exp((-0.100000*constants[2]*states[0])/(constants[0]*constants[1]))+0.0365000*constants[45]*exp((-constants[2]*states[0])/(constants[0]*constants[1])), -1.00000)
    algebraic[58] = (((constants[3]*constants[20]*algebraic[57]*1.00000)/(1.00000+power(constants[18]/states[1], 1.50000)))*constants[12])/(constants[12]+constants[19])
    algebraic[60] = constants[3]*constants[23]*(states[0]-algebraic[40])
    algebraic[17] = ((constants[0]*constants[1])/constants[2])*log(constants[10]/states[1])
    algebraic[30] = constants[3]*constants[9]*(power(states[2], 3.00000))*states[3]*states[4]*(states[0]-algebraic[17])
    algebraic[63] = (constants[3]*constants[25]*(exp((constants[29]*constants[2]*states[0])/(constants[0]*constants[1]))*(power(states[1], 3.00000))*constants[24]-exp(((constants[29]-1.00000)*constants[2]*states[0])/(constants[0]*constants[1]))*(power(constants[10], 3.00000))*states[12]))/((power(constants[26], 3.00000)+power(constants[10], 3.00000))*(constants[27]+constants[24])*(1.00000+constants[28]*exp(((constants[29]-1.00000)*states[0]*constants[2])/(constants[0]*constants[1]))))
    algebraic[61] = constants[3]*constants[21]*(states[0]-algebraic[17])
    algebraic[0] = custom_piecewise([greater_equal(voi , constants[4]) & less_equal(voi , constants[5]) & less_equal((voi-constants[4])-floor((voi-constants[4])/constants[6])*constants[6] , constants[7]), constants[8] , True, 0.00000])
    algebraic[56] = constants[3]*constants[17]*states[13]*states[14]*states[15]*(states[0]-65.0000)
    algebraic[64] = (constants[3]*constants[30]*states[12])/(0.000500000+states[12])
    algebraic[59] = ((constants[0]*constants[1])/(2.00000*constants[2]))*log(constants[24]/states[12])
    algebraic[62] = constants[3]*constants[22]*(states[0]-algebraic[59])
    algebraic[65] = constants[31]*(power(states[17], 2.00000))*states[18]*states[19]*(states[16]-states[12])
    algebraic[67] = (states[20]-states[16])/constants[32]
    algebraic[66] = 1000.00*(1.00000e-15*constants[47]*algebraic[65]-(1.00000e-15/(2.00000*constants[2]))*(0.500000*algebraic[56]-0.200000*algebraic[63]))
    algebraic[68] = power(1.00000+exp(-(algebraic[66]-3.41750e-13)/1.36700e-15), -1.00000)
    algebraic[69] = 1.91000+2.09000*(power(1.00000+exp(-(algebraic[66]-3.41750e-13)/1.36700e-15), -1.00000))
    algebraic[71] = 1.00000-power(1.00000+exp(-(algebraic[66]-6.83500e-14)/1.36700e-15), -1.00000)
    algebraic[70] = constants[33]/(1.00000+constants[34]/states[12])
    algebraic[72] = (constants[33]*states[20])/constants[35]
    algebraic[73] = (2.00000*algebraic[63]-(algebraic[64]+algebraic[56]+algebraic[62]))/(2.00000*constants[43]*constants[2])+(constants[48]*(algebraic[72]-algebraic[70])+algebraic[65]*constants[47])/constants[43]
    algebraic[74] = 1.00000+(constants[37]*constants[40])/(power(states[12]+constants[40], 2.00000))+(constants[36]*constants[39])/(power(states[12]+constants[39], 2.00000))
    algebraic[14] = (constants[36]*states[12])/(states[12]+constants[39])
    algebraic[15] = (constants[37]*states[12])/(states[12]+constants[40])
    algebraic[16] = (constants[38]*states[16])/(states[16]+constants[41])
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