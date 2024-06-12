# Size of variable arrays:
sizeAlgebraic = 49
sizeStates = 18
sizeConstants = 46
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_constants[0] = "R in component constants (joule_per_kilomole_kelvin)"
    legend_constants[1] = "T in component constants (kelvin)"
    legend_constants[2] = "F in component constants (coulomb_per_mole)"
    legend_states[0] = "E in component membrane (millivolt)"
    legend_constants[3] = "C in component membrane (nanoF)"
    legend_algebraic[47] = "i_tot in component membrane (picoA)"
    legend_algebraic[18] = "i_CaL in component L_type_calcium_current (picoA)"
    legend_algebraic[19] = "i_CaT in component T_type_calcium_current (picoA)"
    legend_algebraic[20] = "i_Na in component fast_sodium_current (picoA)"
    legend_algebraic[23] = "i_K in component delayed_rectifying_potassium_current (picoA)"
    legend_algebraic[26] = "i_f in component hyperpolarising_activated_current (picoA)"
    legend_algebraic[27] = "i_p in component sodium_potassium_pump (picoA)"
    legend_algebraic[42] = "i_NaCa in component sodium_calcium_exchange_current (picoA)"
    legend_algebraic[43] = "i_bNa in component background_sodium_current (picoA)"
    legend_algebraic[45] = "i_bK in component background_potassium_current (picoA)"
    legend_algebraic[8] = "E_Ca in component reversal_potentials (millivolt)"
    legend_algebraic[16] = "E_Na in component reversal_potentials (millivolt)"
    legend_algebraic[17] = "E_K in component reversal_potentials (millivolt)"
    legend_states[1] = "Cai in component ion_concentrations (millimolar)"
    legend_states[2] = "Cao in component ion_concentrations (millimolar)"
    legend_states[3] = "Nai in component ion_concentrations (millimolar)"
    legend_states[4] = "Nao in component ion_concentrations (millimolar)"
    legend_states[5] = "Ki in component ion_concentrations (millimolar)"
    legend_states[6] = "Ko in component ion_concentrations (millimolar)"
    legend_constants[4] = "g_CaL in component L_type_calcium_current (nanoS)"
    legend_states[7] = "dL in component L_type_calcium_current_d_gate (dimensionless)"
    legend_states[8] = "fL in component L_type_calcium_current_f_gate (dimensionless)"
    legend_states[9] = "fL2 in component L_type_calcium_current_f2_gate (dimensionless)"
    legend_algebraic[0] = "dL_infinity in component L_type_calcium_current_d_gate (dimensionless)"
    legend_constants[5] = "tau_dL in component L_type_calcium_current_d_gate (second)"
    legend_algebraic[1] = "fL_infinity in component L_type_calcium_current_f_gate (dimensionless)"
    legend_algebraic[9] = "tau_fL in component L_type_calcium_current_f_gate (second)"
    legend_constants[6] = "alpha_fL2 in component L_type_calcium_current_f2_gate (per_second)"
    legend_constants[7] = "beta_fL2 in component L_type_calcium_current_f2_gate (per_millimolar_second)"
    legend_constants[8] = "g_CaT in component T_type_calcium_current (nanoS)"
    legend_states[10] = "dT in component T_type_calcium_current_d_gate (dimensionless)"
    legend_states[11] = "fT in component T_type_calcium_current_f_gate (dimensionless)"
    legend_algebraic[2] = "dT_infinity in component T_type_calcium_current_d_gate (dimensionless)"
    legend_algebraic[10] = "tau_dT in component T_type_calcium_current_d_gate (second)"
    legend_algebraic[3] = "fT_infinity in component T_type_calcium_current_f_gate (dimensionless)"
    legend_algebraic[11] = "tau_fT in component T_type_calcium_current_f_gate (second)"
    legend_constants[9] = "g_Na in component fast_sodium_current (nanoS)"
    legend_states[12] = "m in component fast_sodium_current_m_gate (dimensionless)"
    legend_states[13] = "h in component fast_sodium_current_h_gate (dimensionless)"
    legend_algebraic[4] = "alpha_m in component fast_sodium_current_m_gate (per_second)"
    legend_algebraic[12] = "beta_m in component fast_sodium_current_m_gate (per_second)"
    legend_algebraic[5] = "alpha_h in component fast_sodium_current_h_gate (per_second)"
    legend_algebraic[13] = "beta_h in component fast_sodium_current_h_gate (per_second)"
    legend_algebraic[21] = "i_KK in component delayed_rectifying_potassium_current (picoA)"
    legend_algebraic[22] = "i_KNa in component delayed_rectifying_potassium_current (picoA)"
    legend_constants[10] = "Kk in component delayed_rectifying_potassium_current (picoA_per_millimolar)"
    legend_constants[11] = "P_KNa in component delayed_rectifying_potassium_current (dimensionless)"
    legend_states[14] = "x in component delayed_rectifying_potassium_current_x_gate (dimensionless)"
    legend_algebraic[6] = "x_infinity in component delayed_rectifying_potassium_current_x_gate (dimensionless)"
    legend_algebraic[14] = "tau_x in component delayed_rectifying_potassium_current_x_gate (second)"
    legend_algebraic[24] = "i_fNa in component hyperpolarising_activated_current (picoA)"
    legend_algebraic[25] = "i_fK in component hyperpolarising_activated_current (picoA)"
    legend_constants[12] = "Kmf in component hyperpolarising_activated_current (millimolar)"
    legend_constants[13] = "g_fNa in component hyperpolarising_activated_current (nanoS)"
    legend_constants[14] = "g_fK in component hyperpolarising_activated_current (nanoS)"
    legend_states[15] = "y in component hyperpolarising_activated_current_y_gate (dimensionless)"
    legend_algebraic[7] = "alpha_y in component hyperpolarising_activated_current_y_gate (per_second)"
    legend_algebraic[15] = "beta_y in component hyperpolarising_activated_current_y_gate (per_second)"
    legend_constants[15] = "KmNa in component sodium_potassium_pump (millimolar)"
    legend_constants[16] = "KmK in component sodium_potassium_pump (millimolar)"
    legend_constants[17] = "i_pmax in component sodium_potassium_pump (picoA)"
    legend_constants[18] = "kNaCa in component sodium_calcium_exchange_current (picoA)"
    legend_algebraic[38] = "x1 in component sodium_calcium_exchange_current (dimensionless)"
    legend_algebraic[39] = "x2 in component sodium_calcium_exchange_current (dimensionless)"
    legend_algebraic[40] = "x3 in component sodium_calcium_exchange_current (dimensionless)"
    legend_algebraic[41] = "x4 in component sodium_calcium_exchange_current (dimensionless)"
    legend_algebraic[34] = "k41 in component sodium_calcium_exchange_current (dimensionless)"
    legend_algebraic[32] = "k34 in component sodium_calcium_exchange_current (dimensionless)"
    legend_algebraic[30] = "k23 in component sodium_calcium_exchange_current (dimensionless)"
    legend_algebraic[31] = "k21 in component sodium_calcium_exchange_current (dimensionless)"
    legend_algebraic[29] = "k32 in component sodium_calcium_exchange_current (dimensionless)"
    legend_algebraic[37] = "k43 in component sodium_calcium_exchange_current (dimensionless)"
    legend_algebraic[36] = "k12 in component sodium_calcium_exchange_current (dimensionless)"
    legend_algebraic[35] = "k14 in component sodium_calcium_exchange_current (dimensionless)"
    legend_constants[19] = "Qci in component sodium_calcium_exchange_current (dimensionless)"
    legend_constants[20] = "Qn in component sodium_calcium_exchange_current (dimensionless)"
    legend_constants[21] = "Qco in component sodium_calcium_exchange_current (dimensionless)"
    legend_constants[22] = "K3ni in component sodium_calcium_exchange_current (millimolar)"
    legend_constants[23] = "Kci in component sodium_calcium_exchange_current (millimolar)"
    legend_constants[24] = "K1ni in component sodium_calcium_exchange_current (millimolar)"
    legend_constants[25] = "K2ni in component sodium_calcium_exchange_current (millimolar)"
    legend_constants[26] = "Kcni in component sodium_calcium_exchange_current (millimolar)"
    legend_constants[27] = "K3no in component sodium_calcium_exchange_current (millimolar)"
    legend_constants[28] = "K1no in component sodium_calcium_exchange_current (millimolar)"
    legend_constants[29] = "K2no in component sodium_calcium_exchange_current (millimolar)"
    legend_constants[30] = "Kco in component sodium_calcium_exchange_current (millimolar)"
    legend_algebraic[28] = "do in component sodium_calcium_exchange_current (dimensionless)"
    legend_algebraic[33] = "di in component sodium_calcium_exchange_current (dimensionless)"
    legend_constants[31] = "g_Nab in component background_sodium_current (nanoS)"
    legend_constants[32] = "KbK in component background_potassium_current (picoA_per_millimolar)"
    legend_algebraic[44] = "i_up in component sarcoplasmic_reticulum_kinetics (picoA)"
    legend_algebraic[46] = "i_tr in component sarcoplasmic_reticulum_kinetics (picoA)"
    legend_algebraic[48] = "i_rel in component sarcoplasmic_reticulum_kinetics (picoA)"
    legend_constants[33] = "V_i in component ion_concentrations (microlitre)"
    legend_constants[43] = "V_rel in component sarcoplasmic_reticulum_kinetics (microlitre)"
    legend_constants[45] = "V_up in component sarcoplasmic_reticulum_kinetics (microlitre)"
    legend_constants[34] = "i_up_max in component sarcoplasmic_reticulum_kinetics (picoA)"
    legend_constants[35] = "KmCaup in component sarcoplasmic_reticulum_kinetics (millimolar)"
    legend_constants[36] = "KmCarel in component sarcoplasmic_reticulum_kinetics (millimolar)"
    legend_constants[37] = "tau_rel in component sarcoplasmic_reticulum_kinetics (second)"
    legend_constants[38] = "tau_tr in component sarcoplasmic_reticulum_kinetics (second)"
    legend_states[16] = "Caup in component ion_concentrations (millimolar)"
    legend_states[17] = "Carel in component ion_concentrations (millimolar)"
    legend_constants[44] = "V_e in component ion_concentrations (microlitre)"
    legend_constants[39] = "tau_b in component ion_concentrations (second)"
    legend_constants[40] = "Nab in component ion_concentrations (millimolar)"
    legend_constants[41] = "Cab in component ion_concentrations (millimolar)"
    legend_constants[42] = "Kb in component ion_concentrations (millimolar)"
    legend_rates[0] = "d/dt E in component membrane (millivolt)"
    legend_rates[7] = "d/dt dL in component L_type_calcium_current_d_gate (dimensionless)"
    legend_rates[8] = "d/dt fL in component L_type_calcium_current_f_gate (dimensionless)"
    legend_rates[9] = "d/dt fL2 in component L_type_calcium_current_f2_gate (dimensionless)"
    legend_rates[10] = "d/dt dT in component T_type_calcium_current_d_gate (dimensionless)"
    legend_rates[11] = "d/dt fT in component T_type_calcium_current_f_gate (dimensionless)"
    legend_rates[12] = "d/dt m in component fast_sodium_current_m_gate (dimensionless)"
    legend_rates[13] = "d/dt h in component fast_sodium_current_h_gate (dimensionless)"
    legend_rates[14] = "d/dt x in component delayed_rectifying_potassium_current_x_gate (dimensionless)"
    legend_rates[15] = "d/dt y in component hyperpolarising_activated_current_y_gate (dimensionless)"
    legend_rates[3] = "d/dt Nai in component ion_concentrations (millimolar)"
    legend_rates[4] = "d/dt Nao in component ion_concentrations (millimolar)"
    legend_rates[5] = "d/dt Ki in component ion_concentrations (millimolar)"
    legend_rates[6] = "d/dt Ko in component ion_concentrations (millimolar)"
    legend_rates[1] = "d/dt Cai in component ion_concentrations (millimolar)"
    legend_rates[2] = "d/dt Cao in component ion_concentrations (millimolar)"
    legend_rates[16] = "d/dt Caup in component ion_concentrations (millimolar)"
    legend_rates[17] = "d/dt Carel in component ion_concentrations (millimolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 8314.472
    constants[1] = 310
    constants[2] = 96485.3415
    states[0] = -64.9
    constants[3] = 3.2e-5
    states[1] = 0.000034
    states[2] = 2.0004
    states[3] = 7.4994
    states[4] = 139.9929
    states[5] = 140.0073
    states[6] = 5.4243
    constants[4] = 0.4
    states[7] = 0.0001
    states[8] = 0.1505
    states[9] = 0.219
    constants[5] = 0.002
    constants[6] = 3
    constants[7] = 40000
    constants[8] = 0.085
    states[10] = 0.001
    states[11] = 0.1328
    constants[9] = 0.25
    states[12] = 0.0139
    states[13] = 0.0087
    constants[10] = 0.00026
    constants[11] = 0.035
    states[14] = 0.5682
    constants[12] = 10.3
    constants[13] = 0.0081
    constants[14] = 0.0135
    states[15] = 0.0287
    constants[15] = 40
    constants[16] = 1
    constants[17] = 0.226
    constants[18] = 4
    constants[19] = 0.1369
    constants[20] = 0.4315
    constants[21] = 0
    constants[22] = 26.44
    constants[23] = 0.0207
    constants[24] = 395.3
    constants[25] = 2.289
    constants[26] = 26.44
    constants[27] = 4.663
    constants[28] = 1628
    constants[29] = 561.4
    constants[30] = 3.663
    constants[31] = 0.00024
    constants[32] = 0.00007
    constants[33] = 2.5e-6
    constants[34] = 0.0212
    constants[35] = 0.0005
    constants[36] = 0.001
    constants[37] = 0.005
    constants[38] = 0.4
    states[16] = 0.5832
    states[17] = 0.1101
    constants[39] = 0.1
    constants[40] = 140
    constants[41] = 2
    constants[42] = 5.4
    constants[43] = 0.00600000*constants[33]
    constants[44] = 0.200000*constants[33]
    constants[45] = 0.0140000*constants[33]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[9] = constants[6]*(1.00000-states[9])-constants[7]*states[1]*states[9]
    algebraic[0] = 1.00000/(1.00000+exp((states[0]+6.60000)/-6.60000))
    rates[7] = (algebraic[0]-states[7])/constants[5]
    algebraic[1] = 1.00000/(1.00000+exp((states[0]+25.0000)/6.00000))
    algebraic[9] = 0.0310000+1.00000/(1.00000+exp((states[0]+37.6000)/8.10000))
    rates[8] = (algebraic[1]-states[8])/algebraic[9]
    algebraic[2] = 1.00000/(1.00000+exp((states[0]+23.0000)/-6.10000))
    algebraic[10] = 0.000600000+0.00540000/(1.00000+exp(0.0300000*(states[0]+100.000)))
    rates[10] = (algebraic[2]-states[10])/algebraic[10]
    algebraic[3] = 1.00000/(1.00000+exp((states[0]+75.0000)/6.60000))
    algebraic[11] = 0.00100000+0.0400000/(1.00000+exp(0.0800000*(states[0]+65.0000)))
    rates[11] = (algebraic[3]-states[11])/algebraic[11]
    algebraic[4] = (200.000*(states[0]+34.3000))/(1.00000-exp(-0.0900000*(states[0]+34.3000)))
    algebraic[12] = 8000.00*exp(-0.150000*(states[0]+56.2000))
    rates[12] = algebraic[4]*(1.00000-states[12])-algebraic[12]*states[12]
    algebraic[5] = 32.4000*exp(-0.140000*(states[0]+93.4000))
    algebraic[13] = 709.000/(1.00000+4.20000*exp(-0.0600000*(states[0]+45.4000)))
    rates[13] = algebraic[5]*(1.00000-states[13])-algebraic[13]*states[13]
    algebraic[6] = 1.00000/(1.00000+exp((states[0]+25.1000)/-7.40000))
    algebraic[14] = 1.00000/(17.0000*exp(0.0398000*states[0])+0.211000*exp(-0.0510000*states[0]))
    rates[14] = (algebraic[6]-states[14])/algebraic[14]
    algebraic[7] = (0.360000*(states[0]+137.800))/(exp(0.0660000*(states[0]+137.800))-1.00000)
    algebraic[15] = (0.100000*(states[0]+76.3000))/(1.00000-exp(-0.210000*(states[0]+76.3000)))
    rates[15] = algebraic[7]*(1.00000-states[15])-algebraic[15]*states[15]
    algebraic[8] = ((constants[0]*constants[1])/(2.00000*constants[2]))*log(states[2]/states[1])
    algebraic[18] = constants[4]*states[7]*states[8]*states[9]*((states[0]-algebraic[8])+75.0000)
    algebraic[19] = constants[8]*states[10]*states[11]*((states[0]-algebraic[8])+75.0000)
    algebraic[34] = exp((-constants[20]*states[0]*constants[2])/(2.00000*constants[0]*constants[1]))
    algebraic[32] = states[4]/(constants[27]+states[4])
    algebraic[28] = 1.00000+states[2]/constants[30]+(states[2]/constants[30])*exp((constants[21]*states[0]*constants[2])/(constants[0]*constants[1]))+states[4]/constants[28]+(power(states[4], 2.00000))/(constants[28]*constants[29])+(power(states[4], 3.00000))/(constants[28]*constants[29]*constants[27])
    algebraic[30] = (((power(states[4], 2.00000))/(constants[28]*constants[29])+(power(states[4], 3.00000))/(constants[28]*constants[29]*constants[27]))*exp((-constants[20]*states[0]*constants[2])/(2.00000*constants[0]*constants[1])))/algebraic[28]
    algebraic[31] = ((states[2]/constants[30])*exp((-constants[21]*states[0]*constants[2])/(constants[0]*constants[1])))/algebraic[28]
    algebraic[29] = exp((constants[20]*states[0]*constants[2])/(2.00000*constants[0]*constants[1]))
    algebraic[37] = states[3]/(constants[22]+states[3])
    algebraic[38] = algebraic[34]*algebraic[32]*(algebraic[30]+algebraic[31])+algebraic[31]*algebraic[29]*(algebraic[37]+algebraic[34])
    algebraic[33] = 1.00000+states[1]/constants[23]+(states[1]/constants[23])*exp((-constants[19]*states[0]*constants[2])/(constants[0]*constants[1]))+(states[1]*states[3])/(constants[23]*constants[26])+states[3]/constants[24]+(power(states[3], 2.00000))/(constants[24]*constants[25])+(power(states[3], 3.00000))/(constants[24]*constants[25]*constants[22])
    algebraic[36] = ((states[1]/constants[23])*exp((-constants[19]*states[0]*constants[2])/(constants[0]*constants[1])))/algebraic[33]
    algebraic[35] = (((power(states[3], 2.00000))/(constants[24]*constants[25])+(power(states[3], 3.00000))/(constants[24]*constants[25]*constants[22]))*exp((constants[20]*states[0]*constants[2])/(2.00000*constants[0]*constants[1])))/algebraic[33]
    algebraic[39] = algebraic[29]*algebraic[37]*(algebraic[35]+algebraic[36])+algebraic[34]*algebraic[36]*(algebraic[32]+algebraic[29])
    algebraic[40] = algebraic[35]*algebraic[37]*(algebraic[30]+algebraic[31])+algebraic[36]*algebraic[30]*(algebraic[37]+algebraic[34])
    algebraic[41] = algebraic[30]*algebraic[32]*(algebraic[35]+algebraic[36])+algebraic[35]*algebraic[31]*(algebraic[32]+algebraic[29])
    algebraic[42] = (constants[18]*(algebraic[39]*algebraic[31]-algebraic[38]*algebraic[36]))/(algebraic[38]+algebraic[39]+algebraic[40]+algebraic[41])
    rates[2] = (1.00000*((algebraic[18]+algebraic[19])-2.00000*algebraic[42]))/(2.00000*constants[2]*1.00000*constants[44])+(constants[41]-states[2])/constants[39]
    algebraic[16] = ((constants[0]*constants[1])/constants[2])*log(states[4]/states[3])
    algebraic[20] = constants[9]*(power(states[12], 3.00000))*states[13]*(states[0]-algebraic[16])
    algebraic[27] = ((((constants[17]*states[3])/(states[3]+constants[15]))*states[6])/(states[6]+constants[16]))*(1.00000-power((states[0]-40.0000)/211.000, 2.00000))
    algebraic[43] = constants[31]*(states[0]-algebraic[16])
    algebraic[22] = states[14]*constants[10]*constants[11]*(power(states[6]/1.00000, 0.590000))*(states[3]-states[4]*exp((-states[0]*constants[2])/(constants[0]*constants[1])))
    algebraic[24] = ((states[15]*(power(states[6], 1.83000)))/(power(states[6], 1.83000)+power(constants[12], 1.83000)))*constants[13]*(states[0]-algebraic[16])
    rates[3] = (-1.00000*(algebraic[43]+algebraic[24]+algebraic[20]+3.00000*algebraic[27]+3.00000*algebraic[42]+algebraic[22]))/(constants[2]*1.00000*constants[33])
    rates[4] = (1.00000*(algebraic[43]+algebraic[24]+algebraic[20]+3.00000*algebraic[27]+3.00000*algebraic[42]+algebraic[22]))/(constants[2]*1.00000*constants[44])+(constants[40]-states[4])/constants[39]
    algebraic[45] = constants[32]*(power(states[6]/1.00000, 0.410000))*(states[5]-states[6]*exp((-states[0]*constants[2])/(constants[0]*constants[1])))
    algebraic[21] = states[14]*constants[10]*(power(states[6]/1.00000, 0.590000))*(states[5]-states[6]*exp((-states[0]*constants[2])/(constants[0]*constants[1])))
    algebraic[17] = ((constants[0]*constants[1])/constants[2])*log(states[6]/states[5])
    algebraic[25] = ((states[15]*(power(states[6], 1.83000)))/(power(states[6], 1.83000)+power(constants[12], 1.83000)))*constants[14]*(states[0]-algebraic[17])
    rates[5] = (-1.00000*(((algebraic[21]+algebraic[25])-2.00000*algebraic[27])+algebraic[45]))/(constants[2]*1.00000*constants[33])
    rates[6] = (1.00000*(((algebraic[21]+algebraic[25])-2.00000*algebraic[27])+algebraic[45]))/(constants[2]*1.00000*constants[44])+(constants[42]-states[6])/constants[39]
    algebraic[44] = (constants[34]*(power(states[1], 2.00000)))/(power(states[1], 2.00000)+power(constants[35], 2.00000))
    algebraic[46] = ((2.00000*1.00000*constants[43]*constants[2])/(1.00000*constants[38]))*states[16]
    rates[16] = (1.00000*(algebraic[44]-algebraic[46]))/(2.00000*1.00000*constants[45]*constants[2])
    algebraic[23] = algebraic[21]+algebraic[22]
    algebraic[26] = algebraic[25]+algebraic[24]
    algebraic[47] = algebraic[18]+algebraic[19]+algebraic[20]+algebraic[23]+algebraic[26]+algebraic[27]+algebraic[42]+algebraic[43]+algebraic[45]
    rates[0] = -algebraic[47]/constants[3]
    algebraic[48] = (((2.00000*1.00000*constants[43]*constants[2])/(1.00000*constants[37]))*states[17]*(power(states[1], 2.00000)))/(power(states[1], 2.00000)+power(constants[36], 2.00000))
    rates[1] = (-1.00000*(((algebraic[18]+algebraic[19])-2.00000*algebraic[42])+algebraic[44]+-algebraic[48]))/(2.00000*constants[2]*1.00000*constants[33])
    rates[17] = (1.00000*(algebraic[46]-algebraic[48]))/(2.00000*1.00000*constants[43]*constants[2])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = 1.00000/(1.00000+exp((states[0]+6.60000)/-6.60000))
    algebraic[1] = 1.00000/(1.00000+exp((states[0]+25.0000)/6.00000))
    algebraic[9] = 0.0310000+1.00000/(1.00000+exp((states[0]+37.6000)/8.10000))
    algebraic[2] = 1.00000/(1.00000+exp((states[0]+23.0000)/-6.10000))
    algebraic[10] = 0.000600000+0.00540000/(1.00000+exp(0.0300000*(states[0]+100.000)))
    algebraic[3] = 1.00000/(1.00000+exp((states[0]+75.0000)/6.60000))
    algebraic[11] = 0.00100000+0.0400000/(1.00000+exp(0.0800000*(states[0]+65.0000)))
    algebraic[4] = (200.000*(states[0]+34.3000))/(1.00000-exp(-0.0900000*(states[0]+34.3000)))
    algebraic[12] = 8000.00*exp(-0.150000*(states[0]+56.2000))
    algebraic[5] = 32.4000*exp(-0.140000*(states[0]+93.4000))
    algebraic[13] = 709.000/(1.00000+4.20000*exp(-0.0600000*(states[0]+45.4000)))
    algebraic[6] = 1.00000/(1.00000+exp((states[0]+25.1000)/-7.40000))
    algebraic[14] = 1.00000/(17.0000*exp(0.0398000*states[0])+0.211000*exp(-0.0510000*states[0]))
    algebraic[7] = (0.360000*(states[0]+137.800))/(exp(0.0660000*(states[0]+137.800))-1.00000)
    algebraic[15] = (0.100000*(states[0]+76.3000))/(1.00000-exp(-0.210000*(states[0]+76.3000)))
    algebraic[8] = ((constants[0]*constants[1])/(2.00000*constants[2]))*log(states[2]/states[1])
    algebraic[18] = constants[4]*states[7]*states[8]*states[9]*((states[0]-algebraic[8])+75.0000)
    algebraic[19] = constants[8]*states[10]*states[11]*((states[0]-algebraic[8])+75.0000)
    algebraic[34] = exp((-constants[20]*states[0]*constants[2])/(2.00000*constants[0]*constants[1]))
    algebraic[32] = states[4]/(constants[27]+states[4])
    algebraic[28] = 1.00000+states[2]/constants[30]+(states[2]/constants[30])*exp((constants[21]*states[0]*constants[2])/(constants[0]*constants[1]))+states[4]/constants[28]+(power(states[4], 2.00000))/(constants[28]*constants[29])+(power(states[4], 3.00000))/(constants[28]*constants[29]*constants[27])
    algebraic[30] = (((power(states[4], 2.00000))/(constants[28]*constants[29])+(power(states[4], 3.00000))/(constants[28]*constants[29]*constants[27]))*exp((-constants[20]*states[0]*constants[2])/(2.00000*constants[0]*constants[1])))/algebraic[28]
    algebraic[31] = ((states[2]/constants[30])*exp((-constants[21]*states[0]*constants[2])/(constants[0]*constants[1])))/algebraic[28]
    algebraic[29] = exp((constants[20]*states[0]*constants[2])/(2.00000*constants[0]*constants[1]))
    algebraic[37] = states[3]/(constants[22]+states[3])
    algebraic[38] = algebraic[34]*algebraic[32]*(algebraic[30]+algebraic[31])+algebraic[31]*algebraic[29]*(algebraic[37]+algebraic[34])
    algebraic[33] = 1.00000+states[1]/constants[23]+(states[1]/constants[23])*exp((-constants[19]*states[0]*constants[2])/(constants[0]*constants[1]))+(states[1]*states[3])/(constants[23]*constants[26])+states[3]/constants[24]+(power(states[3], 2.00000))/(constants[24]*constants[25])+(power(states[3], 3.00000))/(constants[24]*constants[25]*constants[22])
    algebraic[36] = ((states[1]/constants[23])*exp((-constants[19]*states[0]*constants[2])/(constants[0]*constants[1])))/algebraic[33]
    algebraic[35] = (((power(states[3], 2.00000))/(constants[24]*constants[25])+(power(states[3], 3.00000))/(constants[24]*constants[25]*constants[22]))*exp((constants[20]*states[0]*constants[2])/(2.00000*constants[0]*constants[1])))/algebraic[33]
    algebraic[39] = algebraic[29]*algebraic[37]*(algebraic[35]+algebraic[36])+algebraic[34]*algebraic[36]*(algebraic[32]+algebraic[29])
    algebraic[40] = algebraic[35]*algebraic[37]*(algebraic[30]+algebraic[31])+algebraic[36]*algebraic[30]*(algebraic[37]+algebraic[34])
    algebraic[41] = algebraic[30]*algebraic[32]*(algebraic[35]+algebraic[36])+algebraic[35]*algebraic[31]*(algebraic[32]+algebraic[29])
    algebraic[42] = (constants[18]*(algebraic[39]*algebraic[31]-algebraic[38]*algebraic[36]))/(algebraic[38]+algebraic[39]+algebraic[40]+algebraic[41])
    algebraic[16] = ((constants[0]*constants[1])/constants[2])*log(states[4]/states[3])
    algebraic[20] = constants[9]*(power(states[12], 3.00000))*states[13]*(states[0]-algebraic[16])
    algebraic[27] = ((((constants[17]*states[3])/(states[3]+constants[15]))*states[6])/(states[6]+constants[16]))*(1.00000-power((states[0]-40.0000)/211.000, 2.00000))
    algebraic[43] = constants[31]*(states[0]-algebraic[16])
    algebraic[22] = states[14]*constants[10]*constants[11]*(power(states[6]/1.00000, 0.590000))*(states[3]-states[4]*exp((-states[0]*constants[2])/(constants[0]*constants[1])))
    algebraic[24] = ((states[15]*(power(states[6], 1.83000)))/(power(states[6], 1.83000)+power(constants[12], 1.83000)))*constants[13]*(states[0]-algebraic[16])
    algebraic[45] = constants[32]*(power(states[6]/1.00000, 0.410000))*(states[5]-states[6]*exp((-states[0]*constants[2])/(constants[0]*constants[1])))
    algebraic[21] = states[14]*constants[10]*(power(states[6]/1.00000, 0.590000))*(states[5]-states[6]*exp((-states[0]*constants[2])/(constants[0]*constants[1])))
    algebraic[17] = ((constants[0]*constants[1])/constants[2])*log(states[6]/states[5])
    algebraic[25] = ((states[15]*(power(states[6], 1.83000)))/(power(states[6], 1.83000)+power(constants[12], 1.83000)))*constants[14]*(states[0]-algebraic[17])
    algebraic[44] = (constants[34]*(power(states[1], 2.00000)))/(power(states[1], 2.00000)+power(constants[35], 2.00000))
    algebraic[46] = ((2.00000*1.00000*constants[43]*constants[2])/(1.00000*constants[38]))*states[16]
    algebraic[23] = algebraic[21]+algebraic[22]
    algebraic[26] = algebraic[25]+algebraic[24]
    algebraic[47] = algebraic[18]+algebraic[19]+algebraic[20]+algebraic[23]+algebraic[26]+algebraic[27]+algebraic[42]+algebraic[43]+algebraic[45]
    algebraic[48] = (((2.00000*1.00000*constants[43]*constants[2])/(1.00000*constants[37]))*states[17]*(power(states[1], 2.00000)))/(power(states[1], 2.00000)+power(constants[36], 2.00000))
    return algebraic

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