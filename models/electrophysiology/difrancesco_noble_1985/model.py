# Size of variable arrays:
sizeAlgebraic = 45
sizeStates = 16
sizeConstants = 50
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
    legend_constants[45] = "RTONF in component membrane (millivolt)"
    legend_constants[3] = "C in component membrane (microF)"
    legend_constants[4] = "i_pulse in component membrane (nanoA)"
    legend_algebraic[25] = "i_f in component hyperpolarising_activated_current (nanoA)"
    legend_algebraic[27] = "i_K in component time_dependent_potassium_current (nanoA)"
    legend_algebraic[28] = "i_K1 in component time_independent_potassium_current (nanoA)"
    legend_algebraic[29] = "i_to in component transient_outward_current (nanoA)"
    legend_algebraic[30] = "i_Na_b in component sodium_background_current (nanoA)"
    legend_algebraic[32] = "i_Ca_b in component calcium_background_current (nanoA)"
    legend_algebraic[33] = "i_p in component sodium_potassium_pump (nanoA)"
    legend_algebraic[34] = "i_NaCa in component Na_Ca_exchanger (nanoA)"
    legend_algebraic[36] = "i_Na in component fast_sodium_current (nanoA)"
    legend_algebraic[43] = "i_si in component second_inward_current (nanoA)"
    legend_algebraic[23] = "i_fNa in component hyperpolarising_activated_current (nanoA)"
    legend_algebraic[0] = "E_Na in component hyperpolarising_activated_current (millivolt)"
    legend_algebraic[10] = "E_K in component hyperpolarising_activated_current (millivolt)"
    legend_algebraic[24] = "i_fK in component hyperpolarising_activated_current (nanoA)"
    legend_constants[5] = "g_f_Na in component hyperpolarising_activated_current (microS)"
    legend_constants[6] = "g_f_K in component hyperpolarising_activated_current (microS)"
    legend_constants[7] = "Km_f in component hyperpolarising_activated_current (millimolar)"
    legend_states[1] = "Kc in component extracellular_potassium_concentration (millimolar)"
    legend_states[2] = "Ki in component intracellular_potassium_concentration (millimolar)"
    legend_states[3] = "Nai in component intracellular_sodium_concentration (millimolar)"
    legend_constants[8] = "Nao in component extracellular_sodium_concentration (millimolar)"
    legend_states[4] = "y in component hyperpolarising_activated_current_y_gate (dimensionless)"
    legend_algebraic[1] = "alpha_y in component hyperpolarising_activated_current_y_gate (per_second)"
    legend_algebraic[19] = "beta_y in component hyperpolarising_activated_current_y_gate (per_second)"
    legend_constants[9] = "delta_y in component hyperpolarising_activated_current_y_gate (millivolt)"
    legend_algebraic[11] = "E0_y in component hyperpolarising_activated_current_y_gate (millivolt)"
    legend_algebraic[26] = "I_K in component time_dependent_potassium_current (nanoA)"
    legend_constants[10] = "i_K_max in component time_dependent_potassium_current (nanoA)"
    legend_states[5] = "x in component time_dependent_potassium_current_x_gate (dimensionless)"
    legend_algebraic[2] = "alpha_x in component time_dependent_potassium_current_x_gate (per_second)"
    legend_algebraic[12] = "beta_x in component time_dependent_potassium_current_x_gate (per_second)"
    legend_constants[11] = "g_K1 in component time_independent_potassium_current (microS)"
    legend_constants[12] = "Km_K1 in component time_independent_potassium_current (millimolar)"
    legend_constants[13] = "Km_to in component transient_outward_current (millimolar)"
    legend_constants[14] = "Km_Ca in component transient_outward_current (millimolar)"
    legend_constants[15] = "g_to in component transient_outward_current (microS_per_millimolar)"
    legend_states[6] = "Cai in component intracellular_calcium_concentration (millimolar)"
    legend_states[7] = "s in component transient_outward_current_s_gate (dimensionless)"
    legend_algebraic[3] = "alpha_s in component transient_outward_current_s_gate (per_second)"
    legend_algebraic[13] = "beta_s in component transient_outward_current_s_gate (per_second)"
    legend_constants[16] = "g_Nab in component sodium_background_current (microS)"
    legend_algebraic[31] = "E_Ca in component calcium_background_current (millivolt)"
    legend_constants[17] = "g_Cab in component calcium_background_current (microS)"
    legend_constants[18] = "Cao in component extracellular_calcium_concentration (millimolar)"
    legend_constants[19] = "I_p in component sodium_potassium_pump (nanoA)"
    legend_constants[20] = "K_mK in component sodium_potassium_pump (millimolar)"
    legend_constants[21] = "K_mNa in component sodium_potassium_pump (millimolar)"
    legend_constants[22] = "n_NaCa in component Na_Ca_exchanger (dimensionless)"
    legend_constants[23] = "K_NaCa in component Na_Ca_exchanger (nanoA)"
    legend_constants[24] = "d_NaCa in component Na_Ca_exchanger (dimensionless)"
    legend_constants[25] = "gamma in component Na_Ca_exchanger (dimensionless)"
    legend_constants[26] = "g_Na in component fast_sodium_current (microS)"
    legend_algebraic[35] = "E_mh in component fast_sodium_current (millivolt)"
    legend_states[8] = "m in component fast_sodium_current_m_gate (dimensionless)"
    legend_states[9] = "h in component fast_sodium_current_h_gate (dimensionless)"
    legend_algebraic[14] = "alpha_m in component fast_sodium_current_m_gate (per_second)"
    legend_algebraic[20] = "beta_m in component fast_sodium_current_m_gate (per_second)"
    legend_constants[27] = "delta_m in component fast_sodium_current_m_gate (millivolt)"
    legend_algebraic[4] = "E0_m in component fast_sodium_current_m_gate (millivolt)"
    legend_algebraic[5] = "alpha_h in component fast_sodium_current_h_gate (per_second)"
    legend_algebraic[15] = "beta_h in component fast_sodium_current_h_gate (per_second)"
    legend_algebraic[37] = "i_siCa in component second_inward_current (nanoA)"
    legend_algebraic[38] = "i_siK in component second_inward_current (nanoA)"
    legend_algebraic[40] = "i_siNa in component second_inward_current (nanoA)"
    legend_constants[28] = "P_si in component second_inward_current (nanoA_per_millimolar)"
    legend_states[10] = "d in component second_inward_current_d_gate (dimensionless)"
    legend_states[11] = "f in component second_inward_current_f_gate (dimensionless)"
    legend_states[12] = "f2 in component second_inward_current_f2_gate (dimensionless)"
    legend_algebraic[16] = "alpha_d in component second_inward_current_d_gate (per_second)"
    legend_algebraic[21] = "beta_d in component second_inward_current_d_gate (per_second)"
    legend_constants[29] = "delta_d in component second_inward_current_d_gate (millivolt)"
    legend_algebraic[6] = "E0_d in component second_inward_current_d_gate (millivolt)"
    legend_algebraic[17] = "alpha_f in component second_inward_current_f_gate (per_second)"
    legend_algebraic[22] = "beta_f in component second_inward_current_f_gate (per_second)"
    legend_constants[30] = "delta_f in component second_inward_current_f_gate (millivolt)"
    legend_algebraic[7] = "E0_f in component second_inward_current_f_gate (millivolt)"
    legend_constants[31] = "alpha_f2 in component second_inward_current_f2_gate (per_second)"
    legend_algebraic[8] = "beta_f2 in component second_inward_current_f2_gate (per_second)"
    legend_constants[32] = "K_mf2 in component second_inward_current_f2_gate (millimolar)"
    legend_constants[33] = "radius in component intracellular_sodium_concentration (micrometre)"
    legend_constants[34] = "length in component intracellular_sodium_concentration (micrometre)"
    legend_constants[35] = "V_e_ratio in component intracellular_sodium_concentration (dimensionless)"
    legend_constants[46] = "V_Cell in component intracellular_sodium_concentration (micrometre3)"
    legend_constants[47] = "Vi in component intracellular_sodium_concentration (micrometre3)"
    legend_constants[48] = "V_up in component intracellular_calcium_concentration (micrometre3)"
    legend_constants[49] = "V_rel in component intracellular_calcium_concentration (micrometre3)"
    legend_algebraic[39] = "i_up in component intracellular_calcium_concentration (nanoA)"
    legend_algebraic[41] = "i_tr in component intracellular_calcium_concentration (nanoA)"
    legend_algebraic[44] = "i_rel in component intracellular_calcium_concentration (nanoA)"
    legend_states[13] = "Ca_up in component intracellular_calcium_concentration (millimolar)"
    legend_states[14] = "Ca_rel in component intracellular_calcium_concentration (millimolar)"
    legend_constants[36] = "Ca_up_max in component intracellular_calcium_concentration (millimolar)"
    legend_constants[37] = "K_mCa in component intracellular_calcium_concentration (millimolar)"
    legend_states[15] = "p in component intracellular_calcium_concentration (dimensionless)"
    legend_algebraic[9] = "alpha_p in component intracellular_calcium_concentration (per_second)"
    legend_algebraic[18] = "beta_p in component intracellular_calcium_concentration (per_second)"
    legend_constants[38] = "tau_up in component intracellular_calcium_concentration (second)"
    legend_constants[39] = "tau_rep in component intracellular_calcium_concentration (second)"
    legend_constants[40] = "tau_rel in component intracellular_calcium_concentration (second)"
    legend_constants[41] = "rCa in component intracellular_calcium_concentration (dimensionless)"
    legend_constants[42] = "Ve in component extracellular_potassium_concentration (micrometre3)"
    legend_constants[43] = "Kb in component extracellular_potassium_concentration (millimolar)"
    legend_algebraic[42] = "i_mK in component extracellular_potassium_concentration (nanoA)"
    legend_constants[44] = "pf in component extracellular_potassium_concentration (per_second)"
    legend_rates[0] = "d/dt V in component membrane (millivolt)"
    legend_rates[4] = "d/dt y in component hyperpolarising_activated_current_y_gate (dimensionless)"
    legend_rates[5] = "d/dt x in component time_dependent_potassium_current_x_gate (dimensionless)"
    legend_rates[7] = "d/dt s in component transient_outward_current_s_gate (dimensionless)"
    legend_rates[8] = "d/dt m in component fast_sodium_current_m_gate (dimensionless)"
    legend_rates[9] = "d/dt h in component fast_sodium_current_h_gate (dimensionless)"
    legend_rates[10] = "d/dt d in component second_inward_current_d_gate (dimensionless)"
    legend_rates[11] = "d/dt f in component second_inward_current_f_gate (dimensionless)"
    legend_rates[12] = "d/dt f2 in component second_inward_current_f2_gate (dimensionless)"
    legend_rates[3] = "d/dt Nai in component intracellular_sodium_concentration (millimolar)"
    legend_rates[15] = "d/dt p in component intracellular_calcium_concentration (dimensionless)"
    legend_rates[13] = "d/dt Ca_up in component intracellular_calcium_concentration (millimolar)"
    legend_rates[14] = "d/dt Ca_rel in component intracellular_calcium_concentration (millimolar)"
    legend_rates[6] = "d/dt Cai in component intracellular_calcium_concentration (millimolar)"
    legend_rates[1] = "d/dt Kc in component extracellular_potassium_concentration (millimolar)"
    legend_rates[2] = "d/dt Ki in component intracellular_potassium_concentration (millimolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = -87
    constants[0] = 8314.472
    constants[1] = 310
    constants[2] = 96485.3415
    constants[3] = 0.075
    constants[4] = 0
    constants[5] = 3
    constants[6] = 3
    constants[7] = 45
    states[1] = 4
    states[2] = 140
    states[3] = 8
    constants[8] = 140
    states[4] = 0.2
    constants[9] = 1e-5
    constants[10] = 180
    states[5] = 0.01
    constants[11] = 920
    constants[12] = 210
    constants[13] = 10
    constants[14] = 0.0005
    constants[15] = 0.28
    states[6] = 5e-5
    states[7] = 1
    constants[16] = 0.18
    constants[17] = 0.02
    constants[18] = 2
    constants[19] = 125
    constants[20] = 1
    constants[21] = 40
    constants[22] = 3
    constants[23] = 0.02
    constants[24] = 0.001
    constants[25] = 0.5
    constants[26] = 750
    states[8] = 0.01
    states[9] = 0.8
    constants[27] = 1e-5
    constants[28] = 15
    states[10] = 0.005
    states[11] = 1
    states[12] = 1
    constants[29] = 0.0001
    constants[30] = 0.0001
    constants[31] = 5
    constants[32] = 0.001
    constants[33] = 0.05
    constants[34] = 2
    constants[35] = 0.1
    states[13] = 2
    states[14] = 1
    constants[36] = 5
    constants[37] = 0.001
    states[15] = 1
    constants[38] = 0.025
    constants[39] = 2
    constants[40] = 0.05
    constants[41] = 2
    constants[42] = 0.00157
    constants[43] = 4
    constants[44] = 0.7
    constants[45] = (constants[0]*constants[1])/constants[2]
    constants[46] = 3.14159*(power(constants[33], 2.00000))*constants[34]
    constants[47] = constants[46]*(1.00000-constants[35])
    constants[48] = constants[47]*0.0500000
    constants[49] = constants[47]*0.0200000
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[8] = (states[6]*constants[31])/constants[32]
    rates[12] = constants[31]-states[12]*(constants[31]+algebraic[8])
    algebraic[2] = (0.500000*exp(0.0826000*(states[0]+50.0000)))/(1.00000+exp(0.0570000*(states[0]+50.0000)))
    algebraic[12] = (1.30000*exp(-0.0600000*(states[0]+20.0000)))/(1.00000+exp(-0.0400000*(states[0]+20.0000)))
    rates[5] = algebraic[2]*(1.00000-states[5])-algebraic[12]*states[5]
    algebraic[3] = 0.0330000*exp(-states[0]/17.0000)
    algebraic[13] = 33.0000/(1.00000+exp(-(states[0]+10.0000)/8.00000))
    rates[7] = algebraic[3]*(1.00000-states[7])-algebraic[13]*states[7]
    algebraic[5] = 20.0000*exp(-0.125000*(states[0]+75.0000))
    algebraic[15] = 2000.00/(320.000*exp(-0.100000*(states[0]+75.0000))+1.00000)
    rates[9] = algebraic[5]*(1.00000-states[9])-algebraic[15]*states[9]
    algebraic[9] = (0.625000*(states[0]+34.0000))/(exp((states[0]+34.0000)/4.00000)-1.00000)
    algebraic[18] = 5.00000/(1.00000+exp((-1.00000*(states[0]+34.0000))/4.00000))
    rates[15] = algebraic[9]*(1.00000-states[15])-algebraic[18]*states[15]
    algebraic[1] = 0.0500000*exp(-0.0670000*((states[0]+52.0000)-10.0000))
    algebraic[11] = (states[0]+52.0000)-10.0000
    algebraic[19] = custom_piecewise([less(fabs(algebraic[11]) , constants[9]), 2.50000 , True, (1.00000*algebraic[11])/(1.00000-exp(-0.200000*algebraic[11]))])
    rates[4] = algebraic[1]*(1.00000-states[4])-algebraic[19]*states[4]
    algebraic[4] = states[0]+41.0000
    algebraic[14] = custom_piecewise([less(fabs(algebraic[4]) , constants[27]), 2000.00 , True, (200.000*algebraic[4])/(1.00000-exp(-0.100000*algebraic[4]))])
    algebraic[20] = 8000.00*exp(-0.0560000*(states[0]+66.0000))
    rates[8] = algebraic[14]*(1.00000-states[8])-algebraic[20]*states[8]
    algebraic[6] = (states[0]+24.0000)-5.00000
    algebraic[16] = custom_piecewise([less(fabs(algebraic[6]) , constants[29]), 120.000 , True, (30.0000*algebraic[6])/(1.00000-exp((-1.00000*algebraic[6])/4.00000))])
    algebraic[21] = custom_piecewise([less(fabs(algebraic[6]) , constants[29]), 120.000 , True, (12.0000*algebraic[6])/(exp(algebraic[6]/10.0000)-1.00000)])
    rates[10] = algebraic[16]*(1.00000-states[10])-algebraic[21]*states[10]
    algebraic[7] = states[0]+34.0000
    algebraic[17] = custom_piecewise([less(fabs(algebraic[7]) , constants[30]), 25.0000 , True, (6.25000*algebraic[7])/(exp(algebraic[7]/4.00000)-1.00000)])
    algebraic[22] = 50.0000/(1.00000+exp((-1.00000*(states[0]+34.0000))/4.00000))
    rates[11] = algebraic[17]*(1.00000-states[11])-algebraic[22]*states[11]
    algebraic[0] = constants[45]*log(constants[8]/states[3])
    algebraic[30] = constants[16]*(states[0]-algebraic[0])
    algebraic[33] = (((constants[19]*states[1])/(constants[20]+states[1]))*states[3])/(constants[21]+states[3])
    algebraic[34] = (constants[23]*(exp((constants[25]*(constants[22]-2.00000)*states[0])/constants[45])*(power(states[3], constants[22]))*constants[18]-exp(((constants[25]-1.00000)*(constants[22]-2.00000)*states[0])/constants[45])*(power(constants[8], constants[22]))*states[6]))/((1.00000+constants[24]*(states[6]*(power(constants[8], constants[22]))+constants[18]*(power(states[3], constants[22]))))*(1.00000+states[6]/0.00690000))
    algebraic[35] = constants[45]*log((constants[8]+0.120000*states[1])/(states[3]+0.120000*states[2]))
    algebraic[36] = constants[26]*(power(states[8], 3.00000))*states[9]*(states[0]-algebraic[35])
    algebraic[23] = ((states[4]*states[1])/(states[1]+constants[7]))*constants[5]*(states[0]-algebraic[0])
    algebraic[40] = ((0.0100000*constants[28]*(states[0]-50.0000))/(constants[45]*(1.00000-exp((-1.00000*(states[0]-50.0000))/constants[45]))))*(states[3]*exp(50.0000/constants[45])-constants[8]*exp((-1.00000*(states[0]-50.0000))/constants[45]))*states[10]*states[11]*states[12]
    rates[3] = (-1.00000*(algebraic[36]+algebraic[30]+algebraic[23]+algebraic[40]+algebraic[33]*3.00000+(algebraic[34]*constants[22])/(constants[22]-2.00000)))/(1.00000*constants[47]*constants[2])
    algebraic[39] = ((2.00000*1.00000*constants[47]*constants[2])/(1.00000*constants[38]*constants[36]))*states[6]*(constants[36]-states[13])
    algebraic[41] = ((2.00000*1.00000*constants[49]*constants[2])/(1.00000*constants[39]))*states[15]*(states[13]-states[14])
    rates[13] = (1.00000*(algebraic[39]-algebraic[41]))/(2.00000*1.00000*constants[48]*constants[2])
    algebraic[26] = (constants[10]*(states[2]-states[1]*exp(-states[0]/constants[45])))/140.000
    algebraic[27] = states[5]*algebraic[26]
    algebraic[10] = constants[45]*log(states[1]/states[2])
    algebraic[28] = (((constants[11]*states[1])/(states[1]+constants[12]))*(states[0]-algebraic[10]))/(1.00000+exp((((states[0]+10.0000)-algebraic[10])*2.00000)/constants[45]))
    algebraic[29] = ((((states[7]*constants[15]*(0.200000+states[1]/(constants[13]+states[1]))*states[6])/(constants[14]+states[6]))*(states[0]+10.0000))/(1.00000-exp(-0.200000*(states[0]+10.0000))))*(states[2]*exp((0.500000*states[0])/constants[45])-states[1]*exp((-0.500000*states[0])/constants[45]))
    algebraic[24] = ((states[4]*states[1])/(states[1]+constants[7]))*constants[6]*(states[0]-algebraic[10])
    algebraic[38] = ((0.0100000*constants[28]*(states[0]-50.0000))/(constants[45]*(1.00000-exp((-1.00000*(states[0]-50.0000))/constants[45]))))*(states[2]*exp(50.0000/constants[45])-states[1]*exp((-1.00000*(states[0]-50.0000))/constants[45]))*states[10]*states[11]*states[12]
    algebraic[42] = (algebraic[28]+algebraic[27]+algebraic[24]+algebraic[38]+algebraic[29])-2.00000*algebraic[33]
    rates[1] = -constants[44]*(states[1]-constants[43])+(1.00000*algebraic[42])/(1.00000*constants[42]*constants[2])
    rates[2] = (-1.00000*algebraic[42])/(1.00000*constants[47]*constants[2])
    algebraic[25] = algebraic[23]+algebraic[24]
    algebraic[31] = 0.500000*constants[45]*log(constants[18]/states[6])
    algebraic[32] = constants[17]*(states[0]-algebraic[31])
    algebraic[37] = ((4.00000*constants[28]*(states[0]-50.0000))/(constants[45]*(1.00000-exp((-1.00000*(states[0]-50.0000)*2.00000)/constants[45]))))*(states[6]*exp(100.000/constants[45])-constants[18]*exp((-2.00000*(states[0]-50.0000))/constants[45]))*states[10]*states[11]*states[12]
    algebraic[43] = algebraic[37]+algebraic[38]+algebraic[40]
    rates[0] = -(algebraic[25]+algebraic[27]+algebraic[28]+algebraic[29]+algebraic[30]+algebraic[32]+algebraic[33]+algebraic[34]+algebraic[36]+algebraic[43]+constants[4])/constants[3]
    algebraic[44] = (((2.00000*1.00000*constants[49]*constants[2])/(1.00000*constants[40]))*states[14]*(power(states[6], constants[41])))/(power(states[6], constants[41])+power(constants[37], constants[41]))
    rates[14] = (1.00000*(algebraic[41]-algebraic[44]))/(2.00000*1.00000*constants[49]*constants[2])
    rates[6] = (-1.00000*((((algebraic[37]+algebraic[32])-(2.00000*algebraic[34])/(constants[22]-2.00000))-algebraic[44])+algebraic[39]))/(2.00000*1.00000*constants[47]*constants[2])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[8] = (states[6]*constants[31])/constants[32]
    algebraic[2] = (0.500000*exp(0.0826000*(states[0]+50.0000)))/(1.00000+exp(0.0570000*(states[0]+50.0000)))
    algebraic[12] = (1.30000*exp(-0.0600000*(states[0]+20.0000)))/(1.00000+exp(-0.0400000*(states[0]+20.0000)))
    algebraic[3] = 0.0330000*exp(-states[0]/17.0000)
    algebraic[13] = 33.0000/(1.00000+exp(-(states[0]+10.0000)/8.00000))
    algebraic[5] = 20.0000*exp(-0.125000*(states[0]+75.0000))
    algebraic[15] = 2000.00/(320.000*exp(-0.100000*(states[0]+75.0000))+1.00000)
    algebraic[9] = (0.625000*(states[0]+34.0000))/(exp((states[0]+34.0000)/4.00000)-1.00000)
    algebraic[18] = 5.00000/(1.00000+exp((-1.00000*(states[0]+34.0000))/4.00000))
    algebraic[1] = 0.0500000*exp(-0.0670000*((states[0]+52.0000)-10.0000))
    algebraic[11] = (states[0]+52.0000)-10.0000
    algebraic[19] = custom_piecewise([less(fabs(algebraic[11]) , constants[9]), 2.50000 , True, (1.00000*algebraic[11])/(1.00000-exp(-0.200000*algebraic[11]))])
    algebraic[4] = states[0]+41.0000
    algebraic[14] = custom_piecewise([less(fabs(algebraic[4]) , constants[27]), 2000.00 , True, (200.000*algebraic[4])/(1.00000-exp(-0.100000*algebraic[4]))])
    algebraic[20] = 8000.00*exp(-0.0560000*(states[0]+66.0000))
    algebraic[6] = (states[0]+24.0000)-5.00000
    algebraic[16] = custom_piecewise([less(fabs(algebraic[6]) , constants[29]), 120.000 , True, (30.0000*algebraic[6])/(1.00000-exp((-1.00000*algebraic[6])/4.00000))])
    algebraic[21] = custom_piecewise([less(fabs(algebraic[6]) , constants[29]), 120.000 , True, (12.0000*algebraic[6])/(exp(algebraic[6]/10.0000)-1.00000)])
    algebraic[7] = states[0]+34.0000
    algebraic[17] = custom_piecewise([less(fabs(algebraic[7]) , constants[30]), 25.0000 , True, (6.25000*algebraic[7])/(exp(algebraic[7]/4.00000)-1.00000)])
    algebraic[22] = 50.0000/(1.00000+exp((-1.00000*(states[0]+34.0000))/4.00000))
    algebraic[0] = constants[45]*log(constants[8]/states[3])
    algebraic[30] = constants[16]*(states[0]-algebraic[0])
    algebraic[33] = (((constants[19]*states[1])/(constants[20]+states[1]))*states[3])/(constants[21]+states[3])
    algebraic[34] = (constants[23]*(exp((constants[25]*(constants[22]-2.00000)*states[0])/constants[45])*(power(states[3], constants[22]))*constants[18]-exp(((constants[25]-1.00000)*(constants[22]-2.00000)*states[0])/constants[45])*(power(constants[8], constants[22]))*states[6]))/((1.00000+constants[24]*(states[6]*(power(constants[8], constants[22]))+constants[18]*(power(states[3], constants[22]))))*(1.00000+states[6]/0.00690000))
    algebraic[35] = constants[45]*log((constants[8]+0.120000*states[1])/(states[3]+0.120000*states[2]))
    algebraic[36] = constants[26]*(power(states[8], 3.00000))*states[9]*(states[0]-algebraic[35])
    algebraic[23] = ((states[4]*states[1])/(states[1]+constants[7]))*constants[5]*(states[0]-algebraic[0])
    algebraic[40] = ((0.0100000*constants[28]*(states[0]-50.0000))/(constants[45]*(1.00000-exp((-1.00000*(states[0]-50.0000))/constants[45]))))*(states[3]*exp(50.0000/constants[45])-constants[8]*exp((-1.00000*(states[0]-50.0000))/constants[45]))*states[10]*states[11]*states[12]
    algebraic[39] = ((2.00000*1.00000*constants[47]*constants[2])/(1.00000*constants[38]*constants[36]))*states[6]*(constants[36]-states[13])
    algebraic[41] = ((2.00000*1.00000*constants[49]*constants[2])/(1.00000*constants[39]))*states[15]*(states[13]-states[14])
    algebraic[26] = (constants[10]*(states[2]-states[1]*exp(-states[0]/constants[45])))/140.000
    algebraic[27] = states[5]*algebraic[26]
    algebraic[10] = constants[45]*log(states[1]/states[2])
    algebraic[28] = (((constants[11]*states[1])/(states[1]+constants[12]))*(states[0]-algebraic[10]))/(1.00000+exp((((states[0]+10.0000)-algebraic[10])*2.00000)/constants[45]))
    algebraic[29] = ((((states[7]*constants[15]*(0.200000+states[1]/(constants[13]+states[1]))*states[6])/(constants[14]+states[6]))*(states[0]+10.0000))/(1.00000-exp(-0.200000*(states[0]+10.0000))))*(states[2]*exp((0.500000*states[0])/constants[45])-states[1]*exp((-0.500000*states[0])/constants[45]))
    algebraic[24] = ((states[4]*states[1])/(states[1]+constants[7]))*constants[6]*(states[0]-algebraic[10])
    algebraic[38] = ((0.0100000*constants[28]*(states[0]-50.0000))/(constants[45]*(1.00000-exp((-1.00000*(states[0]-50.0000))/constants[45]))))*(states[2]*exp(50.0000/constants[45])-states[1]*exp((-1.00000*(states[0]-50.0000))/constants[45]))*states[10]*states[11]*states[12]
    algebraic[42] = (algebraic[28]+algebraic[27]+algebraic[24]+algebraic[38]+algebraic[29])-2.00000*algebraic[33]
    algebraic[25] = algebraic[23]+algebraic[24]
    algebraic[31] = 0.500000*constants[45]*log(constants[18]/states[6])
    algebraic[32] = constants[17]*(states[0]-algebraic[31])
    algebraic[37] = ((4.00000*constants[28]*(states[0]-50.0000))/(constants[45]*(1.00000-exp((-1.00000*(states[0]-50.0000)*2.00000)/constants[45]))))*(states[6]*exp(100.000/constants[45])-constants[18]*exp((-2.00000*(states[0]-50.0000))/constants[45]))*states[10]*states[11]*states[12]
    algebraic[43] = algebraic[37]+algebraic[38]+algebraic[40]
    algebraic[44] = (((2.00000*1.00000*constants[49]*constants[2])/(1.00000*constants[40]))*states[14]*(power(states[6], constants[41])))/(power(states[6], constants[41])+power(constants[37], constants[41]))
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