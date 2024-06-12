# Size of variable arrays:
sizeAlgebraic = 34
sizeStates = 9
sizeConstants = 54
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_constants[0] = "isotonic in component parameters (dimensionless)"
    legend_constants[1] = "alpha_1 in component parameters (per_micrometre)"
    legend_constants[2] = "beta_1 in component parameters (millinewton)"
    legend_constants[3] = "alpha_2 in component parameters (per_micrometre)"
    legend_constants[4] = "beta_2 in component parameters (millinewton)"
    legend_constants[5] = "alpha_3 in component parameters (per_micrometre)"
    legend_constants[6] = "beta_3 in component parameters (millinewton)"
    legend_constants[7] = "lambda in component parameters (millinewton)"
    legend_constants[8] = "A_half in component parameters (dimensionless)"
    legend_constants[9] = "mu in component parameters (dimensionless)"
    legend_constants[10] = "chi in component parameters (dimensionless)"
    legend_constants[11] = "chi_0 in component parameters (dimensionless)"
    legend_constants[12] = "m_0 in component parameters (dimensionless)"
    legend_constants[13] = "v_max in component parameters (micrometre_per_second)"
    legend_constants[14] = "a in component parameters (dimensionless)"
    legend_constants[15] = "d_h in component parameters (dimensionless)"
    legend_constants[16] = "alpha_P in component parameters (dimensionless)"
    legend_algebraic[9] = "l in component length (micrometre)"
    legend_algebraic[5] = "F_muscle in component force (millinewton)"
    legend_algebraic[11] = "flag in component isotonic (dimensionless)"
    legend_constants[17] = "F_afterload in component isotonic (millinewton)"
    legend_algebraic[13] = "isotonic_mode in component isotonic (dimensionless)"
    legend_constants[18] = "l_0 in component isotonic (micrometre)"
    legend_constants[19] = "S_0 in component parameters_izakov_et_al_1991 (micrometre)"
    legend_algebraic[0] = "q_v in component parameters_izakov_et_al_1991 (per_second)"
    legend_constants[20] = "q_1 in component parameters_izakov_et_al_1991 (per_second)"
    legend_constants[21] = "q_2 in component parameters_izakov_et_al_1991 (per_second)"
    legend_constants[22] = "q_3 in component parameters_izakov_et_al_1991 (per_second)"
    legend_constants[23] = "q_4 in component parameters_izakov_et_al_1991 (per_second)"
    legend_constants[24] = "v_star in component parameters_izakov_et_al_1991 (micrometre_per_second)"
    legend_constants[50] = "v_1 in component parameters_izakov_et_al_1991 (micrometre_per_second)"
    legend_constants[25] = "alpha_G in component parameters_izakov_et_al_1991 (dimensionless)"
    legend_constants[26] = "a_on in component parameters_izakov_et_al_1991 (per_second)"
    legend_constants[27] = "a_off in component parameters_izakov_et_al_1991 (per_second)"
    legend_constants[28] = "k_A in component parameters_izakov_et_al_1991 (dimensionless)"
    legend_states[0] = "v in component CE_velocity (micrometre_per_second)"
    legend_constants[29] = "alpha_Q in component parameters_izakov_et_al_1991 (dimensionless)"
    legend_constants[30] = "beta_Q in component parameters_izakov_et_al_1991 (dimensionless)"
    legend_algebraic[31] = "F_CE in component force (millinewton)"
    legend_algebraic[4] = "F_XSE in component force (millinewton)"
    legend_algebraic[1] = "F_SE in component force (millinewton)"
    legend_algebraic[2] = "F_PE in component force (millinewton)"
    legend_states[1] = "N in component crossbridge_kinetics (dimensionless)"
    legend_algebraic[17] = "k_P_vis in component CE_velocity (millinewton_second_per_micrometre)"
    legend_algebraic[20] = "k_S_vis in component PE_velocity (millinewton_second_per_micrometre)"
    legend_states[2] = "w in component PE_velocity (micrometre_per_second)"
    legend_states[3] = "l_1 in component length (micrometre)"
    legend_states[4] = "l_2 in component length (micrometre)"
    legend_states[5] = "l_3 in component length (micrometre)"
    legend_algebraic[30] = "p_v in component average_crossbridge_force (dimensionless)"
    legend_algebraic[27] = "K_chi in component crossbridge_kinetics (per_second)"
    legend_algebraic[6] = "M_A in component crossbridge_kinetics (dimensionless)"
    legend_algebraic[7] = "n_1 in component crossbridge_kinetics (dimensionless)"
    legend_algebraic[8] = "L_oz in component crossbridge_kinetics (dimensionless)"
    legend_algebraic[25] = "k_p_v in component crossbridge_kinetics (per_second)"
    legend_algebraic[26] = "k_m_v in component crossbridge_kinetics (per_second)"
    legend_states[6] = "A in component calcium_handling (dimensionless)"
    legend_algebraic[24] = "G_star in component average_crossbridge_force (dimensionless)"
    legend_algebraic[21] = "P_star in component average_crossbridge_force (dimensionless)"
    legend_constants[31] = "v_0 in component crossbridge_kinetics (micrometre_per_second)"
    legend_constants[32] = "q_star in component crossbridge_kinetics (per_second)"
    legend_algebraic[3] = "dl_1_dt in component length (micrometre_per_second)"
    legend_algebraic[22] = "dl_2_dt in component length (micrometre_per_second)"
    legend_algebraic[23] = "dl_3_dt in component length (micrometre_per_second)"
    legend_algebraic[18] = "phi_chi_2 in component CE_velocity (micrometre_per_second)"
    legend_algebraic[33] = "phi_chi in component CE_velocity (micrometre_per_second2)"
    legend_algebraic[32] = "p_prime_v in component average_crossbridge_force (second_per_micrometre)"
    legend_constants[33] = "alpha_P_lengthening in component CE_velocity (per_micrometre)"
    legend_constants[34] = "beta_P_lengthening in component CE_velocity (millinewton_second_per_micrometre)"
    legend_constants[35] = "alpha_P_shortening in component CE_velocity (per_micrometre)"
    legend_constants[36] = "beta_P_shortening in component CE_velocity (millinewton_second_per_micrometre)"
    legend_algebraic[15] = "alp_p in component CE_velocity (per_micrometre)"
    legend_constants[37] = "alpha_S_lengthening in component PE_velocity (per_micrometre)"
    legend_constants[38] = "beta_S_lengthening in component PE_velocity (millinewton_second_per_micrometre)"
    legend_constants[39] = "alpha_S_shortening in component PE_velocity (per_micrometre)"
    legend_constants[40] = "beta_S_shortening in component PE_velocity (millinewton_second_per_micrometre)"
    legend_algebraic[19] = "alp_s in component PE_velocity (per_micrometre)"
    legend_constants[53] = "gamma in component average_crossbridge_force (dimensionless)"
    legend_constants[51] = "case_1 in component average_crossbridge_force (second_per_micrometre)"
    legend_algebraic[28] = "case_2 in component average_crossbridge_force (second_per_micrometre)"
    legend_constants[52] = "case_3 in component average_crossbridge_force (second_per_micrometre)"
    legend_algebraic[29] = "case_4 in component average_crossbridge_force (second_per_micrometre)"
    legend_algebraic[14] = "dA_dt in component calcium_handling (per_second)"
    legend_algebraic[10] = "N_A in component calcium_handling (dimensionless)"
    legend_algebraic[12] = "pi_N_A in component calcium_handling (dimensionless)"
    legend_states[7] = "B in component calcium_handling (dimensionless)"
    legend_algebraic[16] = "dB_dt in component calcium_handling (per_second)"
    legend_states[8] = "Ca_C in component calcium_handling (dimensionless)"
    legend_constants[41] = "A_tot in component calcium_handling (dimensionless)"
    legend_constants[42] = "B_tot in component calcium_handling (dimensionless)"
    legend_constants[43] = "b_on in component calcium_handling (per_second)"
    legend_constants[44] = "b_off in component calcium_handling (per_second)"
    legend_constants[45] = "a_c in component calcium_handling (per_second2)"
    legend_constants[46] = "r_Ca in component calcium_handling (per_second)"
    legend_constants[47] = "q_Ca in component calcium_handling (dimensionless)"
    legend_constants[48] = "t_d in component calcium_handling (second)"
    legend_constants[49] = "Ca_m in component calcium_handling (dimensionless)"
    legend_rates[1] = "d/dt N in component crossbridge_kinetics (dimensionless)"
    legend_rates[3] = "d/dt l_1 in component length (micrometre)"
    legend_rates[4] = "d/dt l_2 in component length (micrometre)"
    legend_rates[5] = "d/dt l_3 in component length (micrometre)"
    legend_rates[0] = "d/dt v in component CE_velocity (micrometre_per_second)"
    legend_rates[2] = "d/dt w in component PE_velocity (micrometre_per_second)"
    legend_rates[6] = "d/dt A in component calcium_handling (dimensionless)"
    legend_rates[7] = "d/dt B in component calcium_handling (dimensionless)"
    legend_rates[8] = "d/dt Ca_C in component calcium_handling (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 0
    constants[1] = 19
    constants[2] = 0.29
    constants[3] = 14.6
    constants[4] = 0.000924
    constants[5] = 48
    constants[6] = 0.01
    constants[7] = 96
    constants[8] = 0.6
    constants[9] = 3
    constants[10] = 0.705
    constants[11] = 3
    constants[12] = 0.9
    constants[13] = 5.6
    constants[14] = 0.25
    constants[15] = 0.5
    constants[16] = 4
    constants[17] = 2
    constants[18] = 0.527
    constants[19] = 1.14
    constants[20] = 17.3
    constants[21] = 259
    constants[22] = 17.3
    constants[23] = 15
    constants[24] = 5.3035675
    constants[25] = 1
    constants[26] = 2300
    constants[27] = 290
    constants[28] = 2.8
    states[0] = 0
    constants[29] = 10
    constants[30] = 5000
    states[1] = 0.0001
    states[2] = 0
    states[3] = 0.437
    states[4] = 0.439
    states[5] = 0.089
    states[6] = 0.01
    constants[31] = 560
    constants[32] = 1000
    constants[33] = 16
    constants[34] = 0.0015
    constants[35] = 16
    constants[36] = 0.0015
    constants[37] = 39
    constants[38] = 0.008
    constants[39] = 46
    constants[40] = 0.006
    states[7] = 0
    states[8] = 0
    constants[41] = 1
    constants[42] = 0.4
    constants[43] = 2600
    constants[44] = 182
    constants[45] = 5200
    constants[46] = 650
    constants[47] = 50
    constants[48] = 0.033
    constants[49] = 0.03
    constants[50] = constants[13]/10.0000
    constants[51] = (constants[14]*(0.400000+0.400000*constants[14]))/(constants[13]*(power((constants[14]+1.00000)*0.400000, 2.00000)))
    constants[52] = (0.400000*constants[14]+1.00000)/(constants[14]*constants[13])
    constants[53] = (constants[14]*constants[15]*(power(constants[50]/constants[13], 2.00000)))/(3.00000*constants[14]*constants[15]-((constants[14]+1.00000)*constants[50])/constants[13])
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[3] = states[0]
    rates[3] = algebraic[3]
    algebraic[8] = (states[3]+constants[19])/(0.460000+constants[19])
    algebraic[10] = states[1]/(algebraic[8]*states[6])
    algebraic[12] = custom_piecewise([greater_equal(algebraic[10] , 1.00000), 1.00000 , True, power(0.0200000, algebraic[10])])
    algebraic[14] = constants[26]*(constants[41]-states[6])*states[8]-constants[27]*exp(-constants[28]*states[6])*algebraic[12]*states[6]
    rates[6] = algebraic[14]
    algebraic[16] = constants[43]*(constants[42]-states[7])*states[8]-constants[44]*states[7]
    rates[7] = algebraic[16]
    rates[8] = custom_piecewise([less(voi , constants[48]), 4.00000*constants[45]*constants[49]*voi*(1.00000-exp(-constants[45]*(power(voi, 2.00000))))*exp(-constants[45]*(power(voi, 2.00000))) , True, (-algebraic[14]-algebraic[16])-constants[46]*exp(-constants[47]*states[8])*states[8]])
    algebraic[20] = custom_piecewise([less_equal(states[2] , states[0]), constants[38]*exp(constants[37]*(states[4]-states[3])) , True, constants[40]*exp(constants[39]*(states[4]-states[3]))])
    algebraic[9] = states[4]+states[5]
    algebraic[4] = constants[6]*(exp(constants[5]*states[5])-1.00000)
    algebraic[5] = algebraic[4]
    algebraic[11] = custom_piecewise([greater_equal(algebraic[9] , constants[18]) & less(voi , 0.150000), 0.00000 , True, 1.00000])
    algebraic[13] = custom_piecewise([equal(constants[0] , 0.00000), 0.00000 , equal(constants[0] , 1.00000) & greater_equal(algebraic[5] , constants[17]), 1.00000 , equal(constants[0] , 1.00000) & greater_equal(algebraic[9] , constants[18]) & equal(algebraic[11] , 1.00000), 0.00000 , True, float('nan')])
    algebraic[18] = custom_piecewise([equal(algebraic[13] , 1.00000), (constants[1]*constants[2]*exp(constants[1]*(states[4]-states[3]))*states[0])/(constants[1]*constants[2]*exp(constants[1]*(states[4]-states[3]))+constants[3]*constants[4]*exp(constants[3]*states[4])) , True, (constants[1]*constants[2]*exp(constants[1]*(states[4]-states[3]))*states[0])/(constants[1]*constants[2]*exp(constants[1]*(states[4]-states[3]))+constants[3]*constants[4]*exp(constants[3]*states[4])+constants[5]*constants[6]*exp(constants[5]*states[5]))])
    algebraic[22] = custom_piecewise([equal(algebraic[20] , 0.00000), algebraic[18] , True, states[2]])
    rates[4] = algebraic[22]
    algebraic[23] = custom_piecewise([equal(algebraic[13] , 1.00000), 0.00000 , equal(algebraic[13] , 0.00000) & equal(algebraic[20] , 0.00000), -algebraic[18] , True, -states[2]])
    rates[5] = algebraic[23]
    algebraic[6] = (power(states[6], constants[9]))/(power(states[6], constants[9])+power(constants[8], constants[9]))
    algebraic[7] = 0.600000*states[3]+0.500000
    algebraic[0] = custom_piecewise([less_equal(states[0] , 0.00000), constants[20]-(constants[21]*states[0])/constants[13] , less_equal(states[0] , constants[24]) & less(0.00000 , states[0]), ((constants[23]-constants[22])*states[0])/constants[24]+constants[22] , True, constants[23]/(power(1.00000+(constants[30]*(states[0]-constants[24]))/constants[13], constants[29]))])
    algebraic[21] = custom_piecewise([less_equal(states[0] , 0.00000), (constants[14]*(1.00000+states[0]/constants[13]))/(constants[14]-states[0]/constants[13]) , True, (1.00000+constants[15])-((power(constants[15], 2.00000))*constants[14])/(((constants[14]*constants[15])/constants[53])*(power(states[0]/constants[13], 2.00000))+((constants[14]+1.00000)*states[0])/constants[13]+constants[14]*constants[15])])
    algebraic[24] = custom_piecewise([less_equal(-constants[13] , states[0]) & less_equal(states[0] , 0.00000), 1.00000+(0.600000*states[0])/constants[13] , less(0.00000 , states[0]) & less_equal(states[0] , constants[50]), algebraic[21]/(((0.400000*constants[14]+1.00000)*states[0])/(constants[14]*constants[13])+1.00000) , True, (algebraic[21]*exp(-constants[25]*(power((states[0]-constants[50])/constants[13], constants[16]))))/(((0.400000*constants[14]+1.00000)*states[0])/(constants[14]*constants[13])+1.00000)])
    algebraic[25] = constants[10]*constants[11]*algebraic[0]*constants[12]*algebraic[24]
    algebraic[26] = custom_piecewise([less_equal(states[0] , constants[24]), constants[11]*algebraic[0]*(1.00000-constants[10]*constants[12]*algebraic[24]) , True, constants[11]*(constants[23]*(1.00000-constants[10]*constants[12]*algebraic[24])+(constants[32]*(states[0]-constants[24]))/(constants[31]-constants[24]))])
    algebraic[27] = algebraic[25]*algebraic[6]*algebraic[7]*algebraic[8]*(1.00000-states[1])-algebraic[26]*states[1]
    rates[1] = algebraic[27]
    algebraic[17] = custom_piecewise([less_equal(states[0] , 0.00000), constants[34]*exp(constants[33]*states[3]) , True, constants[36]*exp(constants[35]*states[3])])
    algebraic[30] = algebraic[21]/algebraic[24]
    algebraic[28] = (constants[14]*1.00000*(1.00000+0.400000*constants[14]+(1.20000*states[0])/constants[13]+0.600000*(power(states[0]/constants[13], 2.00000))))/(constants[13]*(power((constants[14]-states[0]/constants[13])*(1.00000+(0.600000*states[0])/constants[13]), 2.00000)))
    algebraic[29] = (1.00000/constants[13])*exp(-constants[25]*(power(states[0]/constants[13]-constants[50]/constants[13], constants[16])))*((0.400000*constants[14]+1.00000)/constants[14]+constants[25]*constants[16]*(1.00000+((0.400000*constants[14]+1.00000)*states[0])/(constants[14]*constants[13]))*(power(states[0]/constants[13]-constants[50]/constants[13], constants[16]-1.00000)))
    algebraic[32] = custom_piecewise([less_equal(states[0] , -constants[13]), constants[51] , less(-constants[13] , states[0]) & less_equal(states[0] , 0.00000), algebraic[28] , less(0.00000 , states[0]) & less_equal(states[0] , constants[50]), constants[52] , True, algebraic[29]])
    algebraic[15] = custom_piecewise([less_equal(states[0] , 0.00000), constants[33] , True, constants[35]])
    algebraic[33] = custom_piecewise([equal(algebraic[13] , 1.00000), -(constants[7]*algebraic[27]*algebraic[30]+algebraic[15]*algebraic[17]*(power(states[0], 2.00000))+constants[3]*constants[4]*exp(constants[3]*states[4])*states[2])/(constants[7]*states[1]*algebraic[32]+algebraic[17]) , True, -(constants[7]*algebraic[27]*algebraic[30]+algebraic[15]*algebraic[17]*(power(states[0], 2.00000))+(constants[3]*constants[4]*exp(constants[3]*states[4])+constants[5]*constants[6]*exp(constants[5]*states[5]))*states[2])/(constants[7]*states[1]*algebraic[32]+algebraic[17])])
    rates[0] = custom_piecewise([equal(algebraic[20] , 0.00000), (constants[1]*constants[2]*exp(constants[1]*(states[4]-states[3]))*(algebraic[18]-states[0])-(constants[7]*algebraic[27]*algebraic[30]+algebraic[15]*algebraic[17]*(power(states[0], 2.00000))))/(constants[7]*states[1]*algebraic[32]+algebraic[17]) , True, algebraic[33]])
    algebraic[19] = custom_piecewise([less_equal(states[2] , states[0]), constants[37] , True, constants[39]])
    rates[2] = custom_piecewise([equal(algebraic[13] , 1.00000), ((algebraic[20]*(algebraic[33]-algebraic[19]*(power(states[2]-states[0], 2.00000)))-constants[1]*constants[2]*exp(constants[1]*(states[4]-states[3]))*(states[2]-states[0]))-constants[3]*constants[4]*exp(constants[3]*states[4])*states[2])/algebraic[20] , True, (algebraic[33]-algebraic[19]*(power(states[2]-states[0], 2.00000)))-(constants[1]*constants[2]*exp(constants[1]*(states[4]-states[3]))*(states[2]-states[0])+(constants[3]*constants[4]*exp(constants[3]*states[4])+constants[5]*constants[6]*exp(constants[5]*states[5]))*states[2])/algebraic[20]])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[3] = states[0]
    algebraic[8] = (states[3]+constants[19])/(0.460000+constants[19])
    algebraic[10] = states[1]/(algebraic[8]*states[6])
    algebraic[12] = custom_piecewise([greater_equal(algebraic[10] , 1.00000), 1.00000 , True, power(0.0200000, algebraic[10])])
    algebraic[14] = constants[26]*(constants[41]-states[6])*states[8]-constants[27]*exp(-constants[28]*states[6])*algebraic[12]*states[6]
    algebraic[16] = constants[43]*(constants[42]-states[7])*states[8]-constants[44]*states[7]
    algebraic[20] = custom_piecewise([less_equal(states[2] , states[0]), constants[38]*exp(constants[37]*(states[4]-states[3])) , True, constants[40]*exp(constants[39]*(states[4]-states[3]))])
    algebraic[9] = states[4]+states[5]
    algebraic[4] = constants[6]*(exp(constants[5]*states[5])-1.00000)
    algebraic[5] = algebraic[4]
    algebraic[11] = custom_piecewise([greater_equal(algebraic[9] , constants[18]) & less(voi , 0.150000), 0.00000 , True, 1.00000])
    algebraic[13] = custom_piecewise([equal(constants[0] , 0.00000), 0.00000 , equal(constants[0] , 1.00000) & greater_equal(algebraic[5] , constants[17]), 1.00000 , equal(constants[0] , 1.00000) & greater_equal(algebraic[9] , constants[18]) & equal(algebraic[11] , 1.00000), 0.00000 , True, float('nan')])
    algebraic[18] = custom_piecewise([equal(algebraic[13] , 1.00000), (constants[1]*constants[2]*exp(constants[1]*(states[4]-states[3]))*states[0])/(constants[1]*constants[2]*exp(constants[1]*(states[4]-states[3]))+constants[3]*constants[4]*exp(constants[3]*states[4])) , True, (constants[1]*constants[2]*exp(constants[1]*(states[4]-states[3]))*states[0])/(constants[1]*constants[2]*exp(constants[1]*(states[4]-states[3]))+constants[3]*constants[4]*exp(constants[3]*states[4])+constants[5]*constants[6]*exp(constants[5]*states[5]))])
    algebraic[22] = custom_piecewise([equal(algebraic[20] , 0.00000), algebraic[18] , True, states[2]])
    algebraic[23] = custom_piecewise([equal(algebraic[13] , 1.00000), 0.00000 , equal(algebraic[13] , 0.00000) & equal(algebraic[20] , 0.00000), -algebraic[18] , True, -states[2]])
    algebraic[6] = (power(states[6], constants[9]))/(power(states[6], constants[9])+power(constants[8], constants[9]))
    algebraic[7] = 0.600000*states[3]+0.500000
    algebraic[0] = custom_piecewise([less_equal(states[0] , 0.00000), constants[20]-(constants[21]*states[0])/constants[13] , less_equal(states[0] , constants[24]) & less(0.00000 , states[0]), ((constants[23]-constants[22])*states[0])/constants[24]+constants[22] , True, constants[23]/(power(1.00000+(constants[30]*(states[0]-constants[24]))/constants[13], constants[29]))])
    algebraic[21] = custom_piecewise([less_equal(states[0] , 0.00000), (constants[14]*(1.00000+states[0]/constants[13]))/(constants[14]-states[0]/constants[13]) , True, (1.00000+constants[15])-((power(constants[15], 2.00000))*constants[14])/(((constants[14]*constants[15])/constants[53])*(power(states[0]/constants[13], 2.00000))+((constants[14]+1.00000)*states[0])/constants[13]+constants[14]*constants[15])])
    algebraic[24] = custom_piecewise([less_equal(-constants[13] , states[0]) & less_equal(states[0] , 0.00000), 1.00000+(0.600000*states[0])/constants[13] , less(0.00000 , states[0]) & less_equal(states[0] , constants[50]), algebraic[21]/(((0.400000*constants[14]+1.00000)*states[0])/(constants[14]*constants[13])+1.00000) , True, (algebraic[21]*exp(-constants[25]*(power((states[0]-constants[50])/constants[13], constants[16]))))/(((0.400000*constants[14]+1.00000)*states[0])/(constants[14]*constants[13])+1.00000)])
    algebraic[25] = constants[10]*constants[11]*algebraic[0]*constants[12]*algebraic[24]
    algebraic[26] = custom_piecewise([less_equal(states[0] , constants[24]), constants[11]*algebraic[0]*(1.00000-constants[10]*constants[12]*algebraic[24]) , True, constants[11]*(constants[23]*(1.00000-constants[10]*constants[12]*algebraic[24])+(constants[32]*(states[0]-constants[24]))/(constants[31]-constants[24]))])
    algebraic[27] = algebraic[25]*algebraic[6]*algebraic[7]*algebraic[8]*(1.00000-states[1])-algebraic[26]*states[1]
    algebraic[17] = custom_piecewise([less_equal(states[0] , 0.00000), constants[34]*exp(constants[33]*states[3]) , True, constants[36]*exp(constants[35]*states[3])])
    algebraic[30] = algebraic[21]/algebraic[24]
    algebraic[28] = (constants[14]*1.00000*(1.00000+0.400000*constants[14]+(1.20000*states[0])/constants[13]+0.600000*(power(states[0]/constants[13], 2.00000))))/(constants[13]*(power((constants[14]-states[0]/constants[13])*(1.00000+(0.600000*states[0])/constants[13]), 2.00000)))
    algebraic[29] = (1.00000/constants[13])*exp(-constants[25]*(power(states[0]/constants[13]-constants[50]/constants[13], constants[16])))*((0.400000*constants[14]+1.00000)/constants[14]+constants[25]*constants[16]*(1.00000+((0.400000*constants[14]+1.00000)*states[0])/(constants[14]*constants[13]))*(power(states[0]/constants[13]-constants[50]/constants[13], constants[16]-1.00000)))
    algebraic[32] = custom_piecewise([less_equal(states[0] , -constants[13]), constants[51] , less(-constants[13] , states[0]) & less_equal(states[0] , 0.00000), algebraic[28] , less(0.00000 , states[0]) & less_equal(states[0] , constants[50]), constants[52] , True, algebraic[29]])
    algebraic[15] = custom_piecewise([less_equal(states[0] , 0.00000), constants[33] , True, constants[35]])
    algebraic[33] = custom_piecewise([equal(algebraic[13] , 1.00000), -(constants[7]*algebraic[27]*algebraic[30]+algebraic[15]*algebraic[17]*(power(states[0], 2.00000))+constants[3]*constants[4]*exp(constants[3]*states[4])*states[2])/(constants[7]*states[1]*algebraic[32]+algebraic[17]) , True, -(constants[7]*algebraic[27]*algebraic[30]+algebraic[15]*algebraic[17]*(power(states[0], 2.00000))+(constants[3]*constants[4]*exp(constants[3]*states[4])+constants[5]*constants[6]*exp(constants[5]*states[5]))*states[2])/(constants[7]*states[1]*algebraic[32]+algebraic[17])])
    algebraic[19] = custom_piecewise([less_equal(states[2] , states[0]), constants[37] , True, constants[39]])
    algebraic[1] = constants[2]*(exp(constants[1]*(states[4]-states[3]))-1.00000)
    algebraic[2] = constants[4]*(exp(constants[3]*states[4])-1.00000)
    algebraic[31] = constants[7]*algebraic[30]*states[1]
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