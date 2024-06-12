# Size of variable arrays:
sizeAlgebraic = 43
sizeStates = 7
sizeConstants = 35
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_constants[0] = "k0_12 in component parameters (per_M2_per_second)"
    legend_constants[1] = "k0_21 in component parameters (per_second)"
    legend_constants[2] = "k0_23 in component parameters (per_M_per_second)"
    legend_constants[3] = "k0_32 in component parameters (per_second)"
    legend_constants[4] = "k0_34 in component parameters (per_second)"
    legend_constants[5] = "k0_43 in component parameters (per_second)"
    legend_constants[6] = "k0_45 in component parameters (per_second)"
    legend_constants[7] = "k0_54 in component parameters (per_M_per_second)"
    legend_constants[8] = "k0_25 in component parameters (per_second)"
    legend_constants[9] = "k0_52 in component parameters (per_second)"
    legend_constants[10] = "k0_56 in component parameters (per_second)"
    legend_constants[11] = "k0_65 in component parameters (per_M2_per_second)"
    legend_constants[12] = "k0_61 in component parameters (per_second)"
    legend_constants[13] = "k0_16 in component parameters (per_second)"
    legend_constants[14] = "delta in component parameters (dimensionless)"
    legend_constants[29] = "alpha_p in component parameters (dimensionless)"
    legend_constants[15] = "alpha_pp in component parameters (dimensionless)"
    legend_constants[16] = "N_C in component parameters (dimensionless)"
    legend_constants[17] = "N_Avo in component parameters (per_mol)"
    legend_constants[18] = "area in component parameters (um2)"
    legend_constants[30] = "C_T in component parameters (umol)"
    legend_constants[19] = "n in component parameters (dimensionless)"
    legend_constants[20] = "z_c in component parameters (dimensionless)"
    legend_constants[21] = "z_Na in component parameters (dimensionless)"
    legend_constants[22] = "F in component parameters (C_per_mol)"
    legend_constants[23] = "R in component parameters (J_per_K_per_mol)"
    legend_constants[24] = "T in component parameters (kelvin)"
    legend_states[0] = "V in component ion_concentrations (volt)"
    legend_algebraic[0] = "mu in component parameters (dimensionless)"
    legend_constants[25] = "Na_i in component ion_concentrations (M)"
    legend_constants[26] = "Na_o in component ion_concentrations (M)"
    legend_constants[27] = "glucose_i in component ion_concentrations (M)"
    legend_constants[28] = "glucose_o in component ion_concentrations (M)"
    legend_algebraic[3] = "k_12 in component rate_constants (per_second)"
    legend_algebraic[4] = "k_21 in component rate_constants (per_second)"
    legend_constants[31] = "k_23 in component rate_constants (per_second)"
    legend_constants[32] = "k_32 in component rate_constants (per_second)"
    legend_algebraic[5] = "k_34 in component rate_constants (per_second)"
    legend_algebraic[6] = "k_43 in component rate_constants (per_second)"
    legend_constants[33] = "k_45 in component rate_constants (per_second)"
    legend_algebraic[14] = "k_54 in component rate_constants (per_second)"
    legend_algebraic[7] = "k_25 in component rate_constants (per_second)"
    legend_algebraic[12] = "k_52 in component rate_constants (per_second)"
    legend_algebraic[8] = "k_56 in component rate_constants (per_second)"
    legend_algebraic[9] = "k_65 in component rate_constants (per_second)"
    legend_algebraic[10] = "k_61 in component rate_constants (per_second)"
    legend_algebraic[11] = "k_16 in component rate_constants (per_second)"
    legend_algebraic[1] = "ks_12 in component rate_constants (per_M2_per_second)"
    legend_algebraic[13] = "k0_54_temp in component rate_constants (per_M_per_second)"
    legend_algebraic[2] = "k_52_temp in component rate_constants (per_second)"
    legend_states[1] = "C_1 in component kinetic_equations (umol)"
    legend_states[2] = "C_2 in component kinetic_equations (umol)"
    legend_states[3] = "C_3 in component kinetic_equations (umol)"
    legend_states[4] = "C_4 in component kinetic_equations (umol)"
    legend_states[5] = "C_5 in component kinetic_equations (umol)"
    legend_algebraic[15] = "C_6 in component kinetic_equations (umol)"
    legend_states[6] = "C_6_temp in component kinetic_equations (umol)"
    legend_algebraic[16] = "C1_sum in component king_altman_states (per_second5)"
    legend_algebraic[18] = "C2_sum in component king_altman_states (per_second5)"
    legend_algebraic[22] = "C3_sum in component king_altman_states (per_second5)"
    legend_algebraic[24] = "C4_sum in component king_altman_states (per_second5)"
    legend_algebraic[27] = "C5_sum in component king_altman_states (per_second5)"
    legend_algebraic[31] = "C6_sum in component king_altman_states (per_second5)"
    legend_algebraic[35] = "C_sum in component king_altman_states (per_second5)"
    legend_algebraic[36] = "C1 in component king_altman_states (umol)"
    legend_algebraic[37] = "C2 in component king_altman_states (umol)"
    legend_algebraic[38] = "C3 in component king_altman_states (umol)"
    legend_algebraic[39] = "C4 in component king_altman_states (umol)"
    legend_algebraic[40] = "C5 in component king_altman_states (umol)"
    legend_algebraic[41] = "C6 in component king_altman_states (umol)"
    legend_algebraic[19] = "I_NaGl_pSS in component NBC_current (uA)"
    legend_algebraic[42] = "I_NaGl_SS in component NBC_current (uA)"
    legend_algebraic[28] = "epsilon in component phenomonological_constants (per_second)"
    legend_algebraic[17] = "lambda in component phenomonological_constants (per_M3_per_second5)"
    legend_algebraic[20] = "chi in component phenomonological_constants (M)"
    legend_algebraic[26] = "alpha in component phenomonological_constants (M3)"
    legend_algebraic[23] = "beta in component phenomonological_constants (M2)"
    legend_algebraic[21] = "gamma in component phenomonological_constants (M3_per_second)"
    legend_algebraic[25] = "phi in component phenomonological_constants (M_per_second)"
    legend_algebraic[32] = "Imax_Na in component phenomonological_constants (uA)"
    legend_algebraic[33] = "Imax_gluc in component phenomonological_constants (uA)"
    legend_algebraic[29] = "Khalf_Na_sq in component phenomonological_constants (M2)"
    legend_algebraic[34] = "Khalf_Na in component phenomonological_constants (M)"
    legend_algebraic[30] = "Khalf_gluc in component phenomonological_constants (M)"
    legend_rates[0] = "d/dt V in component ion_concentrations (volt)"
    legend_rates[1] = "d/dt C_1 in component kinetic_equations (umol)"
    legend_rates[2] = "d/dt C_2 in component kinetic_equations (umol)"
    legend_rates[3] = "d/dt C_3 in component kinetic_equations (umol)"
    legend_rates[4] = "d/dt C_4 in component kinetic_equations (umol)"
    legend_rates[5] = "d/dt C_5 in component kinetic_equations (umol)"
    legend_rates[6] = "d/dt C_6_temp in component kinetic_equations (umol)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 80000
    constants[1] = 500
    constants[2] = 1e5
    constants[3] = 20
    constants[4] = 50
    constants[5] = 50
    constants[6] = 800
    constants[7] = 1.83e7
    constants[8] = 0.3
    constants[9] = 1.37
    constants[10] = 10
    constants[11] = 50
    constants[12] = 5
    constants[13] = 35
    constants[14] = 0.7
    constants[15] = 0
    constants[16] = 6e10
    constants[17] = 6.022e23
    constants[18] = 1e6
    constants[19] = 2
    constants[20] = -2
    constants[21] = 1
    constants[22] = 96485.34
    constants[23] = 8.314
    constants[24] = 310
    states[0] = -50e-3
    constants[25] = 20e-3
    constants[26] = 100e-3
    constants[27] = 10e-6
    constants[28] = 0e-3
    states[1] = 1.505e-8
    states[2] = 7.3976e-8
    states[3] = 2.4603e-10
    states[4] = 3.4444e-10
    states[5] = 1.5338e-9
    states[6] = 8.484e-9
    constants[29] = (1.00000-constants[14])-constants[15]
    constants[34] = 0.00000
    constants[30] = (1.00000e+06*constants[16])/constants[17]
    constants[31] = constants[2]*constants[28]
    constants[32] = constants[3]
    constants[33] = constants[6]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = constants[34]
    algebraic[0] = (constants[22]*states[0])/(constants[23]*constants[24])
    algebraic[5] = constants[4]*exp((-(constants[20]+constants[19])*constants[14]*algebraic[0])/2.00000)
    algebraic[6] = constants[5]*exp(((constants[20]+constants[19])*constants[14]*algebraic[0])/2.00000)
    rates[3] = (constants[31]*states[2]+algebraic[6]*states[4])-(constants[32]+algebraic[5])*states[3]
    algebraic[8] = constants[10]*exp((-constants[19]*constants[21]*constants[15]*algebraic[0])/2.00000)
    algebraic[9] = constants[11]*(power(constants[25], constants[19]))*exp((constants[19]*constants[21]*constants[15]*algebraic[0])/2.00000)
    algebraic[10] = constants[12]*exp((constants[20]*constants[14]*algebraic[0])/2.00000)
    algebraic[11] = constants[13]*exp((-constants[20]*constants[14]*algebraic[0])/2.00000)
    rates[6] = (algebraic[11]*states[1]+algebraic[8]*states[5])-(algebraic[10]+algebraic[9])*states[6]
    algebraic[1] = constants[0]*exp((-constants[19]*constants[29]*algebraic[0])/2.00000)
    algebraic[3] = algebraic[1]*(power(constants[26], constants[19]))
    algebraic[4] = constants[1]*exp((constants[19]*constants[21]*constants[29]*algebraic[0])/2.00000)
    algebraic[7] = constants[8]*exp((-(constants[20]+constants[19])*constants[14]*algebraic[0])/2.00000)
    algebraic[12] = (constants[0]*algebraic[7]*constants[10]*constants[12])/(constants[1]*constants[13]*constants[11])
    rates[2] = (algebraic[3]*states[1]+constants[32]*states[3]+algebraic[12]*states[5])-(algebraic[4]+constants[31]+algebraic[7])*states[2]
    algebraic[13] = (constants[2]*algebraic[5]*constants[33]*algebraic[12])/(algebraic[6]*constants[32]*algebraic[7])
    algebraic[14] = algebraic[13]*constants[27]
    rates[4] = (algebraic[5]*states[3]+algebraic[14]*states[5])-(constants[33]+algebraic[6])*states[4]
    algebraic[15] = constants[30]-(states[1]+states[2]+states[3]+states[4]+states[5])
    rates[1] = (algebraic[4]*states[2]+algebraic[10]*algebraic[15])-(algebraic[3]+algebraic[11])*states[1]
    rates[5] = (constants[33]*states[4]+algebraic[9]*algebraic[15]+algebraic[7]*states[2])-(algebraic[14]+algebraic[12]+algebraic[8])*states[5]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = (constants[22]*states[0])/(constants[23]*constants[24])
    algebraic[5] = constants[4]*exp((-(constants[20]+constants[19])*constants[14]*algebraic[0])/2.00000)
    algebraic[6] = constants[5]*exp(((constants[20]+constants[19])*constants[14]*algebraic[0])/2.00000)
    algebraic[8] = constants[10]*exp((-constants[19]*constants[21]*constants[15]*algebraic[0])/2.00000)
    algebraic[9] = constants[11]*(power(constants[25], constants[19]))*exp((constants[19]*constants[21]*constants[15]*algebraic[0])/2.00000)
    algebraic[10] = constants[12]*exp((constants[20]*constants[14]*algebraic[0])/2.00000)
    algebraic[11] = constants[13]*exp((-constants[20]*constants[14]*algebraic[0])/2.00000)
    algebraic[1] = constants[0]*exp((-constants[19]*constants[29]*algebraic[0])/2.00000)
    algebraic[3] = algebraic[1]*(power(constants[26], constants[19]))
    algebraic[4] = constants[1]*exp((constants[19]*constants[21]*constants[29]*algebraic[0])/2.00000)
    algebraic[7] = constants[8]*exp((-(constants[20]+constants[19])*constants[14]*algebraic[0])/2.00000)
    algebraic[12] = (constants[0]*algebraic[7]*constants[10]*constants[12])/(constants[1]*constants[13]*constants[11])
    algebraic[13] = (constants[2]*algebraic[5]*constants[33]*algebraic[12])/(algebraic[6]*constants[32]*algebraic[7])
    algebraic[14] = algebraic[13]*constants[27]
    algebraic[15] = constants[30]-(states[1]+states[2]+states[3]+states[4]+states[5])
    algebraic[2] = constants[9]*exp(((constants[20]+constants[19])*constants[14]*algebraic[0])/2.00000)
    algebraic[16] = algebraic[4]*constants[32]*algebraic[6]*algebraic[14]*algebraic[9]+algebraic[4]*algebraic[5]*constants[33]*algebraic[12]*algebraic[9]+algebraic[4]*constants[32]*constants[33]*algebraic[12]*algebraic[9]+algebraic[4]*constants[32]*algebraic[6]*algebraic[12]*algebraic[9]+algebraic[7]*algebraic[5]*constants[33]*algebraic[8]*algebraic[10]+constants[31]*algebraic[5]*constants[33]*algebraic[8]*algebraic[10]+algebraic[4]*algebraic[5]*constants[33]*algebraic[8]*algebraic[10]+algebraic[7]*constants[32]*constants[33]*algebraic[8]*algebraic[10]+algebraic[4]*constants[32]*constants[33]*algebraic[8]*algebraic[10]+algebraic[7]*constants[32]*algebraic[6]*algebraic[8]*algebraic[10]+algebraic[4]*constants[32]*algebraic[6]*algebraic[8]*algebraic[10]+algebraic[4]*constants[32]*algebraic[6]*algebraic[14]*algebraic[10]+algebraic[4]*algebraic[5]*constants[33]*algebraic[12]*algebraic[10]+algebraic[4]*constants[32]*constants[33]*algebraic[12]*algebraic[10]+algebraic[4]*constants[32]*algebraic[6]*algebraic[12]*algebraic[10]
    algebraic[17] = algebraic[1]*constants[2]*algebraic[6]*algebraic[14]*algebraic[9]+algebraic[1]*constants[2]*algebraic[5]*algebraic[14]*algebraic[9]+algebraic[1]*constants[2]*constants[33]*algebraic[12]*algebraic[9]+algebraic[1]*constants[2]*algebraic[6]*algebraic[12]*algebraic[9]+algebraic[1]*constants[2]*algebraic[5]*algebraic[12]*algebraic[9]+algebraic[1]*constants[2]*algebraic[5]*constants[33]*algebraic[9]+algebraic[1]*constants[2]*constants[33]*algebraic[8]*algebraic[10]+algebraic[1]*constants[2]*algebraic[6]*algebraic[8]*algebraic[10]+algebraic[1]*constants[2]*algebraic[5]*algebraic[8]*algebraic[10]+algebraic[1]*constants[2]*algebraic[6]*algebraic[14]*algebraic[10]+algebraic[1]*constants[2]*algebraic[5]*algebraic[14]*algebraic[10]+algebraic[1]*constants[2]*constants[33]*algebraic[12]*algebraic[10]+algebraic[1]*constants[2]*algebraic[6]*algebraic[12]*algebraic[10]+algebraic[1]*constants[2]*algebraic[5]*algebraic[12]*algebraic[10]+algebraic[1]*constants[2]*algebraic[5]*constants[33]*algebraic[10]+algebraic[1]*constants[2]*algebraic[5]*constants[33]*algebraic[8]
    algebraic[18] = algebraic[11]*constants[32]*algebraic[6]*algebraic[14]*algebraic[9]+algebraic[3]*constants[32]*algebraic[6]*algebraic[14]*algebraic[9]+algebraic[11]*algebraic[5]*constants[33]*algebraic[12]*algebraic[9]+algebraic[3]*algebraic[5]*constants[33]*algebraic[12]*algebraic[9]+algebraic[11]*constants[32]*constants[33]*algebraic[12]*algebraic[9]+algebraic[3]*constants[32]*constants[33]*algebraic[12]*algebraic[9]+algebraic[11]*constants[32]*algebraic[6]*algebraic[12]*algebraic[9]+algebraic[3]*constants[32]*algebraic[6]*algebraic[12]*algebraic[9]+algebraic[3]*algebraic[5]*constants[33]*algebraic[8]*algebraic[10]+algebraic[3]*constants[32]*constants[33]*algebraic[8]*algebraic[10]+algebraic[3]*constants[32]*algebraic[6]*algebraic[8]*algebraic[10]+algebraic[3]*constants[32]*algebraic[6]*algebraic[14]*algebraic[10]+algebraic[3]*algebraic[5]*constants[33]*algebraic[12]*algebraic[10]+algebraic[3]*constants[32]*constants[33]*algebraic[12]*algebraic[10]+algebraic[3]*constants[32]*algebraic[6]*algebraic[12]*algebraic[10]
    algebraic[19] = -constants[22]*(constants[19]*constants[21]*constants[29]*(algebraic[3]*states[1]-algebraic[4]*states[2])+constants[20]*constants[14]*(algebraic[11]*states[1]-algebraic[10]*algebraic[15])+constants[19]*constants[21]*constants[15]*(algebraic[8]*states[5]-algebraic[9]*algebraic[15]))
    algebraic[20] = (1.00000/algebraic[17])*(algebraic[1]*constants[32]*algebraic[6]*algebraic[14]*algebraic[9]+algebraic[1]*algebraic[7]*algebraic[6]*algebraic[14]*algebraic[9]+algebraic[1]*algebraic[7]*algebraic[5]*algebraic[14]*algebraic[9]+algebraic[1]*algebraic[7]*constants[32]*algebraic[14]*algebraic[9]+algebraic[1]*algebraic[5]*constants[33]*algebraic[12]*algebraic[9]+algebraic[1]*constants[32]*constants[33]*algebraic[12]*algebraic[9]+algebraic[1]*constants[32]*algebraic[6]*algebraic[12]*algebraic[9]+algebraic[1]*algebraic[7]*algebraic[5]*constants[33]*algebraic[9]+algebraic[1]*algebraic[7]*constants[32]*constants[33]*algebraic[9]+algebraic[1]*algebraic[7]*constants[32]*algebraic[6]*algebraic[9]+algebraic[1]*algebraic[5]*constants[33]*algebraic[8]*algebraic[10]+algebraic[1]*constants[32]*constants[33]*algebraic[8]*algebraic[10]+algebraic[1]*constants[32]*algebraic[6]*algebraic[8]*algebraic[10]+algebraic[1]*constants[32]*algebraic[6]*algebraic[14]*algebraic[10]+algebraic[1]*algebraic[7]*algebraic[6]*algebraic[14]*algebraic[10]+algebraic[1]*algebraic[7]*algebraic[5]*algebraic[14]*algebraic[10]+algebraic[1]*algebraic[7]*constants[32]*algebraic[14]*algebraic[10]+algebraic[1]*algebraic[5]*constants[33]*algebraic[12]*algebraic[10]+algebraic[1]*constants[32]*constants[33]*algebraic[12]*algebraic[10]+algebraic[1]*constants[32]*algebraic[6]*algebraic[12]*algebraic[10]+algebraic[1]*algebraic[7]*algebraic[5]*constants[33]*algebraic[10]+algebraic[1]*algebraic[7]*constants[32]*constants[33]*algebraic[10]+algebraic[1]*algebraic[7]*constants[32]*algebraic[6]*algebraic[10]+algebraic[1]*algebraic[7]*algebraic[5]*constants[33]*algebraic[8]+algebraic[1]*algebraic[7]*constants[32]*constants[33]*algebraic[8]+algebraic[1]*algebraic[7]*constants[32]*algebraic[6]*algebraic[8])
    algebraic[21] = (1.00000/algebraic[17])*(algebraic[11]*algebraic[4]*constants[32]*algebraic[6]*algebraic[14]*algebraic[9]+algebraic[11]*algebraic[4]*algebraic[5]*constants[33]*algebraic[12]*algebraic[9]+algebraic[11]*algebraic[4]*constants[32]*constants[33]*algebraic[12]*algebraic[9]+algebraic[11]*algebraic[4]*constants[32]*algebraic[6]*algebraic[12]*algebraic[9])
    algebraic[22] = algebraic[11]*algebraic[7]*algebraic[6]*algebraic[14]*algebraic[9]+algebraic[3]*algebraic[7]*algebraic[6]*algebraic[14]*algebraic[9]+algebraic[11]*constants[31]*algebraic[6]*algebraic[14]*algebraic[9]+algebraic[3]*constants[31]*algebraic[6]*algebraic[14]*algebraic[9]+algebraic[11]*algebraic[4]*algebraic[6]*algebraic[14]*algebraic[9]+algebraic[11]*constants[31]*constants[33]*algebraic[12]*algebraic[9]+algebraic[3]*constants[31]*constants[33]*algebraic[12]*algebraic[9]+algebraic[11]*constants[31]*algebraic[6]*algebraic[12]*algebraic[9]+algebraic[3]*constants[31]*algebraic[6]*algebraic[12]*algebraic[9]+algebraic[3]*constants[31]*constants[33]*algebraic[8]*algebraic[10]+algebraic[3]*constants[31]*algebraic[6]*algebraic[8]*algebraic[10]+algebraic[3]*algebraic[7]*algebraic[6]*algebraic[14]*algebraic[10]+algebraic[3]*constants[31]*algebraic[6]*algebraic[14]*algebraic[10]+algebraic[3]*constants[31]*constants[33]*algebraic[12]*algebraic[10]+algebraic[3]*constants[31]*algebraic[6]*algebraic[12]*algebraic[10]
    algebraic[23] = (1.00000/algebraic[17])*(constants[2]*algebraic[11]*algebraic[6]*algebraic[14]*algebraic[9]+constants[2]*algebraic[11]*algebraic[5]*algebraic[14]*algebraic[9]+constants[2]*algebraic[11]*constants[33]*algebraic[12]*algebraic[9]+constants[2]*algebraic[11]*algebraic[6]*algebraic[12]*algebraic[9]+constants[2]*algebraic[11]*algebraic[5]*algebraic[12]*algebraic[9]+constants[2]*algebraic[11]*algebraic[5]*constants[33]*algebraic[9]+constants[2]*algebraic[5]*constants[33]*algebraic[8]*algebraic[10]+constants[2]*algebraic[11]*algebraic[5]*constants[33]*algebraic[8])
    algebraic[24] = algebraic[11]*algebraic[7]*algebraic[5]*algebraic[14]*algebraic[9]+algebraic[3]*algebraic[7]*algebraic[5]*algebraic[14]*algebraic[9]+algebraic[11]*constants[31]*algebraic[5]*algebraic[14]*algebraic[9]+algebraic[3]*constants[31]*algebraic[5]*algebraic[14]*algebraic[9]+algebraic[11]*algebraic[4]*algebraic[5]*algebraic[14]*algebraic[9]+algebraic[11]*algebraic[7]*constants[32]*algebraic[14]*algebraic[9]+algebraic[3]*algebraic[7]*constants[32]*algebraic[14]*algebraic[9]+algebraic[11]*algebraic[4]*constants[32]*algebraic[14]*algebraic[9]+algebraic[11]*constants[31]*algebraic[5]*algebraic[12]*algebraic[9]+algebraic[3]*constants[31]*algebraic[5]*algebraic[12]*algebraic[9]+algebraic[3]*constants[31]*algebraic[5]*algebraic[8]*algebraic[10]+algebraic[3]*algebraic[7]*algebraic[5]*algebraic[14]*algebraic[10]+algebraic[3]*constants[31]*algebraic[5]*algebraic[14]*algebraic[10]+algebraic[3]*algebraic[7]*constants[32]*algebraic[14]*algebraic[10]+algebraic[3]*constants[31]*algebraic[5]*algebraic[12]*algebraic[10]
    algebraic[25] = (1.00000/algebraic[17])*((-algebraic[1]*algebraic[7]*algebraic[5]*constants[33]*algebraic[8]*algebraic[10]-algebraic[1]*algebraic[7]*constants[32]*constants[33]*algebraic[8]*algebraic[10])-algebraic[1]*algebraic[7]*constants[32]*algebraic[6]*algebraic[8]*algebraic[10])
    algebraic[26] = (1.00000/algebraic[17])*(algebraic[4]*constants[32]*algebraic[6]*algebraic[14]*algebraic[9]+algebraic[11]*constants[32]*algebraic[6]*algebraic[14]*algebraic[9]+algebraic[11]*algebraic[7]*algebraic[6]*algebraic[14]*algebraic[9]+algebraic[11]*algebraic[4]*algebraic[6]*algebraic[14]*algebraic[9]+algebraic[11]*algebraic[7]*algebraic[5]*algebraic[14]*algebraic[9]+algebraic[11]*algebraic[4]*algebraic[5]*algebraic[14]*algebraic[9]+algebraic[11]*algebraic[7]*constants[32]*algebraic[14]*algebraic[9]+algebraic[11]*algebraic[4]*constants[32]*algebraic[14]*algebraic[9]+algebraic[4]*algebraic[5]*constants[33]*algebraic[12]*algebraic[9]+algebraic[11]*algebraic[5]*constants[33]*algebraic[12]*algebraic[9]+algebraic[4]*constants[32]*constants[33]*algebraic[12]*algebraic[9]+algebraic[11]*constants[32]*constants[33]*algebraic[12]*algebraic[9]+algebraic[4]*constants[32]*algebraic[6]*algebraic[12]*algebraic[9]+algebraic[11]*constants[32]*algebraic[6]*algebraic[12]*algebraic[9]+algebraic[11]*algebraic[7]*algebraic[5]*constants[33]*algebraic[9]+algebraic[11]*algebraic[4]*algebraic[5]*constants[33]*algebraic[9]+algebraic[11]*algebraic[7]*constants[32]*constants[33]*algebraic[9]+algebraic[11]*algebraic[4]*constants[32]*constants[33]*algebraic[9]+algebraic[11]*algebraic[7]*constants[32]*algebraic[6]*algebraic[9]+algebraic[11]*algebraic[4]*constants[32]*algebraic[6]*algebraic[9]+algebraic[7]*algebraic[5]*constants[33]*algebraic[8]*algebraic[10]+algebraic[4]*algebraic[5]*constants[33]*algebraic[8]*algebraic[10]+algebraic[7]*constants[32]*constants[33]*algebraic[8]*algebraic[10]+algebraic[4]*constants[32]*constants[33]*algebraic[8]*algebraic[10]+algebraic[7]*constants[32]*algebraic[6]*algebraic[8]*algebraic[10]+algebraic[4]*constants[32]*algebraic[6]*algebraic[8]*algebraic[10]+algebraic[4]*constants[32]*algebraic[6]*algebraic[14]*algebraic[10]+algebraic[4]*algebraic[5]*constants[33]*algebraic[12]*algebraic[10]+algebraic[4]*constants[32]*constants[33]*algebraic[12]*algebraic[10]+algebraic[4]*constants[32]*algebraic[6]*algebraic[12]*algebraic[10]+algebraic[11]*algebraic[7]*algebraic[5]*constants[33]*algebraic[8]+algebraic[11]*algebraic[4]*algebraic[5]*constants[33]*algebraic[8]+algebraic[11]*algebraic[7]*constants[32]*constants[33]*algebraic[8]+algebraic[11]*algebraic[4]*constants[32]*constants[33]*algebraic[8]+algebraic[11]*algebraic[7]*constants[32]*algebraic[6]*algebraic[8]+algebraic[11]*algebraic[4]*constants[32]*algebraic[6]*algebraic[8]+algebraic[11]*algebraic[4]*constants[32]*algebraic[6]*algebraic[14]+algebraic[11]*algebraic[4]*algebraic[5]*constants[33]*algebraic[12]+algebraic[11]*algebraic[4]*constants[32]*constants[33]*algebraic[12]+algebraic[11]*algebraic[4]*constants[32]*algebraic[6]*algebraic[12])
    algebraic[27] = algebraic[11]*algebraic[7]*algebraic[5]*constants[33]*algebraic[9]+algebraic[3]*algebraic[7]*algebraic[5]*constants[33]*algebraic[9]+algebraic[11]*constants[31]*algebraic[5]*constants[33]*algebraic[9]+algebraic[3]*constants[31]*algebraic[5]*constants[33]*algebraic[9]+algebraic[11]*algebraic[4]*algebraic[5]*constants[33]*algebraic[9]+algebraic[11]*algebraic[7]*constants[32]*constants[33]*algebraic[9]+algebraic[3]*algebraic[7]*constants[32]*constants[33]*algebraic[9]+algebraic[11]*algebraic[4]*constants[32]*constants[33]*algebraic[9]+algebraic[11]*algebraic[7]*constants[32]*algebraic[6]*algebraic[9]+algebraic[3]*algebraic[7]*constants[32]*algebraic[6]*algebraic[9]+algebraic[11]*algebraic[4]*constants[32]*algebraic[6]*algebraic[9]+algebraic[3]*algebraic[7]*algebraic[5]*constants[33]*algebraic[10]+algebraic[3]*constants[31]*algebraic[5]*constants[33]*algebraic[10]+algebraic[3]*algebraic[7]*constants[32]*constants[33]*algebraic[10]+algebraic[3]*algebraic[7]*constants[32]*algebraic[6]*algebraic[10]
    algebraic[28] = (1.00000/algebraic[17])*-algebraic[1]*constants[2]*algebraic[5]*constants[33]*algebraic[8]*algebraic[10]
    algebraic[29] = (algebraic[26]+algebraic[23]*constants[28])/(algebraic[20]+constants[28])
    algebraic[30] = (algebraic[26]+algebraic[20]*(power(constants[26], 2.00000)))/(algebraic[23]+power(constants[26], 2.00000))
    algebraic[31] = algebraic[11]*algebraic[7]*algebraic[5]*constants[33]*algebraic[8]+algebraic[3]*algebraic[7]*algebraic[5]*constants[33]*algebraic[8]+algebraic[11]*constants[31]*algebraic[5]*constants[33]*algebraic[8]+algebraic[3]*constants[31]*algebraic[5]*constants[33]*algebraic[8]+algebraic[11]*algebraic[4]*algebraic[5]*constants[33]*algebraic[8]+algebraic[11]*algebraic[7]*constants[32]*constants[33]*algebraic[8]+algebraic[3]*algebraic[7]*constants[32]*constants[33]*algebraic[8]+algebraic[11]*algebraic[4]*constants[32]*constants[33]*algebraic[8]+algebraic[11]*algebraic[7]*constants[32]*algebraic[6]*algebraic[8]+algebraic[3]*algebraic[7]*constants[32]*algebraic[6]*algebraic[8]+algebraic[11]*algebraic[4]*constants[32]*algebraic[6]*algebraic[8]+algebraic[11]*algebraic[4]*constants[32]*algebraic[6]*algebraic[14]+algebraic[11]*algebraic[4]*algebraic[5]*constants[33]*algebraic[12]+algebraic[11]*algebraic[4]*constants[32]*constants[33]*algebraic[12]+algebraic[11]*algebraic[4]*constants[32]*algebraic[6]*algebraic[12]
    algebraic[32] = (2.00000*constants[22]*constants[30]*(algebraic[25]+algebraic[28]*constants[28]))/(algebraic[20]+constants[28])
    algebraic[33] = (2.00000*constants[22]*constants[30]*algebraic[28]*(power(constants[26], 2.00000)))/(algebraic[23]+power(constants[26], 2.00000))
    algebraic[34] = power(algebraic[29], 1.0/2)
    algebraic[35] = algebraic[16]+algebraic[18]+algebraic[22]+algebraic[24]+algebraic[27]+algebraic[31]
    algebraic[36] = (constants[30]*algebraic[16])/algebraic[35]
    algebraic[37] = (constants[30]*algebraic[18])/algebraic[35]
    algebraic[38] = (constants[30]*algebraic[22])/algebraic[35]
    algebraic[39] = (constants[30]*algebraic[24])/algebraic[35]
    algebraic[40] = (constants[30]*algebraic[27])/algebraic[35]
    algebraic[41] = (constants[30]*algebraic[31])/algebraic[35]
    algebraic[42] = -constants[22]*(constants[20]*(algebraic[11]*algebraic[36]-algebraic[10]*algebraic[41])+(constants[20]+constants[21]*constants[19])*(algebraic[7]*algebraic[37]-algebraic[12]*algebraic[40])+(constants[20]+constants[21]*constants[19])*(algebraic[5]*algebraic[38]-algebraic[6]*algebraic[39]))
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