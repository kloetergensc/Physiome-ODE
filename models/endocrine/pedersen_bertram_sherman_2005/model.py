# Size of variable arrays:
sizeAlgebraic = 75
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
    legend_voi = "time in component environment (millisecond)"
    legend_states[0] = "Gi in component Gi (millimolar)"
    legend_algebraic[6] = "Jglut in component Jglut (millimolar_per_millisecond)"
    legend_algebraic[7] = "Jgk in component Jgk (millimolar_per_millisecond)"
    legend_algebraic[3] = "Ge in component Ge (millimolar)"
    legend_constants[0] = "gluper in component Ge (dimensionless)"
    legend_constants[1] = "gluamp in component Ge (millimolar)"
    legend_constants[2] = "glubasa1 in component Ge (millimolar)"
    legend_constants[3] = "glustep in component Ge (dimensionless)"
    legend_constants[4] = "steptime in component Ge (millisecond)"
    legend_algebraic[0] = "GeStep in component Ge (millimolar)"
    legend_states[1] = "G6P in component G6P (millimolar)"
    legend_constants[5] = "kappa in component model_parameters (dimensionless)"
    legend_algebraic[74] = "JPFK in component JPFK (micromolar_per_millisecond)"
    legend_states[2] = "FBP in component FBP (micromolar)"
    legend_algebraic[8] = "JGPDH in component JGPDH (micromolar_per_millisecond)"
    legend_algebraic[5] = "F6P in component F6P (millimolar)"
    legend_constants[6] = "Kglut in component Jglut (millimolar)"
    legend_constants[7] = "Vglut in component Jglut (millimolar_per_millisecond)"
    legend_constants[8] = "Kgk in component Jgk (millimolar)"
    legend_constants[9] = "Vgk in component Jgk (millimolar_per_millisecond)"
    legend_constants[10] = "ngk in component Jgk (dimensionless)"
    legend_constants[11] = "pfkbas in component JPFK (dimensionless)"
    legend_constants[12] = "cat in component JPFK (micromolar_per_millisecond)"
    legend_algebraic[60] = "topb in component w (dimensionless)"
    legend_algebraic[72] = "topa16 in component w (dimensionless)"
    legend_algebraic[73] = "bottom16 in component w (dimensionless)"
    legend_constants[51] = "weight1 in component w (dimensionless)"
    legend_algebraic[51] = "weight9 in component w (dimensionless)"
    legend_algebraic[9] = "weight5 in component w (dimensionless)"
    legend_algebraic[10] = "weight3 in component w (dimensionless)"
    legend_algebraic[15] = "weight2 in component w (dimensionless)"
    legend_algebraic[53] = "weight13 in component w (dimensionless)"
    legend_algebraic[54] = "weight11 in component w (dimensionless)"
    legend_algebraic[56] = "weight10 in component w (dimensionless)"
    legend_algebraic[23] = "weight6 in component w (dimensionless)"
    legend_algebraic[26] = "weight4 in component w (dimensionless)"
    legend_algebraic[59] = "weight15 in component w (dimensionless)"
    legend_algebraic[45] = "weight8 in component w (dimensionless)"
    legend_algebraic[61] = "weight12 in component w (dimensionless)"
    legend_algebraic[66] = "weight14 in component w (dimensionless)"
    legend_algebraic[71] = "weight16 in component w (dimensionless)"
    legend_algebraic[12] = "weight7 in component w (dimensionless)"
    legend_constants[50] = "topa1 in component w (dimensionless)"
    legend_constants[53] = "topa2 in component w (dimensionless)"
    legend_algebraic[11] = "topa3 in component w (dimensionless)"
    legend_algebraic[29] = "topa4 in component w (dimensionless)"
    legend_algebraic[32] = "topa5 in component w (dimensionless)"
    legend_algebraic[35] = "topa6 in component w (dimensionless)"
    legend_algebraic[43] = "topa7 in component w (dimensionless)"
    legend_algebraic[46] = "topa8 in component w (dimensionless)"
    legend_algebraic[47] = "topa9 in component w (dimensionless)"
    legend_algebraic[48] = "topa10 in component w (dimensionless)"
    legend_algebraic[55] = "topa11 in component w (dimensionless)"
    legend_algebraic[62] = "topa12 in component w (dimensionless)"
    legend_algebraic[63] = "topa13 in component w (dimensionless)"
    legend_algebraic[67] = "topa14 in component w (dimensionless)"
    legend_algebraic[68] = "topa15 in component w (dimensionless)"
    legend_constants[52] = "bottom1 in component w (dimensionless)"
    legend_algebraic[18] = "bottom2 in component w (dimensionless)"
    legend_algebraic[21] = "bottom3 in component w (dimensionless)"
    legend_algebraic[37] = "bottom4 in component w (dimensionless)"
    legend_algebraic[39] = "bottom5 in component w (dimensionless)"
    legend_algebraic[41] = "bottom6 in component w (dimensionless)"
    legend_algebraic[44] = "bottom7 in component w (dimensionless)"
    legend_algebraic[49] = "bottom8 in component w (dimensionless)"
    legend_algebraic[52] = "bottom9 in component w (dimensionless)"
    legend_algebraic[57] = "bottom10 in component w (dimensionless)"
    legend_algebraic[58] = "bottom11 in component w (dimensionless)"
    legend_algebraic[64] = "bottom12 in component w (dimensionless)"
    legend_algebraic[65] = "bottom13 in component w (dimensionless)"
    legend_algebraic[69] = "bottom14 in component w (dimensionless)"
    legend_algebraic[70] = "bottom15 in component w (dimensionless)"
    legend_constants[13] = "famp in component w (dimensionless)"
    legend_constants[14] = "ffbp in component w (dimensionless)"
    legend_constants[15] = "fmt in component w (dimensionless)"
    legend_constants[16] = "fbt in component w (dimensionless)"
    legend_constants[17] = "fatp in component w (dimensionless)"
    legend_constants[18] = "K1 in component w (micromolar)"
    legend_constants[19] = "K2 in component w (micromolar)"
    legend_constants[20] = "K3 in component w (micromolar)"
    legend_constants[21] = "K4 in component w (micromolar)"
    legend_algebraic[50] = "AMP in component AMP (micromolar)"
    legend_algebraic[14] = "ATP in component ATP (micromolar)"
    legend_constants[22] = "Atot in component ATP (micromolar)"
    legend_algebraic[13] = "rad in component ATP (micromolar)"
    legend_states[3] = "ADP in component ADP (micromolar)"
    legend_algebraic[16] = "y in component ADP (dimensionless)"
    legend_constants[23] = "tau_a in component ADP (millisecond)"
    legend_constants[24] = "r in component ADP (dimensionless)"
    legend_constants[25] = "r1 in component ADP (micromolar)"
    legend_constants[26] = "autoadp in component ADP (dimensionless)"
    legend_constants[27] = "adpknot in component ADP (micromolar)"
    legend_algebraic[19] = "fback in component ADP (dimensionless)"
    legend_constants[28] = "ky in component ADP (dimensionless)"
    legend_constants[29] = "kg in component ADP (micromolar_per_millisecond)"
    legend_states[4] = "Ca in component Ca (micromolar)"
    legend_states[5] = "v in component membrane (millivolt)"
    legend_constants[30] = "cm in component membrane (femtofarad)"
    legend_algebraic[22] = "I_Ca in component I_Ca (femtoampere)"
    legend_algebraic[17] = "I_K in component I_K (femtoampere)"
    legend_algebraic[24] = "I_K_Ca in component I_K_Ca (femtoampere)"
    legend_algebraic[42] = "I_K_ATP in component I_K_ATP (femtoampere)"
    legend_constants[31] = "gK in component I_K (picosiemens)"
    legend_constants[32] = "vK in component model_parameters (millivolt)"
    legend_states[6] = "n in component n (dimensionless)"
    legend_algebraic[4] = "n_infinity in component n (dimensionless)"
    legend_algebraic[1] = "tau_n in component n (millisecond)"
    legend_constants[33] = "gCa in component I_Ca (picosiemens)"
    legend_constants[34] = "vCa in component model_parameters (millivolt)"
    legend_algebraic[20] = "m_infinity in component m (dimensionless)"
    legend_constants[35] = "gkCa in component I_K_Ca (picosiemens)"
    legend_constants[36] = "KD in component I_K_Ca (micromolar)"
    legend_constants[37] = "nh in component I_K_Ca (dimensionless)"
    legend_constants[38] = "gkATP_bar in component I_K_ATP (picosiemens)"
    legend_algebraic[40] = "katpo in component I_K_ATP (dimensionless)"
    legend_algebraic[30] = "topo in component I_K_ATP (dimensionless)"
    legend_algebraic[38] = "bottomo in component I_K_ATP (dimensionless)"
    legend_algebraic[27] = "MgADP in component I_K_ATP (micromolar)"
    legend_algebraic[33] = "ADP3 in component I_K_ATP (micromolar)"
    legend_algebraic[36] = "ATP4 in component I_K_ATP (micromolar)"
    legend_constants[39] = "fcyt in component Ca (dimensionless)"
    legend_algebraic[25] = "Jmem in component Jmem (micromolar_per_millisecond)"
    legend_algebraic[34] = "Jer in component Jer (micromolar_per_millisecond)"
    legend_states[7] = "Caer in component Caer (micromolar)"
    legend_constants[40] = "fer in component Caer (dimensionless)"
    legend_constants[41] = "sigmav in component Caer (dimensionless)"
    legend_constants[42] = "kPMCA in component Jmem (first_order_rate_constant)"
    legend_constants[43] = "alpha in component Jmem (micromolar_per_millisecond)"
    legend_algebraic[31] = "Jleak in component Jleak (micromolar_per_millisecond)"
    legend_algebraic[28] = "JSERCA in component JSERCA (micromolar_per_millisecond)"
    legend_constants[44] = "kSERCA in component JSERCA (first_order_rate_constant)"
    legend_constants[45] = "pleak in component Jleak (first_order_rate_constant)"
    legend_states[8] = "I in component I (dimensionless)"
    legend_algebraic[2] = "I_infinity in component I (dimensionless)"
    legend_constants[46] = "tau_I in component I (millisecond)"
    legend_constants[47] = "I_max in component I (dimensionless)"
    legend_constants[48] = "delta in component I (dimensionless)"
    legend_constants[49] = "ki in component I (micromolar)"
    legend_rates[0] = "d/dt Gi in component Gi (millimolar)"
    legend_rates[1] = "d/dt G6P in component G6P (millimolar)"
    legend_rates[2] = "d/dt FBP in component FBP (micromolar)"
    legend_rates[3] = "d/dt ADP in component ADP (micromolar)"
    legend_rates[5] = "d/dt v in component membrane (millivolt)"
    legend_rates[6] = "d/dt n in component n (dimensionless)"
    legend_rates[4] = "d/dt Ca in component Ca (micromolar)"
    legend_rates[7] = "d/dt Caer in component Caer (micromolar)"
    legend_rates[8] = "d/dt I in component I (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 6.0637
    constants[0] = 7.0
    constants[1] = 0.0
    constants[2] = 7.0
    constants[3] = 0.0
    constants[4] = 480000
    states[1] = 525.97
    constants[5] = 0.005
    states[2] = 0.2088
    constants[6] = 7
    constants[7] = 8
    constants[8] = 7
    constants[9] = 0.8
    constants[10] = 4
    constants[11] = 0.06
    constants[12] = 2
    constants[13] = 0.02
    constants[14] = 0.2
    constants[15] = 20
    constants[16] = 20
    constants[17] = 20
    constants[18] = 30
    constants[19] = 1
    constants[20] = 50000
    constants[21] = 1000
    constants[22] = 3000
    states[3] = 537.6
    constants[23] = 300000
    constants[24] = 0.5
    constants[25] = 0.35
    constants[26] = 1.0
    constants[27] = 800.0
    constants[28] = 2.2
    constants[29] = 0.1
    states[4] = 0.05626
    states[5] = -66.7
    constants[30] = 5300
    constants[31] = 2700
    constants[32] = -75
    states[6] = 0.00012
    constants[33] = 1000
    constants[34] = 25
    constants[35] = 400
    constants[36] = 0.5
    constants[37] = 2.0
    constants[38] = 2000
    constants[39] = 0.01
    states[7] = 121.8
    constants[40] = 0.01
    constants[41] = 31
    constants[42] = 0.18
    constants[43] = 4.5e-6
    constants[44] = 0.4
    constants[45] = 0.0002
    states[8] = 0.316
    constants[46] = 10000
    constants[47] = 20
    constants[48] = 8.0
    constants[49] = 0.1
    constants[50] = 0.00000
    constants[51] = 1.00000
    constants[52] = 1.00000
    constants[53] = constants[50]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[2] = constants[47]*((power(states[4], constants[48]))/(power(constants[49], constants[48])+power(states[4], constants[48])))
    rates[8] = (algebraic[2]-states[8])/constants[46]
    algebraic[4] = 0.500000*(1.00000+tanh((states[5]--16.0000)/11.2000))
    algebraic[1] = 1.00000/(0.0350000*cosh((states[5]--16.0000)/22.4000))
    rates[6] = (algebraic[4]-states[6])/algebraic[1]
    algebraic[0] = custom_piecewise([greater((voi-constants[4])/1000.00 , 1.00000), constants[3]*constants[1] , greater((voi-constants[4])/1000.00 , 0.00000) & less((voi-constants[4])/1000.00 , 1.00000), ((voi-constants[4])/1000.00)*constants[3]*constants[1] , less((voi-constants[4])/1000.00 , 0.00000), 0.00000 , True, float('nan')])
    algebraic[3] = constants[2]+constants[1]*(1.00000-constants[3])*cos((voi*2.00000* pi)/(60000.0*constants[0]))+algebraic[0]
    algebraic[6] = constants[7]*(((algebraic[3]-states[0])*constants[6])/((constants[6]+algebraic[3])*(constants[6]+states[0])))
    algebraic[7] = (constants[9]*(power(states[0], constants[10])))/(power(constants[8], constants[10])+power(states[0], constants[10]))
    rates[0] = algebraic[6]-algebraic[7]
    algebraic[13] = power(power(states[3]-constants[22], 2.00000)-4.00000*(power(states[3], 2.00000)), 1.0/2)
    algebraic[14] = 0.500000*((constants[22]+algebraic[13])-states[3])
    algebraic[8] = 0.200000*(power(states[2]/1.00000, 1.0/2))
    algebraic[16] = (constants[28]*algebraic[8])/(constants[29]+algebraic[8])
    algebraic[19] = constants[24]+algebraic[16]
    rates[3] = constants[26]*((algebraic[14]-states[3]*exp(algebraic[19]*((1.00000-states[4])/constants[25])))/constants[23])+1.00000*(1.00000-constants[26])*(constants[27]-states[3])
    algebraic[20] = 0.500000*(1.00000+tanh((states[5]--20.0000)/24.0000))
    algebraic[22] = constants[33]*algebraic[20]*(states[5]-constants[34])
    algebraic[25] = -(constants[43]*1.00000*algebraic[22]+constants[42]*states[4])
    algebraic[31] = constants[45]*(states[7]-states[4])
    algebraic[28] = constants[44]*states[4]
    algebraic[34] = algebraic[31]-algebraic[28]
    rates[4] = constants[39]*(algebraic[25]+algebraic[34])
    rates[7] = -constants[40]*constants[41]*algebraic[34]
    algebraic[17] = constants[31]*states[6]*(states[5]-constants[32])
    algebraic[24] = (constants[35]/(1.00000+power(constants[36]/states[4], constants[37])))*(states[5]-constants[32])
    algebraic[27] = 0.165000*states[3]
    algebraic[30] = 0.0800000*(1.00000+(2.00000*algebraic[27])/17.0000)+0.890000*(power(algebraic[27]/17.0000, 2.00000))
    algebraic[33] = 0.135000*states[3]
    algebraic[36] = 0.0500000*algebraic[14]
    algebraic[38] = (power(1.00000+algebraic[27]/17.0000, 2.00000))*(1.00000+algebraic[33]/26.0000+algebraic[36]/1.00000)
    algebraic[40] = 20.0000*(algebraic[30]/algebraic[38])
    algebraic[42] = constants[38]*algebraic[40]*(states[5]-constants[32])
    rates[5] = -(algebraic[17]+algebraic[22]+algebraic[24]+algebraic[42])/constants[30]
    algebraic[5] = 0.300000*states[1]
    algebraic[50] = (power(states[3], 2.00000))/algebraic[14]
    algebraic[59] = (algebraic[50]*states[2]*(power(algebraic[5], 2.00000)))/(1.00000*constants[18]*constants[19]*constants[20]*constants[14]*constants[13])
    algebraic[60] = algebraic[59]
    algebraic[71] = (algebraic[50]*states[2]*(power(algebraic[5], 2.00000))*(power(algebraic[14], 2.00000)))/(1.00000*constants[18]*constants[19]*constants[20]*constants[21]*constants[14]*constants[13]*constants[16]*constants[15]*constants[17])
    algebraic[61] = (algebraic[50]*(power(algebraic[5], 2.00000))*(power(algebraic[14], 2.00000)))/(1.00000*constants[18]*constants[20]*constants[21]*constants[13]*constants[15]*constants[17])
    algebraic[54] = (algebraic[50]*(power(algebraic[5], 2.00000)))/(1.00000*constants[18]*constants[20]*constants[13])
    algebraic[45] = (states[2]*(power(algebraic[5], 2.00000))*(power(algebraic[14], 2.00000)))/(1.00000*constants[19]*constants[20]*constants[21]*constants[14]*constants[16]*constants[17])
    algebraic[12] = (states[2]*(power(algebraic[5], 2.00000)))/(1.00000*constants[19]*constants[20]*constants[14])
    algebraic[26] = (power(algebraic[5]*algebraic[14], 2.00000))/(1.00000*constants[17]*constants[20]*constants[21])
    algebraic[10] = (power(algebraic[5], 2.00000))/(1.00000*constants[20])
    algebraic[11] = constants[53]+algebraic[10]
    algebraic[29] = algebraic[11]+algebraic[26]
    algebraic[32] = algebraic[29]
    algebraic[35] = algebraic[32]
    algebraic[43] = algebraic[35]+algebraic[12]
    algebraic[46] = algebraic[43]+algebraic[45]
    algebraic[47] = algebraic[46]
    algebraic[48] = algebraic[47]
    algebraic[55] = algebraic[48]+algebraic[54]
    algebraic[62] = algebraic[55]+algebraic[61]
    algebraic[63] = algebraic[62]
    algebraic[67] = algebraic[63]
    algebraic[68] = algebraic[67]
    algebraic[72] = algebraic[68]+algebraic[71]
    algebraic[66] = (algebraic[50]*states[2]*(power(algebraic[14], 2.00000)))/(1.00000*constants[18]*constants[19]*constants[21]*constants[16]*constants[15])
    algebraic[53] = (algebraic[50]*states[2])/(constants[18]*constants[19])
    algebraic[56] = (algebraic[50]*(power(algebraic[14], 2.00000)))/(1.00000*constants[18]*constants[21]*constants[15])
    algebraic[51] = algebraic[50]/constants[18]
    algebraic[23] = (states[2]*(power(algebraic[14], 2.00000)))/(1.00000*constants[19]*constants[21]*constants[16])
    algebraic[9] = states[2]/constants[19]
    algebraic[15] = (power(algebraic[14], 2.00000))/(1.00000*constants[21])
    algebraic[18] = constants[52]+algebraic[15]
    algebraic[21] = algebraic[18]+algebraic[10]
    algebraic[37] = algebraic[21]+algebraic[26]
    algebraic[39] = algebraic[37]+algebraic[9]
    algebraic[41] = algebraic[39]+algebraic[23]
    algebraic[44] = algebraic[41]+algebraic[12]
    algebraic[49] = algebraic[44]+algebraic[45]
    algebraic[52] = algebraic[49]+algebraic[51]
    algebraic[57] = algebraic[52]+algebraic[56]
    algebraic[58] = algebraic[57]+algebraic[54]
    algebraic[64] = algebraic[58]+algebraic[61]
    algebraic[65] = algebraic[64]+algebraic[53]
    algebraic[69] = algebraic[65]+algebraic[66]
    algebraic[70] = algebraic[69]+algebraic[59]
    algebraic[73] = algebraic[70]+algebraic[71]
    algebraic[74] = (constants[11]*constants[12]*algebraic[72]+constants[12]*algebraic[60])/algebraic[73]
    rates[1] = constants[5]*(algebraic[7]-algebraic[74])
    rates[2] = constants[5]*(algebraic[74]-0.500000*algebraic[8])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[2] = constants[47]*((power(states[4], constants[48]))/(power(constants[49], constants[48])+power(states[4], constants[48])))
    algebraic[4] = 0.500000*(1.00000+tanh((states[5]--16.0000)/11.2000))
    algebraic[1] = 1.00000/(0.0350000*cosh((states[5]--16.0000)/22.4000))
    algebraic[0] = custom_piecewise([greater((voi-constants[4])/1000.00 , 1.00000), constants[3]*constants[1] , greater((voi-constants[4])/1000.00 , 0.00000) & less((voi-constants[4])/1000.00 , 1.00000), ((voi-constants[4])/1000.00)*constants[3]*constants[1] , less((voi-constants[4])/1000.00 , 0.00000), 0.00000 , True, float('nan')])
    algebraic[3] = constants[2]+constants[1]*(1.00000-constants[3])*cos((voi*2.00000* pi)/(60000.0*constants[0]))+algebraic[0]
    algebraic[6] = constants[7]*(((algebraic[3]-states[0])*constants[6])/((constants[6]+algebraic[3])*(constants[6]+states[0])))
    algebraic[7] = (constants[9]*(power(states[0], constants[10])))/(power(constants[8], constants[10])+power(states[0], constants[10]))
    algebraic[13] = power(power(states[3]-constants[22], 2.00000)-4.00000*(power(states[3], 2.00000)), 1.0/2)
    algebraic[14] = 0.500000*((constants[22]+algebraic[13])-states[3])
    algebraic[8] = 0.200000*(power(states[2]/1.00000, 1.0/2))
    algebraic[16] = (constants[28]*algebraic[8])/(constants[29]+algebraic[8])
    algebraic[19] = constants[24]+algebraic[16]
    algebraic[20] = 0.500000*(1.00000+tanh((states[5]--20.0000)/24.0000))
    algebraic[22] = constants[33]*algebraic[20]*(states[5]-constants[34])
    algebraic[25] = -(constants[43]*1.00000*algebraic[22]+constants[42]*states[4])
    algebraic[31] = constants[45]*(states[7]-states[4])
    algebraic[28] = constants[44]*states[4]
    algebraic[34] = algebraic[31]-algebraic[28]
    algebraic[17] = constants[31]*states[6]*(states[5]-constants[32])
    algebraic[24] = (constants[35]/(1.00000+power(constants[36]/states[4], constants[37])))*(states[5]-constants[32])
    algebraic[27] = 0.165000*states[3]
    algebraic[30] = 0.0800000*(1.00000+(2.00000*algebraic[27])/17.0000)+0.890000*(power(algebraic[27]/17.0000, 2.00000))
    algebraic[33] = 0.135000*states[3]
    algebraic[36] = 0.0500000*algebraic[14]
    algebraic[38] = (power(1.00000+algebraic[27]/17.0000, 2.00000))*(1.00000+algebraic[33]/26.0000+algebraic[36]/1.00000)
    algebraic[40] = 20.0000*(algebraic[30]/algebraic[38])
    algebraic[42] = constants[38]*algebraic[40]*(states[5]-constants[32])
    algebraic[5] = 0.300000*states[1]
    algebraic[50] = (power(states[3], 2.00000))/algebraic[14]
    algebraic[59] = (algebraic[50]*states[2]*(power(algebraic[5], 2.00000)))/(1.00000*constants[18]*constants[19]*constants[20]*constants[14]*constants[13])
    algebraic[60] = algebraic[59]
    algebraic[71] = (algebraic[50]*states[2]*(power(algebraic[5], 2.00000))*(power(algebraic[14], 2.00000)))/(1.00000*constants[18]*constants[19]*constants[20]*constants[21]*constants[14]*constants[13]*constants[16]*constants[15]*constants[17])
    algebraic[61] = (algebraic[50]*(power(algebraic[5], 2.00000))*(power(algebraic[14], 2.00000)))/(1.00000*constants[18]*constants[20]*constants[21]*constants[13]*constants[15]*constants[17])
    algebraic[54] = (algebraic[50]*(power(algebraic[5], 2.00000)))/(1.00000*constants[18]*constants[20]*constants[13])
    algebraic[45] = (states[2]*(power(algebraic[5], 2.00000))*(power(algebraic[14], 2.00000)))/(1.00000*constants[19]*constants[20]*constants[21]*constants[14]*constants[16]*constants[17])
    algebraic[12] = (states[2]*(power(algebraic[5], 2.00000)))/(1.00000*constants[19]*constants[20]*constants[14])
    algebraic[26] = (power(algebraic[5]*algebraic[14], 2.00000))/(1.00000*constants[17]*constants[20]*constants[21])
    algebraic[10] = (power(algebraic[5], 2.00000))/(1.00000*constants[20])
    algebraic[11] = constants[53]+algebraic[10]
    algebraic[29] = algebraic[11]+algebraic[26]
    algebraic[32] = algebraic[29]
    algebraic[35] = algebraic[32]
    algebraic[43] = algebraic[35]+algebraic[12]
    algebraic[46] = algebraic[43]+algebraic[45]
    algebraic[47] = algebraic[46]
    algebraic[48] = algebraic[47]
    algebraic[55] = algebraic[48]+algebraic[54]
    algebraic[62] = algebraic[55]+algebraic[61]
    algebraic[63] = algebraic[62]
    algebraic[67] = algebraic[63]
    algebraic[68] = algebraic[67]
    algebraic[72] = algebraic[68]+algebraic[71]
    algebraic[66] = (algebraic[50]*states[2]*(power(algebraic[14], 2.00000)))/(1.00000*constants[18]*constants[19]*constants[21]*constants[16]*constants[15])
    algebraic[53] = (algebraic[50]*states[2])/(constants[18]*constants[19])
    algebraic[56] = (algebraic[50]*(power(algebraic[14], 2.00000)))/(1.00000*constants[18]*constants[21]*constants[15])
    algebraic[51] = algebraic[50]/constants[18]
    algebraic[23] = (states[2]*(power(algebraic[14], 2.00000)))/(1.00000*constants[19]*constants[21]*constants[16])
    algebraic[9] = states[2]/constants[19]
    algebraic[15] = (power(algebraic[14], 2.00000))/(1.00000*constants[21])
    algebraic[18] = constants[52]+algebraic[15]
    algebraic[21] = algebraic[18]+algebraic[10]
    algebraic[37] = algebraic[21]+algebraic[26]
    algebraic[39] = algebraic[37]+algebraic[9]
    algebraic[41] = algebraic[39]+algebraic[23]
    algebraic[44] = algebraic[41]+algebraic[12]
    algebraic[49] = algebraic[44]+algebraic[45]
    algebraic[52] = algebraic[49]+algebraic[51]
    algebraic[57] = algebraic[52]+algebraic[56]
    algebraic[58] = algebraic[57]+algebraic[54]
    algebraic[64] = algebraic[58]+algebraic[61]
    algebraic[65] = algebraic[64]+algebraic[53]
    algebraic[69] = algebraic[65]+algebraic[66]
    algebraic[70] = algebraic[69]+algebraic[59]
    algebraic[73] = algebraic[70]+algebraic[71]
    algebraic[74] = (constants[11]*constants[12]*algebraic[72]+constants[12]*algebraic[60])/algebraic[73]
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