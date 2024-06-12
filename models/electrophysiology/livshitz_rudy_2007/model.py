# Size of variable arrays:
sizeAlgebraic = 67
sizeStates = 18
sizeConstants = 68
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component Environment (ms)"
    legend_constants[0] = "F in component Environment (C_per_mole)"
    legend_constants[1] = "R in component Environment (mJ_per_mole_K)"
    legend_constants[2] = "Temp in component Environment (kelvin)"
    legend_constants[53] = "FonRT in component Environment (per_mV)"
    legend_constants[3] = "K_o in component Environment (mM)"
    legend_constants[4] = "Na_o in component Environment (mM)"
    legend_constants[5] = "Ca_o in component Environment (mM)"
    legend_states[0] = "V in component cell (mV)"
    legend_algebraic[52] = "ilca in component ICaL (uA_per_uF)"
    legend_algebraic[62] = "icab in component ICab (uA_per_uF)"
    legend_algebraic[61] = "ipca in component IpCa (uA_per_uF)"
    legend_algebraic[59] = "inaca in component INaCa (uA_per_uF)"
    legend_algebraic[64] = "icat in component ICaT (uA_per_uF)"
    legend_algebraic[30] = "ina in component INa (uA_per_uF)"
    legend_algebraic[31] = "inab in component INab (uA_per_uF)"
    legend_algebraic[53] = "ilcana in component ICaL (uA_per_uF)"
    legend_algebraic[28] = "inak in component INaK (uA_per_uF)"
    legend_algebraic[37] = "ikr in component IKr (uA_per_uF)"
    legend_algebraic[57] = "iks in component IKs (uA_per_uF)"
    legend_algebraic[36] = "IK1 in component IK1 (uA_per_uF)"
    legend_algebraic[38] = "ikp in component IKp (uA_per_uF)"
    legend_algebraic[55] = "ilcak in component ICaL (uA_per_uF)"
    legend_algebraic[65] = "caiont in component cell (uA_per_uF)"
    legend_algebraic[60] = "naiont in component cell (uA_per_uF)"
    legend_algebraic[58] = "kiont in component cell (uA_per_uF)"
    legend_constants[6] = "l in component cell (cm)"
    legend_constants[7] = "ra in component cell (cm)"
    legend_constants[56] = "vcell in component cell (uL)"
    legend_constants[57] = "ageo in component cell (cm2)"
    legend_constants[61] = "Acap in component cell (uF)"
    legend_constants[65] = "AF in component cell (uF_mole_per_C)"
    legend_constants[62] = "vmyo in component cell (uL)"
    legend_constants[58] = "vmito in component cell (uL)"
    legend_constants[59] = "vsr in component cell (uL)"
    legend_constants[63] = "vnsr in component cell (uL)"
    legend_constants[64] = "vjsr in component cell (uL)"
    legend_constants[60] = "vss in component cell (uL)"
    legend_algebraic[9] = "i_Stim in component cell (uA_per_uF)"
    legend_constants[8] = "stim_offset in component cell (ms)"
    legend_constants[9] = "stim_period in component cell (ms)"
    legend_constants[10] = "stim_duration in component cell (ms)"
    legend_constants[11] = "stim_amplitude in component cell (uA_per_uF)"
    legend_algebraic[0] = "past in component cell (ms)"
    legend_algebraic[29] = "ENa in component reversal_potentials (mV)"
    legend_states[1] = "H in component INa (dimensionless)"
    legend_states[2] = "m in component INa (dimensionless)"
    legend_states[3] = "J in component INa (dimensionless)"
    legend_constants[12] = "GNa in component INa (mS_per_uF)"
    legend_algebraic[1] = "a in component INa (per_ms)"
    legend_algebraic[10] = "aH in component INa (per_ms)"
    legend_algebraic[20] = "bH in component INa (per_ms)"
    legend_algebraic[11] = "aj in component INa (per_ms)"
    legend_algebraic[21] = "bj in component INa (per_ms)"
    legend_algebraic[2] = "am in component INa (per_ms)"
    legend_algebraic[12] = "bm in component INa (per_ms)"
    legend_algebraic[49] = "Ca_i in component Ca (mM)"
    legend_states[4] = "Na_i in component Na (mM)"
    legend_states[5] = "K_i in component K (mM)"
    legend_states[6] = "d in component ICaL (dimensionless)"
    legend_states[7] = "f in component ICaL (dimensionless)"
    legend_algebraic[3] = "dss0 in component ICaL (dimensionless)"
    legend_algebraic[13] = "taud in component ICaL (ms)"
    legend_algebraic[22] = "dss1 in component ICaL (dimensionless)"
    legend_algebraic[25] = "dss in component ICaL (dimensionless)"
    legend_algebraic[4] = "fss in component ICaL (dimensionless)"
    legend_algebraic[14] = "tauf in component ICaL (ms)"
    legend_constants[13] = "gacai in component ICaL (dimensionless)"
    legend_constants[14] = "gacao in component ICaL (dimensionless)"
    legend_constants[15] = "kmca in component ICaL (mM)"
    legend_constants[16] = "pca in component ICaL (L_per_F_ms)"
    legend_constants[17] = "pna in component ICaL (L_per_F_ms)"
    legend_constants[18] = "ganai in component ICaL (dimensionless)"
    legend_constants[19] = "ganao in component ICaL (dimensionless)"
    legend_constants[20] = "pk in component ICaL (L_per_F_ms)"
    legend_constants[21] = "gaki in component ICaL (dimensionless)"
    legend_constants[22] = "gako in component ICaL (dimensionless)"
    legend_algebraic[50] = "ibarca in component ICaL (uA_per_uF)"
    legend_algebraic[19] = "ibarna in component ICaL (uA_per_uF)"
    legend_algebraic[24] = "ibark in component ICaL (uA_per_uF)"
    legend_algebraic[51] = "fca in component ICaL (dimensionless)"
    legend_algebraic[32] = "EK in component reversal_potentials (mV)"
    legend_constants[23] = "GK1max in component IK1 (mS_per_uF)"
    legend_constants[54] = "GK1_ in component IK1 (mS_per_uF)"
    legend_algebraic[33] = "ak1 in component IK1 (per_ms)"
    legend_algebraic[34] = "bk1 in component IK1 (per_ms)"
    legend_algebraic[35] = "gK1 in component IK1 (mS_per_uF)"
    legend_constants[24] = "gkrmax in component IKr (mS_per_uF)"
    legend_states[8] = "xr in component IKr (dimensionless)"
    legend_algebraic[26] = "r in component IKr (dimensionless)"
    legend_algebraic[5] = "xrss in component IKr (dimensionless)"
    legend_algebraic[15] = "tauxr in component IKr (ms)"
    legend_algebraic[39] = "EKs in component reversal_potentials (mV)"
    legend_constants[25] = "GKsmax in component IKs (mS_per_uF)"
    legend_states[9] = "xs1 in component IKs (dimensionless)"
    legend_states[10] = "xs2 in component IKs (dimensionless)"
    legend_algebraic[56] = "gks in component IKs (mS_per_uF)"
    legend_algebraic[6] = "xss in component IKs (dimensionless)"
    legend_algebraic[16] = "tauxs in component IKs (ms)"
    legend_constants[26] = "kmnai in component INaK (mM)"
    legend_constants[27] = "kmko in component INaK (mM)"
    legend_constants[28] = "ibarnak in component INaK (uA_per_uF)"
    legend_algebraic[27] = "fnak in component INaK (dimensionless)"
    legend_constants[55] = "sigma in component INaK (dimensionless)"
    legend_constants[29] = "c1 in component INaCa (uA_per_uF)"
    legend_constants[30] = "c2 in component INaCa (dimensionless)"
    legend_constants[31] = "gammas in component INaCa (dimensionless)"
    legend_constants[32] = "GKpmax in component IKp (mS_per_uF)"
    legend_constants[33] = "ibarpca in component IpCa (uA_per_uF)"
    legend_constants[34] = "kmpca in component IpCa (mM)"
    legend_constants[35] = "gcab in component ICab (mS_per_uF)"
    legend_constants[36] = "GNab in component INab (mS_per_uF)"
    legend_algebraic[63] = "ECa in component reversal_potentials (mV)"
    legend_states[11] = "b in component ICaT (dimensionless)"
    legend_states[12] = "g in component ICaT (dimensionless)"
    legend_constants[37] = "gcat in component ICaT (mS_per_uF)"
    legend_algebraic[7] = "bss in component ICaT (dimensionless)"
    legend_algebraic[17] = "taub in component ICaT (ms)"
    legend_algebraic[8] = "gss in component ICaT (dimensionless)"
    legend_algebraic[18] = "aa in component ICaT (dimensionless)"
    legend_algebraic[23] = "taug in component ICaT (ms)"
    legend_constants[38] = "prnak in component reversal_potentials (dimensionless)"
    legend_states[13] = "Ca_JSR_T in component Ca (mM)"
    legend_states[14] = "Rel in component Irel (mM_per_ms)"
    legend_algebraic[54] = "Rel_ss in component Irel (mM_per_ms)"
    legend_algebraic[43] = "tau_Rel in component Irel (ms)"
    legend_constants[39] = "K_Relss in component Irel (mM)"
    legend_constants[66] = "alpha_Rel in component Irel (mM_per_mV)"
    legend_constants[40] = "tau in component Irel (ms)"
    legend_constants[41] = "kappa in component Irel (mM_per_mV_ms)"
    legend_constants[42] = "qn in component Irel (dimensionless)"
    legend_algebraic[42] = "Ca_JSR_free in component Irel (mM)"
    legend_algebraic[40] = "bbb in component Irel (mM)"
    legend_algebraic[41] = "c in component Irel (mM2)"
    legend_constants[43] = "kmcsqn in component Irel (mM)"
    legend_constants[44] = "csqnbar in component Irel (mM)"
    legend_states[15] = "Ca_NSR in component Ca (mM)"
    legend_constants[45] = "kmup in component Iup_Ileak (mM)"
    legend_constants[46] = "iupbar in component Iup_Ileak (mM_per_ms)"
    legend_constants[47] = "nsrbar in component Iup_Ileak (mM)"
    legend_algebraic[66] = "iup in component Iup_Ileak (mM_per_ms)"
    legend_algebraic[44] = "ileak in component Iup_Ileak (mM_per_ms)"
    legend_constants[48] = "tautr in component Itr (ms)"
    legend_algebraic[45] = "itr in component Itr (mM_per_ms)"
    legend_states[16] = "Ca_T in component Ca (mM)"
    legend_states[17] = "Over in component Ca (dimensionless)"
    legend_constants[49] = "cmdnbar in component Ca (mM)"
    legend_constants[50] = "trpnbar in component Ca (mM)"
    legend_constants[51] = "kmcmdn in component Ca (mM)"
    legend_constants[52] = "kmtrpn in component Ca (mM)"
    legend_algebraic[46] = "bmyo in component Ca (mM)"
    legend_algebraic[47] = "cmyo in component Ca (mM2)"
    legend_algebraic[48] = "dmyo in component Ca (mM3)"
    legend_rates[0] = "d/dt V in component cell (mV)"
    legend_rates[1] = "d/dt H in component INa (dimensionless)"
    legend_rates[2] = "d/dt m in component INa (dimensionless)"
    legend_rates[3] = "d/dt J in component INa (dimensionless)"
    legend_rates[6] = "d/dt d in component ICaL (dimensionless)"
    legend_rates[7] = "d/dt f in component ICaL (dimensionless)"
    legend_rates[8] = "d/dt xr in component IKr (dimensionless)"
    legend_rates[9] = "d/dt xs1 in component IKs (dimensionless)"
    legend_rates[10] = "d/dt xs2 in component IKs (dimensionless)"
    legend_rates[11] = "d/dt b in component ICaT (dimensionless)"
    legend_rates[12] = "d/dt g in component ICaT (dimensionless)"
    legend_rates[14] = "d/dt Rel in component Irel (mM_per_ms)"
    legend_rates[4] = "d/dt Na_i in component Na (mM)"
    legend_rates[5] = "d/dt K_i in component K (mM)"
    legend_rates[17] = "d/dt Over in component Ca (dimensionless)"
    legend_rates[16] = "d/dt Ca_T in component Ca (mM)"
    legend_rates[15] = "d/dt Ca_NSR in component Ca (mM)"
    legend_rates[13] = "d/dt Ca_JSR_T in component Ca (mM)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 96485
    constants[1] = 8314
    constants[2] = 310
    constants[3] = 4.5
    constants[4] = 140
    constants[5] = 1.8
    states[0] = -89.4356034692784
    constants[6] = 0.01
    constants[7] = 0.0011
    constants[8] = 0
    constants[9] = 400
    constants[10] = 3
    constants[11] = -15
    states[1] = 0.994401369032678
    states[2] = 0.000734780346592185
    states[3] = 0.996100688673679
    constants[12] = 16
    states[4] = 16.612739313555
    states[5] = 139.730914103161
    states[6] = 3.2514786721066e-27
    states[7] = 0.997404948824816
    constants[13] = 1
    constants[14] = 0.341
    constants[15] = 0.0006
    constants[16] = 0.00054
    constants[17] = 6.75e-7
    constants[18] = 0.75
    constants[19] = 0.75
    constants[20] = 1.93e-7
    constants[21] = 0.75
    constants[22] = 0.75
    constants[23] = 0.75
    constants[24] = 0.02614
    states[8] = 0.000162194715543637
    constants[25] = 0.433
    states[9] = 0.0285147332973946
    states[10] = 0.0764114040188678
    constants[26] = 10
    constants[27] = 1.5
    constants[28] = 2.25
    constants[29] = 0.00025
    constants[30] = 0.0001
    constants[31] = 0.15
    constants[32] = 0.00552
    constants[33] = 1.15
    constants[34] = 0.0005
    constants[35] = 0.003016
    constants[36] = 0.004
    states[11] = 0.000927461915392873
    states[12] = 0.952834331760863
    constants[37] = 0.05
    constants[38] = 0.01833
    states[13] = 7.87371650296567
    states[14] = 1.06874246141923e-23
    constants[39] = 1
    constants[40] = 4.75
    constants[41] = 0.125
    constants[42] = 9
    constants[43] = 0.8
    constants[44] = 10
    states[15] = 2.71518235696672
    constants[45] = 0.00092
    constants[46] = 0.00875
    constants[47] = 15
    constants[48] = 120
    states[16] = 0.0257059808595638
    states[17] = 1e-12
    constants[49] = 0.05
    constants[50] = 0.07
    constants[51] = 0.00238
    constants[52] = 0.0005
    constants[53] = (constants[0]/constants[2])/constants[1]
    constants[54] = constants[23]*(power(constants[3]/5.40000, 1.0/2))
    constants[55] = (exp(constants[4]/67.3000)-1.00000)/7.00000
    constants[67] = 0.00000
    constants[56] = 1000.00* pi*constants[7]*constants[7]*constants[6]
    constants[57] = 2.00000* pi*constants[7]*constants[7]+2.00000* pi*constants[7]*constants[6]
    constants[58] = constants[56]*0.240000
    constants[59] = constants[56]*0.0600000
    constants[60] = constants[56]*0.0200000
    constants[61] = constants[57]*2.00000
    constants[62] = constants[56]*0.680000
    constants[63] = constants[56]*0.0552000
    constants[64] = constants[56]*0.00480000
    constants[65] = constants[61]/constants[0]
    constants[66] = constants[40]*constants[41]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[17] = constants[67]
    algebraic[2] = (0.320000*1.00000*(states[0]+47.1300))/(1.00000-exp(-0.100000*(states[0]+47.1300)))
    algebraic[12] = 0.0800000*exp(-states[0]/11.0000)
    rates[2] = algebraic[2]*(1.00000-states[2])-algebraic[12]*states[2]
    algebraic[4] = 1.00000/(1.00000+exp((states[0]+32.0000)/8.00000))+0.600000/(1.00000+exp((50.0000-states[0])/20.0000))
    algebraic[14] = 1.00000/(0.0197000*exp(-(power(0.0337000*(states[0]+10.0000), 2.00000)))+0.0200000)
    rates[7] = (algebraic[4]-states[7])/algebraic[14]
    algebraic[5] = 1.00000/(1.00000+exp(-(states[0]+10.0850)/4.25000))
    algebraic[15] = 1.00000/((0.00138000*(states[0]+14.2000))/(1.00000-exp(-0.123000*(states[0]+14.2000)))+(0.000610000*(states[0]+38.9000))/(exp(0.145000*(states[0]+38.9000))-1.00000))
    rates[8] = (algebraic[5]-states[8])/algebraic[15]
    algebraic[6] = 1.00000/(1.00000+exp(-(states[0]-1.50000)/16.7000))
    algebraic[16] = 1.00000/((7.19000e-05*(states[0]+30.0000))/(1.00000-exp(-0.148000*(states[0]+30.0000)))+(0.000131000*(states[0]+30.0000))/(exp(0.0687000*(states[0]+30.0000))-1.00000))
    rates[9] = (algebraic[6]-states[9])/algebraic[16]
    rates[10] = ((algebraic[6]-states[10])/algebraic[16])/4.00000
    algebraic[7] = 1.00000/(1.00000+exp(-(states[0]+14.0000)/10.8000))
    algebraic[17] = 3.70000+6.10000/(1.00000+exp((states[0]+25.0000)/4.50000))
    rates[11] = (algebraic[7]-states[11])/algebraic[17]
    algebraic[1] = 1.00000-1.00000/(1.00000+exp(-(states[0]+40.0000)/0.0240000))
    algebraic[10] = algebraic[1]*0.135000*exp((80.0000+states[0])/-6.80000)
    algebraic[20] = (1.00000-algebraic[1])/(0.130000*(1.00000+exp((states[0]+10.6600)/-11.1000)))+algebraic[1]*(3.56000*exp(0.0790000*states[0])+3.10000*100000.*exp(0.350000*states[0]))
    rates[1] = algebraic[10]*(1.00000-states[1])-algebraic[20]*states[1]
    algebraic[11] = (algebraic[1]*(-127140.*exp(0.244400*states[0])-3.47400e-05*exp(-0.0439100*states[0]))*1.00000*(states[0]+37.7800))/(1.00000+exp(0.311000*(states[0]+79.2300)))
    algebraic[21] = ((1.00000-algebraic[1])*0.300000*exp(-2.53500e-07*states[0]))/(1.00000+exp(-0.100000*(states[0]+32.0000)))+(algebraic[1]*0.121200*exp(-0.0105200*states[0]))/(1.00000+exp(-0.137800*(states[0]+40.1400)))
    rates[3] = algebraic[11]*(1.00000-states[3])-algebraic[21]*states[3]
    algebraic[8] = 1.00000/(1.00000+exp((states[0]+60.0000)/5.60000))
    algebraic[18] = 1.00000-1.00000/(1.00000+exp(-states[0]/0.00240000))
    algebraic[23] = algebraic[18]*1.00000*(-0.875000*states[0]+12.0000)+12.0000*(1.00000-algebraic[18])
    rates[12] = (algebraic[8]-states[12])/algebraic[23]
    algebraic[3] = 1.00000/(1.00000+exp(-(states[0]+10.0000)/6.24000))
    algebraic[13] = (algebraic[3]*1.00000*(1.00000-exp(-(states[0]+10.0000)/6.24000)))/(0.0350000*(states[0]+10.0000))
    algebraic[22] = 1.00000/(1.00000+exp(-(states[0]+60.0000)/0.0240000))
    algebraic[25] = algebraic[3]*algebraic[22]
    rates[6] = (algebraic[25]-states[6])/algebraic[13]
    algebraic[40] = (constants[44]+constants[43])-states[13]
    algebraic[41] = states[13]*constants[43]
    algebraic[42] = -algebraic[40]/2.00000+(power(power(algebraic[40], 2.00000)+4.00000*algebraic[41], 1.0/2))/2.00000
    algebraic[45] = (states[15]-algebraic[42])/constants[48]
    rates[13] = algebraic[45]-states[14]
    algebraic[46] = ((constants[49]+constants[50])-states[16])+constants[52]+constants[51]
    algebraic[47] = (constants[51]*constants[52]-states[16]*(constants[52]+constants[51]))+constants[50]*constants[51]+constants[49]*constants[52]
    algebraic[48] = -constants[52]*constants[51]*states[16]
    algebraic[49] = ((2.00000*(power(algebraic[46]*algebraic[46]-3.00000*algebraic[47], 1.0/2)))/3.00000)*cos(arccos(((9.00000*algebraic[46]*algebraic[47]-2.00000*algebraic[46]*algebraic[46]*algebraic[46])-27.0000*algebraic[48])/(2.00000*(power(algebraic[46]*algebraic[46]-3.00000*algebraic[47], 1.50000))))/3.00000)-algebraic[46]/3.00000
    algebraic[50] = (constants[16]*4.00000*states[0]*constants[0]*constants[53]*(constants[13]*algebraic[49]*exp(2.00000*states[0]*constants[53])-constants[14]*constants[5]))/(exp(2.00000*states[0]*constants[53])-1.00000)
    algebraic[51] = 1.00000/(1.00000+algebraic[49]/constants[15])
    algebraic[52] = states[6]*states[7]*algebraic[51]*algebraic[50]
    algebraic[54] = (algebraic[52]*constants[66])/(1.00000+power(constants[39]/algebraic[42], constants[42]))
    algebraic[43] = constants[40]/(1.00000+0.0123000/algebraic[42])
    rates[14] = -(algebraic[54]+states[14])/algebraic[43]
    algebraic[27] = 1.00000/(1.00000+0.124500*exp(-0.100000*states[0]*constants[53])+0.0365000*constants[55]*exp(-states[0]*constants[53]))
    algebraic[28] = ((constants[28]*algebraic[27])/(1.00000+power(constants[26]/states[4], 2.00000)))/(1.00000+constants[27]/constants[3])
    algebraic[32] = log(constants[3]/states[5])/constants[53]
    algebraic[26] = 1.00000/(1.00000+exp((states[0]+9.00000)/22.4000))
    algebraic[37] = constants[24]*(power(constants[3]/5.40000, 1.0/2))*states[8]*algebraic[26]*(states[0]-algebraic[32])
    algebraic[39] = log((constants[3]+constants[38]*constants[4])/(states[5]+constants[38]*states[4]))/constants[53]
    algebraic[56] = constants[25]*(1.00000+0.600000/(1.00000+power(3.80000e-05/algebraic[49], 1.40000)))
    algebraic[57] = algebraic[56]*states[9]*states[10]*(states[0]-algebraic[39])
    algebraic[33] = 1.02000/(1.00000+exp(0.238500*((states[0]-algebraic[32])-59.2150)))
    algebraic[34] = (0.491240*exp(0.0803200*((states[0]-algebraic[32])+5.47600))+1.00000*exp(0.0617500*((states[0]-algebraic[32])-594.310)))/(1.00000+exp(-0.514300*((states[0]-algebraic[32])+4.75300)))
    algebraic[35] = (constants[54]*algebraic[33])/(algebraic[33]+algebraic[34])
    algebraic[36] = algebraic[35]*(states[0]-algebraic[32])
    algebraic[38] = (constants[32]*(states[0]-algebraic[32]))/(1.00000+exp((7.48800-states[0])/5.98000))
    algebraic[24] = (constants[20]*states[0]*constants[0]*constants[53]*(constants[21]*states[5]*exp(states[0]*constants[53])-constants[22]*constants[3]))/(exp(states[0]*constants[53])-1.00000)
    algebraic[55] = states[6]*states[7]*algebraic[51]*algebraic[24]
    algebraic[0] = floor(voi/constants[9])*constants[9]
    algebraic[9] = custom_piecewise([greater_equal(voi-algebraic[0] , constants[8]) & less_equal(voi-algebraic[0] , constants[8]+constants[10]), constants[11] , True, 0.00000])
    algebraic[58] = ((algebraic[37]+algebraic[57]+algebraic[36]+algebraic[38]+algebraic[55])-2.00000*algebraic[28])+algebraic[9]
    rates[5] = (-algebraic[58]*constants[65])/constants[62]
    algebraic[59] = (constants[29]*exp((constants[31]-1.00000)*states[0]*constants[53])*(exp(states[0]*constants[53])*(power(states[4], 3.00000))*constants[5]-(power(constants[4], 3.00000))*algebraic[49]))/(1.00000+constants[30]*exp((constants[31]-1.00000)*states[0]*constants[53])*(exp(states[0]*constants[53])*(power(states[4], 3.00000))*constants[5]+(power(constants[4], 3.00000))*algebraic[49]))
    algebraic[29] = log(constants[4]/states[4])/constants[53]
    algebraic[30] = constants[12]*states[2]*states[2]*states[2]*states[1]*states[3]*(states[0]-algebraic[29])
    algebraic[31] = constants[36]*(states[0]-algebraic[29])
    algebraic[19] = (constants[17]*states[0]*constants[0]*constants[53]*(constants[18]*states[4]*exp(states[0]*constants[53])-constants[19]*constants[4]))/(exp(states[0]*constants[53])-1.00000)
    algebraic[53] = states[6]*states[7]*algebraic[51]*algebraic[19]
    algebraic[60] = algebraic[30]+algebraic[31]+3.00000*algebraic[59]+algebraic[53]+3.00000*algebraic[28]
    rates[4] = (-algebraic[60]*constants[65])/constants[62]
    algebraic[62] = constants[35]*(states[0]-log(constants[5]/algebraic[49])/(2.00000*constants[53]))
    algebraic[61] = (constants[33]*algebraic[49])/(constants[34]+algebraic[49])
    algebraic[63] = (log(constants[5]/algebraic[49])/2.00000)/constants[53]
    algebraic[64] = constants[37]*states[11]*states[11]*states[12]*(states[0]-algebraic[63])
    algebraic[65] = ((algebraic[52]+algebraic[62]+algebraic[61])-2.00000*algebraic[59])+algebraic[64]
    rates[0] = -(algebraic[60]+algebraic[58]+algebraic[65])
    algebraic[66] = (constants[46]*algebraic[49])/(algebraic[49]+constants[45])
    algebraic[44] = (constants[46]*states[15])/constants[47]
    rates[16] = (-algebraic[65]*constants[65])/(constants[62]*2.00000)+((algebraic[44]-algebraic[66])*constants[63])/constants[62]+(states[14]*constants[64])/constants[62]
    rates[15] = (algebraic[66]-(algebraic[45]*constants[64])/constants[63])-algebraic[44]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[2] = (0.320000*1.00000*(states[0]+47.1300))/(1.00000-exp(-0.100000*(states[0]+47.1300)))
    algebraic[12] = 0.0800000*exp(-states[0]/11.0000)
    algebraic[4] = 1.00000/(1.00000+exp((states[0]+32.0000)/8.00000))+0.600000/(1.00000+exp((50.0000-states[0])/20.0000))
    algebraic[14] = 1.00000/(0.0197000*exp(-(power(0.0337000*(states[0]+10.0000), 2.00000)))+0.0200000)
    algebraic[5] = 1.00000/(1.00000+exp(-(states[0]+10.0850)/4.25000))
    algebraic[15] = 1.00000/((0.00138000*(states[0]+14.2000))/(1.00000-exp(-0.123000*(states[0]+14.2000)))+(0.000610000*(states[0]+38.9000))/(exp(0.145000*(states[0]+38.9000))-1.00000))
    algebraic[6] = 1.00000/(1.00000+exp(-(states[0]-1.50000)/16.7000))
    algebraic[16] = 1.00000/((7.19000e-05*(states[0]+30.0000))/(1.00000-exp(-0.148000*(states[0]+30.0000)))+(0.000131000*(states[0]+30.0000))/(exp(0.0687000*(states[0]+30.0000))-1.00000))
    algebraic[7] = 1.00000/(1.00000+exp(-(states[0]+14.0000)/10.8000))
    algebraic[17] = 3.70000+6.10000/(1.00000+exp((states[0]+25.0000)/4.50000))
    algebraic[1] = 1.00000-1.00000/(1.00000+exp(-(states[0]+40.0000)/0.0240000))
    algebraic[10] = algebraic[1]*0.135000*exp((80.0000+states[0])/-6.80000)
    algebraic[20] = (1.00000-algebraic[1])/(0.130000*(1.00000+exp((states[0]+10.6600)/-11.1000)))+algebraic[1]*(3.56000*exp(0.0790000*states[0])+3.10000*100000.*exp(0.350000*states[0]))
    algebraic[11] = (algebraic[1]*(-127140.*exp(0.244400*states[0])-3.47400e-05*exp(-0.0439100*states[0]))*1.00000*(states[0]+37.7800))/(1.00000+exp(0.311000*(states[0]+79.2300)))
    algebraic[21] = ((1.00000-algebraic[1])*0.300000*exp(-2.53500e-07*states[0]))/(1.00000+exp(-0.100000*(states[0]+32.0000)))+(algebraic[1]*0.121200*exp(-0.0105200*states[0]))/(1.00000+exp(-0.137800*(states[0]+40.1400)))
    algebraic[8] = 1.00000/(1.00000+exp((states[0]+60.0000)/5.60000))
    algebraic[18] = 1.00000-1.00000/(1.00000+exp(-states[0]/0.00240000))
    algebraic[23] = algebraic[18]*1.00000*(-0.875000*states[0]+12.0000)+12.0000*(1.00000-algebraic[18])
    algebraic[3] = 1.00000/(1.00000+exp(-(states[0]+10.0000)/6.24000))
    algebraic[13] = (algebraic[3]*1.00000*(1.00000-exp(-(states[0]+10.0000)/6.24000)))/(0.0350000*(states[0]+10.0000))
    algebraic[22] = 1.00000/(1.00000+exp(-(states[0]+60.0000)/0.0240000))
    algebraic[25] = algebraic[3]*algebraic[22]
    algebraic[40] = (constants[44]+constants[43])-states[13]
    algebraic[41] = states[13]*constants[43]
    algebraic[42] = -algebraic[40]/2.00000+(power(power(algebraic[40], 2.00000)+4.00000*algebraic[41], 1.0/2))/2.00000
    algebraic[45] = (states[15]-algebraic[42])/constants[48]
    algebraic[46] = ((constants[49]+constants[50])-states[16])+constants[52]+constants[51]
    algebraic[47] = (constants[51]*constants[52]-states[16]*(constants[52]+constants[51]))+constants[50]*constants[51]+constants[49]*constants[52]
    algebraic[48] = -constants[52]*constants[51]*states[16]
    algebraic[49] = ((2.00000*(power(algebraic[46]*algebraic[46]-3.00000*algebraic[47], 1.0/2)))/3.00000)*cos(arccos(((9.00000*algebraic[46]*algebraic[47]-2.00000*algebraic[46]*algebraic[46]*algebraic[46])-27.0000*algebraic[48])/(2.00000*(power(algebraic[46]*algebraic[46]-3.00000*algebraic[47], 1.50000))))/3.00000)-algebraic[46]/3.00000
    algebraic[50] = (constants[16]*4.00000*states[0]*constants[0]*constants[53]*(constants[13]*algebraic[49]*exp(2.00000*states[0]*constants[53])-constants[14]*constants[5]))/(exp(2.00000*states[0]*constants[53])-1.00000)
    algebraic[51] = 1.00000/(1.00000+algebraic[49]/constants[15])
    algebraic[52] = states[6]*states[7]*algebraic[51]*algebraic[50]
    algebraic[54] = (algebraic[52]*constants[66])/(1.00000+power(constants[39]/algebraic[42], constants[42]))
    algebraic[43] = constants[40]/(1.00000+0.0123000/algebraic[42])
    algebraic[27] = 1.00000/(1.00000+0.124500*exp(-0.100000*states[0]*constants[53])+0.0365000*constants[55]*exp(-states[0]*constants[53]))
    algebraic[28] = ((constants[28]*algebraic[27])/(1.00000+power(constants[26]/states[4], 2.00000)))/(1.00000+constants[27]/constants[3])
    algebraic[32] = log(constants[3]/states[5])/constants[53]
    algebraic[26] = 1.00000/(1.00000+exp((states[0]+9.00000)/22.4000))
    algebraic[37] = constants[24]*(power(constants[3]/5.40000, 1.0/2))*states[8]*algebraic[26]*(states[0]-algebraic[32])
    algebraic[39] = log((constants[3]+constants[38]*constants[4])/(states[5]+constants[38]*states[4]))/constants[53]
    algebraic[56] = constants[25]*(1.00000+0.600000/(1.00000+power(3.80000e-05/algebraic[49], 1.40000)))
    algebraic[57] = algebraic[56]*states[9]*states[10]*(states[0]-algebraic[39])
    algebraic[33] = 1.02000/(1.00000+exp(0.238500*((states[0]-algebraic[32])-59.2150)))
    algebraic[34] = (0.491240*exp(0.0803200*((states[0]-algebraic[32])+5.47600))+1.00000*exp(0.0617500*((states[0]-algebraic[32])-594.310)))/(1.00000+exp(-0.514300*((states[0]-algebraic[32])+4.75300)))
    algebraic[35] = (constants[54]*algebraic[33])/(algebraic[33]+algebraic[34])
    algebraic[36] = algebraic[35]*(states[0]-algebraic[32])
    algebraic[38] = (constants[32]*(states[0]-algebraic[32]))/(1.00000+exp((7.48800-states[0])/5.98000))
    algebraic[24] = (constants[20]*states[0]*constants[0]*constants[53]*(constants[21]*states[5]*exp(states[0]*constants[53])-constants[22]*constants[3]))/(exp(states[0]*constants[53])-1.00000)
    algebraic[55] = states[6]*states[7]*algebraic[51]*algebraic[24]
    algebraic[0] = floor(voi/constants[9])*constants[9]
    algebraic[9] = custom_piecewise([greater_equal(voi-algebraic[0] , constants[8]) & less_equal(voi-algebraic[0] , constants[8]+constants[10]), constants[11] , True, 0.00000])
    algebraic[58] = ((algebraic[37]+algebraic[57]+algebraic[36]+algebraic[38]+algebraic[55])-2.00000*algebraic[28])+algebraic[9]
    algebraic[59] = (constants[29]*exp((constants[31]-1.00000)*states[0]*constants[53])*(exp(states[0]*constants[53])*(power(states[4], 3.00000))*constants[5]-(power(constants[4], 3.00000))*algebraic[49]))/(1.00000+constants[30]*exp((constants[31]-1.00000)*states[0]*constants[53])*(exp(states[0]*constants[53])*(power(states[4], 3.00000))*constants[5]+(power(constants[4], 3.00000))*algebraic[49]))
    algebraic[29] = log(constants[4]/states[4])/constants[53]
    algebraic[30] = constants[12]*states[2]*states[2]*states[2]*states[1]*states[3]*(states[0]-algebraic[29])
    algebraic[31] = constants[36]*(states[0]-algebraic[29])
    algebraic[19] = (constants[17]*states[0]*constants[0]*constants[53]*(constants[18]*states[4]*exp(states[0]*constants[53])-constants[19]*constants[4]))/(exp(states[0]*constants[53])-1.00000)
    algebraic[53] = states[6]*states[7]*algebraic[51]*algebraic[19]
    algebraic[60] = algebraic[30]+algebraic[31]+3.00000*algebraic[59]+algebraic[53]+3.00000*algebraic[28]
    algebraic[62] = constants[35]*(states[0]-log(constants[5]/algebraic[49])/(2.00000*constants[53]))
    algebraic[61] = (constants[33]*algebraic[49])/(constants[34]+algebraic[49])
    algebraic[63] = (log(constants[5]/algebraic[49])/2.00000)/constants[53]
    algebraic[64] = constants[37]*states[11]*states[11]*states[12]*(states[0]-algebraic[63])
    algebraic[65] = ((algebraic[52]+algebraic[62]+algebraic[61])-2.00000*algebraic[59])+algebraic[64]
    algebraic[66] = (constants[46]*algebraic[49])/(algebraic[49]+constants[45])
    algebraic[44] = (constants[46]*states[15])/constants[47]
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