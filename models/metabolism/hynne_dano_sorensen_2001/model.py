# Size of variable arrays:
sizeAlgebraic = 24
sizeStates = 22
sizeConstants = 60
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_constants[0] = "Glc_x_0 in component Glc_x_0 (millimolar)"
    legend_states[0] = "Glc_x in component Glc_x (millimolar)"
    legend_algebraic[1] = "GlcTrans in component GlcTrans (flux)"
    legend_algebraic[0] = "inGlc in component inGlc (flux)"
    legend_states[1] = "Glc in component Glc (millimolar)"
    legend_algebraic[2] = "HK in component HK (flux)"
    legend_states[2] = "G6P in component G6P (millimolar)"
    legend_algebraic[3] = "PGI in component PGI (flux)"
    legend_algebraic[10] = "storage in component storage (flux)"
    legend_states[3] = "F6P in component F6P (millimolar)"
    legend_algebraic[4] = "PFK in component PFK (flux)"
    legend_states[4] = "FBP in component FBP (millimolar)"
    legend_algebraic[5] = "ALD in component ALD (flux)"
    legend_states[5] = "GAP in component GAP (millimolar)"
    legend_algebraic[7] = "GAPDH in component GAPDH (flux)"
    legend_algebraic[6] = "TIM in component TIM (flux)"
    legend_states[6] = "DHAP in component DHAP (millimolar)"
    legend_algebraic[14] = "lpGlyc in component lpGlyc (flux)"
    legend_states[7] = "BPG in component BPG (millimolar)"
    legend_algebraic[8] = "lpPEP in component lpPEP (flux)"
    legend_states[8] = "PEP in component PEP (millimolar)"
    legend_algebraic[9] = "PK in component PK (flux)"
    legend_states[9] = "Pyr in component Pyr (millimolar)"
    legend_algebraic[11] = "PDC in component PDC (flux)"
    legend_states[10] = "ATP in component ATP (millimolar)"
    legend_algebraic[12] = "consum in component consum (flux)"
    legend_algebraic[15] = "AK in component AK (flux)"
    legend_states[11] = "ADP in component ADP (millimolar)"
    legend_states[12] = "AMP in component AMP (millimolar)"
    legend_constants[1] = "CN_x_0 in component CN_x_0 (millimolar)"
    legend_states[13] = "CN_x in component CN_x (millimolar)"
    legend_algebraic[21] = "lacto in component lacto (flux)"
    legend_algebraic[23] = "inCN in component inCN (flux)"
    legend_states[14] = "ACA in component ACA (millimolar)"
    legend_algebraic[18] = "difACA in component difACA (flux)"
    legend_algebraic[13] = "ADH in component ADH (flux)"
    legend_states[15] = "ACA_x in component ACA_x (millimolar)"
    legend_algebraic[16] = "outACA in component outACA (flux)"
    legend_states[16] = "EtOH in component EtOH (millimolar)"
    legend_algebraic[19] = "difEtOH in component difEtOH (flux)"
    legend_states[17] = "EtOH_x in component EtOH_x (millimolar)"
    legend_algebraic[17] = "outEtOH in component outEtOH (flux)"
    legend_states[18] = "Glyc in component Glyc (millimolar)"
    legend_algebraic[22] = "difGlyc in component difGlyc (flux)"
    legend_states[19] = "Glyc_x in component Glyc_x (millimolar)"
    legend_algebraic[20] = "outGlyc in component outGlyc (flux)"
    legend_states[20] = "NADH in component NADH (millimolar)"
    legend_states[21] = "NAD in component NAD (millimolar)"
    legend_constants[2] = "k0 in component model_parameters (first_order_rate_constant)"
    legend_constants[3] = "K2_Glc in component GlcTrans (millimolar)"
    legend_constants[4] = "K2_IG6P in component GlcTrans (millimolar)"
    legend_constants[5] = "K2_IIG6P in component GlcTrans (millimolar)"
    legend_constants[6] = "V_2m in component GlcTrans (flux)"
    legend_constants[7] = "P2 in component GlcTrans (dimensionless)"
    legend_constants[8] = "y_vol in component model_parameters (dimensionless)"
    legend_constants[9] = "K3_DGlc in component HK (millimolar)"
    legend_constants[10] = "K3_Glc in component HK (millimolar)"
    legend_constants[11] = "K3_ATP in component HK (millimolar)"
    legend_constants[12] = "V_3m in component HK (flux)"
    legend_constants[13] = "V_4m in component PGI (flux)"
    legend_constants[14] = "K4_G6P in component PGI (millimolar)"
    legend_constants[15] = "K4_F6P in component PGI (millimolar)"
    legend_constants[16] = "K4_eq in component PGI (dimensionless)"
    legend_constants[17] = "K5 in component PFK (millimolar2)"
    legend_constants[18] = "kappa5 in component PFK (dimensionless)"
    legend_constants[19] = "V_5m in component PFK (flux)"
    legend_constants[20] = "K6_eq in component ALD (millimolar)"
    legend_constants[21] = "K6_FBP in component ALD (millimolar)"
    legend_constants[22] = "K6_DHAP in component ALD (millimolar)"
    legend_constants[23] = "K6_GAP in component ALD (millimolar)"
    legend_constants[24] = "K6_IGAP in component ALD (millimolar)"
    legend_constants[25] = "V_6r in component ALD (flux)"
    legend_constants[26] = "V_6f in component ALD (flux)"
    legend_constants[27] = "K7_eq in component TIM (dimensionless)"
    legend_constants[28] = "K7_DHAP in component TIM (millimolar)"
    legend_constants[29] = "K7_GAP in component TIM (millimolar)"
    legend_constants[30] = "V_7m in component TIM (flux)"
    legend_constants[31] = "K8_NAD in component GAPDH (millimolar)"
    legend_constants[32] = "K8_NADH in component GAPDH (millimolar)"
    legend_constants[33] = "K8_GAP in component GAPDH (millimolar)"
    legend_constants[34] = "K8_BPG in component GAPDH (millimolar)"
    legend_constants[35] = "K8_eq in component GAPDH (dimensionless)"
    legend_constants[36] = "V_8m in component GAPDH (flux)"
    legend_constants[37] = "k9f in component lpPEP (second_order_rate_constant)"
    legend_constants[38] = "k9r in component lpPEP (second_order_rate_constant)"
    legend_constants[39] = "K10_PEP in component PK (millimolar)"
    legend_constants[40] = "K10_ADP in component PK (millimolar)"
    legend_constants[41] = "V_10m in component PK (flux)"
    legend_constants[42] = "K11 in component PDC (millimolar)"
    legend_constants[43] = "V_11m in component PDC (flux)"
    legend_constants[44] = "K12_NADH in component ADH (millimolar)"
    legend_constants[45] = "K12_ACA in component ADH (millimolar)"
    legend_constants[46] = "V_12m in component ADH (flux)"
    legend_constants[47] = "k13 in component difEtOH (first_order_rate_constant)"
    legend_constants[48] = "K15_NADH in component lpGlyc (millimolar)"
    legend_constants[49] = "K15_INADH in component lpGlyc (millimolar)"
    legend_constants[50] = "K15_INAD in component lpGlyc (millimolar)"
    legend_constants[51] = "K15_DHAP in component lpGlyc (millimolar)"
    legend_constants[52] = "V_15m in component lpGlyc (flux)"
    legend_constants[53] = "k16 in component difGlyc (first_order_rate_constant)"
    legend_constants[54] = "k18 in component difACA (first_order_rate_constant)"
    legend_constants[55] = "k20 in component lacto (second_order_rate_constant)"
    legend_constants[56] = "k22 in component storage (second_order_rate_constant)"
    legend_constants[57] = "k23 in component consum (first_order_rate_constant)"
    legend_constants[58] = "k24f in component AK (second_order_rate_constant)"
    legend_constants[59] = "k24r in component AK (second_order_rate_constant)"
    legend_rates[0] = "d/dt Glc_x in component Glc_x (millimolar)"
    legend_rates[1] = "d/dt Glc in component Glc (millimolar)"
    legend_rates[2] = "d/dt G6P in component G6P (millimolar)"
    legend_rates[3] = "d/dt F6P in component F6P (millimolar)"
    legend_rates[4] = "d/dt FBP in component FBP (millimolar)"
    legend_rates[5] = "d/dt GAP in component GAP (millimolar)"
    legend_rates[6] = "d/dt DHAP in component DHAP (millimolar)"
    legend_rates[7] = "d/dt BPG in component BPG (millimolar)"
    legend_rates[8] = "d/dt PEP in component PEP (millimolar)"
    legend_rates[9] = "d/dt Pyr in component Pyr (millimolar)"
    legend_rates[10] = "d/dt ATP in component ATP (millimolar)"
    legend_rates[11] = "d/dt ADP in component ADP (millimolar)"
    legend_rates[12] = "d/dt AMP in component AMP (millimolar)"
    legend_rates[13] = "d/dt CN_x in component CN_x (millimolar)"
    legend_rates[14] = "d/dt ACA in component ACA (millimolar)"
    legend_rates[15] = "d/dt ACA_x in component ACA_x (millimolar)"
    legend_rates[16] = "d/dt EtOH in component EtOH (millimolar)"
    legend_rates[17] = "d/dt EtOH_x in component EtOH_x (millimolar)"
    legend_rates[18] = "d/dt Glyc in component Glyc (millimolar)"
    legend_rates[19] = "d/dt Glyc_x in component Glyc_x (millimolar)"
    legend_rates[20] = "d/dt NADH in component NADH (millimolar)"
    legend_rates[21] = "d/dt NAD in component NAD (millimolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 24.0
    states[0] = 6.7
    states[1] = 0.573074
    states[2] = 4.2
    states[3] = 0.49
    states[4] = 4.64
    states[5] = 0.115
    states[6] = 2.95
    states[7] = 0.00027
    states[8] = 0.04
    states[9] = 8.7
    states[10] = 2.1
    states[11] = 1.5
    states[12] = 0.33
    constants[1] = 5.60
    states[13] = 5.20358
    states[14] = 1.48153
    states[15] = 1.28836
    states[16] = 19.2379
    states[17] = 16.4514
    states[18] = 4.196
    states[19] = 1.68478
    states[20] = 0.33
    states[21] = 0.65
    constants[2] = 0.048
    constants[3] = 1.7
    constants[4] = 1.2
    constants[5] = 7.2
    constants[6] = 1014.96
    constants[7] = 1.0
    constants[8] = 59.0
    constants[9] = 0.37
    constants[10] = 0.0
    constants[11] = 0.1
    constants[12] = 51.7547
    constants[13] = 496.042
    constants[14] = 0.8
    constants[15] = 0.15
    constants[16] = 0.13
    constants[17] = 0.021
    constants[18] = 0.15
    constants[19] = 45.4327
    constants[20] = 0.081
    constants[21] = 0.3
    constants[22] = 2.0
    constants[23] = 4.0
    constants[24] = 10.0
    constants[25] = 1.10391E4
    constants[26] = 2.20782E3
    constants[27] = 0.055
    constants[28] = 1.23
    constants[29] = 1.27
    constants[30] = 1.16365E2
    constants[31] = 0.1
    constants[32] = 0.06
    constants[33] = 0.6
    constants[34] = 0.01
    constants[35] = 0.0055
    constants[36] = 8.33858E2
    constants[37] = 4.43866E5
    constants[38] = 1.52862E3
    constants[39] = 0.2
    constants[40] = 0.17
    constants[41] = 3.43096E2
    constants[42] = 0.3
    constants[43] = 5.31328E1
    constants[44] = 0.1
    constants[45] = 0.71
    constants[46] = 8.98023E1
    constants[47] = 16.72
    constants[48] = 0.13
    constants[49] = 0.034
    constants[50] = 0.13
    constants[51] = 25.0
    constants[52] = 8.14797E1
    constants[53] = 1.9
    constants[54] = 24.7
    constants[55] = 2.83828E-3
    constants[56] = 2.25932
    constants[57] = 3.20760
    constants[58] = 4.32900E2
    constants[59] = 1.33333E2
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[1] = (constants[6]/constants[8])*((states[0]/constants[3])/(1.00000+states[0]/constants[3]+((constants[7]*(states[0]/constants[3])+1.00000)/(constants[7]*(states[1]/constants[3])+1.00000))*(1.00000+states[1]/constants[3]+states[2]/constants[4]+(states[1]*states[2])/(constants[3]*constants[5]))))-(constants[6]/constants[8])*((states[1]/constants[3])/(1.00000+states[1]/constants[3]+((constants[7]*(states[1]/constants[3])+1.00000)/(constants[7]*(states[0]/constants[3])+1.00000))*(1.00000+states[0]/constants[3])+states[2]/constants[4]+(states[1]*states[2])/(constants[3]*constants[5])))
    algebraic[0] = constants[2]*(constants[0]-states[0])
    rates[0] = algebraic[0]-algebraic[1]
    algebraic[2] = (constants[12]*states[10]*states[1])/(constants[9]*constants[11]+constants[10]*states[10]+constants[11]*states[1]+states[1]*states[10])
    rates[1] = 59.0000*algebraic[1]-algebraic[2]
    algebraic[3] = (constants[13]*states[2])/(constants[14]+states[2]+(constants[14]/constants[15])*states[3])-(constants[13]*(states[3]/constants[16]))/(constants[14]+states[2]+(constants[14]/constants[15])*states[3])
    algebraic[4] = (constants[19]*(power(states[3], 2.00000)))/(constants[17]*(1.00000+constants[18]*(states[10]/states[12])*(states[10]/states[12]))+power(states[3], 2.00000))
    rates[3] = algebraic[3]-algebraic[4]
    algebraic[5] = (constants[26]*states[4])/(constants[21]+states[4]+(states[5]*constants[22]*constants[26])/(constants[20]*constants[25])+(states[6]*constants[23]*constants[26])/(constants[20]*constants[25])+(states[4]*states[5])/constants[24]+(states[5]*states[6]*constants[26])/(constants[20]*constants[25]))-(constants[26]*((states[5]*states[6])/constants[20]))/(constants[21]+states[4]+(states[5]*constants[22]*constants[26])/(constants[20]*constants[25])+(states[6]*constants[23]*constants[26])/(constants[20]*constants[25])+(states[4]*states[5])/constants[24]+(states[5]*states[6]*constants[26])/(constants[20]*constants[25]))
    rates[4] = algebraic[4]-algebraic[5]
    algebraic[7] = (constants[36]*states[5]*states[21])/(constants[33]*constants[31]*(1.00000+states[5]/constants[33]+states[7]/constants[34])*(1.00000+states[21]/constants[31]+states[20]/constants[32]))-(constants[36]*((states[7]*states[20])/constants[35]))/(constants[33]*constants[31]*(1.00000+states[5]/constants[33]+states[7]/constants[34])*(1.00000+states[21]/constants[31]+states[20]/constants[32]))
    algebraic[6] = (constants[30]*states[6])/(constants[28]+states[6]+(constants[28]/constants[29])*states[5])-(constants[30]*(states[5]/constants[27]))/(constants[28]+states[6]+(constants[28]/constants[29])*states[5])
    rates[5] = (algebraic[5]+algebraic[6])-algebraic[7]
    algebraic[8] = constants[37]*states[7]*states[11]-constants[38]*states[8]*states[10]
    rates[7] = algebraic[7]-algebraic[8]
    algebraic[9] = (constants[41]*states[11]*states[8])/((constants[39]+states[8])*(constants[40]+states[11]))
    rates[8] = algebraic[8]-algebraic[9]
    algebraic[10] = constants[56]*states[10]*states[2]
    rates[2] = algebraic[2]-(algebraic[3]+algebraic[10])
    algebraic[11] = (constants[43]*states[9])/(constants[42]+states[9])
    rates[9] = algebraic[9]-algebraic[11]
    algebraic[14] = (constants[52]*states[6])/(constants[51]*(1.00000+(constants[49]/states[20])*(1.00000+states[21]/constants[50]))+states[6]*(1.00000+(constants[48]/states[20])*(1.00000+states[21]/constants[50])))
    rates[6] = algebraic[5]-(algebraic[6]+algebraic[14])
    algebraic[12] = constants[57]*states[10]
    algebraic[15] = constants[58]*states[12]*states[10]-constants[59]*(power(states[11], 2.00000))
    rates[10] = (algebraic[9]+algebraic[8])-(algebraic[4]+algebraic[10]+algebraic[2]+algebraic[12]+algebraic[15])
    rates[11] = (algebraic[4]+algebraic[10]+algebraic[2]+algebraic[12]+2.00000*algebraic[15])-(algebraic[9]+algebraic[8])
    rates[12] = -algebraic[15]
    algebraic[13] = (constants[46]*states[14]*states[20])/((constants[44]+states[20])*(constants[45]+states[14]))
    rates[20] = algebraic[7]-(algebraic[13]+algebraic[14])
    rates[21] = (algebraic[13]+algebraic[14])-algebraic[7]
    algebraic[18] = (constants[54]/constants[8])*(states[14]-states[15])
    rates[14] = algebraic[11]-(59.0000*algebraic[18]+algebraic[13])
    algebraic[19] = (constants[47]/constants[8])*(states[16]-states[17])
    rates[16] = algebraic[13]-59.0000*algebraic[19]
    algebraic[17] = constants[2]*states[17]
    rates[17] = algebraic[19]-algebraic[17]
    algebraic[21] = constants[55]*states[15]*states[13]
    algebraic[16] = constants[2]*states[15]
    rates[15] = algebraic[18]-(algebraic[21]+algebraic[16])
    algebraic[22] = (constants[53]/constants[8])*(states[18]-states[19])
    rates[18] = algebraic[14]-59.0000*algebraic[22]
    algebraic[20] = constants[2]*states[19]
    rates[19] = algebraic[22]-algebraic[20]
    algebraic[23] = constants[2]*(constants[1]-states[13])
    rates[13] = algebraic[23]-algebraic[21]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[1] = (constants[6]/constants[8])*((states[0]/constants[3])/(1.00000+states[0]/constants[3]+((constants[7]*(states[0]/constants[3])+1.00000)/(constants[7]*(states[1]/constants[3])+1.00000))*(1.00000+states[1]/constants[3]+states[2]/constants[4]+(states[1]*states[2])/(constants[3]*constants[5]))))-(constants[6]/constants[8])*((states[1]/constants[3])/(1.00000+states[1]/constants[3]+((constants[7]*(states[1]/constants[3])+1.00000)/(constants[7]*(states[0]/constants[3])+1.00000))*(1.00000+states[0]/constants[3])+states[2]/constants[4]+(states[1]*states[2])/(constants[3]*constants[5])))
    algebraic[0] = constants[2]*(constants[0]-states[0])
    algebraic[2] = (constants[12]*states[10]*states[1])/(constants[9]*constants[11]+constants[10]*states[10]+constants[11]*states[1]+states[1]*states[10])
    algebraic[3] = (constants[13]*states[2])/(constants[14]+states[2]+(constants[14]/constants[15])*states[3])-(constants[13]*(states[3]/constants[16]))/(constants[14]+states[2]+(constants[14]/constants[15])*states[3])
    algebraic[4] = (constants[19]*(power(states[3], 2.00000)))/(constants[17]*(1.00000+constants[18]*(states[10]/states[12])*(states[10]/states[12]))+power(states[3], 2.00000))
    algebraic[5] = (constants[26]*states[4])/(constants[21]+states[4]+(states[5]*constants[22]*constants[26])/(constants[20]*constants[25])+(states[6]*constants[23]*constants[26])/(constants[20]*constants[25])+(states[4]*states[5])/constants[24]+(states[5]*states[6]*constants[26])/(constants[20]*constants[25]))-(constants[26]*((states[5]*states[6])/constants[20]))/(constants[21]+states[4]+(states[5]*constants[22]*constants[26])/(constants[20]*constants[25])+(states[6]*constants[23]*constants[26])/(constants[20]*constants[25])+(states[4]*states[5])/constants[24]+(states[5]*states[6]*constants[26])/(constants[20]*constants[25]))
    algebraic[7] = (constants[36]*states[5]*states[21])/(constants[33]*constants[31]*(1.00000+states[5]/constants[33]+states[7]/constants[34])*(1.00000+states[21]/constants[31]+states[20]/constants[32]))-(constants[36]*((states[7]*states[20])/constants[35]))/(constants[33]*constants[31]*(1.00000+states[5]/constants[33]+states[7]/constants[34])*(1.00000+states[21]/constants[31]+states[20]/constants[32]))
    algebraic[6] = (constants[30]*states[6])/(constants[28]+states[6]+(constants[28]/constants[29])*states[5])-(constants[30]*(states[5]/constants[27]))/(constants[28]+states[6]+(constants[28]/constants[29])*states[5])
    algebraic[8] = constants[37]*states[7]*states[11]-constants[38]*states[8]*states[10]
    algebraic[9] = (constants[41]*states[11]*states[8])/((constants[39]+states[8])*(constants[40]+states[11]))
    algebraic[10] = constants[56]*states[10]*states[2]
    algebraic[11] = (constants[43]*states[9])/(constants[42]+states[9])
    algebraic[14] = (constants[52]*states[6])/(constants[51]*(1.00000+(constants[49]/states[20])*(1.00000+states[21]/constants[50]))+states[6]*(1.00000+(constants[48]/states[20])*(1.00000+states[21]/constants[50])))
    algebraic[12] = constants[57]*states[10]
    algebraic[15] = constants[58]*states[12]*states[10]-constants[59]*(power(states[11], 2.00000))
    algebraic[13] = (constants[46]*states[14]*states[20])/((constants[44]+states[20])*(constants[45]+states[14]))
    algebraic[18] = (constants[54]/constants[8])*(states[14]-states[15])
    algebraic[19] = (constants[47]/constants[8])*(states[16]-states[17])
    algebraic[17] = constants[2]*states[17]
    algebraic[21] = constants[55]*states[15]*states[13]
    algebraic[16] = constants[2]*states[15]
    algebraic[22] = (constants[53]/constants[8])*(states[18]-states[19])
    algebraic[20] = constants[2]*states[19]
    algebraic[23] = constants[2]*(constants[1]-states[13])
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