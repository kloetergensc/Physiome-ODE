# Size of variable arrays:
sizeAlgebraic = 6
sizeStates = 12
sizeConstants = 43
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_states[0] = "Asp in component Asp (millimolar)"
    legend_algebraic[0] = "vAKI in component vAKI (flux)"
    legend_algebraic[4] = "vAKIII in component vAKIII (flux)"
    legend_states[1] = "aspp in component aspp (millimolar)"
    legend_algebraic[1] = "vASD in component vASD (flux)"
    legend_states[2] = "ASA in component ASA (millimolar)"
    legend_algebraic[2] = "vHDH in component vHDH (flux)"
    legend_states[3] = "hs in component hs (millimolar)"
    legend_algebraic[3] = "vHK in component vHK (flux)"
    legend_states[4] = "hsp in component hsp (millimolar)"
    legend_algebraic[5] = "vTS in component vTS (flux)"
    legend_states[5] = "Thr in component Thr (millimolar)"
    legend_states[6] = "ATP in component ATP (millimolar)"
    legend_states[7] = "ADP in component ADP (millimolar)"
    legend_states[8] = "NADPH in component NADPH (millimolar)"
    legend_states[9] = "NADP in component NADP (millimolar)"
    legend_states[10] = "Pi in component Pi (millimolar)"
    legend_states[11] = "Lys in component Lys (millimolar)"
    legend_constants[0] = "K_asp in component vAKI (millimolar)"
    legend_constants[1] = "K_ATP in component vAKI (millimolar)"
    legend_constants[2] = "K_aspp in component vAKI (millimolar)"
    legend_constants[3] = "K_ADP in component vAKI (millimolar)"
    legend_constants[4] = "K_iThr in component vAKI (millimolar)"
    legend_constants[5] = "alpha in component vAKI (dimensionless)"
    legend_constants[6] = "h_Thr in component vAKI (dimensionless)"
    legend_constants[7] = "K_eq in component vAKI (dimensionless)"
    legend_constants[8] = "V_max_AK_I in component vAKI (flux)"
    legend_constants[9] = "K_aspp in component vASD (millimolar)"
    legend_constants[10] = "K_NADPH in component vASD (millimolar)"
    legend_constants[11] = "K_ASA in component vASD (millimolar)"
    legend_constants[12] = "K_NADP in component vASD (millimolar)"
    legend_constants[13] = "K_Pi in component vASD (millimolar)"
    legend_constants[14] = "K_eq in component vASD (millimolar)"
    legend_constants[15] = "V_max_ASD in component vASD (flux)"
    legend_constants[16] = "K_ASA in component vHDH (millimolar)"
    legend_constants[17] = "K_NADPH in component vHDH (millimolar)"
    legend_constants[18] = "K_hs in component vHDH (millimolar)"
    legend_constants[19] = "K_NADP in component vHDH (millimolar)"
    legend_constants[20] = "K_iThr in component vHDH (millimolar)"
    legend_constants[21] = "alpha in component vHDH (dimensionless)"
    legend_constants[22] = "h in component vHDH (dimensionless)"
    legend_constants[23] = "K_eq in component vHDH (dimensionless)"
    legend_constants[24] = "V_max_HDH in component vHDH (flux)"
    legend_constants[25] = "K_hs in component vHK (millimolar)"
    legend_constants[26] = "K_ATP in component vHK (millimolar)"
    legend_constants[27] = "K_iThr in component vHK (millimolar)"
    legend_constants[28] = "K_iLys in component vHK (millimolar)"
    legend_constants[29] = "K_ihs in component vHK (millimolar)"
    legend_constants[30] = "K_iATP in component vHK (millimolar)"
    legend_constants[31] = "V_max_HK in component vHK (flux)"
    legend_constants[32] = "K_hsp in component vTS (millimolar)"
    legend_constants[33] = "V_max_TS in component vTS (flux)"
    legend_constants[34] = "K_asp in component vAKIII (millimolar)"
    legend_constants[35] = "K_ATP in component vAKIII (millimolar)"
    legend_constants[36] = "K_aspp in component vAKIII (millimolar)"
    legend_constants[37] = "K_ADP in component vAKIII (millimolar)"
    legend_constants[38] = "K_iLys in component vAKIII (millimolar)"
    legend_constants[39] = "h_Lys in component vAKIII (dimensionless)"
    legend_constants[40] = "K_eq in component vAKIII (dimensionless)"
    legend_constants[41] = "V_max_AK_III in component vAKIII (flux)"
    legend_rates[0] = "d/dt Asp in component Asp (millimolar)"
    legend_rates[1] = "d/dt aspp in component aspp (millimolar)"
    legend_rates[2] = "d/dt ASA in component ASA (millimolar)"
    legend_rates[3] = "d/dt hs in component hs (millimolar)"
    legend_rates[4] = "d/dt hsp in component hsp (millimolar)"
    legend_rates[5] = "d/dt Thr in component Thr (millimolar)"
    legend_rates[6] = "d/dt ATP in component ATP (millimolar)"
    legend_rates[7] = "d/dt ADP in component ADP (millimolar)"
    legend_rates[8] = "d/dt NADPH in component NADPH (millimolar)"
    legend_rates[9] = "d/dt NADP in component NADP (millimolar)"
    legend_rates[10] = "d/dt Pi in component Pi (millimolar)"
    legend_rates[11] = "d/dt Lys in component Lys (millimolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 10
    states[1] = 0
    states[2] = 0
    states[3] = 0
    states[4] = 0
    states[5] = 0
    states[6] = 30
    states[7] = 0
    states[8] = 0
    states[9] = 0
    states[10] = 0
    states[11] = 0
    constants[0] = 0.97
    constants[1] = 0.98
    constants[2] = 0.017
    constants[3] = 0.25
    constants[4] = 0.167
    constants[5] = 2.47
    constants[6] = 4.09
    constants[7] = 0.00064
    constants[8] = 463
    constants[9] = 0.022
    constants[10] = 0.029
    constants[11] = 0.11
    constants[12] = 0.144
    constants[13] = 10.2
    constants[14] = 284000
    constants[15] = 598
    constants[16] = 0.24
    constants[17] = 0.037
    constants[18] = 3.39
    constants[19] = 0.067
    constants[20] = 0.097
    constants[21] = 3.93
    constants[22] = 1.41
    constants[23] = 100000000000
    constants[24] = 2585
    constants[25] = 0.11
    constants[26] = 0.072
    constants[27] = 1.09
    constants[28] = 9.45
    constants[29] = 4.7
    constants[30] = 4.35
    constants[31] = 483
    constants[32] = 0.31
    constants[33] = 208
    constants[34] = 0.32
    constants[35] = 0.22
    constants[36] = 0.017
    constants[37] = 0.25
    constants[38] = 0.391
    constants[39] = 2.8
    constants[40] = 0.00064
    constants[41] = 299
    constants[42] = 0.00000
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[11] = constants[42]
    algebraic[1] = (constants[15]*(states[1]*states[8]-(states[2]*states[9]*states[10])/constants[14]))/((constants[9]*(1.00000+states[2]/constants[11])*(1.00000+states[10]/constants[13])+states[1])*(constants[10]*(1.00000+states[9]/constants[12])+states[8]))
    algebraic[2] = (constants[24]*(states[2]*states[8]-(states[3]*states[9])/constants[23]))/(((1.00000+power(states[5]/constants[20], constants[22]))/(1.00000+power(states[5]/(constants[21]*constants[20]), constants[22])))*(constants[16]+(states[3]*constants[16])/constants[18]+states[2])*(constants[17]*(1.00000+states[9]/constants[19])+states[8]))
    rates[2] = algebraic[1]-algebraic[2]
    rates[8] = -algebraic[1]-algebraic[2]
    rates[9] = algebraic[1]+algebraic[2]
    algebraic[3] = (constants[31]*states[3]*states[6])/(((constants[25]*(1.00000+states[6]/constants[30]))/(1.00000+states[5]/constants[27])+states[3])*(constants[26]*(1.00000+states[3]/constants[29])+states[6])*(1.00000+states[11]/constants[28]))
    rates[3] = algebraic[2]-algebraic[3]
    algebraic[0] = (constants[8]*(states[0]*states[6]-(states[1]*states[7])/constants[7]))/(((constants[0]*(1.00000+power(states[5]/constants[4], constants[6])))/(1.00000+power(states[5]/(constants[4]*constants[5]), constants[6]))+(states[1]*constants[0])/constants[2]+states[0])*(constants[1]*(1.00000+states[7]/constants[3])+states[6]))
    algebraic[4] = (constants[41]*(states[0]*states[6]-(states[1]*states[7])/constants[40]))/((1.00000+power(states[11]/constants[38], constants[39]))*(constants[34]*(1.00000+states[1]/constants[36])+states[0])*(constants[35]*(1.00000+states[7]/constants[37])+states[6]))
    rates[0] = -algebraic[0]-algebraic[4]
    rates[1] = (algebraic[0]+algebraic[4])-algebraic[1]
    algebraic[5] = (constants[33]*states[4])/(constants[32]+states[4])
    rates[4] = algebraic[3]+algebraic[5]
    rates[5] = algebraic[5]
    rates[6] = (-algebraic[0]-algebraic[4])-algebraic[3]
    rates[7] = algebraic[0]+algebraic[4]+algebraic[3]
    rates[10] = algebraic[1]+algebraic[5]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[1] = (constants[15]*(states[1]*states[8]-(states[2]*states[9]*states[10])/constants[14]))/((constants[9]*(1.00000+states[2]/constants[11])*(1.00000+states[10]/constants[13])+states[1])*(constants[10]*(1.00000+states[9]/constants[12])+states[8]))
    algebraic[2] = (constants[24]*(states[2]*states[8]-(states[3]*states[9])/constants[23]))/(((1.00000+power(states[5]/constants[20], constants[22]))/(1.00000+power(states[5]/(constants[21]*constants[20]), constants[22])))*(constants[16]+(states[3]*constants[16])/constants[18]+states[2])*(constants[17]*(1.00000+states[9]/constants[19])+states[8]))
    algebraic[3] = (constants[31]*states[3]*states[6])/(((constants[25]*(1.00000+states[6]/constants[30]))/(1.00000+states[5]/constants[27])+states[3])*(constants[26]*(1.00000+states[3]/constants[29])+states[6])*(1.00000+states[11]/constants[28]))
    algebraic[0] = (constants[8]*(states[0]*states[6]-(states[1]*states[7])/constants[7]))/(((constants[0]*(1.00000+power(states[5]/constants[4], constants[6])))/(1.00000+power(states[5]/(constants[4]*constants[5]), constants[6]))+(states[1]*constants[0])/constants[2]+states[0])*(constants[1]*(1.00000+states[7]/constants[3])+states[6]))
    algebraic[4] = (constants[41]*(states[0]*states[6]-(states[1]*states[7])/constants[40]))/((1.00000+power(states[11]/constants[38], constants[39]))*(constants[34]*(1.00000+states[1]/constants[36])+states[0])*(constants[35]*(1.00000+states[7]/constants[37])+states[6]))
    algebraic[5] = (constants[33]*states[4])/(constants[32]+states[4])
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