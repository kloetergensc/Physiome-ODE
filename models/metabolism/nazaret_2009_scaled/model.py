# Size of variable arrays:
sizeAlgebraic = 13
sizeStates = 8
sizeConstants = 55
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_algebraic[0] = "time in component environment (second)"
    legend_states[0] = "p in component p (dimensionless)"
    legend_constants[30] = "v1 in component v1 (dimensionless)"
    legend_algebraic[1] = "v2 in component v2 (dimensionless)"
    legend_algebraic[6] = "v7 in component v7 (dimensionless)"
    legend_voi = "tau in component normalised_constants (dimensionless)"
    legend_states[1] = "a in component a (dimensionless)"
    legend_algebraic[2] = "v3 in component v3 (dimensionless)"
    legend_constants[48] = "epsilon1 in component normalised_constants (dimensionless)"
    legend_states[2] = "c in component c (dimensionless)"
    legend_algebraic[3] = "v4 in component v4 (dimensionless)"
    legend_constants[49] = "epsilon2 in component normalised_constants (dimensionless)"
    legend_states[3] = "k in component k (dimensionless)"
    legend_algebraic[4] = "v5 in component v5 (dimensionless)"
    legend_algebraic[5] = "v6 in component v6 (dimensionless)"
    legend_constants[50] = "epsilon3 in component normalised_constants (dimensionless)"
    legend_states[4] = "o in component o (dimensionless)"
    legend_algebraic[7] = "v8 in component v8 (dimensionless)"
    legend_constants[51] = "epsilon4 in component normalised_constants (dimensionless)"
    legend_states[5] = "n in component n (dimensionless)"
    legend_algebraic[10] = "vresp in component vresp (dimensionless)"
    legend_constants[52] = "epsilon5 in component normalised_constants (dimensionless)"
    legend_states[6] = "en in component en (dimensionless)"
    legend_algebraic[12] = "vATP in component vATP (dimensionless)"
    legend_algebraic[8] = "vANT in component vANT (dimensionless)"
    legend_constants[53] = "epsilon6 in component normalised_constants (dimensionless)"
    legend_states[7] = "s in component s (dimensionless)"
    legend_algebraic[9] = "vleak in component vleak (dimensionless)"
    legend_constants[54] = "epsilon7 in component normalised_constants (dimensionless)"
    legend_constants[31] = "beta2 in component normalised_constants (dimensionless)"
    legend_constants[32] = "beta3 in component normalised_constants (dimensionless)"
    legend_constants[33] = "beta4 in component normalised_constants (dimensionless)"
    legend_constants[34] = "beta5 in component normalised_constants (dimensionless)"
    legend_constants[35] = "delta_6 in component normalised_constants (dimensionless)"
    legend_constants[36] = "beta6 in component normalised_constants (dimensionless)"
    legend_constants[37] = "beta7 in component normalised_constants (dimensionless)"
    legend_constants[38] = "beta8 in component normalised_constants (dimensionless)"
    legend_constants[39] = "beta_ANT in component normalised_constants (dimensionless)"
    legend_constants[40] = "beta_leak in component normalised_constants (dimensionless)"
    legend_constants[41] = "beta_resp in component normalised_constants (dimensionless)"
    legend_constants[42] = "delta_r1 in component normalised_constants (dimensionless)"
    legend_constants[43] = "delta_r2 in component normalised_constants (dimensionless)"
    legend_constants[44] = "beta_ATP in component normalised_constants (dimensionless)"
    legend_constants[45] = "delta_atp in component normalised_constants (dimensionless)"
    legend_algebraic[11] = "en_crit in component en_crit (dimensionless)"
    legend_constants[46] = "Kapp_dash in component normalised_constants (dimensionless)"
    legend_constants[47] = "delta_crit in component normalised_constants (dimensionless)"
    legend_constants[0] = "At in component normalised_constants (millimolar)"
    legend_constants[1] = "Nt in component normalised_constants (millimolar)"
    legend_constants[2] = "Pyr_bar in component normalised_constants (dimensionless)"
    legend_constants[3] = "Cit_bar in component normalised_constants (dimensionless)"
    legend_constants[4] = "AcCoA_bar in component normalised_constants (dimensionless)"
    legend_constants[5] = "KG_bar in component normalised_constants (dimensionless)"
    legend_constants[6] = "OAA_bar in component normalised_constants (dimensionless)"
    legend_constants[7] = "k1 in component normalised_constants (micromolar_per_second)"
    legend_constants[8] = "k2 in component normalised_constants (second_order_rate_constant)"
    legend_constants[9] = "k3 in component normalised_constants (second_order_rate_constant)"
    legend_constants[10] = "k4 in component normalised_constants (second_order_rate_constant)"
    legend_constants[11] = "k5 in component normalised_constants (third_order_rate_constant)"
    legend_constants[12] = "k6 in component normalised_constants (first_order_rate_constant)"
    legend_constants[13] = "k7 in component normalised_constants (second_order_rate_constant)"
    legend_constants[14] = "k8 in component normalised_constants (first_order_rate_constant)"
    legend_constants[15] = "kresp in component normalised_constants (millimolar_per_second)"
    legend_constants[16] = "kATP in component normalised_constants (millimolar_per_second)"
    legend_constants[17] = "kANT in component normalised_constants (dimensionless)"
    legend_constants[18] = "kleak in component normalised_constants (molar_per_millivolt_per_second)"
    legend_constants[19] = "Keq in component normalised_constants (dimensionless)"
    legend_constants[20] = "K in component normalised_constants (millimolar)"
    legend_constants[21] = "alpha in component normalised_constants (per_millivolt)"
    legend_constants[22] = "b in component normalised_constants (per_millimolar)"
    legend_constants[23] = "delta_psi_m in component normalised_constants (millivolt)"
    legend_constants[24] = "R in component normalised_constants (joule_per_mole_kelvin)"
    legend_constants[25] = "T in component normalised_constants (kelvin)"
    legend_constants[26] = "F in component normalised_constants (coulomb_per_mole)"
    legend_constants[27] = "C in component normalised_constants (millimolar_per_millivolt)"
    legend_constants[28] = "Kapp in component normalised_constants (per_millimolar)"
    legend_constants[29] = "Pi in component normalised_constants (millimolar)"
    legend_rates[0] = "d/dt p in component p (dimensionless)"
    legend_rates[1] = "d/dt a in component a (dimensionless)"
    legend_rates[2] = "d/dt c in component c (dimensionless)"
    legend_rates[3] = "d/dt k in component k (dimensionless)"
    legend_rates[4] = "d/dt o in component o (dimensionless)"
    legend_rates[5] = "d/dt n in component n (dimensionless)"
    legend_rates[6] = "d/dt en in component en (dimensionless)"
    legend_rates[7] = "d/dt s in component s (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.154
    states[1] = 0.063
    states[2] = 0.44
    states[3] = 0.225
    states[4] = 0.005
    states[5] = 0.856
    states[6] = 3.536
    states[7] = 150.0
    constants[0] = 4.160
    constants[1] = 1.070
    constants[2] = 0.161
    constants[3] = 0.460
    constants[4] = 0.105
    constants[5] = 0.146
    constants[6] = 0.004
    constants[7] = 38
    constants[8] = 152
    constants[9] = 57142
    constants[10] = 53
    constants[11] = 40
    constants[12] = 82361
    constants[13] = 3.2e-3
    constants[14] = 3.6
    constants[15] = 2.5
    constants[16] = 131.9
    constants[17] = 0.1
    constants[18] = 0.426
    constants[19] = 0.3975
    constants[20] = 2
    constants[21] = 0.100
    constants[22] = 0.004
    constants[23] = 150.0
    constants[24] = 8.314
    constants[25] = 298
    constants[26] = 96485
    constants[27] = 6.75e-06
    constants[28] = 4.4e-6
    constants[29] = 2.440
    constants[30] = 1.00000
    constants[31] = (constants[8]/constants[7])*constants[1]*constants[2]
    constants[32] = (constants[9]/constants[7])*constants[6]*constants[4]
    constants[33] = (constants[10]/constants[7])*constants[1]*constants[3]
    constants[34] = (constants[11]/constants[7])*constants[1]*constants[0]*constants[5]
    constants[35] = constants[5]/(constants[6]*constants[19])
    constants[36] = (constants[12]/constants[7])*constants[6]
    constants[37] = (constants[13]/constants[7])*constants[0]*constants[2]
    constants[38] = (constants[14]/constants[7])*constants[6]
    constants[39] = (constants[17]/constants[7])*constants[0]
    constants[40] = (constants[18]/constants[7])*constants[23]
    constants[41] = constants[15]/constants[7]
    constants[42] = constants[20]/constants[1]
    constants[43] = constants[21]*constants[23]
    constants[44] = constants[16]/constants[7]
    constants[45] = constants[22]*constants[0]
    constants[46] = constants[28]*constants[29]
    constants[47] = 3.00000*((1.20000*constants[26]*constants[23])/(constants[24]*constants[25]))
    constants[48] = constants[4]/constants[2]
    constants[49] = constants[3]/constants[2]
    constants[50] = constants[5]/constants[2]
    constants[51] = constants[6]/constants[2]
    constants[52] = constants[1]/constants[2]
    constants[53] = constants[0]/constants[2]
    constants[54] = (constants[23]/constants[2])*constants[27]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[1] = constants[31]*states[0]*states[5]
    algebraic[2] = constants[32]*states[4]*states[1]
    rates[1] = (algebraic[1]-algebraic[2])/constants[48]
    algebraic[3] = constants[33]*states[2]*states[5]
    rates[2] = (algebraic[2]-algebraic[3])/constants[49]
    algebraic[4] = constants[34]*states[3]*states[5]*(1.00000-states[6])
    algebraic[5] = constants[36]*(states[4]-constants[35]*states[3])
    rates[3] = ((algebraic[3]+algebraic[5])-algebraic[4])/constants[50]
    algebraic[6] = constants[37]*states[0]*states[6]
    rates[0] = constants[30]-(algebraic[1]+algebraic[6])
    algebraic[7] = constants[38]*states[4]
    rates[4] = ((algebraic[4]+algebraic[6])-(algebraic[2]+algebraic[7]+algebraic[5]))/constants[51]
    algebraic[10] = constants[41]*((1.00000-states[5])/((constants[42]+1.00000)-states[5]))*(1.00000/(1.00000+exp(constants[43]*(states[7]-1.00000))))
    rates[5] = (algebraic[10]-(algebraic[1]+algebraic[3]+2.00000*algebraic[4]))/constants[52]
    algebraic[11] = constants[46]/(constants[46]+exp(-1.00000*constants[47]*states[7]))
    algebraic[12] = constants[44]*(2.00000/(1.00000+exp(constants[45]*(states[6]-algebraic[11]*states[7])))-1.00000)
    algebraic[8] = constants[39]*states[6]
    rates[6] = ((algebraic[12]+algebraic[4])-(algebraic[8]+algebraic[6]))/constants[53]
    algebraic[9] = constants[40]*states[7]
    rates[7] = (10.0000*algebraic[10]-(3.00000*algebraic[12]+algebraic[9]+algebraic[8]))/constants[54]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[1] = constants[31]*states[0]*states[5]
    algebraic[2] = constants[32]*states[4]*states[1]
    algebraic[3] = constants[33]*states[2]*states[5]
    algebraic[4] = constants[34]*states[3]*states[5]*(1.00000-states[6])
    algebraic[5] = constants[36]*(states[4]-constants[35]*states[3])
    algebraic[6] = constants[37]*states[0]*states[6]
    algebraic[7] = constants[38]*states[4]
    algebraic[10] = constants[41]*((1.00000-states[5])/((constants[42]+1.00000)-states[5]))*(1.00000/(1.00000+exp(constants[43]*(states[7]-1.00000))))
    algebraic[11] = constants[46]/(constants[46]+exp(-1.00000*constants[47]*states[7]))
    algebraic[12] = constants[44]*(2.00000/(1.00000+exp(constants[45]*(states[6]-algebraic[11]*states[7])))-1.00000)
    algebraic[8] = constants[39]*states[6]
    algebraic[9] = constants[40]*states[7]
    rootfind_0(voi, constants, rates, states, algebraic)
    return algebraic

initialGuess0 = None
def rootfind_0(voi, constants, states, algebraic):
    """Calculate value of algebraic variable for DAE"""
    from scipy.optimize import fsolve
    global initialGuess0
    if initialGuess0 is None: initialGuess0 = 0.1
    if not iterable(voi):
        algebraic[0] = fsolve(residualSN_0, initialGuess0, args=(algebraic, voi, constants, rates, states), xtol=1E-6)
        initialGuess0 = algebraic[0]
    else:
        for (i,t) in enumerate(voi):
            algebraic[0][i] = fsolve(residualSN_0, initialGuess0, args=(algebraic[:,i], voi[i], constants, rates, states[:,i]), xtol=1E-6)
            initialGuess0 = algebraic[0][i]

def residualSN_0(algebraicCandidate, algebraic, voi, constants, rates, states):
    algebraic[0] = algebraicCandidate
    return (voi) - ((constants[7]/constants[2])*algebraic[0])

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