# Size of variable arrays:
sizeAlgebraic = 13
sizeStates = 8
sizeConstants = 25
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_states[0] = "Pyr in component Pyr (micromolar)"
    legend_constants[24] = "v1 in component v1 (micromolar_per_second)"
    legend_algebraic[0] = "v2 in component v2 (micromolar_per_second)"
    legend_algebraic[5] = "v7 in component v7 (micromolar_per_second)"
    legend_states[1] = "AcCoA in component AcCoA (micromolar)"
    legend_algebraic[1] = "v3 in component v3 (micromolar_per_second)"
    legend_states[2] = "Cit in component Cit (micromolar)"
    legend_algebraic[2] = "v4 in component v4 (micromolar_per_second)"
    legend_states[3] = "KG in component KG (micromolar)"
    legend_algebraic[3] = "v5 in component v5 (micromolar_per_second)"
    legend_algebraic[4] = "v6 in component v6 (micromolar_per_second)"
    legend_states[4] = "OAA in component OAA (micromolar)"
    legend_algebraic[6] = "v8 in component v8 (micromolar_per_second)"
    legend_states[5] = "NAD in component NAD (micromolar)"
    legend_algebraic[9] = "vresp in component vresp (micromolar_per_second)"
    legend_states[6] = "ATP in component ATP (micromolar)"
    legend_algebraic[12] = "vATP in component vATP (micromolar_per_second)"
    legend_algebraic[7] = "vANT in component vANT (micromolar_per_second)"
    legend_states[7] = "delta_psi in component delta_psi (millivolt)"
    legend_constants[0] = "C in component delta_psi (millimolar_per_millivolt)"
    legend_algebraic[8] = "vleak in component vleak (micromolar_per_second)"
    legend_constants[1] = "k1 in component v1 (micromolar_per_second)"
    legend_constants[2] = "k2 in component v2 (second_order_rate_constant)"
    legend_constants[3] = "k3 in component v3 (second_order_rate_constant)"
    legend_constants[4] = "k4 in component v4 (second_order_rate_constant)"
    legend_constants[5] = "k5 in component v5 (third_order_rate_constant)"
    legend_constants[6] = "At in component model_parameters (millimolar)"
    legend_constants[7] = "k6 in component v6 (first_order_rate_constant)"
    legend_constants[8] = "k7 in component v7 (second_order_rate_constant)"
    legend_constants[9] = "k8 in component v8 (first_order_rate_constant)"
    legend_constants[10] = "kANT in component vANT (first_order_rate_constant)"
    legend_constants[11] = "kleak in component vleak (molar_per_millivolt_per_second)"
    legend_constants[12] = "kresp in component vresp (millimolar_per_second)"
    legend_constants[13] = "K in component vresp (millimolar)"
    legend_constants[14] = "a in component vresp (per_millivolt)"
    legend_constants[15] = "delta_psi_m in component vresp (millivolt)"
    legend_constants[16] = "Nt in component model_parameters (millimolar)"
    legend_constants[17] = "kATP in component vATP (millimolar_per_second)"
    legend_constants[18] = "b in component vATP (per_micromolar)"
    legend_algebraic[11] = "ATP_crit_delta_psi in component ATP_crit_delta_psi (micromolar)"
    legend_constants[19] = "R in component ATP_crit_delta_psi (joule_per_mole_kelvin)"
    legend_constants[20] = "T in component ATP_crit_delta_psi (kelvin)"
    legend_constants[21] = "F in component ATP_crit_delta_psi (coulomb_per_mole)"
    legend_constants[22] = "Kapp in component ATP_crit_delta_psi (per_millimolar)"
    legend_constants[23] = "Pi in component ATP_crit_delta_psi (millimolar)"
    legend_algebraic[10] = "delta_G_transport in component ATP_crit_delta_psi (joule_per_mole)"
    legend_rates[0] = "d/dt Pyr in component Pyr (micromolar)"
    legend_rates[1] = "d/dt AcCoA in component AcCoA (micromolar)"
    legend_rates[2] = "d/dt Cit in component Cit (micromolar)"
    legend_rates[3] = "d/dt KG in component KG (micromolar)"
    legend_rates[4] = "d/dt OAA in component OAA (micromolar)"
    legend_rates[5] = "d/dt NAD in component NAD (micromolar)"
    legend_rates[6] = "d/dt ATP in component ATP (micromolar)"
    legend_rates[7] = "d/dt delta_psi in component delta_psi (millivolt)"
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
    constants[0] = 6.75e-06
    constants[1] = 38.0
    constants[2] = 152.0
    constants[3] = 57142.0
    constants[4] = 53.0
    constants[5] = 82361.0
    constants[6] = 4.160
    constants[7] = 3.2e-3
    constants[8] = 40.0
    constants[9] = 3.6
    constants[10] = 0.1
    constants[11] = 0.426
    constants[12] = 2.5
    constants[13] = 2
    constants[14] = 0.1
    constants[15] = 150.0
    constants[16] = 1.070
    constants[17] = 131.9
    constants[18] = 4
    constants[19] = 8.314
    constants[20] = 298
    constants[21] = 96485
    constants[22] = 4.4e-6
    constants[23] = 2.440
    constants[24] = constants[1]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[0] = constants[2]*states[0]*states[5]
    algebraic[1] = constants[3]*states[4]*states[1]
    rates[1] = algebraic[0]-algebraic[1]
    algebraic[2] = constants[4]*states[2]*states[5]
    rates[2] = algebraic[1]-algebraic[2]
    algebraic[3] = constants[5]*states[3]*states[5]*(constants[6]-states[6])
    algebraic[4] = constants[7]*(states[4]-states[3])
    rates[3] = (algebraic[2]+algebraic[4])-algebraic[3]
    algebraic[5] = constants[8]*states[0]*states[6]
    rates[0] = constants[24]-(algebraic[0]+algebraic[5])
    algebraic[6] = constants[9]*states[4]
    rates[4] = (algebraic[3]+algebraic[5])-(algebraic[1]+algebraic[6]+algebraic[4])
    algebraic[9] = constants[12]*((constants[16]-states[5])/((constants[13]+constants[16])-states[5]))*(1.00000/(1.00000+exp(constants[14]*(states[7]-constants[15]))))
    rates[5] = algebraic[9]-(algebraic[0]+algebraic[2]+2.00000*algebraic[3])
    algebraic[10] = 0.00120000*constants[21]*states[7]
    algebraic[11] = constants[6]/(1.00000+exp((-3.00000*algebraic[10])/(constants[19]*constants[20]))/(constants[22]*constants[23]))
    algebraic[12] = constants[17]*(2.00000/(1.00000+exp(constants[18]*(states[6]-algebraic[11])))-1.00000)
    algebraic[7] = constants[10]*states[6]
    rates[6] = (algebraic[12]+algebraic[3])-(algebraic[7]+algebraic[5])
    algebraic[8] = constants[11]*states[7]
    rates[7] = (1.00000/constants[0])*(10.0000*algebraic[9]-(3.00000*algebraic[12]+algebraic[8]+algebraic[7]))
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = constants[2]*states[0]*states[5]
    algebraic[1] = constants[3]*states[4]*states[1]
    algebraic[2] = constants[4]*states[2]*states[5]
    algebraic[3] = constants[5]*states[3]*states[5]*(constants[6]-states[6])
    algebraic[4] = constants[7]*(states[4]-states[3])
    algebraic[5] = constants[8]*states[0]*states[6]
    algebraic[6] = constants[9]*states[4]
    algebraic[9] = constants[12]*((constants[16]-states[5])/((constants[13]+constants[16])-states[5]))*(1.00000/(1.00000+exp(constants[14]*(states[7]-constants[15]))))
    algebraic[10] = 0.00120000*constants[21]*states[7]
    algebraic[11] = constants[6]/(1.00000+exp((-3.00000*algebraic[10])/(constants[19]*constants[20]))/(constants[22]*constants[23]))
    algebraic[12] = constants[17]*(2.00000/(1.00000+exp(constants[18]*(states[6]-algebraic[11])))-1.00000)
    algebraic[7] = constants[10]*states[6]
    algebraic[8] = constants[11]*states[7]
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