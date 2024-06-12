# Size of variable arrays:
sizeAlgebraic = 4
sizeStates = 3
sizeConstants = 15
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (min)"
    legend_states[0] = "Z in component Ca (uM)"
    legend_states[1] = "Y in component Ca (uM)"
    legend_states[2] = "X in component Ca (uM)"
    legend_constants[14] = "V_in in component V_in (uM_per_min)"
    legend_algebraic[0] = "V_2i in component V_2i (uM_per_min)"
    legend_algebraic[1] = "V_3i in component V_3i (uM_per_min)"
    legend_algebraic[2] = "V_2s in component V_2s (uM_per_min)"
    legend_algebraic[3] = "V_3s in component V_3s (uM_per_min)"
    legend_constants[0] = "K_f in component Ca (per_min)"
    legend_constants[1] = "K in component Ca (per_min)"
    legend_constants[2] = "beta in component Ca_flux (dimensionless)"
    legend_constants[3] = "v_0 in component V_in (uM_per_min)"
    legend_constants[4] = "v_1 in component V_in (uM_per_min)"
    legend_constants[5] = "V_M2i in component V_2i (uM_per_min)"
    legend_constants[6] = "K_2i in component V_2i (uM)"
    legend_constants[7] = "V_M3i in component V_3i (uM_per_min)"
    legend_constants[8] = "K_3z in component V_3i (uM)"
    legend_constants[9] = "K_3y in component V_3i (uM)"
    legend_constants[10] = "V_M2s in component V_2s (uM_per_min)"
    legend_constants[11] = "K_2s in component V_2s (uM)"
    legend_constants[12] = "V_M3s in component V_3s (uM_per_min)"
    legend_constants[13] = "K_3s in component V_3s (uM)"
    legend_rates[0] = "d/dt Z in component Ca (uM)"
    legend_rates[1] = "d/dt Y in component Ca (uM)"
    legend_rates[2] = "d/dt X in component Ca (uM)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.0
    states[1] = 0.0
    states[2] = 0.5
    constants[0] = 0.5
    constants[1] = 1
    constants[2] = 1
    constants[3] = 0.015
    constants[4] = 0.012
    constants[5] = 3.1
    constants[6] = 0.005
    constants[7] = 25
    constants[8] = 0.022
    constants[9] = 0.065
    constants[10] = 1.5
    constants[11] = 0.0265
    constants[12] = 0.169
    constants[13] = 0.1
    constants[14] = constants[3]+constants[4]*constants[2]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[0] = constants[5]*((power(states[0], 2.00000))/(power(constants[6], 2.00000)+power(states[0], 2.00000)))
    algebraic[1] = constants[7]*((power(states[1], 2.00000))/(power(constants[9], 2.00000)+power(states[1], 2.00000)))*((power(states[0], 2.00000))/(power(constants[8], 2.00000)+power(states[0], 2.00000)))
    rates[1] = algebraic[0]+-algebraic[1]+-(constants[0]*states[1])
    algebraic[2] = constants[10]*((power(states[0], 2.00000))/(power(constants[11], 2.00000)+power(states[0], 2.00000)))
    algebraic[3] = constants[2]*constants[12]*((power(states[2], 2.00000))/(power(constants[13], 2.00000)+power(states[2], 2.00000)))
    rates[0] = constants[14]+-algebraic[0]+algebraic[1]+constants[0]*states[1]+-algebraic[2]+algebraic[3]+constants[0]*states[2]+-(constants[1]*states[0])
    rates[2] = algebraic[2]+-algebraic[3]+-(constants[0]*states[2])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = constants[5]*((power(states[0], 2.00000))/(power(constants[6], 2.00000)+power(states[0], 2.00000)))
    algebraic[1] = constants[7]*((power(states[1], 2.00000))/(power(constants[9], 2.00000)+power(states[1], 2.00000)))*((power(states[0], 2.00000))/(power(constants[8], 2.00000)+power(states[0], 2.00000)))
    algebraic[2] = constants[10]*((power(states[0], 2.00000))/(power(constants[11], 2.00000)+power(states[0], 2.00000)))
    algebraic[3] = constants[2]*constants[12]*((power(states[2], 2.00000))/(power(constants[13], 2.00000)+power(states[2], 2.00000)))
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