# Size of variable arrays:
sizeAlgebraic = 1
sizeStates = 14
sizeConstants = 20
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_states[0] = "Fe3 in component Fe3 (micromolar)"
    legend_constants[0] = "Arg in component model_constants (micromolar)"
    legend_states[1] = "Fe3_Arg in component Fe3_Arg (micromolar)"
    legend_states[2] = "Fe3_NO in component Fe3_NO (micromolar)"
    legend_states[3] = "Fe2_NO in component Fe2_NO (micromolar)"
    legend_constants[1] = "O2 in component model_constants (micromolar)"
    legend_constants[2] = "k1 in component model_constants (second_order_rate_constant)"
    legend_constants[3] = "k_1 in component model_constants (first_order_rate_constant)"
    legend_constants[4] = "k2 in component model_constants (first_order_rate_constant)"
    legend_constants[5] = "k13 in component model_constants (first_order_rate_constant)"
    legend_constants[6] = "k12 in component model_constants (second_order_rate_constant)"
    legend_constants[7] = "k3 in component model_constants (first_order_rate_constant)"
    legend_states[4] = "Fe2 in component Fe2 (micromolar)"
    legend_states[5] = "Fe2_Arg in component Fe2_Arg (micromolar)"
    legend_constants[8] = "k_4 in component model_constants (first_order_rate_constant)"
    legend_constants[9] = "k4 in component model_constants (second_order_rate_constant)"
    legend_states[6] = "Fe3_O2_Arg in component Fe3_O2_Arg (micromolar)"
    legend_constants[10] = "k5 in component model_constants (second_order_rate_constant)"
    legend_constants[11] = "k_5 in component model_constants (first_order_rate_constant)"
    legend_constants[12] = "k6 in component model_constants (first_order_rate_constant)"
    legend_states[7] = "Fe3_NOHA in component Fe3_NOHA (micromolar)"
    legend_constants[13] = "k7 in component model_constants (first_order_rate_constant)"
    legend_states[8] = "Fe2_NOHA in component Fe2_NOHA (micromolar)"
    legend_states[9] = "NOHA in component NOHA (micromolar)"
    legend_states[10] = "Fe3_O2_NOHA in component Fe3_O2_NOHA (micromolar)"
    legend_constants[14] = "k9 in component model_constants (second_order_rate_constant)"
    legend_constants[15] = "k_9 in component model_constants (first_order_rate_constant)"
    legend_constants[16] = "k_8 in component model_constants (first_order_rate_constant)"
    legend_constants[17] = "k8 in component model_constants (second_order_rate_constant)"
    legend_constants[18] = "k10 in component model_constants (first_order_rate_constant)"
    legend_constants[19] = "k11 in component model_constants (first_order_rate_constant)"
    legend_states[11] = "NO in component NO (micromolar)"
    legend_algebraic[0] = "dNOdt in component NO (flux)"
    legend_states[12] = "citrulline in component citrulline (micromolar)"
    legend_states[13] = "NO3 in component NO3 (micromolar)"
    legend_rates[0] = "d/dt Fe3 in component Fe3 (micromolar)"
    legend_rates[1] = "d/dt Fe3_Arg in component Fe3_Arg (micromolar)"
    legend_rates[4] = "d/dt Fe2 in component Fe2 (micromolar)"
    legend_rates[5] = "d/dt Fe2_Arg in component Fe2_Arg (micromolar)"
    legend_rates[6] = "d/dt Fe3_O2_Arg in component Fe3_O2_Arg (micromolar)"
    legend_rates[7] = "d/dt Fe3_NOHA in component Fe3_NOHA (micromolar)"
    legend_rates[8] = "d/dt Fe2_NOHA in component Fe2_NOHA (micromolar)"
    legend_rates[10] = "d/dt Fe3_O2_NOHA in component Fe3_O2_NOHA (micromolar)"
    legend_rates[2] = "d/dt Fe3_NO in component Fe3_NO (micromolar)"
    legend_rates[3] = "d/dt Fe2_NO in component Fe2_NO (micromolar)"
    legend_rates[11] = "d/dt NO in component NO (micromolar)"
    legend_rates[12] = "d/dt citrulline in component citrulline (micromolar)"
    legend_rates[13] = "d/dt NO3 in component NO3 (micromolar)"
    legend_rates[9] = "d/dt NOHA in component NOHA (micromolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.9
    constants[0] = 100.0
    states[1] = 0.0
    states[2] = 0.0
    states[3] = 0.0
    constants[1] = 100.0
    constants[2] = 6.6
    constants[3] = 6.6
    constants[4] = 20.8
    constants[5] = 39.9
    constants[6] = 0.01
    constants[7] = 20.8
    states[4] = 0.0
    states[5] = 0.0
    constants[8] = 6.6
    constants[9] = 6.6
    states[6] = 0.0
    constants[10] = 8.5
    constants[11] = 215.6
    constants[12] = 175.6
    states[7] = 0.0
    constants[13] = 20.8
    states[8] = 0.0
    states[9] = 0.0
    states[10] = 0.0
    constants[14] = 8.6
    constants[15] = 399.2
    constants[16] = 13.2
    constants[17] = 13.2
    constants[18] = 39.1
    constants[19] = 20.8
    states[11] = 0.0
    states[12] = 0.0
    states[13] = 0.0
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[11] = constants[5]*states[2]
    rates[0] = (constants[3]*states[1]+constants[5]*states[2]+constants[6]*states[3]*constants[1])-(constants[2]*constants[0]*states[0]+constants[4]*states[0])
    rates[1] = constants[2]*states[0]*constants[0]-(constants[3]*states[1]+constants[7]*states[1])
    rates[4] = (constants[4]*states[0]+constants[8]*states[5])-constants[9]*states[4]*constants[0]
    rates[5] = (constants[7]*states[1]+constants[11]*states[6]+constants[9]*states[4]*constants[0])-(constants[10]*states[5]*constants[1]+constants[8]*states[5])
    rates[6] = constants[10]*states[5]*constants[1]-(constants[12]*states[6]+constants[11]*states[6])
    rates[7] = constants[12]*states[6]-constants[13]*states[7]
    rates[8] = (constants[13]*states[7]+constants[15]*states[10]+constants[17]*states[4]*states[9])-(constants[16]*states[8]+constants[14]*states[8]*constants[1])
    rates[10] = constants[14]*states[8]*constants[1]-(constants[18]*states[10]+constants[15]*states[10])
    rates[2] = constants[18]*states[10]-(constants[5]*states[2]+constants[19]*states[2])
    rates[3] = constants[19]*states[2]-constants[6]*states[3]*constants[1]
    rates[12] = constants[18]*states[10]
    rates[13] = constants[6]*states[3]*constants[1]
    rates[9] = constants[16]*states[8]-constants[17]*states[4]*states[9]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = rates[11]
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