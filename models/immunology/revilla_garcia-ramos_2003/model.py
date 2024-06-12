# Size of variable arrays:
sizeAlgebraic = 0
sizeStates = 5
sizeConstants = 10
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (day)"
    legend_states[0] = "x in component x (cell)"
    legend_constants[0] = "lamda in component x (cell_per_mm3_day)"
    legend_constants[1] = "d in component x (first_order_rate_constant)"
    legend_constants[2] = "beta in component model_parameters (mm3_per_vir_day)"
    legend_states[1] = "v in component v (vir)"
    legend_states[2] = "y in component y (cell)"
    legend_constants[3] = "a in component y (first_order_rate_constant)"
    legend_constants[4] = "alpha in component model_parameters (mm3_per_vir_day)"
    legend_states[3] = "w in component w (vir)"
    legend_states[4] = "z in component z (cell)"
    legend_constants[5] = "b in component z (first_order_rate_constant)"
    legend_constants[6] = "k in component v (vir_per_cell_day)"
    legend_constants[7] = "u in component v (first_order_rate_constant)"
    legend_constants[8] = "c in component w (vir_per_cell_day)"
    legend_constants[9] = "q in component w (first_order_rate_constant)"
    legend_rates[0] = "d/dt x in component x (cell)"
    legend_rates[2] = "d/dt y in component y (cell)"
    legend_rates[4] = "d/dt z in component z (cell)"
    legend_rates[1] = "d/dt v in component v (vir)"
    legend_rates[3] = "d/dt w in component w (vir)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 3
    constants[0] = 2.0
    constants[1] = 0.01
    constants[2] = 0.004
    states[1] = 149
    states[2] = 6
    constants[3] = 0.33
    constants[4] = 0.004
    states[3] = 1
    states[4] = 0
    constants[5] = 2.0
    constants[6] = 50.0
    constants[7] = 2.0
    constants[8] = 2000.0
    constants[9] = 2.0
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = 1.00000*constants[0]-(constants[1]*states[0]+1.00000*constants[2]*states[0]*states[1])
    rates[2] = 1.00000*constants[2]*states[0]*states[1]-(constants[3]*states[2]+1.00000*constants[4]*states[3]*states[2])
    rates[4] = 1.00000*constants[4]*states[3]*states[2]-constants[5]*states[4]
    rates[1] = constants[6]*states[2]-constants[7]*states[1]
    rates[3] = constants[8]*states[4]-constants[9]*states[3]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
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