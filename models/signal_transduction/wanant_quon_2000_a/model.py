# Size of variable arrays:
sizeAlgebraic = 1
sizeStates = 4
sizeConstants = 4
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_states[0] = "x1 in component insulin (nanomolar)"
    legend_algebraic[0] = "scatchard in component insulin (dimensionless)"
    legend_constants[0] = "k1 in component rate_constants (second_order_rate_constant)"
    legend_constants[1] = "k1_ in component rate_constants (first_order_rate_constant)"
    legend_constants[2] = "k2_ in component rate_constants (first_order_rate_constant)"
    legend_states[1] = "x2 in component unbound_receptor (nanomolar)"
    legend_states[2] = "x3 in component single_bound_receptor (nanomolar)"
    legend_states[3] = "x4 in component double_bound_receptor (nanomolar)"
    legend_constants[3] = "k2 in component rate_constants (second_order_rate_constant)"
    legend_rates[0] = "d/dt x1 in component insulin (nanomolar)"
    legend_rates[1] = "d/dt x2 in component unbound_receptor (nanomolar)"
    legend_rates[2] = "d/dt x3 in component single_bound_receptor (nanomolar)"
    legend_rates[3] = "d/dt x4 in component double_bound_receptor (nanomolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 1000
    constants[0] = 1000000
    constants[1] = 0.0004
    constants[2] = 0.04
    states[1] = 0.1
    states[2] = 0.0
    states[3] = 0.0
    constants[3] = 1000000
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = ((constants[1]*states[2]-constants[0]*states[0]*states[1])+constants[2]*states[3])-constants[3]*states[0]*states[2]
    rates[1] = constants[1]*states[2]-constants[0]*states[0]*states[1]
    rates[2] = ((constants[0]*states[0]*states[1]-constants[1]*states[2])+constants[2]*states[3])-constants[3]*states[0]*states[2]
    rates[3] = constants[3]*states[0]*states[2]-constants[2]*states[3]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = (states[2]+states[3])/states[0]
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