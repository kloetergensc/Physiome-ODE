# Size of variable arrays:
sizeAlgebraic = 1
sizeStates = 3
sizeConstants = 10
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (hour)"
    legend_states[0] = "M in component M (nanomolar)"
    legend_constants[0] = "vs in component M (flux)"
    legend_constants[1] = "vm in component M (flux)"
    legend_constants[2] = "Km in component M (nanomolar)"
    legend_constants[3] = "KI in component M (nanomolar)"
    legend_constants[4] = "n in component M (dimensionless)"
    legend_states[1] = "FN in component FN (nanomolar)"
    legend_states[2] = "FC in component FC (nanomolar)"
    legend_algebraic[0] = "Ft in component FC (nanomolar)"
    legend_constants[5] = "ks in component FC (first_order_rate_constant)"
    legend_constants[6] = "vd in component FC (flux)"
    legend_constants[7] = "Kd in component FC (nanomolar)"
    legend_constants[8] = "k1 in component parameters (first_order_rate_constant)"
    legend_constants[9] = "k2 in component parameters (first_order_rate_constant)"
    legend_rates[0] = "d/dt M in component M (nanomolar)"
    legend_rates[2] = "d/dt FC in component FC (nanomolar)"
    legend_rates[1] = "d/dt FN in component FN (nanomolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.1
    constants[0] = 1.6
    constants[1] = 0.505
    constants[2] = 0.5
    constants[3] = 1.0
    constants[4] = 4.0
    states[1] = 0.1
    states[2] = 0.1
    constants[5] = 0.5
    constants[6] = 1.4
    constants[7] = 0.13
    constants[8] = 0.5
    constants[9] = 0.6
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = constants[0]*((power(constants[3], constants[4]))/(power(constants[3], constants[4])+power(states[1], constants[4])))-constants[1]*(states[0]/(constants[2]+states[0]))
    rates[2] = (constants[5]*states[0]+constants[9]*states[1])-(constants[6]*(states[2]/(constants[7]+states[2]))+constants[8]*states[2])
    rates[1] = constants[8]*states[2]-constants[9]*states[1]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = states[2]+states[1]
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