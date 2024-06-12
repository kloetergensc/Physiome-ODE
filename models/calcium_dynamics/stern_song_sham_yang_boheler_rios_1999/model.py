# Size of variable arrays:
sizeAlgebraic = 0
sizeStates = 4
sizeConstants = 5
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (millisecond)"
    legend_states[0] = "R in component R (dimensionless)"
    legend_constants[0] = "ki in component reaction_constants (second_order_rate_constant)"
    legend_constants[1] = "ko in component reaction_constants (third_order_rate_constant)"
    legend_constants[2] = "kim in component reaction_constants (first_order_rate_constant)"
    legend_constants[3] = "kom in component reaction_constants (first_order_rate_constant)"
    legend_states[1] = "RI in component RI (dimensionless)"
    legend_states[2] = "O in component O (dimensionless)"
    legend_constants[4] = "Ca in component reaction_constants (millimolar)"
    legend_states[3] = "I in component I (dimensionless)"
    legend_rates[0] = "d/dt R in component R (dimensionless)"
    legend_rates[2] = "d/dt O in component O (dimensionless)"
    legend_rates[3] = "d/dt I in component I (dimensionless)"
    legend_rates[1] = "d/dt RI in component RI (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 1
    constants[0] = 0.5
    constants[1] = 35
    constants[2] = 0.005
    constants[3] = 0.06
    states[1] = 1
    states[2] = 1
    constants[4] = 0.0001
    states[3] = 1
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = (constants[3]*states[2]+constants[2]*states[1])-(constants[1]*(power(constants[4], 2.00000))*states[0]+constants[0]*constants[4]*states[0])
    rates[2] = (constants[2]*states[3]+constants[1]*(power(constants[4], 2.00000))*states[0])-(constants[3]*states[2]+constants[0]*constants[4]*states[2])
    rates[3] = (constants[0]*constants[4]*states[2]+constants[1]*(power(constants[4], 2.00000))*states[1])-(constants[2]*states[3]+constants[3]*states[3])
    rates[1] = (constants[0]*constants[4]*states[0]+constants[3]*states[3])-(constants[2]*states[1]+constants[1]*(power(constants[4], 2.00000))*states[1])
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