# Size of variable arrays:
sizeAlgebraic = 0
sizeStates = 3
sizeConstants = 7
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_states[0] = "x in component x (dimensionless)"
    legend_constants[0] = "alpha in component x (per_second)"
    legend_constants[1] = "omega in component x (per_second)"
    legend_constants[2] = "D in component x (per_second)"
    legend_constants[3] = "a in component x (dimensionless)"
    legend_constants[4] = "b in component model_constants (dimensionless)"
    legend_states[1] = "z in component z (dimensionless)"
    legend_states[2] = "y in component y (dimensionless)"
    legend_constants[5] = "beta in component y (per_second)"
    legend_constants[6] = "gamma in component z (per_second)"
    legend_rates[0] = "d/dt x in component x (dimensionless)"
    legend_rates[2] = "d/dt y in component y (dimensionless)"
    legend_rates[1] = "d/dt z in component z (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 1.5
    constants[0] = 0.5
    constants[1] = 2
    constants[2] = 0.8228
    constants[3] = 8.1252
    constants[4] = 1.091
    states[1] = 1.0
    states[2] = 0.95
    constants[5] = 0.38
    constants[6] = 0.6
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = (constants[0]*exp(constants[3]*(1.00000-power(states[1], 2.00000))+constants[4]*(1.00000-power(states[2], 2.00000)))+constants[2]*cos(constants[1]*voi))-constants[0]*states[0]
    rates[2] = constants[5]*states[0]*exp(constants[4]*(1.00000-power(states[1], 2.00000)))-constants[5]*states[2]
    rates[1] = constants[6]*states[2]-constants[6]*states[1]
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