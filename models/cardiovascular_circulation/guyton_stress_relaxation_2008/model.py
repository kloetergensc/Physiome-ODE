# Size of variable arrays:
sizeAlgebraic = 0
sizeStates = 2
sizeConstants = 5
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_constants[0] = "VVE in component stress_relaxation (litre)"
    legend_states[0] = "VV7 in component short_term_stress_relaxation (litre)"
    legend_constants[1] = "SR in component parameter_values (dimensionless)"
    legend_constants[2] = "SRK in component parameter_values (minute)"
    legend_states[1] = "VV6 in component long_term_stress_relaxation (litre)"
    legend_constants[3] = "SR2 in component parameter_values (dimensionless)"
    legend_constants[4] = "SRK2 in component parameter_values (minute)"
    legend_rates[0] = "d/dt VV7 in component short_term_stress_relaxation (litre)"
    legend_rates[1] = "d/dt VV6 in component long_term_stress_relaxation (litre)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 0.743224
    states[0] = 0.00366525
    constants[1] = 1
    constants[2] = 5
    states[1] = 0.0101913
    constants[3] = 1
    constants[4] = 10000
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = ((constants[0]-0.740000)*constants[1]-states[0])/constants[2]
    rates[1] = ((constants[0]-0.740000)*constants[3]-states[1])/constants[4]
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