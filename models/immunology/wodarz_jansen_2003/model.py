# Size of variable arrays:
sizeAlgebraic = 1
sizeStates = 4
sizeConstants = 12
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (day)"
    legend_states[0] = "T in component T (dimensionless)"
    legend_constants[0] = "k in component T (first_order_rate_constant)"
    legend_constants[1] = "r in component T (first_order_rate_constant)"
    legend_constants[2] = "d in component T (first_order_rate_constant)"
    legend_constants[3] = "gamma in component T (first_order_rate_constant)"
    legend_states[1] = "C in component C (dimensionless)"
    legend_states[2] = "A in component A (dimensionless)"
    legend_constants[4] = "lambda in component A (first_order_rate_constant)"
    legend_constants[5] = "delta_1 in component A (first_order_rate_constant)"
    legend_constants[6] = "alpha in component kinetic_parameters (first_order_rate_constant)"
    legend_states[3] = "A_star in component A_star (dimensionless)"
    legend_constants[7] = "delta_2 in component A_star (first_order_rate_constant)"
    legend_constants[8] = "eta in component C (first_order_rate_constant)"
    legend_constants[9] = "epsilon in component C (dimensionless)"
    legend_constants[10] = "q in component C (first_order_rate_constant)"
    legend_constants[11] = "mu in component C (first_order_rate_constant)"
    legend_algebraic[0] = "R in component ratio (dimensionless)"
    legend_rates[0] = "d/dt T in component T (dimensionless)"
    legend_rates[2] = "d/dt A in component A (dimensionless)"
    legend_rates[3] = "d/dt A_star in component A_star (dimensionless)"
    legend_rates[1] = "d/dt C in component C (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.1
    constants[0] = 10
    constants[1] = 0.5
    constants[2] = 0.1
    constants[3] = 1
    states[1] = 0.015
    states[2] = 1
    constants[4] = 1
    constants[5] = 0.1
    constants[6] = 0.05
    states[3] = 2
    constants[7] = 1.5
    constants[8] = 2
    constants[9] = 1
    constants[10] = 0.5
    constants[11] = 0.1
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = (constants[1]*states[0]*(1.00000-(states[0]*1.00000)/constants[0])-constants[2]*states[0])-constants[3]*states[0]*states[1]
    rates[2] = (constants[4]-constants[5]*states[2])-constants[6]*states[2]*states[0]
    rates[3] = constants[6]*states[2]*states[0]-constants[7]*states[3]
    rates[1] = ((constants[8]*states[3]*states[1])/(constants[9]*states[1]+1.00000)-constants[10]*states[0]*states[1])-constants[11]*states[1]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = (states[1]*states[3])/(constants[10]*1.00000*states[0])
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