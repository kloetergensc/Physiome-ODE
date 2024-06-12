# Size of variable arrays:
sizeAlgebraic = 2
sizeStates = 3
sizeConstants = 13
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (day)"
    legend_states[0] = "x1 in component x1 (cell)"
    legend_constants[0] = "alpha1 in component model_parameters (flux)"
    legend_constants[1] = "beta1 in component model_parameters (first_order_rate_constant)"
    legend_constants[2] = "g11 in component model_parameters (dimensionless)"
    legend_constants[3] = "g21 in component model_parameters (dimensionless)"
    legend_states[1] = "x2 in component x2 (cell)"
    legend_constants[4] = "alpha2 in component model_parameters (first_order_rate_constant)"
    legend_constants[5] = "beta2 in component model_parameters (first_order_rate_constant)"
    legend_constants[6] = "g12 in component model_parameters (dimensionless)"
    legend_constants[7] = "g22 in component model_parameters (dimensionless)"
    legend_states[2] = "z in component z (percent)"
    legend_constants[8] = "k1 in component model_parameters (percent_per_cell_per_day)"
    legend_constants[9] = "k2 in component model_parameters (percent_per_cell_per_day)"
    legend_algebraic[0] = "y1 in component y1 (cell)"
    legend_algebraic[1] = "y2 in component y2 (cell)"
    legend_constants[11] = "x1_bar in component x1_bar (cell)"
    legend_constants[12] = "x2_bar in component x2_bar (cell)"
    legend_constants[10] = "gamma in component model_parameters (dimensionless)"
    legend_rates[0] = "d/dt x1 in component x1 (cell)"
    legend_rates[1] = "d/dt x2 in component x2 (cell)"
    legend_rates[2] = "d/dt z in component z (percent)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 10.06066
    constants[0] = 3
    constants[1] = 0.2
    constants[2] = 1.1
    constants[3] = -0.5
    states[1] = 212.132
    constants[4] = 4
    constants[5] = 0.02
    constants[6] = 1
    constants[7] = 0
    states[2] = 100.0
    constants[8] = 0.093
    constants[9] = 0.0008
    constants[10] = constants[6]*constants[3]-(1.00000-constants[2])*(1.00000-constants[7])
    constants[11] = (power(constants[1]/constants[0], (1.00000-constants[7])/constants[10]))*(power(constants[5]/constants[4], constants[3]/constants[10]))
    constants[12] = (power(constants[1]/constants[0], constants[6]/constants[10]))*(power(constants[5]/constants[4], (1.00000-constants[2])/constants[10]))
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = constants[0]*(power(states[0], constants[2]))*(power(states[1], constants[3]))-constants[1]*states[0]
    rates[1] = constants[4]*(power(states[0], constants[6]))*(power(states[1], constants[7]))-constants[5]*states[1]
    algebraic[0] = custom_piecewise([greater(states[0] , constants[11]), states[0]-constants[11] , True, 0.00000])
    algebraic[1] = custom_piecewise([greater(states[1] , constants[12]), states[1]-constants[12] , True, 0.00000])
    rates[2] = constants[9]*algebraic[1]-constants[8]*algebraic[0]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = custom_piecewise([greater(states[0] , constants[11]), states[0]-constants[11] , True, 0.00000])
    algebraic[1] = custom_piecewise([greater(states[1] , constants[12]), states[1]-constants[12] , True, 0.00000])
    return algebraic

def custom_piecewise(cases):
    """Compute result of a piecewise function"""
    return select(cases[0::2],cases[1::2])

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