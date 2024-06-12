# Size of variable arrays:
sizeAlgebraic = 1
sizeStates = 2
sizeConstants = 11
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_states[0] = "H1 in component H1 (picomolar)"
    legend_algebraic[0] = "A in component A (flux)"
    legend_constants[7] = "beta_I in component model_parameters (first_order_rate_constant)"
    legend_states[1] = "H2 in component H2 (picomolar)"
    legend_constants[8] = "beta_C in component model_parameters (first_order_rate_constant)"
    legend_constants[0] = "t_on in component A (minute)"
    legend_constants[1] = "t_off in component A (minute)"
    legend_constants[2] = "A_max in component A (flux)"
    legend_constants[9] = "alpha in component model_parameters (first_order_rate_constant)"
    legend_constants[10] = "lamda in component model_parameters (first_order_rate_constant)"
    legend_constants[3] = "tau_I in component model_parameters (minute)"
    legend_constants[4] = "tau_C in component model_parameters (minute)"
    legend_constants[5] = "tau_alpha in component model_parameters (minute)"
    legend_constants[6] = "tau_lamda in component model_parameters (minute)"
    legend_rates[0] = "d/dt H1 in component H1 (picomolar)"
    legend_rates[1] = "d/dt H2 in component H2 (picomolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.05
    states[1] = 1.0
    constants[0] = 1316.0
    constants[1] = 1792.0
    constants[2] = 6.51
    constants[3] = 2.82
    constants[4] = 23.67
    constants[5] = 25.92
    constants[6] = 24.04
    constants[7] = log(2.00000)/constants[3]
    constants[8] = log(2.00000)/constants[4]
    constants[9] = log(2.00000)/constants[5]
    constants[10] = log(2.00000)/constants[6]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[1] = constants[7]*states[0]-constants[8]*states[1]
    algebraic[0] = custom_piecewise([less(voi , constants[1]) & greater_equal(voi , constants[0]), constants[2]*((1.00000-exp(-constants[10]*(voi-constants[0])))/(1.00000-exp(-constants[10]*(constants[1]-constants[0])))) , greater_equal(voi , constants[1]), constants[2]*exp(-constants[9]*(voi-constants[1])) , True, 0.00000])
    rates[0] = -(constants[7]*states[0])+algebraic[0]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = custom_piecewise([less(voi , constants[1]) & greater_equal(voi , constants[0]), constants[2]*((1.00000-exp(-constants[10]*(voi-constants[0])))/(1.00000-exp(-constants[10]*(constants[1]-constants[0])))) , greater_equal(voi , constants[1]), constants[2]*exp(-constants[9]*(voi-constants[1])) , True, 0.00000])
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