# Size of variable arrays:
sizeAlgebraic = 0
sizeStates = 3
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
    legend_states[0] = "cortisol in component cortisol (mcg_ml)"
    legend_constants[0] = "k1 in component cortisol (first_order_rate_constant)"
    legend_constants[1] = "k2 in component cortisol (flux)"
    legend_constants[2] = "k3 in component cortisol (first_order_rate_constant)"
    legend_states[1] = "ACTH in component ACTH (mcg_ml)"
    legend_constants[3] = "k4 in component ACTH (first_order_rate_constant)"
    legend_constants[4] = "k5 in component ACTH (flux)"
    legend_constants[5] = "k6 in component ACTH (first_order_rate_constant)"
    legend_constants[6] = "Kd in component ACTH (mcg_ml)"
    legend_constants[7] = "Imax in component ACTH (dimensionless)"
    legend_states[2] = "CRH in component CRH (mcg_ml)"
    legend_constants[8] = "k7 in component CRH (flux)"
    legend_constants[9] = "k8 in component CRH (first_order_rate_constant)"
    legend_constants[10] = "pulse in component CRH (flux)"
    legend_rates[0] = "d/dt cortisol in component cortisol (mcg_ml)"
    legend_rates[1] = "d/dt ACTH in component ACTH (mcg_ml)"
    legend_rates[2] = "d/dt CRH in component CRH (mcg_ml)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.0
    constants[0] = 5E5
    constants[1] = 0.01
    constants[2] = 0.01
    states[1] = 0.0
    constants[3] = 10.0
    constants[4] = 4E-3
    constants[5] = 0.035
    constants[6] = 0.004
    constants[7] = 0.99
    states[2] = 50.0
    constants[8] = 1E-6
    constants[9] = 0.01
    constants[10] = 50.0
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = (constants[0]*states[1]+constants[1])-constants[2]*states[0]
    rates[1] = (constants[3]*states[2]+constants[4])-(constants[5]*states[1]+(constants[3]*states[2]+constants[4])*((constants[7]*states[0])/(constants[6]+states[0])))
    rates[2] = (constants[10]+constants[8])-constants[9]*states[2]
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