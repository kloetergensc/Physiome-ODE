# Size of variable arrays:
sizeAlgebraic = 1
sizeStates = 4
sizeConstants = 7
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (hour)"
    legend_states[0] = "c in component c (dimensionless)"
    legend_algebraic[0] = "f in component c (dimensionless)"
    legend_constants[0] = "kcd in component reaction_constants (first_order_rate_constant)"
    legend_constants[1] = "ki1 in component reaction_constants (dimensionless)"
    legend_states[1] = "o in component o (dimensionless)"
    legend_states[2] = "a in component a (dimensionless)"
    legend_constants[2] = "kad in component reaction_constants (first_order_rate_constant)"
    legend_constants[3] = "ki2 in component reaction_constants (first_order_rate_constant)"
    legend_states[3] = "r in component r (dimensionless)"
    legend_constants[4] = "kcr in component reaction_constants (first_order_rate_constant)"
    legend_constants[5] = "krd in component reaction_constants (first_order_rate_constant)"
    legend_constants[6] = "k in component reaction_constants (dimensionless)"
    legend_rates[0] = "d/dt c in component c (dimensionless)"
    legend_rates[2] = "d/dt a in component a (dimensionless)"
    legend_rates[3] = "d/dt r in component r (dimensionless)"
    legend_rates[1] = "d/dt o in component o (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.6
    constants[0] = 1.0
    constants[1] = 0.1
    states[1] = 0.055
    states[2] = 0.055
    constants[2] = 10.0
    constants[3] = 0.1
    states[3] = 0.08
    constants[4] = 0.05
    constants[5] = 0.9
    constants[6] = 0.001
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[2] = states[0]/(1.00000+(states[1]*states[3])/constants[3])-constants[2]*states[2]
    rates[3] = ((power(states[1]*states[3], 2.00000))/(1.00000*(constants[6]+power(states[1]*states[3], 2.00000)))+constants[4])-constants[5]*states[3]
    rates[1] = 1.00000*(states[2]-states[1])
    algebraic[0] = custom_piecewise([less(voi , 1.00000), 1.00000 , True, 0.00000])
    rates[0] = 1.00000*((1.00000+algebraic[0])/(1.00000+states[1]/constants[1]))-constants[0]*states[0]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = custom_piecewise([less(voi , 1.00000), 1.00000 , True, 0.00000])
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