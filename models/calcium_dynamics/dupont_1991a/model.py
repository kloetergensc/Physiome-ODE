# Size of variable arrays:
sizeAlgebraic = 2
sizeStates = 2
sizeConstants = 13
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_constants[0] = "v0 in component parameters (micromolar_s)"
    legend_constants[1] = "v1 in component parameters (micromolar_s)"
    legend_algebraic[0] = "v2 in component parameters (micromolar_s)"
    legend_algebraic[1] = "v3 in component parameters (micromolar_s)"
    legend_constants[2] = "beta in component parameters (dimensionless)"
    legend_constants[3] = "VM2 in component parameters (micromolar_s)"
    legend_constants[4] = "VM3 in component parameters (micromolar_s)"
    legend_constants[5] = "KR in component parameters (micromolar)"
    legend_constants[6] = "KA in component parameters (micromolar)"
    legend_constants[7] = "kf in component parameters (per_second)"
    legend_constants[8] = "k in component parameters (per_second)"
    legend_constants[9] = "K2 in component parameters (micromolar)"
    legend_constants[10] = "n in component parameters (dimensionless)"
    legend_constants[11] = "m in component parameters (dimensionless)"
    legend_constants[12] = "p in component parameters (dimensionless)"
    legend_states[0] = "Z in component cytosol (micromolar)"
    legend_states[1] = "Y in component insensitive_pool (micromolar)"
    legend_rates[0] = "d/dt Z in component cytosol (micromolar)"
    legend_rates[1] = "d/dt Y in component insensitive_pool (micromolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 1
    constants[1] = 7.3
    constants[2] = 0.6
    constants[3] = 65
    constants[4] = 500
    constants[5] = 2
    constants[6] = 0.9
    constants[7] = 1
    constants[8] = 10
    constants[9] = 1
    constants[10] = 2
    constants[11] = 2
    constants[12] = 4
    states[0] = 0.52
    states[1] = 0.93
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[0] = (constants[3]*(power(states[0], constants[10])))/(power(constants[9], constants[10])+power(states[0], constants[10]))
    algebraic[1] = constants[4]*((power(states[1], constants[11]))/(power(constants[5], constants[11])+power(states[1], constants[11])))*((power(states[0], constants[12]))/(power(constants[6], constants[12])+power(states[0], constants[12])))
    rates[0] = (((constants[0]+constants[1]*constants[2])-algebraic[0])+algebraic[1]+constants[7]*states[1])-constants[8]*states[0]
    rates[1] = (algebraic[0]-algebraic[1])-constants[7]*states[1]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = (constants[3]*(power(states[0], constants[10])))/(power(constants[9], constants[10])+power(states[0], constants[10]))
    algebraic[1] = constants[4]*((power(states[1], constants[11]))/(power(constants[5], constants[11])+power(states[1], constants[11])))*((power(states[0], constants[12]))/(power(constants[6], constants[12])+power(states[0], constants[12])))
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