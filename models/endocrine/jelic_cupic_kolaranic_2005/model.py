# Size of variable arrays:
sizeAlgebraic = 3
sizeStates = 2
sizeConstants = 15
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_algebraic[0] = "time in component environment (second)"
    legend_voi = "tau in component environment (dimensionless)"
    legend_constants[7] = "C_0 in component reaction_constants (per_second)"
    legend_algebraic[1] = "A in component a (molar)"
    legend_states[0] = "a in component a (dimensionless)"
    legend_constants[8] = "alpha in component reaction_constants (dimensionless)"
    legend_constants[9] = "beta in component reaction_constants (dimensionless)"
    legend_constants[10] = "K in component reaction_constants (dimensionless)"
    legend_states[1] = "g in component g (dimensionless)"
    legend_constants[11] = "C_1 in component reaction_constants (molar)"
    legend_algebraic[2] = "G in component g (molar)"
    legend_constants[12] = "gamma in component reaction_constants (dimensionless)"
    legend_constants[13] = "L in component reaction_constants (dimensionless)"
    legend_constants[14] = "C_2 in component reaction_constants (molar)"
    legend_constants[0] = "k2 in component reaction_constants (per_second)"
    legend_constants[1] = "k3 in component reaction_constants (per_second)"
    legend_constants[2] = "k6 in component reaction_constants (per_second)"
    legend_constants[3] = "k7 in component reaction_constants (per_second)"
    legend_constants[4] = "k0 in component reaction_constants (molar_per_second)"
    legend_constants[5] = "k4 in component reaction_constants (per_molar2_per_second)"
    legend_constants[6] = "km in component reaction_constants (molar_per_second)"
    legend_rates[0] = "d/dt a in component a (dimensionless)"
    legend_rates[1] = "d/dt g in component g (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 7.68e-8
    states[1] = 3.43e-8
    constants[0] = 6e-4
    constants[1] = 0.0000048
    constants[2] = 0.000891
    constants[3] = 0.006831
    constants[4] = 8.7831e-11
    constants[5] = 2.1e12
    constants[6] = 6.9001e-14
    constants[7] = constants[0]
    constants[8] = constants[1]/constants[0]
    constants[9] = constants[2]/constants[0]
    constants[10] = power(((power(constants[4], 2.00000))*constants[5])/(power(constants[0], 3.00000)), 1.0/2)
    constants[11] = power(constants[0]/constants[5], 1.0/2)
    constants[12] = constants[3]/constants[0]
    constants[13] = power(((power(constants[6], 2.00000))*constants[5])/(power(constants[0], 3.00000)), 1.0/2)
    constants[14] = power(constants[0]/constants[5], 1.0/2)
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = constants[10]-((1.00000+constants[8]+constants[9])*states[0]+states[0]*(power(states[1], 2.00000)))
    rates[1] = ((1.00000-constants[8])*states[0]+states[0]*(power(states[1], 2.00000)))-(constants[13]+constants[12]*states[1])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = voi/constants[7]
    algebraic[1] = constants[11]*states[0]
    algebraic[2] = constants[14]*states[1]
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