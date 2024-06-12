# Size of variable arrays:
sizeAlgebraic = 0
sizeStates = 1
sizeConstants = 19
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_constants[15] = "C in component C (nanomolar)"
    legend_constants[0] = "kf1 in component model_parameters (second_order_rate_constant)"
    legend_constants[12] = "kr1 in component model_parameters (first_order_rate_constant)"
    legend_constants[13] = "k_x1 in component model_parameters (first_order_rate_constant)"
    legend_constants[1] = "kt in component model_parameters (first_order_rate_constant)"
    legend_constants[2] = "ke in component model_parameters (first_order_rate_constant)"
    legend_constants[3] = "L in component model_parameters (nanomolar)"
    legend_constants[16] = "R in component R (nanomolar)"
    legend_constants[14] = "K_X in component D (per_nanomolar)"
    legend_constants[17] = "D in component D (nanomolar)"
    legend_constants[4] = "kx2 in component model_parameters (second_order_rate_constant)"
    legend_constants[5] = "k_x2 in component model_parameters (first_order_rate_constant)"
    legend_constants[6] = "R_initial in component R (nanomolar)"
    legend_constants[7] = "krec in component model_parameters (first_order_rate_constant)"
    legend_constants[8] = "kdeg in component model_parameters (first_order_rate_constant)"
    legend_states[0] = "Ri in component Ri (nanomolar)"
    legend_constants[18] = "signal in component signal (dimensionless)"
    legend_constants[9] = "kappaE in component model_parameters (dimensionless)"
    legend_constants[10] = "Vs in component model_parameters (flux)"
    legend_constants[11] = "KD in component model_parameters (nanomolar)"
    legend_rates[0] = "d/dt Ri in component Ri (nanomolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 0.1
    constants[1] = 0.005
    constants[2] = 0.10
    constants[3] = 0.01
    constants[4] = 4.83
    constants[5] = 0.016
    constants[6] = 2000.0
    constants[7] = 0.0
    constants[8] = 0.05
    states[0] = 200.0
    constants[9] = 0.20
    constants[10] = 10.0
    constants[11] = 1.0
    constants[12] = constants[11]*constants[0]
    constants[13] = 0.0100000*constants[12]
    constants[14] = constants[4]/(constants[5]+constants[13]+constants[2])
    rootfind_0(voi, constants, rates, states, algebraic)
    constants[18] = ((2.00000*constants[17])/200.000)/(constants[9]+(2.00000*constants[17])/200.000)
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = constants[1]*(constants[16]+constants[15])-(constants[7]+constants[8])*states[0]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    return algebraic

initialGuess0 = None
def rootfind_0(voi, constants, rates, states, algebraic):
    """Calculate values of algebraic variables for DAE"""
    from scipy.optimize import fsolve
    global initialGuess0
    if initialGuess0 is None: initialGuess0 = ones(3)*0.1
    if not iterable(voi):
        soln = fsolve(residualSN_0, initialGuess0, args=(algebraic, voi, constants, rates, states), xtol=1E-6)
        initialGuess0 = soln
        constants[15] = soln[0]
        constants[16] = soln[1]
        constants[17] = soln[2]
    else:
        for (i,t) in enumerate(voi):
            soln = fsolve(residualSN_0, initialGuess0, args=(algebraic[:,i], voi[i], constants, rates[:i], states[:,i]), xtol=1E-6)
            initialGuess0 = soln
            constants[15][i] = soln[0]
            constants[16][i] = soln[1]
            constants[17][i] = soln[2]

def residualSN_0(algebraicCandidate, algebraic, voi, constants, rates, states):
    resid = array([0.0] * 3)
    constants[15] = algebraicCandidate[0]
    constants[16] = algebraicCandidate[1]
    constants[17] = algebraicCandidate[2]
    resid[0] = (constants[15]-(constants[0]*constants[3]*constants[16])/(constants[12]+constants[1]+(constants[13]+constants[2])*constants[14]*constants[16]))
    resid[1] = (constants[17]-constants[14]*constants[16]*constants[15])
    resid[2] = (constants[16]-(constants[6]-(constants[15]+2.00000*(constants[2]/constants[1])*(1.00000+constants[7]/constants[8])*constants[17])))
    return resid

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