# Size of variable arrays:
sizeAlgebraic = 1
sizeStates = 4
sizeConstants = 14
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_states[0] = "C in component C (nanomolar)"
    legend_constants[0] = "kf1 in component model_parameters (second_order_rate_constant)"
    legend_constants[12] = "kr1 in component model_parameters (first_order_rate_constant)"
    legend_constants[1] = "kx2 in component model_parameters (second_order_rate_constant)"
    legend_constants[2] = "k_x2 in component model_parameters (first_order_rate_constant)"
    legend_constants[3] = "kt in component model_parameters (first_order_rate_constant)"
    legend_states[1] = "D in component D (nanomolar)"
    legend_constants[4] = "L in component model_parameters (nanomolar)"
    legend_states[2] = "R in component R (nanomolar)"
    legend_constants[13] = "k_x1 in component model_parameters (first_order_rate_constant)"
    legend_constants[5] = "ke in component model_parameters (first_order_rate_constant)"
    legend_constants[6] = "R_initial in component R (nanomolar)"
    legend_constants[7] = "krec in component model_parameters (first_order_rate_constant)"
    legend_states[3] = "Ri in component Ri (nanomolar)"
    legend_constants[8] = "Vs in component model_parameters (flux)"
    legend_constants[9] = "kdeg in component model_parameters (first_order_rate_constant)"
    legend_algebraic[0] = "signal in component signal (dimensionless)"
    legend_constants[10] = "kappaE in component model_parameters (dimensionless)"
    legend_constants[11] = "KD in component model_parameters (nanomolar)"
    legend_rates[0] = "d/dt C in component C (nanomolar)"
    legend_rates[1] = "d/dt D in component D (nanomolar)"
    legend_rates[2] = "d/dt R in component R (nanomolar)"
    legend_rates[3] = "d/dt Ri in component Ri (nanomolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.0
    constants[0] = 0.1
    constants[1] = 4.83
    constants[2] = 0.016
    constants[3] = 0.005
    states[1] = 0.0
    constants[4] = 0.01
    states[2] = 2000.0
    constants[5] = 0.10
    constants[6] = 2000.0
    constants[7] = 0.0
    states[3] = 200.0
    constants[8] = 10.0
    constants[9] = 0.05
    constants[10] = 0.20
    constants[11] = 1.0
    constants[12] = constants[11]*constants[0]
    constants[13] = 0.0100000*constants[12]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = (constants[0]*constants[4]*states[2]+constants[2]*states[1])-(constants[12]+constants[1]*states[2]+constants[3])*states[0]
    rates[1] = constants[1]*states[2]*states[0]-(constants[2]+constants[13]+constants[5])*states[1]
    rates[2] = (constants[8]+constants[12]*states[0]+(constants[2]+2.00000*constants[13])*states[1]+constants[7]*states[3])-(constants[0]*constants[4]+constants[1]*states[0]+constants[3])*states[2]
    rates[3] = constants[3]*(states[2]+states[0])-(constants[7]+constants[9])*states[3]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = ((2.00000*states[1])/constants[6])/(constants[10]+(2.00000*states[1])/constants[6])
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