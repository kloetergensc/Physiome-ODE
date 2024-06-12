# Size of variable arrays:
sizeAlgebraic = 0
sizeStates = 5
sizeConstants = 15
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_states[0] = "x in component x (dimensionless)"
    legend_constants[0] = "r1 in component x (rate)"
    legend_constants[1] = "r2 in component x (rate)"
    legend_constants[2] = "c1 in component x (rate)"
    legend_states[1] = "z in component z (dimensionless)"
    legend_states[2] = "y in component y (dimensionless)"
    legend_constants[3] = "r3 in component y (rate)"
    legend_constants[4] = "r4 in component y (rate)"
    legend_constants[5] = "c2 in component y (rate)"
    legend_constants[6] = "c3 in component y (rate)"
    legend_constants[7] = "epsilon in component model_constants (dimensionless)"
    legend_states[3] = "u in component u (dimensionless)"
    legend_constants[8] = "r5 in component z (rate)"
    legend_constants[9] = "r6 in component z (rate)"
    legend_constants[10] = "r7 in component z (rate)"
    legend_constants[11] = "z_ in component z (dimensionless)"
    legend_constants[12] = "y_ in component z (dimensionless)"
    legend_constants[13] = "delta in component z (dimensionless)"
    legend_constants[14] = "omega in component u (rate)"
    legend_states[4] = "v in component u (dimensionless)"
    legend_rates[0] = "d/dt x in component x (dimensionless)"
    legend_rates[2] = "d/dt y in component y (dimensionless)"
    legend_rates[1] = "d/dt z in component z (dimensionless)"
    legend_rates[3] = "d/dt u in component u (dimensionless)"
    legend_rates[4] = "d/dt v in component u (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 5
    constants[0] = 0.15
    constants[1] = 0.12
    constants[2] = 0.1
    states[1] = 1
    states[2] = 0
    constants[3] = 0.05
    constants[4] = 0.03
    constants[5] = 0.1
    constants[6] = 0.005
    constants[7] = 0.1
    states[3] = 1
    constants[8] = 0.09
    constants[9] = 0.1
    constants[10] = 0.05
    constants[11] = 1.01
    constants[12] = 1.08
    constants[13] = 0.01
    constants[14] = 0.05
    states[4] = 0
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = states[1]*(constants[0]*states[2]+-constants[1]*states[0]+constants[2])
    rates[2] = constants[7]*(constants[3]/states[1]+-constants[4]*states[0]+constants[5]+constants[6]*states[3])
    rates[1] = constants[7]*constants[13]*((constants[8]*(states[2]-constants[12])*(constants[11]-states[1])+constants[9]*states[1]*(constants[11]-states[1]))-constants[10]*states[1])
    rates[3] = -constants[14]*states[4]
    rates[4] = constants[14]*states[3]
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