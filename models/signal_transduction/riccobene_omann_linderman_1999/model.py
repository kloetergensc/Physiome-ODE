# Size of variable arrays:
sizeAlgebraic = 0
sizeStates = 8
sizeConstants = 11
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_states[0] = "R in component R (molar)"
    legend_constants[0] = "L in component L (molar)"
    legend_states[1] = "LR in component LR (molar)"
    legend_states[2] = "R_star in component R_star (molar)"
    legend_constants[1] = "kf in component model_parameters (second_order_rate_constant)"
    legend_constants[2] = "kr in component model_parameters (first_order_rate_constant)"
    legend_constants[3] = "kfR in component model_parameters (first_order_rate_constant)"
    legend_constants[4] = "Kact in component model_parameters (dimensionless)"
    legend_states[3] = "LR_star in component LR_star (molar)"
    legend_constants[5] = "alpha in component model_parameters (dimensionless)"
    legend_constants[6] = "kds in component model_parameters (first_order_rate_constant)"
    legend_states[4] = "LR_ds in component LR_ds (molar)"
    legend_states[5] = "R_ds in component R_ds (molar)"
    legend_constants[7] = "kf2 in component model_parameters (second_order_rate_constant)"
    legend_constants[8] = "kr2 in component model_parameters (first_order_rate_constant)"
    legend_states[6] = "G_star in component G_star (molar)"
    legend_states[7] = "G in component G (molar)"
    legend_constants[9] = "ka in component model_parameters (second_order_rate_constant)"
    legend_constants[10] = "ki in component model_parameters (first_order_rate_constant)"
    legend_rates[0] = "d/dt R in component R (molar)"
    legend_rates[2] = "d/dt R_star in component R_star (molar)"
    legend_rates[1] = "d/dt LR in component LR (molar)"
    legend_rates[3] = "d/dt LR_star in component LR_star (molar)"
    legend_rates[4] = "d/dt LR_ds in component LR_ds (molar)"
    legend_rates[5] = "d/dt R_ds in component R_ds (molar)"
    legend_rates[6] = "d/dt G_star in component G_star (molar)"
    legend_rates[7] = "d/dt G in component G (molar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.01
    constants[0] = 1E-12
    states[1] = 0.01
    states[2] = 0.01
    constants[1] = 8.4E7
    constants[2] = 0.37
    constants[3] = 10
    constants[4] = 1E-4
    states[3] = 0.01
    constants[5] = 1E1
    constants[6] = 1E-4
    states[4] = 0.01
    states[5] = 0.01
    constants[7] = 8.4E7
    constants[8] = 4.6E-3
    states[6] = 0.01
    states[7] = 0.01
    constants[9] = 1E-7
    constants[10] = 2E-1
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = (constants[2]*states[1]+(constants[3]/constants[4])*states[2])-(constants[1]*constants[0]*states[0]+constants[3]*states[0])
    rates[2] = (constants[2]*states[3]+constants[3]*states[0])-(constants[5]*constants[1]*constants[0]*states[2]+(constants[3]/constants[4])*states[2])
    rates[1] = (constants[1]*constants[0]*states[0]+(constants[3]/(constants[5]*constants[4]))*states[3])-(constants[2]*states[1]+constants[3]*states[1])
    rates[3] = (constants[3]*states[1]+constants[5]*constants[1]*constants[0]*states[2])-((constants[3]/(constants[5]*constants[4]))*states[3]+constants[6]*states[3]+constants[2]*states[3])
    rates[4] = (constants[6]*states[3]+constants[7]*constants[0]*states[5])-constants[8]*states[4]
    rates[5] = constants[8]*states[4]-constants[7]*constants[0]*states[5]
    rates[6] = constants[9]*states[7]*(states[3]+states[2])-constants[10]*states[6]
    rates[7] = constants[10]*states[6]-constants[9]*states[7]*(states[3]+states[2])
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