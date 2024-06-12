# Size of variable arrays:
sizeAlgebraic = 1
sizeStates = 2
sizeConstants = 9
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (hour)"
    legend_states[0] = "S in component S (dimensionless)"
    legend_constants[0] = "rs in component kinetic_parameters (first_order_rate_constant)"
    legend_constants[1] = "epsilon_s in component kinetic_parameters (first_order_rate_constant)"
    legend_constants[2] = "alpha in component kinetic_parameters (first_order_rate_constant)"
    legend_constants[3] = "u in component kinetic_parameters (first_order_rate_constant)"
    legend_constants[4] = "beta in component kinetic_parameters (first_order_rate_constant)"
    legend_algebraic[0] = "phi in component phi (dimensionless)"
    legend_states[1] = "M in component M (dimensionless)"
    legend_constants[5] = "rm in component kinetic_parameters (first_order_rate_constant)"
    legend_constants[6] = "epsilon_m in component kinetic_parameters (first_order_rate_constant)"
    legend_constants[8] = "growth_rate in component phi (first_order_rate_constant)"
    legend_constants[7] = "a in component kinetic_parameters (first_order_rate_constant)"
    legend_rates[0] = "d/dt S in component S (dimensionless)"
    legend_rates[1] = "d/dt M in component M (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.5
    constants[0] = 1
    constants[1] = 0.99
    constants[2] = 0.1
    constants[3] = 0.07
    constants[4] = 0.2
    states[1] = 0.5
    constants[5] = 1.3
    constants[6] = 0.1
    constants[7] = 0.5
    constants[8] = (constants[0]+constants[5])*(1.00000-constants[7]/1.00000)
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[0] = ((states[0]*constants[0])/1.00000)*(1.00000-(constants[3]/1.00000)*((1.00000-(constants[4]*constants[1])/1.00000)-(constants[2]/1.00000)*(1.00000-constants[1]/1.00000)))+((states[1]*constants[5])/1.00000)*(1.00000-(constants[3]/1.00000)*((1.00000-(constants[4]*constants[6])/1.00000)-(constants[2]/1.00000)*(1.00000-constants[6]/1.00000)))
    rates[0] = (constants[0]*states[0]*((1.00000-constants[3]*1.00000)+(constants[4]*constants[1]*constants[3])/1.00000)+((constants[2]*constants[3]*constants[0]*states[0])/1.00000)*(1.00000-constants[1]))-algebraic[0]*states[0]*1.00000
    rates[1] = (constants[5]*states[1]*((1.00000-constants[3]/1.00000)+(constants[4]*constants[6]*constants[3])/1.00000)+((constants[2]*constants[3]*constants[5]*states[1])/1.00000)*(1.00000-constants[6]))-algebraic[0]*states[1]*1.00000
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = ((states[0]*constants[0])/1.00000)*(1.00000-(constants[3]/1.00000)*((1.00000-(constants[4]*constants[1])/1.00000)-(constants[2]/1.00000)*(1.00000-constants[1]/1.00000)))+((states[1]*constants[5])/1.00000)*(1.00000-(constants[3]/1.00000)*((1.00000-(constants[4]*constants[6])/1.00000)-(constants[2]/1.00000)*(1.00000-constants[6]/1.00000)))
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