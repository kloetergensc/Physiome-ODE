# Size of variable arrays:
sizeAlgebraic = 2
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
    legend_voi = "time in component environment (minute)"
    legend_constants[0] = "PMO in component muscle_autoregulatory_local_blood_flow_control (mmHg)"
    legend_constants[6] = "PDO in component M_autoregulatory_driving_force (mmHg)"
    legend_constants[7] = "POE in component M_ST_sensitivity_control (mmHg)"
    legend_constants[1] = "POM in component parameter_values (dimensionless)"
    legend_algebraic[0] = "AMM1 in component M_ST_time_delay_and_limit (dimensionless)"
    legend_constants[2] = "A4K in component parameter_values (minute)"
    legend_constants[3] = "AMM4 in component parameter_values (dimensionless)"
    legend_states[0] = "AMM1T in component M_ST_time_delay_and_limit (dimensionless)"
    legend_constants[8] = "POF in component M_LT_sensitivity_control (mmHg)"
    legend_constants[4] = "POM2 in component parameter_values (dimensionless)"
    legend_states[1] = "AMM2 in component M_LT_time_delay (dimensionless)"
    legend_constants[5] = "A4K2 in component parameter_values (minute)"
    legend_algebraic[1] = "AMM in component global_M_blood_flow_autoregulation_output (dimensionless)"
    legend_rates[0] = "d/dt AMM1T in component M_ST_time_delay_and_limit (dimensionless)"
    legend_rates[1] = "d/dt AMM2 in component M_LT_time_delay (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 38.0666
    constants[1] = 0.04
    constants[2] = 0.1
    constants[3] = 0.005
    states[0] = 1.00269
    constants[4] = 2
    states[1] = 1.09071
    constants[5] = 40000
    constants[6] = constants[0]-38.0000
    constants[7] = constants[6]*constants[1]+1.00000
    constants[8] = constants[4]*constants[6]+1.00000
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = (constants[7]*1.00000-states[0])/constants[2]
    rates[1] = (constants[8]*1.00000-states[1])/constants[5]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = custom_piecewise([less(states[0] , constants[3]), constants[3] , True, states[0]])
    algebraic[1] = algebraic[0]*states[1]
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