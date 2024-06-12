# Size of variable arrays:
sizeAlgebraic = 4
sizeStates = 10
sizeConstants = 14
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (day)"
    legend_states[0] = "S in component S (dimensionless)"
    legend_constants[0] = "r in component S (first_order_rate_constant)"
    legend_constants[1] = "epsilon in component S (dimensionless)"
    legend_algebraic[0] = "H in component S (dimensionless)"
    legend_constants[2] = "d in component kinetic_parameters (first_order_rate_constant)"
    legend_constants[3] = "g in component memory_duration (first_order_rate_constant)"
    legend_constants[4] = "beta_1 in component kinetic_parameters (first_order_rate_constant)"
    legend_constants[5] = "beta_2 in component kinetic_parameters (first_order_rate_constant)"
    legend_states[1] = "R_1 in component R1 (dimensionless)"
    legend_states[2] = "R_2 in component R2 (dimensionless)"
    legend_states[3] = "R_12 in component R12 (dimensionless)"
    legend_states[4] = "P_1 in component P1 (dimensionless)"
    legend_states[5] = "P_2 in component P2 (dimensionless)"
    legend_states[6] = "I_1 in component I1 (dimensionless)"
    legend_states[7] = "I_2 in component I2 (dimensionless)"
    legend_states[8] = "I_12 in component I12 (dimensionless)"
    legend_states[9] = "I_21 in component I21 (dimensionless)"
    legend_algebraic[1] = "P in component S (dimensionless)"
    legend_constants[6] = "a_1 in component kinetic_parameters (first_order_rate_constant)"
    legend_constants[7] = "alpha_1 in component kinetic_parameters (first_order_rate_constant)"
    legend_constants[8] = "a_2 in component kinetic_parameters (first_order_rate_constant)"
    legend_constants[9] = "alpha_2 in component kinetic_parameters (first_order_rate_constant)"
    legend_constants[10] = "k_1 in component P1 (first_order_rate_constant)"
    legend_constants[11] = "u in component kinetic_parameters (first_order_rate_constant)"
    legend_algebraic[2] = "log_P1 in component P1 (dimensionless)"
    legend_constants[12] = "k_2 in component P2 (first_order_rate_constant)"
    legend_algebraic[3] = "log_P2 in component P2 (dimensionless)"
    legend_constants[13] = "G in component memory_duration (dimensionless)"
    legend_rates[0] = "d/dt S in component S (dimensionless)"
    legend_rates[6] = "d/dt I_1 in component I1 (dimensionless)"
    legend_rates[7] = "d/dt I_2 in component I2 (dimensionless)"
    legend_rates[8] = "d/dt I_12 in component I12 (dimensionless)"
    legend_rates[9] = "d/dt I_21 in component I21 (dimensionless)"
    legend_rates[1] = "d/dt R_1 in component R1 (dimensionless)"
    legend_rates[2] = "d/dt R_2 in component R2 (dimensionless)"
    legend_rates[3] = "d/dt R_12 in component R12 (dimensionless)"
    legend_rates[4] = "d/dt P_1 in component P1 (dimensionless)"
    legend_rates[5] = "d/dt P_2 in component P2 (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 100
    constants[0] = 0.5
    constants[1] = 0.1
    constants[2] = 0.01
    constants[3] = 0.01
    constants[4] = 1
    constants[5] = 1
    states[1] = 0
    states[2] = 0
    states[3] = 0
    states[4] = 1
    states[5] = 1
    states[6] = 0
    states[7] = 0
    states[8] = 0
    states[9] = 0
    constants[6] = 0.03
    constants[7] = 0.1
    constants[8] = 1
    constants[9] = 0.1
    constants[10] = 1
    constants[11] = 0.5
    constants[12] = 1
    constants[13] = 1.00000/constants[3]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[6] = (constants[4]*states[0]*states[4]-constants[6]*states[6])-constants[7]*states[6]
    rates[7] = (constants[5]*states[0]*states[5]-constants[8]*states[7])-constants[9]*states[7]
    rates[8] = (constants[5]*states[1]*states[5]-constants[8]*states[8])-constants[9]*states[8]
    rates[9] = (constants[4]*states[2]*states[4]-constants[6]*states[9])-constants[7]*states[9]
    rates[1] = ((constants[7]*states[6]-constants[2]*states[1])-constants[3]*states[1])-constants[5]*states[1]*states[5]
    rates[2] = ((constants[9]*states[7]-constants[2]*states[2])-constants[3]*states[2])-constants[4]*states[2]*states[4]
    rates[3] = ((constants[9]*states[8]+constants[7]*states[9])-constants[2]*states[3])-constants[3]*states[3]
    rates[4] = constants[10]*(states[6]+states[9])-constants[11]*states[4]
    rates[5] = constants[12]*(states[7]+states[8])-constants[11]*states[5]
    algebraic[0] = states[0]+states[6]+states[1]+states[7]+states[2]+states[8]+states[9]+states[3]
    rates[0] = ((((constants[0]*algebraic[0])/(constants[1]*algebraic[0]+1.00000)-constants[2]*states[0])-constants[4]*states[0]*states[4])-constants[5]*states[0]*states[5])+constants[3]*(states[1]+states[2]+states[3])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = states[0]+states[6]+states[1]+states[7]+states[2]+states[8]+states[9]+states[3]
    algebraic[1] = states[4]+states[5]
    algebraic[2] = log(states[4], 10)
    algebraic[3] = log(states[5], 10)
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