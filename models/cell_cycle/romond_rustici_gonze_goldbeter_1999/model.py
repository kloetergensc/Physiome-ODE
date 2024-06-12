# Size of variable arrays:
sizeAlgebraic = 4
sizeStates = 6
sizeConstants = 28
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_constants[0] = "H_1 in component parameters (dimensionless)"
    legend_constants[1] = "H_2 in component parameters (dimensionless)"
    legend_constants[2] = "H_3 in component parameters (dimensionless)"
    legend_constants[3] = "H_4 in component parameters (dimensionless)"
    legend_constants[4] = "K_1 in component parameters (dimensionless)"
    legend_constants[5] = "K_2 in component parameters (dimensionless)"
    legend_constants[6] = "K_3 in component parameters (dimensionless)"
    legend_constants[7] = "K_4 in component parameters (dimensionless)"
    legend_constants[8] = "V_M1 in component parameters (first_order_rate_constant)"
    legend_constants[9] = "V_M3 in component parameters (first_order_rate_constant)"
    legend_constants[10] = "U_M1 in component parameters (first_order_rate_constant)"
    legend_constants[11] = "U_M3 in component parameters (first_order_rate_constant)"
    legend_constants[12] = "V_2 in component parameters (first_order_rate_constant)"
    legend_constants[13] = "V_4 in component parameters (first_order_rate_constant)"
    legend_constants[14] = "U_2 in component parameters (first_order_rate_constant)"
    legend_constants[15] = "U_4 in component parameters (first_order_rate_constant)"
    legend_constants[16] = "K_c1 in component parameters (micromolar)"
    legend_constants[17] = "K_c2 in component parameters (micromolar)"
    legend_constants[18] = "K_d1 in component parameters (micromolar)"
    legend_constants[19] = "K_d2 in component parameters (micromolar)"
    legend_constants[20] = "v_d1 in component parameters (flux)"
    legend_constants[21] = "v_d2 in component parameters (flux)"
    legend_constants[22] = "v_i1 in component parameters (flux)"
    legend_constants[23] = "v_i2 in component parameters (flux)"
    legend_constants[24] = "k_d1 in component parameters (first_order_rate_constant)"
    legend_constants[25] = "k_d2 in component parameters (first_order_rate_constant)"
    legend_constants[26] = "K_im1 in component parameters (dimensionless)"
    legend_constants[27] = "K_im2 in component parameters (dimensionless)"
    legend_states[0] = "C_1 in component C_1 (micromolar)"
    legend_states[1] = "M_2 in component M_2 (dimensionless)"
    legend_states[2] = "X_1 in component X_1 (dimensionless)"
    legend_states[3] = "M_1 in component M_1 (dimensionless)"
    legend_algebraic[0] = "V_1 in component V_1 (first_order_rate_constant)"
    legend_algebraic[1] = "V_3 in component V_3 (first_order_rate_constant)"
    legend_states[4] = "C_2 in component C_2 (micromolar)"
    legend_states[5] = "X_2 in component X_2 (dimensionless)"
    legend_algebraic[2] = "U_1 in component U_1 (first_order_rate_constant)"
    legend_algebraic[3] = "U_3 in component U_3 (first_order_rate_constant)"
    legend_rates[0] = "d/dt C_1 in component C_1 (micromolar)"
    legend_rates[3] = "d/dt M_1 in component M_1 (dimensionless)"
    legend_rates[2] = "d/dt X_1 in component X_1 (dimensionless)"
    legend_rates[4] = "d/dt C_2 in component C_2 (micromolar)"
    legend_rates[1] = "d/dt M_2 in component M_2 (dimensionless)"
    legend_rates[5] = "d/dt X_2 in component X_2 (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 0.01
    constants[1] = 0.01
    constants[2] = 0.01
    constants[3] = 0.01
    constants[4] = 0.01
    constants[5] = 0.01
    constants[6] = 0.01
    constants[7] = 0.01
    constants[8] = 0.3
    constants[9] = 0.1
    constants[10] = 0.3
    constants[11] = 0.1
    constants[12] = 0.15
    constants[13] = 0.05
    constants[14] = 0.15
    constants[15] = 0.05
    constants[16] = 0.5
    constants[17] = 0.5
    constants[18] = 0.02
    constants[19] = 0.02
    constants[20] = 0.025
    constants[21] = 0.025
    constants[22] = 0.05
    constants[23] = 0.05
    constants[24] = 0.001
    constants[25] = 0.001
    constants[26] = 0.03
    constants[27] = 0.03
    states[0] = 2
    states[1] = 0
    states[2] = 0
    states[3] = 1
    states[4] = 0
    states[5] = 0
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = ((constants[22]*constants[26])/(constants[26]+states[1])-(constants[20]*states[2]*states[0])/(constants[18]+states[0]))-constants[24]*states[0]
    rates[4] = ((constants[23]*constants[27])/(constants[27]+states[3])-(constants[21]*states[5]*states[4])/(constants[19]+states[4]))-constants[25]*states[4]
    algebraic[0] = (states[0]/(constants[16]+states[0]))*constants[8]
    rates[3] = (algebraic[0]*(1.00000-states[3]))/(constants[4]+(1.00000-states[3]))-(constants[12]*states[3])/(constants[5]+states[3])
    algebraic[1] = states[3]*constants[9]
    rates[2] = (algebraic[1]*(1.00000-states[2]))/(constants[6]+(1.00000-states[2]))-(constants[13]*states[2])/(constants[7]+states[2])
    algebraic[2] = (states[4]/(constants[17]+states[4]))*constants[10]
    rates[1] = (algebraic[2]*(1.00000-states[1]))/(constants[0]+(1.00000-states[1]))-(constants[14]*states[1])/(constants[1]+states[1])
    algebraic[3] = states[1]*constants[11]
    rates[5] = (algebraic[3]*(1.00000-states[5]))/(constants[2]+(1.00000-states[5]))-(constants[15]*states[5])/(constants[3]+states[5])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = (states[0]/(constants[16]+states[0]))*constants[8]
    algebraic[1] = states[3]*constants[9]
    algebraic[2] = (states[4]/(constants[17]+states[4]))*constants[10]
    algebraic[3] = states[1]*constants[11]
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