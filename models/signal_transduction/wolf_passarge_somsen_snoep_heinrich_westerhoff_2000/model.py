# Size of variable arrays:
sizeAlgebraic = 13
sizeStates = 9
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
    legend_states[0] = "S1 in component S1 (millimolar)"
    legend_constants[0] = "Jo in component glucose_influx_rate (flux)"
    legend_algebraic[3] = "v1 in component v1 (flux)"
    legend_states[1] = "S2 in component S2 (millimolar)"
    legend_algebraic[4] = "v2 in component v2 (flux)"
    legend_states[2] = "S3 in component S3 (millimolar)"
    legend_algebraic[5] = "v3 in component v3 (flux)"
    legend_algebraic[10] = "v8 in component v8 (flux)"
    legend_states[3] = "S4 in component S4 (millimolar)"
    legend_algebraic[6] = "v4 in component v4 (flux)"
    legend_states[4] = "S5 in component S5 (millimolar)"
    legend_algebraic[7] = "v5 in component v5 (flux)"
    legend_states[5] = "S6 in component S6 (millimolar)"
    legend_algebraic[9] = "v6 in component v6 (flux)"
    legend_algebraic[12] = "J in component S6_flux_rate_across_the_plasma_membrane (flux)"
    legend_states[6] = "S6_ex in component S6_ex (millimolar)"
    legend_constants[1] = "phi in component S6_ex (dimensionless)"
    legend_algebraic[11] = "v9 in component v9 (flux)"
    legend_states[7] = "A3 in component A3 (millimolar)"
    legend_algebraic[8] = "v7 in component v7 (flux)"
    legend_constants[2] = "A in component A (millimolar)"
    legend_algebraic[0] = "A2 in component A (millimolar)"
    legend_states[8] = "N2 in component N2 (millimolar)"
    legend_constants[3] = "N in component N (millimolar)"
    legend_algebraic[1] = "N1 in component N (millimolar)"
    legend_constants[4] = "K_i in component v1 (millimolar)"
    legend_constants[5] = "k_1 in component v1 (second_order_rate_constant)"
    legend_constants[6] = "n in component v1 (dimensionless)"
    legend_algebraic[2] = "f_A3 in component v1 (dimensionless)"
    legend_constants[7] = "k_2 in component v2 (first_order_rate_constant)"
    legend_constants[8] = "k_GAPDH_plus in component v3 (second_order_rate_constant)"
    legend_constants[9] = "k_GAPDH_minus in component v3 (second_order_rate_constant)"
    legend_constants[10] = "k_PGK_plus in component v3 (second_order_rate_constant)"
    legend_constants[11] = "k_PGK_minus in component v3 (second_order_rate_constant)"
    legend_constants[12] = "k_4 in component v4 (second_order_rate_constant)"
    legend_constants[13] = "k_5 in component v5 (first_order_rate_constant)"
    legend_constants[14] = "k_6 in component v6 (second_order_rate_constant)"
    legend_constants[15] = "k_7 in component v7 (first_order_rate_constant)"
    legend_constants[16] = "k_8 in component v8 (second_order_rate_constant)"
    legend_constants[17] = "k_9 in component v9 (first_order_rate_constant)"
    legend_constants[18] = "k in component S6_flux_rate_across_the_plasma_membrane (first_order_rate_constant)"
    legend_rates[0] = "d/dt S1 in component S1 (millimolar)"
    legend_rates[1] = "d/dt S2 in component S2 (millimolar)"
    legend_rates[2] = "d/dt S3 in component S3 (millimolar)"
    legend_rates[3] = "d/dt S4 in component S4 (millimolar)"
    legend_rates[4] = "d/dt S5 in component S5 (millimolar)"
    legend_rates[5] = "d/dt S6 in component S6 (millimolar)"
    legend_rates[6] = "d/dt S6_ex in component S6_ex (millimolar)"
    legend_rates[7] = "d/dt A3 in component A3 (millimolar)"
    legend_rates[8] = "d/dt N2 in component N2 (millimolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 1.57981839
    constants[0] = 50
    states[1] = 4.8279999
    states[2] = 0.468657507
    states[3] = 0.589391932
    states[4] = 8.210114438
    states[5] = 0.078042624
    states[6] = 0.025277594
    constants[1] = 0.1
    states[7] = 1.972814237
    constants[2] = 4
    states[8] = 0.384873894
    constants[3] = 1
    constants[4] = 1
    constants[5] = 550
    constants[6] = 4
    constants[7] = 9.8
    constants[8] = 323.8
    constants[9] = 57823.1
    constants[10] = 76411.1
    constants[11] = 23.7
    constants[12] = 80
    constants[13] = 9.7
    constants[14] = 2000
    constants[15] = 28
    constants[16] = 85.7
    constants[17] = 80
    constants[18] = 375
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[2] = power(1.00000+power(states[7]/constants[4], constants[6]), -1.00000)
    algebraic[3] = constants[5]*states[0]*states[7]*algebraic[2]
    rates[0] = constants[0]-algebraic[3]
    algebraic[4] = constants[7]*states[1]
    rates[1] = algebraic[3]-algebraic[4]
    algebraic[0] = constants[2]-states[7]
    algebraic[1] = constants[3]-states[8]
    algebraic[5] = (constants[8]*constants[10]*states[2]*algebraic[1]*algebraic[0]-constants[9]*constants[11]*states[3]*states[7]*states[8])/(constants[9]*states[8]+constants[10]*algebraic[0])
    algebraic[6] = constants[12]*states[3]*algebraic[0]
    rates[3] = algebraic[5]-algebraic[6]
    algebraic[7] = constants[13]*states[4]
    rates[4] = algebraic[6]-algebraic[7]
    algebraic[8] = constants[15]*states[7]
    rates[7] = (algebraic[5]+algebraic[6])-(2.00000*algebraic[3]+algebraic[8])
    algebraic[10] = constants[16]*states[2]*states[8]
    rates[2] = 2.00000*algebraic[4]-(algebraic[5]+algebraic[10])
    algebraic[9] = constants[14]*states[5]*states[8]
    rates[8] = algebraic[5]-(algebraic[9]+algebraic[10])
    algebraic[12] = constants[18]*(states[5]-states[6])
    rates[5] = algebraic[7]-(algebraic[9]+algebraic[12])
    algebraic[11] = constants[17]*states[6]
    rates[6] = constants[1]*algebraic[12]-algebraic[11]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[2] = power(1.00000+power(states[7]/constants[4], constants[6]), -1.00000)
    algebraic[3] = constants[5]*states[0]*states[7]*algebraic[2]
    algebraic[4] = constants[7]*states[1]
    algebraic[0] = constants[2]-states[7]
    algebraic[1] = constants[3]-states[8]
    algebraic[5] = (constants[8]*constants[10]*states[2]*algebraic[1]*algebraic[0]-constants[9]*constants[11]*states[3]*states[7]*states[8])/(constants[9]*states[8]+constants[10]*algebraic[0])
    algebraic[6] = constants[12]*states[3]*algebraic[0]
    algebraic[7] = constants[13]*states[4]
    algebraic[8] = constants[15]*states[7]
    algebraic[10] = constants[16]*states[2]*states[8]
    algebraic[9] = constants[14]*states[5]*states[8]
    algebraic[12] = constants[18]*(states[5]-states[6])
    algebraic[11] = constants[17]*states[6]
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