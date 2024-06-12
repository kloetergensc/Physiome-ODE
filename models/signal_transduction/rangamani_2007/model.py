# Size of variable arrays:
sizeAlgebraic = 21
sizeStates = 32
sizeConstants = 32
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_constants[0] = "k1 in component constants (second_order_rate_constant)"
    legend_constants[1] = "k2 in component constants (first_order_rate_constant)"
    legend_constants[2] = "k3 in component constants (second_order_rate_constant)"
    legend_constants[3] = "k4 in component constants (first_order_rate_constant)"
    legend_constants[4] = "k5 in component constants (second_order_rate_constant)"
    legend_constants[5] = "k6 in component constants (first_order_rate_constant)"
    legend_constants[6] = "k7 in component constants (second_order_rate_constant)"
    legend_constants[7] = "k8 in component constants (first_order_rate_constant)"
    legend_constants[8] = "k9 in component constants (second_order_rate_constant)"
    legend_constants[9] = "k10 in component constants (first_order_rate_constant)"
    legend_constants[10] = "k110 in component constants (first_order_rate_constant)"
    legend_constants[11] = "k120 in component constants (second_order_rate_constant)"
    legend_constants[12] = "k130 in component constants (first_order_rate_constant)"
    legend_constants[13] = "k140 in component constants (first_order_rate_constant)"
    legend_constants[14] = "p in component constants (first_order_rate_constant)"
    legend_constants[15] = "k15 in component constants (second_order_rate_constant)"
    legend_constants[16] = "k16 in component constants (first_order_rate_constant)"
    legend_constants[17] = "k17 in component constants (first_order_rate_constant)"
    legend_constants[18] = "k18 in component constants (second_order_rate_constant)"
    legend_constants[19] = "k19 in component constants (first_order_rate_constant)"
    legend_constants[20] = "k20 in component constants (first_order_rate_constant)"
    legend_constants[21] = "k21 in component constants (second_order_rate_constant)"
    legend_constants[22] = "k22 in component constants (first_order_rate_constant)"
    legend_constants[23] = "k23 in component constants (first_order_rate_constant)"
    legend_constants[24] = "k24 in component constants (second_order_rate_constant)"
    legend_constants[25] = "k25 in component constants (first_order_rate_constant)"
    legend_constants[26] = "k26 in component constants (first_order_rate_constant)"
    legend_constants[27] = "k28 in component constants (second_order_rate_constant)"
    legend_constants[28] = "k29 in component constants (second_order_rate_constant)"
    legend_constants[29] = "d_6 in component constants (first_order_rate_constant)"
    legend_constants[30] = "k_14 in component constants (flux)"
    legend_constants[31] = "k_m1 in component constants (rate3)"
    legend_algebraic[0] = "J1 in component Js (flux)"
    legend_algebraic[2] = "J2 in component Js (flux)"
    legend_algebraic[4] = "J3 in component Js (flux)"
    legend_algebraic[5] = "J4 in component Js (flux)"
    legend_algebraic[6] = "J5 in component Js (flux)"
    legend_algebraic[7] = "J6 in component Js (flux)"
    legend_algebraic[8] = "J7 in component Js (flux)"
    legend_algebraic[9] = "J8 in component Js (flux)"
    legend_algebraic[10] = "J9 in component Js (flux)"
    legend_algebraic[11] = "J10 in component Js (flux)"
    legend_algebraic[12] = "J11 in component Js (flux)"
    legend_algebraic[13] = "J12 in component Js (flux)"
    legend_algebraic[14] = "J13 in component Js (flux)"
    legend_algebraic[15] = "J14 in component Js (flux)"
    legend_algebraic[16] = "J15 in component Js (flux)"
    legend_algebraic[17] = "J16 in component Js (flux)"
    legend_algebraic[18] = "J17 in component Js (flux)"
    legend_algebraic[19] = "J18 in component Js (flux)"
    legend_algebraic[20] = "J19 in component Js (flux)"
    legend_algebraic[1] = "v14 in component Js (flux)"
    legend_algebraic[3] = "deg_no in component Js (flux)"
    legend_states[0] = "R_NOS2 in component cs (nanomolar)"
    legend_states[1] = "c1 in component cs (nanomolar)"
    legend_states[2] = "c2 in component cs (nanomolar)"
    legend_states[3] = "c3 in component cs (nanomolar)"
    legend_states[4] = "c4 in component cs (nanomolar)"
    legend_states[5] = "c5 in component cs (nanomolar)"
    legend_states[6] = "c6 in component cs (nanomolar)"
    legend_states[7] = "c7 in component cs (nanomolar)"
    legend_states[8] = "c8 in component cs (nanomolar)"
    legend_states[9] = "c9 in component cs (nanomolar)"
    legend_states[10] = "c10 in component cs (nanomolar)"
    legend_states[11] = "c11 in component cs (nanomolar)"
    legend_states[12] = "c12 in component cs (nanomolar)"
    legend_states[13] = "c13 in component cs (nanomolar)"
    legend_states[14] = "c14 in component cs (nanomolar)"
    legend_states[15] = "c15 in component cs (nanomolar)"
    legend_states[16] = "c16 in component cs (nanomolar)"
    legend_states[17] = "c17 in component cs (nanomolar)"
    legend_states[18] = "c18 in component cs (nanomolar)"
    legend_states[19] = "c19 in component cs (nanomolar)"
    legend_states[20] = "c20 in component cs (nanomolar)"
    legend_states[21] = "c21 in component cs (nanomolar)"
    legend_states[22] = "c22 in component cs (nanomolar)"
    legend_states[23] = "c23 in component cs (nanomolar)"
    legend_states[24] = "c24 in component cs (nanomolar)"
    legend_states[25] = "c25 in component cs (nanomolar)"
    legend_states[26] = "c26 in component cs (nanomolar)"
    legend_states[27] = "c27 in component cs (nanomolar)"
    legend_states[28] = "c28 in component cs (nanomolar)"
    legend_states[29] = "c29 in component cs (nanomolar)"
    legend_states[30] = "c30 in component cs (nanomolar)"
    legend_states[31] = "c31 in component cs (nanomolar)"
    legend_rates[0] = "d/dt R_NOS2 in component cs (nanomolar)"
    legend_rates[1] = "d/dt c1 in component cs (nanomolar)"
    legend_rates[2] = "d/dt c2 in component cs (nanomolar)"
    legend_rates[3] = "d/dt c3 in component cs (nanomolar)"
    legend_rates[4] = "d/dt c4 in component cs (nanomolar)"
    legend_rates[5] = "d/dt c5 in component cs (nanomolar)"
    legend_rates[6] = "d/dt c6 in component cs (nanomolar)"
    legend_rates[7] = "d/dt c7 in component cs (nanomolar)"
    legend_rates[8] = "d/dt c8 in component cs (nanomolar)"
    legend_rates[9] = "d/dt c9 in component cs (nanomolar)"
    legend_rates[10] = "d/dt c10 in component cs (nanomolar)"
    legend_rates[11] = "d/dt c11 in component cs (nanomolar)"
    legend_rates[12] = "d/dt c12 in component cs (nanomolar)"
    legend_rates[13] = "d/dt c13 in component cs (nanomolar)"
    legend_rates[14] = "d/dt c14 in component cs (nanomolar)"
    legend_rates[15] = "d/dt c15 in component cs (nanomolar)"
    legend_rates[16] = "d/dt c16 in component cs (nanomolar)"
    legend_rates[17] = "d/dt c17 in component cs (nanomolar)"
    legend_rates[18] = "d/dt c18 in component cs (nanomolar)"
    legend_rates[19] = "d/dt c19 in component cs (nanomolar)"
    legend_rates[20] = "d/dt c20 in component cs (nanomolar)"
    legend_rates[21] = "d/dt c21 in component cs (nanomolar)"
    legend_rates[22] = "d/dt c22 in component cs (nanomolar)"
    legend_rates[23] = "d/dt c23 in component cs (nanomolar)"
    legend_rates[24] = "d/dt c24 in component cs (nanomolar)"
    legend_rates[25] = "d/dt c25 in component cs (nanomolar)"
    legend_rates[26] = "d/dt c26 in component cs (nanomolar)"
    legend_rates[27] = "d/dt c27 in component cs (nanomolar)"
    legend_rates[28] = "d/dt c28 in component cs (nanomolar)"
    legend_rates[29] = "d/dt c29 in component cs (nanomolar)"
    legend_rates[30] = "d/dt c30 in component cs (nanomolar)"
    legend_rates[31] = "d/dt c31 in component cs (nanomolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 185e-6
    constants[1] = 1.25e-6
    constants[2] = 185e-6
    constants[3] = 1.25e-6
    constants[4] = 185e-6
    constants[5] = 1.25e-6
    constants[6] = 185e-6
    constants[7] = 1.25e-6
    constants[8] = 185e-6
    constants[9] = 1.25e-6
    constants[10] = 370e-6
    constants[11] = 14e-6
    constants[12] = 1.25e-6
    constants[13] = 370e-6
    constants[14] = 1750e-6
    constants[15] = 185e-6
    constants[16] = 1.25e-6
    constants[17] = 370e-6
    constants[18] = 500e-6
    constants[19] = 200e-6
    constants[20] = 100e-6
    constants[21] = 100e-6
    constants[22] = 60e-6
    constants[23] = 100000e-6
    constants[24] = 185e-6
    constants[25] = 1.25e-6
    constants[26] = 370e-6
    constants[27] = 500e-6
    constants[28] = 750000e-6
    constants[29] = 2.83e-4
    constants[30] = 41.6
    constants[31] = 10e5
    states[0] = 0.616
    states[1] = 0
    states[2] = 100
    states[3] = 0
    states[4] = 150
    states[5] = 0
    states[6] = 100
    states[7] = 0
    states[8] = 100
    states[9] = 0
    states[10] = 100
    states[11] = 0
    states[12] = 0
    states[13] = 250
    states[14] = 0
    states[15] = 0
    states[16] = 0
    states[17] = 100
    states[18] = 0
    states[19] = 0
    states[20] = 80
    states[21] = 0
    states[22] = 0
    states[23] = 200
    states[24] = 0
    states[25] = 0
    states[26] = 0
    states[27] = 0
    states[28] = 0
    states[29] = 800
    states[30] = 0
    states[31] = 0
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[0] = constants[0]*states[1]*states[2]-constants[1]*states[3]
    rates[1] = -algebraic[0]
    algebraic[1] = (constants[30]*(power(states[16], 2.00000)))/(constants[31]+power(states[16], 2.00000))
    algebraic[3] = constants[29]*states[0]
    rates[0] = algebraic[1]-algebraic[3]
    algebraic[2] = constants[2]*states[3]*states[4]-constants[3]*states[5]
    rates[3] = algebraic[0]-algebraic[2]
    algebraic[4] = constants[4]*states[5]*states[6]-constants[5]*states[7]
    rates[5] = algebraic[2]-algebraic[4]
    algebraic[5] = constants[6]*states[7]*states[8]-constants[7]*states[9]
    rates[7] = algebraic[4]-algebraic[5]
    algebraic[6] = constants[8]*states[10]*states[9]-constants[9]*states[11]
    algebraic[7] = constants[10]*states[11]
    rates[11] = algebraic[6]-algebraic[7]
    algebraic[8] = constants[11]*states[12]*states[13]-constants[12]*states[14]
    rates[12] = -algebraic[8]+algebraic[7]
    algebraic[9] = constants[13]*states[14]
    rates[10] = -algebraic[6]+algebraic[9]
    rates[14] = algebraic[8]-algebraic[9]
    rates[15] = algebraic[9]
    algebraic[10] = constants[15]*states[9]*states[17]-constants[16]*states[18]
    rates[9] = (algebraic[5]-algebraic[6])-algebraic[10]
    algebraic[11] = constants[17]*states[18]
    rates[2] = -algebraic[0]+algebraic[11]+algebraic[7]
    rates[18] = algebraic[10]-algebraic[11]
    algebraic[12] = constants[18]*states[19]*states[20]-constants[19]*states[21]
    rates[19] = algebraic[11]-algebraic[12]
    rates[20] = -algebraic[12]
    algebraic[13] = constants[20]*states[21]
    rates[4] = -algebraic[2]+algebraic[7]+algebraic[13]
    rates[6] = -algebraic[4]+algebraic[7]+algebraic[13]
    rates[8] = -algebraic[5]+algebraic[7]+algebraic[13]
    rates[17] = -algebraic[10]+algebraic[13]
    rates[21] = algebraic[12]-algebraic[13]
    algebraic[14] = constants[21]*states[22]*states[23]-constants[22]*states[24]
    rates[23] = -algebraic[14]
    algebraic[15] = constants[23]*states[24]
    rates[22] = (algebraic[13]-algebraic[14])+algebraic[15]
    rates[24] = algebraic[14]-algebraic[15]
    algebraic[16] = constants[24]*states[25]*states[29]-constants[25]*states[30]
    rates[29] = -algebraic[16]
    algebraic[17] = constants[26]*states[30]
    rates[26] = algebraic[17]
    rates[30] = algebraic[16]-algebraic[17]
    algebraic[20] = constants[28]*states[16]*states[31]
    rates[13] = -algebraic[8]+algebraic[20]
    rates[16] = algebraic[9]-algebraic[20]
    algebraic[19] = constants[27]*states[27]*states[25]
    rates[25] = ((algebraic[15]-algebraic[19])-algebraic[16])+algebraic[17]
    algebraic[18] = constants[14]*states[16]
    rates[27] = algebraic[18]-algebraic[19]
    rates[28] = algebraic[19]
    rates[31] = algebraic[18]-algebraic[20]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = constants[0]*states[1]*states[2]-constants[1]*states[3]
    algebraic[1] = (constants[30]*(power(states[16], 2.00000)))/(constants[31]+power(states[16], 2.00000))
    algebraic[3] = constants[29]*states[0]
    algebraic[2] = constants[2]*states[3]*states[4]-constants[3]*states[5]
    algebraic[4] = constants[4]*states[5]*states[6]-constants[5]*states[7]
    algebraic[5] = constants[6]*states[7]*states[8]-constants[7]*states[9]
    algebraic[6] = constants[8]*states[10]*states[9]-constants[9]*states[11]
    algebraic[7] = constants[10]*states[11]
    algebraic[8] = constants[11]*states[12]*states[13]-constants[12]*states[14]
    algebraic[9] = constants[13]*states[14]
    algebraic[10] = constants[15]*states[9]*states[17]-constants[16]*states[18]
    algebraic[11] = constants[17]*states[18]
    algebraic[12] = constants[18]*states[19]*states[20]-constants[19]*states[21]
    algebraic[13] = constants[20]*states[21]
    algebraic[14] = constants[21]*states[22]*states[23]-constants[22]*states[24]
    algebraic[15] = constants[23]*states[24]
    algebraic[16] = constants[24]*states[25]*states[29]-constants[25]*states[30]
    algebraic[17] = constants[26]*states[30]
    algebraic[20] = constants[28]*states[16]*states[31]
    algebraic[19] = constants[27]*states[27]*states[25]
    algebraic[18] = constants[14]*states[16]
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