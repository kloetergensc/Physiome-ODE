# Size of variable arrays:
sizeAlgebraic = 7
sizeStates = 20
sizeConstants = 33
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_algebraic[0] = "x1 in component x1 (molar)"
    legend_states[0] = "x2 in component x2 (molar)"
    legend_constants[0] = "k1 in component reaction_constants (second_order_rate_constant)"
    legend_constants[1] = "k_minus1 in component reaction_constants (first_order_rate_constant)"
    legend_constants[20] = "k4 in component reaction_constants (first_order_rate_constant)"
    legend_constants[2] = "k_minus4 in component reaction_constants (first_order_rate_constant)"
    legend_constants[21] = "k_minus3 in component reaction_constants (second_order_rate_constant)"
    legend_states[1] = "x3 in component x3 (molar)"
    legend_states[2] = "x5 in component x5 (molar)"
    legend_states[3] = "x6 in component x6 (molar)"
    legend_constants[3] = "PTP in component reaction_constants (molar)"
    legend_constants[4] = "k3 in component reaction_constants (first_order_rate_constant)"
    legend_states[4] = "x4 in component x4 (molar)"
    legend_constants[22] = "k2 in component reaction_constants (second_order_rate_constant)"
    legend_constants[23] = "k_minus2 in component reaction_constants (first_order_rate_constant)"
    legend_constants[5] = "k4b in component reaction_constants (first_order_rate_constant)"
    legend_constants[6] = "k_minus4b in component reaction_constants (first_order_rate_constant)"
    legend_states[5] = "x7 in component x7 (molar)"
    legend_states[6] = "x8 in component x8 (molar)"
    legend_algebraic[1] = "k5 in component reaction_constants (rate)"
    legend_constants[7] = "k_minus5 in component reaction_constants (first_order_rate_constant)"
    legend_constants[8] = "k6 in component reaction_constants (second_order_rate_constant)"
    legend_states[7] = "x9 in component x9 (molar)"
    legend_constants[9] = "k7 in component reaction_constants (first_order_rate_constant)"
    legend_constants[24] = "k_minus7 in component reaction_constants (second_order_rate_constant)"
    legend_states[8] = "x10 in component x10 (molar)"
    legend_constants[10] = "IRp in component reaction_constants (molar)"
    legend_constants[25] = "k8 in component reaction_constants (second_order_rate_constant)"
    legend_constants[11] = "k_minus8 in component reaction_constants (first_order_rate_constant)"
    legend_states[9] = "x11 in component x11 (molar)"
    legend_states[10] = "x12 in component x12 (molar)"
    legend_states[11] = "x13 in component x13 (percentage)"
    legend_algebraic[2] = "k9 in component reaction_constants (first_order_rate_constant)"
    legend_constants[26] = "k_minus9 in component reaction_constants (second_order_rate_constant)"
    legend_constants[27] = "k10 in component reaction_constants (first_order_rate_constant)"
    legend_constants[12] = "k_minus10 in component reaction_constants (second_order_rate_constant)"
    legend_states[12] = "x14 in component x14 (percentage)"
    legend_states[13] = "x15 in component x15 (percentage)"
    legend_constants[13] = "PTEN in component reaction_constants (molar)"
    legend_constants[14] = "SHIP in component reaction_constants (molar)"
    legend_states[14] = "x16 in component x16 (percentage)"
    legend_algebraic[3] = "k11 in component reaction_constants (first_order_rate_constant)"
    legend_constants[28] = "k_minus11 in component reaction_constants (first_order_rate_constant)"
    legend_states[15] = "x17 in component x17 (percentage)"
    legend_states[16] = "x18 in component x18 (percentage)"
    legend_algebraic[4] = "k12 in component reaction_constants (first_order_rate_constant)"
    legend_constants[29] = "k_minus12 in component reaction_constants (first_order_rate_constant)"
    legend_states[17] = "x19 in component x19 (percentage)"
    legend_states[18] = "x20 in component x20 (percentage)"
    legend_constants[30] = "k13 in component reaction_constants (first_order_rate_constant)"
    legend_constants[15] = "k_minus13 in component reaction_constants (first_order_rate_constant)"
    legend_algebraic[6] = "k13b in component reaction_constants (first_order_rate_constant)"
    legend_constants[31] = "k14 in component reaction_constants (first_order_rate_constant)"
    legend_constants[16] = "k_minus14 in component reaction_constants (first_order_rate_constant)"
    legend_states[19] = "x21 in component x21 (percentage)"
    legend_constants[32] = "k9_basal in component reaction_constants (first_order_rate_constant)"
    legend_constants[17] = "k9_stimulated in component reaction_constants (first_order_rate_constant)"
    legend_algebraic[5] = "effect in component reaction_constants (dimensionless)"
    legend_constants[18] = "APequil in component reaction_constants (dimensionless)"
    legend_constants[19] = "PI3K in component reaction_constants (molar)"
    legend_rates[0] = "d/dt x2 in component x2 (molar)"
    legend_rates[1] = "d/dt x3 in component x3 (molar)"
    legend_rates[4] = "d/dt x4 in component x4 (molar)"
    legend_rates[2] = "d/dt x5 in component x5 (molar)"
    legend_rates[3] = "d/dt x6 in component x6 (molar)"
    legend_rates[5] = "d/dt x7 in component x7 (molar)"
    legend_rates[6] = "d/dt x8 in component x8 (molar)"
    legend_rates[7] = "d/dt x9 in component x9 (molar)"
    legend_rates[8] = "d/dt x10 in component x10 (molar)"
    legend_rates[9] = "d/dt x11 in component x11 (molar)"
    legend_rates[10] = "d/dt x12 in component x12 (molar)"
    legend_rates[11] = "d/dt x13 in component x13 (percentage)"
    legend_rates[12] = "d/dt x14 in component x14 (percentage)"
    legend_rates[13] = "d/dt x15 in component x15 (percentage)"
    legend_rates[14] = "d/dt x16 in component x16 (percentage)"
    legend_rates[15] = "d/dt x17 in component x17 (percentage)"
    legend_rates[16] = "d/dt x18 in component x18 (percentage)"
    legend_rates[17] = "d/dt x19 in component x19 (percentage)"
    legend_rates[18] = "d/dt x20 in component x20 (percentage)"
    legend_rates[19] = "d/dt x21 in component x21 (percentage)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 9e-13
    constants[0] = 6e7
    constants[1] = 0.2
    constants[2] = 0.003
    states[1] = 0
    states[2] = 0
    states[3] = 1e-13
    constants[3] = 1
    constants[4] = 2500
    states[4] = 0
    constants[5] = 2.1e-3
    constants[6] = 2.1e-4
    states[5] = 0
    states[6] = 0
    constants[7] = 1.67e-18
    constants[8] = 0.461
    states[7] = 1e-13
    constants[9] = 4.16
    states[8] = 1e-13
    constants[10] = 8.97e-13
    constants[11] = 10
    states[9] = 1e-13
    states[10] = 0
    states[11] = 0.31
    constants[12] = 2.77
    states[12] = 99.4
    states[13] = 0.29
    constants[13] = 1
    constants[14] = 1
    states[14] = 100
    states[15] = 0
    states[16] = 100
    states[17] = 0
    states[18] = 96
    constants[15] = 0.167
    constants[16] = 0.001155
    states[19] = 4
    constants[17] = 1.39
    constants[18] = 9.09
    constants[19] = 5e-15
    constants[20] = constants[2]/9.00000
    constants[21] = constants[1]/1.00000
    constants[22] = constants[0]
    constants[23] = 100.000*constants[1]
    constants[24] = (2.50000/7.45000)*constants[9]
    constants[25] = ((constants[11]*5.00000)/70.7750)*1.00000e+12
    constants[26] = (94.0000/3.10000)*constants[17]
    constants[27] = (3.10000/2.90000)*constants[12]
    constants[28] = 10.0000*log(2.00000)*1.00000
    constants[29] = 10.0000*log(2.00000)*1.00000
    constants[30] = (4.00000/96.0000)*constants[15]
    constants[31] = constants[16]/96.0000
    constants[32] = (0.310000/99.4000)*constants[26]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[5] = constants[5]*states[4]-(constants[6]*states[5]+constants[8]*constants[3]*states[5])
    rates[6] = constants[5]*states[2]-(constants[6]*states[6]+constants[8]*constants[3]*states[6])
    rates[7] = constants[24]*constants[3]*states[8]-(constants[9]*states[7]*(states[4]+states[2]))/constants[10]
    rates[8] = ((constants[9]*states[7]*(states[4]+states[2]))/constants[10]+constants[11]*states[10])-(constants[24]*constants[3]+constants[25]*states[9])*states[8]
    rates[9] = constants[11]*states[10]-constants[25]*states[8]*states[9]
    rates[10] = constants[25]*states[8]*states[9]-constants[11]*states[10]
    rates[13] = constants[12]*constants[14]*states[11]-constants[27]*states[13]
    algebraic[0] = custom_piecewise([less(voi , 15.0000), 1.00000e-07 , True, 0.00000])
    rates[0] = (constants[1]*states[1]+constants[21]*constants[3]*states[2]+constants[2]*states[3])-(constants[0]*algebraic[0]*states[0]+constants[20]*states[0])
    rates[1] = constants[0]*algebraic[0]*states[0]-(constants[1]*states[1]+constants[4]*states[1])
    rates[4] = (constants[22]*algebraic[0]*states[2]+constants[6]*states[5])-(constants[23]*states[4]+constants[5]*states[4])
    rates[2] = (constants[4]*states[1]+constants[23]*states[4]+constants[6]*states[6])-(constants[22]*algebraic[0]*states[2]+constants[21]*constants[3]*states[2]+constants[5]*states[2])
    algebraic[1] = custom_piecewise([greater(states[3]+states[5]+states[6] , 1.00000e-13), 10.0000*constants[7] , True, 60.0000*constants[7]])
    rates[3] = (algebraic[1]+constants[8]*constants[3]*(states[5]+states[6])+constants[20]*states[0])-(constants[7]*states[3]+constants[2]*states[3])
    algebraic[2] = ((constants[17]-constants[32])*states[10])/constants[19]+constants[32]
    rates[11] = (algebraic[2]*states[12]+constants[27]*states[13])-(constants[26]*constants[13]+constants[12]*constants[14])*states[11]
    rates[12] = constants[26]*constants[13]*states[11]-algebraic[2]*states[12]
    algebraic[3] = (0.100000*constants[28]*(states[11]-0.310000))/(3.10000-0.310000)
    rates[14] = constants[28]*states[15]-algebraic[3]*states[14]
    rates[15] = algebraic[3]*states[14]-constants[28]*states[15]
    algebraic[4] = (0.100000*constants[29]*(states[11]-0.310000))/(3.10000-0.310000)
    rates[16] = constants[29]*states[17]-algebraic[4]*states[16]
    rates[17] = algebraic[4]*states[16]-constants[29]*states[17]
    algebraic[5] = (0.200000*states[15]+0.800000*states[17])/constants[18]
    algebraic[6] = (40.0000/60.0000-4.00000/96.0000)*constants[15]*algebraic[5]
    rates[18] = (constants[15]*states[19]+constants[31])-((constants[30]+algebraic[6])*states[18]+constants[16]*states[18])
    rates[19] = -constants[15]*states[19]+(constants[30]+algebraic[6])*states[18]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = custom_piecewise([less(voi , 15.0000), 1.00000e-07 , True, 0.00000])
    algebraic[1] = custom_piecewise([greater(states[3]+states[5]+states[6] , 1.00000e-13), 10.0000*constants[7] , True, 60.0000*constants[7]])
    algebraic[2] = ((constants[17]-constants[32])*states[10])/constants[19]+constants[32]
    algebraic[3] = (0.100000*constants[28]*(states[11]-0.310000))/(3.10000-0.310000)
    algebraic[4] = (0.100000*constants[29]*(states[11]-0.310000))/(3.10000-0.310000)
    algebraic[5] = (0.200000*states[15]+0.800000*states[17])/constants[18]
    algebraic[6] = (40.0000/60.0000-4.00000/96.0000)*constants[15]*algebraic[5]
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