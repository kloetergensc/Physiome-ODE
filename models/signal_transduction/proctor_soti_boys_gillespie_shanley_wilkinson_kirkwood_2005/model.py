# Size of variable arrays:
sizeAlgebraic = 20
sizeStates = 14
sizeConstants = 24
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_states[0] = "NatP in component NatP (molecules)"
    legend_constants[22] = "v1 in component v1 (flux)"
    legend_algebraic[0] = "v2 in component v2 (flux)"
    legend_algebraic[4] = "v5 in component v5 (flux)"
    legend_states[1] = "MisP in component MisP (molecules)"
    legend_algebraic[1] = "v3 in component v3 (flux)"
    legend_algebraic[3] = "v4 in component v4 (flux)"
    legend_algebraic[5] = "v6 in component v6 (flux)"
    legend_algebraic[6] = "v7 in component v7 (flux)"
    legend_states[2] = "MCom in component MCom (molecules)"
    legend_states[3] = "Hsp90 in component Hsp90 (molecules)"
    legend_algebraic[8] = "v9 in component v9 (flux)"
    legend_algebraic[16] = "v16 in component v16 (flux)"
    legend_algebraic[7] = "v8 in component v8 (flux)"
    legend_algebraic[17] = "v17 in component v17 (flux)"
    legend_states[4] = "HCom in component HCom (molecules)"
    legend_states[5] = "HSF1 in component HSF1 (molecules)"
    legend_algebraic[14] = "v13 in component v13 (flux)"
    legend_algebraic[12] = "v12 in component v12 (flux)"
    legend_algebraic[10] = "v10 in component v10 (flux)"
    legend_algebraic[11] = "v11 in component v11 (flux)"
    legend_states[6] = "DiH in component DiH (molecules)"
    legend_states[7] = "TriH in component TriH (molecules)"
    legend_algebraic[13] = "v14 in component v14 (flux)"
    legend_algebraic[15] = "v15 in component v15 (flux)"
    legend_states[8] = "HSE_TriH in component HSE_TriH (molecules)"
    legend_states[9] = "HSE in component HSE (molecules)"
    legend_states[10] = "AggP in component AggP (molecules)"
    legend_algebraic[9] = "v7b in component v7b (flux)"
    legend_states[11] = "ROS in component ROS (molecules)"
    legend_constants[23] = "v20 in component v20 (flux)"
    legend_algebraic[2] = "v21 in component v21 (flux)"
    legend_states[12] = "ATP in component ATP (molecules)"
    legend_algebraic[18] = "v18 in component v18 (flux)"
    legend_algebraic[19] = "v19 in component v19 (flux)"
    legend_states[13] = "ADP in component ADP (molecules)"
    legend_constants[0] = "k1 in component v1 (flux)"
    legend_constants[1] = "k2 in component v2 (second_order_rate_constant)"
    legend_constants[2] = "k3 in component v3 (second_order_rate_constant)"
    legend_constants[3] = "k4 in component v4 (first_order_rate_constant)"
    legend_constants[4] = "k5 in component v5 (second_order_rate_constant)"
    legend_constants[5] = "k6 in component v6 (second_order_rate_constant)"
    legend_constants[6] = "k7 in component v7 (second_order_rate_constant)"
    legend_constants[7] = "k7 in component v7b (second_order_rate_constant)"
    legend_constants[8] = "k8 in component v8 (second_order_rate_constant)"
    legend_constants[9] = "k9 in component v9 (first_order_rate_constant)"
    legend_constants[10] = "k10 in component v10 (second_order_rate_constant)"
    legend_constants[11] = "k11 in component v11 (second_order_rate_constant)"
    legend_constants[12] = "k12 in component v12 (first_order_rate_constant)"
    legend_constants[13] = "k13 in component v13 (first_order_rate_constant)"
    legend_constants[14] = "k14 in component v14 (second_order_rate_constant)"
    legend_constants[15] = "k15 in component v15 (first_order_rate_constant)"
    legend_constants[16] = "k16 in component v16 (first_order_rate_constant)"
    legend_constants[17] = "k17 in component v17 (second_order_rate_constant)"
    legend_constants[18] = "k18 in component v18 (first_order_rate_constant)"
    legend_constants[19] = "k19 in component v19 (first_order_rate_constant)"
    legend_constants[20] = "k20 in component v20 (flux)"
    legend_constants[21] = "k21 in component v21 (first_order_rate_constant)"
    legend_rates[0] = "d/dt NatP in component NatP (molecules)"
    legend_rates[1] = "d/dt MisP in component MisP (molecules)"
    legend_rates[2] = "d/dt MCom in component MCom (molecules)"
    legend_rates[3] = "d/dt Hsp90 in component Hsp90 (molecules)"
    legend_rates[4] = "d/dt HCom in component HCom (molecules)"
    legend_rates[5] = "d/dt HSF1 in component HSF1 (molecules)"
    legend_rates[6] = "d/dt DiH in component DiH (molecules)"
    legend_rates[7] = "d/dt TriH in component TriH (molecules)"
    legend_rates[8] = "d/dt HSE_TriH in component HSE_TriH (molecules)"
    legend_rates[9] = "d/dt HSE in component HSE (molecules)"
    legend_rates[10] = "d/dt AggP in component AggP (molecules)"
    legend_rates[11] = "d/dt ROS in component ROS (molecules)"
    legend_rates[12] = "d/dt ATP in component ATP (molecules)"
    legend_rates[13] = "d/dt ADP in component ADP (molecules)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 6000000
    states[1] = 0
    states[2] = 0
    states[3] = 300000
    states[4] = 5900
    states[5] = 100
    states[6] = 0
    states[7] = 0
    states[8] = 0
    states[9] = 1
    states[10] = 0
    states[11] = 100
    states[12] = 10000
    states[13] = 1000
    constants[0] = 10
    constants[1] = 0.00002
    constants[2] = 50
    constants[3] = 0.00001
    constants[4] = 4e-6
    constants[5] = 6e-7
    constants[6] = 1e-7
    constants[7] = 1e-7
    constants[8] = 500
    constants[9] = 1
    constants[10] = 0.01
    constants[11] = 100
    constants[12] = 0.5
    constants[13] = 0.5
    constants[14] = 0.05
    constants[15] = 0.08
    constants[16] = 1000
    constants[17] = 8.02e-9
    constants[18] = 12
    constants[19] = 0.02
    constants[20] = 0.1
    constants[21] = 0.00001
    constants[22] = constants[0]
    constants[23] = constants[20]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[0] = constants[1]*states[0]*states[11]
    algebraic[2] = constants[21]*states[11]
    rates[11] = (algebraic[0]+constants[23])-(algebraic[0]+algebraic[2])
    algebraic[4] = constants[4]*states[2]*states[12]
    rates[0] = (constants[22]+algebraic[4])-algebraic[0]
    algebraic[1] = constants[2]*states[1]*states[3]
    algebraic[3] = constants[3]*states[2]
    rates[2] = algebraic[1]-(algebraic[3]+algebraic[4])
    algebraic[5] = constants[5]*states[1]*states[12]
    algebraic[6] = constants[6]*states[1]*states[1]
    rates[1] = (algebraic[0]+algebraic[3])-(algebraic[1]+algebraic[5]+2.00000*algebraic[6])
    algebraic[8] = constants[9]*states[4]
    algebraic[7] = constants[8]*states[5]*states[3]
    rates[4] = algebraic[7]-algebraic[8]
    algebraic[9] = constants[7]*states[1]*states[10]
    rates[10] = (algebraic[6]+2.00000*algebraic[9])-algebraic[9]
    algebraic[14] = constants[13]*states[6]
    algebraic[12] = constants[12]*states[7]
    algebraic[10] = constants[10]*states[5]*states[5]
    algebraic[11] = constants[11]*states[6]*states[5]
    rates[5] = (algebraic[8]+2.00000*algebraic[14]+algebraic[12])-(algebraic[7]+2.00000*algebraic[10]+algebraic[11])
    rates[6] = (algebraic[10]+algebraic[12])-(algebraic[11]+algebraic[14])
    algebraic[13] = constants[14]*states[9]*states[7]
    algebraic[15] = constants[15]*states[8]
    rates[7] = (algebraic[11]+algebraic[15])-(algebraic[12]+algebraic[13])
    rates[9] = algebraic[15]-algebraic[13]
    algebraic[16] = constants[16]*states[8]
    rates[8] = (algebraic[13]+algebraic[16])-(algebraic[15]+algebraic[16])
    algebraic[17] = constants[17]*states[3]*states[12]
    rates[3] = (algebraic[4]+algebraic[3]+algebraic[8]+algebraic[16])-(algebraic[1]+algebraic[7]+algebraic[17])
    algebraic[18] = constants[18]*states[13]
    algebraic[19] = constants[19]*states[12]
    rates[12] = algebraic[18]-(algebraic[4]+algebraic[5]+algebraic[17]+algebraic[19])
    rates[13] = (algebraic[4]+algebraic[5]+algebraic[17]+algebraic[19])-algebraic[18]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = constants[1]*states[0]*states[11]
    algebraic[2] = constants[21]*states[11]
    algebraic[4] = constants[4]*states[2]*states[12]
    algebraic[1] = constants[2]*states[1]*states[3]
    algebraic[3] = constants[3]*states[2]
    algebraic[5] = constants[5]*states[1]*states[12]
    algebraic[6] = constants[6]*states[1]*states[1]
    algebraic[8] = constants[9]*states[4]
    algebraic[7] = constants[8]*states[5]*states[3]
    algebraic[9] = constants[7]*states[1]*states[10]
    algebraic[14] = constants[13]*states[6]
    algebraic[12] = constants[12]*states[7]
    algebraic[10] = constants[10]*states[5]*states[5]
    algebraic[11] = constants[11]*states[6]*states[5]
    algebraic[13] = constants[14]*states[9]*states[7]
    algebraic[15] = constants[15]*states[8]
    algebraic[16] = constants[16]*states[8]
    algebraic[17] = constants[17]*states[3]*states[12]
    algebraic[18] = constants[18]*states[13]
    algebraic[19] = constants[19]*states[12]
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