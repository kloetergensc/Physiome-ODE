# Size of variable arrays:
sizeAlgebraic = 23
sizeStates = 13
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
    legend_states[0] = "S1 in component S1 (millimolar)"
    legend_algebraic[13] = "v8 in component v8 (flux)"
    legend_algebraic[15] = "v9 in component v9 (flux)"
    legend_constants[0] = "S in component S (millimolar)"
    legend_algebraic[0] = "S2 in component S (millimolar)"
    legend_states[1] = "sul in component sul (millimolar)"
    legend_algebraic[5] = "v1 in component v1 (flux)"
    legend_algebraic[6] = "v2 in component v2 (flux)"
    legend_states[2] = "aps in component aps (millimolar)"
    legend_algebraic[7] = "v3 in component v3 (flux)"
    legend_states[3] = "pap in component pap (millimolar)"
    legend_algebraic[8] = "v4 in component v4 (flux)"
    legend_states[4] = "hyd in component hyd (millimolar)"
    legend_algebraic[9] = "v5 in component v5 (flux)"
    legend_algebraic[11] = "v17 in component v17 (flux)"
    legend_states[5] = "cys in component cys (millimolar)"
    legend_algebraic[12] = "v6 in component v6 (flux)"
    legend_states[6] = "eth in component eth (millimolar)"
    legend_constants[1] = "v13 in component v13 (flux)"
    legend_algebraic[10] = "v7 in component v7 (flux)"
    legend_states[7] = "aco in component aco (millimolar)"
    legend_algebraic[14] = "v15 in component v15 (flux)"
    legend_states[8] = "oxy in component oxy (millimolar)"
    legend_constants[2] = "v10 in component v10 (flux)"
    legend_algebraic[18] = "v11A in component v11A (flux)"
    legend_algebraic[19] = "v14 in component v14 (flux)"
    legend_states[9] = "A3_c in component A3_c (millimolar)"
    legend_algebraic[21] = "v12 in component v12 (flux)"
    legend_algebraic[22] = "v16 in component v16 (flux)"
    legend_states[10] = "A3_m in component A3_m (millimolar)"
    legend_algebraic[20] = "v11B in component v11B (flux)"
    legend_constants[3] = "A_c in component A_c (millimolar)"
    legend_algebraic[1] = "A2_c in component A_c (millimolar)"
    legend_constants[4] = "A_m in component A_m (millimolar)"
    legend_algebraic[2] = "A2_m in component A_m (millimolar)"
    legend_states[11] = "N2 in component N2 (millimolar)"
    legend_constants[5] = "N in component N (millimolar)"
    legend_algebraic[3] = "N1 in component N (millimolar)"
    legend_states[12] = "oah in component oah (millimolar)"
    legend_algebraic[17] = "v18 in component v18 (flux)"
    legend_constants[6] = "v0 in component v1 (flux)"
    legend_algebraic[4] = "f1 in component v1 (dimensionless)"
    legend_constants[7] = "n in component v1 (dimensionless)"
    legend_constants[8] = "Kc in component v1 (millimolar)"
    legend_constants[9] = "k2 in component v2 (second_order_rate_constant)"
    legend_constants[10] = "k3 in component v3 (second_order_rate_constant)"
    legend_constants[11] = "k4 in component v4 (second_order_rate_constant)"
    legend_constants[12] = "k5 in component v5 (second_order_rate_constant)"
    legend_constants[13] = "k6 in component v6 (first_order_rate_constant)"
    legend_constants[14] = "k7 in component v7 (second_order_rate_constant)"
    legend_constants[15] = "k8 in component v8 (second_order_rate_constant)"
    legend_constants[16] = "k9 in component v9 (second_order_rate_constant)"
    legend_constants[17] = "alpha in component v11A (dimensionless)"
    legend_algebraic[16] = "f2 in component v11A (dimensionless)"
    legend_constants[18] = "m in component v11A (dimensionless)"
    legend_constants[19] = "KH in component v11A (millimolar)"
    legend_constants[20] = "k11 in component v11A (first_order_rate_constant)"
    legend_constants[21] = "KA in component v11B (millimolar)"
    legend_constants[22] = "k12 in component v12 (first_order_rate_constant)"
    legend_constants[23] = "k14 in component v14 (first_order_rate_constant)"
    legend_constants[24] = "k15 in component v15 (first_order_rate_constant)"
    legend_constants[25] = "k16 in component v16 (second_order_rate_constant)"
    legend_constants[26] = "k17 in component v17 (first_order_rate_constant)"
    legend_constants[27] = "k18 in component v18 (first_order_rate_constant)"
    legend_rates[0] = "d/dt S1 in component S1 (millimolar)"
    legend_rates[1] = "d/dt sul in component sul (millimolar)"
    legend_rates[2] = "d/dt aps in component aps (millimolar)"
    legend_rates[3] = "d/dt pap in component pap (millimolar)"
    legend_rates[4] = "d/dt hyd in component hyd (millimolar)"
    legend_rates[5] = "d/dt cys in component cys (millimolar)"
    legend_rates[6] = "d/dt eth in component eth (millimolar)"
    legend_rates[7] = "d/dt aco in component aco (millimolar)"
    legend_rates[8] = "d/dt oxy in component oxy (millimolar)"
    legend_rates[9] = "d/dt A3_c in component A3_c (millimolar)"
    legend_rates[10] = "d/dt A3_m in component A3_m (millimolar)"
    legend_rates[11] = "d/dt N2 in component N2 (millimolar)"
    legend_rates[12] = "d/dt oah in component oah (millimolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 1.892468811
    constants[0] = 2
    states[1] = 0.012871557
    states[2] = 0.053366179
    states[3] = 0.082791522
    states[4] = 0.159317029
    states[5] = 0.805117739
    states[6] = 10.25246999
    constants[1] = 4
    states[7] = 1.139360412
    states[8] = 6.146549084
    constants[2] = 80
    states[9] = 1.570847601
    states[10] = 1.835158493
    constants[3] = 2
    constants[4] = 2
    states[11] = 1.933616149
    constants[5] = 2
    states[12] = 5.571685096
    constants[6] = 1.6
    constants[7] = 4
    constants[8] = 0.1
    constants[9] = 0.2
    constants[10] = 0.2
    constants[11] = 0.2
    constants[12] = 0.1
    constants[13] = 0.12
    constants[14] = 10
    constants[15] = 10
    constants[16] = 10
    constants[17] = 0.1
    constants[18] = 4
    constants[19] = 0.5
    constants[20] = 10
    constants[21] = 1
    constants[22] = 5
    constants[23] = 10
    constants[24] = 5
    constants[25] = 10
    constants[26] = 0.02
    constants[27] = 1
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[4] = power(1.00000+power(states[5]/constants[8], constants[7]), -1.00000)
    algebraic[5] = constants[6]*algebraic[4]
    algebraic[6] = constants[9]*states[1]*states[9]
    rates[1] = algebraic[5]-algebraic[6]
    algebraic[7] = constants[10]*states[2]*states[9]
    rates[2] = algebraic[6]-algebraic[7]
    algebraic[8] = constants[11]*states[3]*states[11]
    rates[3] = algebraic[7]-algebraic[8]
    algebraic[9] = constants[12]*states[4]*states[12]
    algebraic[11] = constants[26]*states[4]
    rates[4] = algebraic[8]-(algebraic[9]+algebraic[11])
    algebraic[12] = constants[13]*states[5]
    rates[5] = algebraic[9]-algebraic[12]
    algebraic[3] = constants[5]-states[11]
    algebraic[10] = constants[14]*states[6]*algebraic[3]
    rates[6] = constants[1]-algebraic[10]
    algebraic[0] = constants[0]-states[0]
    algebraic[13] = constants[15]*states[7]*algebraic[0]
    algebraic[15] = constants[16]*states[0]*algebraic[3]
    rates[0] = algebraic[13]-algebraic[15]
    algebraic[14] = constants[24]*states[7]
    rates[7] = algebraic[10]-(algebraic[13]+algebraic[14])
    algebraic[17] = constants[27]*states[12]
    rates[12] = algebraic[14]-(algebraic[9]+algebraic[17])
    algebraic[16] = power(1.00000+power(states[4]/constants[19], constants[18]), -1.00000)
    algebraic[18] = (constants[20]*states[11]*states[8]*algebraic[16])/(constants[17]*states[11]+states[8])
    rates[11] = (2.00000*algebraic[10]+4.00000*algebraic[15])-(3.00000*algebraic[8]+algebraic[18])
    algebraic[19] = constants[23]*states[8]
    rates[8] = constants[2]-(algebraic[18]+algebraic[19])
    algebraic[21] = constants[22]*states[9]
    algebraic[1] = constants[3]-states[9]
    algebraic[22] = constants[25]*states[10]*algebraic[1]
    rates[9] = algebraic[22]-(algebraic[6]+algebraic[7]+algebraic[21])
    algebraic[2] = constants[4]-states[10]
    algebraic[20] = (3.00000*algebraic[18]*algebraic[2])/(constants[21]+algebraic[2])
    rates[10] = algebraic[20]-algebraic[22]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[4] = power(1.00000+power(states[5]/constants[8], constants[7]), -1.00000)
    algebraic[5] = constants[6]*algebraic[4]
    algebraic[6] = constants[9]*states[1]*states[9]
    algebraic[7] = constants[10]*states[2]*states[9]
    algebraic[8] = constants[11]*states[3]*states[11]
    algebraic[9] = constants[12]*states[4]*states[12]
    algebraic[11] = constants[26]*states[4]
    algebraic[12] = constants[13]*states[5]
    algebraic[3] = constants[5]-states[11]
    algebraic[10] = constants[14]*states[6]*algebraic[3]
    algebraic[0] = constants[0]-states[0]
    algebraic[13] = constants[15]*states[7]*algebraic[0]
    algebraic[15] = constants[16]*states[0]*algebraic[3]
    algebraic[14] = constants[24]*states[7]
    algebraic[17] = constants[27]*states[12]
    algebraic[16] = power(1.00000+power(states[4]/constants[19], constants[18]), -1.00000)
    algebraic[18] = (constants[20]*states[11]*states[8]*algebraic[16])/(constants[17]*states[11]+states[8])
    algebraic[19] = constants[23]*states[8]
    algebraic[21] = constants[22]*states[9]
    algebraic[1] = constants[3]-states[9]
    algebraic[22] = constants[25]*states[10]*algebraic[1]
    algebraic[2] = constants[4]-states[10]
    algebraic[20] = (3.00000*algebraic[18]*algebraic[2])/(constants[21]+algebraic[2])
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