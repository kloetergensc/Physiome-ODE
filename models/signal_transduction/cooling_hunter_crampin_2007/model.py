# Size of variable arrays:
sizeAlgebraic = 17
sizeStates = 13
sizeConstants = 38
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "t in component all (s)"
    legend_states[0] = "Ca in component all (uM)"
    legend_algebraic[10] = "L in component all (uM)"
    legend_constants[0] = "Ls in component all (uM)"
    legend_constants[1] = "ts in component all (s)"
    legend_states[1] = "Gd in component all (per_um2)"
    legend_states[2] = "Gt in component all (per_um2)"
    legend_states[3] = "R in component all (per_um2)"
    legend_states[4] = "Rl in component all (per_um2)"
    legend_states[5] = "Rg in component all (per_um2)"
    legend_states[6] = "Rlg in component all (per_um2)"
    legend_states[7] = "Rlgp in component all (per_um2)"
    legend_states[8] = "IP3 in component all (uM)"
    legend_constants[2] = "PIP2 in component all (per_um2)"
    legend_states[9] = "Pc in component all (per_um2)"
    legend_states[10] = "Pcg in component all (per_um2)"
    legend_states[11] = "P in component all (per_um2)"
    legend_states[12] = "Pg in component all (per_um2)"
    legend_algebraic[11] = "J1 in component all (per_um2_per_s)"
    legend_constants[3] = "kf1 in component all (per_uM_per_s)"
    legend_constants[31] = "kr1 in component all (per_s)"
    legend_constants[4] = "Kd1 in component all (uM)"
    legend_algebraic[12] = "J2 in component all (per_um2_per_s)"
    legend_constants[5] = "kf2 in component all (um2_per_s)"
    legend_constants[32] = "kr2 in component all (per_s)"
    legend_constants[6] = "Kd2 in component all (per_um2)"
    legend_algebraic[13] = "J3 in component all (per_um2_per_s)"
    legend_constants[7] = "kf3 in component all (um2_per_s)"
    legend_constants[8] = "kr3 in component all (per_s)"
    legend_algebraic[14] = "J4 in component all (per_um2_per_s)"
    legend_constants[9] = "kf4 in component all (per_uM_per_s)"
    legend_constants[33] = "kr4 in component all (per_s)"
    legend_constants[10] = "Kd4 in component all (uM)"
    legend_algebraic[15] = "J5 in component all (per_um2_per_s)"
    legend_constants[11] = "kf5 in component all (per_s)"
    legend_algebraic[16] = "J6 in component all (per_um2_per_s)"
    legend_constants[12] = "kf6 in component all (per_s)"
    legend_algebraic[9] = "J7 in component all (per_um2_per_s)"
    legend_constants[13] = "kf7 in component all (per_s)"
    legend_algebraic[7] = "J8 in component all (per_um2_per_s)"
    legend_constants[14] = "kf8 in component all (per_uM_per_s)"
    legend_constants[15] = "kr8 in component all (per_s)"
    legend_algebraic[8] = "J9 in component all (per_um2_per_s)"
    legend_constants[16] = "kf9 in component all (um2_per_s)"
    legend_constants[17] = "kr9 in component all (per_s)"
    legend_algebraic[6] = "J10 in component all (per_um2_per_s)"
    legend_constants[18] = "kf10 in component all (um2_per_s)"
    legend_constants[19] = "kr10 in component all (per_s)"
    legend_algebraic[4] = "J11 in component all (per_um2_per_s)"
    legend_constants[20] = "kf11 in component all (per_uM_per_s)"
    legend_constants[34] = "kr11 in component all (per_s)"
    legend_constants[21] = "Kd11 in component all (uM)"
    legend_algebraic[2] = "J12 in component all (per_um2_per_s)"
    legend_constants[22] = "kf12 in component all (per_s)"
    legend_algebraic[0] = "J13 in component all (per_um2_per_s)"
    legend_constants[23] = "kf13 in component all (per_s)"
    legend_algebraic[3] = "J14 in component all (per_um2_per_s)"
    legend_constants[24] = "kf14 in component all (per_s)"
    legend_constants[25] = "Km14 in component all (uM)"
    legend_algebraic[5] = "J15 in component all (per_um2_per_s)"
    legend_constants[26] = "kf15 in component all (per_s)"
    legend_constants[27] = "Km15 in component all (uM)"
    legend_algebraic[1] = "J16 in component all (uM_per_s)"
    legend_constants[28] = "kf16 in component all (per_s)"
    legend_constants[37] = "Cpc in component all (uM_um2)"
    legend_constants[35] = "Cc in component all (uM)"
    legend_constants[36] = "Cp in component all (per_um2)"
    legend_constants[29] = "Vc in component all (um3)"
    legend_constants[30] = "Rpc in component all (per_um)"
    legend_rates[11] = "d/dt P in component all (per_um2)"
    legend_rates[12] = "d/dt Pg in component all (per_um2)"
    legend_rates[9] = "d/dt Pc in component all (per_um2)"
    legend_rates[10] = "d/dt Pcg in component all (per_um2)"
    legend_rates[8] = "d/dt IP3 in component all (uM)"
    legend_rates[1] = "d/dt Gd in component all (per_um2)"
    legend_rates[2] = "d/dt Gt in component all (per_um2)"
    legend_rates[0] = "d/dt Ca in component all (uM)"
    legend_rates[3] = "d/dt R in component all (per_um2)"
    legend_rates[4] = "d/dt Rl in component all (per_um2)"
    legend_rates[5] = "d/dt Rg in component all (per_um2)"
    legend_rates[7] = "d/dt Rlgp in component all (per_um2)"
    legend_rates[6] = "d/dt Rlg in component all (per_um2)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.1
    constants[0] = 0.1
    constants[1] = 10
    states[1] = 10000
    states[2] = 0
    states[3] = 13.9
    states[4] = 0
    states[5] = 5.06
    states[6] = 0
    states[7] = 0
    states[8] = 0.015
    constants[2] = 4000
    states[9] = 9.09
    states[10] = 0
    states[11] = 90.9
    states[12] = 0
    constants[3] = 0.0003
    constants[4] = 3e-5
    constants[5] = 2.75e-4
    constants[6] = 27500
    constants[7] = 1
    constants[8] = 0.001
    constants[9] = 0.3
    constants[10] = 3e-5
    constants[11] = 0.0004
    constants[12] = 1
    constants[13] = 0.15
    constants[14] = 0.0167
    constants[15] = 0.0167
    constants[16] = 0.0042
    constants[17] = 1
    constants[18] = 0.042
    constants[19] = 1
    constants[20] = 0.0334
    constants[21] = 0.1
    constants[22] = 6
    constants[23] = 6
    constants[24] = 0.444
    constants[25] = 19.8
    constants[26] = 3.8
    constants[27] = 5
    constants[28] = 1.25
    constants[29] = 2550
    constants[30] = 4.61
    constants[31] = constants[3]*constants[4]
    constants[32] = constants[5]*constants[6]
    constants[33] = constants[9]*constants[10]
    constants[34] = constants[20]*constants[21]
    constants[35] = 1.00000/(constants[29]*602.200)
    constants[36] = 1.00000/(constants[29]*constants[30])
    constants[37] = constants[35]/constants[36]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[3] = (constants[24]*states[9]*constants[2])/(constants[25]/constants[37]+constants[2])
    algebraic[5] = (constants[26]*states[10]*constants[2])/(constants[27]/constants[37]+constants[2])
    algebraic[1] = constants[28]*states[8]
    rates[8] = constants[37]*(algebraic[3]+algebraic[5])-algebraic[1]
    algebraic[6] = constants[18]*states[9]*states[2]-constants[19]*states[10]
    algebraic[4] = constants[20]*states[12]*states[0]-constants[34]*states[10]
    algebraic[2] = constants[22]*states[10]
    rates[10] = (algebraic[6]+algebraic[4])-algebraic[2]
    algebraic[7] = constants[14]*states[11]*states[0]-constants[15]*states[9]
    rates[9] = (algebraic[7]+algebraic[2])-algebraic[6]
    rates[0] = constants[37]*-1.00000*(algebraic[7]+algebraic[4])
    algebraic[8] = constants[16]*states[11]*states[2]-constants[17]*states[12]
    algebraic[0] = constants[23]*states[12]
    rates[11] = algebraic[0]-(algebraic[8]+algebraic[7])
    rates[12] = algebraic[8]-(algebraic[4]+algebraic[0])
    algebraic[10] = custom_piecewise([less(voi , constants[1]+0.150000) & greater_equal(voi , constants[1]), constants[0]/(1.00000+exp(-80.0000*((voi-constants[1])-0.0500000))) , greater_equal(voi , constants[1]+0.150000), constants[0] , True, 0.00000])
    algebraic[11] = constants[3]*states[3]*algebraic[10]-constants[31]*states[4]
    algebraic[12] = constants[5]*states[3]*states[1]-constants[32]*states[5]
    rates[3] = -1.00000*(algebraic[11]+algebraic[12])
    algebraic[13] = constants[7]*states[4]*states[1]-constants[8]*states[6]
    algebraic[9] = constants[13]*states[2]
    rates[1] = (algebraic[9]+algebraic[0]+algebraic[2])-(algebraic[12]+algebraic[13])
    algebraic[14] = constants[9]*algebraic[10]*states[5]-constants[33]*states[6]
    rates[5] = algebraic[12]-algebraic[14]
    algebraic[15] = constants[11]*states[6]
    rates[7] = algebraic[15]
    algebraic[16] = constants[12]*states[6]
    rates[2] = algebraic[16]-(algebraic[9]+algebraic[8]+algebraic[6])
    rates[4] = (algebraic[11]+algebraic[16])-algebraic[13]
    rates[6] = ((algebraic[13]-algebraic[15])+algebraic[14])-algebraic[16]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[3] = (constants[24]*states[9]*constants[2])/(constants[25]/constants[37]+constants[2])
    algebraic[5] = (constants[26]*states[10]*constants[2])/(constants[27]/constants[37]+constants[2])
    algebraic[1] = constants[28]*states[8]
    algebraic[6] = constants[18]*states[9]*states[2]-constants[19]*states[10]
    algebraic[4] = constants[20]*states[12]*states[0]-constants[34]*states[10]
    algebraic[2] = constants[22]*states[10]
    algebraic[7] = constants[14]*states[11]*states[0]-constants[15]*states[9]
    algebraic[8] = constants[16]*states[11]*states[2]-constants[17]*states[12]
    algebraic[0] = constants[23]*states[12]
    algebraic[10] = custom_piecewise([less(voi , constants[1]+0.150000) & greater_equal(voi , constants[1]), constants[0]/(1.00000+exp(-80.0000*((voi-constants[1])-0.0500000))) , greater_equal(voi , constants[1]+0.150000), constants[0] , True, 0.00000])
    algebraic[11] = constants[3]*states[3]*algebraic[10]-constants[31]*states[4]
    algebraic[12] = constants[5]*states[3]*states[1]-constants[32]*states[5]
    algebraic[13] = constants[7]*states[4]*states[1]-constants[8]*states[6]
    algebraic[9] = constants[13]*states[2]
    algebraic[14] = constants[9]*algebraic[10]*states[5]-constants[33]*states[6]
    algebraic[15] = constants[11]*states[6]
    algebraic[16] = constants[12]*states[6]
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