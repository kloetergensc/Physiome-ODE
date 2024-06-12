# Size of variable arrays:
sizeAlgebraic = 1
sizeStates = 10
sizeConstants = 38
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (hour)"
    legend_states[0] = "MP in component MP (nanomolar)"
    legend_constants[0] = "vsP in component MP (flux)"
    legend_constants[1] = "vmP in component MP (flux)"
    legend_constants[2] = "KmP in component MP (nanomolar)"
    legend_constants[3] = "KIP in component MP (nanomolar)"
    legend_constants[4] = "kd in component parameters (first_order_rate_constant)"
    legend_states[1] = "CN in component CN (nanomolar)"
    legend_constants[5] = "n in component parameters (dimensionless)"
    legend_states[2] = "P0 in component P0 (nanomolar)"
    legend_constants[6] = "ksP in component P0 (first_order_rate_constant)"
    legend_constants[7] = "V1P in component parameters (flux)"
    legend_constants[8] = "V2P in component parameters (flux)"
    legend_constants[9] = "K1P in component parameters (nanomolar)"
    legend_constants[10] = "K2P in component parameters (nanomolar)"
    legend_states[3] = "P1 in component P1 (nanomolar)"
    legend_constants[11] = "V3P in component parameters (flux)"
    legend_constants[12] = "V4P in component parameters (flux)"
    legend_constants[13] = "K3P in component parameters (nanomolar)"
    legend_constants[14] = "K4P in component parameters (nanomolar)"
    legend_states[4] = "P2 in component P2 (nanomolar)"
    legend_constants[15] = "vdP in component P2 (flux)"
    legend_constants[16] = "KdP in component P2 (nanomolar)"
    legend_algebraic[0] = "Pt in component P2 (nanomolar)"
    legend_constants[17] = "k3 in component parameters (second_order_rate_constant)"
    legend_constants[18] = "k4 in component parameters (first_order_rate_constant)"
    legend_states[5] = "T2 in component T2 (nanomolar)"
    legend_states[6] = "C in component C (nanomolar)"
    legend_states[7] = "MT in component MT (nanomolar)"
    legend_constants[19] = "vsT in component MT (flux)"
    legend_constants[20] = "vmT in component MT (flux)"
    legend_constants[21] = "KmT in component MT (nanomolar)"
    legend_constants[22] = "KIT in component MT (nanomolar)"
    legend_states[8] = "T0 in component T0 (nanomolar)"
    legend_constants[23] = "ksT in component T0 (first_order_rate_constant)"
    legend_constants[24] = "V1T in component parameters (flux)"
    legend_constants[25] = "V2T in component parameters (flux)"
    legend_constants[26] = "K1T in component parameters (nanomolar)"
    legend_constants[27] = "K2T in component parameters (nanomolar)"
    legend_states[9] = "T1 in component T1 (nanomolar)"
    legend_constants[28] = "V3T in component parameters (flux)"
    legend_constants[29] = "V4T in component parameters (flux)"
    legend_constants[30] = "K3T in component parameters (nanomolar)"
    legend_constants[31] = "K4T in component parameters (nanomolar)"
    legend_constants[32] = "vdT in component T2 (flux)"
    legend_constants[33] = "KdT in component T2 (nanomolar)"
    legend_constants[34] = "kdC in component C (first_order_rate_constant)"
    legend_constants[35] = "k1 in component parameters (first_order_rate_constant)"
    legend_constants[36] = "k2 in component parameters (first_order_rate_constant)"
    legend_constants[37] = "kdN in component CN (first_order_rate_constant)"
    legend_rates[0] = "d/dt MP in component MP (nanomolar)"
    legend_rates[2] = "d/dt P0 in component P0 (nanomolar)"
    legend_rates[3] = "d/dt P1 in component P1 (nanomolar)"
    legend_rates[4] = "d/dt P2 in component P2 (nanomolar)"
    legend_rates[7] = "d/dt MT in component MT (nanomolar)"
    legend_rates[8] = "d/dt T0 in component T0 (nanomolar)"
    legend_rates[9] = "d/dt T1 in component T1 (nanomolar)"
    legend_rates[5] = "d/dt T2 in component T2 (nanomolar)"
    legend_rates[6] = "d/dt C in component C (nanomolar)"
    legend_rates[1] = "d/dt CN in component CN (nanomolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.1
    constants[0] = 1
    constants[1] = 0.7
    constants[2] = 0.2
    constants[3] = 1.0
    constants[4] = 0.01
    states[1] = 1.25
    constants[5] = 4.0
    states[2] = 0.1
    constants[6] = 0.9
    constants[7] = 8.0
    constants[8] = 1.0
    constants[9] = 2.0
    constants[10] = 2.0
    states[3] = 0.1
    constants[11] = 8.0
    constants[12] = 1.0
    constants[13] = 2.0
    constants[14] = 1
    states[4] = 0.1
    constants[15] = 2
    constants[16] = 0.2
    constants[17] = 1.2
    constants[18] = 0.6
    states[5] = 0.1
    states[6] = 0.1
    states[7] = 1.6
    constants[19] = 1.0
    constants[20] = 0.7
    constants[21] = 0.2
    constants[22] = 1.0
    states[8] = 0.1
    constants[23] = 0.9
    constants[24] = 8.0
    constants[25] = 1.0
    constants[26] = 2.0
    constants[27] = 2.0
    states[9] = 0.1
    constants[28] = 8.0
    constants[29] = 1.0
    constants[30] = 2.0
    constants[31] = 1
    constants[32] = 2
    constants[33] = 0.2
    constants[34] = 0.01
    constants[35] = 0.6
    constants[36] = 0.2
    constants[37] = 0.01
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = constants[0]*((power(constants[3], constants[5]))/(power(constants[3], constants[5])+power(states[1], constants[5])))-(constants[1]*(states[0]/(constants[2]+states[0]))+constants[4]*states[0])
    rates[2] = (constants[6]*states[0]+constants[8]*(states[3]/(constants[10]+states[3])))-(constants[7]*(states[2]/(constants[9]+states[2]))+constants[4]*states[2])
    rates[3] = (constants[7]*(states[2]/(constants[9]+states[2]))+constants[12]*(states[4]/(constants[14]+states[4])))-(constants[8]*(states[3]/(constants[10]+states[3]))+constants[11]*(states[3]/(constants[13]+states[3]))+constants[4]*states[3])
    rates[4] = (constants[11]*(states[3]/(constants[13]+states[3]))+constants[18]*states[6])-(constants[12]*(states[4]/(constants[14]+states[4]))+constants[17]*states[4]*states[5]+constants[15]*(states[4]/(constants[16]+states[4]))+constants[4]*states[4])
    rates[7] = constants[19]*((power(constants[22], constants[5]))/(power(constants[22], constants[5])+power(states[1], constants[5])))-(constants[20]*(states[7]/(constants[21]+states[7]))+constants[4]*states[7])
    rates[8] = (constants[23]*states[7]+constants[25]*(states[9]/(constants[27]+states[9])))-(constants[24]*(states[8]/(constants[26]+states[8]))+constants[4]*states[8])
    rates[9] = (constants[24]*(states[8]/(constants[26]+states[8]))+constants[29]*(states[5]/(constants[31]+states[5])))-(constants[25]*(states[9]/(constants[27]+states[9]))+constants[28]*(states[9]/(constants[30]+states[9]))+constants[4]*states[9])
    rates[5] = (constants[28]*(states[9]/(constants[30]+states[9]))+constants[18]*states[6])-(constants[29]*(states[5]/(constants[31]+states[5]))+constants[17]*states[4]*states[5]+constants[32]*(states[5]/(constants[33]+states[5]))+constants[4]*states[5])
    rates[6] = (constants[17]*states[4]*states[5]+constants[36]*states[1])-(constants[18]*states[6]+constants[35]*states[6]+constants[34]*states[6])
    rates[1] = constants[35]*states[6]-(constants[36]*states[1]+constants[37]*states[1])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = states[2]+states[3]+states[4]+states[6]+states[1]
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