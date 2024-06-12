# Size of variable arrays:
sizeAlgebraic = 0
sizeStates = 10
sizeConstants = 55
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (hour)"
    legend_states[0] = "Per_m in component Per_m (nanomolar)"
    legend_constants[0] = "B1 in component Per_m (dimensionless)"
    legend_constants[1] = "C1 in component Per_m (flux)"
    legend_constants[2] = "S1 in component Per_m (flux)"
    legend_constants[3] = "D1 in component Per_m (flux)"
    legend_constants[4] = "L1 in component Per_m (nanomolar)"
    legend_constants[5] = "R1 in component Per_m (nanomolar)"
    legend_constants[6] = "A1 in component Per_m (nanomolar)"
    legend_states[1] = "PT_n in component PT_n (nanomolar)"
    legend_states[2] = "CC_n in component CC_n (nanomolar)"
    legend_constants[7] = "D0 in component parameters (first_order_rate_constant)"
    legend_constants[8] = "a in component parameters (dimensionless)"
    legend_constants[9] = "r in component parameters (dimensionless)"
    legend_states[3] = "Per_c in component Per_c (nanomolar)"
    legend_constants[10] = "S2 in component Per_c (first_order_rate_constant)"
    legend_constants[11] = "D2 in component Per_c (first_order_rate_constant)"
    legend_constants[12] = "L2 in component Per_c (nanomolar)"
    legend_constants[13] = "Dbt_c in component Per_c (nanomolar)"
    legend_constants[14] = "V1 in component parameters (second_order_rate_constant)"
    legend_constants[15] = "V2 in component parameters (first_order_rate_constant)"
    legend_states[4] = "Tim_c in component Tim_c (nanomolar)"
    legend_states[5] = "PT_c in component PT_c (nanomolar)"
    legend_states[6] = "Tim_m in component Tim_m (nanomolar)"
    legend_constants[16] = "B2 in component Tim_m (dimensionless)"
    legend_constants[17] = "C2 in component Tim_m (flux)"
    legend_constants[18] = "S3 in component Tim_m (flux)"
    legend_constants[19] = "D3 in component Tim_m (flux)"
    legend_constants[20] = "L3 in component Tim_m (nanomolar)"
    legend_constants[21] = "R2 in component Tim_m (nanomolar)"
    legend_constants[22] = "A2 in component Tim_m (nanomolar)"
    legend_constants[23] = "S4 in component Tim_c (first_order_rate_constant)"
    legend_constants[24] = "D4 in component Tim_c (flux)"
    legend_constants[25] = "L4 in component Tim_c (nanomolar)"
    legend_constants[26] = "D5 in component PT_c (flux)"
    legend_constants[27] = "L5 in component PT_c (nanomolar)"
    legend_constants[28] = "K1 in component parameters (nanomolar)"
    legend_constants[29] = "K2 in component parameters (nanomolar)"
    legend_constants[30] = "T1 in component parameters (flux)"
    legend_constants[31] = "T2 in component parameters (flux)"
    legend_constants[32] = "D6 in component PT_n (flux)"
    legend_constants[33] = "L6 in component PT_n (nanomolar)"
    legend_states[7] = "Clk_m in component Clk_m (nanomolar)"
    legend_constants[34] = "B3 in component Clk_m (dimensionless)"
    legend_constants[35] = "C3 in component Clk_m (flux)"
    legend_constants[36] = "S5 in component Clk_m (flux)"
    legend_constants[37] = "D7 in component Clk_m (flux)"
    legend_constants[38] = "L7 in component Clk_m (nanomolar)"
    legend_constants[39] = "R3 in component Clk_m (nanomolar)"
    legend_constants[40] = "A3 in component Clk_m (nanomolar)"
    legend_states[8] = "Clk_c in component Clk_c (nanomolar)"
    legend_constants[41] = "S6 in component Clk_c (first_order_rate_constant)"
    legend_constants[42] = "D8 in component Clk_c (flux)"
    legend_constants[43] = "L8 in component Clk_c (nanomolar)"
    legend_constants[44] = "V3 in component parameters (second_order_rate_constant)"
    legend_constants[45] = "V4 in component parameters (first_order_rate_constant)"
    legend_constants[46] = "Cyc_c in component Cyc_c (nanomolar)"
    legend_states[9] = "CC_c in component CC_c (nanomolar)"
    legend_constants[47] = "D9 in component CC_c (flux)"
    legend_constants[48] = "L9 in component CC_c (nanomolar)"
    legend_constants[49] = "K3 in component parameters (nanomolar)"
    legend_constants[50] = "K4 in component parameters (nanomolar)"
    legend_constants[51] = "T3 in component parameters (flux)"
    legend_constants[52] = "T4 in component parameters (flux)"
    legend_constants[53] = "D10 in component CC_n (flux)"
    legend_constants[54] = "L10 in component CC_n (nanomolar)"
    legend_rates[0] = "d/dt Per_m in component Per_m (nanomolar)"
    legend_rates[3] = "d/dt Per_c in component Per_c (nanomolar)"
    legend_rates[6] = "d/dt Tim_m in component Tim_m (nanomolar)"
    legend_rates[4] = "d/dt Tim_c in component Tim_c (nanomolar)"
    legend_rates[5] = "d/dt PT_c in component PT_c (nanomolar)"
    legend_rates[1] = "d/dt PT_n in component PT_n (nanomolar)"
    legend_rates[7] = "d/dt Clk_m in component Clk_m (nanomolar)"
    legend_rates[8] = "d/dt Clk_c in component Clk_c (nanomolar)"
    legend_rates[9] = "d/dt CC_c in component CC_c (nanomolar)"
    legend_rates[2] = "d/dt CC_n in component CC_n (nanomolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.5
    constants[0] = 0.0
    constants[1] = 0.0
    constants[2] = 1.45
    constants[3] = 0.94
    constants[4] = 0.3
    constants[5] = 1.02
    constants[6] = 0.45
    states[1] = 1.0
    states[2] = 0.4
    constants[7] = 0.012
    constants[8] = 1.0
    constants[9] = 4.0
    states[3] = 0.6
    constants[10] = 0.48
    constants[11] = 0.44
    constants[12] = 0.2
    constants[13] = 1.0
    constants[14] = 1.45
    constants[15] = 1.45
    states[4] = 0.8
    states[5] = 0.9
    states[6] = 0.7
    constants[16] = 0.0
    constants[17] = 0.0
    constants[18] = 1.45
    constants[19] = 0.94
    constants[20] = 0.3
    constants[21] = 1.02
    constants[22] = 0.45
    constants[23] = 0.48
    constants[24] = 0.44
    constants[25] = 0.2
    constants[26] = 0.44
    constants[27] = 0.2
    constants[28] = 2.0
    constants[29] = 2.0
    constants[30] = 1.73
    constants[31] = 0.72
    constants[32] = 0.29
    constants[33] = 0.2
    states[7] = 0.1
    constants[34] = 0.6
    constants[35] = 0.0
    constants[36] = 1.63
    constants[37] = 0.54
    constants[38] = 0.13
    constants[39] = 0.89
    constants[40] = 0.8
    states[8] = 0.2
    constants[41] = 0.47
    constants[42] = 0.6
    constants[43] = 0.2
    constants[44] = 1.63
    constants[45] = 1.63
    constants[46] = 1.0
    states[9] = 0.3
    constants[47] = 0.6
    constants[48] = 0.2
    constants[49] = 2.0
    constants[50] = 2.0
    constants[51] = 1.63
    constants[52] = 0.52
    constants[53] = 0.3
    constants[54] = 0.2
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = (constants[1]+constants[2]*((power(states[2]/constants[6], constants[8])+constants[0])/(1.00000+power(states[1]/constants[5], constants[9])+power(states[2]/constants[6], constants[8])+constants[0])))-(constants[3]*(states[0]/(constants[4]+states[0]))+constants[7]*states[0])
    rates[3] = (constants[10]*states[0]+constants[15]*states[5])-(constants[14]*states[3]*states[4]+constants[11]*constants[13]*(states[3]/(constants[12]+states[3]))+constants[7]*states[3])
    rates[6] = (constants[17]+constants[18]*((power(states[2]/constants[22], constants[8])+constants[16])/(1.00000+power(states[1]/constants[21], constants[9])+power(states[2]/constants[22], constants[8])+constants[16])))-(constants[19]*(states[6]/(constants[20]+states[6]))+constants[7]*states[6])
    rates[4] = (constants[23]*states[6]+constants[15]*states[5])-(constants[14]*states[3]*states[4]+constants[24]*(states[4]/(constants[25]+states[4]))+constants[7]*states[4])
    rates[5] = (constants[14]*states[3]*states[4]+constants[31]*(states[1]/(constants[29]+states[1])))-(constants[15]*states[5]+constants[30]*(states[5]/(constants[28]+states[5]))+constants[26]*(states[5]/(constants[27]+states[5]))+constants[7]*states[5])
    rates[1] = constants[30]*(states[5]/(constants[28]+states[5]))-(constants[31]*(states[1]/(constants[29]+states[1]))+constants[32]*(states[1]/(constants[33]+states[1]))+constants[7]*states[1])
    rates[7] = (constants[35]+constants[36]*((power(states[1]/constants[40], constants[8])+constants[34])/(1.00000+power(states[2]/constants[39], constants[9])+power(states[1]/constants[40], constants[8])+constants[34])))-(constants[37]*(states[7]/(constants[38]+states[7]))+constants[7]*states[7])
    rates[8] = (constants[41]*states[7]+constants[45]*states[9])-(constants[44]*states[8]*constants[46]+constants[42]*(states[8]/(constants[43]+states[8]))+constants[7]*states[8])
    rates[9] = (constants[44]*states[8]*constants[46]+constants[52]*(states[2]/(constants[50]+states[2])))-(constants[45]*states[9]+constants[51]*(states[9]/(constants[49]+states[9]))+constants[47]*(states[9]/(constants[48]+states[9]))+constants[7]*states[9])
    rates[2] = constants[51]*(states[9]/(constants[49]+states[9]))-(constants[52]*(states[2]/(constants[50]+states[2]))+constants[53]*(states[2]/(constants[54]+states[2]))+constants[7]*states[2])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
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