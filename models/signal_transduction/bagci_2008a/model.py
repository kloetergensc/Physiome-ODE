# Size of variable arrays:
sizeAlgebraic = 19
sizeStates = 9
sizeConstants = 30
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_states[0] = "NO in component NO (micromolar)"
    legend_states[1] = "O_2m in component O_2m (micromolar)"
    legend_constants[0] = "O_2 in component NO (micromolar)"
    legend_states[2] = "NO_2 in component NO (micromolar)"
    legend_states[3] = "N2O3 in component N2O3 (micromolar)"
    legend_states[4] = "GSNO in component GSNO (micromolar)"
    legend_states[5] = "CcOX in component NO (micromolar)"
    legend_states[6] = "FeLn in component FeLn (micromolar)"
    legend_constants[28] = "r_1NO in component NO (flux)"
    legend_algebraic[0] = "r_4NO in component NO (flux)"
    legend_algebraic[1] = "r_12aNO in component NO (flux)"
    legend_algebraic[2] = "r_12bNOp in component NO (flux)"
    legend_algebraic[3] = "r_12bNOm in component NO (flux)"
    legend_algebraic[4] = "r_14NO in component NO (flux)"
    legend_algebraic[5] = "r_15NO in component NO (flux)"
    legend_algebraic[6] = "r_16NO in component NO (flux)"
    legend_constants[1] = "k_1NO in component model_constant (flux)"
    legend_constants[2] = "k_4NO in component model_constant (second_order_rate_constant)"
    legend_constants[3] = "k_12aNO in component model_constant (rate2)"
    legend_constants[4] = "k_12bNOp in component model_constant (second_order_rate_constant)"
    legend_constants[5] = "k_12bNOm in component model_constant (first_order_rate_constant)"
    legend_constants[6] = "k_14NO in component model_constant (first_order_rate_constant)"
    legend_constants[7] = "k_15NO in component model_constant (second_order_rate_constant)"
    legend_constants[8] = "k_16NO in component model_constant (second_order_rate_constant)"
    legend_constants[9] = "SOD in component O_2m (micromolar)"
    legend_constants[29] = "r_2NO in component O_2m (flux)"
    legend_algebraic[7] = "r_5NO in component O_2m (flux)"
    legend_algebraic[8] = "r_10NO in component O_2m (flux)"
    legend_constants[10] = "k_2NO in component model_constant (flux)"
    legend_constants[11] = "k_5NO in component model_constant (second_order_rate_constant)"
    legend_constants[12] = "k_10NO in component model_constant (rate2)"
    legend_states[7] = "ONOO_m in component ONOO_m (micromolar)"
    legend_states[8] = "GSH in component GSH (micromolar)"
    legend_constants[13] = "GPX in component ONOO_m (micromolar)"
    legend_constants[14] = "CO_2 in component ONOO_m (micromolar)"
    legend_constants[15] = "Cyt_c in component ONOO_m (micromolar)"
    legend_algebraic[9] = "r_6NO in component ONOO_m (flux)"
    legend_algebraic[10] = "r_7NO in component ONOO_m (flux)"
    legend_algebraic[12] = "r_8NO in component ONOO_m (flux)"
    legend_algebraic[14] = "r_9NO in component ONOO_m (flux)"
    legend_constants[16] = "k_6NO in component model_constant (second_order_rate_constant)"
    legend_constants[17] = "k_7NO in component model_constant (second_order_rate_constant)"
    legend_constants[18] = "k_8NO in component model_constant (second_order_rate_constant)"
    legend_constants[19] = "k_9NO in component model_constant (second_order_rate_constant)"
    legend_algebraic[13] = "GSSG in component GSH (micromolar)"
    legend_algebraic[11] = "FeLnNO in component GSH (micromolar)"
    legend_constants[20] = "FeLn_0 in component model_constant (micromolar)"
    legend_constants[21] = "GSH_0 in component model_constant (micromolar)"
    legend_algebraic[15] = "r_11NO in component GSH (flux)"
    legend_algebraic[16] = "r_m in component GSH (flux)"
    legend_algebraic[18] = "r_17NO in component GSH (flux)"
    legend_constants[22] = "k_11NO in component model_constant (second_order_rate_constant)"
    legend_constants[23] = "v_m in component model_constant (flux)"
    legend_constants[24] = "k_m in component model_constant (micromolar)"
    legend_constants[25] = "k_17NO in component model_constant (second_order_rate_constant)"
    legend_algebraic[17] = "r_13NO in component N2O3 (flux)"
    legend_constants[26] = "k_13NO in component model_constant (first_order_rate_constant)"
    legend_constants[27] = "k_17bNO in component model_constant (second_order_rate_constant)"
    legend_rates[0] = "d/dt NO in component NO (micromolar)"
    legend_rates[5] = "d/dt CcOX in component NO (micromolar)"
    legend_rates[2] = "d/dt NO_2 in component NO (micromolar)"
    legend_rates[1] = "d/dt O_2m in component O_2m (micromolar)"
    legend_rates[7] = "d/dt ONOO_m in component ONOO_m (micromolar)"
    legend_rates[8] = "d/dt GSH in component GSH (micromolar)"
    legend_rates[4] = "d/dt GSNO in component GSNO (micromolar)"
    legend_rates[3] = "d/dt N2O3 in component N2O3 (micromolar)"
    legend_rates[6] = "d/dt FeLn in component FeLn (micromolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0
    states[1] = 0
    constants[0] = 35
    states[2] = 0
    states[3] = 0
    states[4] = 0
    states[5] = 0.1
    states[6] = 0.05
    constants[1] = 1
    constants[2] = 6700
    constants[3] = 0.000006
    constants[4] = 1100
    constants[5] = 81000
    constants[6] = 0.0002
    constants[7] = 100
    constants[8] = 1.21
    constants[9] = 10
    constants[10] = 0.1
    constants[11] = 2400
    constants[12] = 0.0006
    states[7] = 0
    states[8] = 10000
    constants[13] = 5.8
    constants[14] = 1000
    constants[15] = 400
    constants[16] = 0.00135
    constants[17] = 2
    constants[18] = 0.058
    constants[19] = 0.025
    constants[20] = 0.05
    constants[21] = 10000
    constants[22] = 66
    constants[23] = 320
    constants[24] = 50
    constants[25] = 66
    constants[26] = 1600
    constants[27] = 0.0002
    constants[28] = constants[1]
    constants[29] = constants[10]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[1] = constants[3]*(power(states[0], 2.00000))*constants[0]
    algebraic[2] = constants[4]*states[2]*states[0]
    algebraic[3] = constants[5]*states[3]
    rates[2] = (2.00000*algebraic[1]-algebraic[2])+algebraic[3]
    algebraic[5] = constants[7]*states[5]*states[0]
    rates[5] = -algebraic[5]
    algebraic[0] = constants[2]*states[0]*states[1]
    algebraic[4] = constants[6]*states[4]
    algebraic[6] = constants[8]*states[6]*states[0]
    rates[0] = (((((constants[28]-algebraic[0])-2.00000*algebraic[1])-algebraic[2])+algebraic[3]+algebraic[4])-algebraic[5])-algebraic[6]
    algebraic[7] = constants[11]*constants[9]*states[1]
    algebraic[8] = constants[12]*(power(states[4], 2.00000))*states[1]
    rates[1] = ((constants[29]-algebraic[0])-algebraic[7])-algebraic[8]
    algebraic[9] = constants[16]*states[7]*states[8]
    algebraic[10] = constants[17]*states[7]*constants[13]
    algebraic[12] = constants[18]*states[7]*constants[14]
    algebraic[14] = constants[19]*states[7]*constants[15]
    rates[7] = (((algebraic[0]-algebraic[9])-algebraic[10])-algebraic[12])-algebraic[14]
    algebraic[15] = constants[22]*states[3]*states[8]
    algebraic[17] = constants[26]*states[3]
    rates[3] = ((-algebraic[15]+algebraic[2])-algebraic[3])-algebraic[17]
    algebraic[13] = ((constants[21]-states[8])-states[4])/2.00000
    algebraic[16] = (constants[23]*algebraic[13])/(constants[24]+algebraic[13])
    algebraic[11] = constants[20]-states[6]
    algebraic[18] = constants[25]*algebraic[11]*states[8]
    rates[8] = ((-algebraic[9]-algebraic[15])+2.00000*algebraic[16])-algebraic[18]
    rates[4] = (((algebraic[9]-2.00000*algebraic[8])+algebraic[15])-algebraic[4])+algebraic[18]
    rates[6] = -algebraic[6]+algebraic[18]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[1] = constants[3]*(power(states[0], 2.00000))*constants[0]
    algebraic[2] = constants[4]*states[2]*states[0]
    algebraic[3] = constants[5]*states[3]
    algebraic[5] = constants[7]*states[5]*states[0]
    algebraic[0] = constants[2]*states[0]*states[1]
    algebraic[4] = constants[6]*states[4]
    algebraic[6] = constants[8]*states[6]*states[0]
    algebraic[7] = constants[11]*constants[9]*states[1]
    algebraic[8] = constants[12]*(power(states[4], 2.00000))*states[1]
    algebraic[9] = constants[16]*states[7]*states[8]
    algebraic[10] = constants[17]*states[7]*constants[13]
    algebraic[12] = constants[18]*states[7]*constants[14]
    algebraic[14] = constants[19]*states[7]*constants[15]
    algebraic[15] = constants[22]*states[3]*states[8]
    algebraic[17] = constants[26]*states[3]
    algebraic[13] = ((constants[21]-states[8])-states[4])/2.00000
    algebraic[16] = (constants[23]*algebraic[13])/(constants[24]+algebraic[13])
    algebraic[11] = constants[20]-states[6]
    algebraic[18] = constants[25]*algebraic[11]*states[8]
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