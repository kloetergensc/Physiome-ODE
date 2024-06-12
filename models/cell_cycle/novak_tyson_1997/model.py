# Size of variable arrays:
sizeAlgebraic = 9
sizeStates = 13
sizeConstants = 43
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_constants[0] = "k1 in component rate_constants (first_order_rate_constant)"
    legend_algebraic[1] = "k2 in component rate_constants (first_order_rate_constant)"
    legend_constants[1] = "k2_ in component rate_constants (first_order_rate_constant)"
    legend_constants[2] = "k3 in component rate_constants (first_order_rate_constant)"
    legend_constants[3] = "k4 in component rate_constants (first_order_rate_constant)"
    legend_constants[4] = "k5 in component rate_constants (first_order_rate_constant)"
    legend_algebraic[0] = "k6 in component rate_constants (first_order_rate_constant)"
    legend_constants[5] = "k6_ in component rate_constants (first_order_rate_constant)"
    legend_constants[6] = "k7 in component rate_constants (first_order_rate_constant)"
    legend_constants[7] = "k7r in component rate_constants (first_order_rate_constant)"
    legend_constants[8] = "k8 in component rate_constants (first_order_rate_constant)"
    legend_constants[9] = "k8r in component rate_constants (first_order_rate_constant)"
    legend_algebraic[5] = "kwee in component rate_constants (first_order_rate_constant)"
    legend_algebraic[7] = "k25 in component rate_constants (first_order_rate_constant)"
    legend_constants[10] = "ku in component rate_constants (first_order_rate_constant)"
    legend_constants[11] = "kp in component rate_constants (first_order_rate_constant)"
    legend_constants[12] = "kur in component rate_constants (first_order_rate_constant)"
    legend_constants[13] = "ku2 in component rate_constants (first_order_rate_constant)"
    legend_constants[14] = "kur2 in component rate_constants (first_order_rate_constant)"
    legend_constants[15] = "kc in component rate_constants (first_order_rate_constant)"
    legend_constants[16] = "kcr in component rate_constants (first_order_rate_constant)"
    legend_constants[17] = "kw in component rate_constants (first_order_rate_constant)"
    legend_constants[18] = "kwr in component rate_constants (first_order_rate_constant)"
    legend_constants[19] = "ki in component rate_constants (first_order_rate_constant)"
    legend_constants[20] = "kir in component rate_constants (first_order_rate_constant)"
    legend_constants[21] = "mu in component rate_constants (first_order_rate_constant)"
    legend_constants[22] = "Kmu2 in component rate_constants (dimensionless)"
    legend_constants[23] = "Kmur2 in component rate_constants (dimensionless)"
    legend_constants[24] = "Kmi in component rate_constants (dimensionless)"
    legend_constants[25] = "Kmir in component rate_constants (dimensionless)"
    legend_constants[26] = "Kmw in component rate_constants (dimensionless)"
    legend_constants[27] = "Kmwr in component rate_constants (dimensionless)"
    legend_constants[28] = "Kmu in component rate_constants (dimensionless)"
    legend_constants[29] = "Kmur in component rate_constants (dimensionless)"
    legend_constants[30] = "Kmc in component rate_constants (dimensionless)"
    legend_constants[31] = "Kmcr in component rate_constants (dimensionless)"
    legend_constants[32] = "Kmp in component rate_constants (dimensionless)"
    legend_constants[33] = "V2 in component rate_constants (first_order_rate_constant)"
    legend_constants[34] = "V2_ in component rate_constants (first_order_rate_constant)"
    legend_constants[35] = "V6 in component rate_constants (first_order_rate_constant)"
    legend_constants[36] = "V6_ in component rate_constants (first_order_rate_constant)"
    legend_constants[37] = "V25 in component rate_constants (first_order_rate_constant)"
    legend_constants[38] = "V25_ in component rate_constants (first_order_rate_constant)"
    legend_constants[39] = "Vw in component rate_constants (first_order_rate_constant)"
    legend_constants[40] = "Vw_ in component rate_constants (first_order_rate_constant)"
    legend_states[0] = "Cdc25 in component Cdc25 (dimensionless)"
    legend_states[1] = "UbE in component UbE (dimensionless)"
    legend_states[2] = "UbE2 in component UbE2 (dimensionless)"
    legend_states[3] = "Wee1 in component Wee1 (dimensionless)"
    legend_algebraic[8] = "SPF in component concentration_variables (dimensionless)"
    legend_algebraic[6] = "MPF in component concentration_variables (dimensionless)"
    legend_constants[41] = "alpha in component concentration_variables (dimensionless)"
    legend_constants[42] = "beta in component concentration_variables (dimensionless)"
    legend_states[4] = "G1K in component G1K (dimensionless)"
    legend_states[5] = "G2K in component G2K (dimensionless)"
    legend_states[6] = "PG2 in component PG2 (dimensionless)"
    legend_states[7] = "G2R in component G2R (dimensionless)"
    legend_states[8] = "R in component R (dimensionless)"
    legend_states[9] = "G1R in component G1R (dimensionless)"
    legend_states[10] = "PG2R in component PG2R (dimensionless)"
    legend_states[11] = "mass in component mass (dimensionless)"
    legend_states[12] = "IE in component IE (dimensionless)"
    legend_algebraic[2] = "Cig2_total in component Cig2_total (dimensionless)"
    legend_algebraic[3] = "Rum1_total in component Rum1_total (dimensionless)"
    legend_algebraic[4] = "Cdc13_total in component Cdc13_total (dimensionless)"
    legend_rates[5] = "d/dt G2K in component G2K (dimensionless)"
    legend_rates[8] = "d/dt R in component R (dimensionless)"
    legend_rates[4] = "d/dt G1K in component G1K (dimensionless)"
    legend_rates[7] = "d/dt G2R in component G2R (dimensionless)"
    legend_rates[12] = "d/dt IE in component IE (dimensionless)"
    legend_rates[2] = "d/dt UbE2 in component UbE2 (dimensionless)"
    legend_rates[3] = "d/dt Wee1 in component Wee1 (dimensionless)"
    legend_rates[6] = "d/dt PG2 in component PG2 (dimensionless)"
    legend_rates[9] = "d/dt G1R in component G1R (dimensionless)"
    legend_rates[10] = "d/dt PG2R in component PG2R (dimensionless)"
    legend_rates[1] = "d/dt UbE in component UbE (dimensionless)"
    legend_rates[11] = "d/dt mass in component mass (dimensionless)"
    legend_rates[0] = "d/dt Cdc25 in component Cdc25 (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 0.015
    constants[1] = 0.05
    constants[2] = 0.09375
    constants[3] = 0.1875
    constants[4] = 0.00175
    constants[5] = 0
    constants[6] = 100
    constants[7] = 0.1
    constants[8] = 10
    constants[9] = 0.1
    constants[10] = 0.2
    constants[11] = 3.25
    constants[12] = 0.1
    constants[13] = 1
    constants[14] = 0.3
    constants[15] = 1
    constants[16] = 0.25
    constants[17] = 1
    constants[18] = 0.25
    constants[19] = 0.4
    constants[20] = 0.1
    constants[21] = 0.00495
    constants[22] = 0.05
    constants[23] = 0.05
    constants[24] = 0.01
    constants[25] = 0.01
    constants[26] = 0.1
    constants[27] = 0.1
    constants[28] = 0.01
    constants[29] = 0.01
    constants[30] = 0.1
    constants[31] = 0.1
    constants[32] = 0.001
    constants[33] = 0.25
    constants[34] = 0.0075
    constants[35] = 7.5
    constants[36] = 0.0375
    constants[37] = 0.5
    constants[38] = 0.025
    constants[39] = 0.35
    constants[40] = 0.035
    states[0] = 0
    states[1] = 1
    states[2] = 0
    states[3] = 0
    constants[41] = 0.25
    constants[42] = 0.05
    states[4] = 0
    states[5] = 0
    states[6] = 0
    states[7] = 0
    states[8] = 0.4
    states[9] = 0
    states[10] = 0
    states[11] = 0.5
    states[12] = 0
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[9] = constants[8]*states[8]*states[4]-(constants[9]+constants[3]+constants[5])*states[9]
    rates[1] = (constants[10]*states[12]*(1.00000-states[1]))/((constants[28]+1.00000)-states[1])-(constants[12]*states[1])/(constants[29]+states[1])
    rates[11] = constants[21]*states[11]
    algebraic[0] = constants[36]*(1.00000-states[2])+constants[35]*states[2]
    rates[4] = (constants[4]+(constants[9]+constants[3])*states[9])-(algebraic[0]+constants[8]*states[8])*states[4]
    algebraic[1] = constants[34]*(1.00000-states[1])+constants[33]*states[1]
    rates[7] = constants[6]*states[8]*states[5]-(constants[7]+constants[3]+algebraic[1]+constants[1])*states[7]
    rates[10] = constants[6]*states[8]*states[6]-(constants[7]+constants[3]+algebraic[1]+constants[1])*states[10]
    algebraic[6] = states[5]+constants[42]*states[6]
    rates[12] = (constants[19]*algebraic[6]*(1.00000-states[12]))/((constants[24]+1.00000)-states[12])-(constants[20]*states[12])/(constants[25]+states[12])
    rates[2] = (constants[13]*algebraic[6]*(1.00000-states[2]))/((constants[22]+1.00000)-states[2])-(constants[14]*states[2])/(constants[23]+states[2])
    rates[3] = (constants[18]*(1.00000-states[3]))/((constants[27]+1.00000)-states[3])-(constants[17]*algebraic[6]*states[3])/(constants[26]+states[3])
    rates[0] = (constants[15]*algebraic[6]*(1.00000-states[0]))/((constants[30]+1.00000)-states[0])-(constants[16]*states[0])/(constants[31]+states[0])
    algebraic[5] = constants[40]*(1.00000-states[3])+constants[39]*states[3]
    algebraic[7] = constants[38]*(1.00000-states[0])+constants[37]*states[0]
    rates[5] = (constants[0]+algebraic[7]*states[6]+(constants[7]+constants[3])*states[7])-(algebraic[1]+algebraic[5]+constants[6]*states[8])*states[5]
    algebraic[8] = algebraic[6]+constants[41]*states[4]
    rates[8] = (constants[2]+(constants[7]+algebraic[1]+constants[1])*(states[7]+states[10])+(constants[9]+constants[5])*states[9])-(constants[3]*states[8]+constants[6]*states[8]*(states[5]+states[6])+constants[8]*states[8]*states[4]+(constants[11]*states[8]*algebraic[8]*states[11])/(constants[32]+states[8]))
    rates[6] = (algebraic[5]*states[5]+(constants[7]+constants[3])*states[10])-(algebraic[7]+algebraic[1]+constants[6]*states[8])*states[6]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = constants[36]*(1.00000-states[2])+constants[35]*states[2]
    algebraic[1] = constants[34]*(1.00000-states[1])+constants[33]*states[1]
    algebraic[6] = states[5]+constants[42]*states[6]
    algebraic[5] = constants[40]*(1.00000-states[3])+constants[39]*states[3]
    algebraic[7] = constants[38]*(1.00000-states[0])+constants[37]*states[0]
    algebraic[8] = algebraic[6]+constants[41]*states[4]
    algebraic[2] = states[4]+states[9]
    algebraic[3] = states[8]+states[9]+states[7]+states[10]
    algebraic[4] = states[5]+states[7]+states[6]+states[10]
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