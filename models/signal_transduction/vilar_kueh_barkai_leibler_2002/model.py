# Size of variable arrays:
sizeAlgebraic = 16
sizeStates = 9
sizeConstants = 16
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (hour)"
    legend_states[0] = "A in component A (molecules)"
    legend_algebraic[0] = "RXN1 in component RXN1 (flux)"
    legend_algebraic[1] = "RXN2 in component RXN2 (flux)"
    legend_algebraic[4] = "RXN5 in component RXN5 (flux)"
    legend_algebraic[10] = "RXN11 in component RXN11 (flux)"
    legend_algebraic[5] = "RXN6 in component RXN6 (flux)"
    legend_algebraic[9] = "RXN10 in component RXN10 (flux)"
    legend_algebraic[11] = "RXN12 in component RXN12 (flux)"
    legend_states[1] = "C in component C (molecules)"
    legend_algebraic[2] = "RXN3 in component RXN3 (flux)"
    legend_states[2] = "DA in component DA (molecules)"
    legend_algebraic[6] = "RXN7 in component RXN7 (flux)"
    legend_states[3] = "DAp in component DAp (molecules)"
    legend_algebraic[7] = "RXN8 in component RXN8 (flux)"
    legend_states[4] = "DR in component DR (molecules)"
    legend_algebraic[12] = "RXN13 in component RXN13 (flux)"
    legend_states[5] = "DRP in component DRP (molecules)"
    legend_algebraic[13] = "RXN14 in component RXN14 (flux)"
    legend_states[6] = "MA in component MA (molecules)"
    legend_algebraic[8] = "RXN9 in component RXN9 (flux)"
    legend_states[7] = "MR in component MR (molecules)"
    legend_algebraic[14] = "RXN15 in component RXN15 (flux)"
    legend_algebraic[15] = "RXN16 in component RXN16 (flux)"
    legend_states[8] = "R in component R (molecules)"
    legend_algebraic[3] = "RXN4 in component RXN4 (flux)"
    legend_constants[0] = "Gamma_1 in component RXN1 (second_order_rate)"
    legend_constants[1] = "Delta_1 in component RXN2 (first_order_rate)"
    legend_constants[2] = "Delta_2 in component RXN3 (first_order_rate)"
    legend_constants[3] = "Delta_3 in component RXN4 (first_order_rate)"
    legend_constants[4] = "Gamma_2 in component RXN5 (second_order_rate)"
    legend_constants[5] = "Thetha_1 in component RXN6 (first_order_rate)"
    legend_constants[6] = "Alpha_1 in component RXN7 (first_order_rate)"
    legend_constants[7] = "Alpha_2 in component RXN8 (first_order_rate)"
    legend_constants[8] = "Delta_4 in component RXN9 (first_order_rate)"
    legend_constants[9] = "BetaA_1 in component RXN10 (first_order_rate)"
    legend_constants[10] = "Gamma_3 in component RXN11 (second_order_rate)"
    legend_constants[11] = "Theta_2 in component RXN12 (first_order_rate)"
    legend_constants[12] = "Alpha_3 in component RXN13 (first_order_rate)"
    legend_constants[13] = "Alpha_4 in component RXN14 (first_order_rate)"
    legend_constants[14] = "Delta_5 in component RXN15 (first_order_rate)"
    legend_constants[15] = "BetaR_1 in component RXN16 (first_order_rate)"
    legend_rates[0] = "d/dt A in component A (molecules)"
    legend_rates[1] = "d/dt C in component C (molecules)"
    legend_rates[2] = "d/dt DA in component DA (molecules)"
    legend_rates[3] = "d/dt DAp in component DAp (molecules)"
    legend_rates[4] = "d/dt DR in component DR (molecules)"
    legend_rates[5] = "d/dt DRP in component DRP (molecules)"
    legend_rates[6] = "d/dt MA in component MA (molecules)"
    legend_rates[7] = "d/dt MR in component MR (molecules)"
    legend_rates[8] = "d/dt R in component R (molecules)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.0
    states[1] = 0.0
    states[2] = 1
    states[3] = 0.0
    states[4] = 1
    states[5] = 1
    states[6] = 0.0
    states[7] = 0.0
    states[8] = 0.0
    constants[0] = 2
    constants[1] = 1
    constants[2] = 1
    constants[3] = 0.2
    constants[4] = 1
    constants[5] = 50
    constants[6] = 50
    constants[7] = 500
    constants[8] = 10
    constants[9] = 50
    constants[10] = 1
    constants[11] = 100
    constants[12] = 0.01
    constants[13] = 50
    constants[14] = 0.5
    constants[15] = 5
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[0] = states[0]*states[8]*constants[0]
    algebraic[2] = states[1]*constants[2]
    rates[1] = (1.00000/1.00000)*(-1.00000*algebraic[2]+algebraic[0])
    algebraic[4] = states[0]*states[2]*constants[4]
    algebraic[5] = states[3]*constants[5]
    algebraic[6] = states[2]*constants[6]
    rates[2] = (1.00000/1.00000)*(-1.00000*algebraic[4]+-1.00000*algebraic[6]+algebraic[5]+algebraic[6])
    algebraic[7] = states[3]*constants[7]
    rates[3] = (1.00000/1.00000)*(-1.00000*algebraic[5]+-1.00000*algebraic[7]+algebraic[4]+algebraic[7])
    algebraic[9] = states[6]*constants[9]
    algebraic[8] = states[6]*constants[8]
    rates[6] = (1.00000/1.00000)*(-1.00000*algebraic[8]+-1.00000*algebraic[9]+algebraic[6]+algebraic[7]+algebraic[9])
    algebraic[1] = states[0]*constants[1]
    algebraic[10] = states[0]*states[4]*constants[10]
    algebraic[11] = states[5]*constants[11]
    rates[0] = (1.00000/1.00000)*(-1.00000*algebraic[0]+-1.00000*algebraic[1]+-1.00000*algebraic[4]+-1.00000*algebraic[10]+algebraic[5]+algebraic[9]+algebraic[11])
    algebraic[12] = states[4]*constants[12]
    rates[4] = (1.00000/1.00000)*(-1.00000*algebraic[10]+-1.00000*algebraic[12]+algebraic[11]+algebraic[12])
    algebraic[13] = states[5]*constants[13]
    rates[5] = (1.00000/1.00000)*(-1.00000*algebraic[11]+-1.00000*algebraic[13]+algebraic[10]+algebraic[13])
    algebraic[14] = states[7]*constants[14]
    algebraic[15] = states[7]*constants[15]
    rates[7] = (1.00000/1.00000)*(-1.00000*algebraic[14]+-1.00000*algebraic[15]+algebraic[12]+algebraic[13]+algebraic[15])
    algebraic[3] = states[8]*constants[3]
    rates[8] = (1.00000/1.00000)*(-1.00000*algebraic[0]+-1.00000*algebraic[3]+algebraic[2]+algebraic[15])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = states[0]*states[8]*constants[0]
    algebraic[2] = states[1]*constants[2]
    algebraic[4] = states[0]*states[2]*constants[4]
    algebraic[5] = states[3]*constants[5]
    algebraic[6] = states[2]*constants[6]
    algebraic[7] = states[3]*constants[7]
    algebraic[9] = states[6]*constants[9]
    algebraic[8] = states[6]*constants[8]
    algebraic[1] = states[0]*constants[1]
    algebraic[10] = states[0]*states[4]*constants[10]
    algebraic[11] = states[5]*constants[11]
    algebraic[12] = states[4]*constants[12]
    algebraic[13] = states[5]*constants[13]
    algebraic[14] = states[7]*constants[14]
    algebraic[15] = states[7]*constants[15]
    algebraic[3] = states[8]*constants[3]
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