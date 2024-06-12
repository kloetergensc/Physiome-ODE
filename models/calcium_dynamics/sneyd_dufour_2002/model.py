# Size of variable arrays:
sizeAlgebraic = 2
sizeStates = 5
sizeConstants = 26
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_states[0] = "R in component R (dimensionless)"
    legend_constants[19] = "phi_1 in component reaction_rate_constants (first_order_rate_constant)"
    legend_constants[20] = "phi_2 in component reaction_rate_constants (second_order_rate_constant)"
    legend_constants[21] = "phi_2b in component reaction_rate_constants (first_order_rate_constant)"
    legend_constants[0] = "k_1b in component reaction_rate_constants (first_order_rate_constant)"
    legend_constants[1] = "l_2b in component reaction_rate_constants (first_order_rate_constant)"
    legend_states[1] = "I_1 in component I_1 (dimensionless)"
    legend_states[2] = "O in component O (dimensionless)"
    legend_constants[2] = "p in component reaction_rate_constants (micromolar)"
    legend_constants[22] = "phi_3 in component reaction_rate_constants (second_order_rate_constant)"
    legend_constants[23] = "phi_4 in component reaction_rate_constants (first_order_rate_constant)"
    legend_constants[24] = "phi_4b in component reaction_rate_constants (first_order_rate_constant)"
    legend_constants[3] = "k_3b in component reaction_rate_constants (first_order_rate_constant)"
    legend_states[3] = "A in component A (dimensionless)"
    legend_algebraic[0] = "S in component S (dimensionless)"
    legend_states[4] = "I_2 in component I_2 (dimensionless)"
    legend_constants[25] = "phi_5 in component reaction_rate_constants (first_order_rate_constant)"
    legend_algebraic[1] = "open_probability in component open_probability (dimensionless)"
    legend_constants[4] = "k_1a in component reaction_rate_constants (second_order_rate_constant)"
    legend_constants[5] = "k_2a in component reaction_rate_constants (second_order_rate_constant)"
    legend_constants[6] = "k_2b in component reaction_rate_constants (first_order_rate_constant)"
    legend_constants[7] = "k_3a in component reaction_rate_constants (second_order_rate_constant)"
    legend_constants[8] = "k_4a in component reaction_rate_constants (second_order_rate_constant)"
    legend_constants[9] = "k_4b in component reaction_rate_constants (first_order_rate_constant)"
    legend_constants[10] = "l_2a in component reaction_rate_constants (first_order_rate_constant)"
    legend_constants[11] = "l_4a in component reaction_rate_constants (second_order_rate_constant)"
    legend_constants[12] = "l_4b in component reaction_rate_constants (second_order_rate_constant)"
    legend_constants[13] = "l_6a in component reaction_rate_constants (first_order_rate_constant)"
    legend_constants[14] = "l_6b in component reaction_rate_constants (first_order_rate_constant)"
    legend_constants[15] = "L_1 in component reaction_rate_constants (micromolar)"
    legend_constants[16] = "L_3 in component reaction_rate_constants (micromolar)"
    legend_constants[17] = "L_5 in component reaction_rate_constants (micromolar)"
    legend_constants[18] = "c in component reaction_rate_constants (micromolar)"
    legend_rates[0] = "d/dt R in component R (dimensionless)"
    legend_rates[2] = "d/dt O in component O (dimensionless)"
    legend_rates[1] = "d/dt I_1 in component I_1 (dimensionless)"
    legend_rates[4] = "d/dt I_2 in component I_2 (dimensionless)"
    legend_rates[3] = "d/dt A in component A (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 1
    constants[0] = 0.04
    constants[1] = 0.8
    states[1] = 0
    states[2] = 0
    constants[2] = 10
    constants[3] = 29.8
    states[3] = 0
    states[4] = 0
    constants[4] = 0.64
    constants[5] = 37.4
    constants[6] = 1.4
    constants[7] = 0.11
    constants[8] = 4
    constants[9] = 0.54
    constants[10] = 1.7
    constants[11] = 1.7
    constants[12] = 2.5
    constants[13] = 4707
    constants[14] = 11.4
    constants[15] = 0.12
    constants[16] = 0.025
    constants[17] = 54.7
    constants[18] = 1
    constants[19] = ((constants[4]*constants[15]+constants[10])*constants[18])/(constants[15]+constants[18]*(1.00000+constants[15]/constants[16]))
    constants[20] = (constants[5]*constants[16]+constants[11]*constants[18])/(constants[16]+constants[18]*(1.00000+constants[16]/constants[15]))
    constants[21] = (constants[6]+constants[12]*constants[18])/(1.00000+constants[18]/constants[17])
    constants[22] = (constants[7]*constants[17])/(constants[18]+constants[17])
    constants[23] = ((constants[8]*constants[17]+constants[13])*constants[18])/(constants[18]+constants[17])
    constants[24] = (constants[15]*(constants[9]+constants[14]))/(constants[18]+constants[15])
    constants[25] = ((constants[4]*constants[15]+constants[10])*constants[18])/(constants[18]+constants[15])
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = (constants[21]*states[2]+(constants[0]+constants[1])*states[1])-(constants[20]*constants[2]*states[0]+constants[19]*states[0])
    rates[1] = constants[19]*states[0]-(constants[0]+constants[1])*states[1]
    rates[4] = constants[25]*states[3]-(constants[0]+constants[1])*states[4]
    rates[3] = (constants[23]*states[2]+(constants[0]+constants[1])*states[4])-(constants[24]*states[3]+constants[25]*states[3])
    algebraic[0] = 1.00000-(states[0]+states[2]+states[3]+states[1]+states[4])
    rates[2] = (constants[20]*constants[2]*states[0]+constants[24]*states[3]+constants[3]*algebraic[0])-(constants[21]+constants[23]+1.00000*constants[22])*states[2]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = 1.00000-(states[0]+states[2]+states[3]+states[1]+states[4])
    algebraic[1] = power(0.100000*states[2]+0.900000*states[3], 4.00000)
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