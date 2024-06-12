# Size of variable arrays:
sizeAlgebraic = 10
sizeStates = 5
sizeConstants = 27
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_constants[0] = "k_0 in component Constants (flux)"
    legend_constants[1] = "k_1 in component Constants (per_second)"
    legend_constants[2] = "k_2 in component Constants (per_second)"
    legend_constants[3] = "k_3 in component Constants (per_second)"
    legend_constants[4] = "k_4 in component Constants (per_second)"
    legend_constants[5] = "k_5 in component Constants (per_second)"
    legend_constants[6] = "k_6 in component Constants (per_second)"
    legend_constants[7] = "k_7 in component Constants (per_second)"
    legend_constants[8] = "k_8 in component Constants (flux)"
    legend_constants[9] = "k_9 in component Constants (flux)"
    legend_constants[10] = "k_10 in component Constants (flux)"
    legend_constants[11] = "k_11 in component Constants (flux)"
    legend_constants[12] = "C_PLC_T in component Constants (micromolar)"
    legend_constants[13] = "K_D in component Constants (micromolar)"
    legend_constants[14] = "K_P in component Constants (micromolar)"
    legend_constants[15] = "K_R in component Constants (micromolar)"
    legend_constants[16] = "K_G in component Constants (micromolar)"
    legend_constants[17] = "K_S in component Constants (micromolar)"
    legend_constants[18] = "K_ER in component Constants (micromolar)"
    legend_constants[19] = "K_C1 in component Constants (micromolar)"
    legend_constants[20] = "K_C2 in component Constants (micromolar)"
    legend_constants[21] = "beta in component Constants (dimensionless)"
    legend_constants[22] = "lambda in component Constants (dimensionless)"
    legend_constants[23] = "rho in component Constants (dimensionless)"
    legend_constants[24] = "n in component Constants (dimensionless)"
    legend_constants[25] = "m in component Constants (dimensionless)"
    legend_constants[26] = "w in component Constants (dimensionless)"
    legend_algebraic[1] = "R_APLC in component R_values (dimensionless)"
    legend_algebraic[8] = "R_PKC in component R_values (dimensionless)"
    legend_algebraic[3] = "R_G in component R_values (dimensionless)"
    legend_algebraic[9] = "R_DG in component R_values (dimensionless)"
    legend_algebraic[0] = "R_IP_3 in component R_values (dimensionless)"
    legend_algebraic[2] = "R_Cyt1 in component R_values (dimensionless)"
    legend_algebraic[4] = "R_Cyt2 in component R_values (dimensionless)"
    legend_algebraic[6] = "R_ER in component R_values (dimensionless)"
    legend_states[0] = "APLC in component APLC (micromolar)"
    legend_algebraic[7] = "DG in component DG (micromolar)"
    legend_states[1] = "C_cyt in component C_cyt (micromolar)"
    legend_states[2] = "G in component G_GTP (micromolar)"
    legend_states[3] = "IP_3 in component IP_3 (micromolar)"
    legend_states[4] = "C_ER in component C_ER (micromolar)"
    legend_algebraic[5] = "PLC in component APLC (micromolar)"
    legend_rates[2] = "d/dt G in component G_GTP (micromolar)"
    legend_rates[0] = "d/dt APLC in component APLC (micromolar)"
    legend_rates[3] = "d/dt IP_3 in component IP_3 (micromolar)"
    legend_rates[1] = "d/dt C_cyt in component C_cyt (micromolar)"
    legend_rates[4] = "d/dt C_ER in component C_ER (micromolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 1e-4
    constants[1] = 3.4
    constants[2] = 4
    constants[3] = 4.5
    constants[4] = 1.2
    constants[5] = 0.12
    constants[6] = 14
    constants[7] = 2
    constants[8] = 10.5
    constants[9] = 0.6
    constants[10] = 3
    constants[11] = 0.26
    constants[12] = 0.01
    constants[13] = 0.01
    constants[14] = 0.004
    constants[15] = 0.2
    constants[16] = 0.025
    constants[17] = 0.025
    constants[18] = 0.075
    constants[19] = 1
    constants[20] = 2
    constants[21] = 0.05
    constants[22] = 0.001
    constants[23] = 0.2
    constants[24] = 4
    constants[25] = 2
    constants[26] = 3
    states[0] = 0.001
    states[1] = 0.2
    states[2] = 0.001
    states[3] = 0.001
    states[4] = 1
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[3] = constants[6]*states[0]-constants[7]*states[3]
    algebraic[0] = (power(states[3], 3.00000))/(power(constants[17], 3.00000)+power(states[3], 3.00000))
    algebraic[2] = states[1]/(constants[19]+states[1])
    algebraic[4] = states[1]/(constants[20]+states[1])
    algebraic[6] = (power(states[4], constants[26]))/(power(constants[18], constants[26])+power(states[4], constants[26]))
    rates[1] = constants[21]*((constants[23]*(constants[8]*algebraic[0]*algebraic[6]-constants[9]*algebraic[2])-constants[10]*algebraic[4])+constants[11])
    rates[4] = constants[22]*(-constants[8]*algebraic[0]*algebraic[6]+constants[9]*algebraic[2])
    algebraic[1] = states[0]/(constants[14]+states[0])
    algebraic[7] = states[3]
    algebraic[8] = ((algebraic[7]/(constants[13]+algebraic[7]))*states[1])/(constants[15]+states[1])
    rates[2] = ((constants[0]+constants[1]*states[2])-constants[2]*algebraic[1]*states[2])-constants[3]*algebraic[8]*states[2]
    algebraic[3] = (power(states[2], constants[24]))/(power(constants[16], constants[24])+power(states[2], constants[24]))
    algebraic[9] = (power(algebraic[7], constants[25]))/(power(constants[13], constants[25])+power(algebraic[7], constants[25]))
    algebraic[5] = constants[12]-states[0]
    rates[0] = constants[4]*algebraic[3]*algebraic[9]*algebraic[5]-constants[5]*states[0]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = (power(states[3], 3.00000))/(power(constants[17], 3.00000)+power(states[3], 3.00000))
    algebraic[2] = states[1]/(constants[19]+states[1])
    algebraic[4] = states[1]/(constants[20]+states[1])
    algebraic[6] = (power(states[4], constants[26]))/(power(constants[18], constants[26])+power(states[4], constants[26]))
    algebraic[1] = states[0]/(constants[14]+states[0])
    algebraic[7] = states[3]
    algebraic[8] = ((algebraic[7]/(constants[13]+algebraic[7]))*states[1])/(constants[15]+states[1])
    algebraic[3] = (power(states[2], constants[24]))/(power(constants[16], constants[24])+power(states[2], constants[24]))
    algebraic[9] = (power(algebraic[7], constants[25]))/(power(constants[13], constants[25])+power(algebraic[7], constants[25]))
    algebraic[5] = constants[12]-states[0]
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