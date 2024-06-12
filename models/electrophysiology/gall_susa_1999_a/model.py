# Size of variable arrays:
sizeAlgebraic = 9
sizeStates = 3
sizeConstants = 28
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (millisecond)"
    legend_states[0] = "V in component membrane (millivolt)"
    legend_constants[0] = "Cm in component membrane (femtoF)"
    legend_algebraic[5] = "i_Ca in component calcium_current (picoA)"
    legend_algebraic[0] = "i_K in component rapidly_activating_K_current (picoA)"
    legend_algebraic[6] = "i_K_Ca in component calcium_activated_K_current (picoA)"
    legend_algebraic[8] = "i_Na_Ca in component Na_Ca_exchanger_current (picoA)"
    legend_constants[1] = "V_K in component rapidly_activating_K_current (millivolt)"
    legend_constants[2] = "g_K in component rapidly_activating_K_current (picoS)"
    legend_states[1] = "n in component rapidly_activating_K_current_n_gate (dimensionless)"
    legend_algebraic[1] = "n_infinity in component rapidly_activating_K_current_n_gate (dimensionless)"
    legend_constants[3] = "lamda in component rapidly_activating_K_current_n_gate (dimensionless)"
    legend_algebraic[3] = "tau_n in component rapidly_activating_K_current_n_gate (millisecond)"
    legend_constants[4] = "V_n in component rapidly_activating_K_current_n_gate (millivolt)"
    legend_constants[5] = "S_n in component rapidly_activating_K_current_n_gate (millivolt)"
    legend_constants[6] = "a in component rapidly_activating_K_current_n_gate (millivolt)"
    legend_constants[7] = "b in component rapidly_activating_K_current_n_gate (millivolt)"
    legend_constants[8] = "c in component rapidly_activating_K_current_n_gate (millisecond)"
    legend_constants[9] = "V_ in component rapidly_activating_K_current_n_gate (millivolt)"
    legend_constants[10] = "V_Ca in component calcium_current (millivolt)"
    legend_constants[11] = "g_Ca in component calcium_current (picoS)"
    legend_algebraic[2] = "m_infinity in component calcium_current_m_gate (dimensionless)"
    legend_algebraic[4] = "h in component calcium_current_h_gate (dimensionless)"
    legend_constants[12] = "V_m in component calcium_current_m_gate (millivolt)"
    legend_constants[13] = "S_m in component calcium_current_m_gate (millivolt)"
    legend_constants[14] = "V_h in component calcium_current_h_gate (millivolt)"
    legend_constants[15] = "S_h in component calcium_current_h_gate (millivolt)"
    legend_constants[16] = "g_K_Ca in component calcium_activated_K_current (picoS)"
    legend_constants[17] = "K_d in component calcium_activated_K_current (micromolar)"
    legend_states[2] = "Ca_i in component ionic_concentrations (micromolar)"
    legend_constants[18] = "g_Na_Ca in component Na_Ca_exchanger_current (picoS)"
    legend_constants[19] = "K_1_2 in component Na_Ca_exchanger_current (micromolar)"
    legend_algebraic[7] = "V_Na_Ca in component Na_Ca_exchanger_current (millivolt)"
    legend_constants[20] = "RT_F in component Na_Ca_exchanger_current (millivolt)"
    legend_constants[21] = "nH in component Na_Ca_exchanger_current (dimensionless)"
    legend_constants[22] = "Ca_o in component ionic_concentrations (micromolar)"
    legend_constants[23] = "Na_i in component ionic_concentrations (millimolar)"
    legend_constants[24] = "Na_o in component ionic_concentrations (millimolar)"
    legend_constants[25] = "f in component ionic_concentrations (dimensionless)"
    legend_constants[26] = "k_Ca in component ionic_concentrations (per_millisecond)"
    legend_constants[27] = "alpha in component ionic_concentrations (mole_per_microlitre_coulomb)"
    legend_rates[0] = "d/dt V in component membrane (millivolt)"
    legend_rates[1] = "d/dt n in component rapidly_activating_K_current_n_gate (dimensionless)"
    legend_rates[2] = "d/dt Ca_i in component ionic_concentrations (micromolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = -76.0
    constants[0] = 5310.0
    constants[1] = -75.0
    constants[2] = 2500.0
    states[1] = 0.1
    constants[3] = 1.6
    constants[4] = -15.0
    constants[5] = 5.6
    constants[6] = 65.0
    constants[7] = 20.0
    constants[8] = 6.0
    constants[9] = -75.0
    constants[10] = 110.0
    constants[11] = 1400.0
    constants[12] = 4.0
    constants[13] = 14.0
    constants[14] = -10.0
    constants[15] = -10.0
    constants[16] = 30000.0
    constants[17] = 100.0
    states[2] = 0.52
    constants[18] = 234.0
    constants[19] = 1.5
    constants[20] = 26.54
    constants[21] = 5.0
    constants[22] = 2600.0
    constants[23] = 10.0
    constants[24] = 140.0
    constants[25] = 0.001
    constants[26] = 0.03
    constants[27] = 0.0000045055
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[1] = 1.00000/(1.00000+exp((constants[4]-states[0])/constants[5]))
    algebraic[3] = constants[8]/(exp((states[0]-constants[9])/constants[6])+exp((constants[9]-states[0])/constants[7]))
    rates[1] = constants[3]*((algebraic[1]-states[1])/algebraic[3])
    algebraic[2] = 1.00000/(1.00000+exp((constants[12]-states[0])/constants[13]))
    algebraic[4] = 1.00000/(1.00000+exp((constants[14]-states[0])/constants[15]))
    algebraic[5] = constants[11]*algebraic[2]*algebraic[4]*(states[0]-constants[10])
    algebraic[0] = constants[2]*states[1]*(states[0]-constants[1])
    algebraic[6] = constants[16]*(states[2]/(constants[17]+states[2]))*(states[0]-constants[1])
    algebraic[7] = constants[20]*(3.00000*log(constants[24]/constants[23]-log(constants[22]/states[2])))
    algebraic[8] = constants[18]*((power(states[2], constants[21]))/(power(constants[19], constants[21])+power(states[2], constants[21])))*(states[0]-algebraic[7])
    rates[0] = -(algebraic[0]+algebraic[5]+algebraic[6]+algebraic[8])/constants[0]
    rates[2] = constants[25]*(-constants[27]*(algebraic[5]-2.00000*algebraic[8])-constants[26]*states[2])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[1] = 1.00000/(1.00000+exp((constants[4]-states[0])/constants[5]))
    algebraic[3] = constants[8]/(exp((states[0]-constants[9])/constants[6])+exp((constants[9]-states[0])/constants[7]))
    algebraic[2] = 1.00000/(1.00000+exp((constants[12]-states[0])/constants[13]))
    algebraic[4] = 1.00000/(1.00000+exp((constants[14]-states[0])/constants[15]))
    algebraic[5] = constants[11]*algebraic[2]*algebraic[4]*(states[0]-constants[10])
    algebraic[0] = constants[2]*states[1]*(states[0]-constants[1])
    algebraic[6] = constants[16]*(states[2]/(constants[17]+states[2]))*(states[0]-constants[1])
    algebraic[7] = constants[20]*(3.00000*log(constants[24]/constants[23]-log(constants[22]/states[2])))
    algebraic[8] = constants[18]*((power(states[2], constants[21]))/(power(constants[19], constants[21])+power(states[2], constants[21])))*(states[0]-algebraic[7])
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