# Size of variable arrays:
sizeAlgebraic = 13
sizeStates = 4
sizeConstants = 24
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
    legend_constants[0] = "C in component membrane (picoF)"
    legend_constants[1] = "i_app in component membrane (picoA)"
    legend_algebraic[10] = "i_NaP in component persistent_sodium_current (picoA)"
    legend_algebraic[4] = "i_Na in component fast_sodium_current (picoA)"
    legend_algebraic[8] = "i_K in component potassium_current (picoA)"
    legend_algebraic[11] = "i_L in component leakage_current (picoA)"
    legend_algebraic[12] = "i_tonic_e in component tonic_current (picoA)"
    legend_constants[2] = "E_Na in component fast_sodium_current (millivolt)"
    legend_constants[3] = "g_Na in component fast_sodium_current (nanoS)"
    legend_algebraic[0] = "m_infinity in component fast_sodium_current_m_gate (dimensionless)"
    legend_states[1] = "n in component fast_sodium_current_n_gate (dimensionless)"
    legend_constants[4] = "theta_m in component fast_sodium_current_m_gate (millivolt)"
    legend_constants[5] = "sigma_m in component fast_sodium_current_m_gate (millivolt)"
    legend_algebraic[1] = "n_infinity in component fast_sodium_current_n_gate (dimensionless)"
    legend_algebraic[5] = "tau_n in component fast_sodium_current_n_gate (millisecond)"
    legend_constants[6] = "tau_n_max in component fast_sodium_current_n_gate (millisecond)"
    legend_constants[7] = "theta_n in component fast_sodium_current_n_gate (millivolt)"
    legend_constants[8] = "sigma_n in component fast_sodium_current_n_gate (millivolt)"
    legend_constants[9] = "g_K in component potassium_current (nanoS)"
    legend_constants[10] = "E_K in component potassium_current (millivolt)"
    legend_states[2] = "n in component potassium_current_n_gate (dimensionless)"
    legend_algebraic[2] = "n_infinity in component potassium_current_n_gate (dimensionless)"
    legend_algebraic[6] = "tau_n in component potassium_current_n_gate (millisecond)"
    legend_constants[11] = "tau_n_max in component potassium_current_n_gate (millisecond)"
    legend_constants[12] = "theta_n in component potassium_current_n_gate (millivolt)"
    legend_constants[13] = "sigma_n in component potassium_current_n_gate (millivolt)"
    legend_constants[14] = "g_NaP in component persistent_sodium_current (nanoS)"
    legend_algebraic[9] = "m_infinity in component persistent_sodium_current_m_gate (dimensionless)"
    legend_states[3] = "h in component persistent_sodium_current_h_gate (dimensionless)"
    legend_constants[15] = "theta_m in component persistent_sodium_current_m_gate (millivolt)"
    legend_constants[16] = "sigma_m in component persistent_sodium_current_m_gate (millivolt)"
    legend_algebraic[3] = "h_infinity in component persistent_sodium_current_h_gate (dimensionless)"
    legend_algebraic[7] = "tau_h in component persistent_sodium_current_h_gate (millisecond)"
    legend_constants[17] = "tau_h_max in component persistent_sodium_current_h_gate (millisecond)"
    legend_constants[18] = "theta_h in component persistent_sodium_current_h_gate (millivolt)"
    legend_constants[19] = "sigma_h in component persistent_sodium_current_h_gate (millivolt)"
    legend_constants[20] = "g_L in component leakage_current (nanoS)"
    legend_constants[21] = "E_L in component leakage_current (millivolt)"
    legend_constants[22] = "g_tonic_e in component tonic_current (nanoS)"
    legend_constants[23] = "E_syn_e in component tonic_current (millivolt)"
    legend_rates[0] = "d/dt V in component membrane (millivolt)"
    legend_rates[1] = "d/dt n in component fast_sodium_current_n_gate (dimensionless)"
    legend_rates[2] = "d/dt n in component potassium_current_n_gate (dimensionless)"
    legend_rates[3] = "d/dt h in component persistent_sodium_current_h_gate (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = -50.0
    constants[0] = 21.0
    constants[1] = 0.0
    constants[2] = 50.0
    constants[3] = 28.0
    states[1] = 0.01
    constants[4] = -34.0
    constants[5] = -5.0
    constants[6] = 10.0
    constants[7] = -29.0
    constants[8] = -4.0
    constants[9] = 11.2
    constants[10] = -85.0
    states[2] = 0.01
    constants[11] = 10.0
    constants[12] = -29.0
    constants[13] = -4.0
    constants[14] = 2.8
    states[3] = 0.46
    constants[15] = -40.0
    constants[16] = -6.0
    constants[17] = 10000.0
    constants[18] = -48.0
    constants[19] = 6.0
    constants[20] = 2.8
    constants[21] = -57.5
    constants[22] = 0.0
    constants[23] = 0.0
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[1] = 1.00000/(1.00000+exp((states[0]-constants[7])/constants[8]))
    algebraic[5] = constants[6]/cosh((states[0]-constants[7])/(2.00000*constants[8]))
    rates[1] = (algebraic[1]-states[1])/algebraic[5]
    algebraic[2] = 1.00000/(1.00000+exp((states[0]-constants[12])/constants[13]))
    algebraic[6] = constants[11]/cosh((states[0]-constants[12])/(2.00000*constants[13]))
    rates[2] = (algebraic[2]-states[2])/algebraic[6]
    algebraic[3] = 1.00000/(1.00000+exp((states[0]-constants[18])/constants[19]))
    algebraic[7] = constants[17]/cosh((states[0]-constants[18])/(2.00000*constants[19]))
    rates[3] = (algebraic[3]-states[3])/algebraic[7]
    algebraic[9] = 1.00000/(1.00000+exp((states[0]-constants[15])/constants[16]))
    algebraic[10] = constants[14]*algebraic[9]*states[3]*(states[0]-constants[2])
    algebraic[0] = 1.00000/(1.00000+exp((states[0]-constants[4])/constants[5]))
    algebraic[4] = constants[3]*(power(algebraic[0], 3.00000))*(1.00000-states[1])*(states[0]-constants[2])
    algebraic[8] = constants[9]*(power(states[2], 4.00000))*(states[0]-constants[10])
    algebraic[11] = constants[20]*(states[0]-constants[21])
    algebraic[12] = constants[22]*(states[0]-constants[23])
    rates[0] = (-(algebraic[10]+algebraic[4]+algebraic[8]+algebraic[11]+algebraic[12])+constants[1])/constants[0]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[1] = 1.00000/(1.00000+exp((states[0]-constants[7])/constants[8]))
    algebraic[5] = constants[6]/cosh((states[0]-constants[7])/(2.00000*constants[8]))
    algebraic[2] = 1.00000/(1.00000+exp((states[0]-constants[12])/constants[13]))
    algebraic[6] = constants[11]/cosh((states[0]-constants[12])/(2.00000*constants[13]))
    algebraic[3] = 1.00000/(1.00000+exp((states[0]-constants[18])/constants[19]))
    algebraic[7] = constants[17]/cosh((states[0]-constants[18])/(2.00000*constants[19]))
    algebraic[9] = 1.00000/(1.00000+exp((states[0]-constants[15])/constants[16]))
    algebraic[10] = constants[14]*algebraic[9]*states[3]*(states[0]-constants[2])
    algebraic[0] = 1.00000/(1.00000+exp((states[0]-constants[4])/constants[5]))
    algebraic[4] = constants[3]*(power(algebraic[0], 3.00000))*(1.00000-states[1])*(states[0]-constants[2])
    algebraic[8] = constants[9]*(power(states[2], 4.00000))*(states[0]-constants[10])
    algebraic[11] = constants[20]*(states[0]-constants[21])
    algebraic[12] = constants[22]*(states[0]-constants[23])
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