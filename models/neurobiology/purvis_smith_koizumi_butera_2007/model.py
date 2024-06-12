# Size of variable arrays:
sizeAlgebraic = 11
sizeStates = 3
sizeConstants = 21
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
    legend_constants[0] = "C in component membrane (nanoF)"
    legend_constants[1] = "i_app in component membrane (nanoA)"
    legend_algebraic[10] = "i_tonic_e in component tonic_current (nanoA)"
    legend_algebraic[3] = "i_Na in component fast_sodium_current (nanoA)"
    legend_algebraic[8] = "i_NaP in component persistent_sodium_current (nanoA)"
    legend_algebraic[6] = "i_K in component potassium_current (nanoA)"
    legend_algebraic[9] = "i_leak in component leakage_current (nanoA)"
    legend_constants[2] = "E_Na in component fast_sodium_current (millivolt)"
    legend_constants[3] = "g_Na in component fast_sodium_current (nanoS)"
    legend_algebraic[0] = "m_infinity in component fast_sodium_current_m_gate (dimensionless)"
    legend_states[1] = "n in component potassium_current_n_gate (dimensionless)"
    legend_constants[4] = "theta_m in component fast_sodium_current_m_gate (millivolt)"
    legend_constants[5] = "omega_m in component fast_sodium_current_m_gate (millivolt)"
    legend_constants[6] = "g_K in component potassium_current (nanoS)"
    legend_constants[7] = "E_K in component potassium_current (millivolt)"
    legend_algebraic[1] = "n_infinity in component potassium_current_n_gate (dimensionless)"
    legend_algebraic[4] = "tau_n in component potassium_current_n_gate (millisecond)"
    legend_constants[8] = "tau_n_max in component potassium_current_n_gate (millisecond)"
    legend_constants[9] = "theta_n in component potassium_current_n_gate (millivolt)"
    legend_constants[10] = "omega_n in component potassium_current_n_gate (millivolt)"
    legend_constants[11] = "g_NaP in component persistent_sodium_current (nanoS)"
    legend_algebraic[7] = "m_infinity in component persistent_sodium_current_m_gate (dimensionless)"
    legend_states[2] = "h in component persistent_sodium_current_h_gate (dimensionless)"
    legend_constants[12] = "theta_m in component persistent_sodium_current_m_gate (millivolt)"
    legend_constants[13] = "omega_m in component persistent_sodium_current_m_gate (millivolt)"
    legend_algebraic[2] = "h_infinity in component persistent_sodium_current_h_gate (dimensionless)"
    legend_algebraic[5] = "tau_h in component persistent_sodium_current_h_gate (millisecond)"
    legend_constants[14] = "tau_h_max in component persistent_sodium_current_h_gate (millisecond)"
    legend_constants[15] = "theta_h in component persistent_sodium_current_h_gate (millivolt)"
    legend_constants[16] = "omega_h in component persistent_sodium_current_h_gate (millivolt)"
    legend_constants[17] = "g_leak in component leakage_current (nanoS)"
    legend_constants[18] = "E_leak in component leakage_current (millivolt)"
    legend_constants[19] = "g_tonic_e in component tonic_current (nanoS)"
    legend_constants[20] = "E_syn_e in component tonic_current (millivolt)"
    legend_rates[0] = "d/dt V in component membrane (millivolt)"
    legend_rates[1] = "d/dt n in component potassium_current_n_gate (dimensionless)"
    legend_rates[2] = "d/dt h in component persistent_sodium_current_h_gate (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = -50.0
    constants[0] = 0.021
    constants[1] = 0
    constants[2] = 50
    constants[3] = 28
    states[1] = 0.01
    constants[4] = -34
    constants[5] = -5
    constants[6] = 11.2
    constants[7] = -85
    constants[8] = 10
    constants[9] = -29
    constants[10] = -4
    constants[11] = 2.5
    states[2] = 0.92
    constants[12] = -45.1
    constants[13] = -6
    constants[14] = 10000
    constants[15] = -53
    constants[16] = 6
    constants[17] = 2.2
    constants[18] = -57.5
    constants[19] = 0.2
    constants[20] = 0
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[1] = 1.00000/(1.00000+exp((states[0]-constants[9])/constants[10]))
    algebraic[4] = constants[8]/cosh((states[0]-constants[9])/(2.00000*constants[10]))
    rates[1] = (algebraic[1]-states[1])/algebraic[4]
    algebraic[2] = 1.00000/(1.00000+exp((states[0]-constants[15])/constants[16]))
    algebraic[5] = constants[14]/cosh((states[0]-constants[15])/(2.00000*constants[16]))
    rates[2] = (algebraic[2]-states[2])/algebraic[5]
    algebraic[10] = constants[19]*(1.00000/1000.00)*(states[0]-constants[20])
    algebraic[0] = 1.00000/(1.00000+exp((states[0]-constants[4])/constants[5]))
    algebraic[3] = constants[3]*(power(algebraic[0], 3.00000))*(1.00000-states[1])*(1.00000/1000.00)*(states[0]-constants[2])
    algebraic[7] = 1.00000/(1.00000+exp((states[0]-constants[12])/constants[13]))
    algebraic[8] = constants[11]*algebraic[7]*states[2]*(1.00000/1000.00)*(states[0]-constants[2])
    algebraic[6] = constants[6]*(power(states[1], 4.00000))*(1.00000/1000.00)*(states[0]-constants[7])
    algebraic[9] = constants[17]*(1.00000/1000.00)*(states[0]-constants[18])
    rates[0] = (-(algebraic[3]+algebraic[8]+algebraic[6]+algebraic[9])+algebraic[10]+constants[1])/constants[0]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[1] = 1.00000/(1.00000+exp((states[0]-constants[9])/constants[10]))
    algebraic[4] = constants[8]/cosh((states[0]-constants[9])/(2.00000*constants[10]))
    algebraic[2] = 1.00000/(1.00000+exp((states[0]-constants[15])/constants[16]))
    algebraic[5] = constants[14]/cosh((states[0]-constants[15])/(2.00000*constants[16]))
    algebraic[10] = constants[19]*(1.00000/1000.00)*(states[0]-constants[20])
    algebraic[0] = 1.00000/(1.00000+exp((states[0]-constants[4])/constants[5]))
    algebraic[3] = constants[3]*(power(algebraic[0], 3.00000))*(1.00000-states[1])*(1.00000/1000.00)*(states[0]-constants[2])
    algebraic[7] = 1.00000/(1.00000+exp((states[0]-constants[12])/constants[13]))
    algebraic[8] = constants[11]*algebraic[7]*states[2]*(1.00000/1000.00)*(states[0]-constants[2])
    algebraic[6] = constants[6]*(power(states[1], 4.00000))*(1.00000/1000.00)*(states[0]-constants[7])
    algebraic[9] = constants[17]*(1.00000/1000.00)*(states[0]-constants[18])
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