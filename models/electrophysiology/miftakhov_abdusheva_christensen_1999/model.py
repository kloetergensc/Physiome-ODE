# Size of variable arrays:
sizeAlgebraic = 13
sizeStates = 5
sizeConstants = 14
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (millisecond)"
    legend_constants[0] = "alpha in component model_constants (dimensionless)"
    legend_constants[1] = "lamda in component model_constants (dimensionless)"
    legend_states[0] = "V in component membrane (millivolt)"
    legend_constants[2] = "Cm in component membrane (microF_per_cm2)"
    legend_algebraic[8] = "i_Ca_T in component T_type_calcium_current (microA_per_cm2)"
    legend_algebraic[9] = "i_Ca_L in component L_type_calcium_current (microA_per_cm2)"
    legend_algebraic[11] = "i_Ca_K in component calcium_activated_potassium_current (microA_per_cm2)"
    legend_algebraic[10] = "i_K in component potassium_current (microA_per_cm2)"
    legend_algebraic[12] = "i_Cl in component leak_chloride_current (microA_per_cm2)"
    legend_algebraic[0] = "V_tilde in component gate_voltage (millivolt)"
    legend_constants[3] = "E_Ca in component T_type_calcium_current (millivolt)"
    legend_constants[4] = "g_Ca_T in component T_type_calcium_current (milliS_per_cm2)"
    legend_algebraic[7] = "m in component T_type_calcium_current_m_gate (dimensionless)"
    legend_states[1] = "h in component T_type_calcium_current_h_gate (dimensionless)"
    legend_algebraic[1] = "alpha_m in component T_type_calcium_current_m_gate (per_millisecond)"
    legend_algebraic[4] = "beta_m in component T_type_calcium_current_m_gate (per_millisecond)"
    legend_algebraic[2] = "alpha_h in component T_type_calcium_current_h_gate (per_millisecond)"
    legend_algebraic[5] = "beta_h in component T_type_calcium_current_h_gate (per_millisecond)"
    legend_constants[5] = "g_Ca_L in component L_type_calcium_current (milliS_per_cm2)"
    legend_states[2] = "x_Ca in component L_type_calcium_current_x_Ca_gate (dimensionless)"
    legend_constants[6] = "tau_x_Ca in component L_type_calcium_current_x_Ca_gate (millisecond)"
    legend_constants[7] = "E_K in component potassium_current (millivolt)"
    legend_constants[8] = "g_K in component potassium_current (milliS_per_cm2)"
    legend_states[3] = "n in component potassium_current_n_gate (dimensionless)"
    legend_algebraic[3] = "alpha_n in component potassium_current_n_gate (per_millisecond)"
    legend_algebraic[6] = "beta_n in component potassium_current_n_gate (per_millisecond)"
    legend_states[4] = "Ca in component calcium_activated_potassium_current (millimolar)"
    legend_constants[9] = "g_Ca_K in component calcium_activated_potassium_current (milliS_per_cm2)"
    legend_constants[10] = "rho in component calcium_activated_potassium_current (per_millisecond)"
    legend_constants[11] = "K_c in component calcium_activated_potassium_current (millimolar_per_millivolt)"
    legend_constants[12] = "g_Cl in component leak_chloride_current (milliS_per_cm2)"
    legend_constants[13] = "E_Cl in component leak_chloride_current (millivolt)"
    legend_rates[0] = "d/dt V in component membrane (millivolt)"
    legend_rates[1] = "d/dt h in component T_type_calcium_current_h_gate (dimensionless)"
    legend_rates[2] = "d/dt x_Ca in component L_type_calcium_current_x_Ca_gate (dimensionless)"
    legend_rates[3] = "d/dt n in component potassium_current_n_gate (dimensionless)"
    legend_rates[4] = "d/dt Ca in component calcium_activated_potassium_current (millimolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 0.12
    constants[1] = 12.5
    states[0] = -55.0
    constants[2] = 2.5
    constants[3] = 80.0
    constants[4] = 0.51
    states[1] = 0.01
    constants[5] = 0.004
    states[2] = 0.01
    constants[6] = 500.0
    constants[7] = -75.0
    constants[8] = 0.3
    states[3] = 0.01
    states[4] = 1E-4
    constants[9] = 0.03
    constants[10] = 0.125E3
    constants[11] = 425.0E-5
    constants[12] = 0.003
    constants[13] = -40.0
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[4] = (constants[10]/constants[0])*(constants[11]*states[2]*(constants[3]-states[0])-states[4])
    algebraic[0] = (127.000*states[0]+8265.00)/105.000
    rates[2] = (1.00000/(1.00000+exp(0.150000*(-algebraic[0]-50.0000)))-states[2])/(constants[0]*constants[6])
    algebraic[2] = 0.0700000*exp((25.0000-algebraic[0])/20.0000)
    algebraic[5] = 1.00000/(1.00000+exp(5.50000-algebraic[0]*0.100000))
    rates[1] = (algebraic[2]*(1.00000-states[1])-algebraic[5]*states[1])/(constants[0]*constants[1])
    algebraic[3] = (0.0100000*(55.0000-algebraic[0]))/(exp((55.0000-algebraic[0])/10.0000)-1.00000)
    algebraic[6] = 0.125000*exp((45.0000-algebraic[0])/80.0000)
    rates[3] = (algebraic[3]*(1.00000-states[3])-algebraic[6]*states[3])/(constants[0]*constants[1])
    algebraic[1] = (0.100000*(50.0000-algebraic[0]))/(exp(5.00000-algebraic[0]*0.100000)-1.00000)
    algebraic[4] = 4.00000*exp((25.0000-algebraic[0])/18.0000)
    algebraic[7] = algebraic[1]/(algebraic[1]+algebraic[4])
    algebraic[8] = constants[4]*(power(algebraic[7], 3.00000))*states[1]*(states[0]-constants[3])
    algebraic[9] = constants[5]*states[2]*(states[0]-constants[3])
    algebraic[11] = (constants[9]*states[4]*(states[0]-constants[7]))/(0.500000+states[4])
    algebraic[10] = constants[8]*(power(states[3], 4.00000))*(states[0]-constants[7])
    algebraic[12] = constants[12]*(states[0]-constants[13])
    rates[0] = -(1.00000/(constants[2]*constants[0]))*(algebraic[8]+algebraic[9]+algebraic[11]+algebraic[10]+algebraic[12])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = (127.000*states[0]+8265.00)/105.000
    algebraic[2] = 0.0700000*exp((25.0000-algebraic[0])/20.0000)
    algebraic[5] = 1.00000/(1.00000+exp(5.50000-algebraic[0]*0.100000))
    algebraic[3] = (0.0100000*(55.0000-algebraic[0]))/(exp((55.0000-algebraic[0])/10.0000)-1.00000)
    algebraic[6] = 0.125000*exp((45.0000-algebraic[0])/80.0000)
    algebraic[1] = (0.100000*(50.0000-algebraic[0]))/(exp(5.00000-algebraic[0]*0.100000)-1.00000)
    algebraic[4] = 4.00000*exp((25.0000-algebraic[0])/18.0000)
    algebraic[7] = algebraic[1]/(algebraic[1]+algebraic[4])
    algebraic[8] = constants[4]*(power(algebraic[7], 3.00000))*states[1]*(states[0]-constants[3])
    algebraic[9] = constants[5]*states[2]*(states[0]-constants[3])
    algebraic[11] = (constants[9]*states[4]*(states[0]-constants[7]))/(0.500000+states[4])
    algebraic[10] = constants[8]*(power(states[3], 4.00000))*(states[0]-constants[7])
    algebraic[12] = constants[12]*(states[0]-constants[13])
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