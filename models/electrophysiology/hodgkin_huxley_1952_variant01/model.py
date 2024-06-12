# Size of variable arrays:
sizeAlgebraic = 10
sizeStates = 4
sizeConstants = 8
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
    legend_constants[0] = "E_R in component membrane (millivolt)"
    legend_constants[1] = "Cm in component membrane (microF_per_cm2)"
    legend_algebraic[4] = "i_Na in component sodium_channel (microA_per_cm2)"
    legend_algebraic[8] = "i_K in component potassium_channel (microA_per_cm2)"
    legend_algebraic[9] = "i_L in component leakage_current (microA_per_cm2)"
    legend_algebraic[0] = "i_Stim in component membrane (microA_per_cm2)"
    legend_constants[2] = "g_Na in component sodium_channel (milliS_per_cm2)"
    legend_constants[5] = "E_Na in component sodium_channel (millivolt)"
    legend_states[1] = "m in component sodium_channel_m_gate (dimensionless)"
    legend_states[2] = "h in component sodium_channel_h_gate (dimensionless)"
    legend_algebraic[1] = "alpha_m in component sodium_channel_m_gate (per_millisecond)"
    legend_algebraic[5] = "beta_m in component sodium_channel_m_gate (per_millisecond)"
    legend_algebraic[2] = "alpha_h in component sodium_channel_h_gate (per_millisecond)"
    legend_algebraic[6] = "beta_h in component sodium_channel_h_gate (per_millisecond)"
    legend_constants[3] = "g_K in component potassium_channel (milliS_per_cm2)"
    legend_constants[6] = "E_K in component potassium_channel (millivolt)"
    legend_states[3] = "n in component potassium_channel_n_gate (dimensionless)"
    legend_algebraic[3] = "alpha_n in component potassium_channel_n_gate (per_millisecond)"
    legend_algebraic[7] = "beta_n in component potassium_channel_n_gate (per_millisecond)"
    legend_constants[4] = "g_L in component leakage_current (milliS_per_cm2)"
    legend_constants[7] = "E_L in component leakage_current (millivolt)"
    legend_rates[0] = "d/dt V in component membrane (millivolt)"
    legend_rates[1] = "d/dt m in component sodium_channel_m_gate (dimensionless)"
    legend_rates[2] = "d/dt h in component sodium_channel_h_gate (dimensionless)"
    legend_rates[3] = "d/dt n in component potassium_channel_n_gate (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0
    constants[0] = 0
    constants[1] = 1
    constants[2] = 120
    states[1] = 0.05
    states[2] = 0.6
    constants[3] = 36
    states[3] = 0.325
    constants[4] = 0.3
    constants[5] = constants[0]-115.000
    constants[6] = constants[0]+12.0000
    constants[7] = constants[0]-10.6130
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[1] = (0.100000*(states[0]+25.0000))/(exp((states[0]+25.0000)/10.0000)-1.00000)
    algebraic[5] = 4.00000*exp(states[0]/18.0000)
    rates[1] = algebraic[1]*(1.00000-states[1])-algebraic[5]*states[1]
    algebraic[2] = 0.0700000*exp(states[0]/20.0000)
    algebraic[6] = 1.00000/(exp((states[0]+30.0000)/10.0000)+1.00000)
    rates[2] = algebraic[2]*(1.00000-states[2])-algebraic[6]*states[2]
    algebraic[3] = (0.0100000*(states[0]+10.0000))/(exp((states[0]+10.0000)/10.0000)-1.00000)
    algebraic[7] = 0.125000*exp(states[0]/80.0000)
    rates[3] = algebraic[3]*(1.00000-states[3])-algebraic[7]*states[3]
    algebraic[4] = constants[2]*(power(states[1], 3.00000))*states[2]*(states[0]-constants[5])
    algebraic[8] = constants[3]*(power(states[3], 4.00000))*(states[0]-constants[6])
    algebraic[9] = constants[4]*(states[0]-constants[7])
    algebraic[0] = custom_piecewise([greater_equal(voi , 10.0000) & less_equal(voi , 10.5000), -20.0000 , True, 0.00000])
    rates[0] = -(-algebraic[0]+algebraic[4]+algebraic[8]+algebraic[9])/constants[1]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[1] = (0.100000*(states[0]+25.0000))/(exp((states[0]+25.0000)/10.0000)-1.00000)
    algebraic[5] = 4.00000*exp(states[0]/18.0000)
    algebraic[2] = 0.0700000*exp(states[0]/20.0000)
    algebraic[6] = 1.00000/(exp((states[0]+30.0000)/10.0000)+1.00000)
    algebraic[3] = (0.0100000*(states[0]+10.0000))/(exp((states[0]+10.0000)/10.0000)-1.00000)
    algebraic[7] = 0.125000*exp(states[0]/80.0000)
    algebraic[4] = constants[2]*(power(states[1], 3.00000))*states[2]*(states[0]-constants[5])
    algebraic[8] = constants[3]*(power(states[3], 4.00000))*(states[0]-constants[6])
    algebraic[9] = constants[4]*(states[0]-constants[7])
    algebraic[0] = custom_piecewise([greater_equal(voi , 10.0000) & less_equal(voi , 10.5000), -20.0000 , True, 0.00000])
    return algebraic

def custom_piecewise(cases):
    """Compute result of a piecewise function"""
    return select(cases[0::2],cases[1::2])

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