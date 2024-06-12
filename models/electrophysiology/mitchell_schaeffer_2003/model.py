# Size of variable arrays:
sizeAlgebraic = 3
sizeStates = 2
sizeConstants = 10
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (ms)"
    legend_algebraic[0] = "J_stim in component J_stim (per_ms)"
    legend_constants[0] = "IstimStart in component J_stim (ms)"
    legend_constants[1] = "IstimEnd in component J_stim (ms)"
    legend_constants[2] = "IstimAmplitude in component J_stim (per_ms)"
    legend_constants[3] = "IstimPeriod in component J_stim (ms)"
    legend_constants[4] = "IstimPulseDuration in component J_stim (ms)"
    legend_states[0] = "Vm in component membrane (dimensionless)"
    legend_algebraic[1] = "J_in in component J_in (per_ms)"
    legend_algebraic[2] = "J_out in component J_out (per_ms)"
    legend_constants[5] = "tau_in in component J_in (ms)"
    legend_states[1] = "h in component J_in_h_gate (dimensionless)"
    legend_constants[6] = "tau_open in component J_in_h_gate (ms)"
    legend_constants[7] = "tau_close in component J_in_h_gate (ms)"
    legend_constants[8] = "V_gate in component J_in_h_gate (dimensionless)"
    legend_constants[9] = "tau_out in component J_out (ms)"
    legend_rates[0] = "d/dt Vm in component membrane (dimensionless)"
    legend_rates[1] = "d/dt h in component J_in_h_gate (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 0
    constants[1] = 50000
    constants[2] = 0.2
    constants[3] = 500
    constants[4] = 1
    states[0] = 0.00000820413566106744
    constants[5] = 0.3
    states[1] = 0.8789655121804799
    constants[6] = 120.0
    constants[7] = 150.0
    constants[8] = 0.13
    constants[9] = 6.0
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[1] = custom_piecewise([less(states[0] , constants[8]), (1.00000-states[1])/constants[6] , True, -states[1]/constants[7]])
    algebraic[0] = custom_piecewise([greater_equal(voi , constants[0]) & less_equal(voi , constants[1]) & less_equal((voi-constants[0])-floor((voi-constants[0])/constants[3])*constants[3] , constants[4]), constants[2] , True, 0.00000])
    algebraic[1] = (states[1]*((power(states[0], 2.00000))*(1.00000-states[0])))/constants[5]
    algebraic[2] = -(states[0]/constants[9])
    rates[0] = algebraic[1]+algebraic[2]+algebraic[0]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = custom_piecewise([greater_equal(voi , constants[0]) & less_equal(voi , constants[1]) & less_equal((voi-constants[0])-floor((voi-constants[0])/constants[3])*constants[3] , constants[4]), constants[2] , True, 0.00000])
    algebraic[1] = (states[1]*((power(states[0], 2.00000))*(1.00000-states[0])))/constants[5]
    algebraic[2] = -(states[0]/constants[9])
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