# Size of variable arrays:
sizeAlgebraic = 4
sizeStates = 4
sizeConstants = 10
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_algebraic[0] = "F in component equations (force)"
    legend_states[0] = "R_on in component equations (dimensionless)"
    legend_states[1] = "A in component equations (dimensionless)"
    legend_constants[0] = "alpha in component equations (dimensionless)"
    legend_algebraic[1] = "D in component equations (dimensionless)"
    legend_algebraic[3] = "k_XB in component equations (per_second)"
    legend_constants[1] = "k_a in component equations (per_second)"
    legend_states[2] = "x in component equations (um)"
    legend_constants[2] = "x_0 in component undefined_parameters (um)"
    legend_constants[3] = "epsilon in component undefined_parameters (force_per_um)"
    legend_constants[4] = "beta in component undefined_parameters (per_um)"
    legend_constants[5] = "g in component undefined_parameters (per_second)"
    legend_constants[6] = "f in component undefined_parameters (per_second)"
    legend_constants[7] = "k_off in component undefined_parameters (per_second)"
    legend_constants[8] = "k_on in component undefined_parameters (per_second)"
    legend_states[3] = "L in component parameters_stelzer_et_al (um)"
    legend_constants[9] = "L_0 in component parameters_stelzer_et_al (um)"
    legend_algebraic[2] = "dL_dt in component parameters_stelzer_et_al (um_per_second)"
    legend_rates[0] = "d/dt R_on in component equations (dimensionless)"
    legend_rates[1] = "d/dt A in component equations (dimensionless)"
    legend_rates[2] = "d/dt x in component equations (um)"
    legend_rates[3] = "d/dt L in component parameters_stelzer_et_al (um)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 1
    states[1] = 0
    constants[0] = 0.1
    constants[1] = 0
    states[2] = 1
    constants[2] = 1
    constants[3] = 1
    constants[4] = 2
    constants[5] = 1
    constants[6] = 1
    constants[7] = 1
    constants[8] = 1
    states[3] = 2.12
    constants[9] = 2.12
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[1] = 1.00000-states[1]
    rates[1] = constants[6]*algebraic[1]*states[0]-constants[5]*states[1]
    algebraic[2] = custom_piecewise([less(0.00100000 , voi) & less_equal(voi , 0.00300000), 10.6000 , True, 0.00000])
    rates[2] = -constants[5]*(states[2]-constants[2])+algebraic[2]
    rates[3] = algebraic[2]
    algebraic[3] = constants[1]*states[1]
    rates[0] = -((constants[7]+algebraic[3]+constants[0]*constants[8])/(1.00000+constants[0])+constants[6]*algebraic[1])*states[0]+(constants[5]-(algebraic[3]+constants[0]*constants[8])/(1.00000+constants[0]))*states[1]+(algebraic[3]+(constants[0]*constants[8])/(1.00000+constants[0]))*constants[4]*(states[3]-constants[9])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[1] = 1.00000-states[1]
    algebraic[2] = custom_piecewise([less(0.00100000 , voi) & less_equal(voi , 0.00300000), 10.6000 , True, 0.00000])
    algebraic[3] = constants[1]*states[1]
    algebraic[0] = states[1]*constants[3]*states[2]
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