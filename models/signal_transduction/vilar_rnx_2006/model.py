# Size of variable arrays:
sizeAlgebraic = 12
sizeStates = 6
sizeConstants = 9
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_algebraic[11] = "v1 in component v1 (per_minute)"
    legend_algebraic[10] = "l in component l (dimensionless)"
    legend_states[0] = "RI in component RI (dimensionless)"
    legend_states[1] = "RII in component RII (dimensionless)"
    legend_constants[0] = "ka in component model_parameters (per_minute)"
    legend_algebraic[0] = "v2 in component v2 (per_minute)"
    legend_states[2] = "l_RI_RII in component l_RI_RII (dimensionless)"
    legend_constants[1] = "kcd in component model_parameters (per_minute)"
    legend_algebraic[1] = "v3 in component v3 (per_minute)"
    legend_constants[2] = "klid in component model_parameters (per_minute)"
    legend_algebraic[2] = "v4 in component v4 (per_minute)"
    legend_constants[3] = "ki in component model_parameters (per_minute)"
    legend_constants[7] = "v5 in component v5 (per_minute)"
    legend_constants[4] = "p_RI in component model_parameters (per_minute)"
    legend_algebraic[3] = "v6 in component v6 (per_minute)"
    legend_algebraic[4] = "v7 in component v7 (per_minute)"
    legend_algebraic[5] = "v8 in component v8 (per_minute)"
    legend_states[3] = "RI_endo in component RI_endo (dimensionless)"
    legend_constants[5] = "kr in component model_parameters (per_minute)"
    legend_algebraic[6] = "v9 in component v9 (per_minute)"
    legend_states[4] = "l_RI_RII_endo in component l_RI_RII_endo (dimensionless)"
    legend_constants[8] = "v10 in component v10 (per_minute)"
    legend_constants[6] = "p_RII in component model_parameters (per_minute)"
    legend_algebraic[7] = "v11 in component v11 (per_minute)"
    legend_algebraic[8] = "v12 in component v12 (per_minute)"
    legend_algebraic[9] = "v13 in component v13 (per_minute)"
    legend_states[5] = "RII_endo in component RII_endo (dimensionless)"
    legend_rates[2] = "d/dt l_RI_RII in component l_RI_RII (dimensionless)"
    legend_rates[0] = "d/dt RI in component RI (dimensionless)"
    legend_rates[1] = "d/dt RII in component RII (dimensionless)"
    legend_rates[4] = "d/dt l_RI_RII_endo in component l_RI_RII_endo (dimensionless)"
    legend_rates[3] = "d/dt RI_endo in component RI_endo (dimensionless)"
    legend_rates[5] = "d/dt RII_endo in component RII_endo (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 20.0
    states[1] = 20.0
    constants[0] = 1
    states[2] = 0.0
    constants[1] = 0.0277777778
    constants[2] = 0.25
    constants[3] = 0.333333333
    constants[4] = 8
    states[3] = 0.0
    constants[5] = 0.033333333
    states[4] = 40.0
    constants[6] = 4
    states[5] = 0.0
    constants[7] = constants[4]
    constants[8] = constants[6]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[4] = constants[3]*states[0]
    algebraic[5] = constants[5]*states[3]
    rates[3] = algebraic[4]-algebraic[5]
    algebraic[2] = constants[3]*states[2]
    algebraic[6] = constants[5]*states[4]
    rates[4] = algebraic[2]-algebraic[6]
    algebraic[8] = constants[3]*states[1]
    algebraic[9] = constants[5]*states[5]
    rates[5] = algebraic[8]-algebraic[9]
    algebraic[10] = custom_piecewise([greater_equal(voi , 2500.00), 0.0100000 , True, 3.00000e-05])
    algebraic[11] = constants[0]*algebraic[10]*states[0]*states[1]
    algebraic[0] = constants[1]*states[2]
    algebraic[1] = constants[2]*states[2]
    rates[2] = algebraic[11]-(algebraic[0]+algebraic[1]+algebraic[2])
    algebraic[3] = constants[1]*states[0]
    rates[0] = (constants[7]+algebraic[5]+algebraic[6])-(algebraic[11]+algebraic[3]+algebraic[4])
    algebraic[7] = constants[1]*states[1]
    rates[1] = (algebraic[6]+constants[8]+algebraic[9])-(algebraic[11]+algebraic[7]+algebraic[8])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[4] = constants[3]*states[0]
    algebraic[5] = constants[5]*states[3]
    algebraic[2] = constants[3]*states[2]
    algebraic[6] = constants[5]*states[4]
    algebraic[8] = constants[3]*states[1]
    algebraic[9] = constants[5]*states[5]
    algebraic[10] = custom_piecewise([greater_equal(voi , 2500.00), 0.0100000 , True, 3.00000e-05])
    algebraic[11] = constants[0]*algebraic[10]*states[0]*states[1]
    algebraic[0] = constants[1]*states[2]
    algebraic[1] = constants[2]*states[2]
    algebraic[3] = constants[1]*states[0]
    algebraic[7] = constants[1]*states[1]
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