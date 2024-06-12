# Size of variable arrays:
sizeAlgebraic = 0
sizeStates = 0
sizeConstants = 21
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_constants[16] = "v_cystathionine in component v_cystathionine (flux)"
    legend_constants[0] = "Cys in component Cys (micromolar)"
    legend_constants[1] = "CGS in component CGS (micromolar)"
    legend_constants[2] = "Pi in component Pi (micromolar)"
    legend_constants[3] = "Phser in component Phser (micromolar)"
    legend_constants[15] = "Km_CGS_app_Cys in component v_cystathionine (micromolar)"
    legend_constants[4] = "Km_CGS_Cys in component v_cystathionine (micromolar)"
    legend_constants[5] = "kcat_CGS in component v_cystathionine (first_order_rate_constant)"
    legend_constants[14] = "kcat_CGS_app_Cys in component v_cystathionine (first_order_rate_constant)"
    legend_constants[6] = "Km_CGS_Phser in component v_cystathionine (micromolar)"
    legend_constants[7] = "Ki_CGS_Pi in component v_cystathionine (micromolar)"
    legend_constants[19] = "v_Thr in component v_Thr (flux)"
    legend_constants[8] = "TS in component TS (micromolar)"
    legend_constants[9] = "AdoMet in component AdoMet (micromolar)"
    legend_constants[18] = "Km_TS in component v_Thr (micromolar)"
    legend_constants[10] = "kcat_TS_noAdoMet in component v_Thr (first_order_rate_constant)"
    legend_constants[11] = "kcat_TS_AdoMet in component v_Thr (first_order_rate_constant)"
    legend_constants[17] = "kcat_TS in component v_Thr (first_order_rate_constant)"
    legend_constants[12] = "K1K2 in component v_Thr (micromolar2)"
    legend_constants[13] = "Ki_TS_Pi in component v_Thr (micromolar)"
    legend_constants[20] = "J_Phser in component J_Phser (flux)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 15.0
    constants[1] = 0.7
    constants[2] = 10000.0
    constants[3] = 500.0
    constants[4] = 460.0
    constants[5] = 30.0
    constants[6] = 2500.0
    constants[7] = 2500.0
    constants[8] = 5.0
    constants[9] = 20.0
    constants[10] = 0.42
    constants[11] = 3.5
    constants[12] = 73.0
    constants[13] = 1000.0
    constants[14] = constants[5]/(1.00000+(constants[6]/constants[3])*(1.00000+constants[2]/constants[7]))
    constants[15] = constants[4]/(1.00000+(constants[6]/constants[3])*(1.00000+constants[2]/constants[7]))
    constants[16] = (constants[14]*constants[1]*constants[0])/(constants[15]+constants[0])
    constants[17] = (constants[10]+constants[11]*((power(constants[9], 2.00000))/constants[12]))/(1.00000+(power(constants[9], 2.00000))/constants[12])
    constants[18] = ((250.000*((1.00000+constants[9]/0.500000)/(1.00000+constants[9]/1.10000)))/(1.00000+(power(constants[9], 2.00000))/140.000))*(1.00000+constants[2]/constants[13])
    constants[19] = (constants[8]*constants[17]*constants[3])/(constants[18]+constants[3])
    constants[20] = constants[16]+constants[19]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
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