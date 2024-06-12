# Size of variable arrays:
sizeAlgebraic = 1
sizeStates = 1
sizeConstants = 6
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (day)"
    legend_states[0] = "p in component serum_PSA_dynamics (ng_per_mm3)"
    legend_algebraic[0] = "Vc in component serum_PSA_dynamics (mm3)"
    legend_constants[0] = "Vo in component serum_PSA_dynamics (mm3)"
    legend_constants[1] = "Vh in component serum_PSA_dynamics (mm3)"
    legend_constants[2] = "beta_h in component serum_PSA_dynamics (ng_per_mm3_per_day)"
    legend_constants[3] = "beta_c in component serum_PSA_dynamics (ng_per_mm3_per_day)"
    legend_constants[4] = "gamma in component serum_PSA_dynamics (per_day)"
    legend_constants[5] = "rho in component serum_PSA_dynamics (per_day)"
    legend_rates[0] = "d/dt p in component serum_PSA_dynamics (ng_per_mm3)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.0
    constants[0] = 20.0
    constants[1] = 0.0
    constants[2] = 0.0
    constants[3] = 1.7210
    constants[4] = 1.2896
    constants[5] = 0.0655
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[0] = constants[0]*exp(constants[5]*voi)
    rates[0] = (1.00000*constants[2]*constants[1]+1.00000*constants[3]*algebraic[0])-constants[4]*states[0]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = constants[0]*exp(constants[5]*voi)
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