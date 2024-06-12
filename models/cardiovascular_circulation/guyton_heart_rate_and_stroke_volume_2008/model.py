# Size of variable arrays:
sizeAlgebraic = 0
sizeStates = 0
sizeConstants = 10
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_constants[0] = "QLO in component heart_rate_and_stroke_volume (L_per_minute)"
    legend_constants[1] = "AUR in component heart_rate_and_stroke_volume (dimensionless)"
    legend_constants[2] = "PRA in component heart_rate_and_stroke_volume (mmHg)"
    legend_constants[3] = "HMD in component heart_rate_and_stroke_volume (dimensionless)"
    legend_constants[5] = "AUHR in component effect_of_autonomic_stimulation_on_HR (beats_per_minute)"
    legend_constants[6] = "PRHR in component effect_of_PRA_on_HR (beats_per_minute)"
    legend_constants[4] = "PR1LL in component parameter_values (mmHg)"
    legend_constants[7] = "HDHR in component effect_of_heart_deterioration_on_HR (dimensionless)"
    legend_constants[8] = "HR in component heart_rate (beats_per_minute)"
    legend_constants[9] = "SVO in component stroke_volume_output (litre)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 4.9943
    constants[1] = 1.30
    constants[2] = 0.00852183
    constants[3] = 1.0
    constants[4] = 0
    constants[5] = 72.0000*constants[1]
    constants[6] = (power(constants[4], 0.500000))*5.00000
    constants[7] = (constants[3]-1.00000)*0.500000+1.00000
    constants[8] = (constants[5]+constants[6])*constants[7]
    constants[9] = constants[0]/constants[8]
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