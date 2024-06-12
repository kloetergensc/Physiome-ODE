# Size of variable arrays:
sizeAlgebraic = 6
sizeStates = 5
sizeConstants = 20
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_states[0] = "C in component C (dimensionless)"
    legend_algebraic[0] = "Ctot in component C (dimensionless)"
    legend_constants[0] = "vi in component model_parameters (per_minute)"
    legend_constants[1] = "k1 in component model_parameters (per_minute)"
    legend_states[1] = "X in component X (dimensionless)"
    legend_constants[2] = "K5 in component model_parameters (dimensionless)"
    legend_constants[3] = "kd in component model_parameters (per_minute)"
    legend_states[2] = "Z in component Z (dimensionless)"
    legend_states[3] = "M in component M (dimensionless)"
    legend_constants[19] = "M_ in component M (dimensionless)"
    legend_algebraic[1] = "V1 in component model_parameters (per_minute)"
    legend_constants[4] = "K1 in component model_parameters (dimensionless)"
    legend_constants[5] = "V2 in component model_parameters (per_minute)"
    legend_constants[6] = "K2 in component model_parameters (dimensionless)"
    legend_algebraic[2] = "X_ in component X (dimensionless)"
    legend_algebraic[4] = "V3 in component model_parameters (per_minute)"
    legend_constants[7] = "K3 in component model_parameters (dimensionless)"
    legend_constants[8] = "V4 in component model_parameters (per_minute)"
    legend_constants[9] = "K4 in component model_parameters (dimensionless)"
    legend_states[4] = "Y in component Y (dimensionless)"
    legend_algebraic[3] = "Ytot in component Y (dimensionless)"
    legend_constants[10] = "vs in component model_parameters (per_minute)"
    legend_constants[11] = "d1 in component model_parameters (per_minute)"
    legend_constants[12] = "a1 in component model_parameters (per_minute)"
    legend_constants[13] = "a2 in component model_parameters (per_minute)"
    legend_constants[14] = "alpha in component model_parameters (dimensionless)"
    legend_algebraic[5] = "BP in component BP (dimensionless)"
    legend_constants[15] = "Kd in component model_parameters (dimensionless)"
    legend_constants[16] = "K6 in component model_parameters (dimensionless)"
    legend_constants[17] = "V1_dash in component model_parameters (per_minute)"
    legend_constants[18] = "V3_dash in component model_parameters (per_minute)"
    legend_rates[0] = "d/dt C in component C (dimensionless)"
    legend_rates[3] = "d/dt M in component M (dimensionless)"
    legend_rates[1] = "d/dt X in component X (dimensionless)"
    legend_rates[4] = "d/dt Y in component Y (dimensionless)"
    legend_rates[2] = "d/dt Z in component Z (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.01
    constants[0] = 0.1
    constants[1] = 0.5
    states[1] = 0.01
    constants[2] = 0.02
    constants[3] = 0.02
    states[2] = 0.28
    states[3] = 0.01
    constants[4] = 0.02
    constants[5] = 0.25
    constants[6] = 0.02
    constants[7] = 0.1
    constants[8] = 0.1
    constants[9] = 0.1
    states[4] = 0.01
    constants[10] = 0.1
    constants[11] = 0.05
    constants[12] = 1.5
    constants[13] = 1.5
    constants[14] = 0.1
    constants[15] = 1
    constants[16] = 0.3
    constants[17] = 0.75
    constants[18] = 0.3
    constants[19] = -1.00000
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = (constants[0]-(constants[1]*states[1]*states[0])/(states[0]+constants[2]))-constants[3]*states[0]
    rates[4] = ((constants[10]-constants[11]*states[4])-constants[12]*states[0]*states[4])+(constants[13]+constants[14]*constants[3])*states[2]
    rates[2] = constants[12]*states[0]*states[4]-(constants[13]+constants[14]*constants[3]+constants[14]*constants[11])*states[2]
    algebraic[1] = (states[0]*constants[17])/(states[0]+constants[16])
    rates[3] = (algebraic[1]*constants[19])/(constants[19]+constants[4])-(constants[5]*states[3])/(states[3]+constants[6])
    algebraic[2] = 1.00000-states[1]
    algebraic[4] = states[3]*constants[18]
    rates[1] = (algebraic[4]*algebraic[2])/(algebraic[2]+constants[7])-(constants[8]*states[1])/(states[1]+constants[9])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[1] = (states[0]*constants[17])/(states[0]+constants[16])
    algebraic[2] = 1.00000-states[1]
    algebraic[4] = states[3]*constants[18]
    algebraic[0] = states[0]+states[2]
    algebraic[3] = states[4]+states[2]
    algebraic[5] = 1.00000+(constants[15]*algebraic[3])/(power(states[0]+constants[15], 2.00000))
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