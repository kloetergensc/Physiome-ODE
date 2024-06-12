# Size of variable arrays:
sizeAlgebraic = 3
sizeStates = 4
sizeConstants = 7
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (day)"
    legend_constants[0] = "lambda in component model_parameters (cells_per_ml)"
    legend_constants[1] = "delta_1 in component model_parameters (cells_per_day)"
    legend_constants[2] = "delta in component model_parameters (cells_per_day)"
    legend_constants[3] = "np in component model_parameters (dimensionless)"
    legend_constants[4] = "c in component model_parameters (virons_per_day)"
    legend_constants[5] = "k in component model_parameters (ml_per_virons_per_day)"
    legend_constants[6] = "N in component model_parameters (virons_per_cell)"
    legend_states[0] = "T in component uninfected_T_cells (cells_per_ml)"
    legend_states[1] = "VI in component infectious_virus (virons_per_ml)"
    legend_states[2] = "T_star in component infected_T_cells (cells_per_ml)"
    legend_algebraic[0] = "log_VI in component infectious_virus (dimensionless)"
    legend_states[3] = "VNI in component non_infectious_virus (virons_per_ml)"
    legend_algebraic[1] = "virus_total in component total_virus (virons_per_ml)"
    legend_algebraic[2] = "log_virus_total in component total_virus (dimensionless)"
    legend_rates[0] = "d/dt T in component uninfected_T_cells (cells_per_ml)"
    legend_rates[2] = "d/dt T_star in component infected_T_cells (cells_per_ml)"
    legend_rates[1] = "d/dt VI in component infectious_virus (virons_per_ml)"
    legend_rates[3] = "d/dt VNI in component non_infectious_virus (virons_per_ml)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 5.0
    constants[1] = 0.03
    constants[2] = 0.5
    constants[3] = 1.0
    constants[4] = 3
    constants[5] = 3.43e-5
    constants[6] = 480
    states[0] = 180.0
    states[1] = 134e3
    states[2] = 3.6
    states[3] = 0
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = (constants[0]*1.00000-(constants[1]*states[0])/1.00000)-constants[5]*states[1]*states[0]
    rates[2] = constants[5]*states[1]*states[0]-(constants[2]/1.00000)*states[2]
    rates[1] = ((1.00000-constants[3])*constants[6]*constants[2]*states[2])/1.00000-(constants[4]*states[1])/1.00000
    rates[3] = (constants[3]*constants[6]*constants[2]*states[2])/1.00000-(constants[4]*states[3])/1.00000
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = log(states[1]/1.00000, 10)
    algebraic[1] = states[1]+states[3]
    algebraic[2] = log(algebraic[1]/1.00000, 10)
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