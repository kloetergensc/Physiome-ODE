# Size of variable arrays:
sizeAlgebraic = 2
sizeStates = 7
sizeConstants = 13
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (day)"
    legend_constants[9] = "T in component T (per_ml)"
    legend_constants[0] = "k in component kinetic_parameters (ml_per_day)"
    legend_constants[1] = "p in component kinetic_parameters (first_order_rate_constant)"
    legend_constants[2] = "c in component kinetic_parameters (first_order_rate_constant)"
    legend_constants[3] = "delta in component kinetic_parameters (first_order_rate_constant)"
    legend_states[0] = "I in component I (per_ml)"
    legend_constants[10] = "I_0 in component I (per_ml)"
    legend_constants[11] = "k_ in component kinetic_parameters (ml_per_day)"
    legend_states[1] = "E4 in component E4 (per_ml)"
    legend_constants[4] = "VI_0 in component VI (per_ml)"
    legend_states[2] = "VI in component VI (per_ml)"
    legend_algebraic[0] = "h in component Heavyside_function (dimensionless)"
    legend_states[3] = "VNI in component VNI (per_ml)"
    legend_algebraic[1] = "V in component virus_total (per_ml)"
    legend_states[4] = "E1 in component E1 (per_ml)"
    legend_constants[12] = "b_ in component kinetic_parameters (day)"
    legend_states[5] = "E2 in component E2 (per_ml)"
    legend_states[6] = "E3 in component E3 (per_ml)"
    legend_constants[5] = "tau_p in component Heavyside_function (day)"
    legend_constants[6] = "b in component kinetic_parameters (day)"
    legend_constants[7] = "m in component kinetic_parameters (first_order_rate_constant)"
    legend_constants[8] = "n in component kinetic_parameters (dimensionless)"
    legend_rates[0] = "d/dt I in component I (per_ml)"
    legend_rates[2] = "d/dt VI in component VI (per_ml)"
    legend_rates[3] = "d/dt VNI in component VNI (per_ml)"
    legend_rates[4] = "d/dt E1 in component E1 (per_ml)"
    legend_rates[5] = "d/dt E2 in component E2 (per_ml)"
    legend_rates[6] = "d/dt E3 in component E3 (per_ml)"
    legend_rates[1] = "d/dt E4 in component E4 (per_ml)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 2.4e-5
    constants[1] = 774
    constants[2] = 3
    constants[3] = 0.5
    states[0] = 0.1
    states[1] = 0
    constants[4] = 200000
    states[2] = 200000
    states[3] = 0
    states[4] = 0
    states[5] = 0
    states[6] = 0
    constants[5] = 0
    constants[6] = 0.25
    constants[7] = 0.01
    constants[8] = 4
    constants[9] = (constants[2]*constants[3])/(constants[0]*constants[1])
    constants[10] = (constants[2]/constants[1])*constants[4]
    constants[11] = constants[0]/(power(1.00000+constants[7]*constants[6], constants[8]))
    constants[12] = constants[6]/(1.00000+constants[7]*constants[6])
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = constants[11]*constants[9]*states[1]-constants[3]*states[0]
    rates[4] = (states[2]-states[4])/constants[12]
    rates[5] = (states[4]-states[5])/constants[12]
    rates[6] = (states[5]-states[6])/constants[12]
    rates[1] = (states[6]-states[1])/constants[12]
    algebraic[0] = custom_piecewise([less(voi , constants[5]), 0.00000 , True, 1.00000])
    rates[2] = (1.00000-algebraic[0])*constants[1]*states[0]-constants[2]*states[2]
    rates[3] = algebraic[0]*constants[1]*states[0]-constants[2]*states[3]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = custom_piecewise([less(voi , constants[5]), 0.00000 , True, 1.00000])
    algebraic[1] = states[2]+states[3]
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