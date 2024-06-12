# Size of variable arrays:
sizeAlgebraic = 0
sizeStates = 1
sizeConstants = 19
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_constants[0] = "ADHC in component thirst_drinking_and_salt_appetite (dimensionless)"
    legend_constants[1] = "ANM in component thirst_drinking_and_salt_appetite (dimensionless)"
    legend_constants[2] = "POT in component thirst_drinking_and_salt_appetite (mmHg)"
    legend_constants[12] = "STH in component effect_of_salt_appetite_stimulation_on_thirst (dimensionless)"
    legend_constants[3] = "ANMSLT in component parameter_values (dimensionless)"
    legend_constants[4] = "Z10 in component parameter_values (mmHg)"
    legend_constants[5] = "Z11 in component parameter_values (per_mmHg2)"
    legend_constants[10] = "ANMSML in component effect_of_salt_appetite_stimulation_on_thirst (dimensionless)"
    legend_constants[11] = "STH1 in component effect_of_salt_appetite_stimulation_on_thirst (dimensionless)"
    legend_constants[13] = "AHCM in component effect_of_antidiuretic_hormone_on_thirst (dimensionless)"
    legend_constants[6] = "AHTHM in component parameter_values (dimensionless)"
    legend_constants[14] = "ANMTH in component effect_of_angiotensin_on_thirst (dimensionless)"
    legend_constants[7] = "ANMTM in component parameter_values (dimensionless)"
    legend_states[0] = "TVD in component rate_of_fluid_intake (L_per_minute)"
    legend_constants[8] = "DR in component parameter_values (L_per_minute)"
    legend_constants[9] = "TVDDL in component parameter_values (minute)"
    legend_constants[16] = "AHTH in component rate_of_fluid_intake (dimensionless)"
    legend_constants[15] = "AHTH1 in component rate_of_fluid_intake (dimensionless)"
    legend_constants[18] = "TVZ in component rate_of_fluid_intake (L_per_minute)"
    legend_constants[17] = "TVZ1 in component rate_of_fluid_intake (L_per_minute)"
    legend_rates[0] = "d/dt TVD in component rate_of_fluid_intake (L_per_minute)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 1.0
    constants[1] = 0.987545
    constants[2] = 35.1148
    constants[3] = 2
    constants[4] = 45
    constants[5] = 0.01
    constants[6] = 2
    constants[7] = 1.5
    states[0] = 0.000980838
    constants[8] = 0
    constants[9] = 30
    constants[10] = (constants[1]-1.00000)*constants[3]+1.00000
    constants[11] = (power(constants[4]-constants[2], 2.00000))*constants[5]*constants[10]
    constants[12] = custom_piecewise([less(constants[11] , 0.800000), 0.800000 , greater(constants[11] , 8.00000), 8.00000 , True, constants[11]])
    constants[13] = (constants[0]-1.00000)*constants[6]+1.00000
    constants[14] = (constants[1]-1.00000)*constants[7]*0.00100000
    constants[15] = constants[13]*constants[12]*0.00100000
    constants[16] = custom_piecewise([less(constants[15] , 0.00000), 0.00000 , True, constants[15]])
    constants[17] = (constants[14]+constants[16])*1.00000
    constants[18] = custom_piecewise([less(constants[17] , 0.00000), 0.00000 , True, constants[17]])
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = ((constants[18]+constants[8])-states[0])/constants[9]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
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