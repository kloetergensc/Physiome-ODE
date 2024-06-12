# Size of variable arrays:
sizeAlgebraic = 5
sizeStates = 3
sizeConstants = 13
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_constants[0] = "POT in component non_muscle_autoregulatory_local_blood_flow_control (mmHg)"
    legend_constants[9] = "POD in component NM_autoregulatory_driving_force (mmHg)"
    legend_constants[1] = "POR in component parameter_values (mmHg)"
    legend_constants[10] = "POB in component NM_ST_sensitivity_control (mmHg)"
    legend_constants[2] = "POK in component parameter_values (dimensionless)"
    legend_algebraic[0] = "AR1 in component NM_ST_time_delay_and_damping (dimensionless)"
    legend_constants[3] = "A1K in component parameter_values (minute)"
    legend_states[0] = "AR1T in component NM_ST_time_delay_and_damping (dimensionless)"
    legend_constants[11] = "POA in component NM_I_sensitivity_control (mmHg)"
    legend_constants[4] = "PON in component parameter_values (dimensionless)"
    legend_algebraic[1] = "AR2 in component NM_I_time_delay_and_limit (dimensionless)"
    legend_constants[5] = "A2K in component parameter_values (minute)"
    legend_states[1] = "AR2T in component NM_I_time_delay_and_limit (dimensionless)"
    legend_constants[12] = "POC in component NM_LT_sensitivity_control (mmHg)"
    legend_constants[6] = "POZ in component parameter_values (dimensionless)"
    legend_algebraic[2] = "AR3 in component NM_LT_time_delay_and_limit (dimensionless)"
    legend_constants[7] = "A3K in component parameter_values (minute)"
    legend_states[2] = "AR3T in component NM_LT_time_delay_and_limit (dimensionless)"
    legend_algebraic[3] = "ARM1 in component total_NM_autoregulation (dimensionless)"
    legend_algebraic[4] = "ARM in component global_NM_blood_flow_autoregulation_output (dimensionless)"
    legend_constants[8] = "AUTOSN in component parameter_values (dimensionless)"
    legend_rates[0] = "d/dt AR1T in component NM_ST_time_delay_and_damping (dimensionless)"
    legend_rates[1] = "d/dt AR2T in component NM_I_time_delay_and_limit (dimensionless)"
    legend_rates[2] = "d/dt AR3T in component NM_LT_time_delay_and_limit (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 35.1148
    constants[1] = 35
    constants[2] = 0.1
    constants[3] = 0.5
    states[0] = 1.02127
    constants[4] = 0.1
    constants[5] = 60
    states[1] = 1.01179
    constants[6] = 2
    constants[7] = 40000
    states[2] = 1.1448
    constants[8] = 0.9
    constants[9] = constants[0]-constants[1]
    constants[10] = constants[9]*constants[2]+1.00000
    constants[11] = constants[4]*constants[9]+1.00000
    constants[12] = constants[6]*constants[9]+1.00000
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = (constants[10]*1.00000-states[0])/constants[3]
    rates[1] = (constants[11]*1.00000-states[1])/constants[5]
    rates[2] = (constants[12]*1.00000-states[2])/constants[7]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = custom_piecewise([less(states[0] , 0.500000), 0.500000 , True, states[0]])
    algebraic[1] = custom_piecewise([less(states[1] , 0.500000), 0.500000 , True, states[1]])
    algebraic[2] = custom_piecewise([less(states[2] , 0.300000), 0.300000 , True, states[2]])
    algebraic[3] = algebraic[0]*algebraic[1]*algebraic[2]
    algebraic[4] = (algebraic[3]-1.00000)*constants[8]+1.00000
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