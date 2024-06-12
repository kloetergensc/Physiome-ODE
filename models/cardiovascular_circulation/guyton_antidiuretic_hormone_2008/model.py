# Size of variable arrays:
sizeAlgebraic = 4
sizeStates = 1
sizeConstants = 17
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_constants[0] = "CNA in component antidiuretic_hormone (monovalent_mEq_per_litre)"
    legend_constants[1] = "PA1 in component antidiuretic_hormone (mmHg)"
    legend_constants[12] = "ADHNA in component osmotic_control_of_ADH_secretion (dimensionless)"
    legend_constants[2] = "CNR in component parameter_values (monovalent_mEq_per_litre)"
    legend_constants[11] = "ADHNA1 in component osmotic_control_of_ADH_secretion (dimensionless)"
    legend_constants[14] = "ADHPR in component pressure_control_of_ADH_secretion (dimensionless)"
    legend_constants[3] = "ADHPUL in component parameter_values (mmHg)"
    legend_constants[4] = "ADHPAM in component parameter_values (per_mmHg2)"
    legend_constants[13] = "ADHPA in component pressure_control_of_ADH_secretion (mmHg)"
    legend_constants[16] = "ADH in component total_ADH_secretion (dimensionless)"
    legend_constants[5] = "ADHINF in component parameter_values (dimensionless)"
    legend_constants[15] = "ADH1 in component total_ADH_secretion (dimensionless)"
    legend_states[0] = "ADHC in component ADH_in_blood (dimensionless)"
    legend_constants[6] = "ADHTC in component parameter_values (minute)"
    legend_algebraic[2] = "ADHMV in component ADH_effect_on_nonrenal_vascular_resistance (dimensionless)"
    legend_constants[7] = "ADHVUL in component parameter_values (dimensionless)"
    legend_constants[8] = "ADHVLL in component parameter_values (dimensionless)"
    legend_algebraic[0] = "ADHMV1 in component ADH_effect_on_nonrenal_vascular_resistance (dimensionless)"
    legend_algebraic[3] = "ADHMK in component ADH_effect_on_kidney (dimensionless)"
    legend_constants[9] = "ADHKLL in component parameter_values (dimensionless)"
    legend_constants[10] = "ADHKUL in component parameter_values (dimensionless)"
    legend_algebraic[1] = "ADHMK1 in component ADH_effect_on_kidney (dimensionless)"
    legend_rates[0] = "d/dt ADHC in component ADH_in_blood (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 142.035
    constants[1] = 103.525
    constants[2] = 139
    constants[3] = 85
    constants[4] = 0.3
    constants[5] = 0
    states[0] = 1.0
    constants[6] = 15
    constants[7] = 2.5
    constants[8] = 0.93617
    constants[9] = 0.2
    constants[10] = 5
    constants[11] = (constants[0]-constants[2])/(142.000-constants[2])
    constants[12] = custom_piecewise([less(constants[11] , 0.00000), 0.00000 , True, constants[11]])
    constants[13] = custom_piecewise([greater(constants[1] , constants[3]), constants[3] , True, constants[1]])
    constants[14] = (power(constants[3]-constants[13], 2.00000))*constants[4]
    constants[15] = constants[12]+constants[14]+constants[5]
    constants[16] = custom_piecewise([less(constants[15] , 0.00000), 0.00000 , True, constants[15]])
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = (constants[16]-states[0])/constants[6]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = constants[7]-(constants[7]-1.00000)/(((constants[8]-1.00000)/(constants[8]-constants[7]))*(states[0]-1.00000)+1.00000)
    algebraic[1] = constants[10]-(constants[10]-1.00000)/(((constants[9]-1.00000)/(constants[9]-constants[10]))*(states[0]-1.00000)+1.00000)
    algebraic[2] = custom_piecewise([less(algebraic[0] , constants[8]), constants[8] , True, algebraic[0]])
    algebraic[3] = custom_piecewise([less(algebraic[1] , constants[9]), constants[9] , True, algebraic[1]])
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