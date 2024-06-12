# Size of variable arrays:
sizeAlgebraic = 2
sizeStates = 1
sizeConstants = 10
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_constants[0] = "PLA in component atrial_natriuretic_peptide (mmHg)"
    legend_constants[1] = "PRA in component atrial_natriuretic_peptide (mmHg)"
    legend_constants[8] = "ANP in component total_ANP_secreted (dimensionless)"
    legend_constants[6] = "ANPL in component total_ANP_secreted (dimensionless)"
    legend_constants[7] = "ANPR2 in component total_ANP_secreted (dimensionless)"
    legend_constants[9] = "ANP1 in component ANP_into_circulation (dimensionless)"
    legend_constants[2] = "ANPKNS in component parameter_values (dimensionless)"
    legend_constants[3] = "ANPINF in component parameter_values (dimensionless)"
    legend_states[0] = "ANPC in component ANP_in_plasma (dimensionless)"
    legend_constants[4] = "ANPTC in component parameter_values (minute)"
    legend_algebraic[1] = "ANPX in component ANP_effect_on_renal_afferent_arteriolar_resistance (dimensionless)"
    legend_constants[5] = "ANPXUL in component parameter_values (dimensionless)"
    legend_algebraic[0] = "ANPX1 in component ANP_effect_on_renal_afferent_arteriolar_resistance (dimensionless)"
    legend_rates[0] = "d/dt ANPC in component ANP_in_plasma (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 2
    constants[1] = 0.00852183
    constants[2] = 0
    constants[3] = 0
    states[0] = 1.0
    constants[4] = 4
    constants[5] = 10
    constants[6] = custom_piecewise([less((constants[0]-1.00000)*1.00000 , 0.00000), 0.00000 , True, (constants[0]-1.00000)*1.00000])
    constants[7] = custom_piecewise([less((constants[1]+1.00000)*2.00000 , 0.00000), 0.00000 , True, (constants[1]+1.00000)*2.00000])
    constants[8] = (constants[6]+constants[7])/3.00000
    constants[9] = custom_piecewise([greater(constants[2] , 0.00000), constants[2] , True, constants[8]+constants[3]])
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = (constants[9]-states[0])/constants[4]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = constants[5]-constants[5]/(0.555556*(1.00000+states[0]))
    algebraic[1] = custom_piecewise([less(algebraic[0] , -1.00000), -1.00000 , True, algebraic[0]])
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