# Size of variable arrays:
sizeAlgebraic = 13
sizeStates = 2
sizeConstants = 8
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_constants[0] = "PPC in component pulmonary_fluid_dynamics (mmHg)"
    legend_constants[1] = "PPA in component pulmonary_fluid_dynamics (mmHg)"
    legend_constants[2] = "PLA in component pulmonary_fluid_dynamics (mmHg)"
    legend_constants[3] = "CPP in component pulmonary_fluid_dynamics (gram_per_L)"
    legend_constants[4] = "RPV in component pulmonary_fluid_dynamics (mmHg_minute_per_L)"
    legend_constants[5] = "RPA in component pulmonary_fluid_dynamics (mmHg_minute_per_L)"
    legend_constants[7] = "PCP in component pulmonary_capillary_pressure (mmHg)"
    legend_algebraic[4] = "POS in component colloid_osmotic_pressure_of_pulmonary_interstitium (mmHg)"
    legend_algebraic[1] = "PPI in component pulmonary_interstitial_fluid_pressure (mmHg)"
    legend_algebraic[5] = "PFI in component fluid_filtration_into_pulmonary_interstitium (L_per_minute)"
    legend_constants[6] = "CPF in component parameter_values (L_per_minute_per_mmHg)"
    legend_algebraic[7] = "PLF in component lung_lymphatic_protein_flow (L_per_minute)"
    legend_algebraic[10] = "DFP in component pulmonary_interstitial_free_fluid_volume (L_per_minute)"
    legend_algebraic[0] = "VPF in component pulmonary_interstitial_free_fluid_volume (litre)"
    legend_algebraic[8] = "DFZ in component pulmonary_interstitial_free_fluid_volume (L_per_minute)"
    legend_states[0] = "VPF1 in component pulmonary_interstitial_free_fluid_volume (litre)"
    legend_algebraic[9] = "PPO in component lung_lymphatic_protein_flow (gram_per_minute)"
    legend_algebraic[6] = "PPN in component protein_leakage_into_pulmonary_interstitium (gram_per_minute)"
    legend_algebraic[12] = "PPD in component concentration_of_protein_in_pulmonary_interstitium (gram_per_minute)"
    legend_algebraic[3] = "CPN in component concentration_of_protein_in_pulmonary_interstitium (gram_per_L)"
    legend_algebraic[11] = "PPZ in component concentration_of_protein_in_pulmonary_interstitium (gram_per_minute)"
    legend_states[1] = "PPR1 in component concentration_of_protein_in_pulmonary_interstitium (gram)"
    legend_algebraic[2] = "PPR in component concentration_of_protein_in_pulmonary_interstitium (gram)"
    legend_rates[0] = "d/dt VPF1 in component pulmonary_interstitial_free_fluid_volume (litre)"
    legend_rates[1] = "d/dt PPR1 in component concentration_of_protein_in_pulmonary_interstitium (gram)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 29.9941
    constants[1] = 15.6376
    constants[2] = 2
    constants[3] = 71.9719
    constants[4] = 1.55719
    constants[5] = 1.5683
    constants[6] = 0.0003
    states[0] = 0.0123238
    states[1] = 0.419998
    constants[7] = ((constants[1]-constants[2])*constants[4])/(constants[4]+constants[5])+constants[2]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[0] = custom_piecewise([less(states[0] , 0.00100000), 0.00100000 , True, states[0]])
    algebraic[2] = custom_piecewise([less(states[1] , 0.0250000), 0.0250000 , True, states[1]])
    algebraic[3] = algebraic[2]/algebraic[0]
    algebraic[4] = algebraic[3]*0.400000
    algebraic[1] = 2.00000-0.150000/algebraic[0]
    algebraic[5] = (((constants[7]-algebraic[1])+algebraic[4])-constants[0])*constants[6]
    algebraic[7] = (algebraic[1]+11.0000)*0.000300000
    algebraic[8] = algebraic[5]-algebraic[7]
    algebraic[10] = algebraic[8]
    rates[0] = algebraic[10]
    algebraic[9] = algebraic[7]*algebraic[3]
    algebraic[6] = (constants[3]-algebraic[3])*0.000225000
    algebraic[11] = algebraic[6]-algebraic[9]
    algebraic[12] = algebraic[11]
    rates[1] = algebraic[12]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = custom_piecewise([less(states[0] , 0.00100000), 0.00100000 , True, states[0]])
    algebraic[2] = custom_piecewise([less(states[1] , 0.0250000), 0.0250000 , True, states[1]])
    algebraic[3] = algebraic[2]/algebraic[0]
    algebraic[4] = algebraic[3]*0.400000
    algebraic[1] = 2.00000-0.150000/algebraic[0]
    algebraic[5] = (((constants[7]-algebraic[1])+algebraic[4])-constants[0])*constants[6]
    algebraic[7] = (algebraic[1]+11.0000)*0.000300000
    algebraic[8] = algebraic[5]-algebraic[7]
    algebraic[10] = algebraic[8]
    algebraic[9] = algebraic[7]*algebraic[3]
    algebraic[6] = (constants[3]-algebraic[3])*0.000225000
    algebraic[11] = algebraic[6]-algebraic[9]
    algebraic[12] = algebraic[11]
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