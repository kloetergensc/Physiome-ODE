# Size of variable arrays:
sizeAlgebraic = 8
sizeStates = 4
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
    legend_constants[0] = "AMK in component electrolytes (dimensionless)"
    legend_constants[1] = "TVD in component electrolytes (L_per_minute)"
    legend_constants[2] = "NOD in component electrolytes (monovalent_mEq_per_minute)"
    legend_constants[3] = "STH in component electrolytes (dimensionless)"
    legend_constants[4] = "KOD in component electrolytes (monovalent_mEq_per_minute)"
    legend_constants[5] = "VUD in component electrolytes (L_per_minute)"
    legend_algebraic[3] = "VEC in component extracellular_fluid_volume (litre)"
    legend_algebraic[4] = "CNA in component extracellular_Na_concentration (monovalent_mEq_per_litre)"
    legend_constants[6] = "NID in component parameter_values (monovalent_mEq_per_minute)"
    legend_constants[7] = "TRPL in component parameter_values (L_per_minute)"
    legend_constants[11] = "NED in component extracellular_Na_concentration (monovalent_mEq_per_minute)"
    legend_states[0] = "NAE in component extracellular_Na_concentration (monovalent_mEq)"
    legend_constants[12] = "AMK1 in component aldosterone_effect_on_cellular_K_distribution (dimensionless)"
    legend_constants[8] = "ALCLK in component parameter_values (dimensionless)"
    legend_algebraic[5] = "CKE in component extracellular_K_concentration (monovalent_mEq_per_litre)"
    legend_algebraic[0] = "KE in component extracellular_K_concentration (monovalent_mEq)"
    legend_states[1] = "KTOT in component extracellular_K_concentration (monovalent_mEq)"
    legend_constants[9] = "KID in component parameter_values (monovalent_mEq_per_minute)"
    legend_constants[13] = "KTOTD in component extracellular_K_concentration (monovalent_mEq_per_minute)"
    legend_states[2] = "VIC in component intracellular_fluid_volume (litre)"
    legend_algebraic[2] = "CKI in component intracellular_K_concentration (monovalent_mEq_per_litre)"
    legend_algebraic[1] = "KI in component intracellular_K_concentration (monovalent_mEq)"
    legend_algebraic[7] = "VID in component intracellular_fluid_volume (L_per_minute)"
    legend_constants[10] = "VIDML in component parameter_values (litre2_per_monovalent_mEq_per_minute)"
    legend_algebraic[6] = "CCD in component intracellular_fluid_volume (monovalent_mEq_per_litre)"
    legend_states[3] = "VTW in component total_body_water (litre)"
    legend_rates[0] = "d/dt NAE in component extracellular_Na_concentration (monovalent_mEq)"
    legend_rates[1] = "d/dt KTOT in component extracellular_K_concentration (monovalent_mEq)"
    legend_rates[2] = "d/dt VIC in component intracellular_fluid_volume (litre)"
    legend_rates[3] = "d/dt VTW in component total_body_water (litre)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 1.037
    constants[1] = 0.000980838
    constants[2] = 0.0959449
    constants[3] = 0.977181
    constants[4] = 0.0804374
    constants[5] = 0.000989
    constants[6] = 0.1
    constants[7] = 0
    states[0] = 2109.91
    constants[8] = 0.3
    states[1] = 3622.54
    constants[9] = 0.08
    states[2] = 25.0404
    constants[10] = 0.01
    states[3] = 39.8952
    constants[11] = (constants[6]*constants[3]-constants[2])+constants[7]*142.000
    constants[12] = (constants[0]-1.00000)*constants[8]+1.00000
    constants[13] = constants[9]-constants[4]
    constants[14] = constants[1]-constants[5]
    constants[15] = constants[11]
    constants[16] = constants[13]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[3] = constants[14]
    rates[0] = constants[15]
    rates[1] = constants[16]
    algebraic[3] = states[3]-states[2]
    algebraic[4] = states[0]/algebraic[3]
    algebraic[0] = (states[1]-3000.00)/(constants[12]*9.33330)
    algebraic[1] = states[1]-algebraic[0]
    algebraic[2] = algebraic[1]/states[2]
    algebraic[6] = algebraic[2]-algebraic[4]
    algebraic[7] = algebraic[6]*constants[10]
    rates[2] = algebraic[7]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[3] = states[3]-states[2]
    algebraic[4] = states[0]/algebraic[3]
    algebraic[0] = (states[1]-3000.00)/(constants[12]*9.33330)
    algebraic[1] = states[1]-algebraic[0]
    algebraic[2] = algebraic[1]/states[2]
    algebraic[6] = algebraic[2]-algebraic[4]
    algebraic[7] = algebraic[6]*constants[10]
    algebraic[5] = algebraic[0]/algebraic[3]
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