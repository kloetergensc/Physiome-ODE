# Size of variable arrays:
sizeAlgebraic = 0
sizeStates = 0
sizeConstants = 30
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_constants[28] = "SHbO2 in component SHbO2 (dimensionless)"
    legend_constants[26] = "KHbO2 in component KHbO2 (per_molar)"
    legend_constants[20] = "O2 in component O2 (molar)"
    legend_constants[29] = "SHbCO2 in component SHbCO2 (dimensionless)"
    legend_constants[27] = "KHbCO2 in component KHbCO2 (per_molar)"
    legend_constants[22] = "CO2 in component CO2 (molar)"
    legend_constants[24] = "Hrbc in component model_parameters (molar)"
    legend_constants[0] = "K2 in component model_parameters (per_molar)"
    legend_constants[1] = "K2_ in component model_parameters (molar)"
    legend_constants[2] = "K3 in component model_parameters (per_molar)"
    legend_constants[3] = "K3_ in component model_parameters (molar)"
    legend_constants[25] = "K4 in component K4 (per_molar)"
    legend_constants[4] = "K5_ in component model_parameters (molar)"
    legend_constants[5] = "K6_ in component model_parameters (molar)"
    legend_constants[6] = "O2_S in component model_parameters (micromolar)"
    legend_constants[7] = "H_S in component model_parameters (nanomolar)"
    legend_constants[8] = "n1 in component model_parameters (dimensionless)"
    legend_constants[9] = "n2 in component model_parameters (dimensionless)"
    legend_constants[10] = "CO2_S in component model_parameters (millimolar)"
    legend_constants[11] = "K4_ in component model_parameters (per_molar)"
    legend_constants[12] = "n0 in component model_parameters (dimensionless)"
    legend_constants[19] = "alpha_O2 in component alpha_O2 (M_mmHg)"
    legend_constants[13] = "PO2 in component model_parameters (mmHg)"
    legend_constants[21] = "alpha_CO2 in component alpha_CO2 (M_mmHg)"
    legend_constants[14] = "PCO2 in component model_parameters (mmHg)"
    legend_constants[15] = "Wpl in component model_parameters (ml_ml)"
    legend_constants[16] = "T in component model_parameters (celsius)"
    legend_constants[17] = "Rrbc in component model_parameters (dimensionless)"
    legend_constants[23] = "Hpl in component model_parameters (molar)"
    legend_constants[18] = "pHpl in component model_parameters (pH)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 29.5
    constants[1] = 1E-6
    constants[2] = 25.1
    constants[3] = 1E-6
    constants[4] = 2.63E-8
    constants[5] = 1.91E-8
    constants[6] = 146.0
    constants[7] = 57.5
    constants[8] = 1.06
    constants[9] = 0.12
    constants[10] = 1.31
    constants[11] = 202123.0
    constants[12] = 1.7
    constants[13] = 100.0
    constants[14] = 40.0
    constants[15] = 0.94
    constants[16] = 37.0
    constants[17] = 0.69
    constants[18] = 7.24
    constants[19] = ((1.37000-0.0137000*(constants[16]-37.0000))+0.000580000*(power(constants[16]-37.0000, 2.00000)))*(1.00000e-06/constants[15])
    constants[20] = constants[19]*constants[13]
    constants[21] = ((3.07000-0.0570000*(constants[16]-37.0000))+0.00200000*(power(constants[16]-37.0000, 2.00000)))*(1.00000e-05/constants[15])
    constants[22] = constants[21]*constants[14]
    constants[23] = power(10.0000, -constants[18])
    constants[24] = constants[23]/constants[17]
    constants[25] = constants[11]*(power(constants[20]/constants[6], constants[12]))*(power(constants[24]/constants[7], -constants[8]))*(power(constants[22]/constants[10], -constants[9]))
    constants[26] = (constants[25]*(constants[2]*constants[22]*(1.00000+constants[3]/constants[24])+(1.00000+constants[24]/constants[5])))/(constants[0]*constants[22]*(1.00000+constants[1]/constants[24])+(1.00000+constants[24]/constants[4]))
    constants[27] = (constants[0]*(1.00000+constants[1]/constants[24])+constants[2]*constants[25]*(1.00000+constants[3]/constants[24])*constants[20])/(1.00000+constants[24]/constants[4]+constants[25]*(1.00000+constants[24]/constants[5])*constants[20])
    constants[28] = (constants[26]*constants[20])/(1.00000+constants[26]*constants[20])
    constants[29] = (constants[27]*constants[22])/(1.00000+constants[27]*constants[22])
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