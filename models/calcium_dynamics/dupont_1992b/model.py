# Size of variable arrays:
sizeAlgebraic = 3
sizeStates = 3
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
    legend_constants[0] = "VM2 in component parameters (micromolar_min)"
    legend_constants[1] = "VM3 in component parameters (micromolar_min)"
    legend_constants[2] = "KR in component parameters (micromolar)"
    legend_constants[3] = "KA in component parameters (micromolar)"
    legend_constants[4] = "KP in component parameters (micromolar)"
    legend_constants[5] = "n in component parameters (dimensionless)"
    legend_constants[6] = "m in component parameters (dimensionless)"
    legend_constants[7] = "p in component parameters (dimensionless)"
    legend_constants[8] = "kf in component parameters (per_minute)"
    legend_constants[9] = "k in component parameters (per_minute)"
    legend_states[0] = "Y in component insensitive_pool (micromolar)"
    legend_states[1] = "Z in component cytosol (micromolar)"
    legend_algebraic[0] = "v2 in component parameters (micromolar_min)"
    legend_algebraic[2] = "v3 in component parameters (micromolar_min)"
    legend_constants[10] = "v0 in component cytosol (micromolar_min)"
    legend_constants[11] = "v1beta in component cytosol (micromolar_min)"
    legend_constants[12] = "K1 in component phosphorylation (dimensionless)"
    legend_constants[13] = "K2 in component phosphorylation (dimensionless)"
    legend_constants[14] = "WT in component phosphorylation (micromolar)"
    legend_constants[18] = "vP in component phosphorylation (micromolar_min)"
    legend_algebraic[1] = "vK in component kinase_reaction (micromolar_min)"
    legend_constants[15] = "vMK in component kinase_reaction (micromolar_min)"
    legend_states[2] = "Wstar in component phosphorylation (dimensionless)"
    legend_constants[16] = "Ka in component kinase_reaction (micromolar)"
    legend_constants[17] = "q in component kinase_reaction (dimensionless)"
    legend_rates[1] = "d/dt Z in component cytosol (micromolar)"
    legend_rates[0] = "d/dt Y in component insensitive_pool (micromolar)"
    legend_rates[2] = "d/dt Wstar in component phosphorylation (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 65
    constants[1] = 500
    constants[2] = 2
    constants[3] = 0.9
    constants[4] = 1
    constants[5] = 2
    constants[6] = 2
    constants[7] = 4
    constants[8] = 1
    constants[9] = 10
    states[0] = 1.454
    states[1] = 0.2281
    constants[10] = 1
    constants[11] = 2.4
    constants[12] = 0.01
    constants[13] = 0.01
    constants[14] = 10
    constants[15] = 100
    states[2] = 0.8916
    constants[16] = 1
    constants[17] = 4
    constants[18] = constants[15]*(0.890000/20.0000)
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[1] = constants[15]*((power(states[1], constants[17]))/(power(constants[16], constants[17])+power(states[1], constants[17])))
    rates[2] = (constants[18]/constants[14])*(((algebraic[1]/constants[18])*(1.00000-states[2]))/((constants[12]+1.00000)-states[2])-states[2]/(constants[13]+states[2]))
    algebraic[0] = (constants[0]*(power(states[1], constants[5])))/(power(constants[4], constants[5])+power(states[1], constants[5]))
    algebraic[2] = constants[1]*((power(states[0], constants[6]))/(power(constants[2], constants[6])+power(states[0], constants[6])))*((power(states[1], constants[7]))/(power(constants[3], constants[7])+power(states[1], constants[7])))
    rates[1] = (((constants[10]+constants[11])-algebraic[0])+algebraic[2]+constants[8]*states[0])-constants[9]*states[1]
    rates[0] = (algebraic[0]-algebraic[2])-constants[8]*states[0]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[1] = constants[15]*((power(states[1], constants[17]))/(power(constants[16], constants[17])+power(states[1], constants[17])))
    algebraic[0] = (constants[0]*(power(states[1], constants[5])))/(power(constants[4], constants[5])+power(states[1], constants[5]))
    algebraic[2] = constants[1]*((power(states[0], constants[6]))/(power(constants[2], constants[6])+power(states[0], constants[6])))*((power(states[1], constants[7]))/(power(constants[3], constants[7])+power(states[1], constants[7])))
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