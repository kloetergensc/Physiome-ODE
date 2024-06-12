# Size of variable arrays:
sizeAlgebraic = 6
sizeStates = 4
sizeConstants = 22
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_states[0] = "m in component m (micromolar)"
    legend_constants[0] = "kM_plus in component m (fourth_order_rate_constant)"
    legend_constants[1] = "kM_minus in component m (first_order_rate_constant)"
    legend_constants[2] = "CaMtotal in component m (micromolar)"
    legend_states[1] = "x in component x (micromolar)"
    legend_algebraic[0] = "dmdt in component m (flux)"
    legend_states[2] = "z in component z (micromolar)"
    legend_constants[3] = "kN_plus in component z (second_order_rate_constant)"
    legend_constants[4] = "kN_minus in component z (first_order_rate_constant)"
    legend_constants[5] = "CaNtotal in component z (micromolar)"
    legend_algebraic[1] = "dzdt in component z (flux)"
    legend_states[3] = "h in component h (dimensionless)"
    legend_constants[6] = "d in component h (first_order_rate_constant)"
    legend_constants[7] = "f in component h (first_order_rate_constant)"
    legend_algebraic[3] = "phi in component phi (dimensionless)"
    legend_constants[8] = "lamda in component model_parameters (dimensionless)"
    legend_constants[21] = "L0 in component model_parameters (dimensionless)"
    legend_algebraic[2] = "y in component model_parameters (dimensionless)"
    legend_constants[9] = "N in component model_parameters (dimensionless)"
    legend_algebraic[4] = "psi in component psi (dimensionless)"
    legend_constants[10] = "Vx in component x (flux)"
    legend_constants[11] = "Kx in component x (micromolar)"
    legend_constants[12] = "V1 in component x (flux)"
    legend_constants[13] = "K1 in component x (micromolar)"
    legend_constants[14] = "V2 in component x (flux)"
    legend_constants[15] = "K2 in component x (micromolar)"
    legend_constants[16] = "V3 in component x (flux)"
    legend_constants[17] = "K3 in component x (micromolar)"
    legend_constants[18] = "kc in component x (dimensionless)"
    legend_constants[19] = "alpha in component x (first_order_rate_constant)"
    legend_constants[20] = "Caex in component x (micromolar)"
    legend_algebraic[5] = "dxdt in component x (flux)"
    legend_rates[0] = "d/dt m in component m (micromolar)"
    legend_rates[2] = "d/dt z in component z (micromolar)"
    legend_rates[3] = "d/dt h in component h (dimensionless)"
    legend_rates[1] = "d/dt x in component x (micromolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.0
    constants[0] = 500.0
    constants[1] = 100.0
    constants[2] = 25.0
    states[1] = 0.0
    states[2] = 1.0E-8
    constants[3] = 5.0
    constants[4] = 5.0
    constants[5] = 25.0
    states[3] = 0.0
    constants[6] = 0.4
    constants[7] = 0.1
    constants[8] = 5.0
    constants[9] = 13.0
    constants[10] = 1000.0
    constants[11] = 500.0
    constants[12] = 30000.0
    constants[13] = 4.3
    constants[14] = 100.0
    constants[15] = 0.1
    constants[16] = 10000.0
    constants[17] = 100.0
    constants[18] = 10.0
    constants[19] = 0.006
    constants[20] = 1.0
    constants[21] = power(10.0000, -(constants[9]/2.00000))
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = constants[0]*(constants[2]-states[0])*(power(states[1], 3.00000))-constants[1]*states[0]
    rates[2] = constants[3]*(constants[5]-states[2])*states[0]-constants[4]*states[2]
    algebraic[2] = 1.00000/states[2]
    algebraic[3] = 1.00000/(1.00000+(constants[21]*(power(constants[8]*algebraic[2], constants[9]+1.00000)-1.00000))/((constants[8]*algebraic[2]-1.00000)*((algebraic[2]-1.00000)/(power(algebraic[2], constants[9]+1.00000)-1.00000))))
    rates[3] = constants[6]*algebraic[3]*(1.00000/states[2])*(1.00000-states[3])-constants[7]*(1.00000-algebraic[3]*(1.00000/states[2]))*states[3]
    algebraic[4] = (1.00000+constants[21])/((power(algebraic[2], constants[9]+1.00000)-1.00000)/(algebraic[2]-1.00000)+constants[21]*((power(constants[8]*algebraic[2], constants[9]+1.00000)-1.00000)/(constants[8]*algebraic[2]-1.00000)))
    rates[1] = (constants[10]*constants[20])/(constants[11]+constants[20])-(states[3]*algebraic[4]*(1.00000/states[2])*((constants[12]*states[1])/(constants[13]+states[1]))+states[3]*algebraic[4]*(1.00000/states[2])*((constants[14]*states[1])/(constants[15]+states[1]))+(1.00000/(1.00000+constants[18]*states[2]))*((constants[16]*states[1])/(constants[17]+states[1]))+constants[19]*states[1])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[2] = 1.00000/states[2]
    algebraic[3] = 1.00000/(1.00000+(constants[21]*(power(constants[8]*algebraic[2], constants[9]+1.00000)-1.00000))/((constants[8]*algebraic[2]-1.00000)*((algebraic[2]-1.00000)/(power(algebraic[2], constants[9]+1.00000)-1.00000))))
    algebraic[4] = (1.00000+constants[21])/((power(algebraic[2], constants[9]+1.00000)-1.00000)/(algebraic[2]-1.00000)+constants[21]*((power(constants[8]*algebraic[2], constants[9]+1.00000)-1.00000)/(constants[8]*algebraic[2]-1.00000)))
    algebraic[0] = constants[0]*(constants[2]-states[0])*(power(states[1], 3.00000))-constants[1]*states[0]
    algebraic[1] = constants[3]*(constants[5]-states[2])*states[0]-constants[4]*states[2]
    algebraic[5] = (constants[10]*constants[20])/(constants[11]+constants[20])-(states[3]*algebraic[4]*(1.00000/states[2])*((constants[12]*states[1])/(constants[13]+states[1]))+states[3]*algebraic[4]*(1.00000/states[2])*((constants[14]*states[1])/(constants[15]+states[1]))+(1.00000/(1.00000+constants[18]*states[2]))*((constants[16]*states[1])/(constants[17]+states[1]))+constants[19]*states[1])
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