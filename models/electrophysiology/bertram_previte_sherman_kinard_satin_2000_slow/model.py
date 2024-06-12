# Size of variable arrays:
sizeAlgebraic = 10
sizeStates = 4
sizeConstants = 21
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (millisecond)"
    legend_constants[0] = "Cm in component membrane (femtoF)"
    legend_states[0] = "V in component membrane (millivolt)"
    legend_algebraic[4] = "ICa in component Ca_current (femtoA)"
    legend_algebraic[6] = "IK in component rapid_K_current (femtoA)"
    legend_algebraic[9] = "Il in component leak_current (femtoA)"
    legend_algebraic[7] = "Is1 in component slow_K_current (femtoA)"
    legend_algebraic[8] = "Is2 in component very_slow_K_current (femtoA)"
    legend_constants[1] = "Vm in component Ca_current (millivolt)"
    legend_constants[2] = "VCa in component Ca_current (millivolt)"
    legend_constants[3] = "gCa in component Ca_current (picoS)"
    legend_algebraic[0] = "minf in component Ca_current (dimensionless)"
    legend_constants[4] = "sm in component Ca_current (millivolt)"
    legend_constants[5] = "VK in component rapid_K_current (millivolt)"
    legend_constants[6] = "gK in component rapid_K_current (picoS)"
    legend_states[1] = "n in component rapid_K_current (dimensionless)"
    legend_constants[7] = "lambda in component rapid_K_current (dimensionless)"
    legend_constants[8] = "tnbar in component rapid_K_current (dimensionless)"
    legend_constants[9] = "Vn in component rapid_K_current (millivolt)"
    legend_constants[10] = "sn in component rapid_K_current (millivolt)"
    legend_algebraic[5] = "taun in component rapid_K_current (dimensionless)"
    legend_algebraic[1] = "ninf in component rapid_K_current (dimensionless)"
    legend_constants[11] = "gs1 in component slow_K_current (picoS)"
    legend_states[2] = "s1 in component slow_K_current (dimensionless)"
    legend_algebraic[2] = "s1inf in component slow_K_current (dimensionless)"
    legend_constants[12] = "Vs1 in component slow_K_current (millivolt)"
    legend_constants[13] = "ss1 in component slow_K_current (millivolt)"
    legend_constants[14] = "taus1 in component slow_K_current (dimensionless)"
    legend_constants[15] = "Vs2 in component very_slow_K_current (millivolt)"
    legend_states[3] = "s2 in component very_slow_K_current (dimensionless)"
    legend_algebraic[3] = "s2inf in component very_slow_K_current (dimensionless)"
    legend_constants[16] = "ss2 in component very_slow_K_current (millivolt)"
    legend_constants[17] = "gs2 in component very_slow_K_current (picoS)"
    legend_constants[18] = "taus2 in component very_slow_K_current (dimensionless)"
    legend_constants[19] = "gl in component leak_current (picoS)"
    legend_constants[20] = "Vl in component leak_current (millivolt)"
    legend_rates[0] = "d/dt V in component membrane (millivolt)"
    legend_rates[1] = "d/dt n in component rapid_K_current (dimensionless)"
    legend_rates[2] = "d/dt s1 in component slow_K_current (dimensionless)"
    legend_rates[3] = "d/dt s2 in component very_slow_K_current (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 4524
    states[0] = -43
    constants[1] = -22
    constants[2] = 100
    constants[3] = 280
    constants[4] = 7.5
    constants[5] = -80
    constants[6] = 1300
    states[1] = 0.03
    constants[7] = 1.1
    constants[8] = 9.09
    constants[9] = -9
    constants[10] = 10
    constants[11] = 20
    states[2] = 0.1
    constants[12] = -40
    constants[13] = 0.5
    constants[14] = 1000
    constants[15] = -42
    states[3] = 0.434
    constants[16] = 0.4
    constants[17] = 32
    constants[18] = 120000
    constants[19] = 25
    constants[20] = -40
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[2] = 1.00000/(1.00000+exp((constants[12]-states[0])/constants[13]))
    rates[2] = (algebraic[2]-states[2])/(constants[14]*1.00000)
    algebraic[3] = 1.00000/(1.00000+exp((constants[15]-states[0])/constants[16]))
    rates[3] = (algebraic[3]-states[3])/(constants[18]*1.00000)
    algebraic[5] = constants[8]/(1.00000+exp((states[0]-constants[9])/constants[10]))
    algebraic[1] = 1.00000/(1.00000+exp((constants[9]-states[0])/constants[10]))
    rates[1] = (constants[7]*(algebraic[1]-states[1]))/(algebraic[5]*1.00000)
    algebraic[0] = 1.00000/(1.00000+exp((constants[1]-states[0])/constants[4]))
    algebraic[4] = constants[3]*algebraic[0]*(states[0]-constants[2])
    algebraic[6] = constants[6]*states[1]*(states[0]-constants[5])
    algebraic[9] = constants[19]*(states[0]-constants[20])
    algebraic[7] = constants[11]*states[2]*(states[0]-constants[5])
    algebraic[8] = constants[17]*states[3]*(states[0]-constants[5])
    rates[0] = -(algebraic[4]+algebraic[6]+algebraic[9]+algebraic[7]+algebraic[8])/constants[0]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[2] = 1.00000/(1.00000+exp((constants[12]-states[0])/constants[13]))
    algebraic[3] = 1.00000/(1.00000+exp((constants[15]-states[0])/constants[16]))
    algebraic[5] = constants[8]/(1.00000+exp((states[0]-constants[9])/constants[10]))
    algebraic[1] = 1.00000/(1.00000+exp((constants[9]-states[0])/constants[10]))
    algebraic[0] = 1.00000/(1.00000+exp((constants[1]-states[0])/constants[4]))
    algebraic[4] = constants[3]*algebraic[0]*(states[0]-constants[2])
    algebraic[6] = constants[6]*states[1]*(states[0]-constants[5])
    algebraic[9] = constants[19]*(states[0]-constants[20])
    algebraic[7] = constants[11]*states[2]*(states[0]-constants[5])
    algebraic[8] = constants[17]*states[3]*(states[0]-constants[5])
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