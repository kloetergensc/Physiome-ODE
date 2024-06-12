# Size of variable arrays:
sizeAlgebraic = 2
sizeStates = 5
sizeConstants = 20
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "t in component environment (second)"
    legend_constants[0] = "D_Ca in component parameters (second)"
    legend_constants[1] = "k_1 in component parameters (per_second)"
    legend_constants[2] = "k_2 in component parameters (per_second)"
    legend_constants[3] = "f in component parameters (per_second)"
    legend_constants[4] = "g in component parameters (per_second)"
    legend_constants[5] = "Ca_max in component parameters (dimensionless)"
    legend_constants[6] = "Total_Tn in component parameters (dimensionless)"
    legend_constants[7] = "Total_CB in component parameters (dimensionless)"
    legend_algebraic[0] = "Ca_t in component Ca_t (dimensionless)"
    legend_states[0] = "TnCa in component TnCa (dimensionless)"
    legend_states[1] = "CB_on in component CB_on (dimensionless)"
    legend_states[2] = "CumCB_on in component CumCB (dimensionless)"
    legend_states[3] = "CumCB_off in component CumCB (dimensionless)"
    legend_algebraic[1] = "F in component force_development (force)"
    legend_states[4] = "FTI in component force_development (force_second)"
    legend_constants[15] = "FLA in component force_development (energy)"
    legend_constants[8] = "phi in component force_development (force)"
    legend_constants[9] = "s in component force_development (dimensionless)"
    legend_constants[10] = "L in component force_development (meter)"
    legend_constants[11] = "L_0 in component force_development (meter)"
    legend_constants[12] = "F_max in component force_development (force)"
    legend_constants[17] = "ATP in component ATP (dimensionless)"
    legend_constants[18] = "ATP_energy in component ATP (energy)"
    legend_constants[13] = "epsilon in component ATP (energy)"
    legend_constants[14] = "CumCB_on_end in component ATP (dimensionless)"
    legend_constants[19] = "Efficiency in component equations_main (dimensionless)"
    legend_constants[16] = "Economy in component equations_main (second_per_meter)"
    legend_rates[0] = "d/dt TnCa in component TnCa (dimensionless)"
    legend_rates[1] = "d/dt CB_on in component CB_on (dimensionless)"
    legend_rates[2] = "d/dt CumCB_on in component CumCB (dimensionless)"
    legend_rates[3] = "d/dt CumCB_off in component CumCB (dimensionless)"
    legend_rates[4] = "d/dt FTI in component force_development (force_second)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 0.1
    constants[1] = 40
    constants[2] = 20
    constants[3] = 10
    constants[4] = 10
    constants[5] = 1
    constants[6] = 1
    constants[7] = 1
    states[0] = 0
    states[1] = 0
    states[2] = 0
    states[3] = 0
    states[4] = 0
    constants[8] = 1
    constants[9] = 1
    constants[10] = 1
    constants[11] = 0
    constants[12] = 0.228
    constants[13] = 1
    constants[14] = 1
    constants[15] = constants[12]*constants[9]*(constants[10]-constants[11])
    constants[16] = (constants[8]/constants[13])*(1.00000/constants[4])
    constants[17] = constants[14]
    constants[18] = constants[17]*constants[13]
    constants[19] = constants[15]/constants[18]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[1] = constants[3]*states[0]*(constants[7]-states[1])-constants[4]*states[1]
    rates[2] = constants[3]*states[0]*(constants[7]-states[1])
    rates[3] = constants[4]*states[1]
    algebraic[0] = custom_piecewise([greater_equal(voi , 0.00000) & less(voi , 0.300000*constants[0]), (constants[5]*(1.00000+sin(( pi*(voi/constants[0]-0.150000))/0.300000)))/2.00000 , greater_equal(voi , 0.300000*constants[0]) & less(voi , constants[0]), (constants[5]*(1.00000-sin(( pi*(voi/constants[0]-0.650000))/0.700000)))/2.00000 , True, 0.00000])
    rates[0] = constants[1]*algebraic[0]*(constants[6]-states[0])-constants[2]*states[0]
    algebraic[1] = states[1]*constants[8]
    rates[4] = algebraic[1]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = custom_piecewise([greater_equal(voi , 0.00000) & less(voi , 0.300000*constants[0]), (constants[5]*(1.00000+sin(( pi*(voi/constants[0]-0.150000))/0.300000)))/2.00000 , greater_equal(voi , 0.300000*constants[0]) & less(voi , constants[0]), (constants[5]*(1.00000-sin(( pi*(voi/constants[0]-0.650000))/0.700000)))/2.00000 , True, 0.00000])
    algebraic[1] = states[1]*constants[8]
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