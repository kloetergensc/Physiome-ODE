# Size of variable arrays:
sizeAlgebraic = 9
sizeStates = 1
sizeConstants = 29
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_algebraic[8] = "F_CE in component F_CE (newton)"
    legend_algebraic[0] = "f_L_CE in component f_L_CE (newton)"
    legend_algebraic[5] = "g_V_CE in component g_V_CE (dimensionless)"
    legend_constants[0] = "a in component user_defined_constants (dimensionless)"
    legend_constants[1] = "F_min in component f_L_CE (newton)"
    legend_constants[2] = "F_max in component user_defined_constants (newton)"
    legend_states[0] = "L_CE in component L_CE (metre)"
    legend_constants[3] = "L_CE_opt in component user_defined_constants (metre)"
    legend_constants[4] = "W in component f_L_CE (dimensionless)"
    legend_constants[28] = "lambda_a in component lambda_a (second_per_metre)"
    legend_constants[5] = "V_max in component g_V_CE (metre_per_second)"
    legend_algebraic[6] = "V_CE in component V_CE (dimensionless)"
    legend_constants[6] = "A in component g_V_CE (dimensionless)"
    legend_constants[7] = "g_max in component g_V_CE (dimensionless)"
    legend_constants[24] = "d1 in component d1 (dimensionless)"
    legend_constants[26] = "d2 in component d2 (dimensionless)"
    legend_constants[27] = "d3 in component d3 (dimensionless)"
    legend_constants[8] = "gamma in component g_V_CE (dimensionless)"
    legend_constants[9] = "V_max in component d1 (metre_per_second)"
    legend_constants[10] = "A in component d1 (dimensionless)"
    legend_constants[11] = "g_max in component d1 (dimensionless)"
    legend_constants[12] = "S in component d1 (metre_per_second)"
    legend_constants[13] = "S in component d2 (metre_per_second)"
    legend_constants[14] = "A in component d2 (dimensionless)"
    legend_constants[15] = "V_max in component d2 (metre_per_second)"
    legend_constants[16] = "gamma in component d2 (dimensionless)"
    legend_constants[17] = "g_max in component d3 (dimensionless)"
    legend_constants[18] = "gamma in component d3 (dimensionless)"
    legend_algebraic[4] = "F_SEE in component F_SEE (newton)"
    legend_constants[19] = "k_SEE in component F_SEE (newton_per_metre2)"
    legend_algebraic[3] = "L_SEE in component L_SEE (metre)"
    legend_constants[20] = "L_slack in component F_SEE (metre)"
    legend_algebraic[1] = "F_PEE in component F_PEE (newton)"
    legend_constants[25] = "k_PEE in component k_PEE (newton_per_metre2)"
    legend_constants[21] = "L_slack in component F_PEE (metre)"
    legend_constants[22] = "W in component k_PEE (dimensionless)"
    legend_constants[23] = "L_CE_opt in component k_PEE (metre)"
    legend_algebraic[2] = "L_m in component L_m (metre)"
    legend_algebraic[7] = "F_m in component F_m (newton)"
    legend_rates[0] = "d/dt L_CE in component L_CE (metre)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 0.8
    constants[1] = 10
    constants[2] = 7000
    states[0] = 0.038
    constants[3] = 0.093
    constants[4] = 0.63
    constants[5] = 0.93
    constants[6] = 0.25
    constants[7] = 1.5
    constants[8] = 5.67
    constants[9] = 0.93
    constants[10] = 0.25
    constants[11] = 1.5
    constants[12] = 2
    constants[13] = 2
    constants[14] = 0.25
    constants[15] = 0.93
    constants[16] = 5.67
    constants[17] = 1.5
    constants[18] = 5.67
    constants[19] = 1000000
    constants[20] = 0.0025
    constants[21] = 0.0025
    constants[22] = 0.63
    constants[23] = 0.01
    constants[24] = (constants[9]*constants[10]*(constants[11]-1.00000))/(constants[12]*(constants[10]+1.00000))
    constants[25] = constants[2]/(power(constants[22]*constants[23], 2.00000))
    constants[26] = (constants[13]*(constants[14]+1.00000))/(constants[15]*constants[14]*(power(constants[16]+1.00000, 2.00000)))
    constants[27] = ((constants[17]-1.00000)*(power(constants[18], 2.00000)))/(power(constants[18]+1.00000, 2.00000))+1.00000
    constants[28] = 1.00000*((1.00000-exp(-3.82000*constants[0]))+constants[0]*exp(-3.82000))
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[0] = (constants[2]*(1.00000*(1.00000-states[0])-power(constants[3], 2.00000)))/((power(constants[4], 2.00000))*(power(constants[3], 2.00000)))
    algebraic[2] = custom_piecewise([less_equal(voi , 1.00000), 0.0380000 , greater(voi , 1.00000) & less(voi , 2.00000), 0.0380000+0.00200000*(voi-1.00000) , True, 0.0400000])
    algebraic[3] = algebraic[2]-states[0]
    algebraic[4] = custom_piecewise([less_equal(algebraic[3] , constants[20]), 0.00000 , True, constants[19]*(power(algebraic[3]-constants[20], 2.00000))])
    algebraic[1] = custom_piecewise([less_equal(states[0] , constants[21]), 0.00000 , True, constants[25]*(power(states[0]-constants[21], 2.00000))])
    rootfind_0(voi, constants, rates, states, algebraic)
    rates[0] = 1.00000*algebraic[6]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = (constants[2]*(1.00000*(1.00000-states[0])-power(constants[3], 2.00000)))/((power(constants[4], 2.00000))*(power(constants[3], 2.00000)))
    algebraic[2] = custom_piecewise([less_equal(voi , 1.00000), 0.0380000 , greater(voi , 1.00000) & less(voi , 2.00000), 0.0380000+0.00200000*(voi-1.00000) , True, 0.0400000])
    algebraic[3] = algebraic[2]-states[0]
    algebraic[4] = custom_piecewise([less_equal(algebraic[3] , constants[20]), 0.00000 , True, constants[19]*(power(algebraic[3]-constants[20], 2.00000))])
    algebraic[1] = custom_piecewise([less_equal(states[0] , constants[21]), 0.00000 , True, constants[25]*(power(states[0]-constants[21], 2.00000))])
    algebraic[7] = algebraic[4]
    algebraic[8] = algebraic[0]*algebraic[5]*constants[0]
    return algebraic

initialGuess0 = None
def rootfind_0(voi, constants, rates, states, algebraic):
    """Calculate values of algebraic variables for DAE"""
    from scipy.optimize import fsolve
    global initialGuess0
    if initialGuess0 is None: initialGuess0 = ones(2)*0.1
    if not iterable(voi):
        soln = fsolve(residualSN_0, initialGuess0, args=(algebraic, voi, constants, rates, states), xtol=1E-6)
        initialGuess0 = soln
        algebraic[5] = soln[0]
        algebraic[6] = soln[1]
    else:
        for (i,t) in enumerate(voi):
            soln = fsolve(residualSN_0, initialGuess0, args=(algebraic[:,i], voi[i], constants, rates[:i], states[:,i]), xtol=1E-6)
            initialGuess0 = soln
            algebraic[5][i] = soln[0]
            algebraic[6][i] = soln[1]

def residualSN_0(algebraicCandidate, algebraic, voi, constants, rates, states):
    resid = array([0.0] * 2)
    algebraic[5] = algebraicCandidate[0]
    algebraic[6] = algebraicCandidate[1]
    resid[0] = (algebraic[5]-(custom_piecewise([less_equal(algebraic[6] , 0.00000), (constants[28]*constants[5]+algebraic[6])/(constants[28]*constants[5]-algebraic[6]/constants[6]) , less(0.00000 , algebraic[6]) & less_equal(algebraic[6] , constants[8]*constants[24]), (constants[7]*algebraic[6]+constants[24])/(algebraic[6]+constants[24]) , greater(algebraic[6] , constants[8]*constants[24]), constants[27]+constants[26]*algebraic[6] , True, float('nan')])))
    resid[1] = (algebraic[6]-1.00000*(((1.00000/algebraic[5])*(algebraic[4]*(algebraic[2]-states[0])-algebraic[1]*states[0]))/(constants[0]*algebraic[0])))
    return resid

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