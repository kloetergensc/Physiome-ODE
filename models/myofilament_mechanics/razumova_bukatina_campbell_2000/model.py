# Size of variable arrays:
sizeAlgebraic = 13
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
    legend_voi = "time in component environment (second)"
    legend_constants[16] = "F in component main (dimensionless)"
    legend_constants[0] = "R_T in component main (dimensionless)"
    legend_algebraic[0] = "R_off in component main (dimensionless)"
    legend_states[0] = "D in component main (dimensionless)"
    legend_states[1] = "A_1 in component main (dimensionless)"
    legend_states[2] = "A_2 in component main (dimensionless)"
    legend_algebraic[4] = "lambda_off in component main (dimensionless)"
    legend_algebraic[5] = "lambda_on in component main (dimensionless)"
    legend_algebraic[1] = "lambda_D in component main (dimensionless)"
    legend_algebraic[2] = "lambda_A_1 in component main (dimensionless)"
    legend_algebraic[6] = "lambda_A_2 in component main (dimensionless)"
    legend_algebraic[3] = "lambda_A2_cyc in component main (dimensionless)"
    legend_constants[15] = "Ca in component main (dimensionless)"
    legend_constants[14] = "Ca_50 in component main (dimensionless)"
    legend_algebraic[11] = "k_on in component XB_RU_interaction (per_second)"
    legend_algebraic[12] = "k_off in component XB_RU_interaction (per_second)"
    legend_constants[1] = "k_0_on in component main (per_second)"
    legend_constants[2] = "k_0_off in component main (per_second)"
    legend_constants[3] = "k_Ca_on in component main (per_second)"
    legend_constants[4] = "k_Ca_off in component main (per_second)"
    legend_algebraic[9] = "f in component XB_XB_interaction (per_second)"
    legend_algebraic[10] = "f_prime in component XB_XB_interaction (per_second)"
    legend_constants[5] = "f_0 in component main (per_second)"
    legend_constants[6] = "f_prime_0 in component main (per_second)"
    legend_constants[7] = "h in component main (per_second)"
    legend_constants[8] = "h_prime in component main (per_second)"
    legend_constants[9] = "g in component main (per_second)"
    legend_constants[10] = "n_H in component main (dimensionless)"
    legend_constants[11] = "u in component main (dimensionless)"
    legend_constants[12] = "w in component main (dimensionless)"
    legend_constants[13] = "v in component main (dimensionless)"
    legend_constants[17] = "k_u_on in component RU_rate_constant (per_second)"
    legend_constants[18] = "k_u_off in component RU_rate_constant (per_second)"
    legend_algebraic[7] = "k_w_on in component RU_RU_interaction (per_second)"
    legend_algebraic[8] = "k_w_off in component RU_RU_interaction (per_second)"
    legend_rates[0] = "d/dt D in component main (dimensionless)"
    legend_rates[1] = "d/dt A_1 in component main (dimensionless)"
    legend_rates[2] = "d/dt A_2 in component main (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 1
    states[0] = 0
    states[1] = 0
    states[2] = 0
    constants[1] = 0
    constants[2] = 100
    constants[3] = 120
    constants[4] = 50
    constants[5] = 50
    constants[6] = 400
    constants[7] = 8
    constants[8] = 6
    constants[9] = 4
    constants[10] = 1
    constants[11] = 1
    constants[12] = 1
    constants[13] = 1
    constants[14] = constants[4]/constants[3]
    constants[15] = constants[14]*100.000
    constants[16] = 1.00000/(1.00000+power(constants[15]/constants[14], -constants[10]))
    constants[17] = constants[1]+((constants[3]-constants[1])*constants[15])/(constants[14]+constants[15])
    constants[18] = constants[2]+((constants[4]-constants[2])*constants[15])/(constants[14]+constants[15])
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[2] = constants[7]*states[1]-(constants[8]+constants[9])*states[2]
    algebraic[6] = states[2]/constants[0]
    algebraic[9] = constants[5]*(power(1.00000+algebraic[6]*(exp(constants[13]-1.00000)-1.00000), 2.00000))
    algebraic[10] = constants[6]*(power(1.00000+algebraic[6]*(exp(-(constants[13]-1.00000))-1.00000), 2.00000))
    rates[1] = (algebraic[9]*states[0]+constants[8]*states[2])-(algebraic[10]+constants[7])*states[1]
    algebraic[0] = constants[0]-(states[0]+states[1]+states[2])
    algebraic[5] = (states[0]+states[1]+states[2])/constants[0]
    algebraic[7] = constants[17]*(power(1.00000+algebraic[5]*(constants[11]-1.00000), 2.00000))
    algebraic[11] = algebraic[7]*(power(1.00000+algebraic[6]*(exp(constants[12]-1.00000)-1.00000), 2.00000))
    algebraic[8] = constants[18]*(power(constants[11]-algebraic[5]*(constants[11]-1.00000), 2.00000))
    algebraic[12] = algebraic[8]*(power(1.00000+algebraic[6]*(exp(-(constants[12]-1.00000))-1.00000), 2.00000))
    rates[0] = (algebraic[11]*algebraic[0]+algebraic[10]*states[1]+constants[9]*states[2])-(algebraic[12]+algebraic[9])*states[0]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[6] = states[2]/constants[0]
    algebraic[9] = constants[5]*(power(1.00000+algebraic[6]*(exp(constants[13]-1.00000)-1.00000), 2.00000))
    algebraic[10] = constants[6]*(power(1.00000+algebraic[6]*(exp(-(constants[13]-1.00000))-1.00000), 2.00000))
    algebraic[0] = constants[0]-(states[0]+states[1]+states[2])
    algebraic[5] = (states[0]+states[1]+states[2])/constants[0]
    algebraic[7] = constants[17]*(power(1.00000+algebraic[5]*(constants[11]-1.00000), 2.00000))
    algebraic[11] = algebraic[7]*(power(1.00000+algebraic[6]*(exp(constants[12]-1.00000)-1.00000), 2.00000))
    algebraic[8] = constants[18]*(power(constants[11]-algebraic[5]*(constants[11]-1.00000), 2.00000))
    algebraic[12] = algebraic[8]*(power(1.00000+algebraic[6]*(exp(-(constants[12]-1.00000))-1.00000), 2.00000))
    algebraic[1] = states[0]/constants[0]
    algebraic[2] = states[1]/constants[0]
    algebraic[3] = states[2]/(states[0]+states[1]+states[2])
    algebraic[4] = algebraic[0]/constants[0]
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