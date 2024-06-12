# Size of variable arrays:
sizeAlgebraic = 2
sizeStates = 5
sizeConstants = 15
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (day)"
    legend_states[0] = "X in component X (dimensionless)"
    legend_constants[0] = "mu0 in component X (first_order_rate_constant)"
    legend_constants[13] = "s in component X (first_order_rate_constant)"
    legend_states[1] = "K in component K (first_order_rate_constant)"
    legend_states[2] = "V in component V (dimensionless)"
    legend_states[3] = "Y in component Y (dimensionless)"
    legend_constants[1] = "mu1 in component Y (first_order_rate_constant)"
    legend_constants[2] = "delta1 in component Y (first_order_rate_constant)"
    legend_states[4] = "Z in component Z (dimensionless)"
    legend_algebraic[0] = "CD4 in component CD4 (dimensionless)"
    legend_constants[3] = "mu2 in component V (first_order_rate_constant)"
    legend_constants[4] = "phi in component V (first_order_rate_constant)"
    legend_constants[5] = "delta2 in component V (first_order_rate_constant)"
    legend_constants[6] = "alpha in component Z (first_order_rate_constant)"
    legend_constants[7] = "beta in component Z (first_order_rate_constant)"
    legend_constants[8] = "gamma in component Z (first_order_rate_constant)"
    legend_algebraic[1] = "f_X in component Z (dimensionless)"
    legend_constants[9] = "a in component Z (dimensionless)"
    legend_constants[10] = "b in component Z (dimensionless)"
    legend_constants[11] = "X0 in component Z (dimensionless)"
    legend_constants[12] = "omega in component K (first_order_rate_constant)"
    legend_constants[14] = "Kmax in component K (first_order_rate_constant)"
    legend_rates[0] = "d/dt X in component X (dimensionless)"
    legend_rates[3] = "d/dt Y in component Y (dimensionless)"
    legend_rates[2] = "d/dt V in component V (dimensionless)"
    legend_rates[4] = "d/dt Z in component Z (dimensionless)"
    legend_rates[1] = "d/dt K in component K (first_order_rate_constant)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 1.0E11
    constants[0] = 0.004
    states[1] = 1.35E-14
    states[2] = 1000.0
    states[3] = 100.0
    constants[1] = 0.30
    constants[2] = 0.30
    states[4] = 0.0
    constants[3] = 1.0
    constants[4] = 1000.0
    constants[5] = 1.0
    constants[6] = 1.0E-6
    constants[7] = 0.5
    constants[8] = 0.5
    constants[9] = 1000.0
    constants[10] = 0.2
    constants[11] = 1.0E11
    constants[12] = 0.16E-15
    constants[13] = constants[0]*1.00000e+11
    constants[14] = 1.35000e-14*20.0000
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = constants[13]-(constants[0]*states[0]+states[1]*states[2]*states[0])
    rates[3] = states[1]*states[2]*states[3]-(constants[1]+constants[2]*states[4])*states[3]
    rates[2] = constants[4]*states[3]-(constants[3]+constants[5]*states[4])*states[2]
    rates[1] = constants[12]*states[2]*(constants[14]-states[1])
    algebraic[1] = (power(states[0]/constants[11], 2.00000))/(constants[10]+power(states[0]/constants[11], 2.00000))
    rates[4] = (constants[6]*(states[2]/(constants[9]+states[2]))+constants[7]*algebraic[1]*states[4])-constants[8]*(power(states[4], 2.00000))
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[1] = (power(states[0]/constants[11], 2.00000))/(constants[10]+power(states[0]/constants[11], 2.00000))
    algebraic[0] = (states[0]+states[3])/1.00000e+11
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