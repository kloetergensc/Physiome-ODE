# Size of variable arrays:
sizeAlgebraic = 3
sizeStates = 5
sizeConstants = 18
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (day)"
    legend_states[0] = "T in component T (per_ml)"
    legend_constants[0] = "lamda in component T (second_order_rate_constant)"
    legend_constants[1] = "d in component T (first_order_rate_constant)"
    legend_constants[2] = "k in component kinetic_parameters (flux)"
    legend_states[1] = "VI in component VI (per_ml)"
    legend_states[2] = "T_ in component T_ (per_ml)"
    legend_constants[3] = "tau in component T_ (first_order_rate_constant)"
    legend_constants[4] = "m in component T_ (first_order_rate_constant)"
    legend_constants[5] = "delta in component kinetic_parameters (first_order_rate_constant)"
    legend_constants[6] = "N in component kinetic_parameters (dimensionless)"
    legend_constants[7] = "c in component kinetic_parameters (first_order_rate_constant)"
    legend_algebraic[0] = "epsilon_PI in component epsilon_PI (dimensionless)"
    legend_states[3] = "VNI in component VNI (per_ml)"
    legend_constants[8] = "IC50 in component epsilon_PI (mg_per_ml)"
    legend_states[4] = "Cc in component Cc (mg_per_ml)"
    legend_algebraic[1] = "Cb in component Cb (mg_per_ml)"
    legend_constants[9] = "Vd in component Cb (ml)"
    legend_constants[10] = "F in component Cb (dimensionless)"
    legend_constants[11] = "D in component Cb (mg)"
    legend_constants[12] = "ka in component kinetic_parameters (first_order_rate_constant)"
    legend_constants[13] = "ke in component kinetic_parameters (first_order_rate_constant)"
    legend_constants[14] = "kacell in component Cc (first_order_rate_constant)"
    legend_constants[15] = "kecell in component Cc (first_order_rate_constant)"
    legend_algebraic[2] = "Cx in component Cx (mg_per_ml)"
    legend_constants[16] = "H in component Cx (dimensionless)"
    legend_constants[17] = "fb in component Cx (dimensionless)"
    legend_rates[0] = "d/dt T in component T (per_ml)"
    legend_rates[2] = "d/dt T_ in component T_ (per_ml)"
    legend_rates[1] = "d/dt VI in component VI (per_ml)"
    legend_rates[3] = "d/dt VNI in component VNI (per_ml)"
    legend_rates[4] = "d/dt Cc in component Cc (mg_per_ml)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 1e6
    constants[0] = 1e4
    constants[1] = 0.01
    constants[2] = 2.4e-8
    states[1] = 1
    states[2] = 1
    constants[3] = 1.5
    constants[4] = 0.01
    constants[5] = 0.01
    constants[6] = 2500
    constants[7] = 23
    states[3] = 2
    constants[8] = 9e-7
    states[4] = 0
    constants[9] = 28000
    constants[10] = 1
    constants[11] = 600
    constants[12] = 14.64
    constants[13] = 6.86
    constants[14] = 24000
    constants[15] = 1.1
    constants[16] = 0.052
    constants[17] = 0.99
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = constants[0]-(constants[1]*states[0]+constants[2]*states[0]*states[1])
    rates[2] = constants[2]*states[0]*(voi-constants[3])*states[1]*(voi-constants[3])*exp(-constants[4]*constants[3])-constants[5]*states[2]
    algebraic[0] = states[4]/(constants[8]+states[4])
    rates[1] = constants[6]*constants[5]*states[2]*(1.00000-algebraic[0])-constants[7]*states[1]
    rates[3] = constants[6]*constants[5]*states[2]*algebraic[0]-constants[7]*states[3]
    algebraic[1] = ((constants[10]*constants[11])/constants[9])*(constants[12]/(constants[13]-constants[12]))*(exp(-constants[12]*voi)-exp(-constants[13]*voi))
    algebraic[2] = custom_piecewise([greater((1.00000-constants[17])*constants[16]*algebraic[1]-states[4] , 0.00000), (1.00000-constants[17])*constants[16]*algebraic[1]-states[4] , True, 0.00000])
    rates[4] = constants[14]*algebraic[2]-constants[15]*states[4]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = states[4]/(constants[8]+states[4])
    algebraic[1] = ((constants[10]*constants[11])/constants[9])*(constants[12]/(constants[13]-constants[12]))*(exp(-constants[12]*voi)-exp(-constants[13]*voi))
    algebraic[2] = custom_piecewise([greater((1.00000-constants[17])*constants[16]*algebraic[1]-states[4] , 0.00000), (1.00000-constants[17])*constants[16]*algebraic[1]-states[4] , True, 0.00000])
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