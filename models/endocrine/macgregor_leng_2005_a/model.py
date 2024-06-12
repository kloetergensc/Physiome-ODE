# Size of variable arrays:
sizeAlgebraic = 4
sizeStates = 4
sizeConstants = 14
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_states[0] = "r in component r (nanomolar)"
    legend_algebraic[0] = "Ir in component r (flux)"
    legend_constants[0] = "k6 in component r (first_order_rate_constant)"
    legend_states[1] = "s in component s (nanomolar)"
    legend_algebraic[1] = "Is in component model_parameters (flux)"
    legend_constants[1] = "k7 in component model_parameters (first_order_rate_constant)"
    legend_states[2] = "f in component f (dimensionless)"
    legend_constants[2] = "k1 in component f (second_order_rate_constant)"
    legend_constants[3] = "k2 in component f (first_order_rate_constant)"
    legend_constants[4] = "k3 in component f (first_order_rate_constant)"
    legend_algebraic[2] = "phi_b_s in component f (dimensionless)"
    legend_constants[5] = "sb in component f (dimensionless)"
    legend_constants[6] = "delta_b in component f (dimensionless)"
    legend_constants[7] = "c in component model_parameters (nanomolar)"
    legend_states[3] = "h in component h (nanomolar)"
    legend_constants[8] = "k4 in component h (first_order_rate_constant)"
    legend_constants[9] = "k5 in component h (first_order_rate_constant)"
    legend_algebraic[3] = "phi_r_s in component h (dimensionless)"
    legend_constants[10] = "sr in component h (dimensionless)"
    legend_constants[11] = "delta_r in component h (dimensionless)"
    legend_constants[12] = "k8 in component model_parameters (first_order_rate_constant)"
    legend_constants[13] = "j1 in component model_parameters (dimensionless)"
    legend_rates[0] = "d/dt r in component r (nanomolar)"
    legend_rates[1] = "d/dt s in component s (nanomolar)"
    legend_rates[2] = "d/dt f in component f (dimensionless)"
    legend_rates[3] = "d/dt h in component h (nanomolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.0
    constants[0] = 5.0
    states[1] = 0.0
    constants[1] = 5.0
    states[2] = 0.3
    constants[2] = 0.1
    constants[3] = 0.002
    constants[4] = 0.018
    constants[5] = 0.029
    constants[6] = 0.3
    constants[7] = 0.01
    states[3] = 0.0
    constants[8] = 9.0
    constants[9] = 71.0
    constants[10] = -0.56
    constants[11] = 0.3
    constants[12] = 0.07
    constants[13] = 10
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[0] = custom_piecewise([greater_equal(voi , 0.00000) & less_equal(voi , 90.0000), 0.00000 , greater_equal(voi , 91.0000) & less_equal(voi , 92.0000), 10.0000 , greater_equal(voi , 93.0000) & less_equal(voi , 113.000), 0.00000 , greater_equal(voi , 114.000) & less_equal(voi , 115.000), 10.0000 , greater_equal(voi , 116.000) & less_equal(voi , 136.000), 0.00000 , greater_equal(voi , 137.000) & less_equal(voi , 138.000), 10.0000 , greater_equal(voi , 139.000) & less_equal(voi , 159.000), 0.00000 , greater_equal(voi , 160.000) & less_equal(voi , 161.000), 10.0000 , greater_equal(voi , 162.000) & less_equal(voi , 252.000), 0.00000 , greater_equal(voi , 253.000) & less_equal(voi , 254.000), 10.0000 , greater_equal(voi , 255.000) & less_equal(voi , 275.000), 0.00000 , greater_equal(voi , 276.000) & less_equal(voi , 277.000), 10.0000 , greater_equal(voi , 278.000) & less_equal(voi , 298.000), 0.00000 , greater_equal(voi , 299.000) & less_equal(voi , 300.000), 10.0000 , greater_equal(voi , 301.000) & less_equal(voi , 321.000), 0.00000 , greater_equal(voi , 322.000) & less_equal(voi , 323.000), 10.0000 , True, 0.00000])
    rates[0] = algebraic[0]-constants[0]*states[0]
    algebraic[1] = custom_piecewise([greater(voi , 0.00000) & less_equal(voi , 90.0000), 10.0000 , greater(voi , 90.0000) & less_equal(voi , 180.000), 0.00000 , greater(voi , 180.000) & less_equal(voi , 270.000), 10.0000 , greater(voi , 270.000) & less_equal(voi , 360.000), 0.00000 , True, 0.00000])
    rates[1] = algebraic[1]-constants[1]*states[1]
    algebraic[2] = 1.00000/(1.00000+exp(-(log(1.00000*states[1], 10)-constants[5])/constants[6]))
    rates[2] = -(constants[2]*(states[0]+constants[7])*states[2])+(constants[3]+constants[4]*algebraic[2])*(1.00000-states[2])
    algebraic[3] = 1.00000/(1.00000+exp(-(log(1.00000*states[1], 10)-constants[10])/constants[11]))
    rates[3] = constants[13]*((constants[8]+constants[9]*(1.00000-algebraic[3]))*((states[0]+constants[7])*states[2])-constants[12]*states[3])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = custom_piecewise([greater_equal(voi , 0.00000) & less_equal(voi , 90.0000), 0.00000 , greater_equal(voi , 91.0000) & less_equal(voi , 92.0000), 10.0000 , greater_equal(voi , 93.0000) & less_equal(voi , 113.000), 0.00000 , greater_equal(voi , 114.000) & less_equal(voi , 115.000), 10.0000 , greater_equal(voi , 116.000) & less_equal(voi , 136.000), 0.00000 , greater_equal(voi , 137.000) & less_equal(voi , 138.000), 10.0000 , greater_equal(voi , 139.000) & less_equal(voi , 159.000), 0.00000 , greater_equal(voi , 160.000) & less_equal(voi , 161.000), 10.0000 , greater_equal(voi , 162.000) & less_equal(voi , 252.000), 0.00000 , greater_equal(voi , 253.000) & less_equal(voi , 254.000), 10.0000 , greater_equal(voi , 255.000) & less_equal(voi , 275.000), 0.00000 , greater_equal(voi , 276.000) & less_equal(voi , 277.000), 10.0000 , greater_equal(voi , 278.000) & less_equal(voi , 298.000), 0.00000 , greater_equal(voi , 299.000) & less_equal(voi , 300.000), 10.0000 , greater_equal(voi , 301.000) & less_equal(voi , 321.000), 0.00000 , greater_equal(voi , 322.000) & less_equal(voi , 323.000), 10.0000 , True, 0.00000])
    algebraic[1] = custom_piecewise([greater(voi , 0.00000) & less_equal(voi , 90.0000), 10.0000 , greater(voi , 90.0000) & less_equal(voi , 180.000), 0.00000 , greater(voi , 180.000) & less_equal(voi , 270.000), 10.0000 , greater(voi , 270.000) & less_equal(voi , 360.000), 0.00000 , True, 0.00000])
    algebraic[2] = 1.00000/(1.00000+exp(-(log(1.00000*states[1], 10)-constants[5])/constants[6]))
    algebraic[3] = 1.00000/(1.00000+exp(-(log(1.00000*states[1], 10)-constants[10])/constants[11]))
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