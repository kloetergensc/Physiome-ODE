# Size of variable arrays:
sizeAlgebraic = 1
sizeStates = 20
sizeConstants = 17
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_states[0] = "Py in component Py (nanomolar)"
    legend_states[1] = "Zn in component Zn (nanomolar)"
    legend_states[2] = "Py1 in component Py1 (nanomolar)"
    legend_constants[0] = "r3 in component model_parameters (third_order_rate_constant)"
    legend_constants[1] = "r4 in component model_parameters (first_order_rate_constant)"
    legend_states[3] = "Dw in component Dw (nanomolar)"
    legend_states[4] = "Qw2 in component Qw2 (nanomolar)"
    legend_constants[2] = "k_1 in component model_parameters (first_order_rate_constant)"
    legend_constants[3] = "k1a in component model_parameters (second_order_rate_constant)"
    legend_states[5] = "Qw1 in component Qw1 (nanomolar)"
    legend_states[6] = "Rw in component Rw (nanomolar)"
    legend_constants[4] = "k2 in component model_parameters (second_order_rate_constant)"
    legend_constants[5] = "k_2 in component model_parameters (first_order_rate_constant)"
    legend_algebraic[0] = "k3 in component model_parameters (first_order_rate_constant)"
    legend_states[7] = "Mw in component Mw (nanomolar)"
    legend_states[8] = "Px in component Px (nanomolar)"
    legend_states[9] = "Px1 in component Px1 (nanomolar)"
    legend_states[10] = "Dz in component Dz (nanomolar)"
    legend_states[11] = "Qz4 in component Qz4 (nanomolar)"
    legend_constants[6] = "r1 in component model_parameters (second_order_rate_constant)"
    legend_constants[7] = "r2 in component model_parameters (first_order_rate_constant)"
    legend_constants[8] = "k1b in component model_parameters (second_order_rate_constant)"
    legend_states[12] = "Qz2 in component Qz2 (nanomolar)"
    legend_constants[9] = "k1 in component model_parameters (second_order_rate_constant)"
    legend_states[13] = "Tp in component Tp (nanomolar)"
    legend_states[14] = "Tp1 in component Tp1 (nanomolar)"
    legend_constants[10] = "r5 in component model_parameters (second_order_rate_constant)"
    legend_constants[11] = "r6 in component model_parameters (first_order_rate_constant)"
    legend_states[15] = "Qz1 in component Qz1 (nanomolar)"
    legend_states[16] = "Rz in component Rz (nanomolar)"
    legend_constants[12] = "k2a in component model_parameters (second_order_rate_constant)"
    legend_states[17] = "Qz3 in component Qz3 (nanomolar)"
    legend_states[18] = "Qz5 in component Qz5 (nanomolar)"
    legend_constants[13] = "k2b in component model_parameters (second_order_rate_constant)"
    legend_constants[14] = "k2c in component model_parameters (second_order_rate_constant)"
    legend_states[19] = "Mz in component Mz (nanomolar)"
    legend_constants[15] = "td0 in component model_parameters (second)"
    legend_constants[16] = "td in component model_parameters (second)"
    legend_rates[0] = "d/dt Py in component Py (nanomolar)"
    legend_rates[2] = "d/dt Py1 in component Py1 (nanomolar)"
    legend_rates[3] = "d/dt Dw in component Dw (nanomolar)"
    legend_rates[6] = "d/dt Rw in component Rw (nanomolar)"
    legend_rates[5] = "d/dt Qw1 in component Qw1 (nanomolar)"
    legend_rates[4] = "d/dt Qw2 in component Qw2 (nanomolar)"
    legend_rates[7] = "d/dt Mw in component Mw (nanomolar)"
    legend_rates[8] = "d/dt Px in component Px (nanomolar)"
    legend_rates[9] = "d/dt Px1 in component Px1 (nanomolar)"
    legend_rates[13] = "d/dt Tp in component Tp (nanomolar)"
    legend_rates[14] = "d/dt Tp1 in component Tp1 (nanomolar)"
    legend_rates[1] = "d/dt Zn in component Zn (nanomolar)"
    legend_rates[10] = "d/dt Dz in component Dz (nanomolar)"
    legend_rates[16] = "d/dt Rz in component Rz (nanomolar)"
    legend_rates[15] = "d/dt Qz1 in component Qz1 (nanomolar)"
    legend_rates[12] = "d/dt Qz2 in component Qz2 (nanomolar)"
    legend_rates[17] = "d/dt Qz3 in component Qz3 (nanomolar)"
    legend_rates[11] = "d/dt Qz4 in component Qz4 (nanomolar)"
    legend_rates[18] = "d/dt Qz5 in component Qz5 (nanomolar)"
    legend_rates[19] = "d/dt Mz in component Mz (nanomolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 25.0
    states[1] = 10000.0
    states[2] = 0.0
    constants[0] = 4.41E10
    constants[1] = 9E-3
    states[3] = 4.0
    states[4] = 0.0
    constants[2] = 0.9
    constants[3] = 1.0
    states[5] = 0.0
    states[6] = 50.0
    constants[4] = 0.02
    constants[5] = 0.3
    states[7] = 0.0
    states[8] = 50.0
    states[9] = 0.0
    states[10] = 2.0
    states[11] = 0.0
    constants[6] = 2.73E2
    constants[7] = 3.437E-4
    constants[8] = 1.253E-2
    states[12] = 0.0
    constants[9] = 0.025
    states[13] = 10000.0
    states[14] = 0.0
    constants[10] = 3E4
    constants[11] = 1.506E-2
    states[15] = 0.0
    states[16] = 100.0
    constants[12] = 0.00005
    states[17] = 0.0
    states[18] = 0.0
    constants[13] = 0.0002
    constants[14] = 0.0037
    states[19] = 0.0
    constants[15] = 1800.0
    constants[16] = 2700
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = constants[1]*states[2]-constants[0]*(power(states[1], 2.00000))*states[0]
    rates[2] = (constants[0]*(power(states[1], 2.00000))*states[0]+constants[2]*states[4])-(constants[1]*states[2]+constants[3]*states[3]*states[2])
    rates[4] = constants[3]*states[3]*states[2]-constants[2]*states[4]
    rates[8] = (constants[7]*states[9]+constants[2]*states[11])-(constants[6]*states[1]*states[8]+constants[8]*states[10]*states[8])
    rates[9] = (constants[6]*states[1]*states[8]+constants[2]*states[12])-(constants[7]*states[9]+constants[9]*states[10]*states[9])
    rates[13] = constants[11]*states[14]-constants[10]*states[1]*states[13]
    rates[14] = constants[10]*states[1]*states[13]-constants[11]*states[14]
    rates[1] = (constants[7]*states[9]+constants[11]*states[14])-(constants[6]*states[1]*states[8]+constants[10]*states[1]*states[13])
    algebraic[0] = custom_piecewise([greater_equal(voi , 0.00000) & less(voi , constants[15]), 0.00000 , greater_equal(voi , constants[15]) & less(voi , constants[16]), 0.0110000 , True, float('nan')])
    rates[3] = (constants[2]*states[4]+algebraic[0]*states[5]+constants[5]*states[5])-(constants[3]*states[3]*states[2]+constants[4]*states[3]*states[6])
    rates[6] = (algebraic[0]*states[5]+constants[5]*states[5])-constants[4]*states[3]*states[6]
    rates[5] = constants[4]*states[3]*states[6]-(algebraic[0]*states[5]+constants[5]*states[5])
    rates[7] = algebraic[0]*states[5]
    rates[10] = (constants[2]*states[12]+algebraic[0]*states[15]+constants[5]*states[15]+constants[2]*states[11])-(constants[8]*states[10]*states[8]+constants[9]*states[10]*states[9]+constants[12]*states[10]*states[16])
    rates[16] = (algebraic[0]*states[15]+constants[5]*states[15]+algebraic[0]*states[17]+constants[5]*states[17]+algebraic[0]*states[18]+constants[5]*states[18])-(constants[12]*states[10]*states[16]+constants[13]*states[11]*states[16]+constants[14]*states[12]*states[16])
    rates[15] = constants[12]*states[10]*states[16]-(algebraic[0]*states[15]+constants[5]*states[15])
    rates[12] = (constants[9]*states[10]*states[9]+algebraic[0]*states[17]+constants[5]*states[17])-(constants[2]*states[12]+constants[14]*states[12]*states[16])
    rates[17] = constants[14]*states[12]*states[16]-(algebraic[0]*states[17]+constants[5]*states[17])
    rates[11] = (constants[8]*states[10]*states[8]+algebraic[0]*states[18]+constants[5]*states[18])-(constants[2]*states[11]+constants[13]*states[11]*states[16])
    rates[18] = constants[13]*states[11]*states[16]-(algebraic[0]*states[18]+constants[5]*states[18])
    rates[19] = algebraic[0]*states[15]+algebraic[0]*states[17]+algebraic[0]*states[18]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = custom_piecewise([greater_equal(voi , 0.00000) & less(voi , constants[15]), 0.00000 , greater_equal(voi , constants[15]) & less(voi , constants[16]), 0.0110000 , True, float('nan')])
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