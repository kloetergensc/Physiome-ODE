# Size of variable arrays:
sizeAlgebraic = 15
sizeStates = 5
sizeConstants = 24
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_states[0] = "y0 in component y0 (dimensionless)"
    legend_algebraic[10] = "g0 in component y0 (dimensionless)"
    legend_algebraic[5] = "r0 in component y0 (dimensionless)"
    legend_algebraic[0] = "h0 in component y0 (dimensionless)"
    legend_constants[0] = "S0 in component y0 (dimensionless)"
    legend_states[1] = "y2 in component y2 (dimensionless)"
    legend_constants[1] = "a00 in component model_parameters (dimensionless)"
    legend_constants[2] = "a02 in component model_parameters (dimensionless)"
    legend_constants[3] = "c0 in component model_parameters (first_order_rate_constant)"
    legend_constants[4] = "e0 in component model_parameters (first_order_rate_constant)"
    legend_constants[5] = "epsilon in component model_parameters (dimensionless)"
    legend_states[2] = "y1 in component y1 (dimensionless)"
    legend_algebraic[11] = "g1 in component y1 (dimensionless)"
    legend_algebraic[6] = "r1 in component y1 (dimensionless)"
    legend_algebraic[1] = "h1 in component y1 (dimensionless)"
    legend_constants[6] = "S1 in component y1 (dimensionless)"
    legend_constants[7] = "a10 in component model_parameters (dimensionless)"
    legend_constants[8] = "a12 in component model_parameters (dimensionless)"
    legend_constants[9] = "a11 in component model_parameters (dimensionless)"
    legend_constants[10] = "e1 in component model_parameters (first_order_rate_constant)"
    legend_algebraic[12] = "g2 in component y2 (dimensionless)"
    legend_algebraic[7] = "r2 in component y2 (dimensionless)"
    legend_algebraic[2] = "h2 in component y2 (dimensionless)"
    legend_constants[11] = "S2 in component y2 (dimensionless)"
    legend_states[3] = "y3 in component y3 (dimensionless)"
    legend_states[4] = "y4 in component y4 (dimensionless)"
    legend_constants[12] = "a23 in component model_parameters (dimensionless)"
    legend_constants[13] = "a24 in component model_parameters (dimensionless)"
    legend_constants[14] = "a20 in component model_parameters (dimensionless)"
    legend_constants[15] = "a21 in component model_parameters (dimensionless)"
    legend_constants[16] = "a22 in component model_parameters (dimensionless)"
    legend_constants[17] = "e2 in component model_parameters (first_order_rate_constant)"
    legend_algebraic[13] = "g3 in component y3 (dimensionless)"
    legend_algebraic[8] = "r3 in component y3 (dimensionless)"
    legend_algebraic[3] = "h3 in component y3 (dimensionless)"
    legend_constants[18] = "S3 in component y3 (dimensionless)"
    legend_constants[19] = "a32 in component model_parameters (dimensionless)"
    legend_constants[20] = "a33 in component model_parameters (dimensionless)"
    legend_algebraic[14] = "g4 in component y4 (dimensionless)"
    legend_algebraic[9] = "r4 in component y4 (dimensionless)"
    legend_algebraic[4] = "h4 in component y4 (dimensionless)"
    legend_constants[21] = "S4 in component y4 (dimensionless)"
    legend_constants[22] = "a42 in component model_parameters (dimensionless)"
    legend_constants[23] = "a44 in component model_parameters (dimensionless)"
    legend_rates[0] = "d/dt y0 in component y0 (dimensionless)"
    legend_rates[2] = "d/dt y1 in component y1 (dimensionless)"
    legend_rates[1] = "d/dt y2 in component y2 (dimensionless)"
    legend_rates[3] = "d/dt y3 in component y3 (dimensionless)"
    legend_rates[4] = "d/dt y4 in component y4 (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.4
    constants[0] = 0.010
    states[1] = 1.17
    constants[1] = -0.00843
    constants[2] = -0.440
    constants[3] = 0.443
    constants[4] = 0.0
    constants[5] = 0.50
    states[2] = 1.4
    constants[6] = 0.010
    constants[7] = 0.082
    constants[8] = -0.0668
    constants[9] = -0.0040
    constants[10] = 0.0
    constants[11] = 0.010
    states[3] = 0.95
    states[4] = 0.65
    constants[12] = 0.0576
    constants[13] = 3.25E-4
    constants[14] = 0.0
    constants[15] = 0.0310
    constants[16] = -0.0957
    constants[17] = 0.0
    constants[18] = 0.010
    constants[19] = 0.00869
    constants[20] = -0.00857
    constants[21] = 0.010
    constants[22] = 1.39E-4
    constants[23] = -1.43E-4
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[5] = custom_piecewise([less(states[0] , constants[5]) & less(constants[1]*states[0]+constants[2]*states[1] , 0.00000), 1.00000-exp((constants[0]*(power(states[0], 2.00000)))/((constants[1]*states[0]+constants[2]*states[1])*(power(constants[5]-states[0], 2.00000)))) , True, 1.00000])
    algebraic[0] = custom_piecewise([greater(constants[1]*states[0]+constants[2]*states[1] , 0.00000), (constants[1]*states[0]+constants[2]*states[1])/(1.00000+((constants[1]*states[0]+constants[2]*states[1])/constants[0])*(1.00000-exp(-((constants[1]*states[0]+constants[2]*states[1])/constants[0])))) , less_equal(constants[1]*states[0]+constants[2]*states[1] , 0.00000), constants[1]*states[0]+constants[2]*states[1] , True, float('nan')])
    algebraic[10] = algebraic[0]*algebraic[5]
    rates[0] = 1.00000*algebraic[10]+constants[3]+constants[4]
    algebraic[6] = custom_piecewise([less(states[2] , constants[5]) & less(constants[7]*states[0]+constants[9]*states[2]+constants[8]*states[1] , 0.00000), 1.00000-exp((constants[6]*(power(states[2], 2.00000)))/((constants[7]*states[0]+constants[9]*states[2]+constants[8]*states[1])*(power(constants[5]-states[2], 2.00000)))) , True, 1.00000])
    algebraic[1] = custom_piecewise([greater(constants[7]*states[0]+constants[9]*states[2]+constants[8]*states[1] , 0.00000), (constants[7]*states[0]+constants[9]*states[2]+constants[8]*states[1])/(1.00000+((constants[7]*states[0]+constants[9]*states[2]+constants[8]*states[1])/constants[6])*(1.00000-exp(-((constants[7]*states[0]+constants[9]*states[2]+constants[8]*states[1])/constants[6])))) , less_equal(constants[7]*states[0]+constants[9]*states[2]+constants[8]*states[1] , 0.00000), constants[7]*states[0]+constants[9]*states[2]+constants[8]*states[1] , True, float('nan')])
    algebraic[11] = algebraic[1]*algebraic[6]
    rates[2] = 1.00000*algebraic[11]+constants[10]
    algebraic[7] = custom_piecewise([less(states[1] , constants[5]) & less(constants[14]*states[0]+constants[15]*states[2]+constants[16]*states[1]+constants[12]*states[3]+constants[13]*states[4] , 0.00000), 1.00000-exp((constants[11]*(power(states[1], 2.00000)))/((constants[14]*states[0]+constants[15]*states[2]+constants[16]*states[1]+constants[12]*states[3]+constants[13]*states[4])*(power(constants[5]-states[1], 2.00000)))) , True, 1.00000])
    algebraic[2] = custom_piecewise([greater(constants[14]*states[0]+constants[15]*states[2]+constants[16]*states[1]+constants[12]*states[3]+constants[13]*states[4] , 0.00000), (constants[14]*states[0]+constants[15]*states[2]+constants[16]*states[1]+constants[12]*states[3]+constants[13]*states[4])/(1.00000+((constants[14]*states[0]+constants[15]*states[2]+constants[16]*states[1]+constants[12]*states[3]+constants[13]*states[4])/constants[11])*(1.00000-exp(-((constants[14]*states[0]+constants[15]*states[2]+constants[16]*states[1]+constants[12]*states[3]+constants[13]*states[4])/constants[11])))) , less_equal(constants[14]*states[0]+constants[15]*states[2]+constants[16]*states[1]+constants[12]*states[3]+constants[13]*states[4] , 0.00000), constants[14]*states[0]+constants[15]*states[2]+constants[16]*states[1]+constants[12]*states[3]+constants[13]*states[4] , True, float('nan')])
    algebraic[12] = algebraic[2]*algebraic[7]
    rates[1] = 1.00000*algebraic[12]+constants[17]
    algebraic[8] = custom_piecewise([less(states[3] , constants[5]) & less(constants[19]*states[1]+constants[20]*states[3] , 0.00000), 1.00000-exp((constants[18]*(power(states[3], 2.00000)))/((constants[19]*states[1]+constants[20]*states[3])*(power(constants[5]-states[3], 2.00000)))) , True, 1.00000])
    algebraic[3] = custom_piecewise([greater(constants[19]*states[1]+constants[20]*states[3] , 0.00000), (constants[19]*states[1]+constants[20]*states[3])/(1.00000+((constants[19]*states[1]+constants[20]*states[3])/constants[18])*(1.00000-exp(-((constants[19]*states[1]+constants[20]*states[3])/constants[18])))) , less_equal(constants[19]*states[1]+constants[20]*states[3] , 0.00000), constants[19]*states[1]+constants[20]*states[3] , True, float('nan')])
    algebraic[13] = algebraic[3]*algebraic[8]
    rates[3] = 1.00000*algebraic[13]
    algebraic[9] = custom_piecewise([less(states[4] , constants[5]) & less(constants[22]*states[1]+constants[23]*states[4] , 0.00000), 1.00000-exp((constants[21]*(power(states[4], 2.00000)))/((constants[22]*states[1]+constants[23]*states[4])*(power(constants[5]-states[4], 2.00000)))) , True, 1.00000])
    algebraic[4] = custom_piecewise([greater(constants[22]*states[1]+constants[23]*states[4] , 0.00000), (constants[22]*states[1]+constants[23]*states[4])/(1.00000+((constants[22]*states[1]+constants[23]*states[4])/constants[21])*(1.00000-exp(-((constants[22]*states[1]+constants[23]*states[4])/constants[21])))) , less_equal(constants[22]*states[1]+constants[23]*states[4] , 0.00000), constants[22]*states[1]+constants[23]*states[4] , True, float('nan')])
    algebraic[14] = algebraic[4]*algebraic[9]
    rates[4] = 1.00000*algebraic[14]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[5] = custom_piecewise([less(states[0] , constants[5]) & less(constants[1]*states[0]+constants[2]*states[1] , 0.00000), 1.00000-exp((constants[0]*(power(states[0], 2.00000)))/((constants[1]*states[0]+constants[2]*states[1])*(power(constants[5]-states[0], 2.00000)))) , True, 1.00000])
    algebraic[0] = custom_piecewise([greater(constants[1]*states[0]+constants[2]*states[1] , 0.00000), (constants[1]*states[0]+constants[2]*states[1])/(1.00000+((constants[1]*states[0]+constants[2]*states[1])/constants[0])*(1.00000-exp(-((constants[1]*states[0]+constants[2]*states[1])/constants[0])))) , less_equal(constants[1]*states[0]+constants[2]*states[1] , 0.00000), constants[1]*states[0]+constants[2]*states[1] , True, float('nan')])
    algebraic[10] = algebraic[0]*algebraic[5]
    algebraic[6] = custom_piecewise([less(states[2] , constants[5]) & less(constants[7]*states[0]+constants[9]*states[2]+constants[8]*states[1] , 0.00000), 1.00000-exp((constants[6]*(power(states[2], 2.00000)))/((constants[7]*states[0]+constants[9]*states[2]+constants[8]*states[1])*(power(constants[5]-states[2], 2.00000)))) , True, 1.00000])
    algebraic[1] = custom_piecewise([greater(constants[7]*states[0]+constants[9]*states[2]+constants[8]*states[1] , 0.00000), (constants[7]*states[0]+constants[9]*states[2]+constants[8]*states[1])/(1.00000+((constants[7]*states[0]+constants[9]*states[2]+constants[8]*states[1])/constants[6])*(1.00000-exp(-((constants[7]*states[0]+constants[9]*states[2]+constants[8]*states[1])/constants[6])))) , less_equal(constants[7]*states[0]+constants[9]*states[2]+constants[8]*states[1] , 0.00000), constants[7]*states[0]+constants[9]*states[2]+constants[8]*states[1] , True, float('nan')])
    algebraic[11] = algebraic[1]*algebraic[6]
    algebraic[7] = custom_piecewise([less(states[1] , constants[5]) & less(constants[14]*states[0]+constants[15]*states[2]+constants[16]*states[1]+constants[12]*states[3]+constants[13]*states[4] , 0.00000), 1.00000-exp((constants[11]*(power(states[1], 2.00000)))/((constants[14]*states[0]+constants[15]*states[2]+constants[16]*states[1]+constants[12]*states[3]+constants[13]*states[4])*(power(constants[5]-states[1], 2.00000)))) , True, 1.00000])
    algebraic[2] = custom_piecewise([greater(constants[14]*states[0]+constants[15]*states[2]+constants[16]*states[1]+constants[12]*states[3]+constants[13]*states[4] , 0.00000), (constants[14]*states[0]+constants[15]*states[2]+constants[16]*states[1]+constants[12]*states[3]+constants[13]*states[4])/(1.00000+((constants[14]*states[0]+constants[15]*states[2]+constants[16]*states[1]+constants[12]*states[3]+constants[13]*states[4])/constants[11])*(1.00000-exp(-((constants[14]*states[0]+constants[15]*states[2]+constants[16]*states[1]+constants[12]*states[3]+constants[13]*states[4])/constants[11])))) , less_equal(constants[14]*states[0]+constants[15]*states[2]+constants[16]*states[1]+constants[12]*states[3]+constants[13]*states[4] , 0.00000), constants[14]*states[0]+constants[15]*states[2]+constants[16]*states[1]+constants[12]*states[3]+constants[13]*states[4] , True, float('nan')])
    algebraic[12] = algebraic[2]*algebraic[7]
    algebraic[8] = custom_piecewise([less(states[3] , constants[5]) & less(constants[19]*states[1]+constants[20]*states[3] , 0.00000), 1.00000-exp((constants[18]*(power(states[3], 2.00000)))/((constants[19]*states[1]+constants[20]*states[3])*(power(constants[5]-states[3], 2.00000)))) , True, 1.00000])
    algebraic[3] = custom_piecewise([greater(constants[19]*states[1]+constants[20]*states[3] , 0.00000), (constants[19]*states[1]+constants[20]*states[3])/(1.00000+((constants[19]*states[1]+constants[20]*states[3])/constants[18])*(1.00000-exp(-((constants[19]*states[1]+constants[20]*states[3])/constants[18])))) , less_equal(constants[19]*states[1]+constants[20]*states[3] , 0.00000), constants[19]*states[1]+constants[20]*states[3] , True, float('nan')])
    algebraic[13] = algebraic[3]*algebraic[8]
    algebraic[9] = custom_piecewise([less(states[4] , constants[5]) & less(constants[22]*states[1]+constants[23]*states[4] , 0.00000), 1.00000-exp((constants[21]*(power(states[4], 2.00000)))/((constants[22]*states[1]+constants[23]*states[4])*(power(constants[5]-states[4], 2.00000)))) , True, 1.00000])
    algebraic[4] = custom_piecewise([greater(constants[22]*states[1]+constants[23]*states[4] , 0.00000), (constants[22]*states[1]+constants[23]*states[4])/(1.00000+((constants[22]*states[1]+constants[23]*states[4])/constants[21])*(1.00000-exp(-((constants[22]*states[1]+constants[23]*states[4])/constants[21])))) , less_equal(constants[22]*states[1]+constants[23]*states[4] , 0.00000), constants[22]*states[1]+constants[23]*states[4] , True, float('nan')])
    algebraic[14] = algebraic[4]*algebraic[9]
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