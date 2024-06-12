# Size of variable arrays:
sizeAlgebraic = 2
sizeStates = 5
sizeConstants = 35
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (hour)"
    legend_states[0] = "GH in component GH (ng_ml)"
    legend_constants[0] = "GHS in component GH (dimensionless)"
    legend_constants[1] = "k1 in component GH (first_order_rate_constant)"
    legend_constants[2] = "kr1 in component GH (ng_ml_hr)"
    legend_constants[3] = "t1 in component GH (pg_ml)"
    legend_constants[4] = "n1 in component GH (dimensionless)"
    legend_constants[5] = "n2 in component GH (dimensionless)"
    legend_constants[6] = "g0 in component GH (dimensionless)"
    legend_constants[7] = "ng0 in component GH (dimensionless)"
    legend_constants[8] = "tg0 in component GH (dimensionless)"
    legend_constants[9] = "t2 in component GH (pg_ml)"
    legend_states[1] = "SRIF_PeV in component SRIF_PeV (pg_ml)"
    legend_constants[33] = "F1_GHS in component F (dimensionless)"
    legend_states[2] = "GHRH in component GHRH (pg_ml)"
    legend_states[3] = "ghr_GHRH in component ghr_GHRH (pg_ml)"
    legend_constants[10] = "k4 in component SRIF_PeV (first_order_rate_constant)"
    legend_constants[11] = "kr4 in component SRIF_PeV (pg_ml_hr)"
    legend_constants[12] = "t5 in component SRIF_PeV (ng_ml)"
    legend_constants[13] = "n5 in component SRIF_PeV (dimensionless)"
    legend_constants[14] = "S_basal in component SRIF_PeV (pg_ml_hr)"
    legend_states[4] = "SRIF_ArC in component SRIF_ArC (pg_ml)"
    legend_constants[15] = "k2 in component SRIF_ArC (first_order_rate_constant)"
    legend_constants[16] = "kr2 in component SRIF_ArC (pg_ml_hr)"
    legend_constants[17] = "t3 in component SRIF_ArC (pg_ml)"
    legend_constants[18] = "n3 in component SRIF_ArC (dimensionless)"
    legend_constants[19] = "k3 in component GHRH (first_order_rate_constant)"
    legend_constants[20] = "kr3 in component GHRH (pg_ml_hr)"
    legend_constants[21] = "t4 in component GHRH (pg_ml)"
    legend_constants[22] = "n4 in component GHRH (dimensionless)"
    legend_constants[34] = "F2_GHS in component F (dimensionless)"
    legend_constants[23] = "g1 in component F (dimensionless)"
    legend_constants[24] = "g2 in component F (dimensionless)"
    legend_constants[25] = "tg1 in component F (dimensionless)"
    legend_constants[26] = "tg2 in component F (dimensionless)"
    legend_constants[27] = "ng1 in component F (dimensionless)"
    legend_constants[28] = "ng2 in component F (dimensionless)"
    legend_algebraic[1] = "dghr_GHRH_dt in component ghr_GHRH (pg_ml_hr)"
    legend_algebraic[0] = "inject in component ghr_GHRH (pg_ml_hr)"
    legend_constants[29] = "kghr in component ghr_GHRH (first_order_rate_constant)"
    legend_constants[30] = "C in component ghr_GHRH (pg_ml_hr)"
    legend_constants[31] = "onset in component ghr_GHRH (hour)"
    legend_constants[32] = "duration in component ghr_GHRH (hour)"
    legend_rates[0] = "d/dt GH in component GH (ng_ml)"
    legend_rates[1] = "d/dt SRIF_PeV in component SRIF_PeV (pg_ml)"
    legend_rates[4] = "d/dt SRIF_ArC in component SRIF_ArC (pg_ml)"
    legend_rates[2] = "d/dt GHRH in component GHRH (pg_ml)"
    legend_rates[3] = "d/dt ghr_GHRH in component ghr_GHRH (pg_ml)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.0
    constants[0] = 20.0
    constants[1] = 3.0
    constants[2] = 600.0
    constants[3] = 357.0
    constants[4] = 5.0
    constants[5] = 2.0
    constants[6] = 2.0
    constants[7] = 2.9
    constants[8] = 200.0
    constants[9] = 10.0
    states[1] = 0.0
    states[2] = 0.0
    states[3] = 0.0
    constants[10] = 25.0
    constants[11] = 20400.0
    constants[12] = 10.0
    constants[13] = 2.0
    constants[14] = 900.0
    states[4] = 0.0
    constants[15] = 25.0
    constants[16] = 2200.0
    constants[17] = 400.0
    constants[18] = 2.0
    constants[19] = 40.0
    constants[20] = 63000.0
    constants[21] = 28.0
    constants[22] = 5.0
    constants[23] = 90000.0
    constants[24] = 100.0
    constants[25] = 390.0
    constants[26] = 10000.0
    constants[27] = 3.0
    constants[28] = 2.0
    constants[29] = 15.0
    constants[30] = 10000.0
    constants[31] = 2.0
    constants[32] = 0.2
    constants[33] = constants[23]*((power(constants[0]/constants[25], constants[27]))/(1.00000+power(constants[0]/constants[25], constants[27])))
    constants[34] = constants[24]*((power(constants[0]/constants[26], constants[28]))/(1.00000+power(constants[0]/constants[26], constants[28])))
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = constants[2]*((power((states[2]+states[3])/constants[3], constants[4]))/(power((states[2]+states[3])/constants[3], constants[4])+1.00000)+constants[6]*((power(constants[0]/constants[8], constants[7]))/(1.00000+power(constants[0]/constants[8], constants[7])))+(1.00000+constants[33])/(1.00000+power(states[1]/constants[9], constants[5])+constants[33]))-constants[1]*states[0]
    rates[1] = -(constants[10]*states[1])+constants[11]*((power(states[0]/constants[12], constants[13]))/(power(states[0]/constants[12], constants[13])+1.00000))+constants[14]
    rates[4] = constants[16]*((power((states[2]+states[3])/constants[17], constants[18]))/(1.00000+power((states[2]+states[3])/constants[17], constants[18])))-constants[15]*states[4]
    rates[2] = (constants[20]*((1.00000+constants[34])/(1.00000+power((states[1]+states[4])/constants[21], constants[22])+constants[34]))+states[3]*1.00000)-constants[19]*states[2]
    algebraic[0] = custom_piecewise([less(voi , constants[31]), 0.00000 , greater_equal(voi , constants[31]) & less_equal(voi , constants[31]+constants[32]), constants[30] , greater(voi , constants[31]+constants[32]), 0.00000 , True, float('nan')])
    rates[3] = algebraic[0]-constants[29]*states[3]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = custom_piecewise([less(voi , constants[31]), 0.00000 , greater_equal(voi , constants[31]) & less_equal(voi , constants[31]+constants[32]), constants[30] , greater(voi , constants[31]+constants[32]), 0.00000 , True, float('nan')])
    algebraic[1] = algebraic[0]-constants[29]*states[3]
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