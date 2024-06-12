# Size of variable arrays:
sizeAlgebraic = 11
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
    legend_voi = "time in component environment (hour)"
    legend_algebraic[0] = "Metin in component environment (flux)"
    legend_states[0] = "Met in component Met (micromolar)"
    legend_algebraic[7] = "V_MS in component V_MS (flux)"
    legend_algebraic[10] = "V_BHMT in component V_BHMT (flux)"
    legend_algebraic[1] = "V_MATI in component V_MATI (flux)"
    legend_algebraic[3] = "V_MATIII in component V_MATIII (flux)"
    legend_states[1] = "AdoMet in component AdoMet (micromolar)"
    legend_algebraic[6] = "V_METH in component V_METH (flux)"
    legend_algebraic[4] = "V_GNMT in component V_GNMT (flux)"
    legend_states[2] = "AdoHcy in component AdoHcy (micromolar)"
    legend_algebraic[8] = "V_AH in component V_AH (flux)"
    legend_states[3] = "Hcy in component Hcy (micromolar)"
    legend_algebraic[9] = "V_CBS in component V_CBS (flux)"
    legend_constants[0] = "V_MATImax in component V_MATI (flux)"
    legend_constants[1] = "Km_MATI in component V_MATI (micromolar)"
    legend_constants[2] = "Ki_MATI in component V_MATI (micromolar)"
    legend_constants[3] = "V_MATIIImax in component V_MATIII (flux)"
    legend_algebraic[2] = "Km1_MATIII in component V_MATIII (micromolar)"
    legend_constants[4] = "Km2_MATIII in component V_MATIII (micromolar)"
    legend_constants[5] = "V_GNMTmax in component V_GNMT (flux)"
    legend_constants[6] = "Km_GNMT in component V_GNMT (micromolar)"
    legend_constants[7] = "Ki_GNMT in component V_GNMT (micromolar)"
    legend_constants[8] = "V_METHmax in component V_METH (flux)"
    legend_algebraic[5] = "Km1_METH in component V_METH (micromolar)"
    legend_constants[9] = "Km2_METH_A in component V_METH (dimensionless)"
    legend_constants[10] = "five_mTHF in component V_MS (micromolar)"
    legend_constants[11] = "V_MSmax in component V_MS (flux)"
    legend_constants[12] = "Kd_MS in component V_MS (micromolar)"
    legend_constants[13] = "Km_Hcy_MS in component V_MS (micromolar)"
    legend_constants[14] = "Km_five_mTHF_MS in component V_MS (micromolar)"
    legend_constants[15] = "alpha1 in component V_AH (first_order_rate_constant)"
    legend_constants[16] = "alpha2 in component V_AH (dimensionless)"
    legend_constants[17] = "beta1 in component V_CBS (second_order_rate_constant)"
    legend_constants[18] = "beta2 in component V_CBS (first_order_rate_constant)"
    legend_constants[19] = "V_BHMTmax in component V_BHMT (flux)"
    legend_constants[20] = "Km_BHMT in component V_BHMT (micromolar)"
    legend_rates[0] = "d/dt Met in component Met (micromolar)"
    legend_rates[1] = "d/dt AdoMet in component AdoMet (micromolar)"
    legend_rates[2] = "d/dt AdoHcy in component AdoHcy (micromolar)"
    legend_rates[3] = "d/dt Hcy in component Hcy (micromolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 53.5
    states[1] = 137.6
    states[2] = 13.2
    states[3] = 0.88
    constants[0] = 561
    constants[1] = 41
    constants[2] = 50
    constants[3] = 22870
    constants[4] = 21.1
    constants[5] = 10600
    constants[6] = 4500
    constants[7] = 20
    constants[8] = 4521
    constants[9] = 10
    constants[10] = 5.2
    constants[11] = 500
    constants[12] = 1
    constants[13] = 0.1
    constants[14] = 25
    constants[15] = 100
    constants[16] = 10
    constants[17] = 1.7
    constants[18] = 30
    constants[19] = 2500
    constants[20] = 12
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[1] = constants[0]/(1.00000+(constants[1]/states[0])*(1.00000+states[1]/constants[2]))
    algebraic[2] = 20000.0/(1.00000+5.70000*(power(states[1]/(states[1]+600.000), 2.00000)))
    algebraic[3] = constants[3]/(1.00000+(algebraic[2]*constants[4])/(power(states[0], 2.00000)+states[0]*constants[4]))
    algebraic[5] = 1.00000*(1.00000+states[2]/4.00000)
    algebraic[6] = constants[8]/(1.00000+algebraic[5]/states[1]+constants[9]+(constants[9]*algebraic[5])/states[1])
    algebraic[4] = ((constants[5]/(1.00000+power(constants[6]/states[1], 2.30000)))*1.00000)/(1.00000+states[2]/constants[7])
    rates[1] = (algebraic[1]+algebraic[3])-(algebraic[6]+algebraic[4])
    algebraic[8] = constants[15]*(states[2]-constants[16]*states[3])
    rates[2] = (algebraic[6]+algebraic[4])-algebraic[8]
    algebraic[0] = custom_piecewise([less(voi , 2.00000) | greater_equal(voi , 8.00000), 200.000 , greater_equal(voi , 2.00000) & less(voi , 5.00000), 300.000 , greater_equal(voi , 5.00000) & less(voi , 8.00000), 100.000 , True, 200.000])
    algebraic[7] = (constants[11]*constants[10]*states[3])/(constants[12]*constants[13]+constants[13]*constants[10]+constants[14]*states[3]+constants[10]*states[3])
    algebraic[10] = ((0.700000-0.0250000*((states[1]+states[2])-150.000))*constants[19]*states[3])/(constants[20]+states[3])
    rates[0] = (algebraic[7]+algebraic[10]+algebraic[0])-(algebraic[1]+algebraic[3])
    algebraic[9] = (constants[17]*(states[1]+states[2])-constants[18])*states[3]
    rates[3] = algebraic[8]-(algebraic[9]+algebraic[7]+algebraic[10])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[1] = constants[0]/(1.00000+(constants[1]/states[0])*(1.00000+states[1]/constants[2]))
    algebraic[2] = 20000.0/(1.00000+5.70000*(power(states[1]/(states[1]+600.000), 2.00000)))
    algebraic[3] = constants[3]/(1.00000+(algebraic[2]*constants[4])/(power(states[0], 2.00000)+states[0]*constants[4]))
    algebraic[5] = 1.00000*(1.00000+states[2]/4.00000)
    algebraic[6] = constants[8]/(1.00000+algebraic[5]/states[1]+constants[9]+(constants[9]*algebraic[5])/states[1])
    algebraic[4] = ((constants[5]/(1.00000+power(constants[6]/states[1], 2.30000)))*1.00000)/(1.00000+states[2]/constants[7])
    algebraic[8] = constants[15]*(states[2]-constants[16]*states[3])
    algebraic[0] = custom_piecewise([less(voi , 2.00000) | greater_equal(voi , 8.00000), 200.000 , greater_equal(voi , 2.00000) & less(voi , 5.00000), 300.000 , greater_equal(voi , 5.00000) & less(voi , 8.00000), 100.000 , True, 200.000])
    algebraic[7] = (constants[11]*constants[10]*states[3])/(constants[12]*constants[13]+constants[13]*constants[10]+constants[14]*states[3]+constants[10]*states[3])
    algebraic[10] = ((0.700000-0.0250000*((states[1]+states[2])-150.000))*constants[19]*states[3])/(constants[20]+states[3])
    algebraic[9] = (constants[17]*(states[1]+states[2])-constants[18])*states[3]
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