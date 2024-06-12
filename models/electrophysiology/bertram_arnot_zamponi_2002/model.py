# Size of variable arrays:
sizeAlgebraic = 21
sizeStates = 11
sizeConstants = 22
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (ms)"
    legend_constants[0] = "R_gas_const in component parameters (millijoule_per_mole_kelvin)"
    legend_constants[1] = "Temp in component parameters (kelvin)"
    legend_constants[2] = "F in component parameters (coulomb_per_mole)"
    legend_states[0] = "V in component membrane (mV)"
    legend_constants[3] = "Cm in component membrane (uF_per_cm2)"
    legend_algebraic[0] = "i_app in component stimulus_protocol (uA_per_cm2)"
    legend_algebraic[12] = "i_Na in component sodium_current (uA_per_cm2)"
    legend_algebraic[14] = "i_K in component potassium_current (uA_per_cm2)"
    legend_algebraic[16] = "i_leak in component leak_current (uA_per_cm2)"
    legend_constants[4] = "IstimStart in component stimulus_protocol (ms)"
    legend_constants[5] = "IstimEnd in component stimulus_protocol (ms)"
    legend_constants[6] = "IstimAmplitude in component stimulus_protocol (uA_per_cm2)"
    legend_constants[7] = "IstimPeriod in component stimulus_protocol (ms)"
    legend_constants[8] = "IstimPulseDuration in component stimulus_protocol (ms)"
    legend_algebraic[10] = "x_infinity in component sodium_current (dimensionless)"
    legend_algebraic[5] = "alpha_x in component sodium_current (dimensionless)"
    legend_algebraic[8] = "beta_x in component sodium_current (dimensionless)"
    legend_states[1] = "n in component potassium_current_n_gate (dimensionless)"
    legend_algebraic[1] = "alpha_n in component potassium_current_n_gate (per_ms)"
    legend_algebraic[6] = "beta_n in component potassium_current_n_gate (per_ms)"
    legend_algebraic[2] = "T in component transmitter_release (uM)"
    legend_constants[9] = "T_bar in component transmitter_release (uM)"
    legend_states[2] = "R in component transmitter_release (dimensionless)"
    legend_constants[10] = "kr_plus in component transmitter_release (per_uM_per_ms)"
    legend_constants[11] = "kr_minus in component transmitter_release (per_ms)"
    legend_algebraic[19] = "Ca in component calcium_concentration (uM)"
    legend_constants[12] = "Ca_ex in component calcium_concentration (uM)"
    legend_algebraic[9] = "Ca_open in component calcium_concentration (uM)"
    legend_constants[13] = "Dc in component calcium_concentration (um2_per_second)"
    legend_constants[14] = "r in component calcium_concentration (nm)"
    legend_algebraic[7] = "sigma in component calcium_concentration (uM_per_ms)"
    legend_algebraic[3] = "i_V in component calcium_concentration (uA)"
    legend_constants[15] = "g_Ca in component calcium_concentration (pS)"
    legend_constants[16] = "P in component calcium_concentration (mV_per_uM)"
    legend_algebraic[17] = "O in component O (dimensionless)"
    legend_algebraic[11] = "alpha in component rate_constants (per_ms)"
    legend_algebraic[13] = "alpha_ in component rate_constants (per_ms)"
    legend_algebraic[15] = "beta in component rate_constants (per_ms)"
    legend_algebraic[18] = "beta_ in component rate_constants (per_ms)"
    legend_algebraic[20] = "kG_plus in component rate_constants (per_ms)"
    legend_constants[17] = "kG_minus in component rate_constants (per_ms)"
    legend_constants[18] = "kG2_minus in component rate_constants (per_ms)"
    legend_constants[19] = "kG3_minus in component rate_constants (per_ms)"
    legend_states[3] = "a in component rate_constants (dimensionless)"
    legend_constants[20] = "ka_plus in component rate_constants (per_uM_per_ms)"
    legend_constants[21] = "ka_minus in component rate_constants (per_ms)"
    legend_states[4] = "C1 in component C1 (dimensionless)"
    legend_states[5] = "C2 in component C2 (dimensionless)"
    legend_states[6] = "C_G1 in component C_G1 (dimensionless)"
    legend_states[7] = "C3 in component C3 (dimensionless)"
    legend_states[8] = "C_G2 in component C_G2 (dimensionless)"
    legend_states[9] = "C4 in component C4 (dimensionless)"
    legend_states[10] = "C_G3 in component C_G3 (dimensionless)"
    legend_algebraic[4] = "C_G in component O (dimensionless)"
    legend_rates[0] = "d/dt V in component membrane (mV)"
    legend_rates[1] = "d/dt n in component potassium_current_n_gate (dimensionless)"
    legend_rates[2] = "d/dt R in component transmitter_release (dimensionless)"
    legend_rates[3] = "d/dt a in component rate_constants (dimensionless)"
    legend_rates[4] = "d/dt C1 in component C1 (dimensionless)"
    legend_rates[5] = "d/dt C2 in component C2 (dimensionless)"
    legend_rates[7] = "d/dt C3 in component C3 (dimensionless)"
    legend_rates[9] = "d/dt C4 in component C4 (dimensionless)"
    legend_rates[6] = "d/dt C_G1 in component C_G1 (dimensionless)"
    legend_rates[8] = "d/dt C_G2 in component C_G2 (dimensionless)"
    legend_rates[10] = "d/dt C_G3 in component C_G3 (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 8314.41
    constants[1] = 310
    constants[2] = 96485
    states[0] = -65
    constants[3] = 1
    constants[4] = 10
    constants[5] = 50000
    constants[6] = 40.0
    constants[7] = 100
    constants[8] = 1
    states[1] = 0
    constants[9] = 4000.0
    states[2] = 0
    constants[10] = 0.15
    constants[11] = 2.5
    constants[12] = 2000.0
    constants[13] = 220
    constants[14] = 10
    constants[15] = 1.2
    constants[16] = 0.006
    constants[17] = 0.00025
    constants[18] = 0.016
    constants[19] = 1.024
    states[3] = 0
    constants[20] = 200.0
    constants[21] = 0.0015
    states[4] = 1
    states[5] = 0
    states[6] = 0
    states[7] = 0
    states[8] = 0
    states[9] = 0
    states[10] = 0
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[2] = constants[9]*states[2]
    rates[3] = constants[20]*algebraic[2]*(1.00000-states[3])-constants[21]*states[3]
    algebraic[1] = (0.0200000*(states[0]+55.0000))/(1.00000-exp(-(states[0]+55.0000)/10.0000))
    algebraic[6] = 0.250000*exp(-(states[0]+65.0000)/80.0000)
    rates[1] = algebraic[1]*(1.00000-states[1])-algebraic[6]*states[1]
    algebraic[0] = custom_piecewise([greater_equal(voi , constants[4]) & less_equal(voi , constants[5]) & less_equal((voi-constants[4])-floor((voi-constants[4])/constants[7])*constants[7] , constants[8]), constants[6] , True, 0.00000])
    algebraic[5] = (0.200000*(states[0]+40.0000))/(1.00000-1.00000*exp(-(states[0]+40.0000)/10.0000))
    algebraic[8] = 8.00000*exp(1.00000/-(states[0]+65.0000/18.0000))
    algebraic[10] = algebraic[5]/(algebraic[5]+algebraic[8])
    algebraic[12] = 120.000*(power(algebraic[10], 3.00000))*(1.00000-states[1])*(states[0]-120.000)
    algebraic[14] = (36.0000*(power(states[1], 4.00000))*(states[0]+77.0000))/1.00000
    algebraic[16] = 0.300000*(states[0]+54.0000)
    rates[0] = -((algebraic[12]+algebraic[14]+algebraic[16])-algebraic[0])/constants[3]
    algebraic[17] = ((((((1.00000-states[4])-states[5])-states[7])-states[9])-states[6])-states[8])-states[10]
    algebraic[11] = 0.450000*exp(states[0]/22.0000)
    algebraic[15] = 0.0150000*exp(-states[0]/14.0000)
    rates[9] = (2.00000*algebraic[11]*states[7]+4.00000*algebraic[15]*algebraic[17])-states[9]*(3.00000*algebraic[15]+algebraic[11])
    algebraic[3] = (((constants[15]*constants[16]*2.00000*constants[2]*states[0])/(constants[0]*constants[1]))*constants[12])/(1.00000-exp((2.00000*constants[2]*states[0])/(constants[0]*constants[1])))
    algebraic[7] = -5.18200*algebraic[3]
    algebraic[9] = algebraic[7]/(2.00000*constants[13]*constants[14]* pi)
    algebraic[19] = algebraic[17]*algebraic[9]+0.100000
    rates[2] = constants[10]*algebraic[19]*(1.00000-states[2])-constants[11]*states[2]
    algebraic[20] = (3.00000*states[3])/(680.000+320.000*states[3])
    rates[4] = (algebraic[15]*states[5]+constants[17]*states[6])-states[4]*(4.00000*algebraic[11]+algebraic[20])
    rates[5] = (4.00000*algebraic[11]*states[4]+2.00000*algebraic[15]*states[7]+constants[18]*states[8])-states[5]*(algebraic[15]+3.00000*algebraic[11]+algebraic[20])
    rates[7] = (3.00000*algebraic[11]*states[5]+3.00000*algebraic[15]*states[9]+constants[19]*states[10])-states[7]*(2.00000*algebraic[15]+2.00000*algebraic[11]+algebraic[20])
    algebraic[13] = algebraic[11]/8.00000
    algebraic[18] = algebraic[15]*8.00000
    rates[6] = (algebraic[18]*states[8]+algebraic[20]*states[4])-states[6]*(4.00000*algebraic[13]+constants[17])
    rates[8] = (4.00000*algebraic[13]*states[6]+2.00000*algebraic[18]*states[10]+algebraic[20]*states[5])-states[8]*(algebraic[18]+3.00000*algebraic[13]+constants[18])
    rates[10] = (3.00000*algebraic[13]*states[8]+algebraic[20]*states[7])-states[10]*(2.00000*algebraic[18]+constants[19])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[2] = constants[9]*states[2]
    algebraic[1] = (0.0200000*(states[0]+55.0000))/(1.00000-exp(-(states[0]+55.0000)/10.0000))
    algebraic[6] = 0.250000*exp(-(states[0]+65.0000)/80.0000)
    algebraic[0] = custom_piecewise([greater_equal(voi , constants[4]) & less_equal(voi , constants[5]) & less_equal((voi-constants[4])-floor((voi-constants[4])/constants[7])*constants[7] , constants[8]), constants[6] , True, 0.00000])
    algebraic[5] = (0.200000*(states[0]+40.0000))/(1.00000-1.00000*exp(-(states[0]+40.0000)/10.0000))
    algebraic[8] = 8.00000*exp(1.00000/-(states[0]+65.0000/18.0000))
    algebraic[10] = algebraic[5]/(algebraic[5]+algebraic[8])
    algebraic[12] = 120.000*(power(algebraic[10], 3.00000))*(1.00000-states[1])*(states[0]-120.000)
    algebraic[14] = (36.0000*(power(states[1], 4.00000))*(states[0]+77.0000))/1.00000
    algebraic[16] = 0.300000*(states[0]+54.0000)
    algebraic[17] = ((((((1.00000-states[4])-states[5])-states[7])-states[9])-states[6])-states[8])-states[10]
    algebraic[11] = 0.450000*exp(states[0]/22.0000)
    algebraic[15] = 0.0150000*exp(-states[0]/14.0000)
    algebraic[3] = (((constants[15]*constants[16]*2.00000*constants[2]*states[0])/(constants[0]*constants[1]))*constants[12])/(1.00000-exp((2.00000*constants[2]*states[0])/(constants[0]*constants[1])))
    algebraic[7] = -5.18200*algebraic[3]
    algebraic[9] = algebraic[7]/(2.00000*constants[13]*constants[14]* pi)
    algebraic[19] = algebraic[17]*algebraic[9]+0.100000
    algebraic[20] = (3.00000*states[3])/(680.000+320.000*states[3])
    algebraic[13] = algebraic[11]/8.00000
    algebraic[18] = algebraic[15]*8.00000
    algebraic[4] = states[6]+states[8]+states[10]
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