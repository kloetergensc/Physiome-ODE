# Size of variable arrays:
sizeAlgebraic = 26
sizeStates = 12
sizeConstants = 37
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_states[0] = "V_s in component soma_compartment (mV)"
    legend_constants[0] = "V_Na in component soma_compartment (mV)"
    legend_algebraic[25] = "I_Na_s in component soma_compartment (uA_per_cm2)"
    legend_algebraic[0] = "I_K_DR_s in component soma_compartment (uA_per_cm2)"
    legend_algebraic[10] = "I_Ca_T in component soma_compartment (uA_per_cm2)"
    legend_algebraic[15] = "I_K_Ca in component soma_compartment (uA_per_cm2)"
    legend_algebraic[17] = "I_A in component soma_compartment (uA_per_cm2)"
    legend_algebraic[20] = "I_h in component soma_compartment (uA_per_cm2)"
    legend_constants[1] = "g_Na_s in component soma_compartment (mS_per_cm2)"
    legend_constants[2] = "g_K_DR_s in component soma_compartment (mS_per_cm2)"
    legend_constants[3] = "g_Ca_T in component soma_compartment (mS_per_cm2)"
    legend_constants[4] = "g_K_Ca in component soma_compartment (mS_per_cm2)"
    legend_constants[5] = "g_A in component soma_compartment (mS_per_cm2)"
    legend_constants[6] = "g_h in component soma_compartment (mS_per_cm2)"
    legend_states[1] = "Ca in component soma_compartment (mM)"
    legend_constants[7] = "k_Ca in component soma_compartment (per_second)"
    legend_constants[8] = "K_Ca in component soma_compartment (mM)"
    legend_constants[9] = "V_h in component soma_compartment (mV)"
    legend_constants[10] = "beta in component soma_compartment (mMcm2_per_uAs)"
    legend_constants[11] = "I_APP in component soma_compartment (uA_per_cm2)"
    legend_constants[12] = "V_Ca in component general_variables (mV)"
    legend_constants[13] = "V_K in component general_variables (mV)"
    legend_states[2] = "n in component gating_variables (dimensionless)"
    legend_states[3] = "h in component gating_variables (dimensionless)"
    legend_algebraic[22] = "m_infinity in component gating_variables (dimensionless)"
    legend_constants[14] = "g_c in component general_variables (mS_per_cm2)"
    legend_constants[15] = "p in component general_variables (dimensionless)"
    legend_states[4] = "V_D in component dendritic_compartment (mV)"
    legend_constants[16] = "C_m in component general_variables (uF_per_cm2)"
    legend_states[5] = "m_T in component gating_variables (dimensionless)"
    legend_states[6] = "h_T in component gating_variables (dimensionless)"
    legend_states[7] = "a in component gating_variables (dimensionless)"
    legend_states[8] = "b in component gating_variables (dimensionless)"
    legend_states[9] = "m_h in component gating_variables (dimensionless)"
    legend_states[10] = "Na in component dendritic_compartment (mM)"
    legend_constants[17] = "K_p in component dendritic_compartment (mM)"
    legend_algebraic[18] = "I_L in component dendritic_compartment (uA_per_cm2)"
    legend_algebraic[16] = "I_pump in component dendritic_compartment (uA_per_cm2)"
    legend_constants[18] = "R_pump in component dendritic_compartment (uA_per_cm2)"
    legend_constants[19] = "Na_eq in component dendritic_compartment (mM)"
    legend_algebraic[11] = "phi_Na in component dendritic_compartment (dimensionless)"
    legend_constants[31] = "phi_Na_eq in component dendritic_compartment (dimensionless)"
    legend_constants[20] = "alpha in component dendritic_compartment (mMcm2_per_uAs)"
    legend_algebraic[21] = "I_NMDA in component dendritic_compartment (uA_per_cm2)"
    legend_algebraic[19] = "I_Na_NMDA in component dendritic_compartment (uA_per_cm2)"
    legend_constants[21] = "g_NMDA in component dendritic_compartment (mS_per_cm2)"
    legend_constants[22] = "g_Na_NMDA in component dendritic_compartment (mS_per_cm2)"
    legend_constants[23] = "g_L in component dendritic_compartment (mS_per_cm2)"
    legend_constants[24] = "Mg_o in component dendritic_compartment (mM)"
    legend_constants[25] = "K_Mg in component dendritic_compartment (mM)"
    legend_constants[26] = "q in component dendritic_compartment (mV)"
    legend_constants[27] = "V_NMDA in component dendritic_compartment (mV)"
    legend_constants[28] = "V_L in component dendritic_compartment (mV)"
    legend_algebraic[23] = "I_D in component dendritic_compartment (uA_per_cm2)"
    legend_algebraic[24] = "I_Ca_L in component dendritic_compartment (uA_per_cm2)"
    legend_constants[29] = "g_Ca_L in component dendritic_compartment (mS_per_cm2)"
    legend_constants[30] = "g_K_DR_D in component dendritic_compartment (mS_per_cm2)"
    legend_algebraic[1] = "I_K_DR_D in component dendritic_compartment (uA_per_cm2)"
    legend_states[11] = "m_L in component gating_variables (dimensionless)"
    legend_algebraic[2] = "m_T_infinity in component gating_variables (dimensionless)"
    legend_algebraic[3] = "h_T_infinity in component gating_variables (dimensionless)"
    legend_algebraic[4] = "a_infinity in component gating_variables (dimensionless)"
    legend_algebraic[5] = "b_infinity in component gating_variables (dimensionless)"
    legend_algebraic[6] = "m_h_infinity in component gating_variables (dimensionless)"
    legend_algebraic[7] = "m_L_infinity in component gating_variables (dimensionless)"
    legend_algebraic[8] = "n_infinity in component gating_variables (dimensionless)"
    legend_algebraic[9] = "h_infinity in component gating_variables (dimensionless)"
    legend_algebraic[12] = "tau_h in component gating_variables (second)"
    legend_algebraic[13] = "tau_n in component gating_variables (second)"
    legend_constants[32] = "tau_m_T in component gating_variables (second)"
    legend_constants[33] = "tau_h_T in component gating_variables (second)"
    legend_algebraic[14] = "tau_m_L in component gating_variables (second)"
    legend_constants[34] = "tau_a in component gating_variables (second)"
    legend_constants[35] = "tau_b in component gating_variables (second)"
    legend_constants[36] = "tau_m_h in component gating_variables (second)"
    legend_rates[0] = "d/dt V_s in component soma_compartment (mV)"
    legend_rates[1] = "d/dt Ca in component soma_compartment (mM)"
    legend_rates[4] = "d/dt V_D in component dendritic_compartment (mV)"
    legend_rates[10] = "d/dt Na in component dendritic_compartment (mM)"
    legend_rates[3] = "d/dt h in component gating_variables (dimensionless)"
    legend_rates[2] = "d/dt n in component gating_variables (dimensionless)"
    legend_rates[5] = "d/dt m_T in component gating_variables (dimensionless)"
    legend_rates[11] = "d/dt m_L in component gating_variables (dimensionless)"
    legend_rates[9] = "d/dt m_h in component gating_variables (dimensionless)"
    legend_rates[7] = "d/dt a in component gating_variables (dimensionless)"
    legend_rates[8] = "d/dt b in component gating_variables (dimensionless)"
    legend_rates[6] = "d/dt h_T in component gating_variables (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = -64.0
    constants[0] = 55
    constants[1] = 3.2
    constants[2] = 6.4
    constants[3] = 1.5
    constants[4] = 1.2
    constants[5] = 2
    constants[6] = 0.1
    states[1] = 0
    constants[7] = 1
    constants[8] = 0.0004
    constants[9] = -30
    constants[10] = 0.104
    constants[11] = -6.7
    constants[12] = 120
    constants[13] = -85
    states[2] = 0.002
    states[3] = 1.0
    constants[14] = 0.1
    constants[15] = 0.5
    states[4] = -77.0
    constants[16] = 1
    states[5] = 0.1
    states[6] = 0.1
    states[7] = 0.1
    states[8] = 0.1
    states[9] = 0.1
    states[10] = 5.09
    constants[17] = 15
    constants[18] = 18
    constants[19] = 8
    constants[20] = 0.173
    constants[21] = 25
    constants[22] = 5
    constants[23] = 0.18
    constants[24] = 1.4
    constants[25] = 10
    constants[26] = 12.5
    constants[27] = 0
    constants[28] = -50
    constants[29] = 0.19
    constants[30] = 0.14
    states[11] = 0.1
    constants[31] = (power(constants[19], 3.00000))/(power(constants[19], 3.00000)+power(constants[17], 3.00000))
    constants[32] = 1.00000
    constants[33] = 10.0000
    constants[34] = 0.500000
    constants[35] = 10.0000
    constants[36] = 190.000
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[2] = 1.00000/(1.00000+exp(-(states[0]+55.0000)/7.00000))
    rates[5] = (algebraic[2]-states[5])/constants[32]
    algebraic[6] = 1.00000/(1.00000+exp((states[0]+80.0000)/8.00000))
    rates[9] = (algebraic[6]-states[9])/constants[36]
    algebraic[4] = 1.00000/(1.00000+exp(-(states[0]+60.0000)/10.0000))
    rates[7] = (algebraic[4]-states[7])/constants[34]
    algebraic[5] = 1.00000/(1.00000+exp((states[0]+70.0000)/5.70000))
    rates[8] = (algebraic[5]-states[8])/constants[35]
    algebraic[3] = 1.00000/(1.00000+exp((states[0]+81.0000)/11.0000))
    rates[6] = (algebraic[3]-states[6])/constants[33]
    algebraic[10] = constants[3]*states[6]*(states[0]-constants[12])*(power(states[5], 2.00000))
    rates[1] = -(constants[10]*algebraic[10]+constants[7]*states[1])
    algebraic[9] = 1.00000/(1.00000+exp((states[0]+30.0000)/8.30000))
    algebraic[12] = 0.400000*(1.00000+2.00000/(1.00000+exp((states[0]+25.0000)/5.00000)))
    rates[3] = (algebraic[9]-states[3])/algebraic[12]
    algebraic[8] = 1.00000/(1.00000+exp(-(states[0]+31.0000)/5.30000))
    algebraic[13] = (0.800000*(1.00000+2.00000/(1.00000+exp((states[0]+25.0000)/10.0000))))/(1.00000+exp(-(states[0]+70.0000)/10.0000))
    rates[2] = (algebraic[8]-states[2])/algebraic[13]
    algebraic[7] = 1.00000/(1.00000+exp(-(states[4]+20.0000)/5.30000))
    algebraic[14] = 0.400000/(5.00000*exp(-(states[4]+11.0000)/8.30000)+(-(states[4]+11.0000)/8.30000)/(exp(-(states[4]+11.0000)/8.30000)-1.00000))
    rates[11] = (algebraic[7]-states[11])/algebraic[14]
    algebraic[11] = (power(states[10], 3.00000))/(power(states[10], 3.00000)+power(constants[17], 3.00000))
    algebraic[16] = constants[18]*(algebraic[11]-constants[31])
    algebraic[19] = (constants[22]/(1.00000+(constants[24]/constants[25])*exp(-states[4]/constants[26])))*(states[4]-constants[0])
    rates[10] = constants[20]*(-algebraic[19]-algebraic[16]*3.00000)
    algebraic[18] = constants[23]*(states[4]-constants[28])
    algebraic[21] = (constants[21]/(1.00000+(constants[24]/constants[25])*exp(-states[4]/constants[26])))*(states[4]-constants[27])
    algebraic[24] = constants[29]*(states[4]-constants[12])*(power(states[11], 2.00000))
    algebraic[1] = constants[30]*(states[4]-constants[13])*(power(states[2], 2.00000))
    rates[4] = -(1000.00*(algebraic[24]+algebraic[1]+algebraic[21]+algebraic[16]+algebraic[18]+(constants[14]/(1.00000-constants[15]))*(states[4]-states[0])))/constants[16]
    algebraic[22] = 1.00000/(1.00000+exp(-(states[0]+35.0000)/6.20000))
    algebraic[25] = constants[1]*states[3]*(states[0]-constants[0])*(power(algebraic[22], 3.00000))
    algebraic[0] = constants[2]*(states[0]-constants[13])*(power(states[2], 2.00000))
    algebraic[15] = ((constants[4]*(power(states[1], 4.00000)))/(power(states[1], 4.00000)+power(constants[8], 4.00000)))*(states[0]-constants[13])
    algebraic[17] = constants[5]*states[8]*(states[0]-constants[13])*(power(states[7], 4.00000))
    algebraic[20] = constants[6]*states[9]*(states[0]-constants[9])
    rates[0] = (1000.00*(constants[11]-(algebraic[25]+algebraic[10]+algebraic[0]+algebraic[15]+algebraic[17]+algebraic[20]+(constants[14]/constants[15])*(states[0]-states[4]))))/constants[16]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[2] = 1.00000/(1.00000+exp(-(states[0]+55.0000)/7.00000))
    algebraic[6] = 1.00000/(1.00000+exp((states[0]+80.0000)/8.00000))
    algebraic[4] = 1.00000/(1.00000+exp(-(states[0]+60.0000)/10.0000))
    algebraic[5] = 1.00000/(1.00000+exp((states[0]+70.0000)/5.70000))
    algebraic[3] = 1.00000/(1.00000+exp((states[0]+81.0000)/11.0000))
    algebraic[10] = constants[3]*states[6]*(states[0]-constants[12])*(power(states[5], 2.00000))
    algebraic[9] = 1.00000/(1.00000+exp((states[0]+30.0000)/8.30000))
    algebraic[12] = 0.400000*(1.00000+2.00000/(1.00000+exp((states[0]+25.0000)/5.00000)))
    algebraic[8] = 1.00000/(1.00000+exp(-(states[0]+31.0000)/5.30000))
    algebraic[13] = (0.800000*(1.00000+2.00000/(1.00000+exp((states[0]+25.0000)/10.0000))))/(1.00000+exp(-(states[0]+70.0000)/10.0000))
    algebraic[7] = 1.00000/(1.00000+exp(-(states[4]+20.0000)/5.30000))
    algebraic[14] = 0.400000/(5.00000*exp(-(states[4]+11.0000)/8.30000)+(-(states[4]+11.0000)/8.30000)/(exp(-(states[4]+11.0000)/8.30000)-1.00000))
    algebraic[11] = (power(states[10], 3.00000))/(power(states[10], 3.00000)+power(constants[17], 3.00000))
    algebraic[16] = constants[18]*(algebraic[11]-constants[31])
    algebraic[19] = (constants[22]/(1.00000+(constants[24]/constants[25])*exp(-states[4]/constants[26])))*(states[4]-constants[0])
    algebraic[18] = constants[23]*(states[4]-constants[28])
    algebraic[21] = (constants[21]/(1.00000+(constants[24]/constants[25])*exp(-states[4]/constants[26])))*(states[4]-constants[27])
    algebraic[24] = constants[29]*(states[4]-constants[12])*(power(states[11], 2.00000))
    algebraic[1] = constants[30]*(states[4]-constants[13])*(power(states[2], 2.00000))
    algebraic[22] = 1.00000/(1.00000+exp(-(states[0]+35.0000)/6.20000))
    algebraic[25] = constants[1]*states[3]*(states[0]-constants[0])*(power(algebraic[22], 3.00000))
    algebraic[0] = constants[2]*(states[0]-constants[13])*(power(states[2], 2.00000))
    algebraic[15] = ((constants[4]*(power(states[1], 4.00000)))/(power(states[1], 4.00000)+power(constants[8], 4.00000)))*(states[0]-constants[13])
    algebraic[17] = constants[5]*states[8]*(states[0]-constants[13])*(power(states[7], 4.00000))
    algebraic[20] = constants[6]*states[9]*(states[0]-constants[9])
    algebraic[23] = algebraic[21]+algebraic[16]+algebraic[18]
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