# Size of variable arrays:
sizeAlgebraic = 13
sizeStates = 5
sizeConstants = 20
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
    legend_constants[1] = "V_K in component soma_compartment (mV)"
    legend_algebraic[8] = "I_Na in component soma_compartment (uA_per_cm2)"
    legend_algebraic[0] = "I_K_DR in component soma_compartment (uA_per_cm2)"
    legend_constants[2] = "g_K_DR in component soma_compartment (mS_per_cm2)"
    legend_constants[3] = "g_Na in component soma_compartment (mS_per_cm2)"
    legend_constants[4] = "g_c in component model_parameters (mS_per_cm2)"
    legend_states[1] = "V_D in component dendritic_compartment (mV)"
    legend_constants[5] = "C_m in component model_parameters (uF_per_cm2)"
    legend_constants[6] = "p in component model_parameters (dimensionless)"
    legend_states[2] = "n in component gating_variables (dimensionless)"
    legend_states[3] = "h in component gating_variables (dimensionless)"
    legend_algebraic[4] = "m_infinity in component gating_variables (dimensionless)"
    legend_constants[7] = "V_L in component dendritic_compartment (mV)"
    legend_constants[8] = "V_NMDA in component dendritic_compartment (mV)"
    legend_algebraic[1] = "I_L in component dendritic_compartment (uA_per_cm2)"
    legend_algebraic[12] = "I_D in component dendritic_compartment (uA_per_cm2)"
    legend_algebraic[9] = "I_pump in component dendritic_compartment (uA_per_cm2)"
    legend_algebraic[10] = "I_NMDA in component dendritic_compartment (uA_per_cm2)"
    legend_algebraic[11] = "I_Na_NMDA in component dendritic_compartment (uA_per_cm2)"
    legend_constants[9] = "R_pump in component dendritic_compartment (uA_per_cm2)"
    legend_constants[10] = "alpha in component dendritic_compartment (mMcm2_per_uAs)"
    legend_constants[11] = "g_NMDA in component dendritic_compartment (mS_per_cm2)"
    legend_constants[12] = "g_Na_NMDA in component dendritic_compartment (mS_per_cm2)"
    legend_constants[13] = "g_L in component dendritic_compartment (mS_per_cm2)"
    legend_states[4] = "Na in component dendritic_compartment (mM)"
    legend_constants[14] = "Na_eq in component dendritic_compartment (mM)"
    legend_constants[15] = "K_p in component dendritic_compartment (mM)"
    legend_constants[16] = "Mg_o in component dendritic_compartment (mM)"
    legend_constants[17] = "K_Mg in component dendritic_compartment (mM)"
    legend_constants[18] = "q in component dendritic_compartment (mV)"
    legend_algebraic[5] = "phi_Na in component dendritic_compartment (dimensionless)"
    legend_constants[19] = "phi_Na_eq in component dendritic_compartment (dimensionless)"
    legend_algebraic[2] = "n_infinity in component gating_variables (dimensionless)"
    legend_algebraic[3] = "h_infinity in component gating_variables (dimensionless)"
    legend_algebraic[6] = "tau_h in component gating_variables (second)"
    legend_algebraic[7] = "tau_n in component gating_variables (second)"
    legend_rates[0] = "d/dt V_s in component soma_compartment (mV)"
    legend_rates[1] = "d/dt V_D in component dendritic_compartment (mV)"
    legend_rates[4] = "d/dt Na in component dendritic_compartment (mM)"
    legend_rates[3] = "d/dt h in component gating_variables (dimensionless)"
    legend_rates[2] = "d/dt n in component gating_variables (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = -64
    constants[0] = 55
    constants[1] = -85
    constants[2] = 3.2
    constants[3] = 3.2
    constants[4] = 0.0
    states[1] = -25.0
    constants[5] = 1
    constants[6] = 0.5
    states[2] = 0.002
    states[3] = 1
    constants[7] = -50
    constants[8] = 0
    constants[9] = 18.0
    constants[10] = 0.5
    constants[11] = 1.25
    constants[12] = 1.0
    constants[13] = 0.18
    states[4] = 5.09
    constants[14] = 8
    constants[15] = 15
    constants[16] = 1.4
    constants[17] = 10.0
    constants[18] = 12.5
    constants[19] = (power(constants[14], 3.00000))/(power(constants[14], 3.00000)+power(constants[15], 3.00000))
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[3] = 1.00000/(1.00000+exp((states[0]+30.0000)/8.30000))
    algebraic[6] = 0.400000*(1.00000+2.00000/(1.00000+exp((states[0]+25.0000)/5.00000)))
    rates[3] = (algebraic[3]-states[3])/algebraic[6]
    algebraic[2] = 1.00000/(1.00000+exp(-(states[0]+31.0000)/5.30000))
    algebraic[7] = (0.800000+1.60000/(1.00000+exp(0.100000*(states[0]+25.0000))))/(1.00000+exp(-0.100000*(states[0]+70.0000)))
    rates[2] = (algebraic[2]-states[2])/algebraic[7]
    algebraic[4] = 1.00000/(1.00000+exp(-(states[0]+35.0000)/6.20000))
    algebraic[8] = constants[3]*states[3]*(states[0]-constants[0])*(power(algebraic[4], 3.00000))
    algebraic[0] = constants[2]*(states[0]-constants[1])*(power(states[2], 2.00000))
    rates[0] = -(1000.00*(algebraic[0]+algebraic[8]+(constants[4]/constants[6])*(states[0]-states[1])))/constants[5]
    algebraic[1] = constants[13]*(states[1]-constants[7])
    algebraic[5] = (power(states[4], 3.00000))/(power(states[4], 3.00000)+power(constants[15], 3.00000))
    algebraic[9] = constants[9]*(algebraic[5]-constants[19])
    algebraic[10] = (constants[11]/(1.00000+(constants[16]/constants[17])*exp(-(states[1]/constants[18]))))*(states[1]-constants[8])
    rates[1] = -(1000.00*(algebraic[10]+algebraic[9]+algebraic[1]+(constants[4]/(1.00000-constants[6]))*(states[1]-states[0])))/constants[5]
    algebraic[11] = (constants[12]/(1.00000+(constants[16]/constants[17])*exp(-(states[1]/constants[18]))))*(states[1]-constants[0])
    rates[4] = constants[10]*(-algebraic[11]-3.00000*algebraic[9])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[3] = 1.00000/(1.00000+exp((states[0]+30.0000)/8.30000))
    algebraic[6] = 0.400000*(1.00000+2.00000/(1.00000+exp((states[0]+25.0000)/5.00000)))
    algebraic[2] = 1.00000/(1.00000+exp(-(states[0]+31.0000)/5.30000))
    algebraic[7] = (0.800000+1.60000/(1.00000+exp(0.100000*(states[0]+25.0000))))/(1.00000+exp(-0.100000*(states[0]+70.0000)))
    algebraic[4] = 1.00000/(1.00000+exp(-(states[0]+35.0000)/6.20000))
    algebraic[8] = constants[3]*states[3]*(states[0]-constants[0])*(power(algebraic[4], 3.00000))
    algebraic[0] = constants[2]*(states[0]-constants[1])*(power(states[2], 2.00000))
    algebraic[1] = constants[13]*(states[1]-constants[7])
    algebraic[5] = (power(states[4], 3.00000))/(power(states[4], 3.00000)+power(constants[15], 3.00000))
    algebraic[9] = constants[9]*(algebraic[5]-constants[19])
    algebraic[10] = (constants[11]/(1.00000+(constants[16]/constants[17])*exp(-(states[1]/constants[18]))))*(states[1]-constants[8])
    algebraic[11] = (constants[12]/(1.00000+(constants[16]/constants[17])*exp(-(states[1]/constants[18]))))*(states[1]-constants[0])
    algebraic[12] = algebraic[10]+algebraic[9]+algebraic[1]
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