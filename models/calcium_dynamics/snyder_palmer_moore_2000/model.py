# Size of variable arrays:
sizeAlgebraic = 11
sizeStates = 10
sizeConstants = 27
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_algebraic[2] = "Rate_Ca_influx_across_the_SR in component Ca_influx_across_the_SR (flux)"
    legend_constants[0] = "Ca_e in component extracellular_calcium (molar)"
    legend_states[0] = "Ca_f in component fuzzy_space_calcium (molar)"
    legend_constants[1] = "k1 in component Ca_influx_across_the_SR (first_order_rate_constant)"
    legend_states[1] = "Ca_2_S1 in component Ca_bound_to_the_SRRC_fast_activating_binding_site (molar)"
    legend_states[2] = "Ca_S2 in component Ca_bound_to_the_SRRC_slow_inactivating_binding_site (molar)"
    legend_states[3] = "S1 in component SRRC_fast_activating_binding_site (molar)"
    legend_states[4] = "S2 in component SRRC_slow_inactivating_binding_site (molar)"
    legend_states[5] = "Ca_s in component SR_calcium (molar)"
    legend_algebraic[0] = "dCa2_S1_dt in component Ca_movement_through_the_SRRC (flux)"
    legend_algebraic[1] = "dCa_S2_dt in component Ca_movement_through_the_SRRC (flux)"
    legend_constants[2] = "k_on1 in component Ca_movement_through_the_SRRC (second_order_rate_constant)"
    legend_constants[3] = "k_off1 in component Ca_movement_through_the_SRRC (first_order_rate_constant)"
    legend_constants[4] = "k_on2 in component Ca_movement_through_the_SRRC (second_order_rate_constant)"
    legend_constants[5] = "k_off2 in component Ca_movement_through_the_SRRC (first_order_rate_constant)"
    legend_constants[6] = "k_s in component Ca_movement_through_the_SRRC (first_order_rate_constant)"
    legend_states[6] = "cas1 in component Ca_movement_through_the_SRRC (dimensionless)"
    legend_states[7] = "cas2 in component Ca_movement_through_the_SRRC (dimensionless)"
    legend_algebraic[4] = "dcas1_dt in component Ca_movement_through_the_SRRC (first_order_rate_constant)"
    legend_algebraic[5] = "dcas2_dt in component Ca_movement_through_the_SRRC (first_order_rate_constant)"
    legend_algebraic[3] = "r_o in component Ca_movement_through_the_SRRC (dimensionless)"
    legend_algebraic[6] = "Rate_Ca_movement_through_the_SRRC in component Ca_movement_through_the_SRRC (flux)"
    legend_constants[7] = "Km_NaCaX in component Ca_efflux_across_the_SR_by_NaCa_exchange (molar)"
    legend_constants[8] = "Vmax_NaCaX in component Ca_efflux_across_the_SR_by_NaCa_exchange (flux)"
    legend_algebraic[7] = "Rate_Ca_efflux_across_the_SR_by_NaCa_exchange in component Ca_efflux_across_the_SR_by_NaCa_exchange (flux)"
    legend_states[8] = "Ca_c in component cytosolic_calcium (molar)"
    legend_constants[9] = "kf in component Ca_movement_between_the_fuzzy_space_and_cytosol (first_order_rate_constant)"
    legend_algebraic[8] = "Rate_Ca_movement_between_the_fuzzy_space_and_cytosol in component Ca_movement_between_the_fuzzy_space_and_cytosol (flux)"
    legend_constants[10] = "Km_s in component Ca_uptake_by_SR_Ca_ATPase (molar)"
    legend_constants[11] = "Vmax_s in component Ca_uptake_by_SR_Ca_ATPase (flux)"
    legend_algebraic[9] = "Rate_Ca_uptake_by_SR_Ca_ATPase in component Ca_uptake_by_SR_Ca_ATPase (flux)"
    legend_states[9] = "Ca_CSQ in component calsequestrin_bound_calcium (molar)"
    legend_constants[12] = "K_ons in component Ca_buffering_in_the_SR (second_order_rate_constant)"
    legend_constants[13] = "K_offs in component Ca_buffering_in_the_SR (first_order_rate_constant)"
    legend_constants[14] = "Bmax_s in component Ca_buffering_in_the_SR (molar)"
    legend_algebraic[10] = "Rate_Ca_buffering_in_the_SR in component Ca_buffering_in_the_SR (flux)"
    legend_constants[15] = "Rt in component fuzzy_space_calcium (molar)"
    legend_constants[16] = "Bmax_f1 in component fuzzy_space_calcium (molar)"
    legend_constants[17] = "Bmax_f2 in component fuzzy_space_calcium (molar)"
    legend_constants[18] = "Kb_f1 in component fuzzy_space_calcium (molar)"
    legend_constants[19] = "Kb_f2 in component fuzzy_space_calcium (molar)"
    legend_constants[20] = "V_f in component fuzzy_space_calcium (dimensionless)"
    legend_constants[21] = "Bmax_c in component cytosolic_calcium (molar)"
    legend_constants[22] = "dye_c in component cytosolic_calcium (molar)"
    legend_constants[23] = "Kb_c in component cytosolic_calcium (molar)"
    legend_constants[24] = "Kb_dye in component cytosolic_calcium (molar)"
    legend_constants[25] = "V_c in component cytosolic_calcium (dimensionless)"
    legend_constants[26] = "V_s in component SR_calcium (dimensionless)"
    legend_rates[6] = "d/dt cas1 in component Ca_movement_through_the_SRRC (dimensionless)"
    legend_rates[7] = "d/dt cas2 in component Ca_movement_through_the_SRRC (dimensionless)"
    legend_rates[0] = "d/dt Ca_f in component fuzzy_space_calcium (molar)"
    legend_rates[8] = "d/dt Ca_c in component cytosolic_calcium (molar)"
    legend_rates[5] = "d/dt Ca_s in component SR_calcium (molar)"
    legend_rates[9] = "d/dt Ca_CSQ in component calsequestrin_bound_calcium (molar)"
    legend_rates[3] = "d/dt S1 in component SRRC_fast_activating_binding_site (molar)"
    legend_rates[4] = "d/dt S2 in component SRRC_slow_inactivating_binding_site (molar)"
    legend_rates[1] = "d/dt Ca_2_S1 in component Ca_bound_to_the_SRRC_fast_activating_binding_site (molar)"
    legend_rates[2] = "d/dt Ca_S2 in component Ca_bound_to_the_SRRC_slow_inactivating_binding_site (molar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 0.002
    states[0] = 0.12e-6
    constants[1] = 0.2
    states[1] = 0
    states[2] = 0
    states[3] = 0
    states[4] = 0
    states[5] = 201e-6
    constants[2] = 2000000000
    constants[3] = 1400
    constants[4] = 13000000
    constants[5] = 3.9
    constants[6] = 9
    states[6] = 0
    states[7] = 0
    constants[7] = 0.000036
    constants[8] = 0.0012
    states[8] = 1e-7
    constants[9] = 2500
    constants[10] = 0.00000025
    constants[11] = 0.000525
    states[9] = 0
    constants[12] = 8772
    constants[13] = 5.596536
    constants[14] = 0.008
    constants[15] = 0.00000015
    constants[16] = 0.0002
    constants[17] = 0.0011
    constants[18] = 0.000017
    constants[19] = 0.000013
    constants[20] = 0.0013
    constants[21] = 0.00012
    constants[22] = 0
    constants[23] = 0.00000096
    constants[24] = 2e-7
    constants[25] = 0.9287
    constants[26] = 0.07
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[0] = constants[2]*states[0]*states[3]-((power(constants[3], 2.00000))/(constants[2]*states[0]))*states[1]
    rates[3] = -algebraic[0]
    algebraic[1] = constants[4]*states[0]*states[4]-constants[5]*states[2]
    rates[4] = -algebraic[1]
    rates[1] = algebraic[0]
    rates[2] = algebraic[1]
    algebraic[4] = constants[2]*states[0]*(1.00000-states[6])-((power(constants[3], 2.00000))/(constants[2]*states[0]))*states[6]
    rates[6] = algebraic[4]
    algebraic[5] = constants[4]*states[0]*(1.00000-states[7])-constants[5]*states[7]
    rates[7] = algebraic[5]
    algebraic[2] = constants[1]*(constants[0]-states[0])
    algebraic[3] = states[6]*(1.00000-states[7])
    algebraic[6] = constants[6]*algebraic[3]*(states[5]-states[0])
    algebraic[7] = (constants[8]*states[0])/(constants[7]+states[0])
    algebraic[8] = constants[9]*(states[0]-states[8])
    rates[0] = ((algebraic[6]-(constants[15]*(algebraic[4]+algebraic[5])+algebraic[8]+algebraic[7]))+algebraic[2])/((constants[16]*constants[18])/(power(states[0]+constants[18], 2.00000))+(constants[17]*constants[19])/(power(states[0]+constants[19], 2.00000))+constants[20])
    algebraic[9] = (constants[11]*(power(states[8], 2.00000)-(power(states[5], 2.00000))/(power(7000.00, 2.00000))))/(power(constants[10], 2.00000)+power(states[8], 2.00000)+(power(states[5], 2.00000))/(power(7000.00, 2.00000)))
    rates[8] = (algebraic[8]-algebraic[9])/((constants[21]*constants[23])/(power(states[8]+constants[23], 2.00000))+(constants[22]*constants[24])/(power(states[8]+constants[24], 2.00000))+constants[25])
    algebraic[10] = constants[12]*states[5]*(constants[14]-states[9])-constants[13]*states[9]
    rates[5] = (algebraic[9]-algebraic[6])/constants[26]-algebraic[10]
    rates[9] = algebraic[10]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = constants[2]*states[0]*states[3]-((power(constants[3], 2.00000))/(constants[2]*states[0]))*states[1]
    algebraic[1] = constants[4]*states[0]*states[4]-constants[5]*states[2]
    algebraic[4] = constants[2]*states[0]*(1.00000-states[6])-((power(constants[3], 2.00000))/(constants[2]*states[0]))*states[6]
    algebraic[5] = constants[4]*states[0]*(1.00000-states[7])-constants[5]*states[7]
    algebraic[2] = constants[1]*(constants[0]-states[0])
    algebraic[3] = states[6]*(1.00000-states[7])
    algebraic[6] = constants[6]*algebraic[3]*(states[5]-states[0])
    algebraic[7] = (constants[8]*states[0])/(constants[7]+states[0])
    algebraic[8] = constants[9]*(states[0]-states[8])
    algebraic[9] = (constants[11]*(power(states[8], 2.00000)-(power(states[5], 2.00000))/(power(7000.00, 2.00000))))/(power(constants[10], 2.00000)+power(states[8], 2.00000)+(power(states[5], 2.00000))/(power(7000.00, 2.00000)))
    algebraic[10] = constants[12]*states[5]*(constants[14]-states[9])-constants[13]*states[9]
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