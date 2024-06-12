# Size of variable arrays:
sizeAlgebraic = 7
sizeStates = 11
sizeConstants = 57
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (hour)"
    legend_states[0] = "Q_Ca_p in component Q_Ca_p (millimole)"
    legend_constants[0] = "J_bp_Ca in component model_fluxes (millimole_per_hour)"
    legend_algebraic[0] = "J_pb_Ca in component model_fluxes (millimole_per_hour)"
    legend_constants[1] = "J_pu_Ca in component model_fluxes (millimole_per_hour)"
    legend_algebraic[3] = "J_ip_Ca in component model_fluxes (millimole_per_hour)"
    legend_states[1] = "Q_P_p in component Q_P_p (millimole)"
    legend_constants[38] = "J_bp_P in component model_fluxes (millimole_per_hour)"
    legend_algebraic[4] = "J_pb_P in component model_fluxes (millimole_per_hour)"
    legend_constants[2] = "J_pu_P in component model_fluxes (millimole_per_hour)"
    legend_constants[48] = "J_ip_P in component model_fluxes (millimole_per_hour)"
    legend_constants[55] = "J_pc_P in component model_fluxes (millimole_per_hour)"
    legend_algebraic[6] = "J_cp_P in component model_fluxes (millimole_per_hour)"
    legend_states[2] = "Q_P_c in component Q_P_c (millimole)"
    legend_states[3] = "Q_PTH_p in component Q_PTH_p (millimole)"
    legend_algebraic[1] = "S_PTH in component model_fluxes (millimole_per_hour)"
    legend_constants[3] = "C_PTH_p in component model_parameters (millimolar)"
    legend_constants[4] = "k_PTH in component model_parameters (first_order_rate_constant)"
    legend_states[4] = "Q_Ca_b in component Q_Ca_b (millimole)"
    legend_states[5] = "Q_P_b in component Q_P_b (millimole)"
    legend_states[6] = "Q_E_k in component Q_E_k (millimole)"
    legend_constants[5] = "S_E in component model_parameters (millimole_per_hour)"
    legend_constants[6] = "k_E in component model_parameters (first_order_rate_constant)"
    legend_states[7] = "Q_D_p in component Q_D_p (millimole)"
    legend_constants[7] = "k_D in component model_parameters (first_order_rate_constant)"
    legend_states[8] = "Q_TCa_i in component Q_TCa_i (millimole)"
    legend_constants[49] = "k1_D in component model_parameters (first_order_rate_constant)"
    legend_constants[50] = "k2_D in component model_parameters (first_order_rate_constant)"
    legend_states[9] = "Q_C_PT in component Q_C_PT (millimole)"
    legend_constants[51] = "k3_D in component model_parameters (first_order_rate_constant)"
    legend_constants[52] = "k4_D in component model_parameters (first_order_rate_constant)"
    legend_states[10] = "Q_TP_k in component Q_TP_k (millimole)"
    legend_constants[53] = "k1_P in component model_parameters (first_order_rate_constant)"
    legend_constants[8] = "k2_P in component model_parameters (first_order_rate_constant)"
    legend_constants[9] = "J_P_ing in component model_fluxes (millimole_per_hour)"
    legend_constants[10] = "Stoic_Ca_P in component model_fluxes (dimensionless)"
    legend_constants[11] = "k_Ca_i in component model_fluxes (first_order_rate_constant)"
    legend_constants[12] = "C_Ca_i in component model_parameters (millimolar)"
    legend_constants[13] = "C_Ca_p in component model_parameters (millimolar)"
    legend_constants[14] = "C_P_p in component model_parameters (millimolar)"
    legend_constants[15] = "k_Ca_b in component model_parameters (first_order_rate_constant)"
    legend_constants[16] = "k3_P in component model_parameters (first_order_rate_constant)"
    legend_constants[17] = "k4_P in component model_parameters (first_order_rate_constant)"
    legend_constants[39] = "Y_Ca_i_2plus in component model_parameters (first_order_rate_constant)"
    legend_constants[40] = "Y_Ca_p_1minus in component model_parameters (first_order_rate_constant)"
    legend_constants[18] = "C_D_p in component model_parameters (picomolar)"
    legend_constants[54] = "Ca_thr in component model_parameters (first_order_rate_constant)"
    legend_constants[56] = "Ca_T in component model_parameters (second_order_rate_constant)"
    legend_algebraic[2] = "P_thr in component model_parameters (millimole_per_hour)"
    legend_algebraic[5] = "P_T in component model_parameters (litre_per_hour)"
    legend_constants[41] = "Y_i_D_p_1plus in component model_parameters (first_order_rate_constant)"
    legend_constants[42] = "Y_PT_D_p_1plus in component model_parameters (first_order_rate_constant)"
    legend_constants[43] = "Y_k_Jext_1plus in component model_parameters (first_order_rate_constant)"
    legend_constants[44] = "Y_PTH_p_2plus in component model_parameters (first_order_rate_constant)"
    legend_constants[45] = "Y_PTH_p_2minus in component model_parameters (first_order_rate_constant)"
    legend_constants[46] = "Y_i_D_p_1minus in component model_parameters (first_order_rate_constant)"
    legend_constants[47] = "Y_PT_D_p_1minus in component model_parameters (first_order_rate_constant)"
    legend_constants[19] = "Y_Ca_i_2plus_Max in component model_parameters (first_order_rate_constant)"
    legend_constants[20] = "Y_PTH_p_2plus_Max in component model_parameters (first_order_rate_constant)"
    legend_constants[21] = "Y_PTH_p_2minus_Max in component model_parameters (first_order_rate_constant)"
    legend_constants[22] = "Y_Ca_p_1plus_minus_Max in component model_parameters (first_order_rate_constant)"
    legend_constants[23] = "Y_i_D_p_1plus_minus_Max in component model_parameters (first_order_rate_constant)"
    legend_constants[24] = "Y_PT_D_p_1plus_minus_Max in component model_parameters (first_order_rate_constant)"
    legend_constants[25] = "Y_k_Jext_1plus_minus_Max in component model_parameters (first_order_rate_constant)"
    legend_constants[26] = "X_R_PTH_Ca in component model_parameters (millimolar)"
    legend_constants[27] = "X_R_i_Ca in component model_parameters (millimolar)"
    legend_constants[28] = "X_R_PTH_D in component model_parameters (picomolar)"
    legend_constants[29] = "X_R_E_PTH in component model_parameters (picomolar)"
    legend_constants[30] = "X_R_i_D in component model_parameters (picomolar)"
    legend_constants[31] = "X_R_P_k_PTH in component model_parameters (picomolar)"
    legend_constants[32] = "b_PTH_Ca in component model_parameters (litre_per_millimole)"
    legend_constants[33] = "b_PTH_D in component model_parameters (litre_per_picomole)"
    legend_constants[34] = "b_E_PTH in component model_parameters (litre_per_picomole)"
    legend_constants[35] = "b_i_D in component model_parameters (litre_per_picomole)"
    legend_constants[36] = "a in component model_parameters (dimensionless)"
    legend_constants[37] = "d in component model_parameters (dimensionless)"
    legend_rates[0] = "d/dt Q_Ca_p in component Q_Ca_p (millimole)"
    legend_rates[1] = "d/dt Q_P_p in component Q_P_p (millimole)"
    legend_rates[2] = "d/dt Q_P_c in component Q_P_c (millimole)"
    legend_rates[3] = "d/dt Q_PTH_p in component Q_PTH_p (millimole)"
    legend_rates[4] = "d/dt Q_Ca_b in component Q_Ca_b (millimole)"
    legend_rates[5] = "d/dt Q_P_b in component Q_P_b (millimole)"
    legend_rates[6] = "d/dt Q_E_k in component Q_E_k (millimole)"
    legend_rates[7] = "d/dt Q_D_p in component Q_D_p (millimole)"
    legend_rates[8] = "d/dt Q_TCa_i in component Q_TCa_i (millimole)"
    legend_rates[9] = "d/dt Q_C_PT in component Q_C_PT (millimole)"
    legend_rates[10] = "d/dt Q_TP_k in component Q_TP_k (millimole)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 1.000
    constants[0] = 3.3
    constants[1] = 0.21
    states[1] = 1.000
    constants[2] = 1.27
    states[2] = 3226.0
    states[3] = 1.000
    constants[3] = 3.85
    constants[4] = 100.0
    states[4] = 100.0
    states[5] = 1.000
    states[6] = 1.000
    constants[5] = 1.000
    constants[6] = 0.05
    states[7] = 1.000
    constants[7] = 0.1
    states[8] = 1.000
    states[9] = 1.000
    states[10] = 1.000
    constants[8] = 1.000
    constants[9] = 1.000
    constants[10] = 0.464
    constants[11] = 1.000
    constants[12] = 1.000
    constants[13] = 2.4
    constants[14] = 1.2
    constants[15] = 1.000
    constants[16] = 1.000
    constants[17] = 51.8
    constants[18] = 90.0
    constants[19] = 1.000
    constants[20] = 1.000
    constants[21] = 1.000
    constants[22] = 0.02
    constants[23] = 0.02
    constants[24] = 0.01
    constants[25] = 0.01
    constants[26] = 1.0
    constants[27] = 1.0
    constants[28] = 90.0
    constants[29] = 3.85
    constants[30] = 90.0
    constants[31] = 90.0
    constants[32] = 0.05
    constants[33] = 0.03
    constants[34] = 0.55
    constants[35] = 0.03
    constants[36] = 0.85
    constants[37] = 0.15
    constants[38] = constants[10]*constants[0]
    constants[39] = constants[19]*(constants[12]/(constants[12]+constants[27]))
    constants[40] = constants[22]*(constants[36]*(1.00000-tanh(constants[32]*(constants[13]-constants[26])))+constants[37])
    constants[41] = constants[23]*(constants[36]*(1.00000+tanh(constants[35]*(constants[18]-constants[30])))+constants[37])
    constants[42] = constants[24]*(constants[36]*(1.00000+tanh(constants[33]*(constants[18]-constants[28])))+constants[37])
    constants[43] = constants[25]*(constants[36]*(1.00000+tanh(constants[34]*(1.00000*constants[14]-constants[29])))+constants[37])
    constants[44] = constants[20]*(constants[3]/(constants[3]+1.00000*constants[31]))
    constants[45] = constants[21]*(constants[31]/(1.00000*constants[3]+constants[31]))
    constants[46] = constants[23]*(constants[36]*(1.00000-tanh(constants[35]*(constants[18]-constants[30])))+constants[37])
    constants[47] = constants[24]*(constants[36]*(1.00000-tanh(constants[33]*(constants[18]-constants[28])))+constants[37])
    constants[48] = constants[9]
    constants[49] = constants[41]
    constants[50] = constants[46]
    constants[51] = constants[42]
    constants[52] = constants[47]
    constants[53] = constants[43]
    constants[54] = 1.95000+constants[44]
    constants[55] = constants[17]*1.00000*constants[14]
    constants[56] = constants[54]/constants[13]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[6] = constants[5]-constants[6]*states[6]
    rates[7] = 1.00000*states[6]-constants[7]*states[7]
    rates[8] = (1.00000-states[8])*constants[49]-states[8]*constants[50]
    rates[9] = (1.00000-states[9])*constants[51]-states[9]*constants[52]
    rates[10] = (1.00000-states[10])*constants[53]-states[10]*constants[8]
    algebraic[1] = 1.00000*constants[40]*(constants[51]*(1.00000-states[9])-constants[52]*states[9])
    rates[3] = algebraic[1]-constants[4]*1.00000*constants[3]
    algebraic[0] = constants[15]*states[4]
    rates[4] = algebraic[0]-constants[0]
    algebraic[3] = (constants[39]*(1.00000-states[8])*1.00000*constants[49]+constants[11]*1.00000*(constants[12]-constants[13]))-states[8]*constants[50]
    rates[0] = (constants[0]+algebraic[3])-(algebraic[0]+constants[1])
    algebraic[4] = constants[10]*algebraic[0]
    rates[5] = algebraic[4]-constants[38]
    algebraic[6] = constants[16]*states[2]
    rates[1] = (constants[38]+constants[48]+algebraic[6])-(algebraic[4]+constants[2]+constants[55])
    rates[2] = constants[55]-algebraic[6]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[1] = 1.00000*constants[40]*(constants[51]*(1.00000-states[9])-constants[52]*states[9])
    algebraic[0] = constants[15]*states[4]
    algebraic[3] = (constants[39]*(1.00000-states[8])*1.00000*constants[49]+constants[11]*1.00000*(constants[12]-constants[13]))-states[8]*constants[50]
    algebraic[4] = constants[10]*algebraic[0]
    algebraic[6] = constants[16]*states[2]
    algebraic[2] = constants[45]*states[10]
    algebraic[5] = algebraic[2]/constants[14]
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