# Size of variable arrays:
sizeAlgebraic = 19
sizeStates = 7
sizeConstants = 42
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (s)"
    legend_states[0] = "F6P in component F6P (mM)"
    legend_algebraic[0] = "V_hk in component V_hk (mM_per_s)"
    legend_algebraic[12] = "V_pfk in component V_pfk (mM_per_s)"
    legend_algebraic[18] = "V_pfk2 in component V_pfk2 (mM_per_s)"
    legend_states[1] = "F26P in component F26P (mM)"
    legend_states[2] = "GAP in component GAP (mM)"
    legend_algebraic[10] = "V_pk in component V_pk (mM_per_s)"
    legend_states[3] = "PYR in component PYR (mM)"
    legend_algebraic[1] = "V_ldh in component V_ldh (mM_per_s)"
    legend_algebraic[5] = "V_op in component V_op (mM_per_s)"
    legend_states[4] = "LAC in component LAC (mM)"
    legend_algebraic[16] = "V_lac in component V_lac (mM_per_s)"
    legend_states[5] = "ATP in component ATP (mM)"
    legend_algebraic[15] = "V_ATPase in component V_ATPase (mM_per_s)"
    legend_algebraic[6] = "V_ck in component V_ck (mM_per_s)"
    legend_algebraic[7] = "dAMP_dATP in component dAMP_dATP (dimensionless)"
    legend_states[6] = "PCr in component PCr (mM)"
    legend_constants[0] = "Vmax_hk in component V_hk (mM_per_s)"
    legend_constants[1] = "Km_ATP_hk in component V_hk (mM)"
    legend_constants[2] = "KI_F6P in component V_hk (mM)"
    legend_constants[3] = "Vmax_pfk in component V_pfk (mM_per_s)"
    legend_constants[4] = "Km_ATP_pfk in component V_pfk (mM)"
    legend_constants[5] = "Km_F6P_pfk in component V_pfk (mM)"
    legend_constants[6] = "Km_F26P_pfk in component V_pfk (mM)"
    legend_algebraic[11] = "AMP_act in component AMP_act (dimensionless)"
    legend_algebraic[9] = "ATP_inh in component ATP_inh (dimensionless)"
    legend_constants[7] = "Vmaxf_pfk2 in component V_pfk2 (mM_per_s)"
    legend_constants[8] = "Vmaxr_pfk2 in component V_pfk2 (mM_per_s)"
    legend_constants[9] = "Km_ATP_pfk2 in component V_pfk2 (mM)"
    legend_constants[10] = "Km_F6P_pfk2 in component V_pfk2 (mM)"
    legend_constants[11] = "Km_F26P_pfk2 in component V_pfk2 (mM)"
    legend_algebraic[17] = "AMP_pfk2 in component AMP_pfk2 (dimensionless)"
    legend_constants[12] = "Vmax_pk in component V_pk (mM_per_s)"
    legend_constants[13] = "Km_ADP_pk in component V_pk (mM)"
    legend_constants[14] = "Km_GAP_pk in component V_pk (mM)"
    legend_algebraic[4] = "ADP in component ADP (mM)"
    legend_constants[15] = "Vmax_op in component V_op (mM_per_s)"
    legend_constants[16] = "Km_ADP_op in component V_op (mM)"
    legend_constants[17] = "Km_PYR_op in component V_op (mM)"
    legend_constants[18] = "kf_ldh in component V_ldh (per_s)"
    legend_constants[19] = "kr_ldh in component V_ldh (per_s)"
    legend_constants[20] = "kf_ck in component V_ck (per_mM_s)"
    legend_constants[21] = "kr_ck in component V_ck (per_mM_s)"
    legend_algebraic[2] = "Cr in component Cr (mM)"
    legend_constants[22] = "PCrtot in component Cr (mM)"
    legend_constants[23] = "Vmax_ATPase in component V_ATPase (mM_per_s)"
    legend_constants[24] = "Km_ATP in component V_ATPase (mM)"
    legend_algebraic[14] = "v_stim in component v_stim (dimensionless)"
    legend_constants[25] = "Vlac_0 in component V_lac (mM_per_s)"
    legend_constants[26] = "K_LAC_eff in component V_lac (per_s)"
    legend_constants[27] = "K_LAC in component V_lac (dimensionless)"
    legend_algebraic[3] = "u in component ADP (dimensionless)"
    legend_constants[28] = "ANP in component model_parameters (mM)"
    legend_constants[29] = "Q_adk in component model_parameters (dimensionless)"
    legend_algebraic[8] = "AMP in component AMP (mM)"
    legend_constants[30] = "nATP in component ATP_inh (dimensionless)"
    legend_constants[31] = "KI_ATP in component ATP_inh (mM)"
    legend_constants[32] = "nAMP in component AMP_act (dimensionless)"
    legend_constants[33] = "Ka_AMP in component AMP_act (mM)"
    legend_algebraic[13] = "unitpulseSB in component v_stim (dimensionless)"
    legend_constants[34] = "stim in component v_stim (dimensionless)"
    legend_constants[35] = "to in component v_stim (s)"
    legend_constants[36] = "tend in component v_stim (s)"
    legend_constants[37] = "v1_n in component v_stim (dimensionless)"
    legend_constants[38] = "v2_n in component v_stim (dimensionless)"
    legend_constants[39] = "t_n_stim in component v_stim (s)"
    legend_constants[40] = "Kamp_pfk2 in component AMP_pfk2 (mM)"
    legend_constants[41] = "nh_amp in component AMP_pfk2 (dimensionless)"
    legend_rates[0] = "d/dt F6P in component F6P (mM)"
    legend_rates[1] = "d/dt F26P in component F26P (mM)"
    legend_rates[2] = "d/dt GAP in component GAP (mM)"
    legend_rates[3] = "d/dt PYR in component PYR (mM)"
    legend_rates[4] = "d/dt LAC in component LAC (mM)"
    legend_rates[5] = "d/dt ATP in component ATP (mM)"
    legend_rates[6] = "d/dt PCr in component PCr (mM)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.2
    states[1] = 0.001
    states[2] = 0.0405
    states[3] = 0.1
    states[4] = 0.5
    states[5] = 2.402
    states[6] = 18.14
    constants[0] = 2.5
    constants[1] = 0.5
    constants[2] = 0.068
    constants[3] = 3.85
    constants[4] = 0.05
    constants[5] = 0.18
    constants[6] = 0.01
    constants[7] = 0
    constants[8] = 0
    constants[9] = 0.05
    constants[10] = 0.01
    constants[11] = 0.0001
    constants[12] = 5.0
    constants[13] = 0.005
    constants[14] = 0.4
    constants[15] = 1.0
    constants[16] = 0.005
    constants[17] = 0.5
    constants[18] = 0
    constants[19] = 0
    constants[20] = 0
    constants[21] = 0
    constants[22] = 20.0
    constants[23] = 0.9355
    constants[24] = 0.5
    constants[25] = 0.355
    constants[26] = 0.71
    constants[27] = 0.641
    constants[28] = 2.51
    constants[29] = 0.92
    constants[30] = 0.4
    constants[31] = 1.0
    constants[32] = 0.5
    constants[33] = 0.05
    constants[34] = 1
    constants[35] = 50
    constants[36] = 200
    constants[37] = 0.5
    constants[38] = 0.0
    constants[39] = 2
    constants[40] = 0.005
    constants[41] = 2
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[3] = power(constants[29], 2.00000)+4.00000*constants[29]*(constants[28]/states[5]-1.00000)
    algebraic[4] = (states[5]/2.00000)*(-constants[29]+power(algebraic[3], 1.0/2))
    algebraic[2] = constants[22]-states[6]
    algebraic[6] = constants[20]*states[6]*algebraic[4]-constants[21]*algebraic[2]*states[5]
    rates[6] = -algebraic[6]
    algebraic[9] = power((1.00000+constants[30]*(states[5]/constants[31]))/(1.00000+states[5]/constants[31]), 4.00000)
    algebraic[10] = constants[12]*(states[2]/(states[2]+constants[14]))*(algebraic[4]/(algebraic[4]+constants[13]))*algebraic[9]
    algebraic[1] = constants[18]*states[3]-constants[19]*states[4]
    algebraic[5] = constants[15]*(states[3]/(states[3]+constants[17]))*(algebraic[4]/(algebraic[4]+constants[16]))*(1.00000/(1.00000+0.100000*24.0000))
    rates[3] = algebraic[10]-(algebraic[5]+algebraic[1])
    algebraic[8] = constants[28]-(states[5]+algebraic[4])
    algebraic[11] = power((1.00000+algebraic[8]/constants[33])/(1.00000+constants[32]*(algebraic[8]/constants[33])), 4.00000)
    algebraic[12] = constants[3]*(states[0]/(states[0]+constants[5]))*(states[5]/(states[5]+constants[4]))*(states[1]/(states[1]+constants[6]))*algebraic[9]*algebraic[11]
    rates[2] = 2.00000*algebraic[12]-algebraic[10]
    algebraic[13] = custom_piecewise([greater_equal(voi , constants[35]) & less_equal(voi , constants[35]+constants[36]), 1.00000 , True, 0.00000])
    algebraic[14] = constants[34]*(constants[37]+constants[38]*((voi-constants[35])/constants[39])*exp(-((voi-constants[35])*(algebraic[13]/constants[39]))))*algebraic[13]
    algebraic[16] = constants[25]*(1.00000+algebraic[14]*constants[27])-constants[26]*states[4]
    rates[4] = 2.25000*algebraic[1]+algebraic[16]
    algebraic[0] = constants[0]*(states[5]/(states[5]+constants[1]))*(power(1.00000+power(states[0]/constants[2], 4.00000), -1.00000))
    algebraic[17] = (power(algebraic[8]/constants[40], constants[41]))/(1.00000+power(algebraic[8]/constants[40], constants[41]))
    algebraic[18] = constants[7]*(states[5]/(states[5]+constants[9]))*(states[0]/(states[0]+constants[10]))*algebraic[17]-constants[8]*(states[1]/(states[1]+constants[11]))
    rates[0] = algebraic[0]-(algebraic[12]-algebraic[18])
    rates[1] = algebraic[18]
    algebraic[15] = constants[23]*(states[5]/(states[5]+constants[24]))*(1.00000+algebraic[14])
    algebraic[7] = -1.00000+constants[29]/2.00000+-(0.500000*(power(algebraic[3], 1.0/2)))+constants[29]*(constants[28]/(states[5]*(power(algebraic[3], 1.0/2))))
    rates[5] = ((2.00000*algebraic[10]+15.0000*algebraic[5]+algebraic[6])-(algebraic[0]+algebraic[12]+algebraic[18]+algebraic[15]))*(power(1.00000-algebraic[7], -1.00000))
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[3] = power(constants[29], 2.00000)+4.00000*constants[29]*(constants[28]/states[5]-1.00000)
    algebraic[4] = (states[5]/2.00000)*(-constants[29]+power(algebraic[3], 1.0/2))
    algebraic[2] = constants[22]-states[6]
    algebraic[6] = constants[20]*states[6]*algebraic[4]-constants[21]*algebraic[2]*states[5]
    algebraic[9] = power((1.00000+constants[30]*(states[5]/constants[31]))/(1.00000+states[5]/constants[31]), 4.00000)
    algebraic[10] = constants[12]*(states[2]/(states[2]+constants[14]))*(algebraic[4]/(algebraic[4]+constants[13]))*algebraic[9]
    algebraic[1] = constants[18]*states[3]-constants[19]*states[4]
    algebraic[5] = constants[15]*(states[3]/(states[3]+constants[17]))*(algebraic[4]/(algebraic[4]+constants[16]))*(1.00000/(1.00000+0.100000*24.0000))
    algebraic[8] = constants[28]-(states[5]+algebraic[4])
    algebraic[11] = power((1.00000+algebraic[8]/constants[33])/(1.00000+constants[32]*(algebraic[8]/constants[33])), 4.00000)
    algebraic[12] = constants[3]*(states[0]/(states[0]+constants[5]))*(states[5]/(states[5]+constants[4]))*(states[1]/(states[1]+constants[6]))*algebraic[9]*algebraic[11]
    algebraic[13] = custom_piecewise([greater_equal(voi , constants[35]) & less_equal(voi , constants[35]+constants[36]), 1.00000 , True, 0.00000])
    algebraic[14] = constants[34]*(constants[37]+constants[38]*((voi-constants[35])/constants[39])*exp(-((voi-constants[35])*(algebraic[13]/constants[39]))))*algebraic[13]
    algebraic[16] = constants[25]*(1.00000+algebraic[14]*constants[27])-constants[26]*states[4]
    algebraic[0] = constants[0]*(states[5]/(states[5]+constants[1]))*(power(1.00000+power(states[0]/constants[2], 4.00000), -1.00000))
    algebraic[17] = (power(algebraic[8]/constants[40], constants[41]))/(1.00000+power(algebraic[8]/constants[40], constants[41]))
    algebraic[18] = constants[7]*(states[5]/(states[5]+constants[9]))*(states[0]/(states[0]+constants[10]))*algebraic[17]-constants[8]*(states[1]/(states[1]+constants[11]))
    algebraic[15] = constants[23]*(states[5]/(states[5]+constants[24]))*(1.00000+algebraic[14])
    algebraic[7] = -1.00000+constants[29]/2.00000+-(0.500000*(power(algebraic[3], 1.0/2)))+constants[29]*(constants[28]/(states[5]*(power(algebraic[3], 1.0/2))))
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