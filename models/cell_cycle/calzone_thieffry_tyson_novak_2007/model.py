# Size of variable arrays:
sizeAlgebraic = 4
sizeStates = 17
sizeConstants = 43
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_states[0] = "MPF_n in component MPF_n (dimensionless)"
    legend_constants[0] = "k_in in component parameters (first_order_rate_constant)"
    legend_constants[1] = "k_out in component parameters (first_order_rate_constant)"
    legend_constants[2] = "k_dn_ in component parameters (first_order_rate_constant)"
    legend_constants[3] = "k_dn__ in component parameters (first_order_rate_constant)"
    legend_constants[4] = "k_wee_ in component parameters (first_order_rate_constant)"
    legend_constants[5] = "k_wee__ in component parameters (first_order_rate_constant)"
    legend_constants[6] = "k_stg_ in component parameters (first_order_rate_constant)"
    legend_constants[7] = "k_stg__ in component parameters (first_order_rate_constant)"
    legend_states[1] = "MPF_c in component MPF_c (dimensionless)"
    legend_states[2] = "FZY in component FZY (dimensionless)"
    legend_states[3] = "Wee1_n in component Wee1_n (dimensionless)"
    legend_states[4] = "StgP_n in component StgP_n (dimensionless)"
    legend_states[5] = "preMPF_n in component preMPF_n (dimensionless)"
    legend_states[6] = "preMPF_c in component preMPF_c (dimensionless)"
    legend_constants[8] = "k_sc in component parameters (first_order_rate_constant)"
    legend_constants[9] = "k_dc_ in component parameters (first_order_rate_constant)"
    legend_constants[10] = "epsilon in component parameters (dimensionless)"
    legend_states[7] = "N in component N (dimensionless)"
    legend_states[8] = "StgP_c in component StgP_c (dimensionless)"
    legend_states[9] = "Wee1_c in component Wee1_c (dimensionless)"
    legend_states[10] = "IE in component IE (dimensionless)"
    legend_constants[11] = "j_aie in component parameters (dimensionless)"
    legend_constants[12] = "j_iie in component parameters (dimensionless)"
    legend_constants[13] = "k_aie in component parameters (first_order_rate_constant)"
    legend_constants[14] = "k_iie in component parameters (first_order_rate_constant)"
    legend_constants[15] = "j_afz in component parameters (dimensionless)"
    legend_constants[16] = "j_ifz in component parameters (dimensionless)"
    legend_constants[17] = "k_afz in component parameters (first_order_rate_constant)"
    legend_constants[18] = "k_ifz in component parameters (first_order_rate_constant)"
    legend_states[11] = "Stg_m in component Stg_m (dimensionless)"
    legend_constants[19] = "k_dm_ in component parameters (first_order_rate_constant)"
    legend_constants[20] = "k_dm__ in component parameters (first_order_rate_constant)"
    legend_constants[21] = "j_m in component parameters (dimensionless)"
    legend_states[12] = "Xp in component Xp (dimensionless)"
    legend_states[13] = "Xm in component Xm (dimensionless)"
    legend_constants[22] = "k_sxm in component parameters (first_order_rate_constant)"
    legend_constants[23] = "k_sxp in component parameters (first_order_rate_constant)"
    legend_constants[24] = "k_ins in component parameters (first_order_rate_constant)"
    legend_constants[25] = "k_outs in component parameters (first_order_rate_constant)"
    legend_constants[26] = "k_astg_ in component parameters (first_order_rate_constant)"
    legend_constants[27] = "k_astg__ in component parameters (first_order_rate_constant)"
    legend_constants[28] = "k_istg in component parameters (first_order_rate_constant)"
    legend_constants[29] = "k_dstg in component parameters (first_order_rate_constant)"
    legend_constants[30] = "j_astg in component parameters (dimensionless)"
    legend_constants[31] = "j_istg in component parameters (dimensionless)"
    legend_states[14] = "Stg_n in component Stg_n (dimensionless)"
    legend_states[15] = "Stg_c in component Stg_c (dimensionless)"
    legend_constants[32] = "k_sstg in component parameters (first_order_rate_constant)"
    legend_constants[33] = "k_inw in component parameters (first_order_rate_constant)"
    legend_constants[34] = "k_outw in component parameters (first_order_rate_constant)"
    legend_constants[35] = "k_awee in component parameters (first_order_rate_constant)"
    legend_constants[36] = "k_iwee_ in component parameters (first_order_rate_constant)"
    legend_constants[37] = "k_iwee__ in component parameters (first_order_rate_constant)"
    legend_constants[38] = "j_awee in component parameters (dimensionless)"
    legend_constants[39] = "j_iwee in component parameters (dimensionless)"
    legend_states[16] = "Wee1P_n in component Wee1P_n (dimensionless)"
    legend_algebraic[0] = "Wee1P_c in component Wee1P_c (dimensionless)"
    legend_constants[40] = "Wee1_T in component Wee1P_c (dimensionless)"
    legend_algebraic[1] = "CycB_T in component CycB_T (dimensionless)"
    legend_algebraic[2] = "Stg_T in component Stg_T (dimensionless)"
    legend_algebraic[3] = "StgP_T in component StgP_T (dimensionless)"
    legend_constants[41] = "k_ez in component parameters (first_order_rate_constant)"
    legend_rates[0] = "d/dt MPF_n in component MPF_n (dimensionless)"
    legend_rates[5] = "d/dt preMPF_n in component preMPF_n (dimensionless)"
    legend_rates[1] = "d/dt MPF_c in component MPF_c (dimensionless)"
    legend_rates[6] = "d/dt preMPF_c in component preMPF_c (dimensionless)"
    legend_rates[10] = "d/dt IE in component IE (dimensionless)"
    legend_rates[2] = "d/dt FZY in component FZY (dimensionless)"
    legend_rates[11] = "d/dt Stg_m in component Stg_m (dimensionless)"
    legend_rates[13] = "d/dt Xm in component Xm (dimensionless)"
    legend_rates[12] = "d/dt Xp in component Xp (dimensionless)"
    legend_rates[4] = "d/dt StgP_n in component StgP_n (dimensionless)"
    legend_rates[14] = "d/dt Stg_n in component Stg_n (dimensionless)"
    legend_rates[8] = "d/dt StgP_c in component StgP_c (dimensionless)"
    legend_rates[15] = "d/dt Stg_c in component Stg_c (dimensionless)"
    legend_rates[3] = "d/dt Wee1_n in component Wee1_n (dimensionless)"
    legend_rates[9] = "d/dt Wee1_c in component Wee1_c (dimensionless)"
    legend_rates[16] = "d/dt Wee1P_n in component Wee1P_n (dimensionless)"
    legend_rates[7] = "d/dt N in component N (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0
    constants[0] = 0.15
    constants[1] = 0
    constants[2] = 0.01
    constants[3] = 1.5
    constants[4] = 0.005
    constants[5] = 1
    constants[6] = 0.2
    constants[7] = 2
    states[1] = 1
    states[2] = 0
    states[3] = 0
    states[4] = 0
    states[5] = 0
    states[6] = 0
    constants[8] = 0.01
    constants[9] = 0.01
    constants[10] = 0.00007
    states[7] = 1
    states[8] = 0
    states[9] = 0
    states[10] = 0
    constants[11] = 0.01
    constants[12] = 0.01
    constants[13] = 1
    constants[14] = 0.4
    constants[15] = 0.01
    constants[16] = 0.01
    constants[17] = 1
    constants[18] = 0.2
    states[11] = 1
    constants[19] = 0.002
    constants[20] = 0.2
    constants[21] = 0.05
    states[12] = 0
    states[13] = 0
    constants[22] = 0.0005
    constants[23] = 0.001
    constants[24] = 0.08
    constants[25] = 0.02
    constants[26] = 0
    constants[27] = 1
    constants[28] = 0.3
    constants[29] = 0.015
    constants[30] = 0.05
    constants[31] = 0.05
    states[14] = 0
    states[15] = 1
    constants[32] = 0.02
    constants[33] = 0.04
    constants[34] = 0.01
    constants[35] = 0.3
    constants[36] = 0.01
    constants[37] = 1
    constants[38] = 0.05
    constants[39] = 0.05
    states[16] = 0
    constants[40] = 0.8
    constants[41] = 0.5
    constants[42] = 0.00000
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[7] = constants[42]
    rates[0] = (((constants[0]*states[1]-constants[1]*states[0])-(constants[2]+constants[3]*states[2])*states[0])-(constants[4]+constants[5]*states[3])*states[0])+(constants[6]+constants[7]*states[4])*states[5]
    rates[5] = (((constants[0]*states[6]-constants[1]*states[5])-(constants[2]+constants[3]*states[2])*states[5])+(constants[4]+constants[5]*states[3])*states[0])-(constants[4]+constants[7]*states[4])*states[5]
    rates[1] = (((constants[8]-(constants[10]*states[7]*(constants[0]*states[1]-constants[1]*states[0]))/(1.00000-states[7]*constants[10]))-constants[9]*states[1])+(constants[6]+constants[7]*states[8])*states[6])-(constants[4]+constants[5]*states[9])*states[1]
    rates[6] = (((-constants[10]*states[7]*(constants[0]*states[6]-constants[1]*states[5]))/(1.00000-states[7]*constants[10])-constants[9]*states[6])-(constants[6]+constants[7]*states[8])*states[6])+(constants[4]+constants[5]*states[9])*states[1]
    rates[10] = (constants[13]*(1.00000-states[10])*states[0])/((constants[11]+1.00000)-states[10])-(constants[14]*states[10])/(constants[12]+states[10])
    rates[2] = (constants[17]*states[10]*(1.00000-states[2]))/((constants[15]+1.00000)-states[2])-(constants[18]*states[2])/(constants[16]+states[2])
    rates[11] = -(constants[19]/(constants[21]+states[11])+constants[20]*states[12])*states[11]
    rates[13] = constants[22]*states[7]
    rates[12] = constants[23]*states[13]
    rates[4] = (((constants[24]*states[8]-constants[25]*states[4])+((constants[26]+constants[27]*states[0])*states[14])/(constants[30]+states[14]))-(constants[28]*states[4])/(constants[31]+states[4]))-constants[29]*states[4]
    rates[14] = (((constants[24]*states[15]-constants[25]*states[14])-((constants[26]+constants[27]*states[0])*states[14])/(constants[30]+states[14]))+(constants[28]*states[4])/(constants[31]+states[4]))-constants[29]*states[14]
    rates[8] = ((-constants[29]*states[8]-(constants[10]*states[7]*(constants[24]*states[8]-constants[25]*states[4]))/(1.00000-states[7]*constants[10]))+((constants[26]+constants[27]*states[1])*states[15])/(constants[30]+states[15]))-(constants[28]*states[8])/(constants[31]+states[8])
    rates[15] = (((constants[32]*states[11]-constants[29]*states[15])-(constants[10]*states[7]*(constants[24]*states[15]-constants[25]*states[14]))/(1.00000-states[7]*constants[10]))-((constants[26]+constants[27]*states[1])*states[15])/(constants[30]+states[15]))+(constants[28]*states[8])/(constants[31]+states[8])
    rates[3] = ((constants[33]*states[9]-constants[34]*states[3])+(constants[35]*states[16])/(constants[38]+states[16]))-((constants[36]+constants[37]*states[0])*states[3])/(constants[39]+states[3])
    algebraic[0] = (constants[40]-states[7]*constants[10]*(states[3]+states[16]))/(1.00000-states[7]*constants[10])-states[9]
    rates[9] = ((-(constants[33]*states[9]-constants[34]*states[3])*states[7]*constants[10])/(1.00000-states[7]*constants[10])+(constants[35]*algebraic[0])/(constants[38]+algebraic[0]))-((constants[36]+constants[37]*states[1])*states[9])/(constants[39]+states[9])
    rates[16] = ((constants[33]*algebraic[0]-constants[34]*states[16])-(constants[35]*states[16])/(constants[38]+states[16]))+((constants[36]+constants[37]*states[0])*states[3])/(constants[39]+states[3])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = (constants[40]-states[7]*constants[10]*(states[3]+states[16]))/(1.00000-states[7]*constants[10])-states[9]
    algebraic[1] = (1.00000-states[7]*constants[10])*(states[1]+states[6])+states[7]*constants[10]*(states[0]+states[5])
    algebraic[2] = (1.00000-states[7]*constants[10])*(states[15]+states[8])+states[7]*constants[10]*(states[14]+states[4])
    algebraic[3] = (1.00000-states[7]*constants[10])*states[8]+states[7]*constants[10]*states[4]
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