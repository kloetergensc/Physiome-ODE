# Size of variable arrays:
sizeAlgebraic = 15
sizeStates = 13
sizeConstants = 71
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_states[0] = "Cln2 in component Cln2 (dimensionless)"
    legend_constants[0] = "ks_n2 in component Cln2 (first_order_rate_constant)"
    legend_constants[1] = "ks_n2_ in component Cln2 (first_order_rate_constant)"
    legend_constants[2] = "kd_n2 in component Cln2 (first_order_rate_constant)"
    legend_states[1] = "mass in component mass (dimensionless)"
    legend_algebraic[12] = "SBF in component SBF (dimensionless)"
    legend_states[2] = "Clb2_T in component Clb2_T (dimensionless)"
    legend_constants[3] = "Hct1_T in component Clb2_T (dimensionless)"
    legend_constants[4] = "ks_b2 in component Clb2_T (first_order_rate_constant)"
    legend_constants[5] = "ks_b2_ in component Clb2_T (first_order_rate_constant)"
    legend_constants[6] = "kd_b2 in component Clb2_T (first_order_rate_constant)"
    legend_constants[7] = "kd_b2_ in component Clb2_T (first_order_rate_constant)"
    legend_constants[8] = "kd_b2__ in component Clb2_T (first_order_rate_constant)"
    legend_algebraic[0] = "Vd_b2 in component Clb2_T (first_order_rate_constant)"
    legend_algebraic[3] = "Mcm1 in component Mcm1 (dimensionless)"
    legend_states[3] = "Hct1 in component Hct1 (dimensionless)"
    legend_states[4] = "Cdc20 in component Cdc20 (dimensionless)"
    legend_algebraic[2] = "Clb2 in component Clb2 (dimensionless)"
    legend_states[5] = "Clb2_Sic1 in component Clb2_Sic1 (dimensionless)"
    legend_algebraic[4] = "Clb5 in component Clb5 (dimensionless)"
    legend_states[6] = "Clb5_Sic1 in component Clb5_Sic1 (dimensionless)"
    legend_states[7] = "Clb5_T in component Clb5_T (dimensionless)"
    legend_algebraic[5] = "Sic1 in component Sic1 (dimensionless)"
    legend_states[8] = "Sic1_T in component Sic1_T (dimensionless)"
    legend_constants[9] = "ks_b5 in component Clb5_T (first_order_rate_constant)"
    legend_constants[10] = "ks_b5_ in component Clb5_T (first_order_rate_constant)"
    legend_constants[11] = "kd_b5 in component Clb5_T (first_order_rate_constant)"
    legend_constants[12] = "kd_b5_ in component Clb5_T (first_order_rate_constant)"
    legend_algebraic[6] = "Vd_b5 in component Clb5_T (first_order_rate_constant)"
    legend_algebraic[14] = "MBF in component MBF (dimensionless)"
    legend_algebraic[7] = "Bck2 in component Bck2 (dimensionless)"
    legend_constants[13] = "Bck2_0 in component Bck2 (dimensionless)"
    legend_algebraic[8] = "Cln3 in component Cln3 (dimensionless)"
    legend_constants[14] = "Jn3 in component Cln3 (dimensionless)"
    legend_constants[15] = "Dn3 in component Cln3 (dimensionless)"
    legend_constants[16] = "Cln3_max in component Cln3 (dimensionless)"
    legend_constants[17] = "ks_c1 in component Sic1_T (first_order_rate_constant)"
    legend_constants[18] = "ks_c1_ in component Sic1_T (first_order_rate_constant)"
    legend_constants[19] = "kd1_c1 in component parameters (first_order_rate_constant)"
    legend_constants[20] = "Jd2_c1 in component parameters (dimensionless)"
    legend_algebraic[10] = "Vd2_c1 in component Vd2_c1 (first_order_rate_constant)"
    legend_algebraic[13] = "Swi5 in component Swi5 (dimensionless)"
    legend_constants[21] = "kas_b2 in component Clb2_Sic1 (first_order_rate_constant)"
    legend_constants[22] = "kdi_b2 in component Clb2_Sic1 (first_order_rate_constant)"
    legend_constants[23] = "kas_b5 in component Clb5_Sic1 (first_order_rate_constant)"
    legend_constants[24] = "kdi_b5 in component Clb5_Sic1 (first_order_rate_constant)"
    legend_constants[25] = "kd2_c1 in component Vd2_c1 (first_order_rate_constant)"
    legend_constants[26] = "epsilonc1_n3 in component Vd2_c1 (dimensionless)"
    legend_constants[27] = "epsilonc1_k2 in component Vd2_c1 (dimensionless)"
    legend_constants[28] = "epsilonc1_b5 in component Vd2_c1 (dimensionless)"
    legend_constants[29] = "epsilonc1_b2 in component Vd2_c1 (dimensionless)"
    legend_states[9] = "Cdc20_T in component Cdc20_T (dimensionless)"
    legend_constants[30] = "ks_20 in component Cdc20_T (first_order_rate_constant)"
    legend_constants[31] = "ks_20_ in component Cdc20_T (first_order_rate_constant)"
    legend_constants[32] = "kd_20 in component parameters (first_order_rate_constant)"
    legend_constants[33] = "ka_20 in component Cdc20 (first_order_rate_constant)"
    legend_constants[34] = "ki_20 in component Cdc20 (first_order_rate_constant)"
    legend_constants[35] = "ki_20_ in component Cdc20 (first_order_rate_constant)"
    legend_algebraic[1] = "Vi_20 in component Cdc20 (first_order_rate_constant)"
    legend_states[10] = "ORI in component ORI (dimensionless)"
    legend_states[11] = "SPN in component SPN (dimensionless)"
    legend_constants[36] = "ka_t1 in component Hct1 (first_order_rate_constant)"
    legend_constants[37] = "ka_t1_ in component Hct1 (first_order_rate_constant)"
    legend_constants[38] = "ki_t1 in component Hct1 (first_order_rate_constant)"
    legend_constants[39] = "ki_t1_ in component Hct1 (first_order_rate_constant)"
    legend_algebraic[11] = "Vi_t1 in component Hct1 (first_order_rate_constant)"
    legend_constants[40] = "Ji_t1 in component Hct1 (dimensionless)"
    legend_constants[41] = "Ja_t1 in component Hct1 (dimensionless)"
    legend_constants[42] = "epsiloni_t1_n2 in component Hct1 (dimensionless)"
    legend_constants[43] = "epsiloni_t1_b5 in component Hct1 (dimensionless)"
    legend_constants[44] = "epsiloni_t1_b2 in component Hct1 (dimensionless)"
    legend_constants[45] = "mu in component mass (first_order_rate_constant)"
    legend_constants[46] = "ks_ori in component ORI (first_order_rate_constant)"
    legend_constants[47] = "kd_ori in component ORI (first_order_rate_constant)"
    legend_constants[48] = "epsilonori_b2 in component ORI (dimensionless)"
    legend_states[12] = "BUD in component BUD (dimensionless)"
    legend_constants[49] = "ks_bud in component BUD (first_order_rate_constant)"
    legend_constants[50] = "kd_bud in component BUD (first_order_rate_constant)"
    legend_constants[51] = "epsilonbud_b5 in component BUD (dimensionless)"
    legend_constants[52] = "ks_spn in component SPN (first_order_rate_constant)"
    legend_constants[53] = "kd_spn in component SPN (first_order_rate_constant)"
    legend_constants[54] = "J_spn in component SPN (dimensionless)"
    legend_constants[55] = "ka_sbf in component SBF (first_order_rate_constant)"
    legend_constants[56] = "ki_sbf in component SBF (first_order_rate_constant)"
    legend_constants[57] = "ki_sbf_ in component SBF (first_order_rate_constant)"
    legend_algebraic[9] = "Va_sbf in component SBF (first_order_rate_constant)"
    legend_constants[58] = "Ji_sbf in component SBF (dimensionless)"
    legend_constants[59] = "Ja_sbf in component SBF (dimensionless)"
    legend_constants[60] = "epsilonsbf_n3 in component SBF (dimensionless)"
    legend_constants[61] = "epsilonsbf_b5 in component SBF (dimensionless)"
    legend_constants[62] = "ka_mcm in component Mcm1 (first_order_rate_constant)"
    legend_constants[63] = "ki_mcm in component Mcm1 (first_order_rate_constant)"
    legend_constants[64] = "Ji_mcm in component Mcm1 (dimensionless)"
    legend_constants[65] = "Ja_mcm in component Mcm1 (dimensionless)"
    legend_constants[66] = "ka_swi in component Swi5 (first_order_rate_constant)"
    legend_constants[67] = "ki_swi in component Swi5 (first_order_rate_constant)"
    legend_constants[68] = "ki_swi_ in component Swi5 (first_order_rate_constant)"
    legend_constants[69] = "Ji_swi in component Swi5 (dimensionless)"
    legend_constants[70] = "Ja_swi in component Swi5 (dimensionless)"
    legend_rates[0] = "d/dt Cln2 in component Cln2 (dimensionless)"
    legend_rates[2] = "d/dt Clb2_T in component Clb2_T (dimensionless)"
    legend_rates[7] = "d/dt Clb5_T in component Clb5_T (dimensionless)"
    legend_rates[8] = "d/dt Sic1_T in component Sic1_T (dimensionless)"
    legend_rates[5] = "d/dt Clb2_Sic1 in component Clb2_Sic1 (dimensionless)"
    legend_rates[6] = "d/dt Clb5_Sic1 in component Clb5_Sic1 (dimensionless)"
    legend_rates[9] = "d/dt Cdc20_T in component Cdc20_T (dimensionless)"
    legend_rates[4] = "d/dt Cdc20 in component Cdc20 (dimensionless)"
    legend_rates[3] = "d/dt Hct1 in component Hct1 (dimensionless)"
    legend_rates[1] = "d/dt mass in component mass (dimensionless)"
    legend_rates[10] = "d/dt ORI in component ORI (dimensionless)"
    legend_rates[12] = "d/dt BUD in component BUD (dimensionless)"
    legend_rates[11] = "d/dt SPN in component SPN (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.0078
    constants[0] = 0
    constants[1] = 0.05
    constants[2] = 0.1
    states[1] = 0.6608
    states[2] = 0.2342
    constants[3] = 1
    constants[4] = 0.002
    constants[5] = 0.05
    constants[6] = 0.01
    constants[7] = 2
    constants[8] = 0.05
    states[3] = 0.9946
    states[4] = 0.6848
    states[5] = 0.079
    states[6] = 0.0207
    states[7] = 0.0614
    states[8] = 0.1231
    constants[9] = 0.006
    constants[10] = 0.02
    constants[11] = 0.1
    constants[12] = 0.25
    constants[13] = 0.0027
    constants[14] = 6
    constants[15] = 1
    constants[16] = 0.02
    constants[17] = 0.02
    constants[18] = 0.1
    constants[19] = 0.01
    constants[20] = 0.05
    constants[21] = 50
    constants[22] = 0.05
    constants[23] = 50
    constants[24] = 0.05
    constants[25] = 0.3
    constants[26] = 20
    constants[27] = 2
    constants[28] = 1
    constants[29] = 0.067
    states[9] = 0.8332
    constants[30] = 0.005
    constants[31] = 0.06
    constants[32] = 0.08
    constants[33] = 1
    constants[34] = 0.1
    constants[35] = 10
    states[10] = 0
    states[11] = 0
    constants[36] = 0.04
    constants[37] = 2
    constants[38] = 0
    constants[39] = 0.64
    constants[40] = 0.05
    constants[41] = 0.05
    constants[42] = 1
    constants[43] = 0.5
    constants[44] = 1
    constants[45] = 0.005776
    constants[46] = 2
    constants[47] = 0.06
    constants[48] = 0.4
    states[12] = 0
    constants[49] = 0.3
    constants[50] = 0.06
    constants[51] = 1
    constants[52] = 0.08
    constants[53] = 0.06
    constants[54] = 0.2
    constants[55] = 1
    constants[56] = 0.5
    constants[57] = 6
    constants[58] = 0.01
    constants[59] = 0.01
    constants[60] = 75
    constants[61] = 0.5
    constants[62] = 1
    constants[63] = 0.15
    constants[64] = 1
    constants[65] = 1
    constants[66] = 1
    constants[67] = 0.3
    constants[68] = 0.2
    constants[69] = 0.1
    constants[70] = 0.1
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[1] = constants[45]*states[1]
    algebraic[1] = custom_piecewise([greater_equal(states[10] , 1.00000), constants[35] , greater_equal(states[11] , 1.00000), constants[34] , True, 0.100000])
    rates[4] = constants[33]*(states[9]-states[4])-states[4]*(algebraic[1]+constants[32])
    algebraic[2] = states[2]-states[5]
    rates[9] = (constants[30]+constants[31]*algebraic[2])-constants[32]*states[9]
    rates[11] = (constants[52]*algebraic[2])/(constants[54]+algebraic[2])-constants[53]*states[11]
    algebraic[0] = constants[6]*(constants[3]-states[3])+constants[7]*states[3]+constants[8]*states[4]
    algebraic[3] = (2.00000*constants[62]*algebraic[2]*constants[64])/(((constants[63]+constants[62]*algebraic[2]*constants[64]+constants[63]*constants[65])-constants[62]*algebraic[2])+power(power((constants[63]+constants[62]*algebraic[2]*constants[64]+constants[63]*constants[65])-constants[62]*algebraic[2], 2.00000)-4.00000*(constants[63]-constants[62]*algebraic[2])*constants[62]*algebraic[2]*constants[64], 1.0/2))
    rates[2] = states[1]*(constants[4]+constants[5]*algebraic[3])-algebraic[0]*states[2]
    algebraic[4] = states[7]-states[6]
    rates[10] = constants[46]*(algebraic[4]+constants[48]*algebraic[2])-constants[47]*states[10]
    algebraic[8] = (constants[16]*constants[15]*states[1])/(constants[14]+constants[15]*states[1])
    rates[12] = constants[49]*(states[0]+algebraic[8]+constants[51]*algebraic[4])-constants[50]*states[12]
    algebraic[5] = states[8]-(states[5]+states[6])
    algebraic[7] = constants[13]*states[1]
    algebraic[10] = constants[25]*(constants[26]*algebraic[8]+constants[27]*algebraic[7]+states[0]+constants[28]*algebraic[4]+constants[29]*algebraic[2])
    rates[5] = constants[21]*algebraic[2]*algebraic[5]-states[5]*(constants[22]+algebraic[0]+constants[19]+algebraic[10]/(constants[20]+states[8]))
    algebraic[6] = constants[11]+constants[12]*states[4]
    rates[6] = constants[23]*algebraic[4]*algebraic[5]-states[6]*(constants[24]+algebraic[6]+constants[19]+algebraic[10]/(constants[20]+states[8]))
    algebraic[11] = constants[38]+constants[39]*(algebraic[8]+constants[42]*states[0]+constants[43]*algebraic[4]+constants[44]*algebraic[2])
    rates[3] = ((constants[36]+constants[37]*states[4])*(constants[3]-states[3]))/((constants[41]+constants[3])-states[3])-(algebraic[11]*states[3])/(constants[40]+states[3])
    algebraic[9] = constants[55]*(states[0]+constants[60]*(algebraic[8]+algebraic[7])+constants[61]*algebraic[4])
    algebraic[12] = (2.00000*algebraic[9]*constants[58])/(((constants[56]+constants[57]*algebraic[2]+algebraic[9]*constants[58]+(constants[56]+constants[57]*algebraic[2])*constants[59])-algebraic[9])+power(power((constants[56]+constants[57]*algebraic[2]+algebraic[9]*constants[58]+(constants[56]+constants[57]*algebraic[2])*constants[59])-algebraic[9], 2.00000)-4.00000*algebraic[9]*constants[58]*((constants[56]+constants[57]*algebraic[2])-algebraic[9]), 1.0/2))
    rates[0] = states[1]*(constants[0]+constants[1]*algebraic[12])-constants[2]*states[0]
    algebraic[13] = (2.00000*constants[66]*states[4]*constants[69])/(((constants[67]+constants[68]*algebraic[2]+constants[66]*states[4]*constants[69]+(constants[67]+constants[68]*algebraic[2])*constants[70])-constants[66]*states[4])+power(power((constants[67]+constants[68]*algebraic[2]+constants[66]*states[4]*constants[69]+(constants[67]+constants[68]*algebraic[2])*constants[70])-constants[66]*states[4], 2.00000)-4.00000*((constants[67]+constants[68]*algebraic[2])-constants[66]*states[4])*constants[66]*states[4]*constants[69], 1.0/2))
    rates[8] = (constants[17]+constants[18]*algebraic[13])-states[8]*(constants[19]+algebraic[10]/(constants[20]+states[8]))
    algebraic[14] = algebraic[12]
    rates[7] = states[1]*(constants[9]+constants[10]*algebraic[14])-algebraic[6]*states[7]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[1] = custom_piecewise([greater_equal(states[10] , 1.00000), constants[35] , greater_equal(states[11] , 1.00000), constants[34] , True, 0.100000])
    algebraic[2] = states[2]-states[5]
    algebraic[0] = constants[6]*(constants[3]-states[3])+constants[7]*states[3]+constants[8]*states[4]
    algebraic[3] = (2.00000*constants[62]*algebraic[2]*constants[64])/(((constants[63]+constants[62]*algebraic[2]*constants[64]+constants[63]*constants[65])-constants[62]*algebraic[2])+power(power((constants[63]+constants[62]*algebraic[2]*constants[64]+constants[63]*constants[65])-constants[62]*algebraic[2], 2.00000)-4.00000*(constants[63]-constants[62]*algebraic[2])*constants[62]*algebraic[2]*constants[64], 1.0/2))
    algebraic[4] = states[7]-states[6]
    algebraic[8] = (constants[16]*constants[15]*states[1])/(constants[14]+constants[15]*states[1])
    algebraic[5] = states[8]-(states[5]+states[6])
    algebraic[7] = constants[13]*states[1]
    algebraic[10] = constants[25]*(constants[26]*algebraic[8]+constants[27]*algebraic[7]+states[0]+constants[28]*algebraic[4]+constants[29]*algebraic[2])
    algebraic[6] = constants[11]+constants[12]*states[4]
    algebraic[11] = constants[38]+constants[39]*(algebraic[8]+constants[42]*states[0]+constants[43]*algebraic[4]+constants[44]*algebraic[2])
    algebraic[9] = constants[55]*(states[0]+constants[60]*(algebraic[8]+algebraic[7])+constants[61]*algebraic[4])
    algebraic[12] = (2.00000*algebraic[9]*constants[58])/(((constants[56]+constants[57]*algebraic[2]+algebraic[9]*constants[58]+(constants[56]+constants[57]*algebraic[2])*constants[59])-algebraic[9])+power(power((constants[56]+constants[57]*algebraic[2]+algebraic[9]*constants[58]+(constants[56]+constants[57]*algebraic[2])*constants[59])-algebraic[9], 2.00000)-4.00000*algebraic[9]*constants[58]*((constants[56]+constants[57]*algebraic[2])-algebraic[9]), 1.0/2))
    algebraic[13] = (2.00000*constants[66]*states[4]*constants[69])/(((constants[67]+constants[68]*algebraic[2]+constants[66]*states[4]*constants[69]+(constants[67]+constants[68]*algebraic[2])*constants[70])-constants[66]*states[4])+power(power((constants[67]+constants[68]*algebraic[2]+constants[66]*states[4]*constants[69]+(constants[67]+constants[68]*algebraic[2])*constants[70])-constants[66]*states[4], 2.00000)-4.00000*((constants[67]+constants[68]*algebraic[2])-constants[66]*states[4])*constants[66]*states[4]*constants[69], 1.0/2))
    algebraic[14] = algebraic[12]
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