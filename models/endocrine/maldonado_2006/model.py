# Size of variable arrays:
sizeAlgebraic = 6
sizeStates = 9
sizeConstants = 51
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (day)"
    legend_algebraic[0] = "A_B in component A_B (mm2)"
    legend_states[0] = "r_B in component r_B (mm)"
    legend_constants[0] = "X_c in component r_B (pM)"
    legend_constants[1] = "X_b in component r_B (pM)"
    legend_constants[2] = "X_y in component r_B (pM)"
    legend_constants[3] = "k_rB in component r_B (first_order_rate_constant)"
    legend_constants[4] = "k_for in component r_B (mm_per_day)"
    legend_constants[5] = "k_res in component r_B (mm_per_day)"
    legend_states[1] = "xc in component xc (pM)"
    legend_states[2] = "xb in component xb (pM)"
    legend_states[3] = "xy in component xy (pM)"
    legend_constants[6] = "X_bss in component xy (pM)"
    legend_constants[7] = "X_yss in component xy (pM)"
    legend_constants[8] = "k_byp in component xy (first_order_rate_constant)"
    legend_constants[9] = "k_yd in component xy (first_order_rate_constant)"
    legend_algebraic[5] = "F_sti in component F_sti (pM_N_per_mm2)"
    legend_constants[10] = "k_Fs in component F_sti (mm2_per_N)"
    legend_constants[11] = "k_y in component F_sti (per_pM)"
    legend_algebraic[4] = "F_s in component F_s (N_per_mm2)"
    legend_algebraic[2] = "F_a in component F_s (newton)"
    legend_states[4] = "x_no in component x_no (pM)"
    legend_constants[12] = "X_noe in component x_no (flux)"
    legend_constants[13] = "k_yno in component x_no (mm2_day_per_dyn)"
    legend_constants[14] = "k_nod in component x_no (first_order_rate_constant)"
    legend_states[5] = "x_pge in component x_pge (pM)"
    legend_constants[15] = "X_pgex in component x_pge (flux)"
    legend_constants[16] = "k_pged in component x_pge (first_order_rate_constant)"
    legend_constants[17] = "k_nopge in component x_pge (first_order_rate_constant)"
    legend_constants[18] = "k_ypge in component x_pge (mm2_day_per_dyn)"
    legend_states[6] = "x_opg in component x_opg (pM)"
    legend_constants[19] = "k_opgd in component x_opg (first_order_rate_constant)"
    legend_constants[20] = "k_nopg in component x_opg (first_order_rate_constant)"
    legend_constants[21] = "Io in component model_parameters (flux)"
    legend_constants[22] = "K_o_p in component model_parameters (picomole_per_day_picomole_cells)"
    legend_states[7] = "xr in component xr (pM)"
    legend_algebraic[3] = "pi_c in component model_parameters (dimensionless)"
    legend_states[8] = "x_kl in component x_kl (pM)"
    legend_algebraic[1] = "pi_L in component x_kl (dimensionless)"
    legend_constants[23] = "K_l_p in component x_kl (dimensionless)"
    legend_constants[24] = "K in component x_kl (pM)"
    legend_constants[25] = "k_nokl in component x_kl (first_order_rate_constant)"
    legend_constants[26] = "Il in component x_kl (flux)"
    legend_constants[27] = "rl in component x_kl (flux)"
    legend_constants[28] = "k1 in component x_kl (second_order_rate_constant)"
    legend_constants[29] = "k2 in component x_kl (first_order_rate_constant)"
    legend_constants[30] = "k3 in component x_kl (second_order_rate_constant)"
    legend_constants[31] = "k4 in component x_kl (first_order_rate_constant)"
    legend_constants[32] = "ko in component x_kl (first_order_rate_constant)"
    legend_constants[50] = "pi_p in component model_parameters (dimensionless)"
    legend_constants[33] = "D_R in component xr (flux)"
    legend_constants[34] = "k_pger in component xr (first_order_rate_constant)"
    legend_constants[46] = "D_B in component model_parameters (first_order_rate_constant)"
    legend_constants[35] = "k_B in component xb (first_order_rate_constant)"
    legend_constants[36] = "D_C in component xc (flux)"
    legend_constants[37] = "D_A in component xc (first_order_rate_constant)"
    legend_constants[38] = "f0 in component model_parameters (dimensionless)"
    legend_constants[39] = "dB in component model_parameters (first_order_rate_constant)"
    legend_constants[40] = "C_s in component model_parameters (pM)"
    legend_constants[47] = "P in component model_parameters (pM)"
    legend_constants[48] = "P_0 in component model_parameters (pM)"
    legend_constants[49] = "P_s in component model_parameters (pM)"
    legend_constants[41] = "SP in component model_parameters (flux)"
    legend_constants[42] = "k5 in component model_parameters (second_order_rate_constant)"
    legend_constants[43] = "k6 in component model_parameters (first_order_rate_constant)"
    legend_constants[44] = "IP in component model_parameters (flux)"
    legend_constants[45] = "kP in component model_parameters (first_order_rate_constant)"
    legend_rates[0] = "d/dt r_B in component r_B (mm)"
    legend_rates[3] = "d/dt xy in component xy (pM)"
    legend_rates[4] = "d/dt x_no in component x_no (pM)"
    legend_rates[5] = "d/dt x_pge in component x_pge (pM)"
    legend_rates[6] = "d/dt x_opg in component x_opg (pM)"
    legend_rates[8] = "d/dt x_kl in component x_kl (pM)"
    legend_rates[7] = "d/dt xr in component xr (pM)"
    legend_rates[2] = "d/dt xb in component xb (pM)"
    legend_rates[1] = "d/dt xc in component xc (pM)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.9912
    constants[0] = 9.127e-4
    constants[1] = 7.282e-4
    constants[2] = 7.300e-3
    constants[3] = 1.0e0
    constants[4] = 1.0e-3
    constants[5] = 10.0e-3
    states[1] = 9.127e-4
    states[2] = 7.282e-4
    states[3] = 7.300e-3
    constants[6] = 7.282e-4
    constants[7] = 7.300e-3
    constants[8] = 1.00e-1
    constants[9] = 1.00e0
    constants[10] = 1.00e0
    constants[11] = 1.00e0
    states[4] = 0.02
    constants[12] = 0e0
    constants[13] = 2e4
    constants[14] = 1e3
    states[5] = 0.01
    constants[15] = 0e0
    constants[16] = 1e2
    constants[17] = 1e0
    constants[18] = 1e2
    states[6] = 0.01
    constants[19] = 3.5e-1
    constants[20] = 1e1
    constants[21] = 0.0
    constants[22] = 2e5
    states[7] = 7.734e-4
    states[8] = 0.01
    constants[23] = 3e6
    constants[24] = 10
    constants[25] = 1e2
    constants[26] = 0
    constants[27] = 1e3
    constants[28] = 1e-2
    constants[29] = 10
    constants[30] = 5.8e-4
    constants[31] = 1.7e-2
    constants[32] = 0.35
    constants[33] = 7e-4
    constants[34] = 1e-4
    constants[35] = 0.189
    constants[36] = 0.189
    constants[37] = 0.7
    constants[38] = 0.05
    constants[39] = 0.7
    constants[40] = 5e-3
    constants[41] = 250
    constants[42] = 0.02
    constants[43] = 3
    constants[44] = 0
    constants[45] = 86
    constants[46] = constants[38]*constants[39]
    constants[47] = constants[44]/constants[45]
    constants[48] = constants[41]/constants[45]
    constants[49] = constants[43]/constants[42]
    constants[50] = (constants[47]+constants[48])/(constants[47]+constants[49])
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = ((constants[4]/constants[1])*states[2]+(1.00000/constants[2])*states[3])-((constants[5]/constants[0])*states[1]+constants[3]*states[0])
    rates[3] = constants[8]*(states[2]-constants[6])-constants[9]*(states[3]-constants[7])
    rates[8] = (constants[27]+constants[26])-(constants[25]*states[4]+constants[27]*((1.00000+(constants[28]/constants[29])*states[6]+(constants[30]/constants[31])*constants[24])/(constants[23]*constants[50]*states[2]))*states[8])
    algebraic[3] = (states[1]+constants[38]*constants[40])/(states[1]+constants[40])
    rates[6] = ((constants[22]/1.00000)*algebraic[3]*states[7]+constants[21]+constants[20]*states[4])-constants[19]*states[6]
    rates[7] = (constants[33]*algebraic[3]+constants[34]*states[5])-(constants[46]/algebraic[3])*states[7]
    rates[2] = (constants[46]/algebraic[3])*states[7]-constants[35]*states[2]
    algebraic[1] = ((((constants[30]/constants[31])*constants[23])*constants[50]*states[2])/(1.00000+(constants[30]*constants[24])/constants[31]+(constants[28]/(constants[29]*constants[32]))*(((constants[22]/1.00000)/constants[50])*states[7]+constants[21])))*(1.00000+constants[26]/constants[27])
    rates[1] = constants[36]*algebraic[1]-constants[37]*algebraic[3]*states[1]
    algebraic[0] = (power(states[0], 2.00000))* pi
    algebraic[2] = custom_piecewise([greater_equal(voi , 100.000) & less(voi , 130.000), 10000.0 , True, 100.000])
    algebraic[4] = algebraic[2]/algebraic[0]
    algebraic[5] = (algebraic[4]*states[3])/(1.00000+exp(-(constants[10]*algebraic[4]+constants[11]*states[3])))
    rates[4] = (constants[13]*algebraic[5]+constants[12])-constants[14]*states[4]
    rates[5] = (constants[18]*algebraic[5]+constants[17]*states[4]+constants[15])-constants[16]*states[5]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[3] = (states[1]+constants[38]*constants[40])/(states[1]+constants[40])
    algebraic[1] = ((((constants[30]/constants[31])*constants[23])*constants[50]*states[2])/(1.00000+(constants[30]*constants[24])/constants[31]+(constants[28]/(constants[29]*constants[32]))*(((constants[22]/1.00000)/constants[50])*states[7]+constants[21])))*(1.00000+constants[26]/constants[27])
    algebraic[0] = (power(states[0], 2.00000))* pi
    algebraic[2] = custom_piecewise([greater_equal(voi , 100.000) & less(voi , 130.000), 10000.0 , True, 100.000])
    algebraic[4] = algebraic[2]/algebraic[0]
    algebraic[5] = (algebraic[4]*states[3])/(1.00000+exp(-(constants[10]*algebraic[4]+constants[11]*states[3])))
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