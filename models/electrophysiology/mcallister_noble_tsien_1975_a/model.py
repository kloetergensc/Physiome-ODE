# Size of variable arrays:
sizeAlgebraic = 31
sizeStates = 10
sizeConstants = 13
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (millisecond)"
    legend_states[0] = "V in component membrane (millivolt)"
    legend_constants[0] = "C in component membrane (microF_per_cm2)"
    legend_algebraic[0] = "i_Na in component fast_sodium_current (microA_per_cm2)"
    legend_algebraic[20] = "i_si in component secondary_inward_current (microA_per_cm2)"
    legend_algebraic[22] = "i_K2 in component pacemaker_potassium_current (microA_per_cm2)"
    legend_algebraic[24] = "i_x1 in component plateau_potassium_current1 (microA_per_cm2)"
    legend_algebraic[26] = "i_x2 in component plateau_potassium_current2 (microA_per_cm2)"
    legend_algebraic[27] = "i_qr in component transient_chloride_current (microA_per_cm2)"
    legend_algebraic[28] = "i_K1 in component time_independent_outward_current (microA_per_cm2)"
    legend_algebraic[29] = "i_Na_b in component sodium_background_current (microA_per_cm2)"
    legend_algebraic[30] = "i_Cl_b in component chloride_background_current (microA_per_cm2)"
    legend_constants[1] = "E_Na in component fast_sodium_current (millivolt)"
    legend_constants[2] = "g_Na in component fast_sodium_current (milliS_per_cm2)"
    legend_states[1] = "m in component fast_sodium_current_m_gate (dimensionless)"
    legend_states[2] = "h in component fast_sodium_current_h_gate (dimensionless)"
    legend_algebraic[1] = "alpha_m in component fast_sodium_current_m_gate (per_millisecond)"
    legend_algebraic[11] = "beta_m in component fast_sodium_current_m_gate (per_millisecond)"
    legend_algebraic[2] = "alpha_h in component fast_sodium_current_h_gate (per_millisecond)"
    legend_algebraic[12] = "beta_h in component fast_sodium_current_h_gate (per_millisecond)"
    legend_constants[3] = "g_si in component secondary_inward_current (milliS_per_cm2)"
    legend_constants[4] = "g_si_ in component secondary_inward_current (milliS_per_cm2)"
    legend_constants[5] = "E_si in component secondary_inward_current (millivolt)"
    legend_states[3] = "d in component secondary_inward_current_d_gate (dimensionless)"
    legend_states[4] = "f in component secondary_inward_current_f_gate (dimensionless)"
    legend_algebraic[10] = "d1 in component secondary_inward_current_d1_gate (dimensionless)"
    legend_algebraic[3] = "alpha_d in component secondary_inward_current_d_gate (per_millisecond)"
    legend_algebraic[13] = "beta_d in component secondary_inward_current_d_gate (per_millisecond)"
    legend_algebraic[4] = "alpha_f in component secondary_inward_current_f_gate (per_millisecond)"
    legend_algebraic[14] = "beta_f in component secondary_inward_current_f_gate (per_millisecond)"
    legend_algebraic[21] = "I_K2 in component pacemaker_potassium_current (microA_per_cm2)"
    legend_constants[6] = "E_K in component pacemaker_potassium_current (millivolt)"
    legend_states[5] = "s in component pacemaker_potassium_current_s_gate (dimensionless)"
    legend_algebraic[5] = "alpha_s in component pacemaker_potassium_current_s_gate (per_millisecond)"
    legend_algebraic[15] = "beta_s in component pacemaker_potassium_current_s_gate (per_millisecond)"
    legend_constants[7] = "E_s in component pacemaker_potassium_current_s_gate (millivolt)"
    legend_algebraic[23] = "I_x1 in component plateau_potassium_current1 (microA_per_cm2)"
    legend_states[6] = "x1 in component plateau_potassium_current1_x1_gate (dimensionless)"
    legend_algebraic[6] = "alpha_x1 in component plateau_potassium_current1_x1_gate (per_millisecond)"
    legend_algebraic[16] = "beta_x1 in component plateau_potassium_current1_x1_gate (per_millisecond)"
    legend_algebraic[25] = "I_x2 in component plateau_potassium_current2 (microA_per_cm2)"
    legend_states[7] = "x2 in component plateau_potassium_current2_x2_gate (dimensionless)"
    legend_algebraic[7] = "alpha_x2 in component plateau_potassium_current2_x2_gate (per_millisecond)"
    legend_algebraic[17] = "beta_x2 in component plateau_potassium_current2_x2_gate (per_millisecond)"
    legend_constants[8] = "E_Cl in component transient_chloride_current (millivolt)"
    legend_constants[9] = "g_qr in component transient_chloride_current (milliS_per_cm2)"
    legend_states[8] = "q in component transient_chloride_current_q_gate (dimensionless)"
    legend_states[9] = "r in component transient_chloride_current_r_gate (dimensionless)"
    legend_algebraic[8] = "alpha_q in component transient_chloride_current_q_gate (per_millisecond)"
    legend_algebraic[18] = "beta_q in component transient_chloride_current_q_gate (per_millisecond)"
    legend_algebraic[9] = "alpha_r in component transient_chloride_current_r_gate (per_millisecond)"
    legend_algebraic[19] = "beta_r in component transient_chloride_current_r_gate (per_millisecond)"
    legend_constants[10] = "E_K1 in component time_independent_outward_current (millivolt)"
    legend_constants[11] = "g_Nab in component sodium_background_current (milliS_per_cm2)"
    legend_constants[12] = "g_Clb in component chloride_background_current (milliS_per_cm2)"
    legend_rates[0] = "d/dt V in component membrane (millivolt)"
    legend_rates[1] = "d/dt m in component fast_sodium_current_m_gate (dimensionless)"
    legend_rates[2] = "d/dt h in component fast_sodium_current_h_gate (dimensionless)"
    legend_rates[3] = "d/dt d in component secondary_inward_current_d_gate (dimensionless)"
    legend_rates[4] = "d/dt f in component secondary_inward_current_f_gate (dimensionless)"
    legend_rates[5] = "d/dt s in component pacemaker_potassium_current_s_gate (dimensionless)"
    legend_rates[6] = "d/dt x1 in component plateau_potassium_current1_x1_gate (dimensionless)"
    legend_rates[7] = "d/dt x2 in component plateau_potassium_current2_x2_gate (dimensionless)"
    legend_rates[8] = "d/dt q in component transient_chloride_current_q_gate (dimensionless)"
    legend_rates[9] = "d/dt r in component transient_chloride_current_r_gate (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = -78.041367
    constants[0] = 10
    constants[1] = 40
    constants[2] = 150
    states[1] = 0.02566853
    states[2] = 0.78656359
    constants[3] = 0.8
    constants[4] = 0.04
    constants[5] = 70
    states[3] = 0.00293135
    states[4] = 0.80873917
    constants[6] = -110
    states[5] = 0.75473994
    constants[7] = -52
    states[6] = 0.02030289
    states[7] = 0.0176854
    constants[8] = -70
    constants[9] = 2.5
    states[8] = 3.11285794
    states[9] = 0.13500116
    constants[10] = -30
    constants[11] = 0.105
    constants[12] = 0.01
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[1] = (1.00000*(states[0]+47.0000))/(1.00000-exp(-(states[0]+47.0000)/10.0000))
    algebraic[11] = 40.0000*exp(-0.0560000*(states[0]+72.0000))
    rates[1] = algebraic[1]*(1.00000-states[1])-algebraic[11]*states[1]
    algebraic[2] = 0.00850000*exp(-0.184000*(states[0]+71.0000))
    algebraic[12] = 2.50000/(exp(-0.0820000*(states[0]+10.0000))+1.00000)
    rates[2] = algebraic[2]*(1.00000-states[2])-algebraic[12]*states[2]
    algebraic[3] = (0.00200000*(states[0]+40.0000))/(1.00000-exp(-0.100000*(states[0]+40.0000)))
    algebraic[13] = 0.0200000*exp(-0.0888000*(states[0]+40.0000))
    rates[3] = algebraic[3]*(1.00000-states[3])-algebraic[13]*states[3]
    algebraic[4] = 0.000987000*exp(-0.0400000*(states[0]+60.0000))
    algebraic[14] = 0.0200000/(exp(-0.0870000*(states[0]+26.0000))+1.00000)
    rates[4] = algebraic[4]*(1.00000-states[4])-algebraic[14]*states[4]
    algebraic[5] = (0.00100000*(states[0]-constants[7]))/(1.00000-exp(-0.200000*(states[0]-constants[7])))
    algebraic[15] = 5.00000e-05*exp(-0.0670000*(states[0]-constants[7]))
    rates[5] = algebraic[5]*(1.00000-states[5])-algebraic[15]*states[5]
    algebraic[6] = (0.000500000*exp((states[0]+50.0000)/12.1000))/(1.00000+exp((states[0]+50.0000)/17.5000))
    algebraic[16] = (0.00130000*exp(-(states[0]+20.0000)/16.6700))/(1.00000+exp(-(states[0]+20.0000)/25.0000))
    rates[6] = algebraic[6]*(1.00000-states[6])-algebraic[16]*states[6]
    algebraic[7] = (0.000127000*1.00000)/(1.00000+exp(-(states[0]+19.0000)/5.00000))
    algebraic[17] = (0.000300000*exp(-(states[0]+20.0000)/16.6700))/(1.00000+exp(-(states[0]+20.0000)/25.0000))
    rates[7] = algebraic[7]*(1.00000-states[7])-algebraic[17]*states[7]
    algebraic[8] = (0.00800000*states[0])/(1.00000-exp(-0.100000*states[0]))
    algebraic[18] = 0.0800000*exp(-0.0888000*states[0])
    rates[8] = algebraic[8]*(1.00000-states[8])-algebraic[18]*states[8]
    algebraic[9] = 0.000180000*exp(-0.0400000*(states[0]+80.0000))
    algebraic[19] = 0.0200000/(exp(-0.0870000*(states[0]+26.0000))+1.00000)
    rates[9] = algebraic[9]*(1.00000-states[9])-algebraic[19]*states[9]
    algebraic[0] = constants[2]*(power(states[1], 3.00000))*states[2]*(states[0]-constants[1])
    algebraic[10] = 1.00000/(1.00000+exp(-0.150000*(states[0]+40.0000)))
    algebraic[20] = constants[3]*states[3]*states[4]*(states[0]-constants[5])+constants[4]*algebraic[10]*(states[0]-constants[5])
    algebraic[21] = (2.80000*(exp((states[0]-constants[6])/25.0000)-1.00000))/(exp((states[0]+60.0000)/12.5000)+exp((states[0]+60.0000)/25.0000))
    algebraic[22] = algebraic[21]*states[5]
    algebraic[23] = (1.20000*(exp((states[0]+95.0000)/25.0000)-1.00000))/exp((states[0]+45.0000)/25.0000)
    algebraic[24] = states[6]*algebraic[23]
    algebraic[25] = 25.0000+1.00000*0.385000*states[0]
    algebraic[26] = states[7]*algebraic[25]
    algebraic[27] = constants[9]*states[8]*states[9]*(states[0]-constants[8])
    algebraic[28] = algebraic[21]/2.80000+(0.200000*(states[0]-constants[10]))/(1.00000-exp(-(states[0]-constants[10])/25.0000))
    algebraic[29] = constants[11]*(states[0]-constants[1])
    algebraic[30] = constants[12]*(states[0]-constants[8])
    rates[0] = -(algebraic[0]+algebraic[20]+algebraic[22]+algebraic[24]+algebraic[26]+algebraic[27]+algebraic[28]+algebraic[29]+algebraic[30])/constants[0]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[1] = (1.00000*(states[0]+47.0000))/(1.00000-exp(-(states[0]+47.0000)/10.0000))
    algebraic[11] = 40.0000*exp(-0.0560000*(states[0]+72.0000))
    algebraic[2] = 0.00850000*exp(-0.184000*(states[0]+71.0000))
    algebraic[12] = 2.50000/(exp(-0.0820000*(states[0]+10.0000))+1.00000)
    algebraic[3] = (0.00200000*(states[0]+40.0000))/(1.00000-exp(-0.100000*(states[0]+40.0000)))
    algebraic[13] = 0.0200000*exp(-0.0888000*(states[0]+40.0000))
    algebraic[4] = 0.000987000*exp(-0.0400000*(states[0]+60.0000))
    algebraic[14] = 0.0200000/(exp(-0.0870000*(states[0]+26.0000))+1.00000)
    algebraic[5] = (0.00100000*(states[0]-constants[7]))/(1.00000-exp(-0.200000*(states[0]-constants[7])))
    algebraic[15] = 5.00000e-05*exp(-0.0670000*(states[0]-constants[7]))
    algebraic[6] = (0.000500000*exp((states[0]+50.0000)/12.1000))/(1.00000+exp((states[0]+50.0000)/17.5000))
    algebraic[16] = (0.00130000*exp(-(states[0]+20.0000)/16.6700))/(1.00000+exp(-(states[0]+20.0000)/25.0000))
    algebraic[7] = (0.000127000*1.00000)/(1.00000+exp(-(states[0]+19.0000)/5.00000))
    algebraic[17] = (0.000300000*exp(-(states[0]+20.0000)/16.6700))/(1.00000+exp(-(states[0]+20.0000)/25.0000))
    algebraic[8] = (0.00800000*states[0])/(1.00000-exp(-0.100000*states[0]))
    algebraic[18] = 0.0800000*exp(-0.0888000*states[0])
    algebraic[9] = 0.000180000*exp(-0.0400000*(states[0]+80.0000))
    algebraic[19] = 0.0200000/(exp(-0.0870000*(states[0]+26.0000))+1.00000)
    algebraic[0] = constants[2]*(power(states[1], 3.00000))*states[2]*(states[0]-constants[1])
    algebraic[10] = 1.00000/(1.00000+exp(-0.150000*(states[0]+40.0000)))
    algebraic[20] = constants[3]*states[3]*states[4]*(states[0]-constants[5])+constants[4]*algebraic[10]*(states[0]-constants[5])
    algebraic[21] = (2.80000*(exp((states[0]-constants[6])/25.0000)-1.00000))/(exp((states[0]+60.0000)/12.5000)+exp((states[0]+60.0000)/25.0000))
    algebraic[22] = algebraic[21]*states[5]
    algebraic[23] = (1.20000*(exp((states[0]+95.0000)/25.0000)-1.00000))/exp((states[0]+45.0000)/25.0000)
    algebraic[24] = states[6]*algebraic[23]
    algebraic[25] = 25.0000+1.00000*0.385000*states[0]
    algebraic[26] = states[7]*algebraic[25]
    algebraic[27] = constants[9]*states[8]*states[9]*(states[0]-constants[8])
    algebraic[28] = algebraic[21]/2.80000+(0.200000*(states[0]-constants[10]))/(1.00000-exp(-(states[0]-constants[10])/25.0000))
    algebraic[29] = constants[11]*(states[0]-constants[1])
    algebraic[30] = constants[12]*(states[0]-constants[8])
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