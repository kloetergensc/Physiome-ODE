# Size of variable arrays:
sizeAlgebraic = 33
sizeStates = 7
sizeConstants = 47
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component Environment (ms)"
    legend_states[0] = "NADHm in component Mitochondrial_variables (mM)"
    legend_algebraic[15] = "Jpdh in component J_variables (uM_per_ms)"
    legend_algebraic[8] = "Jo in component J_variables (uM_per_ms)"
    legend_constants[0] = "gamma in component Mitochondrial_variables (dimensionless)"
    legend_algebraic[0] = "NADm in component Mitochondrial_variables (mM)"
    legend_states[1] = "ADPm in component Mitochondrial_variables (mM)"
    legend_states[2] = "PSIm in component Mitochondrial_variables (mV)"
    legend_states[3] = "Cam in component Mitochondrial_variables (uM)"
    legend_constants[1] = "NADtot in component Mitochondrial_variables (mM)"
    legend_constants[2] = "fm in component Mitochondrial_variables (dimensionless)"
    legend_constants[3] = "Cmito in component Mitochondrial_variables (uM_per_mV)"
    legend_constants[4] = "Amtot in component Mitochondrial_variables (mM)"
    legend_algebraic[1] = "ATPm in component Mitochondrial_variables (mM)"
    legend_algebraic[5] = "RATm in component Mitochondrial_variables (dimensionless)"
    legend_algebraic[32] = "Jant in component J_variables (uM_per_ms)"
    legend_algebraic[31] = "Jf1f0 in component J_variables (uM_per_ms)"
    legend_algebraic[25] = "Jh_res in component J_variables (uM_per_ms)"
    legend_algebraic[29] = "Jh_atp in component J_variables (uM_per_ms)"
    legend_algebraic[20] = "Jh_leak in component J_variables (uM_per_ms)"
    legend_algebraic[18] = "Jnaca in component J_variables (uM_per_ms)"
    legend_algebraic[16] = "Juni in component J_variables (uM_per_ms)"
    legend_algebraic[21] = "Jmito in component J_variables (uM_per_ms)"
    legend_states[4] = "c in component Clamp_protocol (uM)"
    legend_algebraic[9] = "ATPc in component Cytosol (mM)"
    legend_states[5] = "ADPc in component Cytosol (uM)"
    legend_constants[5] = "Ac_tot in component Cytosol (uM)"
    legend_constants[6] = "khyd in component Cytosol (dimensionless)"
    legend_constants[7] = "Jhydbas in component Cytosol (mM)"
    legend_algebraic[13] = "Jh_yd in component Cytosol (mM)"
    legend_constants[46] = "delta in component Cytosol (dimensionless)"
    legend_algebraic[10] = "Fproto in component Clamp_parameters (uM)"
    legend_algebraic[28] = "Cproto in component Clamp_parameters (uM)"
    legend_states[6] = "FBP in component Clamp_protocol (uM)"
    legend_constants[8] = "Fhold in component Clamp_parameters (uM)"
    legend_constants[9] = "Ftest in component Clamp_parameters (uM)"
    legend_constants[10] = "Fton in component Clamp_parameters (ms)"
    legend_constants[11] = "Ftoff in component Clamp_parameters (ms)"
    legend_constants[12] = "Chold in component Clamp_parameters (uM)"
    legend_constants[13] = "Ctest in component Clamp_parameters (uM)"
    legend_constants[14] = "Cton1 in component Clamp_parameters (ms)"
    legend_constants[15] = "Cton2 in component Clamp_parameters (ms)"
    legend_constants[16] = "Cton3 in component Clamp_parameters (ms)"
    legend_constants[17] = "Ctoff1 in component Clamp_parameters (ms)"
    legend_constants[18] = "Ctoff2 in component Clamp_parameters (ms)"
    legend_constants[19] = "Ctoff3 in component Clamp_parameters (ms)"
    legend_algebraic[17] = "pulse1 in component Clamp_parameters (uM)"
    legend_algebraic[22] = "pulse2 in component Clamp_parameters (uM)"
    legend_algebraic[26] = "pulse3 in component Clamp_parameters (uM)"
    legend_algebraic[2] = "heav_on in component Clamp_parameters (dimensionless)"
    legend_algebraic[6] = "heav_off in component Clamp_parameters (dimensionless)"
    legend_algebraic[3] = "heav_Cton1 in component Clamp_parameters (dimensionless)"
    legend_algebraic[7] = "heav_Cton2 in component Clamp_parameters (dimensionless)"
    legend_algebraic[11] = "heav_Cton3 in component Clamp_parameters (dimensionless)"
    legend_algebraic[14] = "heav_Ctoff1 in component Clamp_parameters (dimensionless)"
    legend_algebraic[19] = "heav_Ctoff2 in component Clamp_parameters (dimensionless)"
    legend_algebraic[24] = "heav_Ctoff3 in component Clamp_parameters (dimensionless)"
    legend_algebraic[12] = "Jgpdh in component J_variables (uM_per_ms)"
    legend_constants[20] = "p1 in component Parameters (dimensionless)"
    legend_constants[21] = "p2 in component Parameters (dimensionless)"
    legend_constants[22] = "p3 in component Parameters (uM)"
    legend_constants[23] = "p21 in component Parameters (per_uM_per_ms_per_mV)"
    legend_constants[24] = "p22 in component Parameters (per_uM_per_ms)"
    legend_constants[25] = "p23 in component Parameters (uM_per_ms)"
    legend_constants[26] = "p24 in component Parameters (per_mV)"
    legend_constants[27] = "p4 in component Parameters (uM_per_ms)"
    legend_constants[28] = "p5 in component Parameters (mM)"
    legend_constants[29] = "p6 in component Parameters (mV)"
    legend_constants[30] = "p7 in component Parameters (mV)"
    legend_constants[31] = "p8 in component Parameters (uM_per_ms)"
    legend_constants[32] = "p9 in component Parameters (mM)"
    legend_constants[33] = "p10 in component Parameters (mV)"
    legend_constants[34] = "p11 in component Parameters (mV)"
    legend_constants[35] = "p12 in component Parameters (uM_per_ms)"
    legend_constants[36] = "p13 in component Parameters (mM)"
    legend_constants[37] = "p14 in component Parameters (mV)"
    legend_constants[38] = "p15 in component Parameters (mV)"
    legend_constants[39] = "p16 in component Parameters (uM_per_ms)"
    legend_constants[40] = "p17 in component Parameters (uM_per_ms_per_mV)"
    legend_constants[41] = "p18 in component Parameters (uM_per_ms)"
    legend_constants[42] = "p19 in component Parameters (uM_per_ms)"
    legend_constants[43] = "p20 in component Parameters (dimensionless)"
    legend_algebraic[4] = "MM1 in component J_variables (uM_per_ms)"
    legend_algebraic[23] = "MM2 in component J_variables (uM_per_ms)"
    legend_algebraic[27] = "b13 in component J_variables (uM_per_ms)"
    legend_algebraic[30] = "b2 in component J_variables (uM_per_ms)"
    legend_constants[44] = "FRT in component J_variables (per_mV)"
    legend_constants[45] = "kgpdh in component J_variables (uM_per_ms)"
    legend_rates[0] = "d/dt NADHm in component Mitochondrial_variables (mM)"
    legend_rates[1] = "d/dt ADPm in component Mitochondrial_variables (mM)"
    legend_rates[2] = "d/dt PSIm in component Mitochondrial_variables (mV)"
    legend_rates[3] = "d/dt Cam in component Mitochondrial_variables (uM)"
    legend_rates[5] = "d/dt ADPc in component Cytosol (uM)"
    legend_rates[6] = "d/dt FBP in component Clamp_protocol (uM)"
    legend_rates[4] = "d/dt c in component Clamp_protocol (uM)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.6
    constants[0] = 0.001
    states[1] = 7.4
    states[2] = 93
    states[3] = 0.1
    constants[1] = 10
    constants[2] = 0.01
    constants[3] = 1.8
    constants[4] = 15
    states[4] = 0.1
    states[5] = 1850
    constants[5] = 2500
    constants[6] = 0.00005
    constants[7] = 0.00005
    states[6] = 0.5
    constants[8] = 1
    constants[9] = 5
    constants[10] = 90000
    constants[11] = 330000
    constants[12] = 0.1
    constants[13] = 0.1
    constants[14] = 120000
    constants[15] = 180000
    constants[16] = 240000
    constants[17] = 150000
    constants[18] = 210000
    constants[19] = 270000
    constants[20] = 400
    constants[21] = 1
    constants[22] = 0.01
    constants[23] = 0.01
    constants[24] = 1.1
    constants[25] = 0.001
    constants[26] = 0.016
    constants[27] = 0.6
    constants[28] = 0.1
    constants[29] = 177
    constants[30] = 5
    constants[31] = 7
    constants[32] = 0.1
    constants[33] = 177
    constants[34] = 5
    constants[35] = 120
    constants[36] = 10
    constants[37] = 190
    constants[38] = 8.5
    constants[39] = 35
    constants[40] = 0.002
    constants[41] = -0.03
    constants[42] = 0.35
    constants[43] = 2
    constants[44] = 0.037
    constants[45] = 0.0005
    constants[46] = 3.90000/53.2000
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[2] = custom_piecewise([greater_equal(voi-constants[10] , 0.00000), 1.00000 , True, 0.00000])
    algebraic[6] = custom_piecewise([greater_equal(voi-constants[11] , 0.00000), 1.00000 , True, 0.00000])
    algebraic[10] = constants[8]+(constants[9]-constants[8])*(algebraic[2]-algebraic[6])
    rates[6] = (algebraic[10]-states[6])/0.100000
    algebraic[0] = constants[1]-states[0]
    algebraic[12] = constants[45]*(power(states[6]/1.00000, 1.0/2))
    algebraic[15] = (constants[20]/(constants[21]+states[0]/algebraic[0]))*(states[3]/(constants[22]+states[3]))*algebraic[12]
    algebraic[4] = (constants[27]*states[0])/(constants[28]+states[0])
    algebraic[8] = algebraic[4]/(1.00000+exp((states[2]-constants[29])/constants[30]))
    rates[0] = constants[0]*(algebraic[15]-algebraic[8])
    algebraic[18] = ((constants[25]*states[3])/states[4])*exp(constants[26]*states[2])
    algebraic[16] = (constants[23]*states[2]-constants[24])*(power(states[4], 2.00000))
    algebraic[21] = algebraic[18]-algebraic[16]
    rates[3] = -constants[2]*algebraic[21]
    algebraic[3] = custom_piecewise([greater_equal(voi-constants[14] , 0.00000), 1.00000 , True, 0.00000])
    algebraic[14] = custom_piecewise([greater_equal(voi-constants[17] , 0.00000), 1.00000 , True, 0.00000])
    algebraic[17] = (constants[13]-constants[12])*(algebraic[3]-algebraic[14])
    algebraic[7] = custom_piecewise([greater_equal(voi-constants[15] , 0.00000), 1.00000 , True, 0.00000])
    algebraic[19] = custom_piecewise([greater_equal(voi-constants[18] , 0.00000), 1.00000 , True, 0.00000])
    algebraic[22] = (constants[13]-constants[12])*(algebraic[7]-algebraic[19])
    algebraic[11] = custom_piecewise([greater_equal(voi-constants[16] , 0.00000), 1.00000 , True, 0.00000])
    algebraic[24] = custom_piecewise([greater_equal(voi-constants[19] , 0.00000), 1.00000 , True, 0.00000])
    algebraic[26] = (constants[13]-constants[12])*(algebraic[11]-algebraic[24])
    algebraic[28] = constants[12]+algebraic[17]+algebraic[22]+algebraic[26]
    rates[4] = (algebraic[28]-states[4])/0.100000
    algebraic[1] = constants[4]-states[1]
    algebraic[5] = algebraic[1]/states[1]
    algebraic[32] = ((constants[42]*algebraic[5])/(algebraic[5]+constants[43]))/exp(-0.500000*constants[44]*states[2])
    algebraic[30] = (constants[39]*constants[36])/(constants[36]+algebraic[1])
    algebraic[31] = algebraic[30]/(1.00000+exp((constants[37]-states[2])/constants[38]))
    rates[1] = constants[0]*(algebraic[32]-algebraic[31])
    algebraic[23] = (constants[31]*states[0])/(constants[32]+states[0])
    algebraic[25] = algebraic[23]/(1.00000+exp((states[2]-constants[33])/constants[34]))
    algebraic[27] = (constants[35]*constants[36])/(constants[36]+algebraic[1])
    algebraic[29] = algebraic[27]/(1.00000+exp((constants[37]-states[2])/constants[38]))
    algebraic[20] = constants[40]*states[2]+constants[41]
    rates[2] = (((((algebraic[25]-algebraic[29])-algebraic[32])-algebraic[20])-algebraic[18])-algebraic[16]*2.00000)/constants[3]
    algebraic[9] = constants[5]-states[5]
    algebraic[13] = ((constants[6]*states[4]+constants[7])*algebraic[9])/1.00000
    rates[5] = -constants[46]*algebraic[32]+algebraic[13]*1.00000
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[2] = custom_piecewise([greater_equal(voi-constants[10] , 0.00000), 1.00000 , True, 0.00000])
    algebraic[6] = custom_piecewise([greater_equal(voi-constants[11] , 0.00000), 1.00000 , True, 0.00000])
    algebraic[10] = constants[8]+(constants[9]-constants[8])*(algebraic[2]-algebraic[6])
    algebraic[0] = constants[1]-states[0]
    algebraic[12] = constants[45]*(power(states[6]/1.00000, 1.0/2))
    algebraic[15] = (constants[20]/(constants[21]+states[0]/algebraic[0]))*(states[3]/(constants[22]+states[3]))*algebraic[12]
    algebraic[4] = (constants[27]*states[0])/(constants[28]+states[0])
    algebraic[8] = algebraic[4]/(1.00000+exp((states[2]-constants[29])/constants[30]))
    algebraic[18] = ((constants[25]*states[3])/states[4])*exp(constants[26]*states[2])
    algebraic[16] = (constants[23]*states[2]-constants[24])*(power(states[4], 2.00000))
    algebraic[21] = algebraic[18]-algebraic[16]
    algebraic[3] = custom_piecewise([greater_equal(voi-constants[14] , 0.00000), 1.00000 , True, 0.00000])
    algebraic[14] = custom_piecewise([greater_equal(voi-constants[17] , 0.00000), 1.00000 , True, 0.00000])
    algebraic[17] = (constants[13]-constants[12])*(algebraic[3]-algebraic[14])
    algebraic[7] = custom_piecewise([greater_equal(voi-constants[15] , 0.00000), 1.00000 , True, 0.00000])
    algebraic[19] = custom_piecewise([greater_equal(voi-constants[18] , 0.00000), 1.00000 , True, 0.00000])
    algebraic[22] = (constants[13]-constants[12])*(algebraic[7]-algebraic[19])
    algebraic[11] = custom_piecewise([greater_equal(voi-constants[16] , 0.00000), 1.00000 , True, 0.00000])
    algebraic[24] = custom_piecewise([greater_equal(voi-constants[19] , 0.00000), 1.00000 , True, 0.00000])
    algebraic[26] = (constants[13]-constants[12])*(algebraic[11]-algebraic[24])
    algebraic[28] = constants[12]+algebraic[17]+algebraic[22]+algebraic[26]
    algebraic[1] = constants[4]-states[1]
    algebraic[5] = algebraic[1]/states[1]
    algebraic[32] = ((constants[42]*algebraic[5])/(algebraic[5]+constants[43]))/exp(-0.500000*constants[44]*states[2])
    algebraic[30] = (constants[39]*constants[36])/(constants[36]+algebraic[1])
    algebraic[31] = algebraic[30]/(1.00000+exp((constants[37]-states[2])/constants[38]))
    algebraic[23] = (constants[31]*states[0])/(constants[32]+states[0])
    algebraic[25] = algebraic[23]/(1.00000+exp((states[2]-constants[33])/constants[34]))
    algebraic[27] = (constants[35]*constants[36])/(constants[36]+algebraic[1])
    algebraic[29] = algebraic[27]/(1.00000+exp((constants[37]-states[2])/constants[38]))
    algebraic[20] = constants[40]*states[2]+constants[41]
    algebraic[9] = constants[5]-states[5]
    algebraic[13] = ((constants[6]*states[4]+constants[7])*algebraic[9])/1.00000
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