# Size of variable arrays:
sizeAlgebraic = 17
sizeStates = 5
sizeConstants = 25
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (ms)"
    legend_algebraic[1] = "Ca_i in component intracellular_ion_concentrations (mM)"
    legend_algebraic[0] = "Ca_b in component intracellular_ion_concentrations (mM)"
    legend_states[0] = "TRPN in component intracellular_ion_concentrations (mM)"
    legend_constants[0] = "Ca_TRPN_Max in component troponin (mM)"
    legend_algebraic[16] = "J_TRPN in component troponin (mM_per_ms)"
    legend_states[1] = "z in component tropomyosin (dimensionless)"
    legend_algebraic[9] = "z_max in component tropomyosin (dimensionless)"
    legend_constants[1] = "k_on in component troponin (per_mM_per_ms)"
    legend_constants[2] = "k_Ref_off in component troponin (per_ms)"
    legend_constants[3] = "gamma_trpn in component troponin (dimensionless)"
    legend_constants[4] = "alpha_0 in component tropomyosin (per_ms)"
    legend_constants[5] = "alpha_r1 in component tropomyosin (per_ms)"
    legend_constants[6] = "alpha_r2 in component tropomyosin (per_ms)"
    legend_constants[7] = "n_Rel in component tropomyosin (dimensionless)"
    legend_constants[8] = "K_z in component tropomyosin (dimensionless)"
    legend_constants[9] = "n_Hill in component tropomyosin (dimensionless)"
    legend_constants[10] = "Ca_50ref in component tropomyosin (mM)"
    legend_constants[11] = "z_p in component tropomyosin (dimensionless)"
    legend_constants[12] = "beta_1 in component tropomyosin (dimensionless)"
    legend_algebraic[6] = "Ca_50 in component tropomyosin (mM)"
    legend_algebraic[7] = "Ca_TRPN_50 in component tropomyosin (mM)"
    legend_constants[22] = "K_2 in component tropomyosin (per_ms)"
    legend_constants[24] = "K_1 in component tropomyosin (per_ms)"
    legend_algebraic[8] = "alpha_Tm in component tropomyosin (per_ms)"
    legend_algebraic[2] = "beta_Tm in component tropomyosin (per_ms)"
    legend_constants[13] = "beta_0 in component filament_overlap (dimensionless)"
    legend_algebraic[4] = "lambda in component Myofilaments (dimensionless)"
    legend_algebraic[15] = "k_off in component troponin (per_ms)"
    legend_algebraic[14] = "Tension in component Cross_Bridges (N_per_mm2)"
    legend_constants[14] = "T_ref in component length_independent_tension (N_per_mm2)"
    legend_algebraic[3] = "ExtensionRatio in component Myofilaments (dimensionless)"
    legend_constants[23] = "dExtensionRatiodt in component Myofilaments (per_ms)"
    legend_algebraic[5] = "lambda_prev in component Myofilaments (dimensionless)"
    legend_algebraic[10] = "overlap in component filament_overlap (dimensionless)"
    legend_algebraic[11] = "T_Base in component length_independent_tension (N_per_mm2)"
    legend_algebraic[12] = "T_0 in component isometric_tension (N_per_mm2)"
    legend_algebraic[13] = "Q in component Cross_Bridges (dimensionless)"
    legend_constants[15] = "a in component Cross_Bridges (dimensionless)"
    legend_states[2] = "Q_1 in component Cross_Bridges (dimensionless)"
    legend_states[3] = "Q_2 in component Cross_Bridges (dimensionless)"
    legend_states[4] = "Q_3 in component Cross_Bridges (dimensionless)"
    legend_constants[16] = "A_1 in component Cross_Bridges (dimensionless)"
    legend_constants[17] = "A_2 in component Cross_Bridges (dimensionless)"
    legend_constants[18] = "A_3 in component Cross_Bridges (dimensionless)"
    legend_constants[19] = "alpha_1 in component Cross_Bridges (per_ms)"
    legend_constants[20] = "alpha_2 in component Cross_Bridges (per_ms)"
    legend_constants[21] = "alpha_3 in component Cross_Bridges (per_ms)"
    legend_rates[0] = "d/dt TRPN in component intracellular_ion_concentrations (mM)"
    legend_rates[1] = "d/dt z in component tropomyosin (dimensionless)"
    legend_rates[2] = "d/dt Q_1 in component Cross_Bridges (dimensionless)"
    legend_rates[3] = "d/dt Q_2 in component Cross_Bridges (dimensionless)"
    legend_rates[4] = "d/dt Q_3 in component Cross_Bridges (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.067593139865
    constants[0] = 70e-3
    states[1] = 0.014417937837
    constants[1] = 100
    constants[2] = 0.2
    constants[3] = 2
    constants[4] = 8e-3
    constants[5] = 2e-3
    constants[6] = 1.75e-3
    constants[7] = 3
    constants[8] = 0.15
    constants[9] = 3
    constants[10] = 1.05e-3
    constants[11] = 0.85
    constants[12] = -4
    constants[13] = 4.9
    constants[14] = 56.2
    constants[15] = 0.35
    states[2] = 0
    states[3] = 0
    states[4] = 0
    constants[16] = -29
    constants[17] = 138
    constants[18] = 129
    constants[19] = 0.03
    constants[20] = 0.13
    constants[21] = 0.625
    constants[22] = ((constants[6]*(power(constants[11], constants[7])))/(power(constants[11], constants[7])+power(constants[8], constants[7])))*(1.00000-(constants[7]*(power(constants[8], constants[7])))/(power(constants[11], constants[7])+power(constants[8], constants[7])))
    constants[23] = 0.00000
    constants[24] = (constants[6]*(power(constants[11], constants[7]-1.00000))*constants[7]*(power(constants[8], constants[7])))/(power(power(constants[11], constants[7])+power(constants[8], constants[7]), 2.00000))
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[2] = constants[16]*constants[23]-constants[19]*states[2]
    rates[3] = constants[17]*constants[23]-constants[20]*states[3]
    rates[4] = constants[18]*constants[23]-constants[21]*states[4]
    algebraic[0] = constants[0]-states[0]
    algebraic[3] = custom_piecewise([greater(voi , 1000.00), 1.00000 , True, 1.00000])
    algebraic[4] = custom_piecewise([greater(algebraic[3] , 0.800000) & less_equal(algebraic[3] , 1.15000), algebraic[3] , greater(algebraic[3] , 1.15000), 1.15000 , True, 0.800000])
    algebraic[6] = constants[10]*(1.00000+constants[12]*(algebraic[4]-1.00000))
    algebraic[7] = (algebraic[6]*constants[0])/(algebraic[6]+(constants[2]/constants[1])*(1.00000-((1.00000+constants[13]*(algebraic[4]-1.00000))*0.500000)/constants[3]))
    algebraic[8] = constants[4]*(power(algebraic[0]/algebraic[7], constants[9]))
    algebraic[2] = constants[5]+(constants[6]*(power(states[1], constants[7]-1.00000)))/(power(states[1], constants[7])+power(constants[8], constants[7]))
    rates[1] = algebraic[8]*(1.00000-states[1])-algebraic[2]*states[1]
    algebraic[1] = custom_piecewise([less(voi , 1.00000), 1000.00*1.84330e-07 , greater_equal(voi , 10.0000) & less(voi , 15.0000), 1000.00*(((1.05500*(power(voi/1000.00, 3.00000))-0.0350700*(power(voi/1000.00, 2.00000)))+(0.000399200*voi)/1000.00)-1.35600e-06) , greater_equal(voi , 15.0000) & less(voi , 55.0000), 1000.00*(((0.0140000*(power(voi/1000.00, 3.00000))-0.00255500*(power(voi/1000.00, 2.00000)))+(0.000149400*voi)/1000.00)-1.42800e-06) , greater_equal(voi , 55.0000) & less(voi , 250.000), 1000.00*(((1.73900e-05*(power(voi/1000.00, 3.00000))-3.20900e-06*(power(voi/1000.00, 2.00000)))-(5.68900e-06*voi)/1000.00)+1.71900e-06) , greater_equal(voi , 250.000) & less(voi , 490.000), 1000.00*((((0.000132100*(power(voi/1000.00, 4.00000))-0.000219700*(power(voi/1000.00, 3.00000)))+0.000137400*(power(voi/1000.00, 2.00000)))-(3.89500e-05*voi)/1000.00)+4.44100e-06) , True, 1000.00*1.21480e-07])
    algebraic[10] = 1.00000+constants[13]*(algebraic[4]-1.00000)
    algebraic[9] = (constants[4]/(power(algebraic[7]/constants[0], constants[9]))-constants[22])/(constants[5]+constants[24]+constants[4]/(power(algebraic[7]/constants[0], constants[9])))
    algebraic[11] = (constants[14]*states[1])/algebraic[9]
    algebraic[12] = algebraic[11]*algebraic[10]
    algebraic[13] = states[2]+states[3]+states[4]
    algebraic[14] = custom_piecewise([less(algebraic[13] , 0.00000), (algebraic[12]*(constants[15]*algebraic[13]+1.00000))/(1.00000-algebraic[13]) , True, (algebraic[12]*(1.00000+(constants[15]+2.00000)*algebraic[13]))/(1.00000+algebraic[13])])
    algebraic[15] = custom_piecewise([greater(1.00000-algebraic[14]/(constants[3]*constants[14]) , 0.100000), constants[2]*(1.00000-algebraic[14]/(constants[3]*constants[14])) , True, constants[2]*0.100000])
    algebraic[16] = (constants[0]-states[0])*algebraic[15]-algebraic[1]*states[0]*constants[1]
    rates[0] = algebraic[16]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = constants[0]-states[0]
    algebraic[3] = custom_piecewise([greater(voi , 1000.00), 1.00000 , True, 1.00000])
    algebraic[4] = custom_piecewise([greater(algebraic[3] , 0.800000) & less_equal(algebraic[3] , 1.15000), algebraic[3] , greater(algebraic[3] , 1.15000), 1.15000 , True, 0.800000])
    algebraic[6] = constants[10]*(1.00000+constants[12]*(algebraic[4]-1.00000))
    algebraic[7] = (algebraic[6]*constants[0])/(algebraic[6]+(constants[2]/constants[1])*(1.00000-((1.00000+constants[13]*(algebraic[4]-1.00000))*0.500000)/constants[3]))
    algebraic[8] = constants[4]*(power(algebraic[0]/algebraic[7], constants[9]))
    algebraic[2] = constants[5]+(constants[6]*(power(states[1], constants[7]-1.00000)))/(power(states[1], constants[7])+power(constants[8], constants[7]))
    algebraic[1] = custom_piecewise([less(voi , 1.00000), 1000.00*1.84330e-07 , greater_equal(voi , 10.0000) & less(voi , 15.0000), 1000.00*(((1.05500*(power(voi/1000.00, 3.00000))-0.0350700*(power(voi/1000.00, 2.00000)))+(0.000399200*voi)/1000.00)-1.35600e-06) , greater_equal(voi , 15.0000) & less(voi , 55.0000), 1000.00*(((0.0140000*(power(voi/1000.00, 3.00000))-0.00255500*(power(voi/1000.00, 2.00000)))+(0.000149400*voi)/1000.00)-1.42800e-06) , greater_equal(voi , 55.0000) & less(voi , 250.000), 1000.00*(((1.73900e-05*(power(voi/1000.00, 3.00000))-3.20900e-06*(power(voi/1000.00, 2.00000)))-(5.68900e-06*voi)/1000.00)+1.71900e-06) , greater_equal(voi , 250.000) & less(voi , 490.000), 1000.00*((((0.000132100*(power(voi/1000.00, 4.00000))-0.000219700*(power(voi/1000.00, 3.00000)))+0.000137400*(power(voi/1000.00, 2.00000)))-(3.89500e-05*voi)/1000.00)+4.44100e-06) , True, 1000.00*1.21480e-07])
    algebraic[10] = 1.00000+constants[13]*(algebraic[4]-1.00000)
    algebraic[9] = (constants[4]/(power(algebraic[7]/constants[0], constants[9]))-constants[22])/(constants[5]+constants[24]+constants[4]/(power(algebraic[7]/constants[0], constants[9])))
    algebraic[11] = (constants[14]*states[1])/algebraic[9]
    algebraic[12] = algebraic[11]*algebraic[10]
    algebraic[13] = states[2]+states[3]+states[4]
    algebraic[14] = custom_piecewise([less(algebraic[13] , 0.00000), (algebraic[12]*(constants[15]*algebraic[13]+1.00000))/(1.00000-algebraic[13]) , True, (algebraic[12]*(1.00000+(constants[15]+2.00000)*algebraic[13]))/(1.00000+algebraic[13])])
    algebraic[15] = custom_piecewise([greater(1.00000-algebraic[14]/(constants[3]*constants[14]) , 0.100000), constants[2]*(1.00000-algebraic[14]/(constants[3]*constants[14])) , True, constants[2]*0.100000])
    algebraic[16] = (constants[0]-states[0])*algebraic[15]-algebraic[1]*states[0]*constants[1]
    algebraic[5] = algebraic[3]
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