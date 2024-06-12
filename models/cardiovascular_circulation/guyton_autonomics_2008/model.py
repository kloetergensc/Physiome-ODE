# Size of variable arrays:
sizeAlgebraic = 13
sizeStates = 2
sizeConstants = 41
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_constants[0] = "PA in component autonomics (mmHg)"
    legend_constants[1] = "PO2ART in component autonomics (mmHg)"
    legend_constants[2] = "PLA in component autonomics (mmHg)"
    legend_constants[3] = "PRA in component autonomics (mmHg)"
    legend_constants[4] = "PPA in component autonomics (mmHg)"
    legend_constants[31] = "PA1 in component pressure_driving_autonomic_receptors (mmHg)"
    legend_constants[5] = "CRRFLX in component parameter_values (mmHg)"
    legend_constants[6] = "EXE in component parameter_values (mmHg)"
    legend_constants[32] = "AUC in component chemoreceptors_effect_of_PA (dimensionless)"
    legend_constants[7] = "AUC1 in component parameter_values (dimensionless)"
    legend_constants[37] = "AUC3 in component chemoreceptors_effect_of_art_PO2 (dimensionless)"
    legend_constants[8] = "O2CHMO in component parameter_values (per_mmHg)"
    legend_constants[34] = "AUC2 in component chemoreceptors_effect_of_art_PO2 (dimensionless)"
    legend_algebraic[0] = "AU6C in component arterial_baroreceptor_reflex (dimensionless)"
    legend_constants[9] = "AUX in component parameter_values (dimensionless)"
    legend_constants[10] = "AUK in component parameter_values (per_minute)"
    legend_constants[11] = "BAROTC in component parameter_values (minute)"
    legend_constants[33] = "AUB in component arterial_baroreceptor_reflex (dimensionless)"
    legend_constants[35] = "A1B in component arterial_baroreceptor_reflex (dimensionless)"
    legend_constants[36] = "AU6A in component arterial_baroreceptor_reflex (dimensionless)"
    legend_constants[12] = "AU4 in component arterial_baroreceptor_reflex (dimensionless)"
    legend_states[0] = "AU6 in component arterial_baroreceptor_reflex (dimensionless)"
    legend_constants[38] = "AUN in component CNS_ischemic_reflex (dimensionless)"
    legend_constants[13] = "AUN1 in component parameter_values (dimensionless)"
    legend_constants[39] = "AULP in component autonomic_response_to_vasculature_pressure (dimensionless)"
    legend_constants[14] = "AULPM in component parameter_values (dimensionless)"
    legend_constants[40] = "AUEX in component autonomic_response_to_exercise (dimensionless)"
    legend_constants[15] = "EXC in component parameter_values (dimensionless)"
    legend_constants[16] = "EXCXP in component parameter_values (dimensionless)"
    legend_algebraic[4] = "AUTTL in component total_autonomic_stimulation (dimensionless)"
    legend_algebraic[2] = "AUTTL1 in component total_autonomic_stimulation (dimensionless)"
    legend_constants[17] = "EXCML in component parameter_values (dimensionless)"
    legend_algebraic[3] = "AU in component actual_autonomic_stimulation (dimensionless)"
    legend_constants[18] = "AUDMP in component parameter_values (minute)"
    legend_constants[19] = "AUMAX in component parameter_values (dimensionless)"
    legend_constants[20] = "AUMIN in component parameter_values (dimensionless)"
    legend_constants[21] = "AUSLP in component parameter_values (dimensionless)"
    legend_algebraic[7] = "DAU in component actual_autonomic_stimulation (per_minute)"
    legend_states[1] = "AU1 in component actual_autonomic_stimulation (dimensionless)"
    legend_algebraic[1] = "AUT in component actual_autonomic_stimulation (dimensionless)"
    legend_algebraic[5] = "VVR in component autonomic_drive_on_target_organs_and_tissues (litre)"
    legend_algebraic[8] = "AUH in component autonomic_drive_on_target_organs_and_tissues (dimensionless)"
    legend_algebraic[9] = "AUR in component autonomic_drive_on_target_organs_and_tissues (dimensionless)"
    legend_algebraic[10] = "AOM in component autonomic_drive_on_target_organs_and_tissues (dimensionless)"
    legend_algebraic[11] = "AUM in component autonomic_drive_on_target_organs_and_tissues (dimensionless)"
    legend_algebraic[12] = "AVE in component autonomic_drive_on_target_organs_and_tissues (dimensionless)"
    legend_constants[22] = "VV9 in component parameter_values (litre)"
    legend_constants[23] = "AUL in component parameter_values (litre)"
    legend_constants[24] = "AUV in component parameter_values (dimensionless)"
    legend_constants[25] = "AUS in component parameter_values (dimensionless)"
    legend_constants[26] = "O2A in component parameter_values (dimensionless)"
    legend_constants[27] = "AUM1 in component parameter_values (dimensionless)"
    legend_constants[28] = "AUM2 in component parameter_values (dimensionless)"
    legend_constants[29] = "AUY in component parameter_values (dimensionless)"
    legend_algebraic[6] = "AUO in component autonomic_drive_on_target_organs_and_tissues (dimensionless)"
    legend_constants[30] = "MDMP in component parameter_values (dimensionless)"
    legend_rates[0] = "d/dt AU6 in component arterial_baroreceptor_reflex (dimensionless)"
    legend_rates[1] = "d/dt AU1 in component actual_autonomic_stimulation (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 103.525
    constants[1] = 97.0439
    constants[2] = 2
    constants[3] = 0.00852183
    constants[4] = 15.6376
    constants[5] = 0
    constants[6] = 0
    constants[7] = 0.3
    constants[8] = 0.01
    constants[9] = 1
    constants[10] = 0.004
    constants[11] = 0.16
    constants[12] = -0.060024
    states[0] = 1.00132
    constants[13] = 0.5
    constants[14] = 0
    constants[15] = 1
    constants[16] = 1.0
    constants[17] = 0.01
    constants[18] = 0.3
    constants[19] = 5.0
    constants[20] = 0.4
    constants[21] = 1.5
    states[1] = 1.00007
    constants[22] = 2.51
    constants[23] = 1.5
    constants[24] = 0.55
    constants[25] = 1
    constants[26] = 0.1
    constants[27] = 3
    constants[28] = 1
    constants[29] = 0
    constants[30] = 0
    constants[31] = custom_piecewise([greater(constants[5] , 1.00000e-07), constants[5] , True, constants[0]-constants[6]])
    constants[32] = custom_piecewise([less(constants[31] , 80.0000) & greater_equal(constants[31] , 40.0000), 0.00500000*(80.0000-constants[31])*constants[7] , less(constants[31] , 40.0000), 0.200000*constants[7] , True, 0.00000])
    constants[33] = custom_piecewise([less(constants[31] , 160.000) & greater_equal(constants[31] , 80.0000), 0.0166670*(160.000-constants[31]) , less(constants[31] , 80.0000), 1.33360 , True, 0.00000])
    constants[34] = custom_piecewise([less(constants[1] , 80.0000) & greater_equal(constants[1] , 40.0000), constants[8]*(80.0000-constants[1]) , less(constants[1] , 40.0000), constants[8]*40.0000 , True, 0.00000])
    constants[35] = (constants[33]-1.00000)*constants[9]+1.00000
    constants[36] = constants[35]-constants[12]
    constants[37] = constants[32]+constants[34]
    constants[38] = custom_piecewise([less(constants[31] , 40.0000), 0.0400000*(40.0000-constants[31])*constants[13] , True, 0.00000])
    constants[39] = (15.0000/(constants[2]+constants[3]+constants[4])-1.00000)*constants[14]+1.00000
    constants[40] = power(constants[15], constants[16])
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = (constants[36]-states[0])/constants[11]
    algebraic[0] = states[0]
    algebraic[2] = (constants[40]*constants[39]*(constants[37]+algebraic[0]+constants[38])-1.00000)*constants[17]+1.00000
    algebraic[4] = custom_piecewise([less(algebraic[2] , 0.00000), 0.00000 , True, algebraic[2]])
    algebraic[7] = (algebraic[4]-states[1])/constants[18]
    rates[1] = algebraic[7]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = states[0]
    algebraic[2] = (constants[40]*constants[39]*(constants[37]+algebraic[0]+constants[38])-1.00000)*constants[17]+1.00000
    algebraic[4] = custom_piecewise([less(algebraic[2] , 0.00000), 0.00000 , True, algebraic[2]])
    algebraic[7] = (algebraic[4]-states[1])/constants[18]
    algebraic[1] = constants[19]-(constants[19]-1.00000)/exp(constants[21]*(states[1]-1.00000))
    algebraic[3] = custom_piecewise([less(algebraic[1] , constants[20]), constants[20] , True, algebraic[1]])
    algebraic[5] = (constants[22]-algebraic[3]*constants[23])+constants[23]
    algebraic[6] = algebraic[3]-1.00000
    algebraic[8] = algebraic[6]*constants[24]+1.00000
    algebraic[9] = algebraic[6]*constants[25]+1.00000
    algebraic[10] = algebraic[6]*constants[26]+1.00000
    algebraic[11] = power(algebraic[6]*constants[27]+1.00000, constants[28])
    algebraic[12] = algebraic[6]*constants[29]+1.00000
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