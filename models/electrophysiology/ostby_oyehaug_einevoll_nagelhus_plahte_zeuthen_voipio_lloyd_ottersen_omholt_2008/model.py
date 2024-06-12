# Size of variable arrays:
sizeAlgebraic = 28
sizeStates = 8
sizeConstants = 31
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_states[0] = "wi in component wi (micrometre)"
    legend_constants[0] = "Lp in component model_parameters (cm_per_millimolar_second)"
    legend_constants[1] = "Xi in component model_parameters (millimolar_micrometre)"
    legend_algebraic[10] = "Nao in component model_parameters (millimolar)"
    legend_algebraic[6] = "Ko in component model_parameters (millimolar)"
    legend_algebraic[13] = "Clo in component model_parameters (millimolar)"
    legend_algebraic[18] = "HCO3o in component model_parameters (millimolar)"
    legend_algebraic[8] = "Nai in component model_parameters (millimolar)"
    legend_algebraic[5] = "Ki in component model_parameters (millimolar)"
    legend_algebraic[12] = "Cli in component model_parameters (millimolar)"
    legend_algebraic[17] = "HCO3i in component model_parameters (millimolar)"
    legend_algebraic[0] = "wo in component wo (micrometre)"
    legend_constants[2] = "w_tot in component wo (micrometre)"
    legend_states[1] = "N_Nai in component N_Nai (millimolar_micrometre)"
    legend_algebraic[25] = "dN_Nai_dt in component N_Nai (millimolar_micrometre_per_second)"
    legend_constants[3] = "gNa in component model_parameters (per_ohm_cm2)"
    legend_algebraic[11] = "ENa in component Nernst_potentials (volt)"
    legend_algebraic[20] = "Vm in component membrane (volt)"
    legend_algebraic[9] = "J_NaKATPase in component J_NaKATPase (mole_per_cm2_second)"
    legend_constants[4] = "F in component model_parameters (C_per_mol)"
    legend_constants[5] = "theta_NKCC1 in component model_parameters (dimensionless)"
    legend_algebraic[16] = "J_NKCC1 in component J_NKCC1 (millimolar_micrometre_per_second)"
    legend_constants[6] = "theta_NBC in component model_parameters (dimensionless)"
    legend_algebraic[24] = "J_NBC in component J_NBC (millimolar_micrometre_per_second)"
    legend_states[2] = "N_Ki in component N_Ki (millimolar_micrometre)"
    legend_algebraic[21] = "dN_Ki_dt in component N_Ki (millimolar_micrometre_per_second)"
    legend_constants[7] = "gK in component model_parameters (per_ohm_cm2)"
    legend_algebraic[7] = "EK in component Nernst_potentials (volt)"
    legend_constants[8] = "theta_KCC1 in component model_parameters (dimensionless)"
    legend_algebraic[15] = "J_KCC1 in component J_KCC1 (millimolar_micrometre_per_second)"
    legend_states[3] = "N_HCO3i in component N_HCO3i (millimolar_micrometre)"
    legend_algebraic[27] = "dN_HCO3i_dt in component N_HCO3i (millimolar_micrometre_per_second)"
    legend_states[4] = "N_Cli in component N_Cli (millimolar_micrometre)"
    legend_states[5] = "N_Nao in component N_Nao (millimolar_micrometre)"
    legend_algebraic[4] = "y in component model_parameters (dimensionless)"
    legend_constants[9] = "kc in component model_parameters (micrometre_millimolar_per_second)"
    legend_states[6] = "N_Ko in component N_Ko (millimolar_micrometre)"
    legend_states[7] = "N_HCO3o in component N_HCO3o (millimolar_micrometre)"
    legend_algebraic[1] = "N_Clo in component N_Clo (millimolar_micrometre)"
    legend_algebraic[14] = "ECl in component Nernst_potentials (volt)"
    legend_constants[10] = "T in component model_parameters (kelvin)"
    legend_constants[11] = "R in component model_parameters (J_per_mol_K)"
    legend_constants[12] = "gKCC1 in component J_KCC1 (per_ohm_cm2)"
    legend_algebraic[2] = "ft in component J_KCC1 (dimensionless)"
    legend_constants[13] = "gNBC in component J_NBC (per_ohm_cm2)"
    legend_algebraic[19] = "ENBC in component ENBC (volt)"
    legend_constants[14] = "gNKCC1 in component J_NKCC1 (per_ohm_cm2)"
    legend_algebraic[3] = "ft in component J_NKCC1 (dimensionless)"
    legend_constants[15] = "zNBC in component ENBC (dimensionless)"
    legend_constants[16] = "gCl in component model_parameters (per_ohm_cm2)"
    legend_constants[17] = "Pmax in component J_NaKATPase (mole_per_cm2_second)"
    legend_constants[18] = "K_Nai in component J_NaKATPase (millimolar)"
    legend_constants[19] = "K_Ko in component J_NaKATPase (millimolar)"
    legend_algebraic[22] = "I_K in component I_K (millimolar_micrometre_per_second)"
    legend_constants[20] = "nA in component model_parameters (per_mole)"
    legend_constants[21] = "elementary_charge in component model_parameters (C)"
    legend_algebraic[23] = "I_Cl in component I_Cl (millimolar_micrometre_per_second)"
    legend_algebraic[26] = "I_Na in component I_Na (millimolar_micrometre_per_second)"
    legend_constants[22] = "t0 in component model_parameters (second)"
    legend_constants[23] = "t1 in component model_parameters (second)"
    legend_constants[24] = "t2 in component model_parameters (second)"
    legend_constants[25] = "alpha in component model_parameters (dimensionless)"
    legend_constants[26] = "beta in component model_parameters (dimensionless)"
    legend_constants[27] = "deltaT in component model_parameters (second)"
    legend_constants[28] = "gamma_alpha_beta in component model_parameters (dimensionless)"
    legend_constants[29] = "gamma_alpha in component model_parameters (dimensionless)"
    legend_constants[30] = "gamma_beta in component model_parameters (dimensionless)"
    legend_rates[0] = "d/dt wi in component wi (micrometre)"
    legend_rates[1] = "d/dt N_Nai in component N_Nai (millimolar_micrometre)"
    legend_rates[2] = "d/dt N_Ki in component N_Ki (millimolar_micrometre)"
    legend_rates[3] = "d/dt N_HCO3i in component N_HCO3i (millimolar_micrometre)"
    legend_rates[4] = "d/dt N_Cli in component N_Cli (millimolar_micrometre)"
    legend_rates[5] = "d/dt N_Nao in component N_Nao (millimolar_micrometre)"
    legend_rates[6] = "d/dt N_Ko in component N_Ko (millimolar_micrometre)"
    legend_rates[7] = "d/dt N_HCO3o in component N_HCO3o (millimolar_micrometre)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.061
    constants[0] = 2.1e-4
    constants[1] = 12.41
    constants[2] = 0.0879
    states[1] = 0.99796
    constants[3] = 1.314e-4
    constants[4] = 9.649e4
    constants[5] = 1
    constants[6] = 1
    states[2] = 5.52782
    constants[7] = 0.004
    constants[8] = 1
    states[3] = 0.58804
    states[4] = 0.32879
    states[5] = 4.301041
    constants[9] = 7.35e-2
    states[6] = 0.0807
    states[7] = 0.432552
    constants[10] = 300
    constants[11] = 8.315
    constants[12] = 1e-6
    constants[13] = 9.03e-5
    constants[14] = 5.54e-6
    constants[15] = -1
    constants[16] = 8.797e-5
    constants[17] = 1.4207e-10
    constants[18] = 10
    constants[19] = 1.5
    constants[20] = 6.0221415e23
    constants[21] = 1.6e-19
    constants[22] = 10
    constants[23] = 20
    constants[24] = 30
    constants[25] = 2
    constants[26] = 14
    constants[27] = constants[23]-constants[22]
    constants[28] = factorial((constants[25]+constants[26])-1.00000)
    constants[29] = factorial(constants[25]-1.00000)
    constants[30] = factorial(constants[26]-1.00000)
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[0] = constants[2]-states[0]
    algebraic[10] = custom_piecewise([greater(states[5]/algebraic[0] , 0.00000), states[5]/algebraic[0] , True, 1.00000e-180])
    algebraic[6] = custom_piecewise([greater(states[6]/algebraic[0] , 0.00000), states[6]/algebraic[0] , True, 1.00000e-180])
    algebraic[1] = (states[5]+states[6])-states[7]
    algebraic[13] = custom_piecewise([greater(algebraic[1]/algebraic[0] , 0.00000), algebraic[1]/algebraic[0] , True, 1.00000e-180])
    algebraic[18] = custom_piecewise([greater(states[7]/algebraic[0] , 0.00000), states[7]/algebraic[0] , True, 1.00000e-180])
    algebraic[8] = custom_piecewise([greater(states[1]/states[0] , 0.00000), states[1]/states[0] , True, 1.00000e-180])
    algebraic[5] = custom_piecewise([greater(states[2]/states[0] , 0.00000), states[2]/states[0] , True, 1.00000e-180])
    algebraic[12] = custom_piecewise([greater(states[4]/states[0] , 0.00000), states[4]/states[0] , True, 1.00000e-180])
    algebraic[17] = custom_piecewise([greater(states[3]/states[0] , 0.00000), states[3]/states[0] , True, 1.00000e-180])
    rates[0] = 1.00000*constants[0]*((algebraic[8]+algebraic[5]+algebraic[12]+algebraic[17]+constants[1]/states[0])-(algebraic[10]+algebraic[6]+algebraic[13]+algebraic[18]))
    algebraic[11] = ((constants[11]*constants[10])/(1.00000*constants[4]))*log(algebraic[10]/algebraic[8])
    algebraic[9] = (((constants[17]*(power(algebraic[8], 1.50000)))/(power(algebraic[8], 1.50000)+power(constants[18], 1.50000)))*algebraic[6])/(algebraic[6]+constants[19])
    algebraic[7] = ((constants[11]*constants[10])/(1.00000*constants[4]))*log(algebraic[6]/algebraic[5])
    algebraic[14] = ((constants[11]*constants[10])/(-1.00000*constants[4]))*log(algebraic[13]/algebraic[12])
    algebraic[19] = ((constants[11]*constants[10])/(constants[15]*constants[4]))*log(((algebraic[10]/algebraic[8])*(power(algebraic[18], 2.00000)))/(power(algebraic[17], 2.00000)))
    algebraic[20] = ((constants[3]*algebraic[11]+constants[7]*algebraic[7]+constants[16]*algebraic[14]+constants[6]*constants[13]*algebraic[19])-algebraic[9]*constants[4])/(constants[3]+constants[7]+constants[16]+constants[6]*constants[13])
    algebraic[3] = custom_piecewise([greater_equal(voi , 10.0000) & less(voi , 20.0000), 1.00000 , True, 0.00000])
    algebraic[16] = ((((1.00000e+10*constants[14]*algebraic[3])/constants[4])*constants[11]*constants[10])/constants[4])*log((((algebraic[10]/algebraic[8])*algebraic[6])/algebraic[5])*(power(algebraic[13]/algebraic[12], 2.00000)))
    algebraic[2] = custom_piecewise([greater_equal(voi , 10.0000) & less(voi , 20.0000), 1.00000 , True, 0.00000])
    algebraic[15] = ((((1.00000e+10*constants[12]*algebraic[2])/constants[4])*constants[11]*constants[10])/constants[4])*log((algebraic[6]*algebraic[13])/(algebraic[5]*algebraic[12]))
    algebraic[4] = custom_piecewise([greater_equal(voi , constants[22]) & less_equal(voi , constants[23]), (constants[28]/(constants[29]*constants[30]))*(power(1.00000-(voi-constants[22])/constants[27], constants[26]-1.00000))*(power((voi-constants[22])/constants[27], constants[25]-1.00000)) , greater(voi , constants[23]) & less(voi , constants[24]), -1.00000 , True, 0.00000])
    rates[6] = (((1.00000e+10*constants[7])/constants[4])*(algebraic[20]-algebraic[7])+constants[9]*algebraic[4])-(constants[5]*algebraic[16]+constants[8]*algebraic[15]+1.00000e+10*2.00000*algebraic[9])
    algebraic[21] = (constants[5]*algebraic[16]+constants[8]*algebraic[15]+1.00000e+10*2.00000*algebraic[9])-((1.00000e+10*constants[7])/constants[4])*(algebraic[20]-algebraic[7])
    rates[2] = algebraic[21]
    algebraic[24] = ((1.00000e+10*constants[13])/constants[4])*(algebraic[20]-algebraic[19])
    rates[5] = 1.00000e+10*((constants[3]/constants[4])*(algebraic[20]-algebraic[11])+3.00000*algebraic[9])-(constants[5]*algebraic[16]+constants[6]*algebraic[24]+constants[9]*algebraic[4])
    rates[7] = -2.00000*constants[6]*algebraic[24]
    algebraic[25] = (constants[5]*algebraic[16]+constants[6]*algebraic[24])-1.00000e+10*((constants[3]/constants[4])*(algebraic[20]-algebraic[11])+3.00000*algebraic[9])
    rates[1] = algebraic[25]
    algebraic[27] = 2.00000*constants[6]*algebraic[24]
    rates[3] = algebraic[27]
    rates[4] = algebraic[21]+algebraic[25]+algebraic[27]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = constants[2]-states[0]
    algebraic[10] = custom_piecewise([greater(states[5]/algebraic[0] , 0.00000), states[5]/algebraic[0] , True, 1.00000e-180])
    algebraic[6] = custom_piecewise([greater(states[6]/algebraic[0] , 0.00000), states[6]/algebraic[0] , True, 1.00000e-180])
    algebraic[1] = (states[5]+states[6])-states[7]
    algebraic[13] = custom_piecewise([greater(algebraic[1]/algebraic[0] , 0.00000), algebraic[1]/algebraic[0] , True, 1.00000e-180])
    algebraic[18] = custom_piecewise([greater(states[7]/algebraic[0] , 0.00000), states[7]/algebraic[0] , True, 1.00000e-180])
    algebraic[8] = custom_piecewise([greater(states[1]/states[0] , 0.00000), states[1]/states[0] , True, 1.00000e-180])
    algebraic[5] = custom_piecewise([greater(states[2]/states[0] , 0.00000), states[2]/states[0] , True, 1.00000e-180])
    algebraic[12] = custom_piecewise([greater(states[4]/states[0] , 0.00000), states[4]/states[0] , True, 1.00000e-180])
    algebraic[17] = custom_piecewise([greater(states[3]/states[0] , 0.00000), states[3]/states[0] , True, 1.00000e-180])
    algebraic[11] = ((constants[11]*constants[10])/(1.00000*constants[4]))*log(algebraic[10]/algebraic[8])
    algebraic[9] = (((constants[17]*(power(algebraic[8], 1.50000)))/(power(algebraic[8], 1.50000)+power(constants[18], 1.50000)))*algebraic[6])/(algebraic[6]+constants[19])
    algebraic[7] = ((constants[11]*constants[10])/(1.00000*constants[4]))*log(algebraic[6]/algebraic[5])
    algebraic[14] = ((constants[11]*constants[10])/(-1.00000*constants[4]))*log(algebraic[13]/algebraic[12])
    algebraic[19] = ((constants[11]*constants[10])/(constants[15]*constants[4]))*log(((algebraic[10]/algebraic[8])*(power(algebraic[18], 2.00000)))/(power(algebraic[17], 2.00000)))
    algebraic[20] = ((constants[3]*algebraic[11]+constants[7]*algebraic[7]+constants[16]*algebraic[14]+constants[6]*constants[13]*algebraic[19])-algebraic[9]*constants[4])/(constants[3]+constants[7]+constants[16]+constants[6]*constants[13])
    algebraic[3] = custom_piecewise([greater_equal(voi , 10.0000) & less(voi , 20.0000), 1.00000 , True, 0.00000])
    algebraic[16] = ((((1.00000e+10*constants[14]*algebraic[3])/constants[4])*constants[11]*constants[10])/constants[4])*log((((algebraic[10]/algebraic[8])*algebraic[6])/algebraic[5])*(power(algebraic[13]/algebraic[12], 2.00000)))
    algebraic[2] = custom_piecewise([greater_equal(voi , 10.0000) & less(voi , 20.0000), 1.00000 , True, 0.00000])
    algebraic[15] = ((((1.00000e+10*constants[12]*algebraic[2])/constants[4])*constants[11]*constants[10])/constants[4])*log((algebraic[6]*algebraic[13])/(algebraic[5]*algebraic[12]))
    algebraic[4] = custom_piecewise([greater_equal(voi , constants[22]) & less_equal(voi , constants[23]), (constants[28]/(constants[29]*constants[30]))*(power(1.00000-(voi-constants[22])/constants[27], constants[26]-1.00000))*(power((voi-constants[22])/constants[27], constants[25]-1.00000)) , greater(voi , constants[23]) & less(voi , constants[24]), -1.00000 , True, 0.00000])
    algebraic[21] = (constants[5]*algebraic[16]+constants[8]*algebraic[15]+1.00000e+10*2.00000*algebraic[9])-((1.00000e+10*constants[7])/constants[4])*(algebraic[20]-algebraic[7])
    algebraic[24] = ((1.00000e+10*constants[13])/constants[4])*(algebraic[20]-algebraic[19])
    algebraic[25] = (constants[5]*algebraic[16]+constants[6]*algebraic[24])-1.00000e+10*((constants[3]/constants[4])*(algebraic[20]-algebraic[11])+3.00000*algebraic[9])
    algebraic[27] = 2.00000*constants[6]*algebraic[24]
    algebraic[22] = (1.00000e+10*2.00000*algebraic[9]*constants[4])/(constants[21]*constants[20])-((1.00000e+10*constants[7])/(constants[21]*constants[20]))*(algebraic[20]-algebraic[7])
    algebraic[23] = ((1.00000e+10*constants[16])/(constants[21]*constants[20]))*(algebraic[20]-algebraic[14])
    algebraic[26] = algebraic[24]-(((1.00000e+10*constants[3])/(constants[21]*constants[20]))*(algebraic[20]-algebraic[11])+(1.00000e+10*3.00000*algebraic[9]*constants[4])/(constants[21]*constants[20]))
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