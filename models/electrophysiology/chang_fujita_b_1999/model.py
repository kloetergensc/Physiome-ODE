# Size of variable arrays:
sizeAlgebraic = 30
sizeStates = 9
sizeConstants = 27
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_constants[0] = "C_m_Imp in component imported_variables (mmol_per_cm3)"
    legend_constants[1] = "C_c_Imp in component imported_variables (mmol_per_cm3)"
    legend_constants[2] = "psi_m in component imported_variables (millivolt)"
    legend_constants[3] = "psi_c in component imported_variables (millivolt)"
    legend_states[0] = "C_m_Na in component solute_concentrations (mmol_per_cm3)"
    legend_states[1] = "C_m_K in component solute_concentrations (mmol_per_cm3)"
    legend_states[2] = "C_m_Cl in component solute_concentrations (mmol_per_cm3)"
    legend_states[3] = "C_c_Na in component solute_concentrations (mmol_per_cm3)"
    legend_states[4] = "C_c_K in component solute_concentrations (mmol_per_cm3)"
    legend_states[5] = "C_c_Cl in component solute_concentrations (mmol_per_cm3)"
    legend_states[6] = "C_s_Na in component solute_concentrations (mmol_per_cm3)"
    legend_states[7] = "C_s_K in component solute_concentrations (mmol_per_cm3)"
    legend_states[8] = "C_s_Cl in component solute_concentrations (mmol_per_cm3)"
    legend_algebraic[4] = "J_mc_Na in component mc_sodium_flux (flux)"
    legend_algebraic[22] = "J_ms_Na in component ms_sodium_flux (flux)"
    legend_algebraic[16] = "J_sc_Na in component sc_sodium_flux (flux)"
    legend_algebraic[11] = "J_mc_K in component mc_potassium_flux (flux)"
    legend_algebraic[25] = "J_ms_K in component ms_potassium_flux (flux)"
    legend_algebraic[20] = "J_sc_K in component sc_potassium_flux (flux)"
    legend_algebraic[15] = "J_mc_Cl in component mc_chloride_flux (flux)"
    legend_algebraic[26] = "J_ms_Cl in component ms_chloride_flux (flux)"
    legend_algebraic[21] = "J_sc_Cl in component sc_chloride_flux (flux)"
    legend_constants[4] = "RT in component constants (J_per_mmol)"
    legend_constants[5] = "F in component constants (C_per_mmol)"
    legend_constants[6] = "C_s_Imp in component constants (mmol_per_cm3)"
    legend_constants[7] = "psi_s in component constants (millivolt)"
    legend_algebraic[2] = "J_mc_NaCl in component mc_sodium_flux (flux)"
    legend_algebraic[0] = "G_mc_Na in component mc_sodium_flux (flux)"
    legend_constants[8] = "P_mc_Na in component mc_sodium_flux (cm_per_s)"
    legend_constants[9] = "J_mc_NaCl_max in component mc_sodium_flux (flux)"
    legend_constants[10] = "K_mc_Na_NaCl in component mc_sodium_flux (mmol_per_cm3)"
    legend_constants[11] = "K_mc_Cl_NaCl in component mc_sodium_flux (mmol_per_cm3)"
    legend_algebraic[9] = "J_mc_KCl in component mc_potassium_flux (flux)"
    legend_algebraic[6] = "G_mc_K in component mc_potassium_flux (flux)"
    legend_constants[12] = "J_mc_KCl_max in component mc_potassium_flux (flux)"
    legend_constants[13] = "K_mc_K_KCl in component mc_potassium_flux (mmol_per_cm3)"
    legend_constants[14] = "K_mc_Cl_KCl in component mc_potassium_flux (mmol_per_cm3)"
    legend_constants[15] = "P_mc_K in component mc_potassium_flux (cm_per_s)"
    legend_algebraic[12] = "G_mc_Cl in component mc_chloride_flux (flux)"
    legend_constants[16] = "P_mc_Cl in component mc_chloride_flux (cm_per_s)"
    legend_algebraic[14] = "J_a in component sc_sodium_flux (flux)"
    legend_constants[17] = "J_a_max in component sc_sodium_flux (flux)"
    legend_constants[18] = "K_Na_ATPase in component sc_sodium_flux (mmol_per_cm3)"
    legend_algebraic[17] = "G_sc_K in component sc_potassium_flux (flux)"
    legend_constants[19] = "P_sc_K in component sc_potassium_flux (cm_per_s)"
    legend_algebraic[18] = "G_sc_Cl in component sc_chloride_flux (flux)"
    legend_constants[20] = "P_sc_Cl in component sc_chloride_flux (cm_per_s)"
    legend_algebraic[19] = "G_ms_Na in component ms_sodium_flux (flux)"
    legend_constants[21] = "P_ms_Na in component ms_sodium_flux (cm_per_s)"
    legend_algebraic[23] = "G_ms_K in component ms_potassium_flux (flux)"
    legend_constants[22] = "P_ms_K in component ms_potassium_flux (cm_per_s)"
    legend_algebraic[24] = "G_ms_Cl in component ms_chloride_flux (flux)"
    legend_constants[23] = "P_ms_Cl in component ms_chloride_flux (cm_per_s)"
    legend_algebraic[27] = "J_Na in component total_transepithelial_sodium_flux (flux)"
    legend_algebraic[28] = "J_K in component total_transepithelial_potassium_flux (flux)"
    legend_algebraic[29] = "J_Cl in component total_transepithelial_chloride_flux (flux)"
    legend_algebraic[1] = "Osm_m in component osmolarities (mmol_per_cm3)"
    legend_algebraic[3] = "Osm_c in component osmolarities (mmol_per_cm3)"
    legend_algebraic[5] = "Osm_s in component osmolarities (mmol_per_cm3)"
    legend_algebraic[7] = "J_mc_v in component mc_transepithelial_volume_flux (cm_per_s)"
    legend_constants[24] = "L_mc_v in component mc_transepithelial_volume_flux (cm_per_s_mmHg)"
    legend_algebraic[10] = "J_ms_v in component ms_transepithelial_volume_flux (cm_per_s)"
    legend_constants[25] = "L_ms_v in component ms_transepithelial_volume_flux (cm_per_s_mmHg)"
    legend_algebraic[8] = "J_sc_v in component sc_transepithelial_volume_flux (cm_per_s)"
    legend_constants[26] = "L_sc_v in component sc_transepithelial_volume_flux (cm_per_s_mmHg)"
    legend_algebraic[13] = "J_v in component total_transepithelial_volume_flux (cm_per_s)"
    legend_rates[0] = "d/dt C_m_Na in component solute_concentrations (mmol_per_cm3)"
    legend_rates[6] = "d/dt C_s_Na in component solute_concentrations (mmol_per_cm3)"
    legend_rates[3] = "d/dt C_c_Na in component solute_concentrations (mmol_per_cm3)"
    legend_rates[1] = "d/dt C_m_K in component solute_concentrations (mmol_per_cm3)"
    legend_rates[7] = "d/dt C_s_K in component solute_concentrations (mmol_per_cm3)"
    legend_rates[4] = "d/dt C_c_K in component solute_concentrations (mmol_per_cm3)"
    legend_rates[2] = "d/dt C_m_Cl in component solute_concentrations (mmol_per_cm3)"
    legend_rates[8] = "d/dt C_s_Cl in component solute_concentrations (mmol_per_cm3)"
    legend_rates[5] = "d/dt C_c_Cl in component solute_concentrations (mmol_per_cm3)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 0.1033
    constants[1] = 0.1124
    constants[2] = -28.0
    constants[3] = -86.4
    states[0] = 0.05
    states[1] = 0.002
    states[2] = 0.03
    states[3] = 0.0164
    states[4] = 0.1637
    states[5] = 0.0203
    states[6] = 1.438E-1
    states[7] = 4.25E-3
    states[8] = 1.12E-1
    constants[4] = 2.579
    constants[5] = 96.48
    constants[6] = 4.525E-2
    constants[7] = 0.0
    constants[8] = 3.27E-6
    constants[9] = 3.21E-5
    constants[10] = 5.11E-2
    constants[11] = 1.92E-2
    constants[12] = 6.31E-8
    constants[13] = 5.30E-2
    constants[14] = 2.13E-2
    constants[15] = 4.90E-7
    constants[16] = 1.43E-6
    constants[17] = 2.69E-6
    constants[18] = 1.20E-2
    constants[19] = 4.74E-4
    constants[20] = 9.16E-5
    constants[21] = 4.80E-6
    constants[22] = 4.80E-6
    constants[23] = 2.40E-6
    constants[24] = 5.22E-9
    constants[25] = 0.0
    constants[26] = 5.22E-7
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[2] = constants[9]*(((states[0]/constants[10])*(states[2]/constants[11])-(states[3]/constants[10])*(states[5]/constants[11]))/((1.00000+(states[0]/constants[10])*(states[2]/constants[11]))*(1.00000+states[3]/constants[10])*(1.00000+states[5]/constants[11])+(1.00000+(states[3]/constants[10])*(states[5]/constants[11]))*(1.00000+states[0]/constants[10])*(1.00000+states[2]/constants[11])))
    algebraic[0] = constants[8]*((constants[5]*(constants[2]-constants[3]))/constants[4])*((states[0]-states[3]*exp(-(constants[5]/constants[4])*(constants[2]-constants[3])))/(1.00000-exp(-(constants[5]/constants[4])*(constants[2]-constants[3]))))
    algebraic[4] = algebraic[2]+algebraic[0]
    algebraic[14] = constants[17]*(1.00000/(1.00000+power(constants[18]/states[3], 3.00000)))
    algebraic[16] = -3.00000*algebraic[14]
    rates[3] = algebraic[4]+algebraic[16]
    algebraic[9] = constants[12]*(((states[1]/constants[13])*(states[2]/constants[14])-(states[4]/constants[13])*(states[5]/constants[14]))/((1.00000+(states[1]/constants[13])*(states[2]/constants[14]))*(1.00000+states[4]/constants[13])*(1.00000+states[5]/constants[14])+(1.00000+(states[4]/constants[13])*(states[5]/constants[14]))*(1.00000+states[1]/constants[13])*(1.00000+states[2]/constants[14])))
    algebraic[6] = constants[15]*((constants[5]*(constants[2]-constants[3]))/constants[4])*((states[1]-states[4]*exp(-(constants[5]/constants[4])*(constants[2]-constants[3])))/(1.00000-exp(-(constants[5]/constants[4])*(constants[2]-constants[3]))))
    algebraic[11] = algebraic[9]+algebraic[6]
    algebraic[17] = constants[19]*((constants[5]*(constants[7]-constants[3]))/constants[4])*((states[7]-states[4]*exp(-(constants[5]/constants[4])*(constants[7]-constants[3])))/(1.00000-exp(-(constants[5]/constants[4])*(constants[7]-constants[3]))))
    algebraic[20] = 2.00000*algebraic[14]+algebraic[17]
    rates[4] = algebraic[11]+algebraic[20]
    algebraic[12] = constants[16]*((-1.00000*constants[5]*(constants[2]-constants[3]))/constants[4])*((states[2]-states[5]*exp(-((-1.00000*constants[5])/constants[4])*(constants[2]-constants[3])))/(1.00000-exp(-((-1.00000*constants[5])/constants[4])*(constants[2]-constants[3]))))
    algebraic[15] = algebraic[2]+algebraic[9]+algebraic[12]
    algebraic[18] = constants[20]*((-1.00000*constants[5]*(constants[7]-constants[3]))/constants[4])*((states[8]-states[5]*exp(-((-1.00000*constants[5])/constants[4])*(constants[7]-constants[3])))/(1.00000-exp(-((-1.00000*constants[5])/constants[4])*(constants[7]-constants[3]))))
    algebraic[21] = algebraic[18]
    rates[5] = algebraic[15]+algebraic[21]
    algebraic[19] = constants[21]*((constants[5]*(constants[2]-constants[7]))/constants[4])*((states[0]-states[6]*exp(-(constants[5]/constants[4])*(constants[2]-constants[7])))/(1.00000-exp(-(constants[5]/constants[4])*(constants[2]-constants[7]))))
    algebraic[22] = algebraic[19]
    rates[0] = -(algebraic[4]+algebraic[22])
    rates[6] = algebraic[22]-algebraic[16]
    algebraic[23] = constants[22]*((constants[5]*(constants[2]-constants[7]))/constants[4])*((states[1]-states[7]*exp(-(constants[5]/constants[4])*(constants[2]-constants[7])))/(1.00000-exp(-(constants[5]/constants[4])*(constants[2]-constants[7]))))
    algebraic[25] = algebraic[23]
    rates[1] = -(algebraic[11]+algebraic[25])
    rates[7] = algebraic[25]-algebraic[20]
    algebraic[24] = constants[23]*((-1.00000*constants[5]*(constants[2]-constants[7]))/constants[4])*((states[2]-states[8]*exp(-((-1.00000*constants[5])/constants[4])*(constants[2]-constants[7])))/(1.00000-exp(-((-1.00000*constants[5])/constants[4])*(constants[2]-constants[7]))))
    algebraic[26] = algebraic[24]
    rates[2] = -(algebraic[15]+algebraic[26])
    rates[8] = algebraic[26]-algebraic[21]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[2] = constants[9]*(((states[0]/constants[10])*(states[2]/constants[11])-(states[3]/constants[10])*(states[5]/constants[11]))/((1.00000+(states[0]/constants[10])*(states[2]/constants[11]))*(1.00000+states[3]/constants[10])*(1.00000+states[5]/constants[11])+(1.00000+(states[3]/constants[10])*(states[5]/constants[11]))*(1.00000+states[0]/constants[10])*(1.00000+states[2]/constants[11])))
    algebraic[0] = constants[8]*((constants[5]*(constants[2]-constants[3]))/constants[4])*((states[0]-states[3]*exp(-(constants[5]/constants[4])*(constants[2]-constants[3])))/(1.00000-exp(-(constants[5]/constants[4])*(constants[2]-constants[3]))))
    algebraic[4] = algebraic[2]+algebraic[0]
    algebraic[14] = constants[17]*(1.00000/(1.00000+power(constants[18]/states[3], 3.00000)))
    algebraic[16] = -3.00000*algebraic[14]
    algebraic[9] = constants[12]*(((states[1]/constants[13])*(states[2]/constants[14])-(states[4]/constants[13])*(states[5]/constants[14]))/((1.00000+(states[1]/constants[13])*(states[2]/constants[14]))*(1.00000+states[4]/constants[13])*(1.00000+states[5]/constants[14])+(1.00000+(states[4]/constants[13])*(states[5]/constants[14]))*(1.00000+states[1]/constants[13])*(1.00000+states[2]/constants[14])))
    algebraic[6] = constants[15]*((constants[5]*(constants[2]-constants[3]))/constants[4])*((states[1]-states[4]*exp(-(constants[5]/constants[4])*(constants[2]-constants[3])))/(1.00000-exp(-(constants[5]/constants[4])*(constants[2]-constants[3]))))
    algebraic[11] = algebraic[9]+algebraic[6]
    algebraic[17] = constants[19]*((constants[5]*(constants[7]-constants[3]))/constants[4])*((states[7]-states[4]*exp(-(constants[5]/constants[4])*(constants[7]-constants[3])))/(1.00000-exp(-(constants[5]/constants[4])*(constants[7]-constants[3]))))
    algebraic[20] = 2.00000*algebraic[14]+algebraic[17]
    algebraic[12] = constants[16]*((-1.00000*constants[5]*(constants[2]-constants[3]))/constants[4])*((states[2]-states[5]*exp(-((-1.00000*constants[5])/constants[4])*(constants[2]-constants[3])))/(1.00000-exp(-((-1.00000*constants[5])/constants[4])*(constants[2]-constants[3]))))
    algebraic[15] = algebraic[2]+algebraic[9]+algebraic[12]
    algebraic[18] = constants[20]*((-1.00000*constants[5]*(constants[7]-constants[3]))/constants[4])*((states[8]-states[5]*exp(-((-1.00000*constants[5])/constants[4])*(constants[7]-constants[3])))/(1.00000-exp(-((-1.00000*constants[5])/constants[4])*(constants[7]-constants[3]))))
    algebraic[21] = algebraic[18]
    algebraic[19] = constants[21]*((constants[5]*(constants[2]-constants[7]))/constants[4])*((states[0]-states[6]*exp(-(constants[5]/constants[4])*(constants[2]-constants[7])))/(1.00000-exp(-(constants[5]/constants[4])*(constants[2]-constants[7]))))
    algebraic[22] = algebraic[19]
    algebraic[23] = constants[22]*((constants[5]*(constants[2]-constants[7]))/constants[4])*((states[1]-states[7]*exp(-(constants[5]/constants[4])*(constants[2]-constants[7])))/(1.00000-exp(-(constants[5]/constants[4])*(constants[2]-constants[7]))))
    algebraic[25] = algebraic[23]
    algebraic[24] = constants[23]*((-1.00000*constants[5]*(constants[2]-constants[7]))/constants[4])*((states[2]-states[8]*exp(-((-1.00000*constants[5])/constants[4])*(constants[2]-constants[7])))/(1.00000-exp(-((-1.00000*constants[5])/constants[4])*(constants[2]-constants[7]))))
    algebraic[26] = algebraic[24]
    algebraic[1] = states[0]+states[1]+states[2]+constants[0]
    algebraic[3] = states[3]+states[4]+states[5]+constants[1]
    algebraic[5] = states[6]+states[7]+states[8]+constants[6]
    algebraic[7] = constants[24]*constants[4]*(algebraic[1]-algebraic[3])
    algebraic[8] = constants[26]*constants[4]*(algebraic[5]-algebraic[3])
    algebraic[10] = constants[25]*constants[4]*(algebraic[1]-algebraic[5])
    algebraic[13] = algebraic[7]+algebraic[10]
    algebraic[27] = algebraic[4]+algebraic[22]
    algebraic[28] = algebraic[11]+algebraic[25]
    algebraic[29] = algebraic[15]+algebraic[26]
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