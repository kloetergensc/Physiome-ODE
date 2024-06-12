# Size of variable arrays:
sizeAlgebraic = 56
sizeStates = 4
sizeConstants = 72
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_constants[0] = "ADHMK in component kidney (dimensionless)"
    legend_constants[1] = "AMK in component kidney (dimensionless)"
    legend_constants[2] = "AMNA in component kidney (dimensionless)"
    legend_constants[3] = "ANM in component kidney (dimensionless)"
    legend_constants[4] = "ANPX in component kidney (dimensionless)"
    legend_constants[5] = "AUM in component kidney (dimensionless)"
    legend_constants[6] = "CKE in component kidney (monovalent_mEq_per_litre)"
    legend_constants[7] = "CNA in component kidney (monovalent_mEq_per_litre)"
    legend_constants[8] = "HM1 in component kidney (dimensionless)"
    legend_constants[9] = "MYOGRS in component kidney (dimensionless)"
    legend_constants[10] = "PA in component kidney (mmHg)"
    legend_constants[11] = "PAMKRN in component kidney (dimensionless)"
    legend_constants[12] = "PPC in component kidney (mmHg)"
    legend_constants[13] = "VTW in component kidney (litre)"
    legend_algebraic[0] = "PAR in component perfusion_pressure (mmHg)"
    legend_constants[14] = "GBL in component parameter_values (mmHg)"
    legend_constants[15] = "RAPRSP in component parameter_values (mmHg)"
    legend_constants[16] = "RFCDFT in component parameter_values (dimensionless)"
    legend_constants[17] = "RCDFPC in component parameter_values (dimensionless)"
    legend_constants[18] = "RCDFDP in component parameter_values (minute)"
    legend_states[0] = "PAR1 in component perfusion_pressure (mmHg)"
    legend_algebraic[2] = "MDFLW in component proximal_tubular_and_macula_densa_flow (L_per_minute)"
    legend_algebraic[3] = "RNAUG2 in component renal_autoregulatory_feedback_factor (dimensionless)"
    legend_constants[19] = "RNAUGN in component parameter_values (minute_per_L)"
    legend_constants[20] = "RNAULL in component parameter_values (dimensionless)"
    legend_constants[21] = "RNAUUL in component parameter_values (dimensionless)"
    legend_constants[22] = "RNAUAD in component parameter_values (per_minute)"
    legend_algebraic[4] = "RNAUG1 in component renal_autoregulatory_feedback_factor (dimensionless)"
    legend_algebraic[5] = "RNAUG1T in component renal_autoregulatory_feedback_factor (dimensionless)"
    legend_states[1] = "RNAUG3 in component renal_autoregulatory_feedback_factor (dimensionless)"
    legend_constants[63] = "AUMK in component autonomic_effect_on_AAR (dimensionless)"
    legend_constants[23] = "ARF in component parameter_values (dimensionless)"
    legend_constants[62] = "AUMKT in component autonomic_effect_on_AAR (dimensionless)"
    legend_constants[65] = "ANMAR in component angiotensin_effect_on_AAR (dimensionless)"
    legend_constants[24] = "ANMAM in component parameter_values (dimensionless)"
    legend_constants[25] = "ANMARL in component parameter_values (dimensionless)"
    legend_constants[64] = "ANMAR1 in component angiotensin_effect_on_AAR (dimensionless)"
    legend_algebraic[6] = "AAR1 in component AAR_calculation (mmHg_minute_per_L)"
    legend_constants[26] = "AARK in component parameter_values (mmHg_minute_per_L)"
    legend_algebraic[7] = "AAR in component atrial_natriuretic_peptide_effect_on_AAR (mmHg_minute_per_L)"
    legend_constants[27] = "ANPXAF in component parameter_values (mmHg_minute_per_L)"
    legend_constants[28] = "AARLL in component parameter_values (mmHg_minute_per_L)"
    legend_algebraic[8] = "AART in component atrial_natriuretic_peptide_effect_on_AAR (mmHg_minute_per_L)"
    legend_constants[66] = "AUMK2 in component autonomic_effect_on_EAR (dimensionless)"
    legend_constants[29] = "AUMK1 in component parameter_values (dimensionless)"
    legend_constants[67] = "ANMER in component angiotensin_effect_on_EAR (dimensionless)"
    legend_constants[30] = "ANMEM in component parameter_values (dimensionless)"
    legend_algebraic[9] = "RNAUG4 in component effect_of_renal_autoregulatory_feedback_on_EAR (dimensionless)"
    legend_constants[31] = "EFAFR in component parameter_values (dimensionless)"
    legend_algebraic[10] = "EAR in component EAR_calculation (mmHg_minute_per_L)"
    legend_constants[32] = "EARK in component parameter_values (mmHg_minute_per_L)"
    legend_constants[33] = "EARLL in component parameter_values (mmHg_minute_per_L)"
    legend_algebraic[11] = "EAR1 in component EAR_calculation (mmHg_minute_per_L)"
    legend_algebraic[12] = "RR in component total_renal_resistance (mmHg_minute_per_L)"
    legend_algebraic[13] = "RFN in component normal_renal_blood_flow (L_per_minute)"
    legend_algebraic[24] = "RBF in component actual_renal_blood_flow (L_per_minute)"
    legend_constants[34] = "REK in component parameter_values (dimensionless)"
    legend_algebraic[14] = "GFN in component glomerular_filtration_rate (L_per_minute)"
    legend_algebraic[15] = "GLPC in component glomerular_colloid_osmotic_pressure (mmHg)"
    legend_constants[35] = "GPPD in component parameter_values (dimensionless)"
    legend_constants[36] = "GLPCA in component parameter_values (mmHg)"
    legend_algebraic[16] = "EFAFPR in component glomerular_colloid_osmotic_pressure (dimensionless)"
    legend_algebraic[17] = "EFAFPR1 in component glomerular_colloid_osmotic_pressure (dimensionless)"
    legend_algebraic[18] = "GLP in component glomerular_pressure (mmHg)"
    legend_algebraic[19] = "APD in component glomerular_pressure (mmHg)"
    legend_algebraic[25] = "GFR in component glomerular_filtration_rate (L_per_minute)"
    legend_constants[37] = "PXTP in component parameter_values (mmHg)"
    legend_constants[38] = "GFLC in component parameter_values (L_per_minute_per_mmHg)"
    legend_constants[39] = "GFNLL in component parameter_values (L_per_minute)"
    legend_algebraic[20] = "PFL in component glomerular_filtration_rate (mmHg)"
    legend_algebraic[21] = "GFN1 in component glomerular_filtration_rate (L_per_minute)"
    legend_constants[40] = "MDFL1 in component parameter_values (dimensionless)"
    legend_algebraic[22] = "PTFL in component proximal_tubular_and_macula_densa_flow (L_per_minute)"
    legend_algebraic[23] = "MDFLWT in component proximal_tubular_and_macula_densa_flow (L_per_minute)"
    legend_algebraic[27] = "RTSPPC in component renal_tissue_osmotic_pressure (mmHg)"
    legend_constants[41] = "RTPPR in component parameter_values (dimensionless)"
    legend_constants[42] = "RTPPRS in component parameter_values (mmHg)"
    legend_algebraic[26] = "RTSPPC1 in component renal_tissue_osmotic_pressure (mmHg)"
    legend_algebraic[49] = "UROD in component actual_urea_excretion_rate (mOsm_per_minute)"
    legend_states[2] = "PLUR in component glomerular_urea_concentration (mOsm)"
    legend_constants[43] = "URFORM in component parameter_values (mOsm_per_minute)"
    legend_algebraic[1] = "PLURC in component plasma_urea_concentration (mOsm_per_litre)"
    legend_algebraic[28] = "RCPRS in component peritubular_capillary_pressure (mmHg)"
    legend_constants[44] = "RFABX in component parameter_values (dimensionless)"
    legend_constants[45] = "RVRS in component parameter_values (mmHg_minute_per_L)"
    legend_algebraic[33] = "RFABD in component peritubular_capillary_reabsorption_factor (dimensionless)"
    legend_constants[46] = "RTSPRS in component parameter_values (mmHg)"
    legend_constants[47] = "RABSC in component parameter_values (per_mmHg)"
    legend_constants[48] = "RFABDP in component parameter_values (dimensionless)"
    legend_constants[49] = "RFABDM in component parameter_values (dimensionless)"
    legend_algebraic[29] = "RABSPR in component peritubular_capillary_reabsorption_factor (mmHg)"
    legend_algebraic[30] = "RFAB1 in component peritubular_capillary_reabsorption_factor (dimensionless)"
    legend_algebraic[31] = "RFAB in component peritubular_capillary_reabsorption_factor (dimensionless)"
    legend_algebraic[32] = "RFABD1 in component peritubular_capillary_reabsorption_factor (dimensionless)"
    legend_algebraic[34] = "DTNAI in component distal_tubular_Na_delivery (monovalent_mEq_per_minute)"
    legend_algebraic[36] = "DTNARA in component Na_reabsorption_into_distal_tubules (monovalent_mEq_per_minute)"
    legend_constants[50] = "DTNAR in component parameter_values (monovalent_mEq_per_minute)"
    legend_constants[51] = "DIURET in component parameter_values (dimensionless)"
    legend_constants[52] = "AHMNAR in component parameter_values (dimensionless)"
    legend_constants[53] = "DTNARL in component parameter_values (monovalent_mEq_per_minute)"
    legend_algebraic[35] = "DTNARA1 in component Na_reabsorption_into_distal_tubules (monovalent_mEq_per_minute)"
    legend_constants[69] = "DTNANG in component angiotensin_induced_Na_reabsorption_into_distal_tubules (monovalent_mEq_per_minute)"
    legend_constants[54] = "ANMNAM in component parameter_values (dimensionless)"
    legend_constants[68] = "DTNANG1 in component angiotensin_induced_Na_reabsorption_into_distal_tubules (monovalent_mEq_per_minute)"
    legend_algebraic[37] = "DTKI in component distal_tubular_K_delivery (monovalent_mEq_per_minute)"
    legend_algebraic[38] = "RFABK in component effect_of_physical_forces_on_distal_K_reabsorption (monovalent_mEq_per_minute)"
    legend_constants[55] = "RFABKM in component parameter_values (monovalent_mEq_per_minute)"
    legend_algebraic[40] = "MDFLK in component effect_of_fluid_flow_on_distal_K_reabsorption (monovalent_mEq_per_minute)"
    legend_constants[56] = "MDFLKM in component parameter_values (monovalent_mEq_per_litre)"
    legend_algebraic[39] = "MDFLK1 in component effect_of_fluid_flow_on_distal_K_reabsorption (monovalent_mEq_per_minute)"
    legend_algebraic[46] = "KODN in component normal_K_excretion (monovalent_mEq_per_minute)"
    legend_algebraic[54] = "VUDN in component normal_urine_volume (L_per_minute)"
    legend_states[3] = "DTKA in component K_reabsorption_into_distal_tubules (monovalent_mEq_per_minute)"
    legend_algebraic[41] = "DTKSC in component K_secretion_from_distal_tubules (monovalent_mEq_per_minute)"
    legend_constants[57] = "ANMKEM in component parameter_values (dimensionless)"
    legend_constants[58] = "ANMKEL in component parameter_values (dimensionless)"
    legend_constants[59] = "CKEEX in component parameter_values (dimensionless)"
    legend_constants[70] = "ANMKE1 in component K_secretion_from_distal_tubules (dimensionless)"
    legend_constants[71] = "ANMKE in component K_secretion_from_distal_tubules (dimensionless)"
    legend_algebraic[43] = "NODN in component normal_Na_excretion (monovalent_mEq_per_minute)"
    legend_algebraic[42] = "NODN1 in component normal_Na_excretion (monovalent_mEq_per_minute)"
    legend_algebraic[44] = "KODN1 in component normal_K_excretion (monovalent_mEq_per_minute)"
    legend_algebraic[47] = "DTURI in component normal_urea_excretion (mOsm_per_minute)"
    legend_algebraic[50] = "OSMOPN1 in component normal_osmolar_and_water_excretion (mOsm_per_minute)"
    legend_algebraic[51] = "OSMOPN in component normal_osmolar_and_water_excretion (mOsm_per_minute)"
    legend_algebraic[52] = "OSMOP1T in component normal_urine_volume (mOsm_per_minute)"
    legend_algebraic[53] = "OSMOP1 in component normal_urine_volume (mOsm_per_minute)"
    legend_algebraic[45] = "NOD in component actual_Na_excretion_rate (monovalent_mEq_per_minute)"
    legend_algebraic[48] = "KOD in component actual_K_excretion_rate (monovalent_mEq_per_minute)"
    legend_algebraic[55] = "VUD in component actual_urine_volume (L_per_minute)"
    legend_constants[60] = "RNAGTC in component parameter_values (minute)"
    legend_constants[61] = "GFNDMP in component parameter_values (dimensionless)"
    legend_rates[0] = "d/dt PAR1 in component perfusion_pressure (mmHg)"
    legend_rates[1] = "d/dt RNAUG3 in component renal_autoregulatory_feedback_factor (dimensionless)"
    legend_rates[2] = "d/dt PLUR in component glomerular_urea_concentration (mOsm)"
    legend_rates[3] = "d/dt DTKA in component K_reabsorption_into_distal_tubules (monovalent_mEq_per_minute)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 1.0
    constants[1] = 1.037
    constants[2] = 1.0
    constants[3] = 0.987545
    constants[4] = 1.0
    constants[5] = 1.00066
    constants[6] = 4.44092
    constants[7] = 142.035
    constants[8] = 0.39984739
    constants[9] = 1.0
    constants[10] = 103.525
    constants[11] = 1.0
    constants[12] = 29.9941
    constants[13] = 39.8952
    constants[14] = 0
    constants[15] = 0
    constants[16] = 0
    constants[17] = 0
    constants[18] = 2000
    states[0] = 103.525
    constants[19] = 0.6
    constants[20] = 0.3
    constants[21] = 10
    constants[22] = 0
    states[1] = 0.0
    constants[23] = 0.5
    constants[24] = 1.4
    constants[25] = 0.86
    constants[26] = 1
    constants[27] = 1.5
    constants[28] = 4
    constants[29] = 0.3
    constants[30] = 1.6
    constants[31] = 0
    constants[32] = 1
    constants[33] = 24
    constants[34] = 1
    constants[35] = 1.0
    constants[36] = 1.0
    constants[37] = 8
    constants[38] = 0.0208333
    constants[39] = 0.001
    constants[40] = 10
    constants[41] = 0.9
    constants[42] = 15.2
    states[2] = 159.549
    constants[43] = 0.24
    constants[44] = 0.8
    constants[45] = 19.167
    constants[46] = 6
    constants[47] = 0.5
    constants[48] = 1
    constants[49] = 0.3
    constants[50] = 0.675
    constants[51] = 1
    constants[52] = 0.3
    constants[53] = 1e-06
    constants[54] = 1
    constants[55] = 0.03
    constants[56] = 0.667
    states[3] = 0.0367573
    constants[57] = 2
    constants[58] = 0.3
    constants[59] = 4
    constants[60] = 15
    constants[61] = 3
    constants[62] = (constants[5]-1.00000)*constants[23]+1.00000
    constants[63] = custom_piecewise([less(constants[62] , 0.800000), 0.800000 , True, constants[62]])
    constants[64] = (constants[3]-1.00000)*constants[24]+1.00000
    constants[65] = custom_piecewise([less(constants[64] , constants[25]), constants[25] , True, constants[64]])
    constants[66] = (constants[63]-1.00000)*constants[29]+1.00000
    constants[67] = (constants[3]-1.00000)*constants[30]+1.00000
    constants[68] = ((constants[3]-1.00000)*constants[54]+1.00000)*0.100000
    constants[69] = custom_piecewise([less(constants[68] , 0.00000), 0.00000 , True, constants[68]])
    constants[70] = (constants[3]-1.00000)*constants[57]+1.00000
    constants[71] = custom_piecewise([less(constants[70] , constants[58]), constants[58] , True, constants[70]])
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = ((100.000+(constants[10]-100.000)*constants[17])-states[0])/constants[18]
    algebraic[0] = custom_piecewise([greater(constants[15] , 0.00000) & less_equal(constants[16] , 0.00000), constants[15] , greater(constants[16] , 0.00000), states[0] , True, constants[10]-constants[14]])
    rootfind_0(voi, constants, rates, states, algebraic)
    rates[1] = (algebraic[3]-1.00000)*constants[22]
    algebraic[1] = states[2]/constants[13]
    algebraic[47] = (power(algebraic[14], 2.00000))*algebraic[1]*3.84000
    algebraic[49] = algebraic[47]*constants[34]
    rates[2] = constants[43]-algebraic[49]
    algebraic[34] = algebraic[2]*constants[7]*0.00616190
    algebraic[37] = (algebraic[34]*constants[6])/constants[7]
    algebraic[26] = algebraic[15]*constants[41]-constants[42]
    algebraic[27] = custom_piecewise([less(algebraic[26] , 1.00000), 1.00000 , True, algebraic[26]])
    algebraic[28] = ((algebraic[13]-1.20000)*constants[44]+1.20000)*constants[45]
    algebraic[29] = ((algebraic[15]+constants[46])-algebraic[28])-algebraic[27]
    algebraic[30] = algebraic[29]*constants[47]
    algebraic[31] = algebraic[30]
    algebraic[32] = (algebraic[31]-1.00000)*constants[49]+1.00000
    algebraic[33] = custom_piecewise([less(algebraic[32] , 0.000100000), 0.000100000 , True, algebraic[32]])
    algebraic[38] = (algebraic[33]-1.00000)*constants[55]
    algebraic[39] = (algebraic[2]-1.00000)*constants[56]+1.00000
    algebraic[40] = custom_piecewise([less(algebraic[39] , 0.100000), 0.100000 , True, algebraic[39]])
    algebraic[41] = ((power(constants[6]/4.40000, constants[59]))*constants[1]*0.0800000*algebraic[40])/constants[71]
    algebraic[44] = ((algebraic[37]+algebraic[41])-states[3])-algebraic[38]
    algebraic[46] = custom_piecewise([less(algebraic[44] , 0.00000), 0.00000 , True, algebraic[44]])
    algebraic[35] = ((constants[2]*algebraic[33]*constants[50])/constants[51])*((constants[0]-1.00000)*constants[52]+1.00000)
    algebraic[36] = custom_piecewise([less(algebraic[35] , constants[53]), constants[53] , True, algebraic[35]])
    algebraic[42] = (algebraic[34]-algebraic[36])-constants[69]
    algebraic[43] = custom_piecewise([less(algebraic[42] , 1.00000e-08), 1.00000e-08 , True, algebraic[42]])
    algebraic[50] = algebraic[47]+2.00000*(algebraic[43]+algebraic[46])
    algebraic[51] = custom_piecewise([greater(algebraic[50] , 0.600000), 0.600000 , True, algebraic[50]])
    algebraic[52] = algebraic[50]-0.600000
    algebraic[53] = custom_piecewise([less(algebraic[52] , 0.00000), 0.00000 , True, algebraic[52]])
    algebraic[54] = algebraic[51]/(600.000*constants[0])+algebraic[53]/360.000
    rates[3] = ((algebraic[46]/algebraic[54])*0.000451800-states[3])*1.00000
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = custom_piecewise([greater(constants[15] , 0.00000) & less_equal(constants[16] , 0.00000), constants[15] , greater(constants[16] , 0.00000), states[0] , True, constants[10]-constants[14]])
    algebraic[1] = states[2]/constants[13]
    algebraic[47] = (power(algebraic[14], 2.00000))*algebraic[1]*3.84000
    algebraic[49] = algebraic[47]*constants[34]
    algebraic[34] = algebraic[2]*constants[7]*0.00616190
    algebraic[37] = (algebraic[34]*constants[6])/constants[7]
    algebraic[26] = algebraic[15]*constants[41]-constants[42]
    algebraic[27] = custom_piecewise([less(algebraic[26] , 1.00000), 1.00000 , True, algebraic[26]])
    algebraic[28] = ((algebraic[13]-1.20000)*constants[44]+1.20000)*constants[45]
    algebraic[29] = ((algebraic[15]+constants[46])-algebraic[28])-algebraic[27]
    algebraic[30] = algebraic[29]*constants[47]
    algebraic[31] = algebraic[30]
    algebraic[32] = (algebraic[31]-1.00000)*constants[49]+1.00000
    algebraic[33] = custom_piecewise([less(algebraic[32] , 0.000100000), 0.000100000 , True, algebraic[32]])
    algebraic[38] = (algebraic[33]-1.00000)*constants[55]
    algebraic[39] = (algebraic[2]-1.00000)*constants[56]+1.00000
    algebraic[40] = custom_piecewise([less(algebraic[39] , 0.100000), 0.100000 , True, algebraic[39]])
    algebraic[41] = ((power(constants[6]/4.40000, constants[59]))*constants[1]*0.0800000*algebraic[40])/constants[71]
    algebraic[44] = ((algebraic[37]+algebraic[41])-states[3])-algebraic[38]
    algebraic[46] = custom_piecewise([less(algebraic[44] , 0.00000), 0.00000 , True, algebraic[44]])
    algebraic[35] = ((constants[2]*algebraic[33]*constants[50])/constants[51])*((constants[0]-1.00000)*constants[52]+1.00000)
    algebraic[36] = custom_piecewise([less(algebraic[35] , constants[53]), constants[53] , True, algebraic[35]])
    algebraic[42] = (algebraic[34]-algebraic[36])-constants[69]
    algebraic[43] = custom_piecewise([less(algebraic[42] , 1.00000e-08), 1.00000e-08 , True, algebraic[42]])
    algebraic[50] = algebraic[47]+2.00000*(algebraic[43]+algebraic[46])
    algebraic[51] = custom_piecewise([greater(algebraic[50] , 0.600000), 0.600000 , True, algebraic[50]])
    algebraic[52] = algebraic[50]-0.600000
    algebraic[53] = custom_piecewise([less(algebraic[52] , 0.00000), 0.00000 , True, algebraic[52]])
    algebraic[54] = algebraic[51]/(600.000*constants[0])+algebraic[53]/360.000
    algebraic[24] = constants[34]*algebraic[13]
    algebraic[25] = algebraic[14]*constants[34]
    algebraic[45] = algebraic[43]*constants[34]
    algebraic[48] = algebraic[46]*constants[34]
    algebraic[55] = algebraic[54]*constants[34]
    return algebraic

initialGuess0 = None
def rootfind_0(voi, constants, rates, states, algebraic):
    """Calculate values of algebraic variables for DAE"""
    from scipy.optimize import fsolve
    global initialGuess0
    if initialGuess0 is None: initialGuess0 = ones(22)*0.1
    if not iterable(voi):
        soln = fsolve(residualSN_0, initialGuess0, args=(algebraic, voi, constants, rates, states), xtol=1E-6)
        initialGuess0 = soln
        algebraic[2] = soln[0]
        algebraic[3] = soln[1]
        algebraic[4] = soln[2]
        algebraic[5] = soln[3]
        algebraic[6] = soln[4]
        algebraic[7] = soln[5]
        algebraic[8] = soln[6]
        algebraic[9] = soln[7]
        algebraic[10] = soln[8]
        algebraic[11] = soln[9]
        algebraic[12] = soln[10]
        algebraic[13] = soln[11]
        algebraic[14] = soln[12]
        algebraic[15] = soln[13]
        algebraic[16] = soln[14]
        algebraic[17] = soln[15]
        algebraic[18] = soln[16]
        algebraic[19] = soln[17]
        algebraic[20] = soln[18]
        algebraic[21] = soln[19]
        algebraic[22] = soln[20]
        algebraic[23] = soln[21]
    else:
        for (i,t) in enumerate(voi):
            soln = fsolve(residualSN_0, initialGuess0, args=(algebraic[:,i], voi[i], constants, rates[:i], states[:,i]), xtol=1E-6)
            initialGuess0 = soln
            algebraic[2][i] = soln[0]
            algebraic[3][i] = soln[1]
            algebraic[4][i] = soln[2]
            algebraic[5][i] = soln[3]
            algebraic[6][i] = soln[4]
            algebraic[7][i] = soln[5]
            algebraic[8][i] = soln[6]
            algebraic[9][i] = soln[7]
            algebraic[10][i] = soln[8]
            algebraic[11][i] = soln[9]
            algebraic[12][i] = soln[10]
            algebraic[13][i] = soln[11]
            algebraic[14][i] = soln[12]
            algebraic[15][i] = soln[13]
            algebraic[16][i] = soln[14]
            algebraic[17][i] = soln[15]
            algebraic[18][i] = soln[16]
            algebraic[19][i] = soln[17]
            algebraic[20][i] = soln[18]
            algebraic[21][i] = soln[19]
            algebraic[22][i] = soln[20]
            algebraic[23][i] = soln[21]

def residualSN_0(algebraicCandidate, algebraic, voi, constants, rates, states):
    resid = array([0.0] * 22)
    algebraic[2] = algebraicCandidate[0]
    algebraic[3] = algebraicCandidate[1]
    algebraic[4] = algebraicCandidate[2]
    algebraic[5] = algebraicCandidate[3]
    algebraic[6] = algebraicCandidate[4]
    algebraic[7] = algebraicCandidate[5]
    algebraic[8] = algebraicCandidate[6]
    algebraic[9] = algebraicCandidate[7]
    algebraic[10] = algebraicCandidate[8]
    algebraic[11] = algebraicCandidate[9]
    algebraic[12] = algebraicCandidate[10]
    algebraic[13] = algebraicCandidate[11]
    algebraic[14] = algebraicCandidate[12]
    algebraic[15] = algebraicCandidate[13]
    algebraic[16] = algebraicCandidate[14]
    algebraic[17] = algebraicCandidate[15]
    algebraic[18] = algebraicCandidate[16]
    algebraic[19] = algebraicCandidate[17]
    algebraic[20] = algebraicCandidate[18]
    algebraic[21] = algebraicCandidate[19]
    algebraic[22] = algebraicCandidate[20]
    algebraic[23] = algebraicCandidate[21]
    resid[0] = (algebraic[5]-((algebraic[2]-1.00000)*constants[19]+1.00000))
    resid[1] = (algebraic[4]-(custom_piecewise([less(algebraic[5] , constants[20]), constants[20] , greater(algebraic[5] , constants[21]), constants[21] , True, algebraic[5]])))
    resid[2] = (algebraic[3]-(algebraic[4]-states[1]))
    resid[3] = (algebraic[6]-constants[26]*constants[11]*constants[63]*algebraic[3]*constants[65]*40.0000*constants[9])
    resid[4] = (algebraic[8]-((algebraic[6]-constants[4]*constants[27])+constants[27]))
    resid[5] = (algebraic[7]-(custom_piecewise([less(algebraic[8] , constants[28]), constants[28] , True, algebraic[8]])))
    resid[6] = (algebraic[9]-((algebraic[3]-1.00000)*constants[31]+1.00000))
    resid[7] = (algebraic[11]-43.3330*constants[32]*constants[67]*algebraic[9]*constants[9]*constants[66])
    resid[8] = (algebraic[10]-(custom_piecewise([less(algebraic[11] , constants[33]), constants[33] , True, algebraic[11]])))
    resid[9] = (algebraic[12]-(algebraic[7]+algebraic[10]))
    resid[10] = (algebraic[13]-algebraic[0]/algebraic[12])
    resid[11] = (algebraic[17]-(algebraic[13]*(1.00000-constants[8]))/(algebraic[13]*(1.00000-constants[8])-algebraic[14]))
    resid[12] = (algebraic[16]-(custom_piecewise([less(algebraic[17] , 1.00000), 1.00000 , True, algebraic[17]])))
    resid[13] = (algebraic[15]-(custom_piecewise([greater(constants[36] , 0.00000), (power(algebraic[16], 1.35000))*constants[12]*0.980000 , True, constants[12]+4.00000])))
    resid[14] = (algebraic[19]-algebraic[7]*algebraic[13])
    resid[15] = (algebraic[18]-(algebraic[0]-algebraic[19]))
    resid[16] = (algebraic[20]-((algebraic[18]-algebraic[15])-constants[37]))
    resid[17] = (algebraic[21]-algebraic[20]*constants[38])
    resid[18] = (algebraic[14]-(custom_piecewise([less(algebraic[21] , constants[39]), constants[39] , True, algebraic[21]])))
    resid[19] = (algebraic[22]-algebraic[14]*8.00000)
    resid[20] = (algebraic[23]-((algebraic[22]-1.00000)*constants[40]+1.00000))
    resid[21] = (algebraic[2]-(custom_piecewise([less(algebraic[23] , 0.00000), 0.00000 , True, algebraic[23]])))
    return resid

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