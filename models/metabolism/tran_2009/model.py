# Size of variable arrays:
sizeAlgebraic = 50
sizeStates = 11
sizeConstants = 71
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (millisecond)"
    legend_algebraic[3] = "SOVFThick in component sarcomere_geometry (dimensionless)"
    legend_algebraic[4] = "SOVFThin in component sarcomere_geometry (dimensionless)"
    legend_algebraic[0] = "sovr_ze in component sarcomere_geometry (micrometre)"
    legend_algebraic[1] = "sovr_cle in component sarcomere_geometry (micrometre)"
    legend_algebraic[2] = "len_sovr in component sarcomere_geometry (micrometre)"
    legend_constants[0] = "len_thin in component model_parameters (micrometre)"
    legend_constants[1] = "len_thick in component model_parameters (micrometre)"
    legend_constants[2] = "len_hbare in component model_parameters (micrometre)"
    legend_states[0] = "SL in component normalised_active_and_passive_force (micrometre)"
    legend_states[1] = "TRPNCaL in component Ca_binding_to_troponin_to_thin_filament_regulation (dimensionless)"
    legend_states[2] = "TRPNCaH in component Ca_binding_to_troponin_to_thin_filament_regulation (dimensionless)"
    legend_algebraic[7] = "dTRPNCaL in component Ca_binding_to_troponin_to_thin_filament_regulation (first_order_rate_constant)"
    legend_algebraic[10] = "dTRPNCaH in component Ca_binding_to_troponin_to_thin_filament_regulation (first_order_rate_constant)"
    legend_algebraic[12] = "kn_pT in component Ca_binding_to_troponin_to_thin_filament_regulation (first_order_rate_constant)"
    legend_algebraic[18] = "kp_nT in component Ca_binding_to_troponin_to_thin_filament_regulation (first_order_rate_constant)"
    legend_constants[60] = "H in component Ca_binding_to_troponin_to_thin_filament_regulation (micromolar)"
    legend_constants[62] = "H_cons in component Ca_binding_to_troponin_to_thin_filament_regulation (micromolar)"
    legend_constants[63] = "konT in component Ca_binding_to_troponin_to_thin_filament_regulation (second_order_rate_constant)"
    legend_constants[58] = "koffLT in component Ca_binding_to_troponin_to_thin_filament_regulation (first_order_rate_constant)"
    legend_constants[59] = "koffHT in component Ca_binding_to_troponin_to_thin_filament_regulation (first_order_rate_constant)"
    legend_constants[3] = "Qkon in component Ca_binding_to_troponin_to_thin_filament_regulation (dimensionless)"
    legend_constants[4] = "Qkoff in component Ca_binding_to_troponin_to_thin_filament_regulation (dimensionless)"
    legend_constants[5] = "Qkn_p in component Ca_binding_to_troponin_to_thin_filament_regulation (dimensionless)"
    legend_constants[6] = "Qkp_n in component Ca_binding_to_troponin_to_thin_filament_regulation (dimensionless)"
    legend_constants[7] = "kon in component Ca_binding_to_troponin_to_thin_filament_regulation (second_order_rate_constant)"
    legend_constants[8] = "koffL in component Ca_binding_to_troponin_to_thin_filament_regulation (first_order_rate_constant)"
    legend_constants[9] = "koffH in component Ca_binding_to_troponin_to_thin_filament_regulation (first_order_rate_constant)"
    legend_constants[10] = "perm50 in component Ca_binding_to_troponin_to_thin_filament_regulation (dimensionless)"
    legend_constants[11] = "nperm in component Ca_binding_to_troponin_to_thin_filament_regulation (dimensionless)"
    legend_constants[12] = "kn_p in component Ca_binding_to_troponin_to_thin_filament_regulation (first_order_rate_constant)"
    legend_constants[13] = "kp_n in component Ca_binding_to_troponin_to_thin_filament_regulation (first_order_rate_constant)"
    legend_constants[14] = "koffmod in component Ca_binding_to_troponin_to_thin_filament_regulation (dimensionless)"
    legend_algebraic[6] = "Tropreg in component Ca_binding_to_troponin_to_thin_filament_regulation (dimensionless)"
    legend_algebraic[9] = "permtot in component Ca_binding_to_troponin_to_thin_filament_regulation (dimensionless)"
    legend_algebraic[15] = "inprmt in component Ca_binding_to_troponin_to_thin_filament_regulation (dimensionless)"
    legend_constants[15] = "pH in component Ca_binding_to_troponin_to_thin_filament_regulation (dimensionless)"
    legend_constants[16] = "m in component Ca_binding_to_troponin_to_thin_filament_regulation (dimensionless)"
    legend_constants[17] = "kdHCa in component Ca_binding_to_troponin_to_thin_filament_regulation (micromolar)"
    legend_constants[18] = "TmpC in component model_parameters (celsius)"
    legend_constants[19] = "Cai in component model_parameters (micromolar)"
    legend_constants[64] = "fappT in component thin_filament_regulation_and_crossbridge_cycling_rates (first_order_rate_constant)"
    legend_algebraic[16] = "gappT in component thin_filament_regulation_and_crossbridge_cycling_rates (first_order_rate_constant)"
    legend_algebraic[23] = "hfT in component thin_filament_regulation_and_crossbridge_cycling_rates (first_order_rate_constant)"
    legend_algebraic[24] = "hbT in component thin_filament_regulation_and_crossbridge_cycling_rates (first_order_rate_constant)"
    legend_algebraic[26] = "gxbT in component thin_filament_regulation_and_crossbridge_cycling_rates (first_order_rate_constant)"
    legend_constants[20] = "fapp in component thin_filament_regulation_and_crossbridge_cycling_rates (first_order_rate_constant)"
    legend_constants[21] = "gapp in component thin_filament_regulation_and_crossbridge_cycling_rates (first_order_rate_constant)"
    legend_constants[22] = "hf in component thin_filament_regulation_and_crossbridge_cycling_rates (first_order_rate_constant)"
    legend_constants[23] = "hb in component thin_filament_regulation_and_crossbridge_cycling_rates (first_order_rate_constant)"
    legend_constants[24] = "gxb in component thin_filament_regulation_and_crossbridge_cycling_rates (first_order_rate_constant)"
    legend_constants[25] = "gslmod in component thin_filament_regulation_and_crossbridge_cycling_rates (dimensionless)"
    legend_algebraic[19] = "hfmd in component thin_filament_regulation_and_crossbridge_cycling_rates (dimensionless)"
    legend_algebraic[21] = "hbmd in component thin_filament_regulation_and_crossbridge_cycling_rates (dimensionless)"
    legend_constants[26] = "hfmdc in component thin_filament_regulation_and_crossbridge_cycling_rates (dimensionless)"
    legend_constants[27] = "hbmdc in component thin_filament_regulation_and_crossbridge_cycling_rates (dimensionless)"
    legend_constants[28] = "sigmap in component thin_filament_regulation_and_crossbridge_cycling_rates (dimensionless)"
    legend_constants[29] = "sigman in component thin_filament_regulation_and_crossbridge_cycling_rates (dimensionless)"
    legend_constants[30] = "xbmodsp in component thin_filament_regulation_and_crossbridge_cycling_rates (dimensionless)"
    legend_constants[31] = "Qfapp in component thin_filament_regulation_and_crossbridge_cycling_rates (dimensionless)"
    legend_constants[32] = "Qgapp in component thin_filament_regulation_and_crossbridge_cycling_rates (dimensionless)"
    legend_constants[33] = "Qhf in component thin_filament_regulation_and_crossbridge_cycling_rates (dimensionless)"
    legend_constants[34] = "Qhb in component thin_filament_regulation_and_crossbridge_cycling_rates (dimensionless)"
    legend_constants[35] = "Qgxb in component thin_filament_regulation_and_crossbridge_cycling_rates (dimensionless)"
    legend_algebraic[25] = "gxbmd in component thin_filament_regulation_and_crossbridge_cycling_rates (dimensionless)"
    legend_algebraic[13] = "gapslmd in component thin_filament_regulation_and_crossbridge_cycling_rates (dimensionless)"
    legend_constants[36] = "x_0 in component model_parameters (micrometre)"
    legend_states[3] = "xXBpostr in component mean_strain_of_strongly_bound_states (micrometre)"
    legend_states[4] = "xXBprer in component mean_strain_of_strongly_bound_states (micrometre)"
    legend_states[5] = "XBpostr in component regulation_and_crossbridge_cycling_state_equations (dimensionless)"
    legend_states[6] = "XBprer in component regulation_and_crossbridge_cycling_state_equations (dimensionless)"
    legend_algebraic[34] = "dXBpostr in component regulation_and_crossbridge_cycling_state_equations (first_order_rate_constant)"
    legend_algebraic[31] = "dXBprer in component regulation_and_crossbridge_cycling_state_equations (first_order_rate_constant)"
    legend_constants[65] = "alpha1_plus in component regulation_and_crossbridge_cycling_state_equations (first_order_rate_constant)"
    legend_algebraic[27] = "alpha2_plus in component regulation_and_crossbridge_cycling_state_equations (first_order_rate_constant)"
    legend_algebraic[28] = "alpha3_plus in component regulation_and_crossbridge_cycling_state_equations (first_order_rate_constant)"
    legend_algebraic[29] = "alpha1_minus in component regulation_and_crossbridge_cycling_state_equations (first_order_rate_constant)"
    legend_algebraic[30] = "alpha2_minus in component regulation_and_crossbridge_cycling_state_equations (first_order_rate_constant)"
    legend_algebraic[33] = "alpha3_minus in component regulation_and_crossbridge_cycling_state_equations (first_order_rate_constant)"
    legend_constants[37] = "kMgATP in component regulation_and_crossbridge_cycling_state_equations (micromolar2)"
    legend_constants[38] = "kdADP in component regulation_and_crossbridge_cycling_state_equations (micromolar)"
    legend_constants[39] = "xPi_cons in component regulation_and_crossbridge_cycling_state_equations (micromolar)"
    legend_constants[40] = "MgATP_cons in component regulation_and_crossbridge_cycling_state_equations (micromolar)"
    legend_algebraic[32] = "fxbT in component regulation_and_crossbridge_cycling_state_equations (first_order_rate_constant)"
    legend_states[7] = "N_NoXB in component regulation_and_crossbridge_cycling_state_equations (dimensionless)"
    legend_states[8] = "P_NoXB in component regulation_and_crossbridge_cycling_state_equations (dimensionless)"
    legend_states[9] = "P in component regulation_and_crossbridge_cycling_state_equations (dimensionless)"
    legend_algebraic[22] = "N in component regulation_and_crossbridge_cycling_state_equations (dimensionless)"
    legend_constants[41] = "MgADP_cons in component regulation_and_crossbridge_cycling_state_equations (micromolar)"
    legend_constants[42] = "xPi in component regulation_and_crossbridge_cycling_state_equations (micromolar)"
    legend_constants[43] = "MgATP in component regulation_and_crossbridge_cycling_state_equations (micromolar)"
    legend_constants[44] = "MgADP in component regulation_and_crossbridge_cycling_state_equations (micromolar)"
    legend_algebraic[38] = "dxXBpostr in component mean_strain_of_strongly_bound_states (micrometre_per_millisecond)"
    legend_algebraic[36] = "dxXBprer in component mean_strain_of_strongly_bound_states (micrometre_per_millisecond)"
    legend_constants[45] = "xPsi in component mean_strain_of_strongly_bound_states (dimensionless)"
    legend_algebraic[35] = "dutyprer in component mean_strain_of_strongly_bound_states (dimensionless)"
    legend_algebraic[37] = "dutypostr in component mean_strain_of_strongly_bound_states (dimensionless)"
    legend_constants[61] = "dSL in component normalised_active_and_passive_force (micrometre_per_millisecond)"
    legend_constants[67] = "SSXBpostr in component normalised_active_and_passive_force (dimensionless)"
    legend_algebraic[39] = "SSXBprer in component normalised_active_and_passive_force (dimensionless)"
    legend_constants[46] = "kxb in component normalised_active_and_passive_force (millinewton_per_millimetre2)"
    legend_constants[68] = "Fnordv in component normalised_active_and_passive_force (millinewton_micrometre_per_millimetre2)"
    legend_algebraic[5] = "force in component normalised_active_and_passive_force (millinewton_micrometre_per_millimetre2)"
    legend_algebraic[8] = "active in component normalised_active_and_passive_force (unit_normalised_force)"
    legend_algebraic[17] = "ppforce in component normalised_active_and_passive_force (unit_normalised_force)"
    legend_algebraic[11] = "ppforce_t in component normalised_active_and_passive_force (unit_normalised_force)"
    legend_algebraic[14] = "ppforce_c in component normalised_active_and_passive_force (unit_normalised_force)"
    legend_constants[69] = "preload in component normalised_active_and_passive_force (unit_normalised_force)"
    legend_algebraic[20] = "afterload in component normalised_active_and_passive_force (unit_normalised_force)"
    legend_states[10] = "intf in component normalised_active_and_passive_force (unit_normalised_force_millisecond)"
    legend_constants[47] = "SL_c in component normalised_active_and_passive_force (micrometre)"
    legend_constants[48] = "SLrest in component normalised_active_and_passive_force (micrometre)"
    legend_constants[49] = "SLset in component normalised_active_and_passive_force (micrometre)"
    legend_constants[50] = "PCon_t in component normalised_active_and_passive_force (unit_normalised_force)"
    legend_constants[51] = "PExp_t in component normalised_active_and_passive_force (per_micrometre)"
    legend_constants[52] = "PCon_c in component normalised_active_and_passive_force (unit_normalised_force)"
    legend_constants[53] = "PExp_c in component normalised_active_and_passive_force (per_micrometre)"
    legend_constants[54] = "KSE in component normalised_active_and_passive_force (unit_normalised_force_per_micrometre)"
    legend_constants[66] = "fxb in component normalised_active_and_passive_force (first_order_rate_constant)"
    legend_constants[55] = "SEon in component normalised_active_and_passive_force (dimensionless)"
    legend_algebraic[40] = "FrSBXB in component calculation_of_micromolar_per_millisecondes_of_Ca_for_apparent_Ca_binding (dimensionless)"
    legend_algebraic[41] = "dFrSBXB in component calculation_of_micromolar_per_millisecondes_of_Ca_for_apparent_Ca_binding (first_order_rate_constant)"
    legend_algebraic[43] = "dsovr_ze in component calculation_of_micromolar_per_millisecondes_of_Ca_for_apparent_Ca_binding (micrometre_per_millisecond)"
    legend_algebraic[44] = "dsovr_cle in component calculation_of_micromolar_per_millisecondes_of_Ca_for_apparent_Ca_binding (micrometre_per_millisecond)"
    legend_algebraic[45] = "dlen_sovr in component calculation_of_micromolar_per_millisecondes_of_Ca_for_apparent_Ca_binding (micrometre_per_millisecond)"
    legend_algebraic[46] = "dSOVFThick in component calculation_of_micromolar_per_millisecondes_of_Ca_for_apparent_Ca_binding (first_order_rate_constant)"
    legend_algebraic[47] = "dSOVFThin in component calculation_of_micromolar_per_millisecondes_of_Ca_for_apparent_Ca_binding (first_order_rate_constant)"
    legend_constants[56] = "kxb in component calculation_of_micromolar_per_millisecondes_of_Ca_for_apparent_Ca_binding (millinewton_per_millimetre2)"
    legend_algebraic[48] = "dforce in component calculation_of_micromolar_per_millisecondes_of_Ca_for_apparent_Ca_binding (millinewton_micrometre_per_millimetre2_per_millisecond)"
    legend_constants[57] = "Trop_conc in component calculation_of_micromolar_per_millisecondes_of_Ca_for_apparent_Ca_binding (micromolar)"
    legend_algebraic[42] = "TropTot in component calculation_of_micromolar_per_millisecondes_of_Ca_for_apparent_Ca_binding (micromolar)"
    legend_algebraic[49] = "dTropTot in component calculation_of_micromolar_per_millisecondes_of_Ca_for_apparent_Ca_binding (micromolar_per_millisecond)"
    legend_rates[1] = "d/dt TRPNCaL in component Ca_binding_to_troponin_to_thin_filament_regulation (dimensionless)"
    legend_rates[2] = "d/dt TRPNCaH in component Ca_binding_to_troponin_to_thin_filament_regulation (dimensionless)"
    legend_rates[7] = "d/dt N_NoXB in component regulation_and_crossbridge_cycling_state_equations (dimensionless)"
    legend_rates[8] = "d/dt P_NoXB in component regulation_and_crossbridge_cycling_state_equations (dimensionless)"
    legend_rates[9] = "d/dt P in component regulation_and_crossbridge_cycling_state_equations (dimensionless)"
    legend_rates[6] = "d/dt XBprer in component regulation_and_crossbridge_cycling_state_equations (dimensionless)"
    legend_rates[5] = "d/dt XBpostr in component regulation_and_crossbridge_cycling_state_equations (dimensionless)"
    legend_rates[4] = "d/dt xXBprer in component mean_strain_of_strongly_bound_states (micrometre)"
    legend_rates[3] = "d/dt xXBpostr in component mean_strain_of_strongly_bound_states (micrometre)"
    legend_rates[0] = "d/dt SL in component normalised_active_and_passive_force (micrometre)"
    legend_rates[10] = "d/dt intf in component normalised_active_and_passive_force (unit_normalised_force_millisecond)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 1.2
    constants[1] = 1.65
    constants[2] = 0.1
    states[0] = 2.2
    states[1] = 0.0147730085063734
    states[2] = 0.13066096561522
    constants[3] = 1.5
    constants[4] = 1.3
    constants[5] = 1.6
    constants[6] = 1.6
    constants[7] = 0.05
    constants[8] = 0.25
    constants[9] = 0.025
    constants[10] = 0.5
    constants[11] = 15
    constants[12] = 0.5
    constants[13] = 0.05
    constants[14] = 1
    constants[15] = 7.15
    constants[16] = 1
    constants[17] = 2e-2
    constants[18] = 22
    constants[19] = 200.0
    constants[20] = 0.5
    constants[21] = 0.07
    constants[22] = 2
    constants[23] = 0.4
    constants[24] = 0.07
    constants[25] = 6
    constants[26] = 5
    constants[27] = 0
    constants[28] = 8
    constants[29] = 1
    constants[30] = 0.2
    constants[31] = 6.25
    constants[32] = 2.5
    constants[33] = 6.25
    constants[34] = 6.25
    constants[35] = 6.25
    constants[36] = 0.007
    states[3] = 0.00700005394873882
    states[4] = 3.41212828972468e-8
    states[5] = 1.81017564383744e-6
    states[6] = 3.0494964880038e-7
    constants[37] = 15400e6
    constants[38] = 4
    constants[39] = 2e3
    constants[40] = 5e3
    states[7] = 0.999999959256274
    states[8] = 4.07437173988636e-8
    states[9] = 0.999997834540066
    constants[41] = 36
    constants[42] = 2e3
    constants[43] = 5e3
    constants[44] = 36.3
    constants[45] = 2
    constants[46] = 120
    states[10] = -4.5113452510363e-6
    constants[47] = 2.25
    constants[48] = 1.85
    constants[49] = 1.9
    constants[50] = 0.002
    constants[51] = 10
    constants[52] = 0.02
    constants[53] = 70
    constants[54] = 1
    constants[55] = 1
    constants[56] = 120
    constants[57] = 70
    constants[58] = constants[8]*constants[14]*(power(constants[4], (constants[18]-37.0000)/10.0000))
    constants[59] = constants[9]*constants[14]*(power(constants[4], (constants[18]-37.0000)/10.0000))
    constants[60] = 1.00000e+07*(power(10.0000, -constants[15]))
    constants[61] = 0.00000
    constants[62] = 1.00000e+07*(power(10.0000, -7.15000))
    constants[70] = constants[61]
    constants[63] = ((power(constants[17], constants[16])+power(constants[62], constants[16]))/(power(constants[17], constants[16])+power(constants[60], constants[16])))*(constants[7]*(power(constants[3], (constants[18]-37.0000)/10.0000)))
    constants[64] = constants[20]*constants[30]*(power(constants[31], (constants[18]-37.0000)/10.0000))
    constants[65] = constants[64]
    constants[66] = (constants[38]*constants[20]*constants[22]*(constants[24]/constants[40]))/((constants[21]/constants[39])*(constants[23]/constants[62])*constants[37])
    constants[67] = (constants[20]*constants[22]+constants[66]*constants[21]+constants[66]*constants[23])/(constants[22]*constants[24]+constants[23]*constants[21]+constants[24]*constants[21]+constants[66]*constants[23]+constants[24]*constants[20]+constants[23]*constants[20]+constants[66]*constants[21]+constants[20]*constants[22]+constants[66]*constants[23])
    constants[68] = constants[46]*constants[36]*constants[67]
    constants[69] = (fabs(constants[49]-constants[48])/(constants[49]-constants[48]))*constants[50]*(exp(constants[51]*fabs(constants[49]-constants[48]))-1.00000)
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = constants[70]
    algebraic[7] = constants[63]*constants[19]*(1.00000-states[1])-constants[58]*states[1]
    rates[1] = algebraic[7]
    algebraic[10] = constants[63]*constants[19]*(1.00000-states[2])-constants[59]*states[2]
    rates[2] = algebraic[10]
    algebraic[0] = custom_piecewise([less(constants[1]/2.00000 , states[0]/2.00000), constants[1]/2.00000 , True, states[0]/2.00000])
    algebraic[1] = custom_piecewise([greater(states[0]/2.00000-(states[0]-constants[0]) , constants[2]/2.00000), states[0]/2.00000-(states[0]-constants[0]) , True, constants[2]/2.00000])
    algebraic[2] = algebraic[0]-algebraic[1]
    algebraic[4] = algebraic[2]/constants[0]
    algebraic[6] = (1.00000-algebraic[4])*states[1]+algebraic[4]*states[2]
    algebraic[9] = power(fabs(1.00000/(1.00000+power(constants[10]/algebraic[6], constants[11]))), 1.0/2)
    algebraic[12] = constants[12]*algebraic[9]*(power(constants[5], (constants[18]-37.0000)/10.0000))
    algebraic[15] = custom_piecewise([less(1.00000/algebraic[9] , 100.000), 1.00000/algebraic[9] , True, 100.000])
    algebraic[18] = constants[13]*algebraic[15]*(power(constants[6], (constants[18]-37.0000)/10.0000))
    rates[7] = algebraic[18]*states[8]-algebraic[12]*states[7]
    rates[8] = algebraic[12]*states[7]-algebraic[18]*states[8]
    algebraic[3] = (algebraic[2]*2.00000)/(constants[1]-constants[2])
    algebraic[5] = constants[46]*algebraic[3]*(states[3]*states[5]+states[4]*states[6])
    algebraic[8] = (1.00000*algebraic[5])/constants[68]
    algebraic[11] = ((states[0]-constants[48])/fabs(states[0]-constants[48]))*constants[50]*(exp(constants[51]*fabs(states[0]-constants[48]))-1.00000)
    algebraic[14] = custom_piecewise([greater(states[0] , constants[47]), constants[52]*(exp(constants[53]*fabs(states[0]-constants[47]))-1.00000) , True, 0.00000])
    algebraic[17] = algebraic[11]+algebraic[14]
    algebraic[20] = custom_piecewise([equal(constants[55] , 1.00000), constants[54]*(constants[49]-states[0]) , True, 0.00000])
    rates[10] = (constants[69]+algebraic[20])-(algebraic[17]+algebraic[8])
    algebraic[22] = 1.00000-(states[9]+states[6]+states[5])
    rates[9] = algebraic[12]*algebraic[22]-(algebraic[18]+constants[65])*states[9]
    algebraic[19] = exp((-states[4]/fabs(states[4]))*constants[26]*(power(states[4]/constants[36], 2.00000)))
    algebraic[23] = constants[22]*algebraic[19]*constants[30]*(power(constants[33], (constants[18]-37.0000)/10.0000))
    algebraic[27] = algebraic[23]
    algebraic[13] = 1.00000+(1.00000-algebraic[3])*constants[25]
    algebraic[16] = constants[21]*algebraic[13]*constants[30]*(power(constants[32], (constants[18]-37.0000)/10.0000))
    algebraic[29] = constants[42]*(algebraic[16]/constants[39])
    algebraic[21] = exp(((states[3]-constants[36])/fabs(states[3]-constants[36]))*constants[27]*(power((states[3]-constants[36])/constants[36], 2.00000)))
    algebraic[24] = constants[23]*algebraic[21]*constants[30]*(power(constants[34], (constants[18]-37.0000)/10.0000))
    algebraic[30] = constants[60]*(algebraic[24]/constants[62])*((constants[38]+constants[41])/constants[41])*(constants[44]/(constants[38]+constants[44]))
    algebraic[31] = (constants[65]*states[9]+algebraic[30]*states[5])-(algebraic[29]+algebraic[27])*states[6]
    rates[6] = algebraic[31]
    algebraic[25] = custom_piecewise([less(states[3] , constants[36]), exp(constants[28]*(power((constants[36]-states[3])/constants[36], 2.00000))) , True, exp(constants[29]*(power((states[3]-constants[36])/constants[36], 2.00000)))])
    algebraic[26] = constants[24]*algebraic[25]*constants[30]*(power(constants[35], (constants[18]-37.0000)/10.0000))
    algebraic[28] = constants[43]*(algebraic[26]/constants[40])*((constants[38]+constants[41])/(constants[38]+constants[44]))
    algebraic[32] = (constants[38]*constants[64]*algebraic[23]*(algebraic[26]/constants[40]))/((algebraic[16]/constants[39])*(algebraic[24]/constants[62])*constants[37])
    algebraic[33] = algebraic[32]
    algebraic[34] = (algebraic[33]*states[9]+algebraic[27]*states[6])-(algebraic[30]+algebraic[28])*states[5]
    rates[5] = algebraic[34]
    algebraic[35] = (algebraic[33]*algebraic[30]+algebraic[28]*constants[65]+algebraic[30]*constants[65])/(constants[65]*algebraic[27]+algebraic[33]*algebraic[29]+algebraic[33]*algebraic[27]+algebraic[33]*algebraic[30]+algebraic[28]*constants[65]+algebraic[30]*constants[65]+algebraic[27]*algebraic[28]+algebraic[33]*algebraic[29]+algebraic[28]*algebraic[29])
    algebraic[36] = constants[61]/2.00000+(constants[45]/algebraic[35])*(-(constants[65]*states[4])+algebraic[30]*(states[3]-(constants[36]+states[4])))
    rates[4] = algebraic[36]
    algebraic[37] = (constants[65]*algebraic[27]+algebraic[33]*algebraic[29]+algebraic[33]*algebraic[27])/(constants[65]*algebraic[27]+algebraic[33]*algebraic[29]+algebraic[33]*algebraic[27]+algebraic[33]*algebraic[30]+algebraic[28]*constants[65]+algebraic[30]*constants[65]+algebraic[27]*algebraic[28]+algebraic[33]*algebraic[29]+algebraic[28]*algebraic[29])
    algebraic[38] = constants[61]/2.00000+(constants[45]/algebraic[37])*(algebraic[27]*((states[4]+constants[36])-states[3]))
    rates[3] = algebraic[38]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[7] = constants[63]*constants[19]*(1.00000-states[1])-constants[58]*states[1]
    algebraic[10] = constants[63]*constants[19]*(1.00000-states[2])-constants[59]*states[2]
    algebraic[0] = custom_piecewise([less(constants[1]/2.00000 , states[0]/2.00000), constants[1]/2.00000 , True, states[0]/2.00000])
    algebraic[1] = custom_piecewise([greater(states[0]/2.00000-(states[0]-constants[0]) , constants[2]/2.00000), states[0]/2.00000-(states[0]-constants[0]) , True, constants[2]/2.00000])
    algebraic[2] = algebraic[0]-algebraic[1]
    algebraic[4] = algebraic[2]/constants[0]
    algebraic[6] = (1.00000-algebraic[4])*states[1]+algebraic[4]*states[2]
    algebraic[9] = power(fabs(1.00000/(1.00000+power(constants[10]/algebraic[6], constants[11]))), 1.0/2)
    algebraic[12] = constants[12]*algebraic[9]*(power(constants[5], (constants[18]-37.0000)/10.0000))
    algebraic[15] = custom_piecewise([less(1.00000/algebraic[9] , 100.000), 1.00000/algebraic[9] , True, 100.000])
    algebraic[18] = constants[13]*algebraic[15]*(power(constants[6], (constants[18]-37.0000)/10.0000))
    algebraic[3] = (algebraic[2]*2.00000)/(constants[1]-constants[2])
    algebraic[5] = constants[46]*algebraic[3]*(states[3]*states[5]+states[4]*states[6])
    algebraic[8] = (1.00000*algebraic[5])/constants[68]
    algebraic[11] = ((states[0]-constants[48])/fabs(states[0]-constants[48]))*constants[50]*(exp(constants[51]*fabs(states[0]-constants[48]))-1.00000)
    algebraic[14] = custom_piecewise([greater(states[0] , constants[47]), constants[52]*(exp(constants[53]*fabs(states[0]-constants[47]))-1.00000) , True, 0.00000])
    algebraic[17] = algebraic[11]+algebraic[14]
    algebraic[20] = custom_piecewise([equal(constants[55] , 1.00000), constants[54]*(constants[49]-states[0]) , True, 0.00000])
    algebraic[22] = 1.00000-(states[9]+states[6]+states[5])
    algebraic[19] = exp((-states[4]/fabs(states[4]))*constants[26]*(power(states[4]/constants[36], 2.00000)))
    algebraic[23] = constants[22]*algebraic[19]*constants[30]*(power(constants[33], (constants[18]-37.0000)/10.0000))
    algebraic[27] = algebraic[23]
    algebraic[13] = 1.00000+(1.00000-algebraic[3])*constants[25]
    algebraic[16] = constants[21]*algebraic[13]*constants[30]*(power(constants[32], (constants[18]-37.0000)/10.0000))
    algebraic[29] = constants[42]*(algebraic[16]/constants[39])
    algebraic[21] = exp(((states[3]-constants[36])/fabs(states[3]-constants[36]))*constants[27]*(power((states[3]-constants[36])/constants[36], 2.00000)))
    algebraic[24] = constants[23]*algebraic[21]*constants[30]*(power(constants[34], (constants[18]-37.0000)/10.0000))
    algebraic[30] = constants[60]*(algebraic[24]/constants[62])*((constants[38]+constants[41])/constants[41])*(constants[44]/(constants[38]+constants[44]))
    algebraic[31] = (constants[65]*states[9]+algebraic[30]*states[5])-(algebraic[29]+algebraic[27])*states[6]
    algebraic[25] = custom_piecewise([less(states[3] , constants[36]), exp(constants[28]*(power((constants[36]-states[3])/constants[36], 2.00000))) , True, exp(constants[29]*(power((states[3]-constants[36])/constants[36], 2.00000)))])
    algebraic[26] = constants[24]*algebraic[25]*constants[30]*(power(constants[35], (constants[18]-37.0000)/10.0000))
    algebraic[28] = constants[43]*(algebraic[26]/constants[40])*((constants[38]+constants[41])/(constants[38]+constants[44]))
    algebraic[32] = (constants[38]*constants[64]*algebraic[23]*(algebraic[26]/constants[40]))/((algebraic[16]/constants[39])*(algebraic[24]/constants[62])*constants[37])
    algebraic[33] = algebraic[32]
    algebraic[34] = (algebraic[33]*states[9]+algebraic[27]*states[6])-(algebraic[30]+algebraic[28])*states[5]
    algebraic[35] = (algebraic[33]*algebraic[30]+algebraic[28]*constants[65]+algebraic[30]*constants[65])/(constants[65]*algebraic[27]+algebraic[33]*algebraic[29]+algebraic[33]*algebraic[27]+algebraic[33]*algebraic[30]+algebraic[28]*constants[65]+algebraic[30]*constants[65]+algebraic[27]*algebraic[28]+algebraic[33]*algebraic[29]+algebraic[28]*algebraic[29])
    algebraic[36] = constants[61]/2.00000+(constants[45]/algebraic[35])*(-(constants[65]*states[4])+algebraic[30]*(states[3]-(constants[36]+states[4])))
    algebraic[37] = (constants[65]*algebraic[27]+algebraic[33]*algebraic[29]+algebraic[33]*algebraic[27])/(constants[65]*algebraic[27]+algebraic[33]*algebraic[29]+algebraic[33]*algebraic[27]+algebraic[33]*algebraic[30]+algebraic[28]*constants[65]+algebraic[30]*constants[65]+algebraic[27]*algebraic[28]+algebraic[33]*algebraic[29]+algebraic[28]*algebraic[29])
    algebraic[38] = constants[61]/2.00000+(constants[45]/algebraic[37])*(algebraic[27]*((states[4]+constants[36])-states[3]))
    algebraic[39] = (constants[66]*constants[23]+constants[24]*constants[20]+algebraic[24]*constants[20])/(constants[22]*constants[24]+constants[23]*constants[21]+constants[24]*constants[21]+constants[66]*constants[23]+constants[24]*constants[20]+constants[23]*constants[20]+constants[66]*constants[21]+constants[20]*constants[22]+constants[66]*constants[23])
    algebraic[40] = (states[5]+states[6])/(constants[67]+algebraic[39])
    algebraic[41] = (algebraic[34]+algebraic[31])/(constants[67]+algebraic[39])
    algebraic[42] = constants[57]*((1.00000-algebraic[4])*states[1]+algebraic[4]*(algebraic[40]*states[2]+(1.00000-algebraic[40])*states[1]))
    algebraic[43] = custom_piecewise([less(states[0] , constants[1]), -0.500000*constants[61] , True, 0.00000])
    algebraic[44] = custom_piecewise([greater(2.00000*constants[0]-states[0] , constants[2]), -0.500000*constants[61] , True, 0.00000])
    algebraic[45] = algebraic[43]-algebraic[44]
    algebraic[46] = (2.00000*algebraic[45])/(constants[1]-constants[2])
    algebraic[47] = algebraic[45]/constants[0]
    algebraic[48] = constants[56]*algebraic[46]*(states[3]*states[5]+states[4]*states[6])+constants[56]*algebraic[3]*(algebraic[38]*states[5]+states[3]*algebraic[34]+algebraic[36]*states[6]+states[4]*algebraic[31])
    algebraic[49] = constants[57]*(-algebraic[47]*states[1]+(1.00000-algebraic[4])*algebraic[7]+algebraic[47]*(algebraic[40]*states[2]+(1.00000-algebraic[40])*states[1])+algebraic[4]*((algebraic[41]*states[2]+algebraic[40]*algebraic[10]+(1.00000-algebraic[40])*algebraic[7])-algebraic[41]*states[1]))
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