# Size of variable arrays:
sizeAlgebraic = 62
sizeStates = 5
sizeConstants = 45
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_constants[0] = "ADHMV in component circulatory_dynamics (dimensionless)"
    legend_constants[1] = "AMM in component circulatory_dynamics (dimensionless)"
    legend_constants[2] = "ANU in component circulatory_dynamics (dimensionless)"
    legend_constants[3] = "ANUVN in component circulatory_dynamics (dimensionless)"
    legend_constants[4] = "ARM in component circulatory_dynamics (dimensionless)"
    legend_constants[5] = "ATRRFB in component circulatory_dynamics (dimensionless)"
    legend_constants[6] = "ATRVFB in component circulatory_dynamics (litre)"
    legend_constants[7] = "AU in component circulatory_dynamics (dimensionless)"
    legend_constants[8] = "AUH in component circulatory_dynamics (dimensionless)"
    legend_constants[9] = "AUM in component circulatory_dynamics (dimensionless)"
    legend_constants[10] = "AVE in component circulatory_dynamics (dimensionless)"
    legend_constants[11] = "HMD in component circulatory_dynamics (dimensionless)"
    legend_constants[12] = "HPL in component circulatory_dynamics (dimensionless)"
    legend_constants[13] = "HPR in component circulatory_dynamics (dimensionless)"
    legend_constants[14] = "MYOGRS in component circulatory_dynamics (dimensionless)"
    legend_constants[15] = "OSA in component circulatory_dynamics (dimensionless)"
    legend_constants[16] = "PAMK in component circulatory_dynamics (dimensionless)"
    legend_constants[17] = "PC in component circulatory_dynamics (mmHg)"
    legend_constants[18] = "RBF in component circulatory_dynamics (L_per_minute)"
    legend_constants[19] = "VIM in component circulatory_dynamics (dimensionless)"
    legend_constants[20] = "VP in component circulatory_dynamics (litre)"
    legend_constants[21] = "VRC in component circulatory_dynamics (litre)"
    legend_constants[22] = "VV6 in component circulatory_dynamics (litre)"
    legend_constants[23] = "VV7 in component circulatory_dynamics (litre)"
    legend_constants[24] = "VVR in component circulatory_dynamics (litre)"
    legend_states[0] = "VVS1 in component venous_blood_volume (litre)"
    legend_states[1] = "VAS1 in component arterial_blood_volume (litre)"
    legend_states[2] = "VLA1 in component left_atrial_blood_volume (litre)"
    legend_states[3] = "VPA1 in component pulmonary_vasculature_blood_volume (litre)"
    legend_states[4] = "VRA1 in component right_atrial_blood_volume (litre)"
    legend_algebraic[0] = "VBD in component total_blood_volume_change (litre)"
    legend_algebraic[33] = "QVO in component rate_of_blood_flow_from_veins_to_right_atrium (L_per_minute)"
    legend_algebraic[45] = "QRO in component right_ventricular_output (L_per_minute)"
    legend_algebraic[1] = "VRA in component right_atrial_blood_volume (litre)"
    legend_algebraic[47] = "DRA in component right_atrial_blood_volume (L_per_minute)"
    legend_algebraic[3] = "PRA in component right_atrial_pressure (mmHg)"
    legend_algebraic[2] = "VRE in component right_atrial_pressure (litre)"
    legend_algebraic[4] = "PRA1 in component autonomic_stimulation_effect_on_right_atrial_pressure (mmHg)"
    legend_constants[25] = "HTAUML in component parameter_values (dimensionless)"
    legend_algebraic[8] = "PPA in component pulmonary_vasculature_pressure (mmHg)"
    legend_algebraic[10] = "RVM in component pressure_effect_on_right_ventricular_pumping (dimensionless)"
    legend_algebraic[9] = "PP2 in component pressure_effect_on_right_ventricular_pumping (mmHg)"
    legend_algebraic[41] = "QLO in component left_ventricular_output (L_per_minute)"
    legend_algebraic[24] = "QLN in component left_ventricular_output (L_per_minute)"
    legend_algebraic[42] = "HPEF in component pumping_effectiveness_of_right_ventricle (L_per_minute)"
    legend_constants[26] = "QRF in component parameter_values (L_per_minute)"
    legend_constants[27] = "HSR in component parameter_values (dimensionless)"
    legend_algebraic[5] = "QRN in component right_ventricular_output (dimensionless)"
    legend_algebraic[22] = "QPO in component rate_of_blood_flow_from_pulmonary_veins_to_left_atrium (L_per_minute)"
    legend_algebraic[6] = "VPA in component pulmonary_vasculature_blood_volume (litre)"
    legend_algebraic[48] = "DPA in component pulmonary_vasculature_blood_volume (L_per_minute)"
    legend_algebraic[7] = "VPE in component pulmonary_vasculature_pressure (litre)"
    legend_algebraic[14] = "RPA in component pulmonary_arterial_resistance (mmHg_minute_per_L)"
    legend_algebraic[11] = "PP1T in component pulmonary_arterial_resistance (L_per_minute_per_mmHg)"
    legend_algebraic[12] = "PP1 in component pulmonary_arterial_resistance (L_per_minute_per_mmHg)"
    legend_algebraic[13] = "CPA in component pulmonary_arterial_resistance (L_per_minute_per_mmHg)"
    legend_algebraic[17] = "PLA in component left_atrial_pressure (mmHg)"
    legend_algebraic[19] = "RPV in component pulmonary_venous_resistance (mmHg_minute_per_L)"
    legend_algebraic[18] = "PL1 in component pulmonary_venous_resistance (mmHg)"
    legend_algebraic[20] = "RPT in component total_pulmonary_vascular_resistance (mmHg_minute_per_L)"
    legend_algebraic[21] = "PGL in component pressure_gradient_through_the_lungs (mmHg)"
    legend_algebraic[15] = "VLA in component left_atrial_blood_volume (litre)"
    legend_algebraic[43] = "DLA in component left_atrial_blood_volume (L_per_minute)"
    legend_algebraic[16] = "VLE in component left_atrial_pressure (litre)"
    legend_algebraic[23] = "PLA1 in component autonomic_stimulation_effect_on_left_atrial_pressure (mmHg)"
    legend_algebraic[36] = "PA in component arterial_pressure_and_pressure_gradient (mmHg)"
    legend_algebraic[38] = "LVM in component pumping_effectiveness_of_left_ventricle (dimensionless)"
    legend_algebraic[37] = "PA2 in component pumping_effectiveness_of_left_ventricle (mmHg)"
    legend_algebraic[39] = "QLOT in component left_ventricular_output (L_per_minute)"
    legend_constants[28] = "HSL in component parameter_values (dimensionless)"
    legend_algebraic[40] = "QLO1 in component left_ventricular_output (L_per_minute)"
    legend_algebraic[58] = "QAO in component systemic_blood_flow (L_per_minute)"
    legend_algebraic[25] = "VVS in component venous_blood_volume (litre)"
    legend_algebraic[59] = "DVS in component venous_blood_volume (L_per_minute)"
    legend_constants[40] = "VVA in component angiotensin_induced_venous_constriction (litre)"
    legend_constants[29] = "ANY in component parameter_values (litre)"
    legend_algebraic[27] = "VVE in component venous_excess_volume (litre)"
    legend_algebraic[26] = "VVE1 in component venous_excess_volume (litre)"
    legend_algebraic[29] = "PVS in component venous_average_pressure (mmHg)"
    legend_constants[30] = "CV in component parameter_values (L_per_mmHg)"
    legend_algebraic[28] = "PVS1 in component venous_average_pressure (mmHg)"
    legend_algebraic[30] = "PR1 in component venous_outflow_pressure_into_heart (mmHg)"
    legend_constants[31] = "PR1LL in component parameter_values (mmHg)"
    legend_algebraic[31] = "RVG in component resistance_from_veins_to_right_atrium (mmHg_minute_per_L)"
    legend_algebraic[32] = "PGV in component rate_of_blood_flow_from_veins_to_right_atrium (mmHg)"
    legend_constants[43] = "RVS in component venous_resistance (mmHg_minute_per_L)"
    legend_constants[32] = "CN7 in component parameter_values (dimensionless)"
    legend_constants[33] = "CN2 in component parameter_values (per_mmHg)"
    legend_constants[34] = "RVSM in component parameter_values (mmHg_minute_per_L)"
    legend_constants[41] = "CN3 in component venous_resistance (dimensionless)"
    legend_constants[42] = "RV1 in component venous_resistance (mmHg_minute_per_L)"
    legend_constants[44] = "NNRVR in component NM_NR_venous_resistance (mmHg_minute_per_L)"
    legend_algebraic[34] = "VAS in component arterial_blood_volume (litre)"
    legend_algebraic[60] = "DAS in component arterial_blood_volume (L_per_minute)"
    legend_algebraic[44] = "PAG in component arterial_pressure_and_pressure_gradient (mmHg)"
    legend_algebraic[35] = "VAE in component arterial_pressure_and_pressure_gradient (litre)"
    legend_algebraic[46] = "PAM in component pressure_effect_on_arterial_distention (dimensionless)"
    legend_constants[35] = "PAEX in component parameter_values (dimensionless)"
    legend_algebraic[49] = "R1 in component non_renal_systemic_arterial_resistance_multiplier (dimensionless)"
    legend_algebraic[50] = "NNRAR in component NM_NR_arterial_resistance (mmHg_minute_per_L)"
    legend_constants[36] = "RAR in component parameter_values (mmHg_minute_per_L)"
    legend_constants[37] = "RMULT1 in component parameter_values (dimensionless)"
    legend_algebraic[51] = "PGS in component pressure_gradient_from_arteries_to_veins (mmHg)"
    legend_algebraic[52] = "RSM in component M_systemic_resistance (mmHg_minute_per_L)"
    legend_constants[38] = "RAM in component parameter_values (mmHg_minute_per_L)"
    legend_algebraic[53] = "RSN in component total_NM_NR_systemic_resistance (mmHg_minute_per_L)"
    legend_algebraic[54] = "BFM in component blood_flow_through_M_tissues (L_per_minute)"
    legend_algebraic[55] = "BFN in component blood_flow_through_NM_NR_tissues (L_per_minute)"
    legend_algebraic[56] = "FISFLO in component blood_flow_through_AV_fistulas (L_per_minute)"
    legend_constants[39] = "FIS in component parameter_values (L_per_minute_per_mmHg)"
    legend_algebraic[57] = "SYSFLO in component systemic_blood_flow (L_per_minute)"
    legend_algebraic[61] = "RTP in component total_peripheral_resistance (mmHg_minute_per_L)"
    legend_rates[4] = "d/dt VRA1 in component right_atrial_blood_volume (litre)"
    legend_rates[3] = "d/dt VPA1 in component pulmonary_vasculature_blood_volume (litre)"
    legend_rates[2] = "d/dt VLA1 in component left_atrial_blood_volume (litre)"
    legend_rates[0] = "d/dt VVS1 in component venous_blood_volume (litre)"
    legend_rates[1] = "d/dt VAS1 in component arterial_blood_volume (litre)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 1.0
    constants[1] = 1.0
    constants[2] = 0.925271
    constants[3] = 1.0
    constants[4] = 1.16463
    constants[5] = 1.0
    constants[6] = 0.0
    constants[7] = 1.00022
    constants[8] = 1.00012
    constants[9] = 1.00066
    constants[10] = 1.0
    constants[11] = 1.0
    constants[12] = 1.00163
    constants[13] = 1.00237
    constants[14] = 1.0
    constants[15] = 0.97287
    constants[16] = 1.0
    constants[17] = 16.9144
    constants[18] = 1.22057
    constants[19] = 1.00076
    constants[20] = 3.00449
    constants[21] = 2.00439
    constants[22] = 0.0101913
    constants[23] = 0.00366525
    constants[24] = 2.50967
    states[0] = 3.28246
    states[1] = 0.862514
    states[2] = 0.379883
    states[3] = 0.38131
    states[4] = 0.100043
    constants[25] = 0.4
    constants[26] = 0.15
    constants[27] = 1
    constants[28] = 1
    constants[29] = -0.2
    constants[30] = 0.1
    constants[31] = 0
    constants[32] = 0.2
    constants[33] = 0.0212
    constants[34] = 1
    constants[35] = 2
    constants[36] = 30.52
    constants[37] = 1
    constants[38] = 96.3
    constants[39] = 0
    constants[40] = (constants[2]-1.00000)*constants[29]
    constants[41] = ((constants[17]-17.0000)*constants[32]+17.0000)*constants[33]
    constants[42] = constants[34]/constants[41]
    constants[43] = constants[10]*constants[42]*constants[19]*constants[3]
    constants[44] = constants[43]*1.79000
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[0] = ((((((constants[20]+constants[21])-states[0])-states[1])-states[2])-states[3])-states[4])/2.00000
    algebraic[15] = states[2]+algebraic[0]*0.128000
    algebraic[16] = algebraic[15]-0.380000
    algebraic[17] = algebraic[16]/0.0100000
    algebraic[23] = (algebraic[17]+4.00000)*(constants[25]*(constants[7]-1.00000)+1.00000)-4.00000
    algebraic[24] = custom_piecewise([less_equal(algebraic[23] , -2.00000), 0.0100000 , greater(algebraic[23] , -2.00000) & less_equal(algebraic[23] , 1.00000), 0.0100000+((3.60000-0.0100000)*(algebraic[23]--2.00000))/(1.00000--2.00000) , greater(algebraic[23] , 1.00000) & less_equal(algebraic[23] , 5.00000), 3.60000+((9.40000-3.60000)*(algebraic[23]-1.00000))/(5.00000-1.00000) , greater(algebraic[23] , 5.00000) & less_equal(algebraic[23] , 8.00000), 9.40000+((11.6000-9.40000)*(algebraic[23]-5.00000))/(8.00000-5.00000) , greater(algebraic[23] , 8.00000) & less_equal(algebraic[23] , 12.0000), 11.6000+((13.5000-11.6000)*(algebraic[23]-8.00000))/(12.0000-8.00000) , True, 13.5000])
    algebraic[34] = states[1]+algebraic[0]*0.261000
    algebraic[35] = algebraic[34]-0.495000
    algebraic[36] = algebraic[35]/0.00355000
    algebraic[37] = algebraic[36]/(constants[8]*constants[15])
    algebraic[38] = custom_piecewise([less_equal(algebraic[37] , 0.00000), 1.04000 , greater(algebraic[37] , 0.00000) & less_equal(algebraic[37] , 60.0000), 1.04000+((1.02500-1.04000)*(algebraic[37]-0.00000))/(60.0000-0.00000) , greater(algebraic[37] , 60.0000) & less_equal(algebraic[37] , 125.000), 1.02500+((0.970000-1.02500)*(algebraic[37]-60.0000))/(125.000-60.0000) , greater(algebraic[37] , 125.000) & less_equal(algebraic[37] , 160.000), 0.970000+((0.880000-0.970000)*(algebraic[37]-125.000))/(160.000-125.000) , greater(algebraic[37] , 160.000) & less_equal(algebraic[37] , 200.000), 0.880000+((0.590000-0.880000)*(algebraic[37]-160.000))/(200.000-160.000) , greater(algebraic[37] , 200.000) & less_equal(algebraic[37] , 240.000), 0.590000+((0.00000-0.590000)*(algebraic[37]-200.000))/(240.000-200.000) , True, 0.00000])
    algebraic[39] = algebraic[38]*algebraic[24]*constants[8]*constants[28]*constants[11]*constants[12]
    algebraic[40] = (algebraic[17]-algebraic[36])/3.00000
    algebraic[41] = custom_piecewise([greater(algebraic[40] , 0.00000), algebraic[39]+algebraic[40] , True, algebraic[39]])
    algebraic[6] = states[3]+algebraic[0]*0.155000
    algebraic[7] = algebraic[6]-0.306250
    algebraic[8] = algebraic[7]/0.00480000
    algebraic[11] = 0.0260000*algebraic[8]
    algebraic[12] = custom_piecewise([less(algebraic[11] , 1.00000e-05), 1.00000e-05 , True, algebraic[11]])
    algebraic[13] = power(algebraic[12], 0.500000)
    algebraic[14] = 1.00000/algebraic[13]
    algebraic[18] = algebraic[17]+18.0000
    algebraic[19] = 1.00000/(algebraic[18]*0.0357000)
    algebraic[20] = algebraic[19]+algebraic[14]
    algebraic[21] = algebraic[8]-algebraic[17]
    algebraic[22] = algebraic[21]/algebraic[20]
    algebraic[43] = algebraic[22]-algebraic[41]
    rates[2] = algebraic[43]
    algebraic[25] = states[0]+algebraic[0]*0.398600
    algebraic[26] = ((((algebraic[25]-constants[24])-constants[40])-constants[23])-constants[22])-constants[6]
    algebraic[27] = custom_piecewise([less(algebraic[26] , 0.000100000), 0.000100000 , True, algebraic[26]])
    algebraic[28] = 3.70000+(algebraic[27]-0.740000)/constants[30]
    algebraic[29] = custom_piecewise([less(algebraic[28] , 0.000100000), 0.000100000 , True, algebraic[28]])
    algebraic[31] = 0.740000/(power(algebraic[29]/(constants[19]*3.70000), 0.500000))
    algebraic[1] = states[4]+algebraic[0]*0.0574000
    algebraic[2] = algebraic[1]-0.100000
    algebraic[3] = algebraic[2]/0.00500000
    algebraic[30] = custom_piecewise([less(algebraic[3] , constants[31]), constants[31] , True, algebraic[3]])
    algebraic[32] = algebraic[29]-algebraic[30]
    algebraic[33] = algebraic[32]/algebraic[31]
    algebraic[9] = (algebraic[8]/constants[8])/constants[15]
    algebraic[10] = custom_piecewise([less_equal(algebraic[9] , 0.00000), 1.06000 , greater(algebraic[9] , 0.00000) & less_equal(algebraic[9] , 32.0000), 1.06000+((0.970000-1.06000)*(algebraic[9]-0.00000))/(32.0000-0.00000) , greater(algebraic[9] , 32.0000) & less_equal(algebraic[9] , 38.4000), 0.970000+((0.930000-0.970000)*(algebraic[9]-32.0000))/(38.4000-32.0000) , greater(algebraic[9] , 38.4000) & less_equal(algebraic[9] , 48.0000), 0.930000+((0.800000-0.930000)*(algebraic[9]-38.4000))/(48.0000-38.4000) , greater(algebraic[9] , 48.0000) & less_equal(algebraic[9] , 60.8000), 0.800000+((0.460000-0.800000)*(algebraic[9]-48.0000))/(60.8000-48.0000) , greater(algebraic[9] , 60.8000) & less_equal(algebraic[9] , 72.0000), 0.460000+((0.00000-0.460000)*(algebraic[9]-60.8000))/(72.0000-60.8000) , True, 0.00000])
    algebraic[42] = (1.00000-constants[26])*constants[8]*algebraic[10]*constants[27]*constants[11]*constants[13]+(constants[26]*algebraic[41])/algebraic[24]
    algebraic[4] = (algebraic[3]+8.00000)*(constants[25]*(constants[7]-1.00000)+1.00000)-8.00000
    algebraic[5] = custom_piecewise([less_equal(algebraic[4] , -8.00000), 0.00000 , greater(algebraic[4] , -8.00000) & less_equal(algebraic[4] , -6.00000), 0.00000+((0.750000-0.00000)*(algebraic[4]--8.00000))/(-6.00000--8.00000) , greater(algebraic[4] , -6.00000) & less_equal(algebraic[4] , -2.00000), 0.750000+((2.60000-0.750000)*(algebraic[4]--6.00000))/(-2.00000--6.00000) , greater(algebraic[4] , -2.00000) & less_equal(algebraic[4] , 4.00000), 2.60000+((9.80000-2.60000)*(algebraic[4]--2.00000))/(4.00000--2.00000) , greater(algebraic[4] , 4.00000) & less_equal(algebraic[4] , 12.0000), 9.80000+((13.5000-9.80000)*(algebraic[4]-4.00000))/(12.0000-4.00000) , True, 13.5000])
    algebraic[45] = algebraic[5]*algebraic[42]
    algebraic[47] = algebraic[33]-algebraic[45]
    rates[4] = algebraic[47]
    algebraic[48] = algebraic[45]-algebraic[22]
    rates[3] = algebraic[48]
    algebraic[44] = algebraic[36]-algebraic[3]
    algebraic[56] = algebraic[44]*constants[39]
    algebraic[51] = algebraic[36]-algebraic[29]
    algebraic[46] = power(algebraic[36]/100.000, constants[35])
    algebraic[49] = ((constants[2]*constants[0]*constants[9]*constants[19]*constants[16])/algebraic[46])/constants[5]
    algebraic[52] = constants[38]*constants[1]*algebraic[49]*constants[14]*constants[37]
    algebraic[54] = algebraic[51]/algebraic[52]
    algebraic[50] = constants[36]*constants[4]*algebraic[49]*constants[14]*constants[37]
    algebraic[53] = algebraic[50]+constants[44]
    algebraic[55] = algebraic[51]/algebraic[53]
    algebraic[57] = algebraic[54]+algebraic[55]+constants[18]
    algebraic[58] = algebraic[57]+algebraic[56]
    algebraic[59] = algebraic[58]-algebraic[33]
    rates[0] = algebraic[59]
    algebraic[60] = algebraic[41]-algebraic[58]
    rates[1] = algebraic[60]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = ((((((constants[20]+constants[21])-states[0])-states[1])-states[2])-states[3])-states[4])/2.00000
    algebraic[15] = states[2]+algebraic[0]*0.128000
    algebraic[16] = algebraic[15]-0.380000
    algebraic[17] = algebraic[16]/0.0100000
    algebraic[23] = (algebraic[17]+4.00000)*(constants[25]*(constants[7]-1.00000)+1.00000)-4.00000
    algebraic[24] = custom_piecewise([less_equal(algebraic[23] , -2.00000), 0.0100000 , greater(algebraic[23] , -2.00000) & less_equal(algebraic[23] , 1.00000), 0.0100000+((3.60000-0.0100000)*(algebraic[23]--2.00000))/(1.00000--2.00000) , greater(algebraic[23] , 1.00000) & less_equal(algebraic[23] , 5.00000), 3.60000+((9.40000-3.60000)*(algebraic[23]-1.00000))/(5.00000-1.00000) , greater(algebraic[23] , 5.00000) & less_equal(algebraic[23] , 8.00000), 9.40000+((11.6000-9.40000)*(algebraic[23]-5.00000))/(8.00000-5.00000) , greater(algebraic[23] , 8.00000) & less_equal(algebraic[23] , 12.0000), 11.6000+((13.5000-11.6000)*(algebraic[23]-8.00000))/(12.0000-8.00000) , True, 13.5000])
    algebraic[34] = states[1]+algebraic[0]*0.261000
    algebraic[35] = algebraic[34]-0.495000
    algebraic[36] = algebraic[35]/0.00355000
    algebraic[37] = algebraic[36]/(constants[8]*constants[15])
    algebraic[38] = custom_piecewise([less_equal(algebraic[37] , 0.00000), 1.04000 , greater(algebraic[37] , 0.00000) & less_equal(algebraic[37] , 60.0000), 1.04000+((1.02500-1.04000)*(algebraic[37]-0.00000))/(60.0000-0.00000) , greater(algebraic[37] , 60.0000) & less_equal(algebraic[37] , 125.000), 1.02500+((0.970000-1.02500)*(algebraic[37]-60.0000))/(125.000-60.0000) , greater(algebraic[37] , 125.000) & less_equal(algebraic[37] , 160.000), 0.970000+((0.880000-0.970000)*(algebraic[37]-125.000))/(160.000-125.000) , greater(algebraic[37] , 160.000) & less_equal(algebraic[37] , 200.000), 0.880000+((0.590000-0.880000)*(algebraic[37]-160.000))/(200.000-160.000) , greater(algebraic[37] , 200.000) & less_equal(algebraic[37] , 240.000), 0.590000+((0.00000-0.590000)*(algebraic[37]-200.000))/(240.000-200.000) , True, 0.00000])
    algebraic[39] = algebraic[38]*algebraic[24]*constants[8]*constants[28]*constants[11]*constants[12]
    algebraic[40] = (algebraic[17]-algebraic[36])/3.00000
    algebraic[41] = custom_piecewise([greater(algebraic[40] , 0.00000), algebraic[39]+algebraic[40] , True, algebraic[39]])
    algebraic[6] = states[3]+algebraic[0]*0.155000
    algebraic[7] = algebraic[6]-0.306250
    algebraic[8] = algebraic[7]/0.00480000
    algebraic[11] = 0.0260000*algebraic[8]
    algebraic[12] = custom_piecewise([less(algebraic[11] , 1.00000e-05), 1.00000e-05 , True, algebraic[11]])
    algebraic[13] = power(algebraic[12], 0.500000)
    algebraic[14] = 1.00000/algebraic[13]
    algebraic[18] = algebraic[17]+18.0000
    algebraic[19] = 1.00000/(algebraic[18]*0.0357000)
    algebraic[20] = algebraic[19]+algebraic[14]
    algebraic[21] = algebraic[8]-algebraic[17]
    algebraic[22] = algebraic[21]/algebraic[20]
    algebraic[43] = algebraic[22]-algebraic[41]
    algebraic[25] = states[0]+algebraic[0]*0.398600
    algebraic[26] = ((((algebraic[25]-constants[24])-constants[40])-constants[23])-constants[22])-constants[6]
    algebraic[27] = custom_piecewise([less(algebraic[26] , 0.000100000), 0.000100000 , True, algebraic[26]])
    algebraic[28] = 3.70000+(algebraic[27]-0.740000)/constants[30]
    algebraic[29] = custom_piecewise([less(algebraic[28] , 0.000100000), 0.000100000 , True, algebraic[28]])
    algebraic[31] = 0.740000/(power(algebraic[29]/(constants[19]*3.70000), 0.500000))
    algebraic[1] = states[4]+algebraic[0]*0.0574000
    algebraic[2] = algebraic[1]-0.100000
    algebraic[3] = algebraic[2]/0.00500000
    algebraic[30] = custom_piecewise([less(algebraic[3] , constants[31]), constants[31] , True, algebraic[3]])
    algebraic[32] = algebraic[29]-algebraic[30]
    algebraic[33] = algebraic[32]/algebraic[31]
    algebraic[9] = (algebraic[8]/constants[8])/constants[15]
    algebraic[10] = custom_piecewise([less_equal(algebraic[9] , 0.00000), 1.06000 , greater(algebraic[9] , 0.00000) & less_equal(algebraic[9] , 32.0000), 1.06000+((0.970000-1.06000)*(algebraic[9]-0.00000))/(32.0000-0.00000) , greater(algebraic[9] , 32.0000) & less_equal(algebraic[9] , 38.4000), 0.970000+((0.930000-0.970000)*(algebraic[9]-32.0000))/(38.4000-32.0000) , greater(algebraic[9] , 38.4000) & less_equal(algebraic[9] , 48.0000), 0.930000+((0.800000-0.930000)*(algebraic[9]-38.4000))/(48.0000-38.4000) , greater(algebraic[9] , 48.0000) & less_equal(algebraic[9] , 60.8000), 0.800000+((0.460000-0.800000)*(algebraic[9]-48.0000))/(60.8000-48.0000) , greater(algebraic[9] , 60.8000) & less_equal(algebraic[9] , 72.0000), 0.460000+((0.00000-0.460000)*(algebraic[9]-60.8000))/(72.0000-60.8000) , True, 0.00000])
    algebraic[42] = (1.00000-constants[26])*constants[8]*algebraic[10]*constants[27]*constants[11]*constants[13]+(constants[26]*algebraic[41])/algebraic[24]
    algebraic[4] = (algebraic[3]+8.00000)*(constants[25]*(constants[7]-1.00000)+1.00000)-8.00000
    algebraic[5] = custom_piecewise([less_equal(algebraic[4] , -8.00000), 0.00000 , greater(algebraic[4] , -8.00000) & less_equal(algebraic[4] , -6.00000), 0.00000+((0.750000-0.00000)*(algebraic[4]--8.00000))/(-6.00000--8.00000) , greater(algebraic[4] , -6.00000) & less_equal(algebraic[4] , -2.00000), 0.750000+((2.60000-0.750000)*(algebraic[4]--6.00000))/(-2.00000--6.00000) , greater(algebraic[4] , -2.00000) & less_equal(algebraic[4] , 4.00000), 2.60000+((9.80000-2.60000)*(algebraic[4]--2.00000))/(4.00000--2.00000) , greater(algebraic[4] , 4.00000) & less_equal(algebraic[4] , 12.0000), 9.80000+((13.5000-9.80000)*(algebraic[4]-4.00000))/(12.0000-4.00000) , True, 13.5000])
    algebraic[45] = algebraic[5]*algebraic[42]
    algebraic[47] = algebraic[33]-algebraic[45]
    algebraic[48] = algebraic[45]-algebraic[22]
    algebraic[44] = algebraic[36]-algebraic[3]
    algebraic[56] = algebraic[44]*constants[39]
    algebraic[51] = algebraic[36]-algebraic[29]
    algebraic[46] = power(algebraic[36]/100.000, constants[35])
    algebraic[49] = ((constants[2]*constants[0]*constants[9]*constants[19]*constants[16])/algebraic[46])/constants[5]
    algebraic[52] = constants[38]*constants[1]*algebraic[49]*constants[14]*constants[37]
    algebraic[54] = algebraic[51]/algebraic[52]
    algebraic[50] = constants[36]*constants[4]*algebraic[49]*constants[14]*constants[37]
    algebraic[53] = algebraic[50]+constants[44]
    algebraic[55] = algebraic[51]/algebraic[53]
    algebraic[57] = algebraic[54]+algebraic[55]+constants[18]
    algebraic[58] = algebraic[57]+algebraic[56]
    algebraic[59] = algebraic[58]-algebraic[33]
    algebraic[60] = algebraic[41]-algebraic[58]
    algebraic[61] = algebraic[44]/algebraic[58]
    return algebraic

def custom_piecewise(cases):
    """Compute result of a piecewise function"""
    return select(cases[0::2],cases[1::2])

def gcd(A, B):
    """Greatest common divisor"""
    if (iterable(A) and iterable(B)):
        x = [];
        for (a,b) in zip(A,B):
            assert (int(a) == a) and (int(b) == b)
            a = int(a); b = int(b)
            while a:
                a,b = b % a, a
            x.append(b)
        return x
    else:
        while A:
            A,B = B % A, A
        return b

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