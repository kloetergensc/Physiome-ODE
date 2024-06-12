# Size of variable arrays:
sizeAlgebraic = 34
sizeStates = 14
sizeConstants = 88
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "t in component environment (second)"
    legend_algebraic[18] = "Pi in component TempCDa (UnitP)"
    legend_states[0] = "Pi in component TempRLC (UnitP)"
    legend_algebraic[30] = "Qo in component TempRC (UnitQ)"
    legend_algebraic[10] = "Qo in component TempCDv (UnitQ)"
    legend_algebraic[19] = "Pi in component TempCDa (UnitP)"
    legend_states[1] = "Pi in component TempRLC (UnitP)"
    legend_algebraic[31] = "Qo in component TempRC (UnitQ)"
    legend_algebraic[11] = "Qo in component TempCDv (UnitQ)"
    legend_algebraic[6] = "Pi in component TempCDv (UnitP)"
    legend_algebraic[22] = "Qo in component TempCDa (UnitQ)"
    legend_constants[0] = "CVao in component ParaHeart (UnitCV)"
    legend_algebraic[4] = "E in component EVentricle (UnitE)"
    legend_states[2] = "V in component TempCDv (UnitV)"
    legend_constants[1] = "PlvIni in component ParaHeart (UnitP)"
    legend_constants[2] = "VlvIni in component ParaHeart (UnitV)"
    legend_algebraic[8] = "Tao in component TempCDv (dimensionless)"
    legend_constants[3] = "Vlv0 in component ParaHeart (UnitV)"
    legend_constants[4] = "CVmi in component ParaHeart (UnitCV)"
    legend_algebraic[16] = "E in component EAtrium (UnitE)"
    legend_states[3] = "V in component TempCDa (UnitV)"
    legend_constants[5] = "PlaIni in component ParaHeart (UnitP)"
    legend_constants[6] = "VlaIni in component ParaHeart (UnitV)"
    legend_algebraic[20] = "Tao in component TempCDa (dimensionless)"
    legend_constants[7] = "Vla0 in component ParaHeart (UnitV)"
    legend_constants[8] = "ElvMax in component ParaHeart (UnitE)"
    legend_constants[9] = "ElvMin in component ParaHeart (UnitE)"
    legend_constants[10] = "T in component ParaHeart (second)"
    legend_constants[11] = "Ts1 in component ParaHeart (dimensionless)"
    legend_constants[12] = "Ts2 in component ParaHeart (dimensionless)"
    legend_algebraic[0] = "mt in component EVentricle (second)"
    legend_algebraic[2] = "et in component EVentricle (dimensionless)"
    legend_constants[13] = "ElaMax in component ParaHeart (UnitE)"
    legend_constants[14] = "ElaMin in component ParaHeart (UnitE)"
    legend_constants[15] = "Tpwb in component ParaHeart (dimensionless)"
    legend_constants[16] = "Tpww in component ParaHeart (dimensionless)"
    legend_algebraic[12] = "mt in component EAtrium (second)"
    legend_algebraic[14] = "et in component EAtrium (dimensionless)"
    legend_constants[17] = "EraMax in component ParaHeart (UnitE)"
    legend_constants[18] = "EraMin in component ParaHeart (UnitE)"
    legend_constants[19] = "PraIni in component ParaHeart (UnitP)"
    legend_constants[20] = "VraIni in component ParaHeart (UnitV)"
    legend_constants[21] = "ErvMax in component ParaHeart (UnitE)"
    legend_constants[22] = "ErvMin in component ParaHeart (UnitE)"
    legend_constants[23] = "PrvIni in component ParaHeart (UnitP)"
    legend_constants[24] = "VrvIni in component ParaHeart (UnitV)"
    legend_constants[25] = "CVpa in component ParaHeart (UnitCV)"
    legend_constants[26] = "CVti in component ParaHeart (UnitCV)"
    legend_constants[27] = "Vra0 in component ParaHeart (UnitV)"
    legend_constants[28] = "Vrv0 in component ParaHeart (UnitV)"
    legend_algebraic[7] = "Pi in component TempCDv (UnitP)"
    legend_algebraic[23] = "Qo in component TempCDa (UnitQ)"
    legend_constants[29] = "CVpa in component ParaHeart (UnitCV)"
    legend_algebraic[5] = "E in component EVentricle (UnitE)"
    legend_states[4] = "V in component TempCDv (UnitV)"
    legend_constants[30] = "PrvIni in component ParaHeart (UnitP)"
    legend_constants[31] = "VrvIni in component ParaHeart (UnitV)"
    legend_algebraic[9] = "Tao in component TempCDv (dimensionless)"
    legend_constants[32] = "Vrv0 in component ParaHeart (UnitV)"
    legend_constants[33] = "CVti in component ParaHeart (UnitCV)"
    legend_algebraic[17] = "E in component EAtrium (UnitE)"
    legend_states[5] = "V in component TempCDa (UnitV)"
    legend_constants[34] = "PraIni in component ParaHeart (UnitP)"
    legend_constants[35] = "VraIni in component ParaHeart (UnitV)"
    legend_algebraic[21] = "Tao in component TempCDa (dimensionless)"
    legend_constants[36] = "Vra0 in component ParaHeart (UnitV)"
    legend_constants[37] = "ErvMax in component ParaHeart (UnitE)"
    legend_constants[38] = "ErvMin in component ParaHeart (UnitE)"
    legend_constants[39] = "T in component ParaHeart (second)"
    legend_constants[40] = "Ts1 in component ParaHeart (dimensionless)"
    legend_constants[41] = "Ts2 in component ParaHeart (dimensionless)"
    legend_algebraic[1] = "mt in component EVentricle (second)"
    legend_algebraic[3] = "et in component EVentricle (dimensionless)"
    legend_constants[42] = "EraMax in component ParaHeart (UnitE)"
    legend_constants[43] = "EraMin in component ParaHeart (UnitE)"
    legend_constants[44] = "Tpwb in component ParaHeart (dimensionless)"
    legend_constants[45] = "Tpww in component ParaHeart (dimensionless)"
    legend_algebraic[13] = "mt in component EAtrium (second)"
    legend_algebraic[15] = "et in component EAtrium (dimensionless)"
    legend_constants[46] = "ElaMax in component ParaHeart (UnitE)"
    legend_constants[47] = "ElaMin in component ParaHeart (UnitE)"
    legend_constants[48] = "PlaIni in component ParaHeart (UnitP)"
    legend_constants[49] = "VlaIni in component ParaHeart (UnitV)"
    legend_constants[50] = "ElvMax in component ParaHeart (UnitE)"
    legend_constants[51] = "ElvMin in component ParaHeart (UnitE)"
    legend_constants[52] = "PlvIni in component ParaHeart (UnitP)"
    legend_constants[53] = "VlvIni in component ParaHeart (UnitV)"
    legend_constants[54] = "CVao in component ParaHeart (UnitCV)"
    legend_constants[55] = "CVmi in component ParaHeart (UnitCV)"
    legend_constants[56] = "Vlv0 in component ParaHeart (UnitV)"
    legend_constants[57] = "Vla0 in component ParaHeart (UnitV)"
    legend_states[6] = "Pi in component TempRLC (UnitP)"
    legend_states[7] = "Qo in component TempRLC (UnitQ)"
    legend_constants[58] = "Rsas in component ParaSys (UnitR)"
    legend_constants[59] = "Csas in component ParaSys (UnitC)"
    legend_constants[60] = "Lsas in component ParaSys (UnitL)"
    legend_constants[61] = "P0sas in component ParaSys (UnitP)"
    legend_constants[62] = "Q0sas in component ParaSys (UnitQ)"
    legend_algebraic[32] = "Pi in component TempR (UnitP)"
    legend_states[8] = "Qo in component TempRLC (UnitQ)"
    legend_constants[63] = "Rsat in component ParaSys (UnitR)"
    legend_constants[64] = "Csat in component ParaSys (UnitC)"
    legend_constants[65] = "Lsat in component ParaSys (UnitL)"
    legend_constants[66] = "P0sat in component ParaSys (UnitP)"
    legend_constants[67] = "Q0sat in component ParaSys (UnitQ)"
    legend_algebraic[28] = "Pi in component TempR (UnitP)"
    legend_algebraic[25] = "Qo in component TempR (UnitQ)"
    legend_constants[68] = "Rsar in component ParaSys (UnitR)"
    legend_states[9] = "Pi in component TempRC (UnitP)"
    legend_algebraic[27] = "Qo in component TempR (UnitQ)"
    legend_constants[69] = "Rscp in component ParaSys (UnitR)"
    legend_constants[70] = "Rsvn in component ParaSys (UnitR)"
    legend_constants[71] = "Csvn in component ParaSys (UnitC)"
    legend_constants[72] = "P0svn in component ParaSys (UnitP)"
    legend_states[10] = "Pi in component TempRLC (UnitP)"
    legend_states[11] = "Qo in component TempRLC (UnitQ)"
    legend_constants[73] = "Rpas in component ParaPul (UnitR)"
    legend_constants[74] = "Cpas in component ParaPul (UnitC)"
    legend_constants[75] = "Lpas in component ParaPul (UnitL)"
    legend_constants[76] = "P0pas in component ParaPul (UnitP)"
    legend_constants[77] = "Q0pas in component ParaPul (UnitQ)"
    legend_algebraic[33] = "Pi in component TempR (UnitP)"
    legend_states[12] = "Qo in component TempRLC (UnitQ)"
    legend_constants[78] = "Rpat in component ParaPul (UnitR)"
    legend_constants[79] = "Cpat in component ParaPul (UnitC)"
    legend_constants[80] = "Lpat in component ParaPul (UnitL)"
    legend_constants[81] = "P0pat in component ParaPul (UnitP)"
    legend_constants[82] = "Q0pat in component ParaPul (UnitQ)"
    legend_algebraic[29] = "Pi in component TempR (UnitP)"
    legend_algebraic[24] = "Qo in component TempR (UnitQ)"
    legend_constants[83] = "Rpar in component ParaPul (UnitR)"
    legend_states[13] = "Pi in component TempRC (UnitP)"
    legend_algebraic[26] = "Qo in component TempR (UnitQ)"
    legend_constants[84] = "Rpcp in component ParaPul (UnitR)"
    legend_constants[85] = "Rpvn in component ParaPul (UnitR)"
    legend_constants[86] = "Cpvn in component ParaPul (UnitC)"
    legend_constants[87] = "P0pvn in component ParaPul (UnitP)"
    legend_rates[2] = "d/dt V in component TempCDv (UnitV)"
    legend_rates[3] = "d/dt V in component TempCDa (UnitV)"
    legend_rates[4] = "d/dt V in component TempCDv (UnitV)"
    legend_rates[5] = "d/dt V in component TempCDa (UnitV)"
    legend_rates[0] = "d/dt Pi in component TempRLC (UnitP)"
    legend_rates[7] = "d/dt Qo in component TempRLC (UnitQ)"
    legend_rates[6] = "d/dt Pi in component TempRLC (UnitP)"
    legend_rates[8] = "d/dt Qo in component TempRLC (UnitQ)"
    legend_rates[9] = "d/dt Pi in component TempRC (UnitP)"
    legend_rates[1] = "d/dt Pi in component TempRLC (UnitP)"
    legend_rates[11] = "d/dt Qo in component TempRLC (UnitQ)"
    legend_rates[10] = "d/dt Pi in component TempRLC (UnitP)"
    legend_rates[12] = "d/dt Qo in component TempRLC (UnitQ)"
    legend_rates[13] = "d/dt Pi in component TempRC (UnitP)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 350.
    constants[1] = 1.0
    constants[2] = 5.0
    constants[3] = 500
    constants[4] = 400.
    constants[5] = 1.0
    constants[6] = 4.0
    constants[7] = 20
    constants[8] = 2.5
    constants[9] = 0.1
    constants[10] = 1.0
    constants[11] = 0.3
    constants[12] = 0.45
    constants[13] = 0.25
    constants[14] = 0.15
    constants[15] = 0.92
    constants[16] = 0.09
    constants[17] = 0.25
    constants[18] = 0.15
    constants[19] = 1.0
    constants[20] = 4.0
    constants[21] = 1.15
    constants[22] = 0.1
    constants[23] = 1.0
    constants[24] = 10.0
    constants[25] = 350.
    constants[26] = 400.
    constants[27] = 20
    constants[28] = 500
    constants[29] = 350.
    constants[30] = 1.0
    constants[31] = 10.0
    constants[32] = 500
    constants[33] = 400.
    constants[34] = 1.0
    constants[35] = 4.0
    constants[36] = 20
    constants[37] = 1.15
    constants[38] = 0.1
    constants[39] = 1.0
    constants[40] = 0.3
    constants[41] = 0.45
    constants[42] = 0.25
    constants[43] = 0.15
    constants[44] = 0.92
    constants[45] = 0.09
    constants[46] = 0.25
    constants[47] = 0.15
    constants[48] = 1.0
    constants[49] = 4.0
    constants[50] = 2.5
    constants[51] = 0.1
    constants[52] = 1.0
    constants[53] = 5.0
    constants[54] = 350.
    constants[55] = 400.
    constants[56] = 500
    constants[57] = 20
    constants[58] = 0.003
    constants[59] = 0.08
    constants[60] = 0.000062
    constants[61] = 100.
    constants[62] = 0.
    constants[63] = 0.05
    constants[64] = 1.6
    constants[65] = 0.0017
    constants[66] = 100.
    constants[67] = 0.
    constants[68] = 0.5
    constants[69] = 0.52
    constants[70] = 0.075
    constants[71] = 20.5
    constants[72] = 0.
    constants[73] = 0.002
    constants[74] = 0.18
    constants[75] = 0.000052
    constants[76] = 30.
    constants[77] = 0.
    constants[78] = 0.01
    constants[79] = 3.8
    constants[80] = 0.0017
    constants[81] = 30.
    constants[82] = 0.
    constants[83] = 0.05
    constants[84] = 0.25
    constants[85] = 0.0006
    constants[86] = 20.5
    constants[87] = 0.
    states[0] = constants[61]
    states[1] = constants[76]
    states[2] = constants[3]
    states[3] = constants[7]
    states[4] = constants[32]
    states[5] = constants[36]
    states[6] = constants[66]
    states[7] = constants[62]
    states[8] = constants[67]
    states[9] = constants[72]
    states[10] = constants[81]
    states[11] = constants[77]
    states[12] = constants[82]
    states[13] = constants[87]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[7] = ((states[0]-states[6])-constants[58]*states[7])/constants[60]
    rates[6] = (states[7]-states[8])/constants[64]
    rates[11] = ((states[1]-states[10])-constants[73]*states[11])/constants[75]
    rates[10] = (states[11]-states[12])/constants[79]
    algebraic[0] = voi-constants[10]*floor(voi/constants[10])
    algebraic[2] = custom_piecewise([greater_equal(algebraic[0] , 0.00000) & less_equal(algebraic[0] , constants[11]*constants[10]), 1.00000-cos((3.14159*algebraic[0])/(constants[11]*constants[10])) , greater(algebraic[0] , constants[11]*constants[10]) & less_equal(algebraic[0] , constants[12]*constants[10]), 1.00000+cos((3.14159*(algebraic[0]-constants[11]*constants[10]))/((constants[12]-constants[11])*constants[10])) , greater(algebraic[0] , constants[12]*constants[10]) & less(algebraic[0] , constants[10]), 0.00000 , True, float('nan')])
    algebraic[4] = constants[9]+(algebraic[2]*(constants[8]-constants[9]))/2.00000
    algebraic[6] = constants[1]+algebraic[4]*(states[2]-constants[2])
    algebraic[8] = custom_piecewise([greater_equal(algebraic[6] , states[0]), 1.00000 , less(algebraic[6] , states[0]), 0.00000 , True, float('nan')])
    algebraic[10] = custom_piecewise([greater_equal(algebraic[6] , states[0]), constants[0]*algebraic[8]*(power(fabs(algebraic[6]-states[0]), 0.500000)) , less(algebraic[6] , states[0]), -1.00000*constants[0]*algebraic[8]*(power(fabs(states[0]-algebraic[6]), 0.500000)) , True, float('nan')])
    rates[0] = (algebraic[10]-states[7])/constants[59]
    algebraic[1] = voi-constants[39]*floor(voi/constants[39])
    algebraic[3] = custom_piecewise([greater_equal(algebraic[1] , 0.00000) & less_equal(algebraic[1] , constants[40]*constants[39]), 1.00000-cos((3.14159*algebraic[1])/(constants[40]*constants[39])) , greater(algebraic[1] , constants[40]*constants[39]) & less_equal(algebraic[1] , constants[41]*constants[39]), 1.00000+cos((3.14159*(algebraic[1]-constants[40]*constants[39]))/((constants[41]-constants[40])*constants[39])) , greater(algebraic[1] , constants[41]*constants[39]) & less(algebraic[1] , constants[39]), 0.00000 , True, float('nan')])
    algebraic[5] = constants[38]+(algebraic[3]*(constants[37]-constants[38]))/2.00000
    algebraic[7] = constants[30]+algebraic[5]*(states[4]-constants[31])
    algebraic[9] = custom_piecewise([greater_equal(algebraic[7] , states[1]), 1.00000 , less(algebraic[7] , states[1]), 0.00000 , True, float('nan')])
    algebraic[11] = custom_piecewise([greater_equal(algebraic[7] , states[1]), constants[29]*algebraic[9]*(power(fabs(algebraic[7]-states[1]), 0.500000)) , less(algebraic[7] , states[1]), -1.00000*constants[29]*algebraic[9]*(power(fabs(states[1]-algebraic[7]), 0.500000)) , True, float('nan')])
    rates[1] = (algebraic[11]-states[11])/constants[74]
    algebraic[12] = voi-constants[10]*floor(voi/constants[10])
    algebraic[14] = custom_piecewise([greater_equal(algebraic[12] , 0.00000) & less_equal(algebraic[12] , (constants[15]+constants[16])*constants[10]-constants[10]), 1.00000-cos((2.00000*3.14159*((algebraic[12]-constants[15]*constants[10])+constants[10]))/(constants[16]*constants[10])) , greater(algebraic[12] , (constants[15]+constants[16])*constants[10]-constants[10]) & less_equal(algebraic[12] , constants[15]*constants[10]), 0.00000 , greater(algebraic[12] , constants[15]*constants[10]) & less_equal(algebraic[12] , constants[10]), 1.00000-cos((2.00000*3.14159*(algebraic[12]-constants[15]*constants[10]))/(constants[16]*constants[10])) , True, float('nan')])
    algebraic[16] = constants[14]+(algebraic[14]*(constants[13]-constants[14]))/2.00000
    algebraic[18] = constants[5]+algebraic[16]*(states[3]-constants[6])
    algebraic[20] = custom_piecewise([greater_equal(algebraic[18] , algebraic[6]), 1.00000 , less(algebraic[18] , algebraic[6]), 0.00000 , True, float('nan')])
    algebraic[22] = custom_piecewise([greater_equal(algebraic[18] , algebraic[6]), constants[4]*algebraic[20]*(power(fabs(algebraic[18]-algebraic[6]), 0.500000)) , less(algebraic[18] , algebraic[6]), -1.00000*constants[4]*algebraic[20]*(power(fabs(algebraic[6]-algebraic[18]), 0.500000)) , True, float('nan')])
    rates[2] = algebraic[22]-algebraic[10]
    algebraic[13] = voi-constants[39]*floor(voi/constants[39])
    algebraic[15] = custom_piecewise([greater_equal(algebraic[13] , 0.00000) & less_equal(algebraic[13] , (constants[44]+constants[45])*constants[39]-constants[39]), 1.00000-cos((2.00000*3.14159*((algebraic[13]-constants[44]*constants[39])+constants[39]))/(constants[45]*constants[39])) , greater(algebraic[13] , (constants[44]+constants[45])*constants[39]-constants[39]) & less_equal(algebraic[13] , constants[44]*constants[39]), 0.00000 , greater(algebraic[13] , constants[44]*constants[39]) & less_equal(algebraic[13] , constants[39]), 1.00000-cos((2.00000*3.14159*(algebraic[13]-constants[44]*constants[39]))/(constants[45]*constants[39])) , True, float('nan')])
    algebraic[17] = constants[43]+(algebraic[15]*(constants[42]-constants[43]))/2.00000
    algebraic[19] = constants[34]+algebraic[17]*(states[5]-constants[35])
    algebraic[21] = custom_piecewise([greater_equal(algebraic[19] , algebraic[7]), 1.00000 , less(algebraic[19] , algebraic[7]), 0.00000 , True, float('nan')])
    algebraic[23] = custom_piecewise([greater_equal(algebraic[19] , algebraic[7]), constants[33]*algebraic[21]*(power(fabs(algebraic[19]-algebraic[7]), 0.500000)) , less(algebraic[19] , algebraic[7]), -1.00000*constants[33]*algebraic[21]*(power(fabs(algebraic[7]-algebraic[19]), 0.500000)) , True, float('nan')])
    rates[4] = algebraic[23]-algebraic[11]
    algebraic[30] = (states[13]-algebraic[18])/constants[85]
    rates[3] = algebraic[30]-algebraic[22]
    algebraic[31] = (states[9]-algebraic[19])/constants[70]
    rates[5] = algebraic[31]-algebraic[23]
    algebraic[25] = states[8]
    algebraic[28] = states[9]+constants[69]*algebraic[25]
    algebraic[32] = algebraic[28]+constants[68]*states[8]
    rates[8] = ((states[6]-algebraic[32])-constants[63]*states[8])/constants[65]
    algebraic[27] = algebraic[25]
    rates[9] = (algebraic[27]-algebraic[31])/constants[71]
    algebraic[24] = states[12]
    algebraic[29] = states[13]+constants[84]*algebraic[24]
    algebraic[33] = algebraic[29]+constants[83]*states[12]
    rates[12] = ((states[10]-algebraic[33])-constants[78]*states[12])/constants[80]
    algebraic[26] = algebraic[24]
    rates[13] = (algebraic[26]-algebraic[30])/constants[86]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = voi-constants[10]*floor(voi/constants[10])
    algebraic[2] = custom_piecewise([greater_equal(algebraic[0] , 0.00000) & less_equal(algebraic[0] , constants[11]*constants[10]), 1.00000-cos((3.14159*algebraic[0])/(constants[11]*constants[10])) , greater(algebraic[0] , constants[11]*constants[10]) & less_equal(algebraic[0] , constants[12]*constants[10]), 1.00000+cos((3.14159*(algebraic[0]-constants[11]*constants[10]))/((constants[12]-constants[11])*constants[10])) , greater(algebraic[0] , constants[12]*constants[10]) & less(algebraic[0] , constants[10]), 0.00000 , True, float('nan')])
    algebraic[4] = constants[9]+(algebraic[2]*(constants[8]-constants[9]))/2.00000
    algebraic[6] = constants[1]+algebraic[4]*(states[2]-constants[2])
    algebraic[8] = custom_piecewise([greater_equal(algebraic[6] , states[0]), 1.00000 , less(algebraic[6] , states[0]), 0.00000 , True, float('nan')])
    algebraic[10] = custom_piecewise([greater_equal(algebraic[6] , states[0]), constants[0]*algebraic[8]*(power(fabs(algebraic[6]-states[0]), 0.500000)) , less(algebraic[6] , states[0]), -1.00000*constants[0]*algebraic[8]*(power(fabs(states[0]-algebraic[6]), 0.500000)) , True, float('nan')])
    algebraic[1] = voi-constants[39]*floor(voi/constants[39])
    algebraic[3] = custom_piecewise([greater_equal(algebraic[1] , 0.00000) & less_equal(algebraic[1] , constants[40]*constants[39]), 1.00000-cos((3.14159*algebraic[1])/(constants[40]*constants[39])) , greater(algebraic[1] , constants[40]*constants[39]) & less_equal(algebraic[1] , constants[41]*constants[39]), 1.00000+cos((3.14159*(algebraic[1]-constants[40]*constants[39]))/((constants[41]-constants[40])*constants[39])) , greater(algebraic[1] , constants[41]*constants[39]) & less(algebraic[1] , constants[39]), 0.00000 , True, float('nan')])
    algebraic[5] = constants[38]+(algebraic[3]*(constants[37]-constants[38]))/2.00000
    algebraic[7] = constants[30]+algebraic[5]*(states[4]-constants[31])
    algebraic[9] = custom_piecewise([greater_equal(algebraic[7] , states[1]), 1.00000 , less(algebraic[7] , states[1]), 0.00000 , True, float('nan')])
    algebraic[11] = custom_piecewise([greater_equal(algebraic[7] , states[1]), constants[29]*algebraic[9]*(power(fabs(algebraic[7]-states[1]), 0.500000)) , less(algebraic[7] , states[1]), -1.00000*constants[29]*algebraic[9]*(power(fabs(states[1]-algebraic[7]), 0.500000)) , True, float('nan')])
    algebraic[12] = voi-constants[10]*floor(voi/constants[10])
    algebraic[14] = custom_piecewise([greater_equal(algebraic[12] , 0.00000) & less_equal(algebraic[12] , (constants[15]+constants[16])*constants[10]-constants[10]), 1.00000-cos((2.00000*3.14159*((algebraic[12]-constants[15]*constants[10])+constants[10]))/(constants[16]*constants[10])) , greater(algebraic[12] , (constants[15]+constants[16])*constants[10]-constants[10]) & less_equal(algebraic[12] , constants[15]*constants[10]), 0.00000 , greater(algebraic[12] , constants[15]*constants[10]) & less_equal(algebraic[12] , constants[10]), 1.00000-cos((2.00000*3.14159*(algebraic[12]-constants[15]*constants[10]))/(constants[16]*constants[10])) , True, float('nan')])
    algebraic[16] = constants[14]+(algebraic[14]*(constants[13]-constants[14]))/2.00000
    algebraic[18] = constants[5]+algebraic[16]*(states[3]-constants[6])
    algebraic[20] = custom_piecewise([greater_equal(algebraic[18] , algebraic[6]), 1.00000 , less(algebraic[18] , algebraic[6]), 0.00000 , True, float('nan')])
    algebraic[22] = custom_piecewise([greater_equal(algebraic[18] , algebraic[6]), constants[4]*algebraic[20]*(power(fabs(algebraic[18]-algebraic[6]), 0.500000)) , less(algebraic[18] , algebraic[6]), -1.00000*constants[4]*algebraic[20]*(power(fabs(algebraic[6]-algebraic[18]), 0.500000)) , True, float('nan')])
    algebraic[13] = voi-constants[39]*floor(voi/constants[39])
    algebraic[15] = custom_piecewise([greater_equal(algebraic[13] , 0.00000) & less_equal(algebraic[13] , (constants[44]+constants[45])*constants[39]-constants[39]), 1.00000-cos((2.00000*3.14159*((algebraic[13]-constants[44]*constants[39])+constants[39]))/(constants[45]*constants[39])) , greater(algebraic[13] , (constants[44]+constants[45])*constants[39]-constants[39]) & less_equal(algebraic[13] , constants[44]*constants[39]), 0.00000 , greater(algebraic[13] , constants[44]*constants[39]) & less_equal(algebraic[13] , constants[39]), 1.00000-cos((2.00000*3.14159*(algebraic[13]-constants[44]*constants[39]))/(constants[45]*constants[39])) , True, float('nan')])
    algebraic[17] = constants[43]+(algebraic[15]*(constants[42]-constants[43]))/2.00000
    algebraic[19] = constants[34]+algebraic[17]*(states[5]-constants[35])
    algebraic[21] = custom_piecewise([greater_equal(algebraic[19] , algebraic[7]), 1.00000 , less(algebraic[19] , algebraic[7]), 0.00000 , True, float('nan')])
    algebraic[23] = custom_piecewise([greater_equal(algebraic[19] , algebraic[7]), constants[33]*algebraic[21]*(power(fabs(algebraic[19]-algebraic[7]), 0.500000)) , less(algebraic[19] , algebraic[7]), -1.00000*constants[33]*algebraic[21]*(power(fabs(algebraic[7]-algebraic[19]), 0.500000)) , True, float('nan')])
    algebraic[30] = (states[13]-algebraic[18])/constants[85]
    algebraic[31] = (states[9]-algebraic[19])/constants[70]
    algebraic[25] = states[8]
    algebraic[28] = states[9]+constants[69]*algebraic[25]
    algebraic[32] = algebraic[28]+constants[68]*states[8]
    algebraic[27] = algebraic[25]
    algebraic[24] = states[12]
    algebraic[29] = states[13]+constants[84]*algebraic[24]
    algebraic[33] = algebraic[29]+constants[83]*states[12]
    algebraic[26] = algebraic[24]
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