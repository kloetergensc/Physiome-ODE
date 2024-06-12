# Size of variable arrays:
sizeAlgebraic = 16
sizeStates = 3
sizeConstants = 37
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (hour)"
    legend_constants[0] = "T_a in component model_parameters (celsius)"
    legend_constants[1] = "T_b in component model_parameters (celsius)"
    legend_constants[2] = "delta_T in component model_parameters (celsius)"
    legend_constants[3] = "kinc in component model_parameters (W_per_kg_C2)"
    legend_constants[4] = "tdose1 in component model_parameters (hour)"
    legend_constants[5] = "tdose2 in component model_parameters (hour)"
    legend_constants[6] = "tdose3 in component model_parameters (hour)"
    legend_algebraic[5] = "M_c in component M_c (W_per_kg)"
    legend_constants[7] = "t_day in component M_c (hour)"
    legend_constants[8] = "t_night in component M_c (hour)"
    legend_algebraic[1] = "tprime in component M_c (second)"
    legend_constants[9] = "day_length in component M_c (second)"
    legend_constants[35] = "M_day in component M_day (W_per_kg)"
    legend_algebraic[3] = "M_night in component M_night (W_per_kg)"
    legend_states[0] = "M in component M (W_per_kg)"
    legend_constants[10] = "km in component M (per_hour)"
    legend_states[1] = "T in component T (celsius)"
    legend_constants[11] = "c in component T (kJ_per_kg_C)"
    legend_algebraic[0] = "k in component k (W_per_kg_C)"
    legend_states[2] = "BR in component k (dimensionless)"
    legend_constants[12] = "pEtot in component k (dimensionless)"
    legend_constants[13] = "kR in component k (per_day)"
    legend_constants[14] = "AMT_dose in component k (mg_per_kg)"
    legend_constants[15] = "pEf1 in component k (per_day)"
    legend_constants[16] = "pEs1 in component k (kg_per_day_mg)"
    legend_constants[17] = "pEf2 in component k (per_day)"
    legend_constants[18] = "pEs2 in component k (kg_per_day_mg)"
    legend_constants[19] = "pEf3 in component k (per_day)"
    legend_constants[20] = "pEs3 in component k (kg_per_day_mg)"
    legend_algebraic[13] = "E_slow in component k (per_day)"
    legend_algebraic[15] = "E_fast in component k (per_day)"
    legend_constants[30] = "f2_drug in component k (W_per_kg_C)"
    legend_constants[34] = "kb in component kb (W_per_kg_C)"
    legend_algebraic[2] = "f_prime in component M_night (dimensionless)"
    legend_algebraic[6] = "gNsTs1 in component gNT (dimensionless)"
    legend_algebraic[9] = "gNsTs2 in component gNT (dimensionless)"
    legend_algebraic[12] = "gNsTs3 in component gNT (dimensionless)"
    legend_algebraic[7] = "gNfTf1 in component gNT (dimensionless)"
    legend_algebraic[10] = "gNfTf2 in component gNT (dimensionless)"
    legend_algebraic[14] = "gNfTf3 in component gNT (dimensionless)"
    legend_constants[29] = "T_day in component T_day (celsius)"
    legend_constants[32] = "T_night in component T_night (celsius)"
    legend_constants[21] = "M_b in component kb (W_per_kg)"
    legend_constants[22] = "t_prime in component M_night (hour)"
    legend_constants[23] = "alpha in component M_night (per_hour)"
    legend_constants[24] = "delta_high_dose in component M_night (dimensionless)"
    legend_constants[36] = "M_night_baseline in component M_night (W_per_kg)"
    legend_constants[25] = "Ns in component gNT (dimensionless)"
    legend_constants[26] = "Nf in component gNT (dimensionless)"
    legend_constants[27] = "Ts in component gNT (day)"
    legend_constants[28] = "Tf in component gNT (day)"
    legend_algebraic[4] = "X1 in component gNT (day)"
    legend_algebraic[8] = "X2 in component gNT (day)"
    legend_algebraic[11] = "X3 in component gNT (day)"
    legend_constants[31] = "Kf in component gNT (per_day)"
    legend_constants[33] = "Ks in component gNT (per_day)"
    legend_rates[0] = "d/dt M in component M (W_per_kg)"
    legend_rates[1] = "d/dt T in component T (celsius)"
    legend_rates[2] = "d/dt BR in component k (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 21.0
    constants[1] = 38.0
    constants[2] = 1.57
    constants[3] = 0.0258
    constants[4] = 24.0
    constants[5] = 72.0
    constants[6] = 120.0
    constants[7] = 17.5
    constants[8] = 6.73
    constants[9] = 86400
    states[0] = 3.5
    constants[10] = 1.1375
    states[1] = 38.785
    constants[11] = 3.47
    states[2] = 0.0
    constants[12] = 0.144
    constants[13] = 5.35
    constants[14] = 3.0
    constants[15] = 1.0
    constants[16] = 0.2
    constants[17] = 3.57
    constants[18] = 2.43
    constants[19] = 8.0
    constants[20] = 50.0
    constants[21] = 3.0
    constants[22] = 45.12
    constants[23] = 0.2229166
    constants[24] = 1.0
    constants[25] = 4.0
    constants[26] = 4.0
    constants[27] = 2.45
    constants[28] = 0.368
    constants[29] = constants[1]+constants[2]/2.00000
    constants[30] = 0.00000
    constants[31] = constants[26]/constants[28]
    constants[32] = constants[1]-constants[2]/2.00000
    constants[33] = constants[25]/constants[27]
    constants[34] = constants[21]/(constants[1]-constants[0])
    constants[35] = (constants[34]+constants[3]*(constants[29]-constants[1]))*(constants[29]-constants[0])
    constants[36] = (constants[34]+constants[3]*(constants[32]-constants[1]))*(constants[32]-constants[0])
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[0] = constants[34]+constants[3]*(states[1]-constants[1]*(1.00000+constants[12]*states[2]))+constants[30]
    rates[1] = (power(constants[11], -1.00000))*(states[0]-algebraic[0]*(states[1]-constants[0]))
    algebraic[1] =  voi*3600.00*1.00000 % constants[9]
    algebraic[2] = constants[24]*(power(1.00000+exp(-constants[23]*(voi-(constants[4]+constants[22]))), -1.00000))
    algebraic[3] = (1.00000-algebraic[2])*constants[36]+algebraic[2]*constants[35]
    algebraic[5] = custom_piecewise([greater_equal(algebraic[1]/3600.00 , constants[8]) & less(algebraic[1]/3600.00 , constants[7]), algebraic[3] , True, constants[35]])
    rates[0] = -constants[10]*(states[0]-algebraic[5])
    algebraic[4] = (voi-constants[4])/24.0000
    algebraic[6] = custom_piecewise([greater(algebraic[4] , 0.00000), ((power(constants[33], constants[25]))/6.00000)*exp(-constants[33]*algebraic[4])*(power(algebraic[4], constants[25]-1.00000)) , True, 0.00000])
    algebraic[8] = (voi-constants[5])/24.0000
    algebraic[9] = custom_piecewise([greater(algebraic[8] , 0.00000), ((power(constants[33], constants[25]))/6.00000)*exp(-constants[33]*algebraic[8])*(power(algebraic[8], constants[25]-1.00000)) , True, 0.00000])
    algebraic[11] = (voi-constants[6])/24.0000
    algebraic[12] = custom_piecewise([greater(algebraic[11] , 0.00000), ((power(constants[33], constants[25]))/6.00000)*exp(-constants[33]*algebraic[11])*(power(algebraic[11], constants[25]-1.00000)) , True, 0.00000])
    algebraic[13] = constants[14]*constants[18]*(algebraic[6]+algebraic[9]+algebraic[12])
    algebraic[7] = custom_piecewise([greater(algebraic[4] , 0.00000), ((power(constants[31], constants[26]))/6.00000)*exp(-constants[31]*algebraic[4])*(power(algebraic[4], constants[26]-1.00000)) , True, 0.00000])
    algebraic[10] = custom_piecewise([greater(algebraic[8] , 0.00000), ((power(constants[31], constants[26]))/6.00000)*exp(-constants[31]*algebraic[8])*(power(algebraic[8], constants[26]-1.00000)) , True, 0.00000])
    algebraic[14] = custom_piecewise([greater(algebraic[11] , 0.00000), ((power(constants[31], constants[26]))/6.00000)*exp(-constants[31]*algebraic[11])*(power(algebraic[11], constants[26]-1.00000)) , True, 0.00000])
    algebraic[15] = constants[17]*(algebraic[7]+algebraic[10]+algebraic[14])
    rates[2] = (algebraic[2]*(algebraic[13]+algebraic[15]))*(1.00000-states[2])-constants[13]*states[2]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = constants[34]+constants[3]*(states[1]-constants[1]*(1.00000+constants[12]*states[2]))+constants[30]
    algebraic[1] =  voi*3600.00*1.00000 % constants[9]
    algebraic[2] = constants[24]*(power(1.00000+exp(-constants[23]*(voi-(constants[4]+constants[22]))), -1.00000))
    algebraic[3] = (1.00000-algebraic[2])*constants[36]+algebraic[2]*constants[35]
    algebraic[5] = custom_piecewise([greater_equal(algebraic[1]/3600.00 , constants[8]) & less(algebraic[1]/3600.00 , constants[7]), algebraic[3] , True, constants[35]])
    algebraic[4] = (voi-constants[4])/24.0000
    algebraic[6] = custom_piecewise([greater(algebraic[4] , 0.00000), ((power(constants[33], constants[25]))/6.00000)*exp(-constants[33]*algebraic[4])*(power(algebraic[4], constants[25]-1.00000)) , True, 0.00000])
    algebraic[8] = (voi-constants[5])/24.0000
    algebraic[9] = custom_piecewise([greater(algebraic[8] , 0.00000), ((power(constants[33], constants[25]))/6.00000)*exp(-constants[33]*algebraic[8])*(power(algebraic[8], constants[25]-1.00000)) , True, 0.00000])
    algebraic[11] = (voi-constants[6])/24.0000
    algebraic[12] = custom_piecewise([greater(algebraic[11] , 0.00000), ((power(constants[33], constants[25]))/6.00000)*exp(-constants[33]*algebraic[11])*(power(algebraic[11], constants[25]-1.00000)) , True, 0.00000])
    algebraic[13] = constants[14]*constants[18]*(algebraic[6]+algebraic[9]+algebraic[12])
    algebraic[7] = custom_piecewise([greater(algebraic[4] , 0.00000), ((power(constants[31], constants[26]))/6.00000)*exp(-constants[31]*algebraic[4])*(power(algebraic[4], constants[26]-1.00000)) , True, 0.00000])
    algebraic[10] = custom_piecewise([greater(algebraic[8] , 0.00000), ((power(constants[31], constants[26]))/6.00000)*exp(-constants[31]*algebraic[8])*(power(algebraic[8], constants[26]-1.00000)) , True, 0.00000])
    algebraic[14] = custom_piecewise([greater(algebraic[11] , 0.00000), ((power(constants[31], constants[26]))/6.00000)*exp(-constants[31]*algebraic[11])*(power(algebraic[11], constants[26]-1.00000)) , True, 0.00000])
    algebraic[15] = constants[17]*(algebraic[7]+algebraic[10]+algebraic[14])
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