# Size of variable arrays:
sizeAlgebraic = 9
sizeStates = 7
sizeConstants = 26
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component Environment (second)"
    legend_algebraic[0] = "P_L in component PluralPressureFunction (mmHg)"
    legend_algebraic[1] = "dP_Ldt in component PluralPressureFunction (mmHg_per_second)"
    legend_constants[0] = "P_m in component PluralPressureFunction (mmHg)"
    legend_constants[23] = "R in component PluralPressureFunction (mmHg_second_per_litre)"
    legend_constants[1] = "omega in component PluralPressureFunction (radian_per_second)"
    legend_constants[2] = "V_T in component PluralPressureFunction (litre)"
    legend_constants[3] = "E in component PluralPressureFunction (mmHg_per_second)"
    legend_constants[4] = "P_m in component lungMechanics (mmHg)"
    legend_states[0] = "V_A in component lungMechanics (litre)"
    legend_constants[5] = "E in component lungMechanics (mmHg_per_second)"
    legend_states[1] = "P_A in component lungMechanics (mmHg)"
    legend_constants[6] = "R in component lungMechanics (mmHg_second_per_litre)"
    legend_algebraic[6] = "Q_A in component gasExchange (litre_per_second)"
    legend_algebraic[3] = "q in component lungMechanics (litre_per_second)"
    legend_constants[7] = "D_o in component gasExchange (mole_per_second_mmHg)"
    legend_states[2] = "f_o in component gasExchange (dimensionless)"
    legend_algebraic[7] = "f_oi in component gasExchange (dimensionless)"
    legend_constants[8] = "D_c in component gasExchange (mole_per_second_mmHg)"
    legend_states[3] = "f_c in component gasExchange (dimensionless)"
    legend_algebraic[8] = "f_ci in component gasExchange (dimensionless)"
    legend_constants[9] = "P_w in component gasExchange (mmHg)"
    legend_algebraic[4] = "p_ao in component gasExchange (mmHg)"
    legend_states[4] = "p_o in component gasTransport (mmHg)"
    legend_algebraic[5] = "p_ac in component gasExchange (mmHg)"
    legend_constants[10] = "f_om in component gasExchange (dimensionless)"
    legend_constants[11] = "f_cm in component gasExchange (dimensionless)"
    legend_states[5] = "p_c in component gasTransport (mmHg)"
    legend_constants[12] = "V_D in component gasExchange (litre)"
    legend_constants[13] = "V_T in component gasExchange (litre)"
    legend_algebraic[2] = "df_satdp in component gasTransport (dimensionless)"
    legend_constants[14] = "L in component gasTransport (dimensionless)"
    legend_constants[15] = "K_T in component gasTransport (litre_per_mole)"
    legend_constants[16] = "K_R in component gasTransport (litre_per_mole)"
    legend_constants[17] = "sigma in component gasTransport (mole_per_litre_mmHg)"
    legend_constants[18] = "V_c in component gasTransport (litre)"
    legend_constants[19] = "T_h in component gasTransport (mole_per_litre)"
    legend_constants[24] = "delta in component gasTransport (dimensionless)"
    legend_constants[25] = "h in component gasTransport (mole_per_litre)"
    legend_constants[20] = "l_2 in component gasTransport (litre_per_second_mole)"
    legend_constants[21] = "r_2 in component gasTransport (per_second)"
    legend_constants[22] = "sigma_c in component gasTransport (mole_per_litre_mmHg)"
    legend_states[6] = "z in component gasTransport (dimensionless)"
    legend_rates[1] = "d/dt P_A in component lungMechanics (mmHg)"
    legend_rates[0] = "d/dt V_A in component lungMechanics (litre)"
    legend_rates[2] = "d/dt f_o in component gasExchange (dimensionless)"
    legend_rates[3] = "d/dt f_c in component gasExchange (dimensionless)"
    legend_rates[4] = "d/dt p_o in component gasTransport (mmHg)"
    legend_rates[5] = "d/dt p_c in component gasTransport (mmHg)"
    legend_rates[6] = "d/dt z in component gasTransport (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 760
    constants[1] = 1.256637
    constants[2] = 0.41
    constants[3] = 2.5
    constants[4] = 760
    states[0] = 1
    constants[5] = 2.5
    states[1] = 760
    constants[6] = 1
    constants[7] = 0.0000156
    states[2] = 0.1368
    constants[8] = 0.0000316
    states[3] = 0.05263
    constants[9] = 47
    states[4] = 40
    constants[10] = 0.21
    constants[11] = 0
    states[5] = 46
    constants[12] = 0.151
    constants[13] = 0.41
    constants[14] = 171200000
    constants[15] = 10000
    constants[16] = 3600000
    constants[17] = 0.0000014
    constants[18] = 0.071
    constants[19] = 0.002
    constants[20] = 164000
    constants[21] = 0.12
    constants[22] = 0.000033
    states[6] = 0.00000044219
    constants[23] = (2.00000* pi*1.00000)/5.00000
    constants[24] = power(10.0000, 1.90000)
    constants[25] = 1.00000*(power(10.0000, -7.40000))
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[6] = (constants[24]*constants[21]*constants[22]*states[5])/1.00000-constants[24]*constants[20]*constants[25]*states[6]
    algebraic[0] = (constants[0]-((constants[23]*constants[1]*constants[2])/2.00000)*sin((constants[1]*voi)/1.00000))-constants[3]*(2.50000-((constants[2]*1.00000)/2.00000)*cos((constants[1]*voi)/1.00000))
    rates[0] = ((constants[4]-algebraic[0])-(states[0]*constants[5])/1.00000)/constants[6]
    algebraic[2] = ((constants[14]*(power(1.00000+constants[15]*constants[17]*states[4], 4.00000))+power(1.00000+constants[16]*constants[17]*states[4], 4.00000))*(3.00000*constants[14]*(power(constants[15], 2.00000))*(power(constants[17], 2.00000))*states[4]*1.00000*(power(1.00000+constants[15]*constants[17]*states[4], 2.00000))+constants[14]*constants[15]*constants[17]*1.00000*(power(1.00000+constants[15]*constants[17]*states[4], 3.00000))+3.00000*(power(constants[16], 2.00000))*(power(constants[17], 2.00000))*states[4]*1.00000*(power(1.00000+constants[16]*constants[17]*states[4], 2.00000))+constants[16]*constants[17]*1.00000*(power(1.00000+constants[16]*constants[17]*states[4], 3.00000)))-(constants[14]*constants[15]*constants[17]*states[4]*(power(1.00000+constants[15]*constants[17]*states[4], 3.00000))+constants[16]*constants[17]*states[4]*(power(1.00000+constants[16]*constants[17]*states[4], 3.00000)))*(4.00000*constants[14]*constants[15]*constants[17]*1.00000*(power(1.00000+constants[15]*constants[17]*states[4], 3.00000))+4.00000*constants[16]*constants[17]*1.00000*(power(1.00000+constants[16]*constants[17]*states[4], 3.00000))))/(power(constants[14]*(power(1.00000+constants[15]*constants[17]*states[4], 4.00000))+power(1.00000+constants[16]*constants[17]*states[4], 4.00000), 2.00000))
    rates[4] = (constants[7]/(constants[17]*constants[18]))*(power(1.00000+((4.00000*constants[19])/constants[17])*algebraic[2], -1.00000))*(states[2]*(states[1]-constants[9])-states[4])
    algebraic[5] = states[3]*(states[1]-constants[9])
    rates[5] = ((constants[8]/(constants[22]*constants[18]))*(algebraic[5]-states[5])+((1.00000*constants[24]*constants[20])/constants[22])*constants[25]*states[6])-constants[24]*constants[21]*states[5]
    algebraic[1] = ((-constants[23]*(power(constants[1], 2.00000))*constants[2])/(2.00000*1.00000))*cos((constants[1]*voi)/1.00000)-constants[3]*(2.50000-(constants[2]/2.00000)*sin((constants[1]*voi)/1.00000))
    algebraic[3] = (constants[4]-states[1])/constants[6]
    algebraic[4] = states[2]*(states[1]-constants[9])
    algebraic[6] = algebraic[3]+1.00000*constants[8]*(states[5]-algebraic[5])+1.00000*constants[7]*(states[4]-algebraic[4])
    rates[1] = (constants[4]*constants[5]*algebraic[6])/(states[1]*1.00000)+algebraic[1]
    algebraic[7] = custom_piecewise([greater_equal(constants[13] , constants[12]), (states[2]*constants[12]+constants[10]*(constants[13]-constants[12]))/constants[13] , True, states[2]])
    rates[2] = (1.00000/states[0])*((1.00000*constants[7]*(states[4]-algebraic[4])+(algebraic[7]-states[2])*algebraic[3])-states[2]*(1.00000*constants[8]*(states[5]-algebraic[5])+1.00000*constants[7]*(states[4]-algebraic[4])))
    algebraic[8] = custom_piecewise([greater_equal(constants[13] , constants[12]), (states[3]*constants[12]+constants[11]*(constants[13]-constants[12]))/constants[13] , True, states[3]])
    rates[3] = (1.00000/states[0])*((1.00000*constants[8]*(states[5]-algebraic[5])+(algebraic[8]-states[3])*algebraic[3])-states[3]*(1.00000*constants[7]*(states[4]-algebraic[4])+1.00000*constants[8]*(states[5]-algebraic[5])))
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = (constants[0]-((constants[23]*constants[1]*constants[2])/2.00000)*sin((constants[1]*voi)/1.00000))-constants[3]*(2.50000-((constants[2]*1.00000)/2.00000)*cos((constants[1]*voi)/1.00000))
    algebraic[2] = ((constants[14]*(power(1.00000+constants[15]*constants[17]*states[4], 4.00000))+power(1.00000+constants[16]*constants[17]*states[4], 4.00000))*(3.00000*constants[14]*(power(constants[15], 2.00000))*(power(constants[17], 2.00000))*states[4]*1.00000*(power(1.00000+constants[15]*constants[17]*states[4], 2.00000))+constants[14]*constants[15]*constants[17]*1.00000*(power(1.00000+constants[15]*constants[17]*states[4], 3.00000))+3.00000*(power(constants[16], 2.00000))*(power(constants[17], 2.00000))*states[4]*1.00000*(power(1.00000+constants[16]*constants[17]*states[4], 2.00000))+constants[16]*constants[17]*1.00000*(power(1.00000+constants[16]*constants[17]*states[4], 3.00000)))-(constants[14]*constants[15]*constants[17]*states[4]*(power(1.00000+constants[15]*constants[17]*states[4], 3.00000))+constants[16]*constants[17]*states[4]*(power(1.00000+constants[16]*constants[17]*states[4], 3.00000)))*(4.00000*constants[14]*constants[15]*constants[17]*1.00000*(power(1.00000+constants[15]*constants[17]*states[4], 3.00000))+4.00000*constants[16]*constants[17]*1.00000*(power(1.00000+constants[16]*constants[17]*states[4], 3.00000))))/(power(constants[14]*(power(1.00000+constants[15]*constants[17]*states[4], 4.00000))+power(1.00000+constants[16]*constants[17]*states[4], 4.00000), 2.00000))
    algebraic[5] = states[3]*(states[1]-constants[9])
    algebraic[1] = ((-constants[23]*(power(constants[1], 2.00000))*constants[2])/(2.00000*1.00000))*cos((constants[1]*voi)/1.00000)-constants[3]*(2.50000-(constants[2]/2.00000)*sin((constants[1]*voi)/1.00000))
    algebraic[3] = (constants[4]-states[1])/constants[6]
    algebraic[4] = states[2]*(states[1]-constants[9])
    algebraic[6] = algebraic[3]+1.00000*constants[8]*(states[5]-algebraic[5])+1.00000*constants[7]*(states[4]-algebraic[4])
    algebraic[7] = custom_piecewise([greater_equal(constants[13] , constants[12]), (states[2]*constants[12]+constants[10]*(constants[13]-constants[12]))/constants[13] , True, states[2]])
    algebraic[8] = custom_piecewise([greater_equal(constants[13] , constants[12]), (states[3]*constants[12]+constants[11]*(constants[13]-constants[12]))/constants[13] , True, states[3]])
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