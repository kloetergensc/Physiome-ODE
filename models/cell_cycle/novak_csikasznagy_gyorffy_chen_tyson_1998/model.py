# Size of variable arrays:
sizeAlgebraic = 10
sizeStates = 11
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
    legend_states[0] = "Cdc13_Cdc2 in component Cdc13_Cdc2 (dimensionless)"
    legend_constants[0] = "k1 in component Cdc13_Cdc2 (first_order_rate_constant)"
    legend_algebraic[3] = "kwee in component kwee (first_order_rate_constant)"
    legend_constants[1] = "ki in component parameters (first_order_rate_constant)"
    legend_algebraic[2] = "kcdc25 in component kcdc25 (first_order_rate_constant)"
    legend_constants[2] = "kir in component parameters (first_order_rate_constant)"
    legend_algebraic[0] = "k2 in component k2 (first_order_rate_constant)"
    legend_algebraic[5] = "k4 in component k4 (first_order_rate_constant)"
    legend_states[1] = "mass in component mass (dimensionless)"
    legend_states[2] = "Cdc13_P_Cdc2 in component Cdc13_P_Cdc2 (dimensionless)"
    legend_states[3] = "Rum1_Cdc13_Cdc2 in component Rum1_Cdc13_Cdc2 (dimensionless)"
    legend_states[4] = "Rum1 in component Rum1 (dimensionless)"
    legend_algebraic[1] = "k2c in component k2c (first_order_rate_constant)"
    legend_constants[3] = "k3 in component Rum1 (first_order_rate_constant)"
    legend_states[5] = "Cdc25P in component Cdc25P (dimensionless)"
    legend_constants[4] = "k25 in component Cdc25P (first_order_rate_constant)"
    legend_algebraic[7] = "k25r in component Cdc25P (first_order_rate_constant)"
    legend_constants[5] = "k25r_ in component Cdc25P (first_order_rate_constant)"
    legend_constants[6] = "J25 in component Cdc25P (dimensionless)"
    legend_constants[7] = "J25r in component Cdc25P (dimensionless)"
    legend_algebraic[6] = "ks in component parameters (first_order_rate_constant)"
    legend_algebraic[4] = "MPF in component MPF (dimensionless)"
    legend_states[6] = "Wee1 in component Wee1 (dimensionless)"
    legend_constants[8] = "kw in component Wee1 (first_order_rate_constant)"
    legend_algebraic[8] = "kwr in component Wee1 (first_order_rate_constant)"
    legend_constants[9] = "kwr_ in component Wee1 (first_order_rate_constant)"
    legend_constants[10] = "Jw in component Wee1 (dimensionless)"
    legend_constants[11] = "Jwr in component Wee1 (dimensionless)"
    legend_states[7] = "Mik1 in component Mik1 (dimensionless)"
    legend_constants[12] = "km in component Mik1 (first_order_rate_constant)"
    legend_algebraic[9] = "kmr in component Mik1 (first_order_rate_constant)"
    legend_constants[13] = "kmr_ in component Mik1 (first_order_rate_constant)"
    legend_constants[14] = "Jm in component Mik1 (dimensionless)"
    legend_constants[15] = "Jmr in component Mik1 (dimensionless)"
    legend_states[8] = "AAE_total in component AAE_total (dimensionless)"
    legend_constants[16] = "kas in component AAE_total (first_order_rate_constant)"
    legend_constants[17] = "kad in component parameters (first_order_rate_constant)"
    legend_states[9] = "AAE in component AAE (dimensionless)"
    legend_constants[18] = "kaa in component AAE (first_order_rate_constant)"
    legend_constants[19] = "kaa_ in component AAE (first_order_rate_constant)"
    legend_constants[44] = "kai in component AAE (first_order_rate_constant)"
    legend_constants[20] = "kai_ in component AAE (first_order_rate_constant)"
    legend_constants[21] = "kx in component AAE (first_order_rate_constant)"
    legend_constants[22] = "Jaa in component AAE (dimensionless)"
    legend_constants[23] = "Jai in component AAE (dimensionless)"
    legend_states[10] = "APC in component APC (dimensionless)"
    legend_constants[24] = "kapr in component APC (first_order_rate_constant)"
    legend_constants[25] = "kapr_ in component APC (first_order_rate_constant)"
    legend_constants[26] = "kap in component APC (first_order_rate_constant)"
    legend_constants[27] = "Japr in component APC (dimensionless)"
    legend_constants[28] = "Jap in component APC (dimensionless)"
    legend_constants[29] = "Puc1 in component parameters (dimensionless)"
    legend_constants[30] = "mu in component mass (first_order_rate_constant)"
    legend_constants[31] = "V2 in component k2 (first_order_rate_constant)"
    legend_constants[32] = "V2_ in component k2 (first_order_rate_constant)"
    legend_constants[33] = "V2c in component k2c (first_order_rate_constant)"
    legend_constants[34] = "V2c_ in component k2c (first_order_rate_constant)"
    legend_constants[35] = "k4_ in component k4 (first_order_rate_constant)"
    legend_constants[36] = "k4__ in component k4 (first_order_rate_constant)"
    legend_constants[37] = "V25 in component kcdc25 (first_order_rate_constant)"
    legend_constants[38] = "V25_ in component kcdc25 (first_order_rate_constant)"
    legend_constants[39] = "Vwee in component kwee (first_order_rate_constant)"
    legend_constants[40] = "Vwee_ in component kwee (first_order_rate_constant)"
    legend_constants[41] = "Vmik in component kwee (first_order_rate_constant)"
    legend_constants[42] = "Vmik_ in component kwee (first_order_rate_constant)"
    legend_constants[43] = "alpha in component MPF (dimensionless)"
    legend_rates[0] = "d/dt Cdc13_Cdc2 in component Cdc13_Cdc2 (dimensionless)"
    legend_rates[2] = "d/dt Cdc13_P_Cdc2 in component Cdc13_P_Cdc2 (dimensionless)"
    legend_rates[3] = "d/dt Rum1_Cdc13_Cdc2 in component Rum1_Cdc13_Cdc2 (dimensionless)"
    legend_rates[4] = "d/dt Rum1 in component Rum1 (dimensionless)"
    legend_rates[5] = "d/dt Cdc25P in component Cdc25P (dimensionless)"
    legend_rates[6] = "d/dt Wee1 in component Wee1 (dimensionless)"
    legend_rates[7] = "d/dt Mik1 in component Mik1 (dimensionless)"
    legend_rates[8] = "d/dt AAE_total in component AAE_total (dimensionless)"
    legend_rates[9] = "d/dt AAE in component AAE (dimensionless)"
    legend_rates[10] = "d/dt APC in component APC (dimensionless)"
    legend_rates[1] = "d/dt mass in component mass (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 1
    constants[0] = 0.03
    constants[1] = 200
    constants[2] = 1
    states[1] = 1
    states[2] = 0
    states[3] = 0
    states[4] = 0
    constants[3] = 0.15
    states[5] = 0
    constants[4] = 0.5
    constants[5] = 0.2
    constants[6] = 0.2
    constants[7] = 0.2
    states[6] = 0
    constants[8] = 0.5
    constants[9] = 0.2
    constants[10] = 0.2
    constants[11] = 0.2
    states[7] = 0
    constants[12] = 0.1
    constants[13] = 0
    constants[14] = 0.2
    constants[15] = 0.2
    states[8] = 2
    constants[16] = 0.25
    constants[17] = 0.1
    states[9] = 2
    constants[18] = 0.001
    constants[19] = 1
    constants[20] = 0.25
    constants[21] = 0
    constants[22] = 0.1
    constants[23] = 0.1
    states[10] = 0
    constants[24] = 0.04
    constants[25] = 3
    constants[26] = 4
    constants[27] = 0.01
    constants[28] = 0.01
    constants[29] = 0.013
    constants[30] = 0.005776
    constants[31] = 0.03
    constants[32] = 1
    constants[33] = 0.03
    constants[34] = 0.16
    constants[35] = 0.15
    constants[36] = 20
    constants[37] = 0.01
    constants[38] = 0.4
    constants[39] = 0.01
    constants[40] = 0.93
    constants[41] = 0.002
    constants[42] = 0.2
    constants[43] = 0.1
    constants[44] = constants[20]+constants[21]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[1] = constants[30]*states[1]
    algebraic[3] = constants[39]*((1.00000-states[6])*1.00000+constants[40]*states[6]+constants[41]*(1.00000-states[7]))*1.00000+constants[42]*states[7]
    algebraic[2] = constants[37]*(1.00000-states[5])+constants[38]*states[5]
    algebraic[0] = constants[31]*(1.00000-states[10])+constants[32]*states[10]
    rates[2] = algebraic[3]*states[0]-(algebraic[2]*states[2]+algebraic[0]*states[2])
    algebraic[4] = states[0]+constants[43]*states[2]
    rates[8] = constants[16]*algebraic[4]-constants[17]*states[8]
    rates[9] = ((constants[18]+constants[19]*algebraic[4])*(states[8]-states[9]))/((constants[22]+states[8])-states[9])-((constants[44]*states[9])/(constants[23]+states[9])+constants[17]*states[9])
    rates[10] = ((constants[24]+constants[25]*states[9])*(1.00000-states[10]))/((constants[27]+1.00000)-states[10])-(constants[26]*(constants[29]*states[1]+algebraic[4])*states[10])/(constants[28]+states[10])
    algebraic[5] = constants[35]+constants[36]*(constants[29]*states[1]+algebraic[4])
    rates[0] = (constants[0]*states[1]+algebraic[2]*states[2]+states[3]*(constants[2]+algebraic[5]))-(algebraic[3]*states[0]+constants[1]*states[0]*states[4]+algebraic[0]*states[0])
    algebraic[1] = constants[33]*(1.00000-states[10])+constants[34]*states[10]
    rates[3] = constants[1]*states[4]*states[0]-states[3]*(algebraic[5]+algebraic[1]+constants[2])
    rates[4] = (constants[3]+states[3]*(constants[2]+algebraic[1]))-(algebraic[5]*states[4]+constants[1]*states[0]*states[4])
    algebraic[6] = custom_piecewise([less_equal(states[10] , 0.200000), 0.500000 , True, 0.00000])
    algebraic[7] = constants[5]+algebraic[6]
    rates[5] = (constants[4]*algebraic[4]*(1.00000-states[5]))/((constants[6]+1.00000)-states[5])-(algebraic[7]*states[5])/(constants[7]+states[5])
    algebraic[8] = constants[9]+algebraic[6]
    rates[6] = (algebraic[8]*(1.00000-states[6]))/((constants[11]+1.00000)-states[6])-(constants[8]*algebraic[4]*states[6])/(constants[10]+states[6])
    algebraic[9] = constants[13]+algebraic[6]
    rates[7] = (algebraic[9]*(1.00000-states[7]))/((constants[15]+1.00000)-states[7])-(constants[12]*states[7])/(constants[14]+states[7])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[3] = constants[39]*((1.00000-states[6])*1.00000+constants[40]*states[6]+constants[41]*(1.00000-states[7]))*1.00000+constants[42]*states[7]
    algebraic[2] = constants[37]*(1.00000-states[5])+constants[38]*states[5]
    algebraic[0] = constants[31]*(1.00000-states[10])+constants[32]*states[10]
    algebraic[4] = states[0]+constants[43]*states[2]
    algebraic[5] = constants[35]+constants[36]*(constants[29]*states[1]+algebraic[4])
    algebraic[1] = constants[33]*(1.00000-states[10])+constants[34]*states[10]
    algebraic[6] = custom_piecewise([less_equal(states[10] , 0.200000), 0.500000 , True, 0.00000])
    algebraic[7] = constants[5]+algebraic[6]
    algebraic[8] = constants[9]+algebraic[6]
    algebraic[9] = constants[13]+algebraic[6]
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