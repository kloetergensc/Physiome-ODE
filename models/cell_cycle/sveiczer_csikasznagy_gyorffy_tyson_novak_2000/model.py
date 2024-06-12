# Size of variable arrays:
sizeAlgebraic = 8
sizeStates = 16
sizeConstants = 58
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_states[0] = "MPF in component MPF (dimensionless)"
    legend_constants[0] = "k1 in component rate_constants (first_order_rate_constant)"
    legend_algebraic[0] = "k2 in component rate_constants (first_order_rate_constant)"
    legend_algebraic[2] = "kwee in component rate_constants (first_order_rate_constant)"
    legend_algebraic[4] = "kc25 in component rate_constants (first_order_rate_constant)"
    legend_constants[1] = "kj in component rate_constants (first_order_rate_constant)"
    legend_constants[2] = "kjr in component rate_constants (first_order_rate_constant)"
    legend_constants[3] = "k6_ in component rate_constants (first_order_rate_constant)"
    legend_constants[4] = "k6 in component rate_constants (first_order_rate_constant)"
    legend_states[1] = "mass in component mass (dimensionless)"
    legend_states[2] = "preMPF in component preMPF (dimensionless)"
    legend_states[3] = "Rum1 in component Rum1 (dimensionless)"
    legend_states[4] = "Rum1P in component Rum1P (dimensionless)"
    legend_states[5] = "CR in component CR (dimensionless)"
    legend_states[6] = "CRP in component CRP (dimensionless)"
    legend_states[7] = "Ste9 in component Ste9 (dimensionless)"
    legend_constants[5] = "kste9r_ in component rate_constants (first_order_rate_constant)"
    legend_constants[6] = "kste9r in component rate_constants (first_order_rate_constant)"
    legend_constants[7] = "kste9 in component rate_constants (first_order_rate_constant)"
    legend_constants[8] = "Jste9r in component Ste9 (dimensionless)"
    legend_constants[9] = "Jste9 in component Ste9 (dimensionless)"
    legend_algebraic[1] = "MPF_a in component MPF_a (dimensionless)"
    legend_algebraic[3] = "PP in component PP (dimensionless)"
    legend_constants[10] = "SK in component dimensionless_constants (dimensionless)"
    legend_states[8] = "Mik1 in component Mik1 (dimensionless)"
    legend_constants[11] = "ks in component rate_constants (first_order_rate_constant)"
    legend_constants[12] = "kmr_ in component rate_constants (first_order_rate_constant)"
    legend_constants[13] = "kmr in component rate_constants (first_order_rate_constant)"
    legend_constants[14] = "km in component rate_constants (first_order_rate_constant)"
    legend_constants[15] = "Jmikr in component Mik1 (dimensionless)"
    legend_constants[16] = "Jmik in component Mik1 (dimensionless)"
    legend_states[9] = "Wee1 in component Wee1 (dimensionless)"
    legend_constants[17] = "kwr_ in component rate_constants (first_order_rate_constant)"
    legend_constants[18] = "kwr in component rate_constants (first_order_rate_constant)"
    legend_constants[19] = "kw in component rate_constants (first_order_rate_constant)"
    legend_constants[20] = "Jweer in component Wee1 (dimensionless)"
    legend_constants[21] = "Jwee in component Wee1 (dimensionless)"
    legend_states[10] = "Cdc25 in component Cdc25 (dimensionless)"
    legend_constants[22] = "k25 in component rate_constants (first_order_rate_constant)"
    legend_constants[23] = "k5 in component rate_constants (first_order_rate_constant)"
    legend_constants[24] = "k25r_ in component rate_constants (first_order_rate_constant)"
    legend_constants[25] = "k25r in component rate_constants (first_order_rate_constant)"
    legend_constants[26] = "J25 in component Cdc25 (dimensionless)"
    legend_constants[27] = "J25r in component Cdc25 (dimensionless)"
    legend_states[11] = "Slp1 in component Slp1 (dimensionless)"
    legend_constants[28] = "kas in component rate_constants (first_order_rate_constant)"
    legend_constants[29] = "kad in component rate_constants (first_order_rate_constant)"
    legend_states[12] = "Slp1_a in component Slp1_a (dimensionless)"
    legend_constants[30] = "kaa_ in component rate_constants (first_order_rate_constant)"
    legend_constants[31] = "kaa in component rate_constants (first_order_rate_constant)"
    legend_constants[32] = "kai in component rate_constants (first_order_rate_constant)"
    legend_states[13] = "Inh in component Inh (dimensionless)"
    legend_constants[33] = "k3 in component rate_constants (first_order_rate_constant)"
    legend_constants[34] = "ki in component rate_constants (first_order_rate_constant)"
    legend_constants[35] = "kir in component rate_constants (first_order_rate_constant)"
    legend_algebraic[5] = "k4 in component rate_constants (first_order_rate_constant)"
    legend_states[14] = "PI in component PI (dimensionless)"
    legend_constants[36] = "kp in component rate_constants (first_order_rate_constant)"
    legend_constants[37] = "kpp_ in component rate_constants (first_order_rate_constant)"
    legend_constants[38] = "kpp in component rate_constants (first_order_rate_constant)"
    legend_algebraic[6] = "k2c in component rate_constants (first_order_rate_constant)"
    legend_constants[39] = "epsilon_p in component dimensionless_constants (dimensionless)"
    legend_constants[40] = "mu in component mass (first_order_rate_constant)"
    legend_states[15] = "R_dna in component R_dna (dimensionless)"
    legend_constants[41] = "K in component R_dna (dimensionless)"
    legend_constants[42] = "Y in component R_dna (dimensionless)"
    legend_constants[43] = "epsilon in component dimensionless_constants (dimensionless)"
    legend_algebraic[7] = "ratio in component ratio (dimensionless)"
    legend_constants[44] = "V2_ in component rate_constants (first_order_rate_constant)"
    legend_constants[45] = "V2 in component rate_constants (first_order_rate_constant)"
    legend_constants[46] = "V2c in component rate_constants (first_order_rate_constant)"
    legend_constants[47] = "V2c_ in component rate_constants (first_order_rate_constant)"
    legend_constants[48] = "Vwee in component rate_constants (first_order_rate_constant)"
    legend_constants[49] = "Vwee_ in component rate_constants (first_order_rate_constant)"
    legend_constants[50] = "Vmik in component rate_constants (first_order_rate_constant)"
    legend_constants[51] = "Vmik_ in component rate_constants (first_order_rate_constant)"
    legend_constants[52] = "V25 in component rate_constants (first_order_rate_constant)"
    legend_constants[53] = "V25_ in component rate_constants (first_order_rate_constant)"
    legend_constants[54] = "Vpyp in component rate_constants (first_order_rate_constant)"
    legend_constants[55] = "V4 in component rate_constants (first_order_rate_constant)"
    legend_constants[56] = "V4_ in component rate_constants (first_order_rate_constant)"
    legend_constants[57] = "Pyp3 in component rate_constants (dimensionless)"
    legend_rates[0] = "d/dt MPF in component MPF (dimensionless)"
    legend_rates[2] = "d/dt preMPF in component preMPF (dimensionless)"
    legend_rates[7] = "d/dt Ste9 in component Ste9 (dimensionless)"
    legend_rates[8] = "d/dt Mik1 in component Mik1 (dimensionless)"
    legend_rates[9] = "d/dt Wee1 in component Wee1 (dimensionless)"
    legend_rates[10] = "d/dt Cdc25 in component Cdc25 (dimensionless)"
    legend_rates[11] = "d/dt Slp1 in component Slp1 (dimensionless)"
    legend_rates[12] = "d/dt Slp1_a in component Slp1_a (dimensionless)"
    legend_rates[13] = "d/dt Inh in component Inh (dimensionless)"
    legend_rates[14] = "d/dt PI in component PI (dimensionless)"
    legend_rates[6] = "d/dt CRP in component CRP (dimensionless)"
    legend_rates[5] = "d/dt CR in component CR (dimensionless)"
    legend_rates[3] = "d/dt Rum1 in component Rum1 (dimensionless)"
    legend_rates[4] = "d/dt Rum1P in component Rum1P (dimensionless)"
    legend_rates[1] = "d/dt mass in component mass (dimensionless)"
    legend_rates[15] = "d/dt R_dna in component R_dna (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0
    constants[0] = 0.02
    constants[1] = 400
    constants[2] = 1
    constants[3] = 0.1
    constants[4] = 5
    states[1] = 1
    states[2] = 1
    states[3] = 0
    states[4] = 0
    states[5] = 0
    states[6] = 0
    states[7] = 0
    constants[5] = 0.03
    constants[6] = 8
    constants[7] = 5
    constants[8] = 0.01
    constants[9] = 0.01
    constants[10] = 0.018
    states[8] = 0
    constants[11] = 0.1
    constants[12] = 0.01
    constants[13] = 5
    constants[14] = 1
    constants[15] = 0.15
    constants[16] = 0.15
    states[9] = 0
    constants[17] = 0.4
    constants[18] = 1
    constants[19] = 2
    constants[20] = 0.2
    constants[21] = 0.2
    states[10] = 0
    constants[22] = 1
    constants[23] = 0.1
    constants[24] = 0.4
    constants[25] = 2
    constants[26] = 0.05
    constants[27] = 0.05
    states[11] = 0
    constants[28] = 0.1
    constants[29] = 0.1
    states[12] = 0
    constants[30] = 0.01
    constants[31] = 0.1
    constants[32] = 0.1
    states[13] = 0
    constants[33] = 0.1
    constants[34] = 50
    constants[35] = 0.5
    states[14] = 0.2
    constants[36] = 100
    constants[37] = 1
    constants[38] = 100
    constants[39] = 0.025
    constants[40] = 0.00462
    states[15] = 1
    constants[41] = 0.06
    constants[42] = 0
    constants[43] = 0.05
    constants[44] = 0.02
    constants[45] = 1
    constants[46] = 0.5
    constants[47] = 0.02
    constants[48] = 10
    constants[49] = 0.08
    constants[50] = 2
    constants[51] = 0.04
    constants[52] = 10
    constants[53] = 0.05
    constants[54] = 0.07
    constants[55] = 1
    constants[56] = 0.01
    constants[57] = 1
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[1] = constants[40]*states[1]
    algebraic[1] = states[0]+constants[43]*states[2]
    rates[11] = constants[28]*algebraic[1]-constants[29]*states[11]
    rates[12] = (constants[30]+constants[31]*algebraic[1])*(states[11]-states[12])-(constants[32]+constants[29])*states[12]
    rates[15] = (constants[41]*1.00000)/(1.00000+constants[42]*algebraic[1])
    algebraic[3] = 1.00000-states[14]
    rates[7] = ((constants[5]+constants[6]*algebraic[3])*(1.00000-states[7]))/((constants[8]+1.00000)-states[7])-(constants[7]*(algebraic[1]+constants[10]*states[1])*states[7])/(constants[9]+states[7])
    rates[8] = ((constants[11]+constants[12]+constants[13]*algebraic[3])*(1.00000-states[8]))/((constants[15]+1.00000)-states[8])-(constants[14]*algebraic[1]*states[8])/(constants[16]+states[8])
    rates[9] = ((constants[17]+constants[18]*algebraic[3])*(1.00000-states[9]))/((constants[20]+1.00000)-states[9])-(constants[19]*algebraic[1]*states[9])/(constants[21]+states[9])
    rates[10] = (constants[22]*algebraic[1]*(1.00000-states[10]))/((constants[26]+1.00000)-states[10])-((constants[23]+constants[24]+constants[25]*algebraic[3])*states[10])/(constants[27]+states[10])
    algebraic[0] = constants[44]+constants[45]*states[7]
    algebraic[2] = constants[48]*states[9]+constants[49]*(1.00000-states[9])+constants[50]*states[8]+constants[51]*(1.00000-states[8])
    algebraic[4] = constants[52]*states[10]+constants[53]*(1.00000-states[10])+constants[54]*constants[57]
    rates[0] = ((((constants[0]*states[1]-algebraic[0]*states[0])-algebraic[2]*states[0])+algebraic[4]*states[2])-constants[1]*states[0]*(states[3]+states[4]))+constants[2]*(states[5]+states[6])+constants[3]*states[5]+(constants[3]+constants[4])*states[6]
    rates[2] = (algebraic[2]*states[0]-algebraic[4]*states[2])-algebraic[0]*states[2]
    algebraic[5] = constants[56]+constants[55]*states[12]
    rates[13] = ((constants[33]-constants[34]*states[13]*algebraic[3])+constants[35]*states[14])-algebraic[5]*states[13]
    rates[14] = (constants[34]*states[13]*algebraic[3]-constants[35]*states[14])-algebraic[5]*states[14]
    algebraic[6] = constants[47]+constants[46]*states[7]
    rates[6] = ((((constants[36]*(algebraic[1]+constants[39]*constants[10]*states[1])*states[5]-(constants[37]+constants[38]*algebraic[3])*states[6])+constants[1]*states[0]*states[4])-constants[2]*states[6])-algebraic[6]*states[6])-(constants[3]+constants[4])*states[6]
    rates[5] = ((((constants[1]*states[0]*states[3]-constants[2]*states[5])-algebraic[6]*states[5])-constants[3]*states[5])-constants[36]*(algebraic[1]+constants[39]*constants[10]*states[1])*states[5])+(constants[37]+constants[38]*algebraic[3])*states[6]
    rates[3] = ((((constants[23]-constants[3]*states[3])-constants[36]*(algebraic[1]+constants[39]*constants[10]*states[1])*states[3])+(constants[37]+constants[38]*algebraic[3])*states[4])-constants[1]*states[0]*states[3])+constants[2]*states[5]+algebraic[6]*states[5]
    rates[4] = (((constants[36]*(algebraic[1]+constants[39]*constants[10]*states[1])*states[3]-(constants[37]+constants[38]*algebraic[3])*states[4])-(constants[3]+constants[4])*states[4])-constants[1]*states[0]*states[4])+constants[2]*states[6]+algebraic[6]*states[6]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[1] = states[0]+constants[43]*states[2]
    algebraic[3] = 1.00000-states[14]
    algebraic[0] = constants[44]+constants[45]*states[7]
    algebraic[2] = constants[48]*states[9]+constants[49]*(1.00000-states[9])+constants[50]*states[8]+constants[51]*(1.00000-states[8])
    algebraic[4] = constants[52]*states[10]+constants[53]*(1.00000-states[10])+constants[54]*constants[57]
    algebraic[5] = constants[56]+constants[55]*states[12]
    algebraic[6] = constants[47]+constants[46]*states[7]
    algebraic[7] = algebraic[1]/algebraic[3]
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