# Size of variable arrays:
sizeAlgebraic = 0
sizeStates = 12
sizeConstants = 38
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_states[0] = "E in component E (millimolar)"
    legend_constants[0] = "k23 in component reaction_constants (second_order_rate_constant)"
    legend_constants[1] = "k24 in component reaction_constants (first_order_rate_constant)"
    legend_constants[2] = "k17 in component reaction_constants (first_order_rate_constant)"
    legend_constants[3] = "k18 in component reaction_constants (first_order_rate_constant)"
    legend_constants[4] = "k1 in component reaction_constants (second_order_rate_constant)"
    legend_constants[5] = "k2 in component reaction_constants (first_order_rate_constant)"
    legend_constants[6] = "k3 in component reaction_constants (second_order_rate_constant)"
    legend_constants[7] = "k4 in component reaction_constants (first_order_rate_constant)"
    legend_states[1] = "ED in component ED (millimolar)"
    legend_states[2] = "E_ in component E_ (millimolar)"
    legend_states[3] = "ENa in component ENa (millimolar)"
    legend_states[4] = "ECl in component ECl (millimolar)"
    legend_constants[8] = "D in component reaction_constants (millimolar)"
    legend_constants[9] = "Na in component reaction_constants (millimolar)"
    legend_constants[10] = "Cl in component reaction_constants (millimolar)"
    legend_constants[11] = "k29 in component reaction_constants (second_order_rate_constant)"
    legend_constants[12] = "k30 in component reaction_constants (first_order_rate_constant)"
    legend_constants[13] = "k11 in component reaction_constants (second_order_rate_constant)"
    legend_constants[14] = "k12 in component reaction_constants (first_order_rate_constant)"
    legend_constants[15] = "k9 in component reaction_constants (second_order_rate_constant)"
    legend_constants[16] = "k10 in component reaction_constants (first_order_rate_constant)"
    legend_states[5] = "ED_ in component ED_ (millimolar)"
    legend_states[6] = "ENa_ in component ENa_ (millimolar)"
    legend_states[7] = "ECl_ in component ECl_ (millimolar)"
    legend_constants[17] = "D_ in component reaction_constants (millimolar)"
    legend_constants[18] = "Na_ in component reaction_constants (millimolar)"
    legend_constants[19] = "Cl_ in component reaction_constants (millimolar)"
    legend_constants[20] = "k21 in component reaction_constants (second_order_rate_constant)"
    legend_constants[21] = "k22 in component reaction_constants (first_order_rate_constant)"
    legend_states[8] = "ENaD in component ENaD (millimolar)"
    legend_constants[22] = "k27 in component reaction_constants (second_order_rate_constant)"
    legend_constants[23] = "k28 in component reaction_constants (first_order_rate_constant)"
    legend_states[9] = "ENaD_ in component ENaD_ (millimolar)"
    legend_constants[24] = "k25 in component reaction_constants (second_order_rate_constant)"
    legend_constants[25] = "k26 in component reaction_constants (first_order_rate_constant)"
    legend_constants[26] = "k31 in component reaction_constants (second_order_rate_constant)"
    legend_constants[27] = "k32 in component reaction_constants (first_order_rate_constant)"
    legend_constants[28] = "k5 in component reaction_constants (second_order_rate_constant)"
    legend_constants[29] = "k6 in component reaction_constants (first_order_rate_constant)"
    legend_states[10] = "ENaCl in component ENaCl (millimolar)"
    legend_constants[30] = "k13 in component reaction_constants (second_order_rate_constant)"
    legend_constants[31] = "k14 in component reaction_constants (first_order_rate_constant)"
    legend_states[11] = "ENaCl_ in component ENaCl_ (millimolar)"
    legend_constants[32] = "k7 in component reaction_constants (second_order_rate_constant)"
    legend_constants[33] = "k8 in component reaction_constants (first_order_rate_constant)"
    legend_constants[34] = "k15 in component reaction_constants (second_order_rate_constant)"
    legend_constants[35] = "k16 in component reaction_constants (first_order_rate_constant)"
    legend_constants[36] = "k19 in component reaction_constants (first_order_rate_constant)"
    legend_constants[37] = "k20 in component reaction_constants (first_order_rate_constant)"
    legend_rates[0] = "d/dt E in component E (millimolar)"
    legend_rates[2] = "d/dt E_ in component E_ (millimolar)"
    legend_rates[1] = "d/dt ED in component ED (millimolar)"
    legend_rates[5] = "d/dt ED_ in component ED_ (millimolar)"
    legend_rates[8] = "d/dt ENaD in component ENaD (millimolar)"
    legend_rates[9] = "d/dt ENaD_ in component ENaD_ (millimolar)"
    legend_rates[3] = "d/dt ENa in component ENa (millimolar)"
    legend_rates[6] = "d/dt ENa_ in component ENa_ (millimolar)"
    legend_rates[4] = "d/dt ECl in component ECl (millimolar)"
    legend_rates[7] = "d/dt ECl_ in component ECl_ (millimolar)"
    legend_rates[10] = "d/dt ENaCl in component ENaCl (millimolar)"
    legend_rates[11] = "d/dt ENaCl_ in component ENaCl_ (millimolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.08333
    constants[0] = 1.0E5
    constants[1] = 3.192E1
    constants[2] = 4.587E5
    constants[3] = 1.0E5
    constants[4] = 1.0E5
    constants[5] = 4.183E5
    constants[6] = 1.0E5
    constants[7] = 4.928E6
    states[1] = 0.08333
    states[2] = 0.08333
    states[3] = 0.08333
    states[4] = 0.08333
    constants[8] = 1.0E-6
    constants[9] = 50.0
    constants[10] = 96.0
    constants[11] = 1.0E5
    constants[12] = 3.514E-1
    constants[13] = 1.0E5
    constants[14] = 4.982E6
    constants[15] = 1.0E5
    constants[16] = 4.183E5
    states[5] = 0.08333
    states[6] = 0.08333
    states[7] = 0.08333
    constants[17] = 1.0E-6
    constants[18] = 10.0
    constants[19] = 40.0
    constants[20] = 1.0E5
    constants[21] = 4.183E5
    states[8] = 0.08333
    constants[22] = 1.0E5
    constants[23] = 1.389E5
    states[9] = 0.08333
    constants[24] = 1.0E5
    constants[25] = 3.192E1
    constants[26] = 1.0E5
    constants[27] = 1.166E-1
    constants[28] = 1.0E5
    constants[29] = 1.065E6
    states[10] = 0.08333
    constants[30] = 1.0E5
    constants[31] = 1.065E6
    states[11] = 0.08333
    constants[32] = 1.0E5
    constants[33] = 8.940E4
    constants[34] = 1.0E5
    constants[35] = 8.940E4
    constants[36] = 1.0E3
    constants[37] = 2.180E2
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = (constants[1]*states[1]+constants[3]*states[2]+constants[5]*states[3]+constants[7]*states[4])-(constants[0]*constants[8]*states[0]+constants[2]*states[0]+constants[4]*constants[9]*states[0]+constants[6]*constants[10]*states[0])
    rates[2] = (constants[12]*states[5]+constants[2]*states[0]+constants[16]*states[6]+constants[14]*states[7])-(constants[11]*constants[17]*states[2]+constants[3]*states[2]+constants[15]*constants[18]*states[2]+constants[13]*constants[19]*states[2])
    rates[1] = (constants[0]*states[0]*constants[8]+constants[21]*states[8])-(constants[1]*states[1]+constants[20]*constants[9]*states[1])
    rates[5] = (constants[11]*states[2]*constants[17]+constants[23]*states[9])-(constants[12]*states[5]+constants[22]*constants[18]*states[5])
    rates[8] = (constants[20]*constants[9]*states[1]+constants[24]*constants[8]*states[3])-(constants[21]*states[8]+constants[25]*states[8])
    rates[9] = (constants[22]*constants[18]*states[5]+constants[26]*constants[17]*states[6])-(constants[23]*states[9]+constants[27]*states[9])
    rates[3] = (constants[4]*constants[9]*states[0]+constants[25]*states[8]+constants[29]*states[10])-(constants[5]*states[3]+constants[28]*constants[10]*states[3]+constants[24]*constants[8]*states[3])
    rates[6] = (constants[15]*constants[18]*states[2]+constants[27]*states[9]+constants[31]*states[11])-(constants[16]*states[6]+constants[30]*constants[19]*states[6]+constants[26]*constants[17]*states[6])
    rates[4] = (constants[6]*constants[10]*states[0]+constants[33]*states[10])-(constants[32]*constants[9]*states[4]+constants[7]*states[4])
    rates[7] = (constants[13]*constants[19]*states[2]+constants[35]*states[11])-(constants[34]*constants[18]*states[7]+constants[14]*states[7])
    rates[10] = (constants[28]*constants[10]*states[3]+constants[32]*constants[9]*states[4]+constants[37]*states[11])-(constants[29]*states[10]+constants[33]*states[10]+constants[36]*states[10])
    rates[11] = (constants[30]*constants[19]*states[6]+constants[34]*constants[18]*states[7]+constants[36]*states[10])-(constants[31]*states[11]+constants[35]*states[11]+constants[37]*states[11])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
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