# Size of variable arrays:
sizeAlgebraic = 2
sizeStates = 9
sizeConstants = 30
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (hour)"
    legend_constants[29] = "F_b_dopa in component gastrointestinal_compartment_L_dopa (dimensionless)"
    legend_states[0] = "A_dopa_b in component gastrointestinal_compartment_L_dopa (umole)"
    legend_constants[0] = "ka_b_dopa in component gastrointestinal_compartment_L_dopa (per_hour)"
    legend_constants[21] = "F_G in component gastrointestinal_compartment_L_dopa (dimensionless)"
    legend_constants[28] = "F_H in component gastrointestinal_compartment_L_dopa (dimensionless)"
    legend_constants[25] = "CL_H in component gastrointestinal_compartment_L_dopa (liter_per_hour)"
    legend_constants[1] = "Q in component gastrointestinal_compartment_L_dopa (liter_per_hour)"
    legend_constants[2] = "f_H in component gastrointestinal_compartment_L_dopa (dimensionless)"
    legend_constants[3] = "CL_dopa_0 in component L_dopa_clearance (liter_per_hour)"
    legend_states[1] = "C_dopa_b in component body_compartment_L_dopa (uM)"
    legend_constants[4] = "V_dopa in component body_compartment_L_dopa (liter)"
    legend_algebraic[1] = "CL_dopa in component L_dopa_clearance (liter_per_hour)"
    legend_states[2] = "C_OMD_b in component body_compartment_3_OMD (uM)"
    legend_constants[5] = "CL_OMD_b in component body_compartment_3_OMD (liter_per_hour)"
    legend_constants[6] = "V_OMD_b in component body_compartment_3_OMD (liter)"
    legend_constants[22] = "CL_COMT in component L_dopa_clearance (liter_per_hour)"
    legend_algebraic[0] = "CL_AADC in component L_dopa_clearance (liter_per_hour)"
    legend_constants[23] = "CL_AADC0 in component L_dopa_clearance (liter_per_hour)"
    legend_states[3] = "C_Ro_central in component central_compartment_Ro (uM)"
    legend_constants[26] = "CL_REST in component L_dopa_clearance (liter_per_hour)"
    legend_constants[7] = "ki in component L_dopa_clearance (uM)"
    legend_states[4] = "A_bens in component gastrointestinal_compartment_benserazide (umole)"
    legend_constants[8] = "ka_Bens in component gastrointestinal_compartment_benserazide (per_hour)"
    legend_constants[9] = "F_Bens in component gastrointestinal_compartment_benserazide (dimensionless)"
    legend_states[5] = "C_bens_central in component central_compartment_benserazide (uM)"
    legend_constants[10] = "V1_B in component central_compartment_benserazide (liter)"
    legend_constants[11] = "CLd_B in component central_compartment_benserazide (liter_per_hour)"
    legend_states[6] = "C_bens_peripheral in component peripheral_compartment_benserazide (uM)"
    legend_constants[24] = "CL_bens_total in component benserazide_clearance (liter_per_hour)"
    legend_constants[12] = "V2_B in component peripheral_compartment_benserazide (liter)"
    legend_constants[27] = "CL_Ro in component benserazide_clearance (liter_per_hour)"
    legend_constants[13] = "CL_B in component benserazide_clearance (liter_per_hour)"
    legend_constants[14] = "fm in component benserazide_clearance (dimensionless)"
    legend_states[7] = "A_Ro in component gastrointestinal_compartment_Ro (umole)"
    legend_constants[15] = "ka_M in component gastrointestinal_compartment_Ro (per_hour)"
    legend_constants[16] = "F_Ro in component gastrointestinal_compartment_Ro (dimensionless)"
    legend_constants[17] = "V1_M in component central_compartment_Ro (liter)"
    legend_constants[18] = "CLd_M in component central_compartment_Ro (liter_per_hour)"
    legend_constants[19] = "CL_M in component central_compartment_Ro (liter_per_hour)"
    legend_states[8] = "C_Ro_peripheral in component peripheral_compartment_Ro (uM)"
    legend_constants[20] = "V2_M in component peripheral_compartment_Ro (liter)"
    legend_rates[0] = "d/dt A_dopa_b in component gastrointestinal_compartment_L_dopa (umole)"
    legend_rates[1] = "d/dt C_dopa_b in component body_compartment_L_dopa (uM)"
    legend_rates[2] = "d/dt C_OMD_b in component body_compartment_3_OMD (uM)"
    legend_rates[4] = "d/dt A_bens in component gastrointestinal_compartment_benserazide (umole)"
    legend_rates[5] = "d/dt C_bens_central in component central_compartment_benserazide (uM)"
    legend_rates[6] = "d/dt C_bens_peripheral in component peripheral_compartment_benserazide (uM)"
    legend_rates[7] = "d/dt A_Ro in component gastrointestinal_compartment_Ro (umole)"
    legend_rates[3] = "d/dt C_Ro_central in component central_compartment_Ro (uM)"
    legend_rates[8] = "d/dt C_Ro_peripheral in component peripheral_compartment_Ro (uM)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 101
    constants[0] = 2.11
    constants[1] = 0.828
    constants[2] = 0.13
    constants[3] = 0.823
    states[1] = 0
    constants[4] = 0.496
    states[2] = 0
    constants[5] = 0.012
    constants[6] = 0.196
    states[3] = 0
    constants[7] = 0.00246
    states[4] = 19.51
    constants[8] = 0.94
    constants[9] = 0.022
    states[5] = 0
    constants[10] = 0.202
    constants[11] = 0.072
    states[6] = 0
    constants[12] = 0.127
    constants[13] = 1.67
    constants[14] = 0.15
    states[7] = 1.3658
    constants[15] = 2.47
    constants[16] = 1
    constants[17] = 0.0691
    constants[18] = 1.06
    constants[19] = 4.29
    states[8] = 0
    constants[20] = 3.2
    constants[21] = 0.244000
    constants[22] = constants[3]*0.100000
    constants[23] = constants[3]*0.690000
    constants[24] = constants[13]/(1.00000-constants[14])
    constants[25] = constants[2]*constants[3]
    constants[26] = constants[3]*0.210000
    constants[27] = constants[24]*constants[14]
    constants[28] = 1.00000-constants[25]/constants[1]
    constants[29] = constants[28]*constants[21]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = -constants[0]*states[0]
    rates[2] = (1.00000/constants[6])*(constants[22]*states[1]-constants[5]*states[2])
    rates[4] = -constants[8]*states[4]
    rates[5] = (1.00000/constants[10])*((constants[8]*states[4]*constants[9]-constants[24]*states[5])+constants[11]*(states[6]-states[5]))
    rates[6] = (1.00000/constants[12])*constants[11]*(states[5]-states[6])
    rates[7] = -constants[15]*states[7]
    rates[3] = (1.00000/constants[17])*((constants[15]*states[7]*constants[16]-constants[19]*states[3])+constants[27]*states[5]+constants[18]*(states[8]-states[3]))
    rates[8] = (1.00000/constants[20])*constants[18]*(states[3]-states[8])
    algebraic[0] = constants[23]/(1.00000+states[3]/constants[7])
    algebraic[1] = algebraic[0]+constants[22]+constants[26]
    rates[1] = (1.00000/constants[4])*(constants[0]*states[0]*constants[29]-algebraic[1]*states[1])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = constants[23]/(1.00000+states[3]/constants[7])
    algebraic[1] = algebraic[0]+constants[22]+constants[26]
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