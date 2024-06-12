# Size of variable arrays:
sizeAlgebraic = 17
sizeStates = 10
sizeConstants = 19
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_states[0] = "C_6PG in component C_6PG (millimolar)"
    legend_constants[0] = "mu in component model_constants (per_hour)"
    legend_algebraic[3] = "rG6PDH in component rG6PDH (flux)"
    legend_algebraic[9] = "r6PGDH in component r6PGDH (flux)"
    legend_states[1] = "C_Ru5P in component C_Ru5P (millimolar)"
    legend_algebraic[11] = "rRu5PE in component rRu5PE (flux)"
    legend_algebraic[10] = "rR5PI in component rR5PI (flux)"
    legend_states[2] = "C_R5P in component C_R5P (millimolar)"
    legend_algebraic[12] = "rTKL1 in component rTKL1 (flux)"
    legend_algebraic[13] = "rRPPK in component rRPPK (flux)"
    legend_states[3] = "C_X5P in component C_X5P (millimolar)"
    legend_algebraic[15] = "rTKL2 in component rTKL2 (flux)"
    legend_states[4] = "C_S7P in component C_S7P (millimolar)"
    legend_algebraic[14] = "rTAL in component rTAL (flux)"
    legend_states[5] = "C_E4P in component C_E4P (millimolar)"
    legend_algebraic[16] = "rPKDA in component rPKDA (flux)"
    legend_states[6] = "C_G6P in component C_G6P (millimolar)"
    legend_algebraic[0] = "dC_G6P_dt in component C_G6P (flux)"
    legend_states[7] = "C_NADP in component C_NADP (millimolar)"
    legend_states[8] = "C_NADPH in component C_NADPH (millimolar)"
    legend_states[9] = "C_MgATP in component C_MgATP (millimolar)"
    legend_constants[1] = "K_NADP_1 in component rG6PDH (millimolar)"
    legend_constants[2] = "Ki_NADPH_1 in component rG6PDH (millimolar)"
    legend_constants[3] = "Ki_MgATP_1 in component rG6PDH (millimolar)"
    legend_algebraic[1] = "I_NADPH_1 in component rG6PDH (dimensionless)"
    legend_algebraic[2] = "I_MgATP_1 in component rG6PDH (dimensionless)"
    legend_constants[4] = "rmax_G6PDH in component rG6PDH (flux)"
    legend_constants[5] = "K_NADP_2 in component r6PGDH (millimolar)"
    legend_constants[6] = "Ki_NADPH_2 in component r6PGDH (millimolar)"
    legend_constants[7] = "Ki_MgATP_2 in component r6PGDH (millimolar)"
    legend_algebraic[4] = "I_NADPH_2 in component r6PGDH (dimensionless)"
    legend_algebraic[6] = "I_MgATP_2 in component r6PGDH (dimensionless)"
    legend_constants[8] = "rmax_6PGDH in component r6PGDH (flux)"
    legend_constants[9] = "rmax_R5PI in component rR5PI (millimole_per_second)"
    legend_constants[10] = "rmax_Ru5PE in component rRu5PE (millimole_per_second)"
    legend_constants[11] = "rmax_TKL1 in component rTKL1 (millimole_per_second)"
    legend_constants[12] = "C_GAP in component model_constants (millimolar)"
    legend_constants[13] = "rmax_TAL in component rTAL (millimole_per_second)"
    legend_constants[14] = "rmax_TKL2 in component rTKL2 (millimole_per_second)"
    legend_constants[15] = "K_PKDA in component rPKDA (millimolar)"
    legend_constants[16] = "rmax_PKDA in component rPKDA (millimole_per_second)"
    legend_constants[17] = "K_RPPK in component rRPPK (millimolar)"
    legend_constants[18] = "rmax_RPPK in component rRPPK (millimole_per_second)"
    legend_algebraic[7] = "rHK in component rHK (flux)"
    legend_algebraic[5] = "qs in component model_constants (flux)"
    legend_algebraic[8] = "rPGI in component rPGI (flux)"
    legend_rates[0] = "d/dt C_6PG in component C_6PG (millimolar)"
    legend_rates[1] = "d/dt C_Ru5P in component C_Ru5P (millimolar)"
    legend_rates[2] = "d/dt C_R5P in component C_R5P (millimolar)"
    legend_rates[3] = "d/dt C_X5P in component C_X5P (millimolar)"
    legend_rates[4] = "d/dt C_S7P in component C_S7P (millimolar)"
    legend_rates[5] = "d/dt C_E4P in component C_E4P (millimolar)"
    legend_rates[6] = "d/dt C_G6P in component C_G6P (millimolar)"
    legend_rates[7] = "d/dt C_NADP in component C_NADP (millimolar)"
    legend_rates[8] = "d/dt C_NADPH in component C_NADPH (millimolar)"
    legend_rates[9] = "d/dt C_MgATP in component C_MgATP (millimolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.25
    constants[0] = 1.0
    states[1] = 0.033
    states[2] = 0.118
    states[3] = 0.041
    states[4] = 0.082
    states[5] = 0.029
    states[6] = 0.9
    states[7] = 0.168
    states[8] = 0.168
    states[9] = 2.3
    constants[1] = 0.116
    constants[2] = 1.702
    constants[3] = 0.33
    constants[4] = 44.19
    constants[5] = 1.848
    constants[6] = 0.055
    constants[7] = 0.109
    constants[8] = 0.654
    constants[9] = 0.57
    constants[10] = 0.85
    constants[11] = 3.24
    constants[12] = 0.064
    constants[13] = 3.0
    constants[14] = 10.5
    constants[15] = 0.0032
    constants[16] = 0.004
    constants[17] = 0.0034
    constants[18] = 0.003
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[7] = -1.48000/(9.17000+16.1000*voi+0.480000*(power(voi, 2.00000)))+((1.48000*voi)/((9.17000+16.1000*voi+0.480000*(power(voi, 2.00000)))*(9.17000+16.1000*voi+0.480000*(power(voi, 2.00000)))))*(16.1000+0.960000*voi)
    rates[8] = 0.516000/(25.3900+0.370000*voi+0.500000*(power(voi, 2.00000)))-((0.516000*voi)/((25.3900+0.370000*voi+0.500000*(power(voi, 2.00000)))*(25.3900+0.370000*voi+0.500000*(power(voi, 2.00000)))))*(0.370000+1.00000*voi)
    rates[9] = 29.8300/(29.7700+13.4200*voi+0.0500000*(power(voi, 2.00000)))-((29.8300*voi)/((29.7700+13.4200*voi+0.0500000*(power(voi, 2.00000)))*(29.7700+13.4200*voi+0.0500000*(power(voi, 2.00000)))))*(13.4200+0.100000*voi)
    algebraic[0] = 44.1000/(48.0000+1.00000*voi+0.450000*(power(voi, 2.00000)))+((44.1000*voi)/((48.0000+1.00000*voi+0.450000*(power(voi, 2.00000)))*(48.0000+1.00000*voi+0.450000*(power(voi, 2.00000)))))*(1.00000+0.900000*voi)
    rates[6] = algebraic[0]
    algebraic[1] = 1.00000+states[8]/constants[2]
    algebraic[2] = 1.00000+states[9]/constants[3]
    algebraic[3] = constants[4]*(states[7]/((states[7]+constants[1]*algebraic[1])*algebraic[2]))
    algebraic[4] = 1.00000+states[8]/constants[6]
    algebraic[6] = 1.00000+states[9]/constants[7]
    algebraic[9] = constants[8]*(states[7]/((states[7]+constants[5]*algebraic[4])*algebraic[6]))
    rates[0] = algebraic[3]-(algebraic[9]+constants[0]*states[0])
    algebraic[11] = 1.00000*constants[10]*states[1]
    algebraic[10] = 1.00000*constants[9]*states[1]
    rates[1] = algebraic[9]-(algebraic[10]+algebraic[11]+constants[0]*states[1])
    algebraic[12] = 1.00000*constants[11]*states[3]*states[2]
    algebraic[13] = 1.00000*constants[18]*(states[2]/(states[2]+constants[17]))
    rates[2] = algebraic[10]-(algebraic[12]+algebraic[13]+constants[0]*states[2])
    algebraic[14] = 1.00000*constants[13]*constants[12]*states[4]
    rates[4] = algebraic[12]-(algebraic[14]+constants[0]*states[4])
    algebraic[15] = 1.00000*constants[14]*states[5]*states[3]
    rates[3] = algebraic[11]-(algebraic[12]+algebraic[15]+constants[0]*states[3])
    algebraic[16] = 1.00000*constants[16]*(states[5]/(states[5]+constants[15]))
    rates[5] = algebraic[14]-(algebraic[15]+algebraic[16]+constants[0]*states[5])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = 44.1000/(48.0000+1.00000*voi+0.450000*(power(voi, 2.00000)))+((44.1000*voi)/((48.0000+1.00000*voi+0.450000*(power(voi, 2.00000)))*(48.0000+1.00000*voi+0.450000*(power(voi, 2.00000)))))*(1.00000+0.900000*voi)
    algebraic[1] = 1.00000+states[8]/constants[2]
    algebraic[2] = 1.00000+states[9]/constants[3]
    algebraic[3] = constants[4]*(states[7]/((states[7]+constants[1]*algebraic[1])*algebraic[2]))
    algebraic[4] = 1.00000+states[8]/constants[6]
    algebraic[6] = 1.00000+states[9]/constants[7]
    algebraic[9] = constants[8]*(states[7]/((states[7]+constants[5]*algebraic[4])*algebraic[6]))
    algebraic[11] = 1.00000*constants[10]*states[1]
    algebraic[10] = 1.00000*constants[9]*states[1]
    algebraic[12] = 1.00000*constants[11]*states[3]*states[2]
    algebraic[13] = 1.00000*constants[18]*(states[2]/(states[2]+constants[17]))
    algebraic[14] = 1.00000*constants[13]*constants[12]*states[4]
    algebraic[15] = 1.00000*constants[14]*states[5]*states[3]
    algebraic[16] = 1.00000*constants[16]*(states[5]/(states[5]+constants[15]))
    algebraic[5] = custom_piecewise([less(voi , 0.00000), 0.131000 , True, 0.546000])
    algebraic[7] = algebraic[5]
    algebraic[8] = algebraic[5]-(algebraic[3]+constants[0]*states[6]+algebraic[0])
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