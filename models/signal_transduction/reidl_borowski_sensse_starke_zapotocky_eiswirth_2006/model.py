# Size of variable arrays:
sizeAlgebraic = 7
sizeStates = 4
sizeConstants = 11
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (second)"
    legend_constants[0] = "sigma in component parameters (dm)"
    legend_constants[1] = "CNG_tot in component parameters (mole_per_dm_squared)"
    legend_constants[2] = "CaM_tot in component parameters (mole_per_dm_cubed)"
    legend_constants[3] = "km_CNG_0 in component parameters (per_second)"
    legend_constants[4] = "km_CaM4 in component parameters (per_second)"
    legend_constants[5] = "kp_CaM4 in component parameters (dm_6_per_second_per_mole_squared)"
    legend_constants[6] = "kp_CNG_i in component parameters (dm_3_per_second_per_mole)"
    legend_constants[7] = "km_CNG_i in component parameters (per_second)"
    legend_constants[8] = "i_Ca in component parameters (per_second)"
    legend_constants[9] = "k_Ca in component parameters (mole_per_dm_squared_per_second)"
    legend_constants[10] = "K_Ca in component parameters (mole_per_dm_cubed)"
    legend_algebraic[0] = "kp_act in component parameters (per_second)"
    legend_states[0] = "CNG_o in component dCNG_o_dt (mole_per_dm_squared)"
    legend_algebraic[1] = "CNG_o_normalized in component dCNG_o_dt (dimensionless)"
    legend_states[1] = "CNG_i in component dCNG_i_dt (mole_per_dm_squared)"
    legend_states[2] = "CaM4 in component dCaM4_dt (mole_per_dm_cubed)"
    legend_states[3] = "Ca in component dCa_dt (mole_per_dm_cubed)"
    legend_algebraic[2] = "Ca_normalized in component dCa_dt (dimensionless)"
    legend_algebraic[3] = "CaM4_normalized in component dCaM4_dt (dimensionless)"
    legend_algebraic[4] = "CNG_i_normalized in component dCNG_i_dt (dimensionless)"
    legend_algebraic[5] = "CNG_c in component dCNG_c_dt (mole_per_dm_squared)"
    legend_algebraic[6] = "CaM in component dCaM_dt (mole_per_dm_cubed)"
    legend_rates[0] = "d/dt CNG_o in component dCNG_o_dt (mole_per_dm_squared)"
    legend_rates[3] = "d/dt Ca in component dCa_dt (mole_per_dm_cubed)"
    legend_rates[2] = "d/dt CaM4 in component dCaM4_dt (mole_per_dm_cubed)"
    legend_rates[1] = "d/dt CNG_i in component dCNG_i_dt (mole_per_dm_squared)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 5e-7
    constants[1] = 1.3e-13
    constants[2] = 2e-5
    constants[3] = 1e-2
    constants[4] = 2.5
    constants[5] = 1.1e9
    constants[6] = 2.1e6
    constants[7] = 3.4e-1
    constants[8] = 2e4
    constants[9] = 1e-10
    constants[10] = 1.2e-7
    states[0] = 0
    states[1] = 0
    states[2] = 0
    states[3] = 0
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[3] = ((states[0]/constants[0])*constants[8]-((constants[9]/constants[0])*states[3])/(states[3]+constants[10]))-4.00000*(constants[5]*(power(states[3], 2.00000))*((constants[2]-states[2])-states[1]/constants[0])-constants[4]*states[2])
    rates[2] = ((constants[5]*(power(states[3], 2.00000))*((constants[2]-states[2])-states[1]/constants[0])-constants[4]*states[2])-(constants[6]/constants[0])*states[2]*(constants[1]-states[0]))+(constants[7]/constants[0])*states[1]
    rates[1] = -constants[7]*states[1]+constants[6]*states[2]*(constants[1]-states[1])
    algebraic[0] = custom_piecewise([greater(voi , 0.100000) & less(voi , 0.200000), 5.50000 , greater(voi , 4.10000) & less(voi , 4.20000), 5.50000 , True, 1.60000e-05])
    rates[0] = (algebraic[0]*((constants[1]-states[0])-states[1])-constants[3]*states[0])-constants[6]*states[0]*states[2]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = custom_piecewise([greater(voi , 0.100000) & less(voi , 0.200000), 5.50000 , greater(voi , 4.10000) & less(voi , 4.20000), 5.50000 , True, 1.60000e-05])
    algebraic[1] = states[0]/constants[1]
    algebraic[2] = states[3]*10000.0
    algebraic[3] = states[2]/constants[2]
    algebraic[4] = states[1]/constants[1]
    algebraic[5] = (constants[1]-states[0])-states[1]
    algebraic[6] = (constants[2]-states[2])-(1.00000/constants[0])*states[1]
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