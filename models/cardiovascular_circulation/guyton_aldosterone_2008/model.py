# Size of variable arrays:
sizeAlgebraic = 6
sizeStates = 1
sizeConstants = 21
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_constants[0] = "ANM in component aldosterone (dimensionless)"
    legend_constants[1] = "CKE in component aldosterone (monovalent_mEq_per_litre)"
    legend_constants[15] = "ANMAL in component angiotensin_control_of_aldosterone_secretion (dimensionless)"
    legend_constants[2] = "ANMALD in component parameter_values (dimensionless)"
    legend_constants[16] = "OSMAL in component osmotic_control_of_aldosterone_secretion (dimensionless)"
    legend_constants[20] = "AMR1 in component aldosterone_secretion (dimensionless)"
    legend_constants[3] = "AMKMUL in component parameter_values (dimensionless)"
    legend_constants[4] = "ALDINF in component parameter_values (dimensionless)"
    legend_constants[5] = "ALDKNS in component parameter_values (dimensionless)"
    legend_constants[17] = "AMRBSC in component aldosterone_secretion (dimensionless)"
    legend_constants[18] = "AMRT in component aldosterone_secretion (dimensionless)"
    legend_constants[19] = "AMR in component aldosterone_secretion (dimensionless)"
    legend_states[0] = "AMC in component aldosterone_concentration (dimensionless)"
    legend_constants[6] = "AMT in component parameter_values (minute)"
    legend_algebraic[1] = "AM in component general_aldosterone_multiplier (dimensionless)"
    legend_constants[7] = "AM1UL in component parameter_values (dimensionless)"
    legend_constants[8] = "AM1LL in component parameter_values (dimensionless)"
    legend_constants[9] = "AMCSNS in component parameter_values (dimensionless)"
    legend_constants[10] = "ALDMM in component parameter_values (dimensionless)"
    legend_algebraic[0] = "AM1 in component general_aldosterone_multiplier (dimensionless)"
    legend_algebraic[4] = "AMK in component aldosterone_effect_on_cell_membrane_K_transport (dimensionless)"
    legend_constants[11] = "AMKM in component parameter_values (dimensionless)"
    legend_algebraic[2] = "AMKT in component aldosterone_effect_on_cell_membrane_K_transport (dimensionless)"
    legend_algebraic[5] = "AMNA in component aldosterone_effect_on_cell_membrane_Na_transport (dimensionless)"
    legend_constants[12] = "AMNAM in component parameter_values (dimensionless)"
    legend_constants[13] = "AMNAUL in component parameter_values (dimensionless)"
    legend_constants[14] = "AMNALL in component parameter_values (dimensionless)"
    legend_algebraic[3] = "AMNAT in component aldosterone_effect_on_cell_membrane_Na_transport (dimensionless)"
    legend_rates[0] = "d/dt AMC in component aldosterone_concentration (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 0.987545
    constants[1] = 4.44092
    constants[2] = 2.5
    constants[3] = 12
    constants[4] = 0
    constants[5] = 0
    states[0] = 1.0
    constants[6] = 60
    constants[7] = 5
    constants[8] = 0
    constants[9] = 0.65
    constants[10] = 2.5
    constants[11] = 0.5
    constants[12] = 0.8
    constants[13] = 15
    constants[14] = 0.04
    constants[15] = (constants[0]-1.00000)*constants[2]+1.00000
    constants[16] = (constants[1]-3.30000)/1.00000
    constants[17] = constants[15]*0.909000*constants[16]
    constants[18] = (constants[17]-1.00000)*constants[3]+1.00000
    constants[19] = custom_piecewise([less(constants[18] , 0.00000), 0.00000 , True, constants[18]])
    constants[20] = custom_piecewise([greater(constants[5] , 0.00000), constants[5] , True, constants[19]+constants[4]])
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = (constants[20]-states[0])/constants[6]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = constants[7]-(constants[7]-1.00000)/(((constants[8]-1.00000)/(constants[8]-constants[7]))*(states[0]-1.00000)*constants[9]+1.00000)
    algebraic[1] = (algebraic[0]-1.00000)*constants[10]+1.00000
    algebraic[2] = (algebraic[1]-1.00000)*constants[11]+1.00000
    algebraic[3] = (algebraic[1]-1.00000)*constants[12]+1.00000
    algebraic[4] = custom_piecewise([less(algebraic[2] , 0.200000), 0.200000 , True, algebraic[2]])
    algebraic[5] = custom_piecewise([less(algebraic[3] , constants[14]), constants[14] , greater(algebraic[3] , constants[13]), constants[13] , True, algebraic[3]])
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