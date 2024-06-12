# Size of variable arrays:
sizeAlgebraic = 7
sizeStates = 2
sizeConstants = 17
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_constants[0] = "MDFLW in component angiotensin (L_per_minute)"
    legend_constants[15] = "ANGSCR in component instantaneous_angiotensin_formation (dimensionless)"
    legend_constants[14] = "MDFLW3 in component instantaneous_angiotensin_formation (L_per_minute)"
    legend_states[0] = "ANX1 in component time_delayed_angiotensin_formation (dimensionless)"
    legend_constants[1] = "ANXM in component parameter_values (dimensionless)"
    legend_constants[2] = "ANV in component parameter_values (minute)"
    legend_constants[16] = "ANX in component time_delayed_angiotensin_formation (dimensionless)"
    legend_algebraic[2] = "ANPR in component total_angiotensin_formation (dimensionless)"
    legend_constants[3] = "REK in component parameter_values (dimensionless)"
    legend_algebraic[0] = "ANPRT in component total_angiotensin_formation (dimensionless)"
    legend_algebraic[4] = "ANPR1 in component artificial_angiotensin_formation (dimensionless)"
    legend_constants[4] = "ANGKNS in component parameter_values (dimensionless)"
    legend_constants[5] = "ANGINF in component parameter_values (dimensionless)"
    legend_states[1] = "ANC in component angiotensin_concentration (dimensionless)"
    legend_constants[6] = "ANT in component parameter_values (minute)"
    legend_algebraic[1] = "ANM in component general_angiotensin_multiplier (dimensionless)"
    legend_constants[7] = "ANMUL in component parameter_values (dimensionless)"
    legend_constants[8] = "ANMLL in component parameter_values (dimensionless)"
    legend_constants[9] = "ANCSNS in component parameter_values (dimensionless)"
    legend_algebraic[5] = "ANU in component angiotensin_effect_on_circulation (dimensionless)"
    legend_constants[10] = "ANUM in component parameter_values (dimensionless)"
    legend_constants[11] = "ANULL in component parameter_values (dimensionless)"
    legend_algebraic[3] = "ANU1 in component angiotensin_effect_on_circulation (dimensionless)"
    legend_algebraic[6] = "ANUVN in component angiotensin_effect_on_venous_constriction (dimensionless)"
    legend_constants[12] = "ANUVM in component parameter_values (dimensionless)"
    legend_constants[13] = "Z12 in component parameter_values (dimensionless)"
    legend_rates[0] = "d/dt ANX1 in component time_delayed_angiotensin_formation (dimensionless)"
    legend_rates[1] = "d/dt ANC in component angiotensin_concentration (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 1.00051
    states[0] = 0.0
    constants[1] = 0
    constants[2] = 5000
    constants[3] = 1
    constants[4] = 0
    constants[5] = 0
    states[1] = 0.859476
    constants[6] = 12
    constants[7] = 1.8
    constants[8] = 0.7
    constants[9] = 0.4
    constants[10] = 6
    constants[11] = 0.8
    constants[12] = 0
    constants[13] = 5
    constants[14] = constants[0]
    constants[15] = custom_piecewise([greater(constants[14] , 1.00000), 1.00000/(1.00000+(constants[14]-1.00000)*72.0000) , True, 10.0000-9.00000/(1.00000+(1.00000-constants[14])*8.00000)])
    constants[16] = (constants[15]-1.00000)*constants[1]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[0] = (constants[16]-states[0])/constants[2]
    algebraic[0] = (constants[15]+states[0])*constants[3]
    algebraic[2] = custom_piecewise([less(algebraic[0] , 1.00000e-05), 1.00000e-05 , True, algebraic[0]])
    algebraic[4] = custom_piecewise([greater(constants[4] , 0.00000), constants[4] , True, algebraic[2]+constants[5]])
    rates[1] = (algebraic[4]-states[1])/constants[6]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = (constants[15]+states[0])*constants[3]
    algebraic[2] = custom_piecewise([less(algebraic[0] , 1.00000e-05), 1.00000e-05 , True, algebraic[0]])
    algebraic[4] = custom_piecewise([greater(constants[4] , 0.00000), constants[4] , True, algebraic[2]+constants[5]])
    algebraic[1] = constants[7]-(constants[7]-1.00000)/(((constants[8]-1.00000)/(constants[8]-constants[7]))*(states[1]-1.00000)*constants[9]+1.00000)
    algebraic[3] = (algebraic[1]-1.00000)*constants[10]+1.00000
    algebraic[5] = custom_piecewise([less(algebraic[3] , constants[11]), constants[11] , True, algebraic[3]])
    algebraic[6] = (algebraic[5]-1.00000)*constants[12]+1.00000
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