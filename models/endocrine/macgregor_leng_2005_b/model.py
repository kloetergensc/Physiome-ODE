# Size of variable arrays:
sizeAlgebraic = 10
sizeStates = 7
sizeConstants = 28
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_states[0] = "r in component r (nanomolar)"
    legend_algebraic[0] = "ract in component r (dimensionless)"
    legend_algebraic[9] = "hr in component r (dimensionless)"
    legend_algebraic[4] = "Ir in component r (flux)"
    legend_algebraic[6] = "Ir2 in component r (flux)"
    legend_constants[0] = "k6 in component r (first_order_rate_constant)"
    legend_constants[1] = "n1 in component r (dimensionless)"
    legend_constants[2] = "theta_1 in component r (nanomolar)"
    legend_constants[3] = "n3 in component r (dimensionless)"
    legend_constants[4] = "theta_3 in component r (dimensionless)"
    legend_states[1] = "s in component s (nanomolar)"
    legend_algebraic[8] = "h_delta1 in component model_parameters (dimensionless)"
    legend_constants[5] = "j1 in component model_parameters (dimensionless)"
    legend_constants[6] = "Is2 in component s (flux)"
    legend_algebraic[7] = "Is in component model_parameters (flux)"
    legend_constants[7] = "k7 in component model_parameters (first_order_rate_constant)"
    legend_states[2] = "g in component g (dimensionless)"
    legend_states[3] = "se in component se (nanomolar)"
    legend_constants[8] = "Ise in component se (flux)"
    legend_states[4] = "f in component f (dimensionless)"
    legend_constants[9] = "k1 in component f (second_order_rate_constant)"
    legend_constants[10] = "k2 in component f (first_order_rate_constant)"
    legend_constants[11] = "k3 in component f (first_order_rate_constant)"
    legend_algebraic[2] = "phi_b_s in component f (dimensionless)"
    legend_constants[12] = "sb in component f (dimensionless)"
    legend_constants[13] = "delta_b in component f (dimensionless)"
    legend_constants[14] = "c in component model_parameters (nanomolar)"
    legend_states[5] = "h in component h (nanomolar)"
    legend_constants[15] = "k4 in component h (first_order_rate_constant)"
    legend_constants[16] = "k5 in component h (first_order_rate_constant)"
    legend_algebraic[3] = "phi_r_s in component h (dimensionless)"
    legend_constants[17] = "sr in component h (dimensionless)"
    legend_constants[18] = "delta_r in component h (dimensionless)"
    legend_constants[19] = "k8 in component model_parameters (first_order_rate_constant)"
    legend_constants[20] = "g1 in component g (first_order_rate_constant)"
    legend_constants[21] = "gmax in component g (dimensionless)"
    legend_constants[22] = "g2 in component g (per_nanomolar)"
    legend_algebraic[5] = "hact in component g (dimensionless)"
    legend_constants[23] = "n2 in component g (dimensionless)"
    legend_constants[24] = "theta_2 in component g (dimensionless)"
    legend_algebraic[1] = "h_delta in component model_parameters (dimensionless)"
    legend_states[6] = "hh in component hh (nanomolar)"
    legend_constants[25] = "Ih in component hh (flux)"
    legend_constants[26] = "delta in component model_parameters (per_nanomolar)"
    legend_constants[27] = "delta1 in component model_parameters (per_nanomolar)"
    legend_rates[0] = "d/dt r in component r (nanomolar)"
    legend_rates[1] = "d/dt s in component s (nanomolar)"
    legend_rates[3] = "d/dt se in component se (nanomolar)"
    legend_rates[4] = "d/dt f in component f (dimensionless)"
    legend_rates[5] = "d/dt h in component h (nanomolar)"
    legend_rates[2] = "d/dt g in component g (dimensionless)"
    legend_rates[6] = "d/dt hh in component hh (nanomolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 0.0
    constants[0] = 5.0
    constants[1] = 4.0
    constants[2] = 1.0
    constants[3] = 5.0
    constants[4] = 30.0
    states[1] = 0.0
    constants[5] = 10
    constants[6] = 50.0
    constants[7] = 5.0
    states[2] = 2.0
    states[3] = 0.0
    constants[8] = 10.0
    states[4] = 0.3
    constants[9] = 0.1
    constants[10] = 0.002
    constants[11] = 0.018
    constants[12] = 0.029
    constants[13] = 0.3
    constants[14] = 0.01
    states[5] = 0.0
    constants[15] = 0.5
    constants[16] = 71.0
    constants[17] = -0.56
    constants[18] = 0.2
    constants[19] = 0.07
    constants[20] = 1.0
    constants[21] = 5.0
    constants[22] = 0.008
    constants[23] = 2.0
    constants[24] = 30.0
    states[6] = 0.0
    constants[25] = 50.0
    constants[26] = 60.0
    constants[27] = 15.0
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[3] = constants[8]-constants[7]*states[3]
    rates[6] = constants[5]*(constants[25]-constants[19]*states[6])
    algebraic[2] = 1.00000/(1.00000+exp(-(log(1.00000*(states[1]+states[3]), 10)-constants[12])/constants[13]))
    rates[4] = -(constants[9]*(states[0]+constants[14])*states[4])+(constants[10]+constants[11]*algebraic[2])*(1.00000-states[4])
    algebraic[3] = 1.00000/(1.00000+exp(-(log(1.00000*(states[1]+states[3]), 10)-constants[17])/constants[18]))
    rates[5] = constants[5]*((constants[15]+constants[16]*(1.00000-algebraic[3]))*((states[0]+constants[14])*states[4])-constants[19]*states[5])
    algebraic[7] = custom_piecewise([greater(voi , 0.00000) & less_equal(voi , 90.0000), 10.0000 , greater(voi , 90.0000) & less_equal(voi , 180.000), 0.00000 , greater(voi , 180.000) & less_equal(voi , 270.000), 10.0000 , greater(voi , 270.000) & less_equal(voi , 360.000), 0.00000 , True, 0.00000])
    rates[1] = (algebraic[7]*states[2]-constants[7]*states[1])+constants[6]
    algebraic[1] = (states[5]+states[6])*constants[26]
    algebraic[5] = (power(algebraic[1], constants[23]))/(power(algebraic[1], constants[23])+power(constants[5]*constants[24], constants[23]))
    rates[2] = constants[20]*algebraic[5]*((constants[21]-states[2])/constants[21])-constants[22]*algebraic[7]*states[2]
    algebraic[0] = 1.00000-(power(states[1], constants[1]))/(power(states[1], constants[1])+power(constants[2], constants[1]))
    algebraic[8] = (states[5]+states[6])*constants[27]
    algebraic[9] = 1.00000-(power(algebraic[8], constants[3]))/(power(algebraic[8], constants[3])+power(constants[5]*constants[4], constants[3]))
    algebraic[4] = custom_piecewise([greater_equal(voi , 0.00000) & less_equal(voi , 90.0000), 0.00000 , greater_equal(voi , 91.0000) & less_equal(voi , 92.0000), 10.0000 , greater_equal(voi , 93.0000) & less_equal(voi , 113.000), 0.00000 , greater_equal(voi , 114.000) & less_equal(voi , 115.000), 10.0000 , greater_equal(voi , 116.000) & less_equal(voi , 136.000), 0.00000 , greater_equal(voi , 137.000) & less_equal(voi , 138.000), 10.0000 , greater_equal(voi , 139.000) & less_equal(voi , 159.000), 0.00000 , greater_equal(voi , 160.000) & less_equal(voi , 161.000), 10.0000 , greater_equal(voi , 162.000) & less_equal(voi , 252.000), 0.00000 , greater_equal(voi , 253.000) & less_equal(voi , 254.000), 10.0000 , greater_equal(voi , 255.000) & less_equal(voi , 275.000), 0.00000 , greater_equal(voi , 276.000) & less_equal(voi , 277.000), 10.0000 , greater_equal(voi , 278.000) & less_equal(voi , 298.000), 0.00000 , greater_equal(voi , 299.000) & less_equal(voi , 300.000), 10.0000 , greater_equal(voi , 301.000) & less_equal(voi , 321.000), 0.00000 , greater_equal(voi , 322.000) & less_equal(voi , 323.000), 10.0000 , True, 0.00000])
    algebraic[6] = custom_piecewise([greater_equal(voi , 0.00000) & less_equal(voi , 5.00000), 0.00000 , greater_equal(voi , 6.00000) & less_equal(voi , 7.00000), 1000.00 , greater_equal(voi , 8.00000) & less_equal(voi , 12.0000), 0.00000 , greater_equal(voi , 13.0000) & less_equal(voi , 14.0000), 1000.00 , greater_equal(voi , 15.0000) & less_equal(voi , 21.0000), 0.00000 , greater_equal(voi , 22.0000) & less_equal(voi , 23.0000), 1000.00 , greater_equal(voi , 24.0000) & less_equal(voi , 204.000), 0.00000 , greater_equal(voi , 205.000) & less_equal(voi , 206.000), 1000.00 , greater_equal(voi , 207.000) & less_equal(voi , 217.000), 0.00000 , greater_equal(voi , 218.000) & less_equal(voi , 219.000), 1000.00 , greater_equal(voi , 220.000) & less_equal(voi , 227.000), 0.00000 , greater_equal(voi , 228.000) & less_equal(voi , 229.000), 1000.00 , greater_equal(voi , 230.000) & less_equal(voi , 310.000), 0.00000 , greater_equal(voi , 311.000) & less_equal(voi , 312.000), 1000.00 , greater_equal(voi , 313.000) & less_equal(voi , 321.000), 0.00000 , greater_equal(voi , 322.000) & less_equal(voi , 323.000), 1000.00 , True, 0.00000])
    rates[0] = (algebraic[0]*algebraic[9]*algebraic[4]-constants[0]*states[0])+algebraic[6]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[2] = 1.00000/(1.00000+exp(-(log(1.00000*(states[1]+states[3]), 10)-constants[12])/constants[13]))
    algebraic[3] = 1.00000/(1.00000+exp(-(log(1.00000*(states[1]+states[3]), 10)-constants[17])/constants[18]))
    algebraic[7] = custom_piecewise([greater(voi , 0.00000) & less_equal(voi , 90.0000), 10.0000 , greater(voi , 90.0000) & less_equal(voi , 180.000), 0.00000 , greater(voi , 180.000) & less_equal(voi , 270.000), 10.0000 , greater(voi , 270.000) & less_equal(voi , 360.000), 0.00000 , True, 0.00000])
    algebraic[1] = (states[5]+states[6])*constants[26]
    algebraic[5] = (power(algebraic[1], constants[23]))/(power(algebraic[1], constants[23])+power(constants[5]*constants[24], constants[23]))
    algebraic[0] = 1.00000-(power(states[1], constants[1]))/(power(states[1], constants[1])+power(constants[2], constants[1]))
    algebraic[8] = (states[5]+states[6])*constants[27]
    algebraic[9] = 1.00000-(power(algebraic[8], constants[3]))/(power(algebraic[8], constants[3])+power(constants[5]*constants[4], constants[3]))
    algebraic[4] = custom_piecewise([greater_equal(voi , 0.00000) & less_equal(voi , 90.0000), 0.00000 , greater_equal(voi , 91.0000) & less_equal(voi , 92.0000), 10.0000 , greater_equal(voi , 93.0000) & less_equal(voi , 113.000), 0.00000 , greater_equal(voi , 114.000) & less_equal(voi , 115.000), 10.0000 , greater_equal(voi , 116.000) & less_equal(voi , 136.000), 0.00000 , greater_equal(voi , 137.000) & less_equal(voi , 138.000), 10.0000 , greater_equal(voi , 139.000) & less_equal(voi , 159.000), 0.00000 , greater_equal(voi , 160.000) & less_equal(voi , 161.000), 10.0000 , greater_equal(voi , 162.000) & less_equal(voi , 252.000), 0.00000 , greater_equal(voi , 253.000) & less_equal(voi , 254.000), 10.0000 , greater_equal(voi , 255.000) & less_equal(voi , 275.000), 0.00000 , greater_equal(voi , 276.000) & less_equal(voi , 277.000), 10.0000 , greater_equal(voi , 278.000) & less_equal(voi , 298.000), 0.00000 , greater_equal(voi , 299.000) & less_equal(voi , 300.000), 10.0000 , greater_equal(voi , 301.000) & less_equal(voi , 321.000), 0.00000 , greater_equal(voi , 322.000) & less_equal(voi , 323.000), 10.0000 , True, 0.00000])
    algebraic[6] = custom_piecewise([greater_equal(voi , 0.00000) & less_equal(voi , 5.00000), 0.00000 , greater_equal(voi , 6.00000) & less_equal(voi , 7.00000), 1000.00 , greater_equal(voi , 8.00000) & less_equal(voi , 12.0000), 0.00000 , greater_equal(voi , 13.0000) & less_equal(voi , 14.0000), 1000.00 , greater_equal(voi , 15.0000) & less_equal(voi , 21.0000), 0.00000 , greater_equal(voi , 22.0000) & less_equal(voi , 23.0000), 1000.00 , greater_equal(voi , 24.0000) & less_equal(voi , 204.000), 0.00000 , greater_equal(voi , 205.000) & less_equal(voi , 206.000), 1000.00 , greater_equal(voi , 207.000) & less_equal(voi , 217.000), 0.00000 , greater_equal(voi , 218.000) & less_equal(voi , 219.000), 1000.00 , greater_equal(voi , 220.000) & less_equal(voi , 227.000), 0.00000 , greater_equal(voi , 228.000) & less_equal(voi , 229.000), 1000.00 , greater_equal(voi , 230.000) & less_equal(voi , 310.000), 0.00000 , greater_equal(voi , 311.000) & less_equal(voi , 312.000), 1000.00 , greater_equal(voi , 313.000) & less_equal(voi , 321.000), 0.00000 , greater_equal(voi , 322.000) & less_equal(voi , 323.000), 1000.00 , True, 0.00000])
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