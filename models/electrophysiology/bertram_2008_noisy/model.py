# Size of variable arrays:
sizeAlgebraic = 12
sizeStates = 5
sizeConstants = 32
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (millisecond)"
    legend_states[0] = "V in component membrane (millivolt)"
    legend_constants[0] = "Cm in component membrane (femtofarad)"
    legend_algebraic[5] = "Ica in component Ica (femtoampere)"
    legend_algebraic[7] = "Is1 in component Is1 (femtoampere)"
    legend_algebraic[11] = "Is2 in component Is2 (femtoampere)"
    legend_algebraic[10] = "Il in component Il (femtoampere)"
    legend_algebraic[8] = "Ik in component Ik (femtoampere)"
    legend_constants[1] = "gCa in component Ica (picosiemens)"
    legend_constants[2] = "VCa in component Ica (millivolt)"
    legend_algebraic[0] = "m_infinity in component m (dimensionless)"
    legend_constants[3] = "vm in component m (millivolt)"
    legend_constants[4] = "sm in component m (millivolt)"
    legend_constants[5] = "gs1 in component Is1 (picosiemens)"
    legend_constants[6] = "VK in component Ik (millivolt)"
    legend_states[1] = "s1 in component s1 (dimensionless)"
    legend_algebraic[1] = "s1_infinity in component s1 (dimensionless)"
    legend_constants[7] = "autos1 in component s1 (dimensionless)"
    legend_constants[8] = "s1knot in component s1 (dimensionless)"
    legend_constants[9] = "tau_s1 in component s1 (millisecond)"
    legend_constants[10] = "vs1 in component s1 (millivolt)"
    legend_constants[11] = "ss1 in component s1 (millivolt)"
    legend_constants[12] = "gK in component Ik (picosiemens)"
    legend_states[2] = "n in component n (dimensionless)"
    legend_algebraic[2] = "n_infinity in component n (dimensionless)"
    legend_constants[13] = "tau_n_bar in component n (millisecond)"
    legend_algebraic[6] = "tau_n in component n (millisecond)"
    legend_constants[14] = "vn in component n (millivolt)"
    legend_constants[15] = "sn in component n (millivolt)"
    legend_constants[16] = "gl in component Il (picosiemens)"
    legend_constants[17] = "Vl in component Il (millivolt)"
    legend_algebraic[9] = "q in component Il (dimensionless)"
    legend_states[3] = "p in component Il (dimensionless)"
    legend_constants[18] = "alpha_p in component Il (dimensionless)"
    legend_constants[19] = "tau_p in component Il (millisecond)"
    legend_constants[30] = "beta_p in component Il (dimensionless)"
    legend_constants[20] = "p0 in component Il (dimensionless)"
    legend_constants[21] = "noise in component Il (dimensionless)"
    legend_algebraic[3] = "sigma in component Il (dimensionless)"
    legend_constants[22] = "w in component Il (dimensionless)"
    legend_constants[31] = "nstoc in component Il (dimensionless)"
    legend_constants[23] = "delNoise in component Il (dimensionless)"
    legend_constants[24] = "gs2 in component Is2 (picosiemens)"
    legend_states[4] = "s2 in component s2 (dimensionless)"
    legend_algebraic[4] = "s2_infinity in component s2 (dimensionless)"
    legend_constants[25] = "autos2 in component s2 (dimensionless)"
    legend_constants[26] = "s2knot in component s2 (dimensionless)"
    legend_constants[27] = "tau_s2 in component s2 (millisecond)"
    legend_constants[28] = "vs2 in component s2 (millivolt)"
    legend_constants[29] = "ss2 in component s2 (millivolt)"
    legend_rates[0] = "d/dt V in component membrane (millivolt)"
    legend_rates[1] = "d/dt s1 in component s1 (dimensionless)"
    legend_rates[2] = "d/dt n in component n (dimensionless)"
    legend_rates[3] = "d/dt p in component Il (dimensionless)"
    legend_rates[4] = "d/dt s2 in component s2 (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = -40.0
    constants[0] = 4525.0
    constants[1] = 280.0
    constants[2] = 100.0
    constants[3] = -22.0
    constants[4] = 7.5
    constants[5] = 22.0
    constants[6] = -80.0
    states[1] = 0.9
    constants[7] = 1
    constants[8] = 1
    constants[9] = 1000.0
    constants[10] = -50.0
    constants[11] = 5
    constants[12] = 1300.0
    states[2] = 0.0
    constants[13] = 8.25
    constants[14] = -9.0
    constants[15] = 10.0
    constants[16] = 41.0
    constants[17] = -40.0
    states[3] = 0.14
    constants[18] = 1.0
    constants[19] = 100.0
    constants[20] = 0.2
    constants[21] = 1
    constants[22] = 1
    constants[23] = 3
    constants[24] = 16
    states[4] = 0.5
    constants[25] = 1
    constants[26] = 0.47
    constants[27] = 30000.0
    constants[28] = -40.0
    constants[29] = 15
    constants[30] = constants[18]*(1.00000/constants[20]-1.00000)
    constants[31] = 1000.00/(power(constants[23], 2.00000))
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[1] = 1.00000/(1.00000+exp((constants[10]-states[0])/constants[11]))
    rates[1] = constants[7]*((algebraic[1]-states[1])/constants[9])+(1.00000-constants[7])*(constants[8]-states[1])
    algebraic[3] = power((constants[18]*(1.00000-states[3])+constants[30]*states[3])/(constants[19]*constants[31]), 1.0/2)
    rates[3] = (constants[18]*(1.00000-states[3])-constants[30]*states[3])/constants[19]+constants[21]*constants[22]*algebraic[3]
    algebraic[4] = 1.00000/(1.00000+exp((constants[28]-states[0])/constants[29]))
    rates[4] = constants[25]*((algebraic[4]-states[4])/constants[27])+(1.00000-constants[25])*(constants[26]-states[4])
    algebraic[2] = 1.00000/(1.00000+exp((constants[14]-states[0])/constants[15]))
    algebraic[6] = constants[13]/(1.00000+exp((states[0]-constants[14])/constants[15]))
    rates[2] = (algebraic[2]-states[2])/algebraic[6]
    algebraic[0] = 1.00000/(1.00000+exp((constants[3]-states[0])/constants[4]))
    algebraic[5] = constants[1]*algebraic[0]*(states[0]-constants[2])
    algebraic[7] = constants[5]*states[1]*(states[0]-constants[6])
    algebraic[11] = constants[24]*states[4]*(states[0]-constants[6])
    algebraic[9] = (1.00000+states[3])/2.00000
    algebraic[10] = constants[16]*algebraic[9]*(states[0]-constants[17])
    algebraic[8] = constants[12]*states[2]*(states[0]-constants[6])
    rates[0] = -(algebraic[5]+algebraic[7]+algebraic[11]+algebraic[10]+algebraic[8])/constants[0]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[1] = 1.00000/(1.00000+exp((constants[10]-states[0])/constants[11]))
    algebraic[3] = power((constants[18]*(1.00000-states[3])+constants[30]*states[3])/(constants[19]*constants[31]), 1.0/2)
    algebraic[4] = 1.00000/(1.00000+exp((constants[28]-states[0])/constants[29]))
    algebraic[2] = 1.00000/(1.00000+exp((constants[14]-states[0])/constants[15]))
    algebraic[6] = constants[13]/(1.00000+exp((states[0]-constants[14])/constants[15]))
    algebraic[0] = 1.00000/(1.00000+exp((constants[3]-states[0])/constants[4]))
    algebraic[5] = constants[1]*algebraic[0]*(states[0]-constants[2])
    algebraic[7] = constants[5]*states[1]*(states[0]-constants[6])
    algebraic[11] = constants[24]*states[4]*(states[0]-constants[6])
    algebraic[9] = (1.00000+states[3])/2.00000
    algebraic[10] = constants[16]*algebraic[9]*(states[0]-constants[17])
    algebraic[8] = constants[12]*states[2]*(states[0]-constants[6])
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