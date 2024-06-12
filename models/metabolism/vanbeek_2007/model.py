# Size of variable arrays:
sizeAlgebraic = 15
sizeStates = 10
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
    legend_states[0] = "ATP_cyt in component ATP_cyt (micromolar)"
    legend_algebraic[9] = "J_hyd in component J_hyd (flux)"
    legend_algebraic[1] = "J_CKMM in component J_CKMM (flux)"
    legend_algebraic[12] = "J_diff_ATP in component J_diff_ATP (flux)"
    legend_constants[0] = "V_cyt in component fractional_volumes (dimensionless)"
    legend_states[1] = "ADP_cyt in component ADP_cyt (micromolar)"
    legend_algebraic[13] = "J_diff_ADP in component J_diff_ADP (flux)"
    legend_states[2] = "PCr_cyt in component PCr_cyt (micromolar)"
    legend_algebraic[5] = "J_diff_PCr in component J_diff_PCr (flux)"
    legend_states[3] = "Cr_cyt in component Cr_cyt (micromolar)"
    legend_algebraic[6] = "J_diff_Cr in component J_diff_Cr (flux)"
    legend_states[4] = "Pi_cyt in component Pi_cyt (micromolar)"
    legend_algebraic[14] = "J_diff_Pi in component J_diff_Pi (flux)"
    legend_states[5] = "ATP_ims in component ATP_ims (micromolar)"
    legend_algebraic[3] = "J_CKMi in component J_CKMi (flux)"
    legend_algebraic[11] = "J_syn in component J_syn (flux)"
    legend_constants[1] = "V_ims in component fractional_volumes (dimensionless)"
    legend_states[6] = "ADP_ims in component ADP_ims (micromolar)"
    legend_states[7] = "PCr_ims in component PCr_ims (micromolar)"
    legend_states[8] = "Cr_ims in component Cr_ims (micromolar)"
    legend_states[9] = "Pi_ims in component Pi_ims (micromolar)"
    legend_algebraic[0] = "Den_MMCK in component J_CKMM (dimensionless)"
    legend_constants[2] = "Kia in component J_CKMM (micromolar)"
    legend_constants[3] = "Kb in component J_CKMM (micromolar)"
    legend_constants[4] = "Kib in component J_CKMM (micromolar)"
    legend_constants[31] = "KIb in component J_CKMM (micromolar)"
    legend_constants[32] = "Kc in component J_CKMM (micromolar)"
    legend_constants[5] = "Kic in component J_CKMM (micromolar)"
    legend_constants[6] = "Kd in component J_CKMM (micromolar)"
    legend_constants[7] = "Kid in component J_CKMM (micromolar)"
    legend_constants[8] = "Vmax_MM_f in component J_CKMM (flux)"
    legend_constants[9] = "Vmax_MM_b in component J_CKMM (flux)"
    legend_algebraic[2] = "Den_MiCK in component J_CKMi (dimensionless)"
    legend_constants[10] = "Kia in component J_CKMi (micromolar)"
    legend_constants[11] = "Kb in component J_CKMi (micromolar)"
    legend_constants[12] = "Kib in component J_CKMi (micromolar)"
    legend_constants[33] = "KIb in component J_CKMi (micromolar)"
    legend_constants[34] = "Kc in component J_CKMi (micromolar)"
    legend_constants[13] = "Kic in component J_CKMi (micromolar)"
    legend_constants[14] = "Kd in component J_CKMi (micromolar)"
    legend_constants[15] = "Kid in component J_CKMi (micromolar)"
    legend_constants[16] = "Vmax_Mi_f in component J_CKMi (flux)"
    legend_constants[17] = "Vmax_Mi_b in component J_CKMi (flux)"
    legend_algebraic[8] = "H_ATPmax in component J_hyd (flux)"
    legend_constants[18] = "J_hyd_basis_1 in component J_hyd (flux)"
    legend_constants[19] = "J_hyd_basis_2 in component J_hyd (flux)"
    legend_constants[20] = "freq_1 in component J_hyd (dimensionless)"
    legend_constants[21] = "freq_2 in component J_hyd (dimensionless)"
    legend_constants[35] = "t_cycle_1 in component J_hyd (second)"
    legend_constants[36] = "t_cycle_2 in component J_hyd (second)"
    legend_algebraic[7] = "t_cycle in component J_hyd (second)"
    legend_constants[22] = "nb_of_cycles_1 in component J_hyd (dimensionless)"
    legend_constants[37] = "duration_1 in component J_hyd (second)"
    legend_algebraic[4] = "ltime in component J_hyd (second)"
    legend_algebraic[10] = "Den_syn in component J_syn (dimensionless)"
    legend_constants[23] = "KPi in component J_syn (micromolar)"
    legend_constants[24] = "KADP in component J_syn (micromolar)"
    legend_constants[25] = "V_max_syn in component J_syn (flux)"
    legend_constants[26] = "PS_tot_ATP in component J_diff_ATP (first_order_rate_constant)"
    legend_constants[27] = "PS_tot_ADP in component J_diff_ADP (first_order_rate_constant)"
    legend_constants[28] = "PS_tot_PCr in component J_diff_PCr (first_order_rate_constant)"
    legend_constants[29] = "PS_tot_Cr in component J_diff_Cr (first_order_rate_constant)"
    legend_constants[30] = "PS_tot_Pi in component J_diff_Pi (first_order_rate_constant)"
    legend_rates[0] = "d/dt ATP_cyt in component ATP_cyt (micromolar)"
    legend_rates[1] = "d/dt ADP_cyt in component ADP_cyt (micromolar)"
    legend_rates[2] = "d/dt PCr_cyt in component PCr_cyt (micromolar)"
    legend_rates[3] = "d/dt Cr_cyt in component Cr_cyt (micromolar)"
    legend_rates[4] = "d/dt Pi_cyt in component Pi_cyt (micromolar)"
    legend_rates[5] = "d/dt ATP_ims in component ATP_ims (micromolar)"
    legend_rates[6] = "d/dt ADP_ims in component ADP_ims (micromolar)"
    legend_rates[7] = "d/dt PCr_ims in component PCr_ims (micromolar)"
    legend_rates[8] = "d/dt Cr_ims in component Cr_ims (micromolar)"
    legend_rates[9] = "d/dt Pi_ims in component Pi_ims (micromolar)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = 5912.77
    constants[0] = 0.75
    states[1] = 64
    states[2] = 5000
    states[3] = 10500
    states[4] = 913
    states[5] = 5912.77
    constants[1] = 0.0625
    states[6] = 39
    states[7] = 5000
    states[8] = 10500
    states[9] = 910
    constants[2] = 9.0E2
    constants[3] = 1.55E4
    constants[4] = 3.49E4
    constants[5] = 2.224E2
    constants[6] = 1.67E3
    constants[7] = 4.73E3
    constants[8] = 1.144E4
    constants[9] = 4.804E4
    constants[10] = 7.5E2
    constants[11] = 5.2E3
    constants[12] = 2.88E4
    constants[13] = 2.048E2
    constants[14] = 5.0E2
    constants[15] = 1.6E3
    constants[16] = 8.82E2
    constants[17] = 3.704E3
    constants[18] = 4.865e2
    constants[19] = 6.276e2
    constants[20] = 135
    constants[21] = 220
    constants[22] = 5
    constants[23] = 8E2
    constants[24] = 25
    constants[25] = 1.504E4
    constants[26] = 13.3
    constants[27] = 13.3
    constants[28] = 155.0
    constants[29] = 155.0
    constants[30] = 194.0
    constants[31] = constants[4]
    constants[32] = (constants[5]*constants[6])/constants[7]
    constants[33] = constants[12]
    constants[34] = (constants[13]*constants[14])/constants[15]
    constants[35] = 60.0000/constants[20]
    constants[36] = 60.0000/constants[21]
    constants[37] = constants[22]*constants[35]
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[0] = 1.00000+states[3]/constants[4]+states[2]/constants[7]+states[0]*(1.00000/constants[2]+states[3]/(constants[2]*constants[3]))+states[1]*(1.00000/constants[5]+states[2]/(constants[7]*constants[32])+states[3]/(constants[5]*constants[31]))
    algebraic[1] = (constants[8]*((states[0]*states[3])/(constants[2]*constants[3]))-constants[9]*((states[1]*states[2])/(constants[5]*constants[6])))/algebraic[0]
    algebraic[5] = constants[28]*(states[7]-states[2])
    rates[2] = (algebraic[5]+algebraic[1])/constants[0]
    algebraic[6] = constants[29]*(states[8]-states[3])
    rates[3] = (algebraic[6]-algebraic[1])/constants[0]
    algebraic[2] = 1.00000+states[8]/constants[12]+states[7]/constants[15]+states[5]*(1.00000/constants[10]+states[8]/(constants[10]*constants[11]))+states[6]*(1.00000/constants[13]+states[7]/(constants[15]*constants[34])+states[8]/(constants[13]*constants[33]))
    algebraic[3] = (constants[16]*((states[5]*states[8])/(constants[10]*constants[11]))-constants[17]*((states[6]*states[7])/(constants[13]*constants[14])))/algebraic[2]
    rates[7] = (algebraic[3]-algebraic[5])/constants[1]
    rates[8] = -(algebraic[3]+algebraic[6])/constants[1]
    algebraic[8] = custom_piecewise([less_equal(voi , constants[37]), 6.00000*constants[18] , True, 6.00000*constants[19]])
    algebraic[7] = custom_piecewise([less_equal(voi , constants[37]), constants[35] , True, constants[36]])
    algebraic[4] = custom_piecewise([less_equal(voi , constants[37]), voi-constants[35]*floor(voi/constants[35]) , True, (voi-constants[37])-constants[36]*floor((voi-constants[37])/constants[36])])
    algebraic[9] = custom_piecewise([greater_equal(algebraic[4] , 0.00000) & less(algebraic[4] , (1.00000/6.00000)*algebraic[7]), ((algebraic[8]*algebraic[4])/algebraic[7])*6.00000 , greater_equal(algebraic[4] , (1.00000/6.00000)*algebraic[7]) & less(algebraic[4] , (1.00000/3.00000)*algebraic[7]), algebraic[8]*(1.00000-6.00000*(algebraic[4]/algebraic[7]-1.00000/6.00000)) , greater_equal(algebraic[4] , (1.00000/3.00000)*algebraic[7]) & less(algebraic[4] , algebraic[7]), 0.00000 , True, float('nan')])
    algebraic[12] = constants[26]*(states[5]-states[0])
    rates[0] = (algebraic[12]-(algebraic[9]+algebraic[1]))/constants[0]
    algebraic[13] = constants[27]*(states[6]-states[1])
    rates[1] = (algebraic[13]+algebraic[9]+algebraic[1])/constants[0]
    algebraic[14] = constants[30]*(states[9]-states[4])
    rates[4] = (algebraic[14]+algebraic[9])/constants[0]
    algebraic[10] = 1.00000+states[6]/constants[24]+states[9]/constants[23]+(states[6]*states[9])/(constants[24]*constants[23])
    algebraic[11] = constants[25]*((states[6]*states[9])/(constants[23]*constants[24]*algebraic[10]))
    rates[5] = (algebraic[11]-(algebraic[12]+algebraic[3]))/constants[1]
    rates[6] = (algebraic[3]-(algebraic[11]+algebraic[13]))/constants[1]
    rates[9] = -(algebraic[11]+algebraic[14])/constants[1]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[0] = 1.00000+states[3]/constants[4]+states[2]/constants[7]+states[0]*(1.00000/constants[2]+states[3]/(constants[2]*constants[3]))+states[1]*(1.00000/constants[5]+states[2]/(constants[7]*constants[32])+states[3]/(constants[5]*constants[31]))
    algebraic[1] = (constants[8]*((states[0]*states[3])/(constants[2]*constants[3]))-constants[9]*((states[1]*states[2])/(constants[5]*constants[6])))/algebraic[0]
    algebraic[5] = constants[28]*(states[7]-states[2])
    algebraic[6] = constants[29]*(states[8]-states[3])
    algebraic[2] = 1.00000+states[8]/constants[12]+states[7]/constants[15]+states[5]*(1.00000/constants[10]+states[8]/(constants[10]*constants[11]))+states[6]*(1.00000/constants[13]+states[7]/(constants[15]*constants[34])+states[8]/(constants[13]*constants[33]))
    algebraic[3] = (constants[16]*((states[5]*states[8])/(constants[10]*constants[11]))-constants[17]*((states[6]*states[7])/(constants[13]*constants[14])))/algebraic[2]
    algebraic[8] = custom_piecewise([less_equal(voi , constants[37]), 6.00000*constants[18] , True, 6.00000*constants[19]])
    algebraic[7] = custom_piecewise([less_equal(voi , constants[37]), constants[35] , True, constants[36]])
    algebraic[4] = custom_piecewise([less_equal(voi , constants[37]), voi-constants[35]*floor(voi/constants[35]) , True, (voi-constants[37])-constants[36]*floor((voi-constants[37])/constants[36])])
    algebraic[9] = custom_piecewise([greater_equal(algebraic[4] , 0.00000) & less(algebraic[4] , (1.00000/6.00000)*algebraic[7]), ((algebraic[8]*algebraic[4])/algebraic[7])*6.00000 , greater_equal(algebraic[4] , (1.00000/6.00000)*algebraic[7]) & less(algebraic[4] , (1.00000/3.00000)*algebraic[7]), algebraic[8]*(1.00000-6.00000*(algebraic[4]/algebraic[7]-1.00000/6.00000)) , greater_equal(algebraic[4] , (1.00000/3.00000)*algebraic[7]) & less(algebraic[4] , algebraic[7]), 0.00000 , True, float('nan')])
    algebraic[12] = constants[26]*(states[5]-states[0])
    algebraic[13] = constants[27]*(states[6]-states[1])
    algebraic[14] = constants[30]*(states[9]-states[4])
    algebraic[10] = 1.00000+states[6]/constants[24]+states[9]/constants[23]+(states[6]*states[9])/(constants[24]*constants[23])
    algebraic[11] = constants[25]*((states[6]*states[9])/(constants[23]*constants[24]*algebraic[10]))
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