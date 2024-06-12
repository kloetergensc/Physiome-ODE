# Size of variable arrays:
sizeAlgebraic = 26
sizeStates = 4
sizeConstants = 27
from math import *
from numpy import *

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (minute)"
    legend_constants[0] = "VEC in component capillary_dynamics (litre)"
    legend_constants[1] = "PPD in component capillary_dynamics (gram_per_minute)"
    legend_constants[2] = "RVS in component capillary_dynamics (mmHg_minute_per_L)"
    legend_constants[3] = "DFP in component capillary_dynamics (L_per_minute)"
    legend_constants[4] = "VPF in component capillary_dynamics (litre)"
    legend_constants[5] = "BFN in component capillary_dynamics (L_per_minute)"
    legend_constants[6] = "PVS in component capillary_dynamics (mmHg)"
    legend_constants[24] = "PC in component capillary_pressure (mmHg)"
    legend_algebraic[15] = "PGH in component hydrostatic_pressure_of_tissue_gel (mmHg)"
    legend_algebraic[13] = "PTC in component total_osmotic_pressure_of_tissue_gel (mmHg)"
    legend_algebraic[3] = "PPC in component plasma_colloid_osmotic_pressure (mmHg)"
    legend_constants[26] = "VTCPL in component plasma_leakage (L_per_minute)"
    legend_algebraic[16] = "VTC in component rate_of_fluid_out_of_capillaries (L_per_minute)"
    legend_constants[7] = "CFC in component parameter_values (L_per_minute_per_mmHg)"
    legend_algebraic[21] = "VTL in component lymph_flow (L_per_minute)"
    legend_states[0] = "VP in component plasma_volume (litre)"
    legend_constants[8] = "TRPL in component parameter_values (L_per_minute)"
    legend_algebraic[22] = "VPD in component plasma_volume (L_per_minute)"
    legend_states[1] = "PRP in component total_plasma_protein (gram)"
    legend_algebraic[0] = "CPP in component plasma_protein_concentration (gram_per_L)"
    legend_algebraic[2] = "DLP in component protein_destruction_and_formation (gram_per_minute)"
    legend_constants[9] = "CPR in component parameter_values (gram_per_L)"
    legend_constants[10] = "LPPR in component parameter_values (gram_per_minute)"
    legend_constants[11] = "LPDE in component parameter_values (dimensionless)"
    legend_constants[12] = "LPK in component parameter_values (L_per_minute)"
    legend_algebraic[1] = "CPPD in component protein_destruction_and_formation (gram_per_L)"
    legend_constants[13] = "PCR in component parameter_values (mmHg)"
    legend_constants[14] = "CPK in component parameter_values (L_per_minute_per_mmHg)"
    legend_constants[15] = "PCE in component parameter_values (dimensionless)"
    legend_constants[25] = "PRCD in component plasma_leakage (mmHg)"
    legend_algebraic[7] = "CPI in component interstitial_protein_concentration (gram_per_L)"
    legend_algebraic[9] = "DPC in component protein_influx_into_interstitium (gram_per_minute)"
    legend_algebraic[23] = "DPL in component lymph_protein_flow (gram_per_minute)"
    legend_algebraic[24] = "DPP in component total_plasma_protein (gram_per_minute)"
    legend_algebraic[4] = "VTS in component total_systemic_fluid_volume (litre)"
    legend_algebraic[5] = "VTS1 in component interstitial_fluid_volume (litre)"
    legend_constants[16] = "TSSLML in component parameter_values (dimensionless)"
    legend_constants[17] = "TSSLTC in component parameter_values (per_minute)"
    legend_states[2] = "VTS2 in component interstitial_fluid_volume (litre)"
    legend_states[3] = "TSP in component total_interstitial_protein (gram)"
    legend_algebraic[25] = "DPI in component total_interstitial_protein (gram_per_minute)"
    legend_algebraic[10] = "PTCPR in component interstitial_colloid_osmotic_pressure (mmHg)"
    legend_algebraic[14] = "PTT in component total_tissue_pressure (mmHg)"
    legend_algebraic[11] = "CHY in component hydrostatic_pressure_of_tissue_gel (gram_per_L)"
    legend_constants[18] = "HYL in component parameter_values (gram)"
    legend_constants[19] = "CMPTSS in component parameter_values (dimensionless)"
    legend_constants[20] = "PGHF in component parameter_values (L_mmHg_per_gram)"
    legend_algebraic[12] = "POSHYL in component total_osmotic_pressure_of_tissue_gel (mmHg)"
    legend_constants[21] = "GCOPF in component parameter_values (per_mmHg)"
    legend_constants[22] = "VTSF in component parameter_values (litre)"
    legend_algebraic[17] = "PIF in component interstial_free_fluid_pressure (mmHg)"
    legend_algebraic[19] = "PTS in component interstitial_solid_tissue_pressure (mmHg)"
    legend_constants[23] = "PLDF in component parameter_values (mmHg)"
    legend_algebraic[18] = "PLD1 in component lymph_flow (mmHg)"
    legend_algebraic[20] = "PLD in component lymph_flow (mmHg)"
    legend_algebraic[6] = "VG in component interstitial_gel_volume (litre)"
    legend_algebraic[8] = "VIF in component interstitial_free_fluid_volume (litre)"
    legend_rates[0] = "d/dt VP in component plasma_volume (litre)"
    legend_rates[1] = "d/dt PRP in component total_plasma_protein (gram)"
    legend_rates[2] = "d/dt VTS2 in component interstitial_fluid_volume (litre)"
    legend_rates[3] = "d/dt TSP in component total_interstitial_protein (gram)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 14.8548
    constants[1] = 4.4805e-06
    constants[2] = 2.77751
    constants[3] = -4.078e-07
    constants[4] = 0.0123238
    constants[5] = 2.79521
    constants[6] = 3.71612
    constants[7] = 0.01167
    states[0] = 3.00449
    constants[8] = 0
    states[1] = 216.243
    constants[9] = 40
    constants[10] = 0.03
    constants[11] = 8
    constants[12] = 2.728e-14
    constants[13] = 15
    constants[14] = 0.000253
    constants[15] = 1
    constants[16] = 0.15
    constants[17] = 0.005
    states[2] = 0.0
    states[3] = 279.945
    constants[18] = 60
    constants[19] = 2
    constants[20] = -2
    constants[21] = 0.8092
    constants[22] = 6
    constants[23] = 4.2
    constants[24] = constants[2]*1.70000*constants[5]+constants[6]
    constants[25] = custom_piecewise([less(constants[24]-constants[13] , 0.00000), 0.00000 , True, constants[24]-constants[13]])
    constants[26] = power(constants[25]*constants[14], constants[15])
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[4] = (constants[0]-states[0])-constants[4]
    rates[2] = ((algebraic[4]-12.0000)*constants[16]-states[2])*constants[17]
    algebraic[5] = algebraic[4]-states[2]
    algebraic[14] = (power((algebraic[5]-constants[22])/constants[22], 2.00000))*1.00000
    algebraic[11] = power((constants[18]/algebraic[4])/5.00000, constants[19])
    algebraic[15] = algebraic[11]*constants[20]+algebraic[14]
    algebraic[7] = states[3]/algebraic[4]
    algebraic[10] = 0.280000*algebraic[7]+0.00190000*(power(algebraic[7], 2.00000))
    algebraic[12] = algebraic[11]*2.00000
    algebraic[13] = algebraic[12]*algebraic[10]*constants[21]
    algebraic[0] = states[1]/states[0]
    algebraic[3] = 0.280000*algebraic[0]+0.00190000*(power(algebraic[0], 2.00000))
    algebraic[16] = (((constants[24]-algebraic[3])-algebraic[15])+algebraic[13])*constants[7]+constants[26]
    algebraic[17] = algebraic[15]-algebraic[12]
    algebraic[18] = (algebraic[17]+constants[23])-algebraic[14]
    algebraic[20] = custom_piecewise([greater(algebraic[18] , 7.00000), 7.00000 , True, algebraic[18]])
    algebraic[21] = custom_piecewise([less(algebraic[20] , 0.00000), 0.00000 , True, algebraic[20]*0.0200000])
    algebraic[22] = ((algebraic[21]-algebraic[16])-constants[3])+constants[8]
    rates[0] = algebraic[22]
    algebraic[1] = custom_piecewise([less(algebraic[0]-constants[9] , 0.00000), 0.00000 , True, algebraic[0]-constants[9]])
    algebraic[2] = constants[10]-(power(algebraic[1], constants[11]))*constants[12]
    algebraic[9] = constants[26]*algebraic[0]+(algebraic[0]-algebraic[7])*0.00104000
    algebraic[23] = algebraic[7]*algebraic[21]
    algebraic[24] = (((algebraic[2]+algebraic[23])-algebraic[9])-constants[1])+constants[8]*72.0000
    rates[1] = algebraic[24]
    algebraic[25] = algebraic[9]-algebraic[23]
    rates[3] = algebraic[25]
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[4] = (constants[0]-states[0])-constants[4]
    algebraic[5] = algebraic[4]-states[2]
    algebraic[14] = (power((algebraic[5]-constants[22])/constants[22], 2.00000))*1.00000
    algebraic[11] = power((constants[18]/algebraic[4])/5.00000, constants[19])
    algebraic[15] = algebraic[11]*constants[20]+algebraic[14]
    algebraic[7] = states[3]/algebraic[4]
    algebraic[10] = 0.280000*algebraic[7]+0.00190000*(power(algebraic[7], 2.00000))
    algebraic[12] = algebraic[11]*2.00000
    algebraic[13] = algebraic[12]*algebraic[10]*constants[21]
    algebraic[0] = states[1]/states[0]
    algebraic[3] = 0.280000*algebraic[0]+0.00190000*(power(algebraic[0], 2.00000))
    algebraic[16] = (((constants[24]-algebraic[3])-algebraic[15])+algebraic[13])*constants[7]+constants[26]
    algebraic[17] = algebraic[15]-algebraic[12]
    algebraic[18] = (algebraic[17]+constants[23])-algebraic[14]
    algebraic[20] = custom_piecewise([greater(algebraic[18] , 7.00000), 7.00000 , True, algebraic[18]])
    algebraic[21] = custom_piecewise([less(algebraic[20] , 0.00000), 0.00000 , True, algebraic[20]*0.0200000])
    algebraic[22] = ((algebraic[21]-algebraic[16])-constants[3])+constants[8]
    algebraic[1] = custom_piecewise([less(algebraic[0]-constants[9] , 0.00000), 0.00000 , True, algebraic[0]-constants[9]])
    algebraic[2] = constants[10]-(power(algebraic[1], constants[11]))*constants[12]
    algebraic[9] = constants[26]*algebraic[0]+(algebraic[0]-algebraic[7])*0.00104000
    algebraic[23] = algebraic[7]*algebraic[21]
    algebraic[24] = (((algebraic[2]+algebraic[23])-algebraic[9])-constants[1])+constants[8]*72.0000
    algebraic[25] = algebraic[9]-algebraic[23]
    algebraic[6] = custom_piecewise([less_equal(algebraic[4] , 0.00000), 0.00000 , greater(algebraic[4] , 0.00000) & less_equal(algebraic[4] , 12.0000), 0.00000+((11.4000-0.00000)*(algebraic[4]-0.00000))/(12.0000-0.00000) , greater(algebraic[4] , 12.0000) & less_equal(algebraic[4] , 15.0000), 11.4000+((14.0000-11.4000)*(algebraic[4]-12.0000))/(15.0000-12.0000) , greater(algebraic[4] , 15.0000) & less_equal(algebraic[4] , 18.0000), 14.0000+((16.0000-14.0000)*(algebraic[4]-15.0000))/(18.0000-15.0000) , greater(algebraic[4] , 18.0000) & less_equal(algebraic[4] , 21.0000), 16.0000+((17.3000-16.0000)*(algebraic[4]-18.0000))/(21.0000-18.0000) , greater(algebraic[4] , 21.0000) & less_equal(algebraic[4] , 24.0000), 17.3000+((18.0000-17.3000)*(algebraic[4]-21.0000))/(24.0000-21.0000) , True, 18.0000])
    algebraic[8] = algebraic[4]-algebraic[6]
    algebraic[19] = algebraic[14]-algebraic[17]
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