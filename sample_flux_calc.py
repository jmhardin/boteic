import numpy as np
import scipy.interpolate
import nuflux
import nuSQuIDS as nsq
import nuSQUIDSTools
import time


flux = nuflux.makeFlux('H3a_SIBYLL23C')
#flux = nuflux.makeFlux('CORSIKA_GaisserH3a_average')
#flux = nuflux.makeFlux('sarcevic_std')

units = nsq.Const()


#This, like most of the setup, stolen from Austin
def setup_SM_oscillations(nuSQ):
    nuSQ.Set_MixingAngle(0,1,0.563942)
    nuSQ.Set_MixingAngle(0,2,0.154085)
    nuSQ.Set_MixingAngle(1,2,0.785398)

    nuSQ.Set_SquareMassDifference(1,7.65e-05)
    nuSQ.Set_SquareMassDifference(2,0.00247)

    nuSQ.Set_CPPhase(0,2,0.0)
    nuSQ.Set_rel_error(1.0e-08)
    nuSQ.Set_abs_error(1.0e-08)


#Use a lot of czbins and a lot of e bins, respectively
n_czbins = 100
n_ebins = 100
czbins = np.linspace(-1,1,n_czbins+1)
ebins = np.logspace(1,6,n_ebins+1)
num_nu = 3

init_state = np.zeros([n_czbins+1,n_ebins+1,2,num_nu])


for ei in range (0,n_ebins+1):
    ebins[ei] *= units.GeV

for ei in range(0,n_ebins+1):
    cur_e = ebins[ei]
    for czi in range(0,n_czbins+1):
        cur_cosz = czbins[czi]
        if (cur_e < 10001*units.GeV):
            init_state[czi][ei][0][0] = flux.getFlux(nuflux.NuE,cur_e/units.GeV,cur_cosz);
            init_state[czi][ei][0][1] = flux.getFlux(nuflux.NuMu,cur_e/units.GeV,cur_cosz);
            init_state[czi][ei][0][2] = flux.getFlux(nuflux.NuTau,cur_e/units.GeV,cur_cosz);

            init_state[czi][ei][1][0] = flux.getFlux(nuflux.NuEBar,cur_e/units.GeV,cur_cosz);
            init_state[czi][ei][1][1] = flux.getFlux(nuflux.NuMuBar,cur_e/units.GeV,cur_cosz);
            init_state[czi][ei][1][2] = flux.getFlux(nuflux.NuTauBar,cur_e/units.GeV,cur_cosz);

interactions = True
nuSQ = nsq.nuSQUIDSAtm(czbins, ebins, num_nu, nsq.NeutrinoType.both, interactions)

setup_SM_oscillations(nuSQ)

nuSQ.Set_ProgressBar(False)
nuSQ.Set_IncludeOscillations(True)
nuSQ.Set_GlashowResonance(True);
nuSQ.Set_TauRegeneration(True);


# nuSQuIDS inputs should be given in natural units. In order to make this convenient we have define a units class called *Const*. We can instanciate it as follows

nuSQ.Set_initial_state(init_state,nsq.Basis.flavor)

# Finally we can tell $\nu$-SQuIDS to perform the calculation. In case one (or all) of the above parameters is not set $\nu$-SQuIDS will throw an exception and tell you to fix it, but if you have defined everything then it will evolve the given state.

print("Evolving flux through earth")
elapsed = time.time()
nuSQ.EvolveState()
elapsed = time.time() - elapsed
print(f'Evolved state in {elapsed:0.1f} seconds')

#print(nuSQ.EvalFlavor(0,-.96,1000*units.GeV))


def get_weights_function(pref):
    weights_2d = np.genfromtxt(pref + 'avgweights.csv', delimiter=',')
    e_vals = np.genfromtxt(pref + 'evals.csv', delimiter=',')
    cosz_vals = np.genfromtxt(pref + 'coszvals.csv', delimiter=',')

    return scipy.interpolate.interp2d(cosz_vals,e_vals,weights_2d)

def get_mean_err_function(pref):
    e_vals = np.genfromtxt(pref + 'evals.csv', delimiter=',')
    de_vals = np.genfromtxt(pref + 'varvals.csv', delimiter=',')[1]

    return scipy.interpolate.interp1d(e_vals,de_vals)

def get_stddev_err_function(pref):
    e_vals = np.genfromtxt(pref + 'evals.csv', delimiter=',')
    var_vals = np.genfromtxt(pref + 'varvals.csv', delimiter=',')[2]

    #I'm sure there is a nicer numpy way to do this, but I like my for loops
    for i in range(len(var_vals)):
        var_vals[i] = np.sqrt(var_vals[i])
    return scipy.interpolate.interp1d(e_vals,var_vals)

def integrate_v_weightfunc(wf,flavor,rho,emin,emax,czmin,czmax):
    #Should return the rate per second in these ranges
    #flavor is 0, 1 or 2 for e, mu, tau
    #rho is 0 for matter, 1 for antimatter
    #Match these to the weights files when you can
    #weights are in m^2, nuflux is in 1/cm^2*GeV*sr*s, so we account for that
    ebins = 100
    czbins = 100
    inv_dedsr = 0    
    oval = 0.0

    for e in np.linspace(emin,emax,ebins):
        for cz in np.linspace(czmin,czmax,czbins):
            oval += 10000*wf(cz,e)*nuSQ.EvalFlavor(flavor,cz,e*units.GeV,rho)
            inv_dedsr += 1
    return oval*(emax-emin)*(czmax-czmin)/inv_dedsr

test_func = get_weights_function('ic86_weights/numu_track_')
test_func_nubar = get_weights_function('ic86_weights/numubar_track_')

test_de = get_mean_err_function('ic86_weights/numu_track_')
test_stddev = get_stddev_err_function('ic86_weights/numu_track_')

ebins = np.logspace(2.60206,4.30103,10+1)

rng = np.random.default_rng()

spread_e = []

#Likely you need slightly finer binning and bins outside the range to make this work exactly
#but this is an idea
#The mean "reco" energy is well below the true energy for the ic86 sample
#for i in range(10):
#    a = 3.14e7*integrate_v_weightfunc(test_func,1,0,ebins[i],ebins[i+1],-1.,1.0)
#    b = 3.14e7*integrate_v_weightfunc(test_func_nubar,1,1,ebins[i],ebins[i+1],-1.,1.0)
#
#    for j in range(int(a+b)):
#        e_val = rng.uniform(ebins[i],ebins[i+1])
#        se_val = rng.normal(e_val + test_de(e_val),test_stddev(e_val))
#        spread_e.append(se_val)


#histo, earr = np.histogram(spread_e,ebins)

#print(histo)
