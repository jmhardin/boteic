#==============================================================================
# This sample script reproduces the energy and zenith distributions for the
# nominal monte carlo and experimental data. As mentioned in the ReadMe,
# the file NuFSGenMC_nominal.dat contains an example flux,
# pre-calculated using the Honda+Gaisser pion and kaon flux model.
#
#==============================================================================

# First, we calculate the event weight via:
#     weight = flux * mcweight    ...(1)
# and plot the MC with the data

import sys
import numpy as np
import pylab


font = {'family':'serif',
        'serif':'Computer Modern Roman',
        'weight':200,
        'size':22}

pylab.rc('font', **font)

# Import the experimental data
class DataClass():
    def __init__(self,infile):
        self.name = infile.split("/")[-1]
        print('Loading '+str(infile.split("/")[-1])+' ...')
        exp_energy = []
        exp_cosZ = []
        try:
            expData = np.loadtxt(infile, skiprows = 12)
            for col in expData:
                #print(float(col[0]))
                exp_energy.append(float(col[0]))
                exp_cosZ.append(np.cos(float(col[1])))
            self.exp_energy = exp_energy
            self.exp_cosZ = exp_cosZ
        except:
            print('We could not find the file. Check paths.')

#Data = DataClass('../../data/observed_events.dat')


# import the monte carlo
class MCClass():
    def __init__(self,infile):
        self.name = infile.split("/")[-1]
        print('Loading '+str(infile.split("/")[-1])+' ...')
        id = []
        reco_energy = []
        true_energy = []
        reco_cosZ = []
        true_cosZ = []
        weights = []
        mc_weight = []
        pion_flux = []
        kaon_flux = []
        total_flux = []
        total_count_nominal = []
        mcData=np.loadtxt(infile,delimiter=' ',skiprows = 11)
        for col in mcData:
            id.append(float(col[0]))
            reco_energy.append(float(col[1]))
            reco_cosZ.append(float(col[2]))
            true_energy.append(float(col[3]))
            true_cosZ.append(float(col[4]))
            mc_weight.append(float(col[5]))
            pion_flux.append(float(col[6]))
            kaon_flux.append(float(col[7]))
            Ftot = float(col[7]) + float(col[6])
            total_flux.append(Ftot)
            weights.append(Ftot*float(col[5]))
            Ftot = float(col[7]) + float(col[6])
            total_count_nominal.append(Ftot*float(col[5]))
        self.id = id
        self.reco_energy = reco_energy
        self.true_energy = true_energy
        self.reco_cosZ = reco_cosZ
        self.true_cosZ = true_cosZ
        self.mc_weight = mc_weight
        self.pion_flux = pion_flux
        self.kaon_flux = kaon_flux
        self.mc_weight = mc_weight
        self.total_flux = total_flux
        self.weights = weights
        self.total_count_nominal = total_count_nominal

MonteCarlo = MCClass('../../monte_carlo/NuFSGenMC_nominal.dat')

def output_csvs(ipref,ipdg,ipid):
    pref = ipref

    num_cosz_bins = 20
    cosz_low = -1.
    cosz_high = 1.
    num_e_bins = 240
    log_e_low = 2
    log_e_high = 6

    ebins = np.logspace(log_e_low,log_e_high,num_e_bins + 1)
    coszbins = np.linspace(cosz_low,cosz_high,num_cosz_bins+1)

    numu_e = [] 
    numu_cosz = [] 
    numu_weights = []

    delta_e = []
    delta_sq_e = []

    numubar_e = [] 
    numubar_cosz = []
    numubar_weights = []

    print("MCEvents:")
    print(len(MonteCarlo.id))

    for i in range(len(MonteCarlo.id)):
        pdg = MonteCarlo.id[i]
        true_e = MonteCarlo.true_energy[i]
        reco_e = MonteCarlo.reco_energy[i]
        true_cosz = MonteCarlo.true_cosZ[i]
        weight = MonteCarlo.mc_weight[i]

        if (pdg == ipdg):
            numu_e.append(true_e)
            numu_cosz.append(true_cosz)
            numu_weights.append(weight)
            delta_e.append(reco_e - true_e)
            delta_sq_e.append((reco_e - true_e)*(reco_e - true_e))



    #We really just want the average weight in each bin so that we can interpolate later

    numu_weights_vals,x_array,y_array = np.histogram2d(numu_e,numu_cosz,[ebins,coszbins],weights=numu_weights)
    numu_num_vals,x_array,y_array = np.histogram2d(numu_e,numu_cosz,[ebins,coszbins])

    numu_weights_e_vals,x_e_array = np.histogram(numu_e,ebins,weights=numu_weights)
    numu_num_e_vals, x_e_array = np.histogram(numu_e,ebins)

    numu_weights_cosz_vals, x_cosz_array = np.histogram(numu_cosz,coszbins,weights=numu_weights)
    numu_num_cosz_vals, x_cosz_array = np.histogram(numu_cosz,coszbins)


    numu_delta_e_vals,x_de_array = np.histogram(numu_e,ebins,weights=delta_e)
    numu_delta_sq_e_vals,x_de_array = np.histogram(numu_e,ebins,weights=delta_sq_e)

    to_sec_m2 = 347*3600*24*10000

    for i in range(len(numu_weights_vals)):
        for j in range(len(numu_weights_vals[i])):
            #cval = numu_num_vals[i][j]
            cval = (ebins[i+1] - ebins[i])*(coszbins[j+1] - coszbins[j])
            if (cval > 0):
                numu_weights_vals[i][j] /= (cval*to_sec_m2)
                #numu_weights_vals[i][j] /= (to_sec_m2)

    for i in range(len(numu_weights_e_vals)):
        #cval = numu_num_e_vals[i]
        cval = ebins[i+1] - ebins[i]
        if (cval > 0):
            numu_weights_e_vals[i] /= (cval*to_sec_m2)
            #numu_weights_e_vals[i] /= (to_sec_m2)

    for i in range(len(numu_weights_cosz_vals)):
        #cval = numu_num_cosz_vals[i]
        cval = ebins[i+1] - ebins[i]
        if (cval > 0):
            numu_weights_cosz_vals[i] /= (cval*to_sec_m2)
            #numu_weights_cosz_vals[i] /= (to_sec_m2)

    stddev_e = []
    for i in range(len(numu_delta_e_vals)):
        cval = numu_num_e_vals[i]
        stddev_e.append(0)
        if (cval > 0):
            numu_delta_e_vals[i] /= cval
            stddev_e[i] = numu_delta_sq_e_vals[i]/cval - numu_delta_e_vals[i]*numu_delta_e_vals[i]


    font = {'family':'serif',
            'serif':'Computer Modern Roman',
            'weight':200,
            'size':22}

    pylab.rc('font', **font)

    pylab.figure(figsize=(20, 5))
    pylab.subplot(121)
    xmid = (y_array[1] - y_array[0]) / 2
    cosz_midbins = np.linspace(cosz_low+xmid,cosz_high-xmid,num_cosz_bins)
    pylab.plot(cosz_midbins,numu_weights_cosz_vals,label='weight',linewidth=2.0)
    pylab.xscale('linear')
    pylab.yscale('log')
    pylab.grid(b=True, which='major', color='black', alpha = 0.3, linestyle='-')
    pylab.grid(b=True, which='minor', color='black', alpha = .1, linestyle='-')
    pylab.ylabel('Average Weight (1/m^2*s^2)')
    pylab.xlabel(r'cos($\theta_z$)')
    pylab.axis([-1.0, 1.0, 1e-10, 1e2])
    pylab.legend(fontsize=18, loc='upper left', fancybox=True)

    pylab.subplot(122)
    xmid = (log_e_low + log_e_high) / (2*(num_e_bins - 1))
    e_midbins = np.logspace(log_e_low+xmid,log_e_high-xmid,num_e_bins)
    pylab.plot(e_midbins,numu_weights_e_vals,label='weight',linewidth=2.0)
    pylab.xscale('log')
    pylab.yscale('log')
    pylab.grid(b=True, which='major', color='black', alpha=0.3, linestyle='-')
    pylab.grid(b=True, which='minor', color='black', alpha=0.1, linestyle='-')
    pylab.ylabel('Average Weight (1/m^2*s^2)')
    pylab.xlabel('Thrown Energy [GeV]')
    pylab.axis([ebins[0],ebins[-1],1e-12,1])
    pylab.legend(fontsize=18,loc='upper right',fancybox = True)

    pylab.savefig(pref+"distributions.png", bbox_inches='tight')
    pylab.clf()

    np.savetxt(pref+'avgweights.csv',numu_weights_vals,delimiter=",")
    np.savetxt(pref+'evals.csv',e_midbins,delimiter=",")
    spread_array = []
    spread_array.append(e_midbins)
    spread_array.append(numu_delta_e_vals)
    spread_array.append(stddev_e)
    np.savetxt(pref+'varvals.csv',spread_array,delimiter=",")
    np.savetxt(pref+'coszvals.csv',cosz_midbins,delimiter=",")


output_csvs("ic86_weights/numu_track_",13,1)
#output_csvs("ic86_weights/numu_cascade_",14,0)
output_csvs("ic86_weights/numubar_track_",-13,1)
#output_csvs("ic86_weights/numubar_cascade_",-14,0)
#output_csvs("ic86_weights/nue_track_",12,1)
#output_csvs("ic86_weights/nue_cascade_",12,0)
#output_csvs("ic86_weights/nuebar_track_",-12,1)
#output_csvs("ic86_weights/nuebar_cascade_",-12,0)



        
