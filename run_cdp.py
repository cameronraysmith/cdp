# -*- coding: utf-8 -*-
"""
coupled Dirichlet processes
"""
import sys
import numpy as np
import pymc as mc
import matplotlib.pyplot as plt
import subprocess
from scipy.stats.mstats import mquantiles
from pymc.Matplot import plot as mcplot

import cdp

def load_cdp_run(pickledb="cdp.pickle"):
    pass

def fit_cdp():
    #model = vars(cdp)
    model = cdp.gaussian_dpmm()

    #mc.MAP(model).fit(method='fmin_powell')
    mc.MAP(model).fit(method='fmin')

    try:
        with open('cdp.pickle'):
            m = mc.MCMC(model)
    except IOError:
        print "Saving simulation data to new pickle database"
        m = mc.MCMC(model, db='pickle', dbname='cdp.pickle')

    # impose Adaptive Metropolis
    #m.use_step_method(mc.AdaptiveMetropolis,
    #     [m.lagtime, m.linslope, m.carcap, m.sigma])

    # low for testing
    #m.sample(iter=6000, burn=5000, thin=1)
    # medium
    m.sample(iter=10000, burn=8000, thin=2)
    # long
    #m.sample(iter=50000, burn=40000, thin=10)
    # longer
    #m.sample(iter=150000, burn=140000, thin=10)
    m.db.close()
    return m

def plot_cdp(m, ffname):
    z = m.z.trace()[:]
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.hist([len(np.unique(z[i,:])) for i in range(len(z))])
    plt.xlabel('Components used')
    plt.ylabel('Frequency')
    plt.title('truncated K = %d' % m.N_dp)
    plt.xlim([0, m.N_dp])
    # plt.subplot(1, 2, 2)
    # plt.hist([len(np.unique(z10[i,:])) for i in range(len(z10))])
    # plt.xlabel('Components used')
    # plt.ylabel('Frequency')
    # plt.title('K = 30')
    # plt.xlim([0, 8])
    plt.savefig(ffname)
    plt.close()

def plot_parcorr(m):
    parnames = ["a","z","p"]
    for par in parnames:
        mcplot(m.trace(par), common_scale=False)
        plt.savefig(par + ".pdf", bbox_inches='tight', edgecolor='none')
        plt.close()

def main(argv):
    #simulate model
    m = fit_cdp()
    print " "
    print m.dic

    print "Saving graphical representation"
    mc.graph.graph(m, name="cdpgraph", format="pdf",
                   prog="dot", legend=False, consts=True)

    #plot distribution of components to which each data point belongs
    plot_cdp(m, "components.pdf")

    #plot parameter autocorrelation
    #print "Plotting parameter distributions"
    #plot_parcorr(m)

    # save parameter estimation data to csv file
    # print "Saving parameter estimates"
    # pars_ffname = "parests.csv"
    # cdpmodvars = ["a","z","p"]
    # m.write_csv(pars_ffname, variables=cdpmodvars)

    #combine figures
    # print "Combining figures"
    # pdfcombine = subprocess.check_output(["pdftk", "a.pdf", "z.pdf", "p.pdf", "cat", "output", "allfigs.pdf"])

if __name__ == "__main__":
    #import doctest
    #doctest.testmod()
    main(sys.argv[1:])
