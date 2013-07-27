import numpy as np
import pymc

def gaussian_dpmm(N_dp=20, data=np.array([10., 11., 12., -10., -11., -12.])):
    """
        Dirichlet process sample measure
        G = \Sum_{k=1}^{\infty} \pi_k \delta_{phi_k}

        N_dp:: Size of truncated DP
        data:: data
        G_0:: Base measure - Gaussian
        phi_k:: Samples from base measure
        pi_k:: Stick-breaking weights
    """
    # Hyperpriors
    ## for phi_k
    mu_0 = pymc.Normal('mu_0', mu=0, tau=0.01, value=0)
    sig_0 = pymc.Uniform('sig_0', lower=0, upper=100, value=1)
    tau_0 = pymc.Lambda('tau_0', lambda s=sig_0: s**-2)

    ## for data
    sig_x = pymc.Uniform('sig_x', lower=0, upper=100)
    tau_x = pymc.Lambda('tau_x', lambda s=sig_x: s**-2)

    # Concentration parameter
    alpha = pymc.Uniform('alpha', lower=0.3, upper=10)

    # Samples from baseline measure (G_0) for DP
    # determine potential cluster locations (\delta_{phi_k})
    phi_k = pymc.Normal('phi_k', mu=mu_0, tau=tau_0, size=N_dp)

    pi_kp = pymc.Beta('pi_kp', alpha=1, beta=alpha, size=N_dp)

    @pymc.deterministic
    def pi_k(pi_kp=pi_kp, value = np.ones(N_dp)/N_dp):
        """
            Calculate Dirichlet probabilities
            aka Stick-breaking weights
        """
        # Probabilities from betas
        value = [u*np.prod(1-pi_kp[:i]) for i,u in enumerate(pi_kp)]
        # Enforce sum to unity constraint
        value[-1] = 1-sum(value[:-1])

        return value

    # Component to which each data point belongs
    z = pymc.Categorical('z', pi_k, size=len(data))

    # "Location" of clusters in terms of baseline (G_0)
    # (\delta_{phi_k})
    #g = pymc.Lambda('g', lambda z=z, phi_k=phi_k: phi_k[z])

    # Observation model
    x = pymc.Normal('x', mu = phi_k[z], tau = tau_x, value = data,
                    observed = True)
    # Model posterior predictive
    x_sim = pymc.Normal('x_sim', mu = phi_k[z], tau = tau_x, value = data)

    # Expected value of random measure
    #E_dp = pymc.Lambda('E_dp', lambda pi_k=pi_k, phi_k=phi_k: np.inner(pi_k, phi_k))

    return vars()

def dm_dpmm(N_dp=20, data=np.array([10., 11., 12., -10., -11., -12.])):
    """
        Dirichlet process sample measure
        G = \Sum_{k=1}^{\infty} \pi_k \delta_{phi_k}

        N_dp:: Size of truncated DP
        data:: data
        G_0:: Base measure - Multinomial
        phi_k:: Samples from base measure
        pi_k:: Stick-breaking weights
    """
    # Hyperpriors
    mu_0 = pymc.Normal('mu_0', mu=0, tau=0.01, value=0)
    sig_0 = pymc.Uniform('sig_0', lower=0, upper=100, value=1)
    tau_0 = pymc.Lambda('tau_0', lambda s=sig_0: s**-2)
    #tau_0 = sig_0 ** -2

    sig_x = pymc.Uniform('sig_x', lower=0, upper=100)
    tau_x = pymc.Lambda('tau_x', lambda s=sig_x: s**-2)

    # Concentration parameter
    alpha = pymc.Uniform('alpha', lower=0.3, upper=10)

    # Samples from baseline measure (G_0) for DP
    # determine potential cluster locations (\delta_{phi_k})

    #ps = np.random.dirichlet([10, 5, 3, 7, 18, 6])
    #np.random.multinomial(20, ps, size=1)

    phi_k = pymc.Normal('phi_k', mu=mu_0, tau=tau_0, size=N_dp)

    pi_kp = pymc.Beta('pi_kp', alpha=1, beta=alpha, size=N_dp)

    @pymc.deterministic
    def pi_k(pi_kp=pi_kp, value = np.ones(N_dp)/N_dp):
        """
            Calculate Dirichlet probabilities
            aka Stick-breaking weights
        """
        # Probabilities from betas
        value = [u*np.prod(1-pi_kp[:i]) for i,u in enumerate(pi_kp)]
        # Enforce sum to unity constraint
        value[-1] = 1-sum(value[:-1])

        return value

    # Component to which each data point belongs
    z = pymc.Categorical('z', pi_k, size=len(data))

    # "Location" of clusters in terms of baseline (G_0)
    # (\delta_{phi_k})
    #g = pymc.Lambda('g', lambda z=z, phi_k=phi_k: phi_k[z])

    # Observation model
    x = pymc.Normal('x', mu = phi_k[z], tau = tau_x, value = data,
                    observed = True)
    # Model posterior predictive
    x_sim = pymc.Normal('x_sim', mu = phi_k[z], tau = tau_x, value = data)

    # Expected value of random measure
    #E_dp = pymc.Lambda('E_dp', lambda pi_k=pi_k, phi_k=phi_k: np.inner(pi_k, phi_k))

    return vars()
