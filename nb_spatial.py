import pandas as pd
import numpy as np
import os
import sys
import time
from joblib import Parallel, delayed
import h5py
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import expm
from scipy.special import loggamma
from scipy.stats import invwishart, rankdata
import torch
import torch.optim as optim
from numba import jit
import pypolyagamma as pypolyagamma
import networkx as nx

#####################################
# Data and Utility Classes
#####################################

class Data:
    """Holds the training data for Negative Binomial modeling.
    
    Attributes:
        y (array): dependent count data.
        N (int): number of dependent data.
        x_fix (array): covariates for fixed link function parameters.
        n_fix (int): number of covariates for fixed link function parameters.
        x_rnd (array): covariates for random link function parameters.
        n_rnd (int): number of covariates for random link function parameters.
        W (array): row-normalized spatial weight matrix.
    """   
    
    def __init__(self, y, x_fix, x_rnd, W):
        self.y = y
        self.N = y.shape[0]
        
        self.x_fix = x_fix
        if x_fix is not None:
            self.n_fix = x_fix.shape[1]
        else:
            self.n_fix = 0
            
        self.x_rnd = x_rnd
        if x_rnd is not None:
            self.n_rnd = x_rnd.shape[1] 
        else:
            self.n_rnd = 0
            
        self.W = W
        

class Options:
    """Contains options for MCMC algorithm.
    
    Attributes:
        model_name (string): name of the model to be estimated.
        nChain (int): number of Markov chains.
        nBurn (int): number of samples for burn-in. 
        nSample (int): number of samples after burn-in period.
        nThin (int): thinning factors.
        disp (int): number of samples after which progress is displayed. 
        mh_step_initial (float): Initial MH step size.
        mh_target (float): target MH acceptance rate.
        mh_correct (float): correction of MH step.
        mh_window (int): number of samples after which to adjust MG step size.
        delete_draws (bool): Whether simulation draws should be deleted.
        seed (int): random seed.
    """   

    def __init__(
            self, 
            model_name='test',
            nChain=1, nBurn=500, nSample=500, nThin=2, nMem=None, disp=100, 
            mh_step_initial=0.1, mh_target=0.3, mh_correct=0.01, mh_window=50,
            delete_draws=True, seed=4711
            ):
        self.model_name = model_name
        self.nChain = nChain
        self.nBurn = nBurn
        self.nSample = nSample
        self.nIter = nBurn + nSample
        self.nThin = nThin
        self.nKeep = int(nSample / nThin)
        if nMem is None:
            self.nMem = self.nKeep
        else:
            self.nMem = nMem
        self.disp = disp
        
        self.mh_step_initial = mh_step_initial
        self.mh_target = mh_target
        self.mh_correct = mh_correct
        self.mh_window = mh_window
        
        self.delete_draws = delete_draws
        self.seed = seed
        
        
class Results:
    """Holds simulation results.
    
    Attributes:
        options (Options): Options used for MCMC algorithm.
        estimation time (float): Estimation time.
        lppd (float): Log pointwise predictive density.
        waic (float): Widely applicable information criterion.
        post_mean_lam (array): Posterior mean of expected count.
        rmse, mae, rmsle (float): Error metrics.
        post_mean_ranking (array): Posterior mean site rank.
        post_mean_ranking_top (array): Posterior probability that a site
          belongs to the top m most hazardous sites.
        ranking_top_m_list (list): List of m values for top site calculations.
        post_r (DataFrame): Posterior summary of negative binomial success rate.
        post_beta (DataFrame): Posterior summary of fixed link function parameters.
        post_mu (DataFrame): Posterior summary of random link function parameters.
        post_sigma (DataFrame): Posterior summary of standard deviations.
        post_Sigma (DataFrame): Posterior covariance matrix.
        post_sigma_mess: Posterior summary of spatial error scale.
        post_tau: Posterior summary of MESS association parameters.
    """   
    
    def __init__(self, 
                 options, bart_options, toc, 
                 lppd, waic,
                 post_mean_lam, 
                 rmse, mae, rmsle,
                 post_mean_ranking, 
                 post_mean_ranking_top, ranking_top_m_list,
                 post_r, post_mean_f, 
                 post_beta,
                 post_mu, post_sigma, post_Sigma,
                 post_variable_inclusion_props,
                 post_sigma_mess, post_tau):
        self.options = options
        self.bart_options = bart_options
        self.estimation_time = toc
        
        self.lppd = lppd
        self.waic = waic
        self.post_mean_lam = post_mean_lam
        self.rmse = rmse
        self.mae = mae
        self.rmsle = rmsle
        
        self.post_mean_ranking = post_mean_ranking
        self.post_mean_ranking_top = post_mean_ranking_top
        self.ranking_top_m_list = ranking_top_m_list
        
        self.post_r = post_r
        self.post_mean_f = post_mean_f
        
        self.post_variable_inclusion_props = post_variable_inclusion_props
        
        self.post_beta = post_beta
        
        self.post_mu = post_mu
        self.post_sigma = post_sigma
        self.post_Sigma = post_Sigma
        
        self.post_sigma_mess = post_sigma_mess
        self.post_tau = post_tau
        
        # Add test predictions attribute for cross-validation
        self.test_predictions = None
        self.test_actual = None


#####################################
# NegativeBinomial Model Class
#####################################

class NegativeBinomial:
    """MCMC method for posterior inference in negative binomial model."""
    
    @staticmethod
    def _F_matrix(y):
        """Calculates F matrix."""
        y_max = np.max(y)
        F = np.zeros((y_max, y_max))
        for m in np.arange(y_max):
            for j in np.arange(m+1):
                if m==0 and j==0:
                    F[m,j] = 1
                else:
                    F[m,j] = m/(m+1) * F[m-1,j] + (1/(m+1)) * F[m-1,j-1]
        return F
    
    def __init__(self, data, data_bart=None):
        self.data = data
        self.data_bart = data_bart
        self.F = self._F_matrix(data.y)
        self.N = data.N
        self.n_fix = data.n_fix
        self.n_rnd = data.n_rnd
        self.mess = data.W is not None

    # Convenience methods
    @staticmethod
    def _pg_rnd(a, b):
        """Takes draws from Polya-Gamma distribution with parameters a, b."""
        ppg = pypolyagamma.PyPolyaGamma(np.random.randint(2**16))
        N = a.shape[0]
        r = np.zeros((N,))
        ppg.pgdrawv(a, b, r)
        return r
    
    @staticmethod
    def _mvn_prec_rnd(mu, P):
        """Takes draws from multivariate normal with mean mu and precision P."""
        d = mu.shape[0]
        P_chT = np.linalg.cholesky(P).T
        r = mu + np.linalg.solve(P_chT, np.random.randn(d,1)).reshape(d,)
        return r 
    
    @staticmethod
    def _mvn_prec_rnd_arr(mu, P):
        """Takes draws from multivariate normal with array of mean vectors."""
        N, d = mu.shape
        P_chT = np.moveaxis(np.linalg.cholesky(P), [1,2], [2,1])
        r = mu + np.linalg.solve(P_chT, np.random.randn(N,d,1)).reshape(N,d)
        return r 
    
    @staticmethod
    def _nb_lpmf(y, psi, r):
        """Calculates log pmf of negative binomial."""
        lc = loggamma(y + r) - loggamma(r) - loggamma(y + 1)
        lp = y * psi - (y + r) * np.log1p(np.exp(psi))
        lpmf = lc + lp
        return lpmf
    
    # MCMC parameter sampling methods
    def _next_omega(self, r, psi):
        """Updates omega."""
        # Clip psi to prevent extreme values that could cause numerical problems
        psi_clipped = np.clip(psi, -50, 50)
        omega = self._pg_rnd(self.data.y + r, psi_clipped)
        # Add small epsilon to omega to prevent division by zero
        omega = np.maximum(omega, 1e-10)
        Omega = np.diag(omega)
        z = (self.data.y - r) / 2 / omega
        return omega, Omega, z

    def _next_beta(self, z, psi_rnd, psi_bart, phi, 
                   Omega, beta_mu0, beta_Si0Inv):
        """Updates beta."""
        beta_SiInv = self.data.x_fix.T @ Omega @ self.data.x_fix + beta_Si0Inv
        beta_mu = np.linalg.solve(beta_SiInv,
                                  self.data.x_fix.T @ Omega @ (z - psi_rnd - psi_bart - phi) + 
                                  beta_Si0Inv @ beta_mu0)
        beta = self._mvn_prec_rnd(beta_mu, beta_SiInv)
        return beta
    
    def _next_gamma(self, z, psi_fix, psi_bart, phi, omega, mu, SigmaInv):
        """Updates gamma."""
        gamma_SiInv = omega.reshape(self.N,1,1) * \
            self.data.x_rnd.reshape(self.N,self.n_rnd,1) @ \
            self.data.x_rnd.reshape(self.N,1,self.n_rnd) + \
            SigmaInv.reshape(1,self.n_rnd,self.n_rnd)
        gamma_mu_pre = (omega * (z - psi_fix - psi_bart - phi)).reshape(self.N,1) * \
            self.data.x_rnd + (SigmaInv @ mu).reshape(1,self.n_rnd)
        gamma_mu = np.linalg.solve(gamma_SiInv, gamma_mu_pre)
        gamma = self._mvn_prec_rnd_arr(gamma_mu, gamma_SiInv)            
        return gamma
 
    def _next_mu(self, gamma, SigmaInv, mu_mu0, mu_Si0Inv):
        """Updates mu."""
        mu_SiInv = self.N * SigmaInv + mu_Si0Inv
        mu_mu = np.linalg.solve(mu_SiInv,
                                SigmaInv @ np.sum(gamma, axis=0) + 
                                mu_Si0Inv @ mu_mu0)
        mu = self._mvn_prec_rnd(mu_mu, mu_SiInv)
        return mu

    def _next_Sigma(self, gamma, mu, a, nu):    
        """Updates Sigma."""
        diff = gamma - mu
        Sigma = (invwishart.rvs(nu + self.N + self.n_rnd - 1, 
                                2 * nu * np.diag(a) + diff.T @ diff))\
            .reshape((self.n_rnd, self.n_rnd))
        SigmaInv = np.linalg.inv(Sigma)
        return Sigma, SigmaInv

    def _next_a(self, SigmaInv, nu, A):
        """Updates a."""
        a = np.random.gamma((nu + self.n_rnd) / 2, 
                            1 / (1 / A**2 + nu * np.diag(SigmaInv)))
        return a

    def _next_phi(self, z, psi_fix, psi_rnd, psi_bart, Omega, Omega_tilde):
        """Updates phi."""
        phi_SiInv = Omega + Omega_tilde
        phi_mu = np.linalg.solve(phi_SiInv, 
                                 Omega @ (z - psi_fix - psi_rnd - psi_bart))
        phi = self._mvn_prec_rnd(phi_mu, phi_SiInv)
        return phi      
    
    def _next_sigma2(self, phi, S, sigma2_b0, sigma2_c0):
        """Updates sigma_mess**2."""
        b = sigma2_b0 + self.data.N / 2
        c = sigma2_c0 + phi.T @ S.T @ S @ phi / 2
        sigma2 = 1 / np.random.gamma(b, 1 / c)
        return sigma2
    
    def _log_target_tau(self, tau, S, phi, sigma2, tau_mu0, tau_si0):
        """Calculates target density for MH."""
        if S is None:
            S = expm(tau * self.data.W)
        Omega_tilde = S.T @ S / sigma2
        lt = - phi.T @ Omega_tilde @ phi / 2 - (tau - tau_mu0)**2 / 2 / tau_si0**2
        return lt, S
    
    def _next_tau(self, tau, S, phi, sigma2, tau_mu0, tau_si0, mh_step):
        """Updates tau."""
        lt_tau, S = self._log_target_tau(tau, S, phi, sigma2, tau_mu0, tau_si0)
        tau_star = tau + np.sqrt(mh_step) * np.random.randn()
        lt_tau_star, S_star = self._log_target_tau(
            tau_star, None, phi, sigma2, tau_mu0, tau_si0
            )
        log_r = np.log(np.random.rand())
        log_alpha = lt_tau_star - lt_tau
        if log_r <= log_alpha:
            tau = tau_star
            S = np.array(S_star)
            mh_tau_accept = True
        else:
            mh_tau_accept = False
        return tau, S, mh_tau_accept
    
    @staticmethod
    def _next_h(r, r0, b0, c0):
        """Updates h."""
        h = np.random.gamma(r0 + b0, 1/(r + c0))
        return h
    
    @staticmethod
    @jit(nopython=True)
    def _next_L(y, r, F):
        """Updates L."""
        N = y.shape[0]
        L = np.zeros((N,))
        for n in np.arange(N):
            if y[n]:
                numer = np.zeros((y[n],))
                for j in np.arange(y[n]):
                    numer[j] = F[y[n]-1,j] * r**(j+1)
                L_p = numer / np.sum(numer)
                L[n] = np.searchsorted(np.cumsum(L_p), np.random.rand()) + 1
        return L
    
    @staticmethod
    def _next_r(r0, L, h, psi):
        """Updates r."""
        # Clip psi to avoid overflow in exp
        psi_clipped = np.clip(psi, -50, 50)
        sum_p = np.sum(np.log1p(np.exp(psi_clipped)))
        r = np.random.gamma(r0 + np.sum(L), 1 / (h + sum_p))
        # Ensure r is positive and not too small
        r = max(r, 1e-5)
        return r
    
    @staticmethod
    def _rank(lam, ranking_top_m_list):
        """Computes the rank of each site."""
        ranking = rankdata(-lam, method='min')
        ranking_top = np.zeros((lam.shape[0], len(ranking_top_m_list)))
        for j, m in enumerate(ranking_top_m_list):
            ranking_top[:,j] = ranking <= m
        return ranking, ranking_top
    
    def _mcmc_chain(
            self,
            chainID, 
            options, bart_options,
            r0, b0, c0,
            beta_mu0, beta_Si0Inv,
            mu_mu0, mu_Si0Inv, nu, A,
            sigma2_b0, sigma2_c0, tau_mu0, tau_si0,
            r_init, beta_init, mu_init, Sigma_init,
            ranking_top_m_list,
            seed=None):
        """Markov chain for MCMC simulation for negative binomial model."""
        
        # Set unique seed for this chain
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(options.seed + chainID)  # Alternative approach
        
        # Storage setup
        file_name = options.model_name + '_draws_chain' + str(chainID + 1) + '.hdf5'
        if os.path.exists(file_name):
            os.remove(file_name) 
        file = h5py.File(file_name, 'a')    
        
        lp_store = file.create_dataset('lp_store', (options.nKeep,self.N), dtype='float64')
        lp_store_tmp = np.zeros((options.nMem,self.N))
        
        lam_store = file.create_dataset('lam_store', (options.nKeep,self.N), dtype='float64')
        lam_store_tmp = np.zeros((options.nMem,self.N))
        
        ranking_store = file.create_dataset('ranking_store', (options.nKeep,self.N), dtype='float64')
        ranking_store_tmp = np.zeros((options.nMem,self.N))
        
        ranking_top_store = file.create_dataset('ranking_top_store', 
                                               (options.nKeep,self.N,len(ranking_top_m_list)), 
                                               dtype='float64')
        ranking_top_store_tmp = np.zeros((options.nMem, self.N, len(ranking_top_m_list)))
        
        r_store = file.create_dataset('r_store', (options.nKeep,), dtype='float64')
        r_store_tmp = np.zeros((options.nMem,))
        
        f_store = file.create_dataset('f_store', (options.nKeep,self.N), dtype='float64')
        f_store_tmp = np.zeros((options.nMem,self.N))
        
        if self.n_fix:
            beta_store = file.create_dataset('beta_store', 
                                           (options.nKeep, self.data.n_fix), 
                                           dtype='float64')
            beta_store_tmp = np.zeros((options.nMem, self.data.n_fix))
            
        if self.n_rnd:
            mu_store = file.create_dataset('mu_store', 
                                         (options.nKeep, self.data.n_rnd), 
                                         dtype='float64')
            mu_store_tmp = np.zeros((options.nMem, self.data.n_rnd))
            
            sigma_store = file.create_dataset('sigma_store', 
                                            (options.nKeep, self.data.n_rnd), 
                                            dtype='float64')
            sigma_store_tmp = np.zeros((options.nMem, self.data.n_rnd))
            
            Sigma_store = file.create_dataset('Sigma_store', 
                                             (options.nKeep, self.data.n_rnd, self.data.n_rnd), 
                                             dtype='float64')
            Sigma_store_tmp = np.zeros((options.nMem, self.data.n_rnd, self.data.n_rnd))
        
        if self.data_bart is not None:
            avg_tree_acceptance_store = np.zeros((options.nIter,))
            avg_tree_depth_store = np.zeros((options.nIter,))
                
            variable_inclusion_props_store = file.create_dataset(
                'variable_inclusion_props_store', 
                (options.nKeep, self.data_bart.J), 
                dtype='float64')
            variable_inclusion_props_store_tmp = np.zeros((options.nMem, self.data_bart.J))              
        
        if self.mess:
            sigma_mess_store = file.create_dataset('sigma_mess_store', 
                                                 (options.nKeep,), 
                                                 dtype='float64')
            sigma_mess_store_tmp = np.zeros((options.nMem,))

            tau_store = file.create_dataset('tau_store', 
                                          (options.nKeep,), 
                                          dtype='float64')
            tau_store_tmp = np.zeros((options.nMem,))
            
            mh_tau_accept_store = np.zeros((options.nIter,))
        
        # Initialize parameters
        r = max(r_init - 0.5 + 1.0 * np.random.rand(), 0.25)
        
        if self.n_fix:
            beta = beta_init - 0.5 + 1 * np.random.rand(self.n_fix,)
            psi_fix = self.data.x_fix @ beta
        else:
            beta = None
            psi_fix = 0
            
        if self.n_rnd:
            mu = mu_init - 0.5 + 1 * np.random.rand(self.n_rnd,)
            Sigma = Sigma_init.copy()
            SigmaInv = np.linalg.inv(Sigma)
            a = np.random.gamma(1/2, 1/A**2)
            gamma = mu + (np.linalg.cholesky(Sigma) @ np.random.randn(self.n_rnd,self.N)).T
            psi_rnd = np.sum(self.data.x_rnd * gamma, axis=1)
        else:
            psi_rnd = 0
            
        if self.data_bart is not None:
            forest = bt.Forest(bart_options, self.data_bart)
            psi_bart = self.data_bart.unscale(forest.y_hat)
        else:
            forest = None
            psi_bart = 0
        
        if self.mess:
            sigma2 = np.sqrt(0.4)
            tau = -float(0.5 + np.random.rand())
            S = expm(tau * self.data.W)
            Omega_tilde = S.T @ S / sigma2         
            eps = np.sqrt(sigma2) * np.random.randn(self.data.N,)
            phi = np.linalg.solve(S, eps)
            mh_step = options.mh_step_initial
        else:
            sigma2 = None
            tau = None       
            S = None
            Omega_tilde = None
            phi = 0
            mh_step = None
  
        psi = psi_fix + psi_rnd + psi_bart + phi
        
        # MCMC sampling
        j = -1
        ll = 0
        sample_state = 'burn in'    
        
        for i in np.arange(options.nIter):
            # Update parameters
            omega, Omega, z = self._next_omega(r, psi)
            
            if self.n_fix:
                beta = self._next_beta(z, psi_rnd, psi_bart, phi, 
                                      Omega, beta_mu0, beta_Si0Inv)
                psi_fix = self.data.x_fix @ beta
                
            if self.n_rnd:
                gamma = self._next_gamma(z, psi_fix, psi_bart, phi, 
                                        omega, mu, SigmaInv)
                mu = self._next_mu(gamma, SigmaInv, mu_mu0, mu_Si0Inv)
                Sigma, SigmaInv = self._next_Sigma(gamma, mu, a, nu)
                a = self._next_a(SigmaInv, nu, A)
                psi_rnd = np.sum(self.data.x_rnd * gamma, axis=1) 
                
            if self.data_bart is not None:
                sigma_weights = np.sqrt(1/omega)
                f = z - psi_fix - psi_rnd - phi
                self.data_bart.update(f, sigma_weights)
                avg_tree_acceptance_store[i], avg_tree_depth_store[i] = \
                    forest.update(self.data_bart)
                psi_bart = self.data_bart.unscale(forest.y_hat)
                
            if self.mess:
                phi = self._next_phi(z, psi_fix, psi_rnd, psi_bart,
                                    Omega, Omega_tilde)
            
            psi = psi_fix + psi_rnd + psi_bart + phi 
            
            if self.mess:
                sigma2 = self._next_sigma2(phi, S, sigma2_b0, sigma2_c0)
                tau, S, mh_tau_accept_store[i] = self._next_tau(
                    tau, S, phi, sigma2, tau_mu0, tau_si0, mh_step
                    )
                Omega_tilde = S @ S.T / sigma2
            
            h = self._next_h(r, r0, b0, c0)
            L = self._next_L(self.data.y, r, self.F)
            r = self._next_r(r0, L, h, psi)
            
            # Adjust MH step size
            if self.mess and ((i+1) % options.mh_window) == 0:
                sl = slice(max(i+1-options.mh_window,0), i+1)
                mean_accept = mh_tau_accept_store[sl].mean()
                if mean_accept >= options.mh_target:
                    mh_step += options.mh_correct
                else:
                    mh_step -= options.mh_correct
                     
            # Display sampling progress
            if ((i+1) % options.disp) == 0:  
                if (i+1) > options.nBurn:
                    sample_state = 'sampling'
                verbose = f'Chain {chainID + 1}; iteration: {i + 1} ({sample_state})'
                if self.data_bart is not None:
                    sl = slice(max(i+1-100,0),i+1)
                    ravg_depth = np.round(np.mean(avg_tree_depth_store[sl]), 2)
                    ravg_acceptance = np.round(np.mean(avg_tree_acceptance_store[sl]), 2)
                    verbose += f'; avg. tree depth: {ravg_depth}; avg. tree acceptance: {ravg_acceptance}'
                if self.mess:
                    verbose += f'; avg. tau acceptance: {mh_tau_accept_store[sl].mean()}'
                print(verbose)
                sys.stdout.flush()
            
            # Store samples
            if (i+1) > options.nBurn:                  
                if ((i+1) % options.nThin) == 0:
                    j+=1
                    
                    lp_store_tmp[j,:] = self._nb_lpmf(self.data.y, psi, r)
                    lam = np.exp(psi + np.log(r))
                    lam_store_tmp[j,:] = lam
                    ranking_store_tmp[j,:], ranking_top_store_tmp[j,:,:] = self._rank(lam, ranking_top_m_list)
                    r_store_tmp[j] = r
                    f_store_tmp[j,:] = z - phi
                    
                    if self.n_fix:
                        beta_store_tmp[j,:] = beta
                        
                    if self.n_rnd:
                        mu_store_tmp[j,:] = mu
                        sigma_store_tmp[j,:] = np.sqrt(np.diag(Sigma))
                        Sigma_store_tmp[j,:,:] = Sigma
                        
                    if self.data_bart is not None:
                        variable_inclusion_props_store_tmp[j,:] = forest.variable_inclusion()
                        
                    if self.mess:
                        sigma_mess_store_tmp[j] = np.sqrt(sigma2)
                        tau_store_tmp[j] = tau
                
                # Write to storage
                if (j+1) == options.nMem:
                    l = ll
                    ll += options.nMem
                    sl = slice(l, ll)
                    
                    print(f'Storing chain {chainID + 1}')
                    sys.stdout.flush()
                    
                    lp_store[sl,:] = lp_store_tmp
                    lam_store[sl,:] = lam_store_tmp
                    ranking_store[sl,:] = ranking_store_tmp
                    ranking_top_store[sl,:,:] = ranking_top_store_tmp
                    r_store[sl] = r_store_tmp
                    f_store[sl,:] = f_store_tmp
                    
                    if self.n_fix:
                        beta_store[sl,:] = beta_store_tmp
                        
                    if self.n_rnd:
                        mu_store[sl,:] = mu_store_tmp
                        sigma_store[sl,:] = sigma_store_tmp
                        Sigma_store[sl,:,:] = Sigma_store_tmp
                        
                    if self.data_bart is not None:
                        variable_inclusion_props_store[sl,:] = variable_inclusion_props_store_tmp
                    
                    if self.mess:
                        sigma_mess_store[sl] = sigma_mess_store_tmp
                        tau_store[sl] = tau_store_tmp
                    
                    j = -1 
        
        if self.data_bart is not None:
            file.create_dataset('avg_tree_acceptance_store', data=avg_tree_acceptance_store)
            file.create_dataset('avg_tree_depth_store', data=avg_tree_depth_store)
            
    # Posterior summary methods
    @staticmethod
    def _posterior_summary(options, param_name, nParam, nParam2, verbose):
        """Returns summary of posterior draws of parameters of interest."""
        headers = ['mean', 'std. dev.', '2.5%', '97.5%', 'Rhat']
        q = (0.025, 0.975)
        nSplit = 2
        
        draws = np.zeros((options.nChain, options.nKeep, nParam, nParam2))
        for c in range(options.nChain):
            file = h5py.File(options.model_name + '_draws_chain' + str(c + 1) + '.hdf5', 'r')
            draws[c,:,:,:] = np.array(file[param_name + '_store']).reshape((options.nKeep, nParam, nParam2))
            
        mat = np.zeros((nParam * nParam2, len(headers)))
        post_mean = np.mean(draws, axis=(0,1))
        mat[:, 0] = np.array(post_mean).reshape((nParam * nParam2,))
        mat[:, 1] = np.array(np.std(draws, axis=(0,1))).reshape((nParam * nParam2,))
        mat[:, 2] = np.array(np.quantile(draws, q[0], axis=(0,1))).reshape((nParam * nParam2,))
        mat[:, 3] = np.array(np.quantile(draws, q[1], axis=(0,1))).reshape((nParam * nParam2,))
        
        m = int(options.nChain * nSplit)
        n = int(options.nKeep / nSplit)
        draws_split = np.zeros((m, n, nParam, nParam2))
        draws_split[:options.nChain,:,:,:] = draws[:,:n,:,:]
        draws_split[options.nChain:,:,:,:] = draws[:,n:,:,:]
        mu_chain = np.mean(draws_split, axis=1, keepdims=True)
        mu = np.mean(mu_chain, axis=0, keepdims=True)
        B = (n / (m - 1)) * np.sum((mu_chain - mu)**2, axis=(0,1))
        ssq = (1 / (n - 1)) * np.sum((draws_split - mu_chain)**2, axis=1)
        W = np.mean(ssq, axis=0)
        varPlus = ((n - 1) / n) * W + B / n
        Rhat = np.empty((nParam, nParam2)) * np.nan
        W_idx = W > 0
        Rhat[W_idx] = np.sqrt(varPlus[W_idx] / W[W_idx])
        mat[:,4] = np.array(Rhat).reshape((nParam * nParam2,))
            
        df = pd.DataFrame(mat, columns=headers) 
        if verbose:
            print(' ')
            print(f'{param_name}:')
            print(df)
        return df  
    
    @staticmethod
    def _posterior_mean(options, param_name, nParam, nParam2): 
        """Calculates mean of posterior draws of parameter of interest."""
        draws = np.zeros((options.nChain, options.nKeep, nParam, nParam2))
        for c in range(options.nChain):
            file = h5py.File(options.model_name + '_draws_chain' + str(c + 1) + '.hdf5', 'r')
            draws[c,:,:,:] = np.array(file[param_name + '_store']).reshape((options.nKeep, nParam, nParam2))
        post_mean = draws.mean(axis=(0,1))
        return post_mean
    
    def _posterior_fit(self, options, verbose):
        """Calculates LPPD and WAIC."""
        lp_draws = np.zeros((options.nChain, options.nKeep, self.N))
        for c in range(options.nChain):
            file = h5py.File(options.model_name + '_draws_chain' + str(c + 1) + '.hdf5', 'r')
            lp_draws[c,:,:] = np.array(file['lp' + '_store'])
        
        p_draws = np.exp(lp_draws)
        lppd = np.log(p_draws.mean(axis=(0,1))).sum()
        p_waic = lp_draws.var(axis=(0,1)).sum()
        waic = -2 * (lppd - p_waic)
        
        if verbose:
            print(' ')
            print(f'LPPD: {lppd}')
            print(f'WAIC: {waic}')
        return lppd, waic

    def estimate(
            self,
            options, bart_options,
            r0, b0, c0,
            beta_mu0, beta_Si0Inv,
            mu_mu0, mu_Si0Inv, nu, A,
            sigma2_b0, sigma2_c0, tau_mu0, tau_si0,
            r_init, beta_init, mu_init, Sigma_init,
            ranking_top_m_list):
        """Executes MCMC simulation for all chains and summarizes results."""
        
        # Set random seed
        np.random.seed(options.seed)
        
        # Create unique seeds for each chain
        chain_seeds = np.random.randint(0, 10000, options.nChain)
        
        # Start timer
        tic = time.time()
        
        # Run chains sequentially instead of in parallel when nChain > 1
        if options.nChain > 1:
            # Run each chain sequentially 
            for c in range(options.nChain):
                self._mcmc_chain(
                    c, 
                    options, 
                    bart_options,
                    r0, b0, c0,
                    beta_mu0, beta_Si0Inv,
                    mu_mu0, mu_Si0Inv, nu, A,
                    sigma2_b0, sigma2_c0, tau_mu0, tau_si0,
                    r_init, beta_init, mu_init, Sigma_init,
                    ranking_top_m_list,
                    seed=chain_seeds[c]
                )
        else:
            # Use Parallel for a single chain (mostly for compatibility)
            Parallel(n_jobs=1)(
                delayed(self._mcmc_chain)(
                    c, 
                    options, 
                    bart_options,
                    r0, b0, c0,
                    beta_mu0, beta_Si0Inv,
                    mu_mu0, mu_Si0Inv, nu, A,
                    sigma2_b0, sigma2_c0, tau_mu0, tau_si0,
                    r_init, beta_init, mu_init, Sigma_init,
                    ranking_top_m_list,
                    seed=chain_seeds[c]
                ) for c in range(options.nChain)
            )
            
        # End timer
        toc = time.time() - tic
        
        # Calculate posterior summaries
        post_r = self._posterior_summary(
            options, 'r', 1, 1, True
            ).set_index(pd.Index(['r']))
        
        post_mean_f = self._posterior_mean(
            options, 'f', self.N, 1
            )
        
        post_variable_inclusion_props = None
        if self.data_bart is not None:
            post_variable_inclusion_props = self._posterior_summary(
                options, 'variable_inclusion_props',
                self.data_bart.J, 1, True
                ).set_index(pd.Index(self.data_bart.var_names))
                
        post_beta = None
        if self.n_fix:
            post_beta = self._posterior_summary(
                options, 'beta', self.n_fix, 1, True
                ).set_index(pd.Index([f'beta[{j}]' for j in range(self.n_fix)]))
                
        post_mu = None
        post_sigma = None
        post_Sigma = None
        if self.n_rnd:
            post_mu = self._posterior_summary(
                options, 'mu', self.n_rnd, 1, True
                ).set_index(pd.Index(['mu[' + str(j) + ']' for j in range(self.n_rnd)]))
            
            post_sigma = self._posterior_summary(
                options, 'sigma', self.n_rnd, 1, True
                ).set_index(pd.Index(['sigma[' + str(j) + ']' for j in range(self.n_rnd)]))
                
            post_Sigma = self._posterior_summary(
                options, 'Sigma', self.n_rnd, self.n_rnd, True
                ).set_index(pd.Index(['Sigma[' + str(j) + ',' + str(k) + ']' 
                            for j in range(self.n_rnd) for k in range(self.n_rnd)]))
                
        post_sigma_mess = None
        post_tau = None
        if self.mess:
            post_sigma_mess = self._posterior_summary(
                options, 'sigma_mess', 1, 1, True
                ).set_index(pd.Index(['sigma_mess']))
                
            post_tau = self._posterior_summary(
                options, 'tau', 1, 1, True
                ).set_index(pd.Index(['tau']))                
                
        # Compute fitted values
        post_mean_lam = self._posterior_mean(
            options, 'lam', self.N, 1
            )
            
        residuals = self.data.y - post_mean_lam
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        rmsle = np.sqrt(np.mean((np.log1p(self.data.y) - np.log1p(post_mean_lam))**2))
            
        # Compute rankings
        post_mean_ranking = self._posterior_mean(
            options, 'ranking', self.N, 1
            )
            
        post_mean_ranking_top = np.zeros((self.N, len(ranking_top_m_list)))
        for c in range(options.nChain):
            file = h5py.File(options.model_name + '_draws_chain' + str(c + 1) + '.hdf5', 'r')
            post_mean_ranking_top += np.mean(np.array(file['ranking_top_store']), axis=0) / options.nChain
            
        # Compute goodness-of-fit statistics
        lppd, waic = self._posterior_fit(options, True)
            
        # Create results instance
        res = Results(
            options, bart_options, toc, 
            lppd, waic,
            post_mean_lam, 
            rmse, mae, rmsle,
            post_mean_ranking, 
            post_mean_ranking_top, ranking_top_m_list,
            post_r, post_mean_f, 
            post_beta,
            post_mu, post_sigma, post_Sigma,
            post_variable_inclusion_props,
            post_sigma_mess, post_tau
            )
            
        # Delete MCMC draws
        if options.delete_draws:
            for c in range(options.nChain):
                file_name = options.model_name + '_draws_chain' + str(c + 1) + '.hdf5'
                os.remove(file_name)        
        
        return res


#####################################
# Data Processing and Model Execution
#####################################

def load_and_prepare_data():
    """
    Load and prepare the Montreal intersection data for modeling
    """
    data = pd.read_csv("data/data_final.csv", sep=";")
    
    # Replace NaN values in ln_distdt with 0
    data['ln_distdt'] = data['ln_distdt'].fillna(0)
    
    # Filter where pi != 0
    data = data[data['pi'] != 0]
    
    # First correct borough names with encoding issues
    borough_corrections = {
        '?le-Bizard-Sainte-GeneviÞve': 'Île-Bizard-Sainte-Geneviève',
        'C¶te-Saint-Luc': 'Côte-Saint-Luc',
        'C¶te-des-Neiges-Notre-Dame-de-Graces': 'Côte-des-Neiges-Notre-Dame-de-Grâce',
        'MontrÚal-Est': 'Montréal-Est',
        'MontrÚal-Nord': 'Montréal-Nord',
        'Pointe-aux-Trembles-RiviÞres-des-Prairies': 'Rivière-des-Prairies-Pointe-aux-Trembles',
        'St-LÚonard': 'Saint-Léonard'
    }
    
    # Then group boroughs into zones
    borough_zones = {
        # Zone ouest
        'Kirkland': 'Zone ouest',
        'Beaconsfield': 'Zone ouest',
        'Île-Bizard-Sainte-Geneviève': 'Zone ouest',
        'Pierrefonds-Roxboro': 'Zone ouest',
        'Dollard-des-Ormeaux': 'Zone ouest',
        'Dorval': 'Zone ouest',
        
        # Zone est
        'Rivière-des-Prairies-Pointe-aux-Trembles': 'Zone est',
        'Montréal-Est': 'Zone est',
        'Anjou': 'Zone est',
        
        # Zone centre
        'Outremont': 'Zone centre',
        'Mont-Royal': 'Zone centre',
        
        # Zone sud
        'Sud-Ouest': 'Zone sud',
        'Côte-Saint-Luc': 'Zone sud',
        'Verdun': 'Zone sud',
        'Lasalle': 'Zone sud',
        'Lachine': 'Zone sud',
        
        # Zone centre-sud
        'Côte-des-Neiges-Notre-Dame-de-Grâce': 'Zone centre-sud',
        'Hampstead': 'Zone centre-sud',
        'Westmount': 'Zone centre-sud'
    }
    
    # Apply corrections if borough column exists
    if 'borough' in data.columns:
        # First fix encoding issues
        data['borough'] = data['borough'].replace(borough_corrections)
        # Then group into zones
        data['borough'] = data['borough'].replace(borough_zones)
        
    # Include existing spatial columns
    spatial_cols = ['x', 'y']
    
    # Base features (non-spatial)
    feature_cols = ['ln_fi', 'ln_fri', 'ln_fli', 'ln_pi', 'traffic_10000', 'ln_cti', 'ln_cli', 'ln_cri',"ln_distdt",
                    'total_lane', 'avg_crossw', 'tot_road_w', 'tot_crossw',
                    'commercial', 'number_of_', 'of_exclusi', 'curb_exten', 'median', 'all_pedest', 'half_phase', 'new_half_r',
                    'any_ped_pr', 'ped_countd', 'lt_restric', 'lt_prot_re', 'lt_protect', 'any_exclus', 'all_red_an', 'green_stra',
                    'parking', 'north_veh', 'north_ped', 'east_veh', 'east_ped', 'south_veh', 'south_ped', 'west_veh', 'west_ped']
    
    # Check which columns actually exist in the data
    feature_cols = [col for col in feature_cols if col in data.columns]
    spatial_cols = [col for col in spatial_cols if col in data.columns]
    
    # Combine base features with spatial variables
    X_base = data[feature_cols].fillna(0)
    X_spatial = data[spatial_cols].fillna(0)
    
    # Apply quadratic transformation to specified variables
    for col in ['ln_cti', 'ln_cri', 'ln_cli', 'ln_pi', 'ln_fri', 'ln_fli', 'ln_fi']:
        if col in X_base.columns:
            X_base[f'{col}_squared'] = X_base[col] ** 2

    # Instead of creating random effects for borough, create dummy variables
    X_rnd = None  # We won't use random effects anymore
    
    # Create dummy variables for borough if it exists
    borough_dummies = None
    if 'borough' in data.columns:
        # Create dummies but drop the first category to avoid the dummy variable trap
        borough_dummies = pd.get_dummies(data['borough'], prefix='borough', drop_first=True)
        # Add borough dummies to X_features
        X_features = pd.concat([X_base, X_spatial, borough_dummies], axis=1)
    else:
        X_features = pd.concat([X_base, X_spatial], axis=1)
    
    return {
        'X': X_features, 
        'X_non_spatial': X_base,
        'X_spatial': X_spatial,
        'y': data['acc'], 
        'int_no': data['int_no'],
        'pi': data['pi'],
        'X_rnd': None,  # No random effects
        'borough_codes': None,  # No borough codes needed
        'borough_dummies': borough_dummies  # Return borough dummies separately
    }


def create_spatial_weight_matrix(X_spatial):
    """
    Create a spatial weight matrix based on distances between coordinates.
    Uses GPU acceleration if available.
    """
    # Extract coordinates
    coords = X_spatial[['x', 'y']].values
    print(f"Coordinates shape: {coords.shape}")
    
    # Use GPU if available
    if torch.cuda.is_available():
        coords_tensor = torch.tensor(coords, device='cuda', dtype=torch.float32)
        
        # Calculate pairwise distances on GPU
        n = len(coords_tensor)
        distances = torch.zeros((n, n), device='cuda')
        
        # More efficient batched calculation to avoid potential OOM errors
        batch_size = 100  # Adjust based on GPU memory
        for i in range(0, n, batch_size):
            end_idx = min(i + batch_size, n)
            batch_coords = coords_tensor[i:end_idx]
            
            # Calculate distances for this batch to all other points
            diff = batch_coords.unsqueeze(1) - coords_tensor.unsqueeze(0)
            batch_distances = torch.sqrt((diff ** 2).sum(dim=2))
            distances[i:end_idx] = batch_distances
        
        # Create weight matrix based on distance bands
        W = torch.zeros((n, n), device='cuda')
        
        # Define weights based on adjacency orders
        W[(distances > 0) & (distances <= 250)] = 1.0
        W[(distances > 250) & (distances <= 500)] = 0.5
        W[(distances > 500) & (distances <= 1000)] = 0.25
        
        # Row-normalize the weight matrix
        row_sums = W.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        W = W / row_sums
        
        # Convert back to numpy
        W = W.cpu().numpy()
    else:
        # Original CPU implementation
        distances = squareform(pdist(coords, 'euclidean'))
        n = len(X_spatial)
        W = np.zeros((n, n))
        # Define weights based on adjacency orders
        W[(distances > 0) & (distances <= 250)] = 1.0
        W[(distances > 250) & (distances <= 500)] = 0.5
        W[(distances > 500) & (distances <= 1000)] = 0.25
        
        row_sums = W.sum(axis=1)
        W[row_sums > 0] = W[row_sums > 0] / row_sums[row_sums > 0].reshape(-1, 1)
    
    return W


def select_features_with_lasso(X, y, borough_dummies=None):
    """
    Select features using Lasso regression
    Always keeps specified important features and borough dummies
    """
    # List of features to always keep
    important_features = ['ln_cti', 'ln_cri', 'ln_cli', 'ln_pi', 'ln_fri', 'ln_fli', 'ln_fi', 
                         'ln_cti_squared', 'ln_cri_squared', 'ln_cli_squared', 'ln_pi_squared', 
                         'ln_fri_squared', 'ln_fli_squared', 'ln_fi_squared']
    
    # Add borough dummy columns to important features if they exist
    if borough_dummies is not None:
        important_features.extend(borough_dummies.columns.tolist())
    
    # Filter to only include features that exist in the dataset
    important_features = [f for f in important_features if f in X.columns]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lasso = LassoCV(cv=5, random_state=42, max_iter=500000, tol=1e-4)
    lasso.fit(X_scaled, y)
    
    # Get features selected by Lasso
    lasso_selected = X.columns[lasso.coef_ != 0].tolist()
    
    # Combine Lasso-selected features with important features (avoiding duplicates)
    selected_features = list(set(lasso_selected + important_features))
    
    print(f"Selected {len(selected_features)} features:")
    print(f"  - Always included: {important_features}")
    print(f"  - Lasso selected: {[f for f in lasso_selected if f not in important_features]}")
    
    return selected_features


def run_nb_model(X_train, y_train, X_test, y_test, pi_train, W_train):
    """
    Run Negative Binomial model with fixed parameters, random effects for borough, and spatial correlation
    """
    # Convert to numpy arrays
    X_train_np = X_train.values.astype(np.float64)
    y_train_np = y_train.values.astype(np.int64)
    
    # Create Data object for NegativeBinomial model
    nb_data = Data(y=y_train_np, x_fix=X_train_np, x_rnd=None, W=W_train)
    
    # Initialize model
    nb_model = NegativeBinomial(nb_data, data_bart=None)
    
    # Define model options - reduced settings for faster computation
    options = Options(
        model_name='fixed_random',
        nChain=2, nBurn=1000, nSample=1000, nThin=2, 
        mh_step_initial=0.1, mh_target=0.3, mh_correct=0.01, mh_window=50,
        disp=100, delete_draws=False, seed=42
    )
    
    # Prior parameters
    n_fix = X_train_np.shape[1]
    r0 = 1e-2
    b0 = 1e-2
    c0 = 1e-2
    beta_mu0 = np.zeros(n_fix)
    beta_Si0Inv = 1e-2 * np.eye(n_fix)
    
    # Random effects parameters (for borough)
    mu_mu0 = np.zeros(0)
    mu_Si0Inv = np.eye(0)
    nu = 3  # Degrees of freedom for inverse-Wishart
    A = np.ones(0)
    
    # Spatial parameters
    sigma2_b0 = 1e-2
    sigma2_c0 = 1e-2
    tau_mu0 = 0
    tau_si0 = 2
    
    # Initialize parameters
    r_init = 5.0
    beta_init = np.zeros(n_fix)
    mu_init = np.zeros(0)
    Sigma_init = np.eye(0)
    
    # Define ranking thresholds
    ranking_top_m_list = [10, 25, 50, 100]
    
    # Run model estimation
    try:
        results = nb_model.estimate(
            options, None,  # No BART options
            r0, b0, c0,
            beta_mu0, beta_Si0Inv,
            mu_mu0, mu_Si0Inv, nu, A,
            sigma2_b0, sigma2_c0, tau_mu0, tau_si0,
            r_init, beta_init, mu_init, Sigma_init,
            ranking_top_m_list
        )
        
        # Predict for test data
        X_test_np = X_test.values.astype(np.float64)
        
        # Handle potential NaN values in beta_mean
        beta_mean = results.post_beta['mean'].values
        if np.isnan(beta_mean).any():
            print("Warning: NaN values detected in beta coefficients. Using zeros instead.")
            beta_mean = np.zeros_like(beta_mean)
        
        # Get random effects mean if available
        mu_mean = None
        if hasattr(results, 'post_mu') and results.post_mu is not None:
            mu_mean = results.post_mu['mean'].values[0] if results.post_mu.shape[0] > 0 else 0
            
        # For negative binomial, calculate lambda = exp(X*beta) * r
        r_mean = results.post_r['mean'].values[0]
        if np.isnan(r_mean):
            print("Warning: NaN value detected in r parameter. Using 1.0 instead.")
            r_mean = 1.0
            
        # Fixed effects contribution
        psi_test = X_test_np @ beta_mean
        
        # Random effects contribution (if available)
        # For test set predictions, we use the mean random effect across all boroughs
        if mu_mean is not None:
            psi_test += mu_mean
        
        # Apply clipping to prevent overflow
        psi_test = np.clip(psi_test, -25, 25)
        
        # Calculate expected value
        lam_test = np.exp(psi_test) * r_mean
        
        # Final safeguard against NaN or infinity values
        lam_test = np.nan_to_num(lam_test, nan=0.0, posinf=100, neginf=0)
        
        # Store predictions
        results.test_predictions = lam_test
        results.test_actual = y_test.values
        
    except Exception as e:
        print(f"Error in model estimation: {e}")
        results = None
        
    return results


def run_cross_validation(X_non_spatial, X_spatial, y, int_no, pi, borough_dummies=None, k=5):
    """
    Run k-fold cross-validation with the Negative Binomial model
    """
    # Create fold indices
    np.random.seed(42)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Initialize lists to store results
    results_list = []
    all_metrics = []
    combined_results = pd.DataFrame()
    
    # Make sure int_no is present in X_spatial
    if 'int_no' not in X_spatial.columns:
        X_spatial = X_spatial.copy()
        X_spatial['int_no'] = int_no.values
    
    # Print dataset shapes for debugging
    print(f"X_non_spatial shape: {X_non_spatial.shape}")
    print(f"X_spatial shape: {X_spatial.shape}")
    print(f"y shape: {y.shape}")
    print(f"int_no shape: {int_no.shape}")
    print(f"pi shape: {pi.shape}")
    if borough_dummies is not None:
        print(f"borough_dummies shape: {borough_dummies.shape}")
    
    # For each fold
    for i, (train_idx, test_idx) in enumerate(kf.split(X_non_spatial), 1):
        print(f"\n========== Fold {i} ==========")
        
        # Create training and test sets
        X_train_non_spatial = X_non_spatial.iloc[train_idx]
        y_train = y.iloc[train_idx]
        pi_train = pi.iloc[train_idx]
        X_test_non_spatial = X_non_spatial.iloc[test_idx]
        y_test = y.iloc[test_idx]
        pi_test = pi.iloc[test_idx]
        
        # Get spatial features for this fold
        X_train_spatial = X_spatial.iloc[train_idx]
        
        # Get borough dummies for this fold if available
        train_borough_dummies = None
        test_borough_dummies = None
        if borough_dummies is not None:
            train_borough_dummies = borough_dummies.iloc[train_idx]
            test_borough_dummies = borough_dummies.iloc[test_idx]
        
        # Select features using Lasso on non-spatial features
        selected_features = select_features_with_lasso(X_train_non_spatial, y_train, train_borough_dummies)
        X_train_selected = X_train_non_spatial[selected_features]
        X_test_selected = X_test_non_spatial[selected_features]
        
        # Combine selected features with spatial features and borough dummies
        X_train_combined = X_train_selected.copy()
        X_test_combined = X_test_selected.copy()
        
        if train_borough_dummies is not None:
            X_train_combined = pd.concat([X_train_combined, train_borough_dummies], axis=1)
            X_test_combined = pd.concat([X_test_combined, test_borough_dummies], axis=1)
        
        # Create spatial weight matrix for training data
        W_train = create_spatial_weight_matrix(X_train_spatial)
        
        # Run Negative Binomial model (without random effects)
        fold_results = run_nb_model(X_train_combined, y_train, X_test_combined, y_test, 
                                  pi_train, W_train)
        
        # Skip this fold if model failed
        if fold_results is None:
            print(f"Skipping fold {i} due to estimation errors")
            continue
        
        # Store results
        results_list.append(fold_results)
        
        # Get predictions for test set
        y_pred = fold_results.test_predictions
        
        # Combine results for this fold
        fold_results_df = pd.DataFrame({
            'int_no': int_no.iloc[test_idx],
            'true_counts': y_test,
            'pred_counts': y_pred,
            'true_rate': y_test / pi_test,
            'pred_rate': y_pred / pi_test,
            'pi': pi_test
        })
        combined_results = pd.concat([combined_results, fold_results_df], ignore_index=True)
        
        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Add evaluation metrics to list
        all_metrics.append({
            'fold': i,
            'mae': mae,
            'mse': mse,
            'rmse': rmse
        })
        
        # Display fold results
        print(f"Fold {i} - MAE: {mae}")
        print(f"Fold {i} - MSE: {mse}")
        print(f"Fold {i} - RMSE: {rmse}")
    
    # Convert list to DataFrame after the loop
    all_metrics = pd.DataFrame(all_metrics)
    
    # Calculate averages
    avg_metrics = all_metrics[['mae', 'mse', 'rmse']].mean()
    
    # Display results
    print("\n========== Results by fold ==========")
    print(all_metrics)
    
    print("\n========== Average metrics ==========")
    print(f"Average MAE: {avg_metrics['mae']}")
    print(f"Average MSE: {avg_metrics['mse']}")
    print(f"Average RMSE: {avg_metrics['rmse']}")
    
    # Create visualization of fold results
    results_path = "results"
    os.makedirs(results_path, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    all_metrics_long = pd.melt(all_metrics, id_vars=['fold'], value_vars=['mae', 'mse', 'rmse'])
    
    # Create a multi-panel plot
    g = sns.FacetGrid(all_metrics_long, col='variable', col_wrap=3, height=4)
    g.map(sns.barplot, 'fold', 'value', order=sorted(all_metrics_long['fold'].unique()))
    g.set_axis_labels('Fold', 'Value')
    g.set_titles('{col_name}')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(results_path, "nb_model_cv_metrics.png"))
    
    # Save combined results
    combined_results.to_csv(os.path.join(results_path, "nb_model_combined_cv_results.csv"), index=False)
    
    # Create and save combined intersection ranking
    combined_rankings = combined_results[['int_no', 'pred_counts']].sort_values(by='pred_counts', ascending=False)
    combined_rankings['ranking'] = range(1, len(combined_rankings) + 1)
    combined_rankings.to_csv(os.path.join(results_path, "nb_model_combined_intersections_cv_ranking.csv"), index=False)
    
    # ADDED: Generate spatial correlation plot using the last fold's model results
    # Since we want a single spatial correlation plot after all CV folds are complete
    if results_list and len(results_list) > 0:
        last_model = results_list[-1]  # Use the last model from cross-validation
        
        # Create a spatial DataFrame with the results
        spatial_df = pd.DataFrame({
            'int_no': int_no,
            'x': X_spatial['x'],
            'y': X_spatial['y'],
            'pred_counts': combined_results['pred_counts']
        })
        
        # Create the last fold's W matrix
        last_fold_idx = list(kf.split(X_non_spatial))[-1][0]  # Get train indices from last fold
        last_fold_spatial = X_spatial.iloc[last_fold_idx]
        W_last = create_spatial_weight_matrix(last_fold_spatial)
        
        # Generate the spatial correlation network plot
        print("\nGenerating spatial correlation network from cross-validation results...")
        plot_spatial_correlation_network(
            spatial_df,
            last_model,
            W_last,
            results_path,
            correlation_threshold=0.2  # Slightly lower threshold for better visualization
        )
    
    return {
        'results_per_fold': results_list,
        'metrics_per_fold': all_metrics,
        'average_metrics': avg_metrics,
        'combined_results': combined_results
    }


def plot_spatial_correlation_network(spatial_results, results, W, results_path, correlation_threshold=0.3):
    """
    Create a network visualization showing the estimated spatial correlation between intersections
    based on the model's tau and sigma_mess parameters.
    
    Args:
        spatial_results: DataFrame containing int_no, x, y, and pred_counts
        results: Results object from the Negative Binomial model
        W: The spatial weight matrix used in modeling
        results_path: Path to save the visualization
        correlation_threshold: Minimum correlation to display a connection
    """
    try:
        import networkx as nx
        from scipy.linalg import expm
        print("Creating estimated spatial correlation network visualization...")
        
        # Check if spatial parameters are available
        if results.post_tau is None or results.post_sigma_mess is None:
            print("Cannot create correlation network: spatial parameters not available in model results")
            return
            
        # Extract spatial parameters
        tau = results.post_tau['mean'].values[0]
        sigma_mess = results.post_sigma_mess['mean'].values[0]
        print(f"Using estimated spatial parameters: tau={tau:.4f}, sigma_mess={sigma_mess:.4f}")
        
        # Calculate spatial correlation matrix
        S = expm(tau * W)
        sigma2 = sigma_mess**2
        precision_matrix = S.T @ S / sigma2
        
        # Convert precision to covariance
        try:
            covariance_matrix = np.linalg.inv(precision_matrix)
        except np.linalg.LinAlgError:
            covariance_matrix = np.linalg.pinv(precision_matrix)
        
        # Convert to correlation matrix
        diag_values = np.abs(np.diag(covariance_matrix))  # Use absolute values to avoid issues
        D_sqrt_inv = np.diag(1.0 / np.sqrt(diag_values))
        correlation_matrix = np.clip(D_sqrt_inv @ covariance_matrix @ D_sqrt_inv, -1, 1)
        
        # Create network graph
        G = nx.Graph()
        
        # Get subset of intersections
        int_indices = spatial_results.index.tolist()[:correlation_matrix.shape[0]]
        
        # Add nodes (intersections)
        for i, row in spatial_results.iterrows():
            if i < len(int_indices):
                G.add_node(i, pos=(row['x'], row['y']), 
                          accidents=row['pred_counts'],
                          int_no=row['int_no'])
        
        # Add edges for correlated intersections
        n = min(len(int_indices), correlation_matrix.shape[0])
        edge_weights = []
        
        for i in range(n):
            for j in range(i+1, n):
                corr = abs(correlation_matrix[i, j])
                if corr > correlation_threshold:
                    G.add_edge(i, j, weight=corr)
                    edge_weights.append(corr)
        
        if not edge_weights:
            print(f"No correlations above threshold {correlation_threshold}. Try lowering the threshold.")
            return
            
        # Create visualization
        fig, ax = plt.subplots(figsize=(15, 12))
        pos = nx.get_node_attributes(G, 'pos')
        
        # Draw nodes (small and transparent)
        nx.draw_networkx_nodes(G, pos, 
                              node_size=10,
                              node_color='lightgray',
                              alpha=0.3,
                              ax=ax)
        
        # Draw edges with color based on correlation strength
        edge_weights_list = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, 
                              width=2.0,
                              alpha=0.7,
                              edge_color=edge_weights_list, 
                              edge_cmap=plt.cm.YlOrRd, 
                              ax=ax)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, 
                                  norm=plt.Normalize(vmin=correlation_threshold, 
                                                    vmax=max(edge_weights)))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Estimated Spatial Correlation (absolute value)')
        
        plt.title(f'Estimated Spatial Correlation Network (tau={tau:.3f}, sigma={sigma_mess:.3f})')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, 'estimated_correlation_network.png'), dpi=300)
        
        print("Estimated spatial correlation network visualization saved.")
        
    except ImportError:
        print("NetworkX library not available. Install with 'pip install networkx' to enable network visualization.")
    except Exception as e:
        print(f"Error creating spatial correlation network: {e}")
        import traceback
        traceback.print_exc()


def report_model_coefficients(results, X_columns, results_path="results"):
    """
    Create a clear, formatted table of model coefficients and save to CSV.
    Also creates a coefficient plot with confidence intervals.
    
    Args:
        results: Model results object containing posterior samples
        X_columns: Column names for the design matrix
        results_path: Path to save results
    """
    if not hasattr(results, 'post_beta') or results.post_beta is None:
        print("No coefficient information available in model results")
        return
    
    # Create coefficient table
    coef_table = results.post_beta.copy()
    coef_table.index = X_columns
    
    # Rename columns for clarity
    coef_table = coef_table.rename(columns={
        'mean': 'Estimate',
        'std. dev.': 'Std. Error',
        '2.5%': 'Lower 95% CI',
        '97.5%': 'Upper 95% CI'
    })
    
    # Add a column for significance
    coef_table['Significant'] = ((coef_table['Lower 95% CI'] > 0) & 
                                (coef_table['Upper 95% CI'] > 0)) | \
                               ((coef_table['Lower 95% CI'] < 0) & 
                                (coef_table['Upper 95% CI'] < 0))
    
    # Sort by absolute effect size
    coef_table = coef_table.sort_values(by='Estimate', key=abs, ascending=False)
    
    # Print nicely formatted table to console
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.width', 120)      # Wider display
    pd.set_option('display.precision', 4)    # Show 4 decimal places
    
    print("\n=== Model Coefficients ===")
    print(coef_table[['Estimate', 'Std. Error', 'Lower 95% CI', 'Upper 95% CI', 'Significant']])
    
    # Save to CSV
    os.makedirs(results_path, exist_ok=True)
    coef_table.to_csv(os.path.join(results_path, "model_coefficients.csv"))
    print(f"Coefficient table saved to {os.path.join(results_path, 'model_coefficients.csv')}")
    
    # Create coefficient plot
    plt.figure(figsize=(12, max(8, len(coef_table) * 0.3)))
    
    # Plot coefficients and confidence intervals
    y_pos = np.arange(len(coef_table))
    
    # Instead of using a list of colors in errorbar, plot each point separately
    for i, (idx, row) in enumerate(coef_table.iterrows()):
        color = 'darkred' if row['Significant'] else 'darkblue'
        plt.errorbar(
            x=row['Estimate'],
            y=i,
            xerr=[[row['Estimate'] - row['Lower 95% CI']], 
                  [row['Upper 95% CI'] - row['Estimate']]],
            fmt='o',
            capsize=5,
            color=color,
            alpha=0.7
        )
    
    # Add vertical line at zero
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add labels and title
    plt.yticks(y_pos, coef_table.index)
    plt.xlabel('Coefficient Estimate')
    plt.title('Model Coefficients with 95% Confidence Intervals')
    plt.grid(axis='x', alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "coefficient_plot.png"), dpi=300)
    print(f"Coefficient plot saved to {os.path.join(results_path, 'coefficient_plot.png')}")


def main():
    """Main function to run the entire workflow"""
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Load and prepare data
    print("Loading and preparing data...")
    data = load_and_prepare_data()
    
    # First fit the full model to the entire dataset
    print("\n========== Full Model on Entire Dataset ==========")
    
    # Select features using Lasso on non-spatial features
    selected_features = select_features_with_lasso(data['X_non_spatial'], data['y'], data.get('borough_dummies'))
    X_selected = data['X_non_spatial'][selected_features]
    
    # Combine selected features with spatial features and borough dummies
    X_combined = X_selected.copy()
    
    if data.get('borough_dummies') is not None:
        X_combined = pd.concat([X_combined, data.get('borough_dummies')], axis=1)
    
    # Create spatial weight matrix
    W = create_spatial_weight_matrix(data['X_spatial'])
    
    # Run Negative Binomial model on full dataset
    full_model_results = run_nb_model(X_combined, data['y'], X_combined, data['y'], 
                                    data['pi'], W)
    
    # Report model information criteria
    if full_model_results is not None:
        print("\n=== Full Model Results ===")
        if hasattr(full_model_results, 'lppd'):
            print(f"LPPD: {full_model_results.lppd:.2f}")
        if hasattr(full_model_results, 'waic'):
            print(f"WAIC: {full_model_results.waic:.2f}")
        
        # Report coefficients in a clear, formatted table
        report_model_coefficients(full_model_results, X_combined.columns)
        
        # Report spatial parameters
        if hasattr(full_model_results, 'post_sigma_mess') and full_model_results.post_sigma_mess is not None:
            print("\nSpatial Parameters:")
            print(f"sigma_mess (mean): {full_model_results.post_sigma_mess['mean'].values[0]:.4f}")
        if hasattr(full_model_results, 'post_tau') and full_model_results.post_tau is not None:
            print(f"tau (mean): {full_model_results.post_tau['mean'].values[0]:.4f}")
        
        # Report negative binomial dispersion parameter
        if hasattr(full_model_results, 'post_r') and full_model_results.post_r is not None:
            print(f"\nNegative Binomial Dispersion Parameter (r):")
            print(f"r (mean): {full_model_results.post_r['mean'].values[0]:.4f}")
    
    # Run cross-validation for predictive performance evaluation
    print("\nRunning cross-validation to evaluate predictive performance...")
    cv_results = run_cross_validation(
        data['X_non_spatial'], 
        data['X_spatial'], 
        data['y'], 
        data['int_no'], 
        data['pi'], 
        data.get('borough_dummies'), 
        k=5
    )
    
    print("\nProcessing complete. Results saved to 'results' directory.")
    return full_model_results, cv_results


# Run the main script
if __name__ == "__main__":
    full_results, cv_results = main()
