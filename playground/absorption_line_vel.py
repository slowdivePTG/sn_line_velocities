import matplotlib as mpl
import corner
import emcee
from numpy.core.fromnumeric import mean
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import norm
import numpy as np
import warnings
import glob
from dust_extinction import calALambda
from logging import raiseExceptions
import sys
sys.path.append('../../tools/')


mpl.rcParams['text.usetex'] = True  # only when producing plots for publication
mpl.rcParams['font.family'] = 'times new roman'
mpl.rcParams['font.size'] = '25'
mpl.rcParams['xtick.labelsize'] = '20'
mpl.rcParams['ytick.labelsize'] = '20'


##################### SpectrumSN class ##########################


class SpectrumSN(object):
    '''1D optical spectrum

    Attributes
    ----------
    spec_name : str
        spectrum name based on the filename
        source name + date + instrument

    fl : array_like
        flux (in arbitrary units)

    wv_rf : array_like
        wavelength [angstrom] in the host galaxy's rest frame:
        wv_rf = wavelength / (1 + z)

    fl_unc : array_like
        uncertainty in flux (in arbitrary units)

    Methods
    -------
    plot_line_region(blue_edge, red_edge) :
        plot the spectrum in the line region

    get_flux_at_lambda(lambda_0, delta_l=50) :
        pget the flux and its uncertainty at some given wavelength
    '''

    def __init__(self, spec1D, z=0, SN_name='ZTF'):
        '''Constructor

        Parameters
        ----------
        spec1D : str
            the spectrum file (directory + filename)

        z : float (default=0)
            host galaxy redshift

        SN_name : str, default='ZTF'
            SN naming system (appeared in the datafile)
        '''

        spec_df = pd.read_csv(spec1D,
                              comment='#',
                              delim_whitespace=True,
                              header=None)

        wv = spec_df[0].values
        wv_rf = wv / (1 + z)
        fl = spec_df[1].values

        self.spec_name = spec1D[spec1D.find(
            SN_name):spec1D.find('.ascii')]

        try:
            if ('Keck' in spec1D) and (len(spec_df.columns) > 3):
                fl_unc = spec_df[3].values
            else:
                fl_unc = spec_df[2].values

                if 'P60' in spec1D or 'P200' in spec1D:
                    fl_unc **= .5
        except:
            warnings.warn("No flux uncertainty in the datafile!")
            # set relative uncertainty to be 10%
            fl_unc = np.ones_like(fl) * 1e-1 * np.median(fl)

        # make sure flux measurements are positive
        self.fl = fl[fl > 0]
        self.wv_rf = wv_rf[fl > 0]
        self.fl_unc = fl_unc[fl > 0]

    def plot_line_region(self, blue_edge, red_edge):
        '''Plot the spectrum in the line region

        Parameters
        ----------
        blue_edge, red_edge : float
            the wavelength [angstrom] (host galaxy frame) 
            at the blue/red edge

        Returns
        -------
        ax : matplotlib.axes
            the axes
        '''
        line_region = np.where(
            (self.wv_rf < red_edge) & (self.wv_rf > blue_edge))[0]

        plt.figure(figsize=(8, 6))
        plt.plot(self.wv_rf[line_region], self.fl[line_region])
        plt.errorbar(self.wv_rf[line_region], self.fl[line_region],
                     yerr=self.fl_unc[line_region], fmt='o')
        plt.tight_layout()
        return plt.gca()

    def get_flux_at_lambda(self, lambda_0, delta_l=50):
        '''Get the flux and uncertainty at some given wavelength

        Returns the mean and uncertainty of flux in the 
        wavelength range: [lambda_0 - delta_l, lambda_0 + delta_l]

        Parameters
        ----------
        lambda_0 : float
            the central wavelength [angstrom]

        delta_l : float, default=50
            the size of the wavelength interval [angstrom]

        Returns
        -------
        mean : float
            mean flux around the central wavelength

        std : float
            multiple measurements:
                standard deviation in flux around the central wavelength
            single measurement:
                flux uncertainty given
        '''

        region = np.where(np.abs(self.wv_rf - lambda_0) < delta_l)[0]
        try:
            if len(region) == 0:
                raise IndexError('No data within this range!')
            elif len(region) == 1:
                warnings.warn('Too few points within the wavelength range!')
                return(self.fl[region[0]], self.fl_unc[region[0]])
            else:
                if len(region) <= 5:
                    warnings.warn(
                        '<=5 points within the wavelength range!')
                mean = np.sum(self.fl[region] / self.fl_unc[region]**2)\
                    / np.sum(1 / self.fl_unc[region]**2)
                std = np.nanmin(self.fl_unc[region])
                #std = np.std(self.fl[region], ddof=1)
            return (mean, std)
        except IndexError as e:
            repr(e)
            return None, None


class SpectrumSN_Lines(SpectrumSN):
    '''A set of measurements on different absorption lines

    Attributes
    ----------
    Spec1D : str
        the spectrum file (directory + filename)

    z : float
        host galaxy redshift

    line : dict
        a dictionary of various lines (AbsorbLine objects)

    Methods
    -------
    add_line(name, blue_edge, red_edge, lines=[])) :
        Add one (series of) absorption line(s)
    '''

    def __init__(self, spec1D, z):
        '''Constructor

        Parameters
        ----------
        spec1D : str
            the spectrum file (directory + filename)

        z : float, default=0
            host galaxy redshift
        '''

        super(SpectrumSN_Lines, self).__init__(spec1D, z)

        self.spec1D = spec1D
        self.z = z
        self.line = {}

    def add_line(self, name, blue_edge, red_edge, lines=[],
                 rel_strength=[], free_rel_strength=[]):
        '''Add one (series of) absorption line(s)

        Construct a new AbsorbLine object, and save it in self.line

        Parameters
        ----------
        name : str
            the name of the absorption line

        blue_edge, red_edge : float
            the wavelength [angstrom] (host galaxy frame) at 
            the blue/red edge

        lines : 2D array_like, default=[]
            wavelength of each absorption line

        rel_strength : array_like, default=[]
            the relative strength between each line in the series
            rel_strength = []: all lines are of the equal strength

        free_rel_strength : array_like, default=[]
            whether to set the relative strength of each line series as
            another free parameter in MCMC fit
        '''

        self.line[name] = AbsorbLine(
            self.spec1D, self.z, blue_edge, red_edge, lines, rel_strength, free_rel_strength)


class AbsorbLine(SpectrumSN):
    '''A (series of) absorption line(s) in a 1D optical spectrum

    Attributes
    ----------
    spec_name : str
        spectrum name based on the filename
        source name + date + instrument

    wv_line : array_like
        the wavelength range [angstrom] of the line 
        (host galaxy frame)

    rel_strength : array_like, default=[]
        the relative strength between each line in the series
        rel_strength = []: all lines are of the equal strength

    free_rel_strength : array_like, default=[]
        whether to set the relative strength of each line series as
        another free parameter in MCMC fit

    lambda_0 : float
        wavelength as a reference for velocity

    norm_fl : array_like
        flux normalized by the median value within the line region

    norm_fl_unc : array_like
        normalized flux uncertainty within the line region

    blue_fl, red_fl : list
        normalized flux and uncertainty at the blue/red edge

    vel_rf : array_like
        relative velocities of each point with respect 
        to the center of absorption

    blue_vel, red_vel : float
        relative velocity at the blue/red edge

    lines : 2D array_like, default=[]
            wavelength of each absorption line

    theta_LS : array_like
        best fit with least square methods, powered by
        scipy.optimize.minimize()

    chi2_LS : float
        the minimized residual with least square methods

    chi2_MCMC : float
        the minimized residual with MCMC

    theta_MCMC : array_like
        best fit with an MCMC sampler, approximated by median 
        values in the MCMC chains

    sig_theta_MCMC : array_like
        1 sigma uncertainty for each parameter, approximated 
        by the half the range of 16 and 84 percentile values 
        in the MCMC chains

    EW : float
        effective width [angstrom]

    sig_EW : float
        uncertainty in effective width [angstrom]

    Methods
    -------
    LS_estimator(guess=(1, 1, -10000, 15, -1000)) :
        Least square point estimation

    MCMC_sampler(mu_pvf=-1e4, var_pvf=1e7,
                 nwalkers=100, nsteps=1500, 
                 nburn=-1, initial=[],
                 normalize_unc=False) :
        An MCMC sampler

    plot_model(theta) :
        Plot the predicted absorption features

    '''

    def __init__(self, spec1D, z,
                 blue_edge, red_edge,
                 lines=[],
                 rel_strength=[],
                 free_rel_strength=[]):
        '''Constructor

        Parameters
        ----------
        spec1D: str
            the spectrum file (directory + filename)

        z: float, default=0
            host galaxy redshift

        blue_edge, red_edge: float
            the wavelength [angstrom] (host galaxy frame) at the blue/red edge

        lines: 2D array_like, default=[]
            the central wavelength(s) [angstrom] of this (series) of line(s)
            1D/2D for single components:
                Si II: [[6371.359, 6347.103]] or [6371.359, 6347.103] 
            2D for multiple components":
                different vel components of one element:
                    Ca II IRT [[8498.018, 8542.089, 8662.140], [8498.018, 8542.089, 8662.140]]    
                multiple elements:
                    He I/Fe II [[10830], [9998, 10500, 10863]]

        rel_strength : 2D array_like, default=[]
            the relative strength between each line in the series
            1D/2D for single component:
                Si II: [] or [[]] - empty for equal strength
            2D for multiple components:
                Ca II IRT [[], []]    
                He I/Fe II [[], [0.382, 0.239, 0.172]]

        free_rel_strength : array_like, default=[]
            whether to set the relative strength of each line series as
            another free parameter in MCMC fit
        '''

        super(AbsorbLine, self).__init__(spec1D, z)

        # line region
        line_region = np.where(
            (self.wv_rf < red_edge) & (self.wv_rf > blue_edge))[0]
        self.wv_line = self.wv_rf[line_region]
        # print('{:.0f} points within {:.2f} and {:.2f} angstroms.'.format(
        # len(line_region), blue_edge, red_edge))

        # normalized flux
        norm_fl = self.fl / np.nanmedian(self.fl)
        norm_fl_unc = self.fl_unc / np.nanmedian(self.fl)
        self.norm_fl = norm_fl[line_region]

        # check if there are points with relative uncertainty
        # two orders of magnitude lower than the median
        norm_fl_unc = norm_fl_unc[line_region]
        rel_unc = norm_fl_unc / self.norm_fl
        med_rel_unc = np.nanmedian(rel_unc)
        if rel_unc.min() < med_rel_unc / 1e2:
            warnings.warn("Some flux with extremely low uncertainty!")
        rel_unc[rel_unc < med_rel_unc / 1e2] = med_rel_unc
        self.norm_fl_unc = rel_unc * self.norm_fl

        # flux at each edge
        range_l = red_edge - blue_edge
        delta_l = min(100, range_l / 10)
        blue_fl = super(AbsorbLine, self).get_flux_at_lambda(
            blue_edge, delta_l=delta_l)
        red_fl = super(AbsorbLine, self).get_flux_at_lambda(
            red_edge, delta_l=delta_l)
        self.blue_fl = blue_fl / np.nanmedian(self.fl)
        self.red_fl = red_fl / np.nanmedian(self.fl)

        # velocity
        try:
            if len(lines[0]) > 0:
                pass
        except:
            lines = np.atleast_2d(lines)
            rel_strength = np.atleast_2d(rel_strength)
        lambda_0 = np.max(lines[0])
        vel_rf = velocity_rf(self.wv_rf, lambda_0)
        self.vel_rf = vel_rf[line_region]

        self.blue_vel = velocity_rf(blue_edge, lambda_0)
        self.red_vel = velocity_rf(red_edge, lambda_0)

        self.rel_strength = []
        self.lines = []

        for k in range(len(lines)):
            if len(rel_strength[k]) == 0:
                rs = np.ones_like(lines[k])
            else:
                rs = rel_strength[k]
            li = np.array(lines[k])[np.argsort(lines[k])]
            rs = np.array(rs) / rs[np.argmax(lines[k])]
            rs = np.array(rs)[np.argsort(lines[k])]
            self.lines.append(li)
            self.rel_strength.append(rs[:-1])
            if k == 0:
                self.lambda_0 = np.max(lines[k])
        if len(free_rel_strength) == 0:
            free_rel_strength = np.array([False] * len(self.rel_strength))
        self.free_rel_strength = free_rel_strength

        self.theta_LS = []
        self.chi2_LS = np.nan

        self.theta_MCMC = []
        self.sig_theta_MCMC = []

    def LS_estimator(self, guess=(1, 1, -10000, 15, -1000), plot_model=False):
        '''Least square point estimation

        Parameters
        ----------
        guess: tuple, default=(1, 1, -10000, 15, -1000)
            an initial guess for the fitting parameter theta

        plot_model : bool, default=False
            whether to plot the best fit result
        '''

        LS_res = minimize(
            neg_lnlike_gaussian_abs,
            guess,
            method='Powell',  # Powell method does not need derivatives
            args=(self.rel_strength, self.lambda_0,
                  self.blue_vel, self.red_vel, self.vel_rf, self.norm_fl,
                  self.lines, self.norm_fl_unc, 'chi2'))

        self.theta_LS = LS_res['x']
        ndim = len(self.theta_LS)
        self.chi2_LS = LS_res['fun']

        ndim = len(self.theta_LS)
        if plot_model:
            self.plot_model(self.theta_LS)

        print('LS estimation:')
        for k in range(ndim // 3):
            print('Velocity {}: {:.0f} km/s'.format(k +
                  1, self.theta_LS[2 + 3 * k]))

    def MCMC_NUTS_sampler(self,
                          vel_mean_mu=[], vel_mean_sig=[],
                          vel_var_lim=[2e1, 1e8],
                          A_lim=[-1e5, 1e5],
                          nburn=2000,
                          target_accept=0.8,
                          initial=[],
                          Plot_structure=False,
                          Plot_model=True,
                          Plot_mcmc=False,
                          Plot_tau=False):
        '''MCMC sampler with pymc

        Parameters
        ----------

        vel_mean_mu : array_like, default=[]
            means of the velocity priors

        vel_mean_sig : float, default=[]
            standard deviations of means of the velocity priors

        vel_var_lim : float, default=[2e1, 1e8]
            allowed range of the velocity dispersion

        A_lim : float, default=[-1e5, 1e5]
            allowed range of the amplitude

        nburn : int, default=2000
            number of "burn-in" steps for the MCMC chains

        target_accept : float in [0, 1], default=0.8 
             the step size is tuned such that we approximate this acceptance rate
             higher values like 0.9 or 0.95 often work better for problematic posteriors

        initial : array_like, default=[]
             initial values for the MCMC sampler

        Plot_model : bool, default=True
            whether to plot the model v.s. data

        Plot_mcmc : bool, default=False
            whether to plot the MCMC chains and corner plots

        Plot_tau : bool, default=False
            whether to plot the evolution of autocorrelation 
            time tau

        Returns
        -------
        trace : arviz.data.inference_data.InferenceData
            the samples drawn by the NUTS sampler
        GaussianProfile : pymc.model.Model
            the Bayesian model

        '''
        import pymc as pm
        import arviz as az
        import corner

        n_lines = len(self.lines)

        with pm.Model() as GaussProfile:
            # continuum fitting
            # model flux at the blue edge
            fl1 = pm.Normal(
                "blue_fl", mu=self.blue_fl[0], sigma=self.blue_fl[1])
            # model flux at the red edge
            fl2 = pm.Normal("red_fl", mu=self.red_fl[0], sigma=self.red_fl[1])

            # Gaussian profile
            # amplitude
            A = pm.Uniform("A", lower=A_lim[0],
                           upper=A_lim[1], shape=(n_lines,))
            # mean velocity
            if len(vel_mean_mu) == n_lines:
                v_mean = pm.Normal("v_mean", mu=vel_mean_mu,
                                   sigma=vel_mean_sig)
            else:
                raise IndexError(
                    'The number of the velocity priors does not match the number of lines')
            # velocity dispersion
            ln_vel_var_lim = np.log(np.array(vel_var_lim))
            ln_v_var = pm.Uniform(
                "ln_v_var", lower=ln_vel_var_lim[0], upper=ln_vel_var_lim[1], shape=(n_lines,))
            v_sig = pm.Deterministic("v_sig", np.exp(ln_v_var / 2))
            theta = [fl1, fl2]
            for k in range(n_lines):
                theta += [v_mean[k], ln_v_var[k], A[k]]
            # relative intensity of lines
            rel_strength = []
            ratio_index = []
            for k, free in enumerate(self.free_rel_strength):
                if free:
                    ratio_index.append(k)
                    log_ratio_0 = np.log10(self.rel_strength[k])
                    log_rel_strength_k = pm.Normal(
                        f"log_ratio_{k}", mu=log_ratio_0, sigma=0.1)
                    rel_strength.append(pm.Deterministic(
                        f"ratio_{k}", 10**log_rel_strength_k))
                else:
                    rel_strength.append(self.rel_strength[k])
                # equivalent width
                EW_k = pm.Deterministic(f'EW_{k}', -A[k] /
                                        (self.red_vel - self.blue_vel) /
                                        ((self.red_fl[0] + self.blue_fl[0]) / 2) *
                                        (self.wv_line[-1] - self.wv_line[0]) *
                                        (pm.math.sum(rel_strength[k]) + 1))

            # flux expectation
            mu = pm.Deterministic("mu", flux_gauss(theta, rel_strength, self.lambda_0,
                                                   self.blue_vel, self.red_vel, self.vel_rf,
                                                   self.lines))

            # uncertainty normalization
            typical_unc = np.median(self.norm_fl_unc)
            sigma_0 = pm.HalfCauchy("sigma_0", beta=typical_unc)
            sigma = pm.Deterministic(
                "sigma", (sigma_0**2 + self.norm_fl_unc**2)**.5)

            Flux = pm.Normal("Flux", mu=mu, sigma=sigma, observed=self.norm_fl)

        if Plot_structure:
            pm.model_to_graphviz(GaussProfile)
            plt.show()

        # initialization
        if len(initial) == 0:
            initial = self.theta_LS

        start = {}
        start['blue_fl'], start['red_fl'] = self.blue_fl[0], self.red_fl[0]
        start['v_mean'] = initial[2::3]
        start['ln_v_var'] = initial[3::3]
        start['A'] = initial[4::3]
        start['sigma_0'] = 1e-3
        for k, free in enumerate(self.free_rel_strength):
            if free:
                start[f'ratio_{k}'] = self.rel_strength[k]
                start[f'log_ratio_{k}'] = np.log10(self.rel_strength[k])

        with GaussProfile:
            trace = pm.sample(return_inferencedata=True,
                              initvals=start,
                              target_accept=target_accept,
                              tune=nburn)

        var_names_summary = ["v_mean", "v_sig", "A", "sigma_0"]
        for k in ratio_index:
            var_names_summary.append(f'ratio_{k}')
        for k in range(n_lines):
            var_names_summary.append(f'EW_{k}')
        summary = az.summary(trace, var_names=var_names_summary,
                             stat_focus="mean", round_to=3, hdi_prob=0.68)
        print(summary)

        all = az.summary(trace, kind="stats")
        theta = [all['mean']['blue_fl'], all['mean']['red_fl']]
        for k in range(n_lines):
            theta.append(all['mean'][f'v_mean[{k}]'])
            theta.append(all['mean'][f'ln_v_var[{k}]'])
            theta.append(all['mean'][f'A[{k}]'])
        self.theta_MCMC = theta

        if Plot_model:
            self.plot_model(theta)

        if Plot_mcmc:
            # by default, show the mean velocity, velocity dispersion, and line ratios
            var_names_plot = ["v_mean", "v_sig", "A"]
            for k in ratio_index:
                var_names_plot.append(f'ratio_{k}')
            corner.corner(trace, var_names=var_names_plot)

        return trace, GaussProfile

    def MCMC_sampler(self,
                     mu_prior=[], var_prior=[],
                     vel_flat=[-1e5, 0],
                     var_max=1e8,
                     nwalkers=100, max_nsteps=50000,
                     nburn=-1, thin=-1, initial=[],
                     normalize_unc='None',
                     Plot_model=True,
                     Plot_mcmc=False,
                     Plot_tau=False):
        from warnings import warn
        warn('The function MCMC_sampler() is deprecated. Use MCMC_NUTS_sampler() instead',
             DeprecationWarning, stacklevel=2)
        '''MCMC sampler with emcee

        Parameters
        ----------

        mu_prior : array_like, default=[]
            means of the profile prior

        var_prior : float, default=[]
            variances of the profile prior

        vel_flat : list, default=[-1e5, 0]
            the range of the flat velocity prior

        var_max : float, default=1e8
            maximum var allowed in a line profile

        nwalkers : int, default=100
            number of MCMC sampler walkers

        max_nsteps : int, default=50000
            maximum MCMC chain length

        nburn : int, default=-1
            number of "burn-in" steps for the MCMC chains
            if nburn<0, it will be recalculated based on
            the autocorrelation timescale tau

        thin : int, default=-1
            yield every 'thin' samples in the chain
            if nthin<0, it will be recalculated based on
            the autocorrelation timescale tau

        initial : array_like, default=[]
             initial values for the MCMC sampler

        normalize_unc : string, default='None'
            whether to normalize the flux uncertainty based 
            on the residual of a former estimation
                'None' : no normalization
                'LS' : based on former least square estimation
                'MCMC' : based on former MCMC

        Plot_model : bool, default=True
            whether to plot the model v.s. data

        Plot_mcmc : bool, default=False
            whether to plot the MCMC chains and corner plots

        Plot_tau : bool, default=False
            whether to plot the evolution of autocorrelation 
            time tau

        Returns
        -------
        sampler : emcee EnsembleSampler object
            emcee affine-invariant multi-chain MCMC sampler

        '''

        if len(initial) == 0:
            initial = self.theta_LS

        initial[0] = self.blue_fl[0]
        initial[1] = self.red_fl[0]

        ndim = len(initial)

        for k, free in enumerate(self.free_rel_strength):
            if free:
                rel = self.rel_strength[k][:-1].copy()
                initial = np.append(initial, np.log10(rel))

        ndim1 = len(initial)
        p0 = [i + initial for i in np.random.randn(nwalkers, ndim1) * 1e-5]

        if normalize_unc != '':
            if normalize_unc == 'LS':
                chi2 = self.chi2_LS
            elif normalize_unc == 'MCMC':
                chi2 = self.chi2_MCMC
            norm_fac = (chi2 / len(self.vel_rf))**.5
            print('Normalize factor = {:.3f}'.format(norm_fac))
        else:
            norm_fac = 1

        # Saving and monitoring process
        # https://emcee.readthedocs.io/en/stable/tutorials/monitor/
        # filename = "{}.h5".format(self.spec_name)  # save the chain
        # if filename in glob.glob('./*h5'):
        #    os.remove(filename)
        #backend = emcee.backends.HDFBackend(filename)
        #backend.reset(nwalkers, ndim)

        autocorr = np.zeros(max_nsteps)
        old_tau = np.inf

        from copy import deepcopy

        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim1,
            ln_prob,
            args=(deepcopy(self.rel_strength), self.lambda_0,
                  self.blue_vel, self.red_vel, self.vel_rf, self.norm_fl,
                  self.lines,
                  vel_flat, var_max,
                  mu_prior, var_prior,
                  self.norm_fl_unc * norm_fac,
                  [self.blue_fl[0], self.blue_fl[1] * norm_fac],
                  [self.red_fl[0], self.red_fl[1] * norm_fac],
                  self.free_rel_strength))
        # backend=backend)

        index = 0
        for sample in sampler.sample(p0,
                                     iterations=max_nsteps,
                                     progress=True):
            # Only check convergence every 500 steps
            if sampler.iteration % 500:
                continue

            # Compute the autocorrelation time so far
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1

            # Check convergence
            converged = np.all(tau * 50 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.05)
            if converged:
                break
            old_tau = tau

        if nburn < 0:
            try:
                nburn = int(2 * np.nanmax(tau))
            except:
                nburn = len(tau) // 2
        if thin < 0:
            try:
                thin = max(int(0.5 * np.nanmax(tau)), 1)
            except:
                thin = 2
        samples = sampler.get_chain(discard=nburn, flat=True)

        # Median as the estimator
        #self.theta_MCMC = np.median(samples, axis=0)
        # self.sig_theta_MCMC = (np.percentile(samples, q=84, axis=0)
        # - np.percentile(samples, q=16, axis=0)) / 2

        # Mode as the estimator
        # 68% credible region for the highest density
        self.theta_MCMC = []
        self.sig_theta_MCMC = []
        for i in range(ndim1):
            hist, bin_edges = np.histogram(
                samples[:, i], bins=50, density=True)
            bins = (bin_edges[1:] + bin_edges[:-1]) / 2
            width = bin_edges[1] - bin_edges[0]
            arg = np.argsort(hist)
            self.theta_MCMC.append(bins[arg][-1])

            cred = 0.68  # credible region
            dens_thres = np.inf
            j = 0
            while cred > 0:
                cred -= width * hist[arg[-1 - j]]
                dens_thres = hist[arg][-1 - j]
                j += 1
            interval = bins[hist > dens_thres]
            try:
                self.sig_theta_MCMC.append((interval[-1] - interval[0]) / 2)
            except:
                warnings.warn('No valid credit interval!')
                self.sig_theta_MCMC.append((bins[-1] - bins[0]) / 2)

            # If the posteriors are asymmetric
            # self.sig_theta_MCMC.append(
            #    [bins[arg][-1] - interval[0], interval[-1] - bins[arg][-1]])

        print('MCMC results:')
        for k in range(ndim // 3):
            print('Velocity {}: {:.0f} pm {:.0f} km/s'.format(k +
                  1, self.theta_MCMC[2 + 3 * k], self.sig_theta_MCMC[2 + 3 * k]))

        self.chi2_MCMC = neg_lnlike_gaussian_abs(self.theta_MCMC, self.rel_strength, self.lambda_0,
                                                 self.blue_vel, self.red_vel, self.vel_rf, self.norm_fl,
                                                 self.lines, self.norm_fl_unc, 'chi2', self.free_rel_strength)

        # convert amplitude to equivalent width
        self.EW = 0
        self.sig_EW = 0
        for k, rs in enumerate(self.rel_strength):
            ratio = 2 / (self.red_vel - self.blue_vel) * \
                (self.wv_line[-1] - self.wv_line[0]) * (np.sum(rs) + 1)
            self.EW += self.theta_MCMC[4 + 3 * k] * -ratio
            self.sig_EW += self.sig_theta_MCMC[4 + 3 * k] * ratio

        # If the posteriors are asymmetric
        # print('Velocity pvf: {:.0f} plus {:.0f} minus {:.0f} km/s'.format(
        #    self.theta_MCMC[2], self.sig_theta_MCMC[2][0], self.sig_theta_MCMC[2][1]))
        # if ndim == 8:
        #    print('Velocity hvf: {:.0f} plus {:.0f} minus {:.0f} km/s'.format(
        #        self.theta_MCMC[5], self.sig_theta_MCMC[5][0], self.sig_theta_MCMC[5][1]))

        if Plot_model:
            self.plot_model(self.theta_MCMC)
        if Plot_mcmc:
            samples = sampler.get_chain(discard=nburn, flat=True, thin=thin)
            plot_MCMC(sampler=sampler, num_vel_com=len(
                self.rel_strength), nplot=20, samples=samples)
        if Plot_tau:
            n = 500 * np.arange(1, index + 1)
            y = autocorr[:index]
            plt.figure(figsize=(10, 10))
            plt.plot(n, n / 50.0, "--k")
            plt.plot(n, y)
            plt.xlim(500, n.max())
            plt.ylim(10, y.max() + 0.1 * (y.max() - y.min()))
            plt.xlabel(r"$\mathrm{Number\ of\ steps}$")
            plt.ylabel(r"$\mathrm{Mean}\ \hat{\tau}$")
            plt.show()

        return sampler

    def plot_model(self, theta, return_ax=False):
        '''Plot the predicted absorption features

        Parameters
        ----------
        theta : array_like
            fitting parameters: flux at the blue edge, flux at the
            red edge, (mean of relative velocity, log variance,
            amplitude) * Number of velocity components

        return_ax : boolean, default=False
            whether to return the axes
            if return_ax == True, a matplotlib axes will be returned
        '''

        plt.figure(figsize=(10, 10))
        # ensure high resolution in predicted model
        if len(self.vel_rf) < 200:
            vel_rf = np.linspace(self.vel_rf[0], self.vel_rf[-1], 200)
        else:
            vel_rf = self.vel_rf
        num = len(self.rel_strength)
        theta0 = theta[:2 + 3 * num]
        j = 2 + 3 * num
        rel_strength = self.rel_strength.copy()
        for k, rel in enumerate(self.free_rel_strength):
            if rel:
                for rel_s in range(len(rel_strength[k])-1):
                    rel_strength[k][rel_s] = 10**theta[j]
                    j += 1
        if j != len(theta):
            raise IndexError(
                'Number of free parameters and relative strength do not match')
        model_flux = flux_gauss(theta0,
                                rel_strength,
                                self.lambda_0,
                                self.blue_vel, self.red_vel,
                                vel_rf, self.lines)
        model_res = flux_gauss(theta0,
                               rel_strength,
                               self.lambda_0,
                               self.blue_vel, self.red_vel,
                               self.vel_rf, self.lines) - self.norm_fl
        plt.errorbar(self.vel_rf, self.norm_fl,
                     yerr=self.norm_fl_unc, alpha=0.5, elinewidth=.5)
        model_plot = plt.plot(vel_rf, model_flux, linewidth=5)
        plt.errorbar([self.vel_rf[0], self.vel_rf[-1]], [
                     model_flux[0], model_flux[-1]],
                     yerr=[self.blue_fl[1], self.red_fl[1]],
                     color=model_plot[0].get_color(), fmt='o', capsize=5)
        plt.plot(self.vel_rf, model_res, color='grey')

        if len(rel_strength) > 1:
            for k in range(len(rel_strength)):
                model_flux = flux_gauss(np.append(theta0[:2], theta0[2 + 3 * k:5 + 3 * k]),
                                        [rel_strength[k]],
                                        self.lambda_0,
                                        self.blue_vel, self.red_vel,
                                        self.vel_rf, [self.lines[k]])
                plt.plot(self.vel_rf, model_flux,
                         color='k', alpha=0.4, linewidth=3)

        plt.xlabel(r'$v\ [\mathrm{km/s}]$')
        plt.ylabel(r'$\mathrm{Normalized\ Flux}$')
        plt.tight_layout()
        if return_ax:
            return plt.gca()
        else:
            plt.show()

###################### Basic Functions ##########################


def velocity_rf(lambda_rf, lambda_0):
    '''convert rest-frame wavelength to relative velocity'''
    c = 2.99792458e5
    v = c * ((lambda_rf / lambda_0)**2 - 1) / ((lambda_rf / lambda_0)**2 + 1)

    return v


def velocity_rf_line(lambda_0, lambda_1, vel):
    '''get the relative velocity of a feature assuming another line

    Parameters
    ----------
    lambda_0 : float
        wavelength of the original line [angstrom]
    lambda_1 : float
        wavelength of the new line [angstrom]
    vel : float
        the velocity relative to lambda_0 [km/s]

    Returns
    -------
    vel_1 : float
        the velocity relative to lambda_1 [km/s]
    '''

    c = 2.99792458e5
    lambda_rf = ((vel / c + 1)/(-vel / c + 1))**.5 * lambda_0
    vel_1 = velocity_rf(lambda_rf, lambda_1)

    return vel_1


def calc_gauss(mean_vel, var_vel, amplitude, vel_rf):
    '''gaussian profile'''
    gauss = amplitude / np.sqrt(2 * np.pi * var_vel) * \
        np.exp(-0.5 * (vel_rf - mean_vel)**2 / var_vel)
    return gauss


def flux_gauss(theta, rel_strength, lambda_0, blue_vel, red_vel, vel_rf,
               lines=[]):
    '''Calculate normalized flux based on a Gaussian model

    Parameters
    ----------
    theta : array_like
        fitting parameters: flux at the blue edge, flux at the
        red edge, (mean of relative velocity, log variance,
        amplitude) * Number of velocity components

    rel_strength : array_like, default=[]
        the relative strength between each line in the series
        rel_strength = []: all lines are of the equal strength

    lambda_0 : float
        the wavelength as a reference for velocity

    blue_vel, red_vel : float
        the relative velocity [km/s] at the blue/red edge

    vel_ref : float
        relative velocities [km/s] for each flux measurement

    norm_flux : float
        normalized flux

    lines : 2D array_like, default=[]
        wavelength of each absorption line

    Returns
    -------
    model_flux : array_like
        predicted (normalized) flux at each relative radial
        velocity
    '''

    y1, y2 = theta[:2]
    m = (y2 - y1) / (red_vel - blue_vel)
    b = y2 - m * red_vel

    model_flux = m * vel_rf + b
    for k in range(len(theta) // 3):
        mean_vel, lnvar, amplitude = theta[3 * k + 2:3 * k + 5]
        var_vel = np.exp(lnvar)

        for rel_s, li in zip(rel_strength[k], lines[k][:-1]):
            vel = velocity_rf_line(li, lambda_0, mean_vel)
            model_flux += rel_s * calc_gauss(vel, var_vel,
                                             amplitude, vel_rf)
        vel = velocity_rf_line(lines[k][-1], lambda_0, mean_vel)
        model_flux += calc_gauss(vel, var_vel, amplitude, vel_rf)
    return model_flux


###################### Likelihood & prior ##########################


def lnlike_gaussian_abs(theta,
                        rel_strength,
                        lambda_0,
                        blue_vel,
                        red_vel,
                        vel_rf,
                        norm_flux,
                        lines=[],
                        flux_unc=1,
                        type='gaussian',
                        free_rel_strength=[]):
    '''Log likelihood function assuming Gaussian profile

    Parameters
    ----------
    theta : array_like
        fitting parameters: flux at the blue edge, flux at the
        red edge, (mean of relative velocity, log variance,
        amplitude) * Number of velocity components, log10 line ratio
        for each velocity components (if set free)

    rel_strength : array_like, default=[]
        the relative strength between each line in the series
        rel_strength = []: all lines are of the equal strength

    blue_vel, red_vel : float
        the relative velocity [km/s] at the blue/red edge

    vel_ref : float
        relative velocities [km/s] for each flux measurement

    norm_flux : float
        normalized flux

    lines : 2D array_like, default=[]
            wavelength of each absorption line

    lambda_0 : float
        the wavelength as a reference for velocity

    flux_unc : float
        uncertainty in normalized flux

    type : ['gaussian', 'chi2'], default='gaussian'
        'gaussian': Gaussian likelihood (for mcmc)
        'chi2': chi2 likelihood (for least square estimation)

    free_rel_strength : array_like, default=[]
        whether to set the relative strength of each line series as
        another free parameter in MCMC fit

    Returns
    -------
    lnl : float
        the log likelihood function
    '''
    num = len(rel_strength)
    theta0 = theta[:2 + 3 * num]
    j = 2 + 3 * num
    for k, rel in enumerate(free_rel_strength):
        if rel:
            for rel_s in range(len(rel_strength[k])-1):
                rel_strength[k][rel_s] = 10**theta[j]
                j += 1
    if j != len(theta):
        print(j, theta)
        raise IndexError(
            'Number of free parameters and relative strength do not match')

    model_flux = flux_gauss(theta0, rel_strength, lambda_0,
                            blue_vel, red_vel, vel_rf,
                            lines)
    if type == 'gaussian':
        lnl = -0.5 * len(model_flux) * np.log(2 * np.pi) - np.sum(
            np.log(flux_unc)) - 0.5 * np.sum(
                (norm_flux - model_flux)**2 / flux_unc**2)
    elif type == 'chi2':
        lnl = -np.sum((norm_flux - model_flux)**2 / flux_unc**2)

    return lnl


def neg_lnlike_gaussian_abs(theta,
                            rel_strength,
                            lambda_0,
                            blue_vel,
                            red_vel,
                            vel_rf,
                            norm_flux,
                            lines=[],
                            flux_unc=1,
                            type='gaussian',
                            free_rel_strength=[]):
    '''negative log-likelihood function'''

    lnl = lnlike_gaussian_abs(theta,
                              rel_strength,
                              lambda_0,
                              blue_vel,
                              red_vel,
                              vel_rf,
                              norm_flux,
                              lines=lines,
                              flux_unc=flux_unc,
                              type=type,
                              free_rel_strength=free_rel_strength)
    return -1 * lnl


def lnprior(
    theta,
    vel_flat=[-1e5, 0],
    var_max=1e8,
    delta_vel=0,
    mu_prior=[],
    var_prior=[],
    blue_fl=[1, .1],
    red_fl=[1, .1],
    free_rel_strength=[]
):
    '''log-prior probability


    Parameters
    ----------
    theta : array_like
        fitting parameters: flux at the blue edge, flux at the
        red edge, (mean of relative velocity, log variance,
        amplitude) * Number of velocity components, line ratio
        for each velocity components (if set free)

    vel_flat : list, default=[-1e5, 0]
        the range of the flat velocity prior [km/s]

    var_max : float, default=1e8
        maximum var allowed in a line profile

    delta_vel : float, default=0
        the relative velocity between the bluest and reddest line

    mu_prior : array_like, default=[]
        means of the profile prior

    var_prior : float, default=[]
        variances of the profile prior

    blue_fl, red_fl : list
        normalized flux and uncertainty at the blue/red edge

    free_rel_strength : array_like, default=[]
        whether to set the relative strength of each line series as
        another free parameter in MCMC fit

    Returns
    -------
    lnprior : float
        the log prior probability
    '''

    y1, y2 = theta[:2]
    lnp_y1 = -np.log(blue_fl[1]) - (blue_fl[0] - y1)**2 / 2 / blue_fl[1]**2
    lnp_y2 = -np.log(red_fl[1]) - (red_fl[0] - y2)**2 / 2 / red_fl[1]**2

    num_vel_com = len(free_rel_strength)

    sig_lim = min((vel_flat[1] - vel_flat[0] - delta_vel) / 2, var_max**.5)

    if len(mu_prior) != len(var_prior):
        raise IndexError('Means and variances of prior do not match.')

    mean_vel = np.array(theta[2:2 + 3 * num_vel_com:3])
    lnvar = np.array(theta[3:2 + 3 * num_vel_com:3])
    amplitude = np.array(theta[4:2 + 3 * num_vel_com:3])
    var_vel = np.exp(lnvar)
    if len(var_prior) == 0:
        lnpvf = 0
    else:
        var_prior = np.array(var_prior)
        mu_prior = np.array(mu_prior)
        lnpvf = -0.5 * np.log(
            2 * np.pi * var_prior) - (mean_vel - mu_prior)**2 / 2 / var_prior
        lnpvf = lnpvf.sum()
    vlim = -np.inf
    for k in range(len(mean_vel)):
        if not (vel_flat[0] + delta_vel < mean_vel[k] < vel_flat[1]
                and 2e1 < var_vel[k]**.5 < sig_lim
                and (-(2 * np.pi * var_vel[k])**.5 * np.mean([y1, y2]) < amplitude[k])
                and mean_vel[k] > vlim):
            return -np.inf
        vlim = mean_vel[k]
    for rel in theta[2 + 3 * num_vel_com:]:
        if not (-1 < rel < 0):
            return -np.inf
    return lnp_y1 + lnp_y2 + lnpvf


def ln_prob(
    theta,
    rel_strength,
    lambda_0,
    blue_vel,
    red_vel,
    vel_rf,
    norm_flux,
    lines=[],
    vel_flat=[-1e5, 0],
    var_max=1e8,
    mu_prior=[],
    var_prior=[],
    norm_flux_unc=1,
    blue_fl=[1, .1],
    red_fl=[1, .1],
    free_rel_strength=[]
):
    '''log-posterior probability

    See lnprior() and lnlike_gaussian_abs() for details
    '''
    if len(free_rel_strength) == 0:
        free_rel_strength = np.array([False] * len(rel_strength))
    delta_vel = 0
    for l in lines:
        delta_vel_min = np.inf
        for k in l:
            delta_vel_temp = velocity_rf_line(lambda_0, k, 0)
            if delta_vel_min > delta_vel_temp:
                delta_vel_min = delta_vel_temp
            if delta_vel_temp - delta_vel_min > delta_vel:
                delta_vel = delta_vel_temp - delta_vel_min
    ln_prior = lnprior(theta=theta, vel_flat=vel_flat, var_max=var_max,
                       delta_vel=delta_vel,
                       mu_prior=mu_prior, var_prior=var_prior, blue_fl=blue_fl, red_fl=red_fl, free_rel_strength=free_rel_strength)
    ln_like = lnlike_gaussian_abs(theta=theta, lambda_0=lambda_0,
                                  rel_strength=rel_strength,
                                  blue_vel=blue_vel, red_vel=red_vel,
                                  vel_rf=vel_rf, norm_flux=norm_flux,
                                  lines=lines,
                                  flux_unc=norm_flux_unc,
                                  free_rel_strength=free_rel_strength)
    return ln_prior + ln_like


######################### MCMC visualization ##############################


def plot_MCMC(sampler=None, num_vel_com=1, nburn=0, thin=1, nplot=None, samples=[]):
    '''plot walker chains and corner plots

    Parameters
    ----------
    sampler : emcee EnsembleSampler object, default=0
        emcee affine-invariant multi-chain MCMC sampler

    num_vel_com : int, default=1
        number of velocity components

    nburn : int, default=0
        number of "burn-in" steps for the MCMC chains

    thin : int, default=1
        take only every thin steps from the chain

    nplot : int, default=None
        number of chains to show in the visualization

    samples : array_like, default=[]
        MCMC samples
    '''

    ndim = sampler.get_chain().shape[2]
    paramsNames = [r'$\mathrm{Blue\ edge\ flux}$',
                   r'$\mathrm{Red\ edge\ flux}$']
    for k in range(num_vel_com):
        v_label = r'$v_{}$'.format(k)
        sig2_label = r'$\ln(\sigma^2_{})$'.format(k)
        A_label = r'$A_{}$'.format(k)
        paramsNames = np.append(paramsNames, [v_label, sig2_label, A_label])
    for k in range(ndim - 2 - 3 * num_vel_com):
        ratio_label = r'$\lg(\mathrm{Ratio}_' + r'{})$'.format(1)
        paramsNames = np.append(paramsNames, ratio_label)

    # plotChains(sampler, nburn, paramsNames, nplot)
    # plt.tight_layout()
    # plt.show()
    if len(samples) == []:
        samples = sampler.get_chain(discard=nburn, flat=True, thin=thin)
    fig = corner.corner(samples,
                        labels=paramsNames,
                        quantiles=[0.16, 0.50, 0.84],
                        show_titles=True)


def plotChains(sampler, nburn, paramsNames, thin=1, nplot=None):
    '''Plot individual chains from the emcee MCMC sampler

    Parameters
    ----------
    sampler : emcee EnsembleSampler object
        emcee affine-invariant multi-chain MCMC sampler

    nburn : int
        number of "burn-in" steps for the MCMC chains

    paramsNames : array_like
        names of the parameters to be shown

    thin : int, default=1
        yield every 'thin' samples in plotting the chain

    nplot : int, default=None
        number of chains to show in the visualization.
        In instances where the number of chains is
        very large (>> 100), then it can be helpful to
        downsample to provide more clarity.

    Returns
    -------
    ax : maptlotlib axes object
        multi panel plot showing the evoltion of
        each chain for the parameters in the model

    '''

    Nparams = len(paramsNames)
    nwalkers = sampler.get_chain(thin=thin).shape[1]

    fig, ax = plt.subplots(Nparams + 1,
                           1,
                           figsize=(12, 3 * (Nparams + 1)),
                           sharex=True)
    fig.subplots_adjust(hspace=0)
    ax[0].set_title(r'$\mathrm{Chains}$')
    xplot = np.arange(sampler.get_chain(thin=thin).shape[0]) * thin

    if nplot is None:
        nplot = nwalkers
    selected_walkers = np.random.choice(range(nwalkers), nplot, replace=False)
    for i, p in enumerate(paramsNames):
        for w in selected_walkers:
            burn = ax[i].plot(xplot[:nburn // thin],
                              sampler.get_chain(thin=thin)[
                :nburn // thin, w, i],
                alpha=0.4,
                lw=0.7,
                zorder=1)
            ax[i].plot(xplot[nburn // thin - 1:],
                       sampler.get_chain(thin=thin)[nburn // thin - 1:, w, i],
                       color=burn[0].get_color(),
                       alpha=0.8,
                       lw=0.7,
                       zorder=1)

            ax[i].set_ylabel(p)
            if i == Nparams - 1:
                ax[i + 1].plot(xplot[:nburn // thin],
                               sampler.get_log_prob(thin=thin)[
                    :nburn // thin, w],
                    color=burn[0].get_color(),
                    alpha=0.4,
                    lw=0.7,
                    zorder=1)
                ax[i + 1].plot(xplot[nburn // thin - 1:],
                               sampler.get_log_prob(
                                   thin=thin)[nburn // thin - 1:, w],
                               color=burn[0].get_color(),
                               alpha=0.8,
                               lw=0.7,
                               zorder=1)
                ax[i + 1].set_ylabel(r'$\ln P$')

    return ax
