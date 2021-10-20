import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import emcee
import corner

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.size'] = '25'
mpl.rcParams['xtick.labelsize'] = '20'
mpl.rcParams['ytick.labelsize'] = '20'

import warnings


##################### SpectrumSN class ##########################


class SpectrumSN(object):
    '''1D optical spectrum

    Attributes
    ----------
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

    def __init__(self, spec1D, z=0):
        '''Constructor

        Parameters
        ----------
        spec1D : str
            the spectrum file (directory + filename)

        z : float (default=0)
            host galaxy redshift
        '''

        spec_df = pd.read_csv(spec1D,
                              comment='#',
                              delim_whitespace=True,
                              header=None)

        wv = spec_df[0].values
        wv_rf = wv / (1 + z)
        fl = spec_df[1].values

        try:
            if 'Keck' in spec1D:
                fl_unc = spec_df[3].values
            else:
                fl_unc = spec_df[2].values

                if 'P60' in spec1D:
                    fl_unc **= .5

            rel_unc = fl_unc / fl
            rel_unc[rel_unc < 1e-3] = np.median(rel_unc[rel_unc > 1e-3])
            fl_unc = rel_unc * fl
        except:
            warnings.warn("No flux uncertainty in the datafile!")
            # set relative uncertainty to be 1%
            fl_unc = np.ones_like(fl) * 1e-2 * np.median(fl)

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
        '''
        line_region = np.where(
            (self.wv_rf < red_edge) & (self.wv_rf > blue_edge))[0]

        plt.figure(figsize=(10, 10))
        plt.plot(self.wv_rf[line_region], self.fl[line_region])
        plt.errorbar(self.wv_rf[line_region], self.fl[line_region],
                     yerr=self.fl_unc[line_region], fmt='o')
        plt.tight_layout()
        plt.show()

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
            if len(region) == 1:
                warnings.warn('Too few points within the wavelength range!')
                return (self.fl[region], self.fl_unc[region])
            else:
                mean = self.fl[region].mean()
                std = np.std(self.fl[region], ddof=1)
            return (mean, std)
        except IndexError as e:
            repr(e)
            return None, None


class SpectrumSN_Lines(SpectrumSN):
    '''A set of measurements on different absorption lines

    Children class of SpectrumSN

    Attributes
    ----------
    Spec1D : str
        the spectrum file (directory + filename)

    z : float
        host galaxy redshift

    spec_name : str
        spectrum name based on the filename
        source name + date + instrument

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

        z : float (default=0)
            host galaxy redshift
        '''
        super(SpectrumSN_Lines, self).__init__(spec1D, z)
        self.spec1D = spec1D
        self.z = z
        self.spec_name = spec1D[spec1D.find('ZTF'):spec1D.find('.ascii')]
        self.line = {}

    def add_line(self, name, blue_edge, red_edge, lines=[]):
        '''Add one (series of) absorption line(s)

        Construct a new AbsorbLine object, and save it in self.line

        Parameters
        ----------
        name : str
            the name of the absorption line

        blue_edge, red_edge : float
            the wavelength [angstrom] (host galaxy frame) at 
            the blue/red edge

        lines : array_like, default=[]
            the central wavelength(s) [angstrom] of this 
            (series) of line(s)
        '''
        self.line[name] = AbsorbLine(
            self.spec1D, self.z, blue_edge, red_edge, lines)


class AbsorbLine(SpectrumSN):
    '''A (series of) absorption line(s) in a 1D optical spectrum

    Attributes
    ----------
    wv_line : array_like
        the wavelength range [angstrom] of the line 
        (host galaxy frame)

    norm_fl : array_like
        flux normalized by the median value

    norm_fl_unc : array_like
        normalized flux uncertainty

    blue_fl, red_fl : list
        normalized flux and uncertainty at the blue/red edge

    vel_rf : array_like
        relative velocities of each point with respect 
        to the center of absorption

    blue_vel, red_vel : float
        relative velocity at the blue/red edge

    delta_vel_components : array_like, default=[]
        relative velocities of other absorption lines (if any)
        with respect to the default one at v=0

    theta_LS : array_like
        best fit with least square methods, powered by
        scipy.optimize.minimize()

    chi2_LS : float
        the minimized residual with least square methods

    theta_MCMC : array_like
        best fit with an MCMC sampler, approximated by median 
        values in the MCMC chains

    sig_theta_MCMC : array_like
        1 sigma uncertainty for each parameter, approximated 
        by the half the range of 16 and 84 percentile values 
        in the MCMC chains



    Methods
    -------
    LS_estimator(guess=(1, 1, -10000, 15, -1000)) :
        Least square point estimation

    MCMC_sampler(mu_pvf=-1e4, var_pvf=1e7,
                 nwalkers=100, nsteps=1500, 
                 nburn=500, initial=[],
                 normalize_unc=False) :
        An MCMC sampler

    plot_model(theta) :
        Plot the predicted absorption features

    '''

    def __init__(self, spec1D, z,
                 blue_edge, red_edge,
                 lines=[]):
        '''Constructor

        Parameters
        ----------
        spec1D: str
            the spectrum file (directory + filename)

        z: float (default=0)
            host galaxy redshift

        blue_edge, red_edge: float
            the wavelength [angstrom] (host galaxy frame) at the blue/red edge

        lines: array_like, default=[]
            the central wavelength(s) [angstrom] of this (series) of line(s)
            (e.g. [6371.359, 6347.103])
        '''

        super(AbsorbLine, self).__init__(spec1D, z)

        # line region
        line_region = np.where(
            (self.wv_rf < red_edge) & (self.wv_rf > blue_edge))[0]
        self.wv_line = self.wv_rf[line_region]
        print('{:.0f} points within {:.2f} and {:.2f} angstroms.'.format(
            len(line_region), blue_edge, red_edge))

        # normalized flux
        norm_fl = self.fl / np.nanmedian(self.fl)
        norm_fl_unc = self.fl_unc / np.nanmedian(self.fl)
        self.norm_fl = norm_fl[line_region]
        self.norm_fl_unc = norm_fl_unc[line_region]

        # flux at each edge
        blue_fl = super(AbsorbLine, self).get_flux_at_lambda(blue_edge)
        red_fl = super(AbsorbLine, self).get_flux_at_lambda(red_edge)
        self.blue_fl = blue_fl / np.nanmedian(self.fl)
        self.red_fl = red_fl / np.nanmedian(self.fl)

        # velocity
        lines = np.sort(lines)
        lambda_0 = lines[-1]
        vel_rf = velocity_rf(self.wv_rf, lambda_0)
        self.vel_rf = vel_rf[line_region]

        self.blue_vel = velocity_rf(blue_edge, lambda_0)
        self.red_vel = velocity_rf(red_edge, lambda_0)
        self.delta_vel_components = [
            velocity_rf(lambda_0, l) for l in lines[:-1]]

        self.theta_LS = []
        self.chi2_LS = np.nan

        self.theta_MCMC = []
        self.sig_theta_MCMC = []

    def LS_estimator(self, guess=(1, 1, -10000, 15, -1000)):
        '''Least square point estimation

        Parameters
        ----------
        guess: tuple, default=(1, 1, -10000, 15, -1000)
            an initial guess for the fitting parameter theta
        '''

        LS_res = minimize(
            neg_lnlike_gaussian_abs,
            guess,
            method='Powell',  # Powell method does not need derivatives
            args=(self.blue_vel, self.red_vel, self.vel_rf, self.norm_fl,
                  self.delta_vel_components, self.norm_fl_unc, 'chi2'))

        self.theta_LS = LS_res['x']
        self.chi2_LS = LS_res['fun']

        self.plot_model(self.theta_LS)

        # print(self.theta_LS)
        ndim = len(self.theta_LS)
        print('LS estimation:')
        if ndim == 5:
            print('Velocity pvf: {:.0f} km/s'.format(
                self.theta_LS[2]))
        elif ndim == 8:
            print('Velocity pvf: {:.0f} km/s'.format(
                self.theta_LS[2]))
            print('Velocity hvf: {:.0f} km/s'.format(
                self.theta_LS[5]))

    def MCMC_sampler(self,
                     mu_pvf=-1e4, var_pvf=1e7,
                     nwalkers=100, nsteps=1500, nburn=500, initial=[],
                     normalize_unc=False):
        '''MCMC sampler

        Parameters
        ----------
        mu_pvf : float, default=-1e4
            mean of the pvf profile prior

        var_pvf : float, default=1e7
            var of the pvf profile prior

        nwalkers : int, default=100
            number of MCMC sampler walkers

        nsteps : int, default=1500
            MCMC chain length

        nburn : int, default=500
            number of "burn-in" steps for the MCMC chains

        initial : array_like, default=[]
             initial values for the MCMC sampler

        normalize_unc : bool, default=False
            whether to normalize the flux uncertainty based on the
            residual of a former LS estimation

        Returns
        -------
        sampler : emcee EnsembleSampler object
            emcee affine-invariant multi-chain MCMC sampler

        '''

        if len(initial) == 0:
            initial = self.theta_LS

        ndim = len(initial)
        p0 = [i + initial for i in np.random.randn(nwalkers, ndim) * 1e-5]

        if normalize_unc:
            norm_fac = (self.chi2_LS / len(vel_rf))**.5
        else:
            norm_fac = 1

        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            ln_prob,
            args=(self.blue_vel, self.red_vel, self.vel_rf, self.norm_fl,
                  self.delta_vel_components,
                  mu_pvf, var_pvf,
                  self.norm_fl_unc * norm_fac,
                  self.blue_fl, self.red_fl))
        sampler.run_mcmc(p0, nsteps, progress=True)

        self.theta_MCMC = np.median(
            sampler.chain[:, nburn:, :].reshape((-1, ndim)), axis=0)
        self.sig_theta_MCMC = (np.percentile(
            sampler.chain[:, nburn:, :].reshape((-1, ndim)), q=84, axis=0) - np.percentile(
            sampler.chain[:, nburn:, :].reshape((-1, ndim)), q=16, axis=0)) / 2

        self.plot_model(self.theta_LS)

        print('MCMC results:')
        if ndim == 5:
            print('Velocity pvf: {:.0f} pm {:.0f} km/s'.format(
                self.theta_MCMC[2], self.sig_theta_MCMC[2]))
        elif ndim == 8:
            print('Velocity pvf: {:.0f} pm {:.0f} km/s'.format(
                self.theta_MCMC[2], self.sig_theta_MCMC[2]))
            print('Velocity hvf: {:.0f} pm {:.0f} km/s'.format(
                self.theta_MCMC[5], self.sig_theta_MCMC[5]))

        return sampler

    def plot_model(self, theta):
        '''Plot the predicted absorption features

        Parameters
        ----------
        theta : array_like
        fitting parameters: flux at the blue edge, flux at the
        red edge, (mean of relative velocity, log variance,
        amplitude) * Number of velocity components
        '''

        plt.figure(figsize=(10, 10))
        model_flux = flux_gauss(theta,
                                self.blue_vel, self.red_vel,
                                self.vel_rf, self.delta_vel_components)
        plt.plot(self.vel_rf, self.norm_fl)
        plt.errorbar(self.vel_rf, model_flux,
                     yerr=self.norm_fl_unc, alpha=0.5, linewidth=5, elinewidth=.5)
        plt.plot(self.vel_rf, model_flux - self.norm_fl, color='grey')

        if len(theta) == 8:
            model1_flux = flux_gauss(theta[:5],
                                     self.blue_vel, self.red_vel,
                                     self.vel_rf, self.delta_vel_components)
            model2_flux = flux_gauss(np.append(theta[:2], theta[5:]),
                                     self.blue_vel, self.red_vel,
                                     self.vel_rf, self.delta_vel_components)
            plt.plot(self.vel_rf, model1_flux,
                     color='k', alpha=0.8, linewidth=3)
            plt.plot(self.vel_rf, model2_flux,
                     color='k', alpha=0.4, linewidth=3)

        plt.xlabel(r'$v\ [\mathrm{km/s}]$')
        plt.ylabel(r'$\mathrm{Normalized\ Flux}$')
        plt.tight_layout()
        plt.show()


###################### Basic Functions ##########################


def velocity_rf(lambda_rf, lambda_0):
    '''convert rest-frame wavelength to relative velocity'''
    c = 2.99792458e5
    v = c * ((lambda_rf / lambda_0)**2 - 1) / ((lambda_rf / lambda_0)**2 + 1)

    return v


def calc_gauss(mean_vel, var_vel, amplitude, vel_rf):
    '''gaussian profile'''
    gauss = amplitude / np.sqrt(2 * np.pi * var_vel) * np.exp(
        -0.5 * (vel_rf - mean_vel)**2 / var_vel)
    return gauss


def flux_gauss(theta, blue_vel, red_vel, vel_rf, delta_vel_components=[]):
    '''Calculate normalized flux based on a Gaussian model

    Parameters
    ----------
    theta : array_like
        fitting parameters: flux at the blue edge, flux at the
        red edge, (mean of relative velocity, log variance,
        amplitude) * Number of velocity components

    blue_vel, red_vel : float
        the relative velocity [km/s] at the blue/red edge

    vel_ref : float
        relative velocities [km/s] for each flux measurement

    norm_flux : float
        normalized flux

    delta_vel_components : array_like (default=[])
        relative velocities of other absorption lines (if any)
        with respect to the default one at v=0

    Returns
    -------
    model_flux : array_like
        predicted (normalized) flux at each relative radial
        velocity
    '''

    y1, y2 = theta[:2]
    m = (y2 - y1) / (blue_vel - red_vel)
    b = y2 - m * blue_vel

    model_flux = m * vel_rf + b

    for i in range(len(theta[2:]) // 3):
        mean_vel, lnvar, amplitude = theta[3 * i + 2:3 * i + 5]
        var_vel = np.exp(lnvar)
        model_flux += calc_gauss(mean_vel, var_vel, amplitude, vel_rf)

        if len(delta_vel_components) > 0:
            for delta_vel in delta_vel_components:
                model_flux += calc_gauss(mean_vel - delta_vel, var_vel,
                                         amplitude, vel_rf)
    return model_flux


###################### Likelihood & prior ##########################


def lnlike_gaussian_abs(theta,
                        blue_vel,
                        red_vel,
                        vel_rf,
                        norm_flux,
                        delta_vel_components=[],
                        flux_unc=1,
                        type='gaussian'):
    '''Log likelihood function assuming Gaussian profile

    Parameters
    ----------
    theta : array_like
        fitting parameters: flux at the blue edge, flux at the
        red edge, (mean of relative velocity, log variance,
        amplitude) * Number of velocity components

    blue_vel, red_vel : float
        the relative velocity [km/s] at the blue/red edge

    vel_ref : float
        relative velocities [km/s] for each flux measurement

    norm_flux : float
        normalized flux

    delta_vel_components : array_like, default=[]
        relative velocities of other absorption lines (if any)
        with respect to the default one at v=0

    flux_unc : float
        uncertainty in normalized flux

    type : ['gaussian', 'chi2'], default='gaussian'
        'gaussian': Gaussian likelihood (for mcmc)
        'chi2': chi2 likelihood (for least square estimation)

    Returns
    -------
    lnl : float
        the log likelihood function
    '''

    model_flux = flux_gauss(theta, blue_vel, red_vel, vel_rf,
                            delta_vel_components)
    if type == 'gaussian':
        lnl = -0.5 * len(model_flux) * np.log(2 * np.pi) - np.sum(
            np.log(flux_unc)) - 0.5 * np.sum(
                (norm_flux - model_flux)**2 / flux_unc**2)
    elif type == 'chi2':
        lnl = -np.sum((norm_flux - model_flux)**2 / flux_unc**2)

    return lnl


def neg_lnlike_gaussian_abs(theta,
                            blue_vel,
                            red_vel,
                            vel_rf,
                            norm_flux,
                            delta_vel_components=[],
                            flux_unc=1,
                            type='gaussian'):
    '''negative log-likelihood function'''

    lnl = lnlike_gaussian_abs(theta,
                              blue_vel,
                              red_vel,
                              vel_rf,
                              norm_flux,
                              delta_vel_components=delta_vel_components,
                              flux_unc=flux_unc,
                              type=type)
    return -1 * lnl


def lnprior(
    theta,
    mu_pvf=-1e4,
    var_pvf=1e7,
    blue_fl=[1, .1],
    red_fl=[1, .1],
):
    '''log-prior probability


    Parameters
    ----------
    theta : array_like
        fitting parameters: flux at the blue edge, flux at the
        red edge, (mean of relative velocity, log variance,
        amplitude) * Number of velocity components

    mu_pvf : float, default=-1e4
        mean of the pvf profile prior

    var_pvf : float, default=1e7
        var of the pvf profile prior

    blue_fl, red_fl : list
        normalized flux and uncertainty at the blue/red edge

    Returns
    -------
    lnprior : float
        the log prior probability
    '''

    y1, y2 = theta[:2]
    lnp_y1 = -np.log(blue_fl[1]) - (blue_fl[0] - y1)**2 / 2 / blue_fl[1]**2
    lnp_y2 = -np.log(red_fl[1]) - (red_fl[0] - y2)**2 / 2 / red_fl[1]**2

    if len(theta[2:]) == 3:
        mean_vel, lnvar, amplitude = theta[2:]
        var_vel = np.exp(lnvar)
        if (-40000 < mean_vel < 0 and 100 < var_vel**0.5 < 22000
                and -1e5 < amplitude < 0):
            return lnp_y1 + lnp_y2
            # lnflat = 0  #-np.log(40000 * (22000 - 100) * 1e5)
        else:
            return -np.inf

    elif len(theta[2:]) == 6:
        mean_vel_pvf, lnvar_pvf, amp_pvf = theta[2:3 + 2]
        mean_vel_hvf, lnvar_hvf, amp_hvf = theta[3 + 2:]
        var_rel_pvf = np.exp(lnvar_pvf)
        var_rel_hvf = np.exp(lnvar_hvf)

        lnpvf = -0.5 * np.log(
            2 * np.pi * var_pvf) - (mean_vel_pvf - mu_pvf)**2 / 2 / var_pvf

        if (0 > mean_vel_pvf > -2e4 and 1e5 < var_rel_pvf < 1e9
                and -1e4 < amp_pvf < 0
                and mean_vel_pvf - 2000 > mean_vel_hvf > -4e4
                and 1e5 < var_rel_hvf < 1e9 and -1e4 < amp_hvf < 0):
            lnflat = 0
        else:
            lnflat = -np.inf
    return lnpvf + lnflat + lnp_y1 + lnp_y2


def ln_prob(
    theta,
    blue_vel,
    red_vel,
    vel_rf,
    norm_flux,
    delta_vel_components=[],
    mu_pvf=-1e4,
    var_pvf=1e7,
    norm_flux_unc=1,
    blue_fl=[1, .1],
    red_fl=[1, .1],
):
    '''log-posterior probability

    See lnprior() and lnlike_gaussian_abs() for details
    '''

    ln_prior = lnprior(theta, mu_pvf, var_pvf, blue_fl, red_fl)
    ln_like = lnlike_gaussian_abs(theta, blue_vel, red_vel, vel_rf, norm_flux,
                                  delta_vel_components, norm_flux_unc)
    return ln_prior + ln_like


######################### MCMC visualization ##############################


def plot_MCMC(sampler, nburn, nplot=None):
    '''plot walker chains and corner plots

    Parameters
    ----------
    sampler : emcee EnsembleSampler object
        emcee affine-invariant multi-chain MCMC sampler

    nburn : int
        number of "burn-in" steps for the MCMC chains

    nplot : int, default=None
        number of chains to show in the visualization.
    '''

    ndim = sampler.get_chain().shape[2]
    if ndim == 5:
        paramsNames = [
            r'$\mathrm{Blue\ edge\ flux}$', r'$\mathrm{Red\ edge\ flux}$',
            r'$v_\mathrm{pvf}$', r'$\ln(\sigma^2_\mathrm{pvf})$', r'$A_\mathrm{pvf}$'
        ]
    elif ndim == 8:
        paramsNames = [
            r'$\mathrm{Blue\ edge\ flux}$', r'$\mathrm{Red\ edge\ flux}$',
            r'$v_\mathrm{pvf}$', r'$\ln(\sigma^2_\mathrm{pvf})$', r'$A_\mathrm{pvf}$',
            r'$v_\mathrm{hvf}$', r'$\ln(\sigma^2_\mathrm{hvf})$', r'$A_\mathrm{hvf}$'
        ]
    else:
        print('Error: wrong parameter number!')

    plotChains(sampler, 0, paramsNames, nplot)
    plt.tight_layout()
    plt.show()

    samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
    fig = corner.corner(samples,
                        labels=paramsNames,
                        quantiles=[0.16, 0.50, 0.84],
                        show_titles=True)


def plotChains(sampler, nburn, paramsNames, nplot=None):
    '''Plot individual chains from the emcee MCMC sampler

    Parameters
    ----------
    sampler : emcee EnsembleSampler object
        emcee affine-invariant multi-chain MCMC sampler

    nburn : int
        number of "burn-in" steps for the MCMC chains

    paramsNames : array_like
        names of the parameters to be shown

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
    nwalkers = sampler.get_chain().shape[1]

    fig, ax = plt.subplots(Nparams + 1,
                           1,
                           figsize=(12, 3 * (Nparams + 1)),
                           sharex=True)
    fig.subplots_adjust(hspace=0)
    ax[0].set_title(r'$\mathrm{Chains}$')
    xplot = np.arange(sampler.get_chain().shape[0])

    if nplot is None:
        nplot = nwalkers
    selected_walkers = np.random.choice(range(nwalkers), nplot, replace=False)
    for i, p in enumerate(paramsNames):
        for w in selected_walkers:
            burn = ax[i].plot(xplot[:nburn],
                              sampler.get_chain()[:nburn, w, i],
                              alpha=0.4,
                              lw=0.7,
                              zorder=1)
            ax[i].plot(xplot[nburn:],
                       sampler.get_chain(discard=nburn)[:, w, i],
                       color=burn[0].get_color(),
                       alpha=0.8,
                       lw=0.7,
                       zorder=1)

            ax[i].set_ylabel(p)
            if i == Nparams - 1:
                ax[i + 1].plot(xplot[:nburn],
                               sampler.get_log_prob()[:nburn, w],
                               color=burn[0].get_color(),
                               alpha=0.4,
                               lw=0.7,
                               zorder=1)
                ax[i + 1].plot(xplot[nburn:],
                               sampler.get_log_prob(discard=nburn)[:, w],
                               color=burn[0].get_color(),
                               alpha=0.8,
                               lw=0.7,
                               zorder=1)
                ax[i + 1].set_ylabel(r'$\ln P$')

    return ax
