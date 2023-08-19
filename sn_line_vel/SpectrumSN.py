import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from logging import raiseExceptions

from .SpecLine import SpecLine

mpl.rcParams["text.usetex"] = True  # only when producing plots for publication
mpl.rcParams["font.family"] = "times new roman"
mpl.rcParams["font.size"] = "25"
mpl.rcParams["xtick.labelsize"] = "20"
mpl.rcParams["ytick.labelsize"] = "20"


##################### SpectrumSN class ##########################


class SpectrumSN(object):
    """1D optical spectrum

    Attributes
    ----------

    fl : array_like
        flux (in arbitrary units)

    wv_rf : array_like
        wavelength [angstrom] in the host galaxy's rest frame:
        wv_rf = wavelength / (1 + z)

    fl_unc : array_like
        uncertainty in flux (in arbitrary units)

    line : dict
        a dictionary of various lines (AbsorbLine objects)

    Methods
    -------
    add_line(name, blue_edge, red_edge, lines=[])) :
        Add one (series of) absorption line(s)

    plot_line_region(blue_edge, red_edge) :
        plot the spectrum in the line region

    get_flux_at_lambda(lambda_0, delta_l=50) :
        pget the flux and its uncertainty at some given wavelength
    """

    def __init__(self, spec1D, z=0, snr=20, spec_resolution=5, force_pos_flux=False):
        """Constructor

        Parameters
        ----------
        spec1D : str
            the spectrum file (directory + filename)
            the inputs should include 3 columns (wavelengths, fluxes, flux uncertainties)
            if only the first two columns are provided, the uncertainties will be estimated with an S/N

        z : float (default=0)
            host galaxy redshift

        snr : float, default=20
            the assigned S/N for spectra with no flux errors

        spec_resolution : float, default=5
            the spectral resolution of the spectrum in Angstrom

        force_pos_flux : bool, default=False
            remove all the non-positive flux measurements
        """

        spec_df = pd.read_csv(spec1D, comment="#", delim_whitespace=True, header=None)

        wv = spec_df[0].values
        wv_rf = wv / (1 + z)
        fl = spec_df[1].values

        try:
            fl_unc = spec_df[2].values
        except:
            warnings.warn("No flux uncertainty in the datafile!")
            # the same uncertainty is assigned to all the flux measurements
            warnings.warn(f"Manual snr = {snr} assigned.")
            fl_unc = np.ones_like(fl) * (np.nanmedian(fl) / snr)

        # make sure flux measurements are positive
        pos_flux = (fl > 0) | (force_pos_flux)
        self.fl = fl[pos_flux]
        self.wv_rf = wv_rf[pos_flux]
        self.fl_unc = fl_unc[pos_flux]

        # self.snr = snr
        self.spec_resolution = spec_resolution

        self.line = {}

    def add_line(
        self,
        name,
        blue_edge,
        red_edge,
        lines=[],
        rel_strength=[],
        free_rel_strength=[],
        bin=False,
        bin_size=1,
        plot_region=False,
    ):
        """Add one (series of) absorption line(s)

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
        """

        self.line[name] = SpecLine(
            np.array([self.wv_rf, self.fl, self.fl_unc]).T,
            spec_resolution=self.spec_resolution,
            blue_edge=blue_edge,
            red_edge=red_edge,
            lines=lines,
            rel_strength=rel_strength,
            free_rel_strength=free_rel_strength,
            bin=bin,
            bin_size=bin_size,
        )
        if plot_region:
            self.plot_line_region(blue_edge=blue_edge, red_edge=red_edge)
            plt.show()

    def plot_line_region(self, blue_edge, red_edge):
        """plot the spectrum in the line region

        Parameters
        ----------
        blue_edge, red_edge : float
            the wavelength [angstrom] (host galaxy frame)
            at the blue/red edge

        Returns
        -------
        ax : matplotlib.axes
            the axes
        """
        line_region = np.where((self.wv_rf < red_edge) & (self.wv_rf > blue_edge))[0]

        plt.figure(figsize=(8, 6))
        plt.plot(self.wv_rf[line_region], self.fl[line_region], color="0.5", lw=2)
        plt.errorbar(
            self.wv_rf[line_region],
            self.fl[line_region],
            yerr=self.fl_unc[line_region],
            fmt="o",
            capsize=2,
            elinewidth=1,
        )
        plt.tight_layout()
        return plt.gca()
