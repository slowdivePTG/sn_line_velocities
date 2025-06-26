import numpy as np
import warnings

from .SpecLine import SpecLine
from .tools.dust_extinction import calALambda
from .tools._plt import plt


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

    def __init__(
        self, spec1D, z=0, ebv=0, snr=None, spec_resolution=5, force_pos_flux=False
    ):
        """Constructor

        Parameters
        ----------
        spec1D : str
            the spectrum file (directory + filename)
            the inputs should include 3 columns (wavelengths, fluxes, flux uncertainties)
            if only the first two columns are provided, the uncertainties will be estimated with an S/N

        z : float (default=0)
            host galaxy redshift

        ebv : float (default=0)
            E(B-V), Galactic reddening

        snr : float, default=None
            the assigned S/N for spectra with no flux errors

        spec_resolution : float, default=5
            the spectral resolution of the spectrum in Angstrom

        force_pos_flux : bool, default=False
            remove all the non-positive flux measurements
        """

        spec = np.loadtxt(spec1D, comments="#", delimiter=None, unpack=False)

        wv = spec[:, 0]
        wv_rf = wv / (1 + z)
        aLambda = calALambda(wv, RV=3.1, EBV=ebv)
        fl = spec[:, 1] * 10 ** (0.4 * aLambda)

        if snr is None:
            if spec.shape[1] == 3:
                fl_unc = spec[:, 2] * 10 ** (0.4 * aLambda)
            else:
                raise ValueError("No flux uncertainty in the datafile!")
        else:
            # the same uncertainty is assigned to all the flux measurements
            warnings.warn("snr = {:.1f} assigned.".format(snr))
            fl_unc = (
                np.ones_like(fl)
                * (np.nanmedian(fl) / snr)
                * (
                    np.where(fl > np.nanmedian(fl) / 10, fl, np.nanmedian(fl) / 10)
                    / np.nanmedian(fl)
                )
                ** -0.5
            )

        # make sure flux measurements are positive
        pos_flux = (fl > 0) | (not force_pos_flux)
        self.fl = fl[pos_flux]
        self.wv_rf = wv_rf[pos_flux]
        self.fl_unc = fl_unc[pos_flux]

        # self.snr = snr
        self.spec_resolution = spec_resolution

        self.line: dict[str, SpecLine] = {}

    def add_line(
        self,
        name,
        blue_edge,
        red_edge,
        lines=[],
        rel_strength=[],
        free_rel_strength=[],
        line_model="Gauss",
        mask=[],
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
            line_model=line_model,
            mask=mask,
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

        _, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        ax.plot(self.wv_rf[line_region], self.fl[line_region], color="0.5", lw=2)
        ax.errorbar(
            self.wv_rf[line_region],
            self.fl[line_region],
            yerr=self.fl_unc[line_region],
            fmt="o",
            capsize=2,
            elinewidth=1,
        )
        ax.set_xlabel(r"$\mathrm{Wavelength}\ [\mathrm{\r{A}}]$")
        ax.set_ylabel(r"$\mathrm{Flux}$")
        return ax
