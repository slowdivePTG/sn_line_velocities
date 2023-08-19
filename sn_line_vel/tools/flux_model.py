###################### conversion between wav and vel ##########################

import numpy as np


def velocity_rf(lambda_rf, lambda_0):
    """convert rest-frame wavelength to relative velocity"""
    c = 2.99792458e5
    v = c * ((lambda_rf / lambda_0) ** 2 - 1) / ((lambda_rf / lambda_0) ** 2 + 1)

    return v


def wv_rf(vel_rf, lambda_0):
    """convert relative velocity to rest-frame wavelength"""
    c = 2.99792458e5
    beta = vel_rf / c
    wv = lambda_0 * ((1 + beta) / (1 - beta)) ** 0.5

    return wv


def velocity_rf_line(lambda_0, lambda_1, vel):
    """get the relative velocity of a feature assuming another line

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
    """

    c = 2.99792458e5
    lambda_rf = ((vel / c + 1) / (-vel / c + 1)) ** 0.5 * lambda_0
    vel_1 = velocity_rf(lambda_rf, lambda_1)

    return vel_1


###################### spectroscopic models ##########################


def calc_gauss(mean_vel, sig_vel, amplitude, vel_rf):
    """Gaussian profile"""
    gauss = (
        amplitude
        / np.sqrt(2 * np.pi * sig_vel**2)
        * np.exp(-0.5 * (vel_rf - mean_vel) ** 2 / sig_vel**2)
    )
    return gauss


def calc_lorentzian(mean_vel, gamma_vel, amplitude, vel_rf):
    """Lorentzian profile"""
    lorentz = (
        amplitude
        / np.sqrt(np.pi * gamma_vel)
        / (1 + (vel_rf - mean_vel) ** 2 / gamma_vel**2)
    )
    return lorentz


def calc_model_flux(
    theta,
    vel_resolution,
    rel_strength,
    lambda_0,
    blue_vel,
    red_vel,
    vel_rf,
    lines=[],
    model="Gauss",
):
    """Calculate normalized flux based on a Gaussian/Lorentzian model

    Parameters
    ----------
    theta : array_like
        fitting parameters: flux at the blue edge, flux at the
        red edge, (mean of relative velocity, log variance,
        amplitude) * Number of velocity components

    vel_resolution : float
        the spectral resolution of the spectrograph in terms of
        the velocity - will broaden the lines

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

    model : string, default="Gauss"
        "Gauss" for a Gaussian profile, "Lorentz" for a Lorentzian profile

    Returns
    -------
    model_flux : array_like
        predicted (normalized) flux at each relative radial
        velocity
    """

    y1, y2 = theta[:2]
    m = (y2 - y1) / (red_vel - blue_vel)
    b = y2 - m * red_vel

    model_flux = m * vel_rf + b
    for k in range(len(theta) // 3):
        mean_vel, lnsig, amplitude = theta[3 * k + 2 : 3 * k + 5]
        sig_vel = np.exp(lnsig)

        # the spectral resolution of the spectrograph limits the line width
        sig_instru = (sig_vel**2 + vel_resolution**2) ** 0.5

        for rel_s, li in zip(rel_strength[k], lines[k][:-1]):
            vel = velocity_rf_line(li, lambda_0, mean_vel)
            if model == "Gauss":
                calc = calc_gauss
            elif model == "Lorentz":
                calc = calc_lorentzian
            else:
                raise NameError("Line model not supported (optional: Gauss & Lorentz)")
            model_flux += rel_s * calc(vel, sig_instru, amplitude, vel_rf)
        vel = velocity_rf_line(lines[k][-1], lambda_0, mean_vel)
        model_flux += calc_gauss(vel, sig_instru, amplitude, vel_rf)
    return model_flux
