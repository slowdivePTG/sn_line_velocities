import numpy as np


def data_binning(
    data, size=2, min_bin=1, spec_resolution=0, sigma_clip=3
):  # size - Angstrom
    """
    binning spectroscopic data with a finite spectral resolution
    """
    from astropy.stats import mad_std

    data_bin = []
    i = 0
    while i < len(data):
        j = i
        while j < len(data):
            if data[j, 0] < data[i, 0] + size:
                j += 1
            else:
                break
        temp = data[i:j, :]
        if len(temp) >= min_bin:
            if len(temp) > 1:
                arg = np.arange(len(temp))
                X, Y, Yerr = temp[arg, 0], temp[arg, 1], temp[arg, 2]
                if sigma_clip != None:
                    clip = np.abs(Y - np.median(Y)) <= sigma_clip * mad_std(Y)
                    X, Y, Yerr = X[clip], Y[clip], Yerr[clip]
                X_bin = X.mean()
                weight = Yerr ** (-2.0) / (Yerr ** (-2.0)).sum()
                Y_bin = (Y * weight).sum()
                ivar = (Yerr ** (-2.0)).sum()
                if spec_resolution <= 0:
                    Yerr_bin = ivar**-0.5
                else:
                    Yerr_bin_sq = 0
                    for idx1 in range(len(X)):
                        for idx2 in range(len(X)):
                            Yerr_bin_sq += (Yerr[idx1] * Yerr[idx2]) ** -1 * np.exp(
                                -((X[idx1] - X[idx2]) ** 2)
                                / (2 * (spec_resolution / 2.355) ** 2)
                            )
                    Yerr_bin = ivar**-1 * Yerr_bin_sq**0.5
                i = j
            else:
                X_bin = data[i, 0]
                Y_bin = data[i, 1]
                Yerr_bin = data[i, 2]
                i += 1
            data_bin.append([X_bin, Y_bin, Yerr_bin])
        else:
            for t in temp.reshape(len(temp.flatten()) // 3, 3):
                data_bin.append(t)
            i = j
    return np.array(data_bin)


def plot_box_spec(wave, flux):
    flux_plot = np.repeat(flux, 2)
    wv_plot = wave.copy()
    wv_plot[:-1] += np.diff(wave) / 2
    wv_plot = np.append(
        wave[0] - (wave[1] - wave[0]) / 2,
        np.append(np.repeat(wv_plot[0:-1], 2), wave[-1] + (wave[-1] - wave[-2]) / 2),
    )

    return wv_plot, flux_plot
