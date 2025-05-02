# ruff: noqa: PGH004
# ruff: noqa
# type: ignore
# Code créé par Gabriel Taillon le 7 Mai 2018
# From https://github.com/Gabinou/2DKS
#  Kolmogorov-Smyrnov Test extended to two dimensions.
# References:s
#  [1] Peacock, J. A. (1983). Two-dimensional goodness-of-fit testing
#  in astronomy. Monthly Notices of the Royal Astronomical Society,
#  202(3), 615-627.
#  [2] Fasano, G., & Franceschini, A. (1987). A multidimensional version of
#  the Kolmogorov–Smirnov test. Monthly Notices of the Royal Astronomical
#  Society, 225(1), 155-170.
#  [3] Flannery, B. P., Press, W. H., Teukolsky, S. A., & Vetterling, W.
#  (1992). Numerical recipes in C. Press Syndicate of the University
#  of Cambridge, New York, 24, 78.
import inspect

import numpy as np
import scipy.stats


def CountQuads(
    Arr2D: np.ndarray, point: np.ndarray
) -> tuple[float, float, float, float]:
    """
    Compute probabilities by counting points in quadrants.

    Computes the probabilities of finding points in each of the 4 quadrants
    defined by a vertical and horizontal line crossing the given `point`.
    The probabilities are determined by counting the proportion of points
    from `Arr2D` that fall into each quadrant.

    Parameters
    ----------
    Arr2D : np.ndarray
        Array of 2D points (shape N x 2) to be counted.
    point : np.ndarray
        A 1D array or list with 2 elements representing the center (x, y)
        of the 4 quadrants.

    Returns
    -------
    tuple[float, float, float, float]
        A tuple containing four floats (fpp, fnp, fpn, fnn), representing the
        normalized fractions (probabilities) of points in each quadrant:
        - fpp: Fraction in the positive-x, positive-y quadrant.
        - fnp: Fraction in the negative-x, positive-y quadrant.
        - fpn: Fraction in the positive-x, negative-y quadrant.
        - fnn: Fraction in the negative-x, negative-y quadrant.

    Raises
    ------
    TypeError
        If `point` or `Arr2D` are not list-like or numpy arrays, or if
        `point` does not have 2 elements, or if `Arr2D` is not 2D.

    """
    if isinstance(point, list):
        point = np.asarray(np.ravel(point))
    elif type(point).__module__ + type(point).__name__ == "numpyndarray":
        point = np.ravel(point.copy())
    else:
        raise TypeError("Input point is neither list nor numpyndarray")
    if len(point) != 2:
        raise TypeError("Input point must have exactly 2 elements")
    if isinstance(Arr2D, list):
        Arr2D = np.asarray(Arr2D)
    elif type(Arr2D).__module__ + type(Arr2D).__name__ == "numpyndarray":
        pass
    else:
        raise TypeError("Input Arr2D is neither list nor numpyndarray")
    if Arr2D.shape[1] > Arr2D.shape[0]:  # Reshape to A[row,column]
        Arr2D = Arr2D.copy().T
    if Arr2D.shape[1] != 2:
        raise TypeError("Input Arr2D is not 2D")
    # The pp of Qpp refer to p for 'positive' and n for 'negative' quadrants.
    # In order. first subscript is x, second is y.
    Qpp = Arr2D[(Arr2D[:, 0] > point[0]) & (Arr2D[:, 1] > point[1]), :]
    Qnp = Arr2D[(Arr2D[:, 0] < point[0]) & (Arr2D[:, 1] > point[1]), :]
    Qpn = Arr2D[(Arr2D[:, 0] > point[0]) & (Arr2D[:, 1] < point[1]), :]
    Qnn = Arr2D[(Arr2D[:, 0] < point[0]) & (Arr2D[:, 1] < point[1]), :]
    # Normalized fractions:
    ff = 1.0 / len(Arr2D)
    fpp = len(Qpp) * ff
    fnp = len(Qnp) * ff
    fpn = len(Qpn) * ff
    fnn = len(Qnn) * ff
    # NOTE:  all the f's are supposed to sum to 1.0. Float representation
    # cause SOMETIMES sum to 1.000000002 or something. I don't know how to
    # test for that reliably, OR what to do about it yet. Keep in mind.
    return fpp, fnp, fpn, fnn


def FuncQuads(func2D, point, xlim, ylim, rounddig=4):
    """
    Compute probabilities by integrating a density function in quadrants.

    Computes the probabilities of finding points in each of the 4 quadrants
    defined by a vertical and horizontal line crossing the given `point`.
    The probabilities are determined by numerically integrating the 2D density
    function `func2D` over each quadrant within the specified limits.

    Parameters
    ----------
    func2D : callable
        A 2D density function that accepts two arguments (x, y).
    point : list or np.ndarray
        A 1D array or list with 2 elements representing the center (x, y)
        of the 4 quadrants.
    xlim : list or np.ndarray
        A list or array with 2 elements defining the integration limits for x.
    ylim : list or np.ndarray
        A list or array with 2 elements defining the integration limits for y.
    rounddig : int, optional
        Number of decimal digits to round the resulting probabilities to,
        by default 4.

    Returns
    -------
    tuple[float, float, float, float]
        A tuple containing four floats (fpp, fnp, fpn, fnn), representing the
        integrated probabilities in each quadrant, normalized by the total integral:
        - fpp: Probability in the positive-x, positive-y quadrant.
        - fnp: Probability in the negative-x, positive-y quadrant.
        - fpn: Probability in the positive-x, negative-y quadrant.
        - fnn: Probability in the negative-x, negative-y quadrant.

    Raises
    ------
    TypeError
        If `func2D` is not a callable function with 2 arguments, or if
        `point`, `xlim`, or `ylim` are not list-like or numpy arrays with
        exactly 2 elements, or if limits in `xlim` or `ylim` are equal.

    """
    if callable(func2D):
        if len(inspect.getfullargspec(func2D)[0]) != 2:
            raise TypeError("Input func2D is not a function with 2 arguments")
    else:
        raise TypeError("Input func2D is not a function")
    # If xlim, ylim and point are not lists or ndarray, exit.
    if isinstance(point, list):
        point = np.asarray(np.ravel(point))
    elif type(point).__module__ + type(point).__name__ == "numpyndarray":
        point = np.ravel(point.copy())
    else:
        raise TypeError("Input point is not a list or numpyndarray")
    if len(point) != 2:
        raise TypeError("Input point has not exactly 2 elements")
    if isinstance(xlim, list):
        xlim = np.asarray(np.sort(np.ravel(xlim)))
    elif type(xlim).__module__ + type(xlim).__name__ == "numpyndarray":
        xlim = np.sort(np.ravel(xlim.copy()))
    else:
        raise TypeError("Input xlim is not a list or ndarray")
    if len(xlim) != 2:
        raise TypeError("Input xlim has not exactly 2 elements")
    if xlim[0] == xlim[1]:
        raise TypeError("Input xlim[0] should be different to xlim[1]")
    if isinstance(ylim, list):
        ylim = np.asarray(np.sort(np.ravel(ylim)))
    elif type(ylim).__module__ + type(ylim).__name__ == "numpyndarray":
        ylim = np.sort(np.ravel(ylim.copy()))
    else:
        raise TypeError("Input ylim is not a list or ndarray")
    if len(ylim) != 2:
        raise TypeError("Input ylim has not exactly 2 elements")
    if ylim[0] == ylim[1]:
        raise TypeError("Input ylim[0] should be different to ylim[1]")
    # Numerical integration to find the quadrant probabilities.
    totInt = scipy.integrate.dblquad(
        func2D, *xlim, lambda x: np.amin(ylim), lambda x: np.amax(ylim)
    )[0]
    Qpp = scipy.integrate.dblquad(
        func2D, point[0], np.amax(xlim), lambda x: point[1], lambda x: np.amax(ylim)
    )[0]
    Qpn = scipy.integrate.dblquad(
        func2D, point[0], np.amax(xlim), lambda x: np.amin(ylim), lambda x: point[1]
    )[0]
    Qnp = scipy.integrate.dblquad(
        func2D, np.amin(xlim), point[0], lambda x: point[1], lambda x: np.amax(ylim)
    )[0]
    Qnn = scipy.integrate.dblquad(
        func2D, np.amin(xlim), point[0], lambda x: np.amin(ylim), lambda x: point[1]
    )[0]
    fpp = round(Qpp / totInt, rounddig)
    fnp = round(Qnp / totInt, rounddig)
    fpn = round(Qpn / totInt, rounddig)
    fnn = round(Qnn / totInt, rounddig)
    return (fpp, fnp, fpn, fnn)


def Qks(alam, iter=100, prec=1e-17):
    """
    Compute the Kolmogorov-Smirnov probability function Q(lambda).

    Calculates the significance level for a given KS statistic `alam` (D).
    This function is based on the approximation given in Numerical Recipes in C,
    page 623. It represents the probability that the KS statistic will exceed
    the observed value `alam` under the null hypothesis.

    Parameters
    ----------
    alam : float
        The KS statistic D (or a related value, often D * sqrt(N_eff)).
    iter : int, optional
        Maximum number of iterations for the series summation, by default 100.
    prec : float, optional
        Convergence precision. The summation stops if the absolute value of
        the term to add is less than `prec`, by default 1e-17.

    Returns
    -------
    float
        The significance level P(D > observed) associated with `alam`.
        Returns 1.0 if the series does not converge within `iter` iterations
        or if the result exceeds 1.0. Returns 0.0 if the result is below `prec`.

    Raises
    ------
    TypeError
        If `alam` is not an integer or float.

    """
    # If j iterations are performed, meaning that toadd
    # is still 2 times larger than the precision.
    if isinstance(alam, int) | isinstance(alam, float):
        pass
    else:
        raise TypeError("Input alam is neither int nor float")
    toadd = [1]
    qks = 0.0
    j = 1
    while (j < iter) & (abs(toadd[-1]) > prec * 2):
        toadd.append(2.0 * (-1.0) ** (j - 1.0) * np.exp(-2.0 * j**2.0 * alam**2.0))
        qks += toadd[-1]
        j += 1
    if (j == iter) | (qks > 1):  # If no convergence after j iter, return 1.0
        return 1.0
    if qks < prec:
        return 0.0
    return qks


def ks2d2s(Arr2D1: np.ndarray, Arr2D2: np.ndarray) -> tuple[float, float]:
    """
    Perform the 2-dimensional, 2-sample Kolmogorov-Smirnov test.

    Tests the null hypothesis that two independent 2D samples, `Arr2D1` and
    `Arr2D2`, are drawn from the same underlying probability distribution.
    This implementation is based on the methods described by Peacock (1983)
    and Fasano & Franceschini (1987).

    Parameters
    ----------
    Arr2D1 : np.ndarray
        First 2D sample array (shape N1 x 2).
    Arr2D2 : np.ndarray
        Second 2D sample array (shape N2 x 2).

    Returns
    -------
    tuple[float, float]
        d : float
            The 2D KS statistic, representing the maximum difference found
            between the cumulative distributions in any of the four quadrants,
            evaluated at all data points.
        prob : float
            The significance level (p-value) of the observed statistic `d`.
            A small `prob` indicates that the two samples are significantly
            different.

    Raises
    ------
    TypeError
        If `Arr2D1` or `Arr2D2` are not numpy arrays or are not 2D.

    """
    if not isinstance(Arr2D1, np.ndarray):
        raise TypeError("Input Arr2D1 is not a numpyndarray")
    if Arr2D1.shape[1] > Arr2D1.shape[0]:
        Arr2D1 = Arr2D1.copy().T
    if not isinstance(Arr2D2, np.ndarray):
        raise TypeError("Input Arr2D2 is not a numpyndarray")

    if Arr2D2.shape[1] > Arr2D2.shape[0]:
        Arr2D2 = Arr2D2.copy().T

    if Arr2D1.shape[1] != 2:
        raise TypeError("Input Arr2D1 is not 2D")
    if Arr2D2.shape[1] != 2:
        raise TypeError("Input Arr2D2 is not 2D")

    d1, d2 = 0.0, 0.0
    for point1 in Arr2D1:
        fpp1, fmp1, fpm1, fmm1 = CountQuads(Arr2D1, point1)
        fpp2, fmp2, fpm2, fmm2 = CountQuads(Arr2D2, point1)
        d1 = max(d1, abs(fpp1 - fpp2))
        d1 = max(d1, abs(fpm1 - fpm2))
        d1 = max(d1, abs(fmp1 - fmp2))
        d1 = max(d1, abs(fmm1 - fmm2))
    for point2 in Arr2D2:
        fpp1, fmp1, fpm1, fmm1 = CountQuads(Arr2D1, point2)
        fpp2, fmp2, fpm2, fmm2 = CountQuads(Arr2D2, point2)
        d2 = max(d2, abs(fpp1 - fpp2))
        d2 = max(d2, abs(fpm1 - fpm2))
        d2 = max(d2, abs(fmp1 - fmp2))
        d2 = max(d2, abs(fmm1 - fmm2))
    d = (d1 + d2) / 2.0
    sqen = np.sqrt(len(Arr2D1) * len(Arr2D2) / (len(Arr2D1) + len(Arr2D2)))
    R1 = scipy.stats.pearsonr(Arr2D1[:, 0], Arr2D1[:, 1]).correlation
    R2 = scipy.stats.pearsonr(Arr2D2[:, 0], Arr2D2[:, 1]).correlation
    RR = np.sqrt(1.0 - (R1 * R1 + R2 * R2) / 2.0)
    prob = Qks(d * sqen / (1.0 + RR * (0.25 - 0.75 / sqen)))
    # Small values of prob show that the two samples are significantly
    # different. Prob is the significance level of an observed value of d.
    # NOT the same as the significance level that ou set and compare to D.
    return d, prob


def ks2d1s(Arr2D, func2D, xlim=[], ylim=[]):
    """
    Perform the 2-dimensional, 1-sample Kolmogorov-Smirnov test.

    Tests the null hypothesis that a 2D sample `Arr2D` is drawn from a
    given 2D probability density distribution `func2D`.

    Parameters
    ----------
    Arr2D : np.ndarray
        The 2D sample array (shape N x 2).
    func2D : callable
        The theoretical 2D probability density function func(x, y).
    xlim : list or np.ndarray, optional
        Integration limits for the x-dimension. If empty, defaults are
        calculated based on the range of `Arr2D`.
    ylim : list or np.ndarray, optional
        Integration limits for the y-dimension. If empty, defaults are
        calculated based on the range of `Arr2D`.

    Returns
    -------
    tuple[float, float]
        d : float
            The 2D KS statistic, representing the maximum difference between
            the empirical distribution (from `Arr2D`) and the theoretical
            distribution (`func2D`) in any of the four quadrants, evaluated
            at all data points.
        prob : float
            The significance level (p-value) of the observed statistic `d`.
            A small `prob` indicates that the sample is significantly
            different from the theoretical distribution.

    Raises
    ------
    TypeError
        If `func2D` is not a callable function with 2 arguments, or if
        `Arr2D` is not a 2D numpy array.

    """
    if callable(func2D):
        if len(inspect.getfullargspec(func2D)[0]) != 2:
            raise TypeError("Input func2D is not a function with 2 input arguments")
    else:
        raise TypeError("Input func2D is not a function")
    if type(Arr2D).__module__ + type(Arr2D).__name__ == "numpyndarray":
        pass
    else:
        raise TypeError("Input Arr2D is neither list nor numpyndarray")
    print(Arr2D.shape)
    if Arr2D.shape[1] > Arr2D.shape[0]:
        Arr2D = Arr2D.copy().T
    if Arr2D.shape[1] != 2:
        raise TypeError("Input Arr2D is not 2D")
    if xlim == []:
        xlim.append(
            np.amin(Arr2D[:, 0]) - abs(np.amin(Arr2D[:, 0]) - np.amax(Arr2D[:, 0])) / 10
        )
        xlim.append(
            np.amax(Arr2D[:, 0]) - abs(np.amin(Arr2D[:, 0]) - np.amax(Arr2D[:, 0])) / 10
        )
    if ylim == []:
        ylim.append(
            np.amin(Arr2D[:, 1]) - abs(np.amin(Arr2D[:, 1]) - np.amax(Arr2D[:, 1])) / 10
        )

        ylim.append(
            np.amax(Arr2D[:, 1]) - abs(np.amin(Arr2D[:, 1]) - np.amax(Arr2D[:, 1])) / 10
        )
    d = 0
    for point in Arr2D:
        fpp1, fmp1, fpm1, fmm1 = FuncQuads(func2D, point, xlim, ylim)
        fpp2, fmp2, fpm2, fmm2 = CountQuads(Arr2D, point)
        d = max(d, abs(fpp1 - fpp2))
        d = max(d, abs(fpm1 - fpm2))
        d = max(d, abs(fmp1 - fmp2))
        d = max(d, abs(fmm1 - fmm2))
    sqen = np.sqrt(len(Arr2D))
    R1 = scipy.stats.pearsonr(Arr2D[:, 0], Arr2D[:, 1])[0]
    RR = np.sqrt(1.0 - R1**2)
    prob = Qks(d * sqen / (1.0 + RR * (0.25 - 0.75 / sqen)))
    return d, prob
