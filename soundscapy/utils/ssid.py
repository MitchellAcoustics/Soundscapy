# %%
import ndtest
import numpy as np
import pandas as pd
from scipy.stats import truncnorm


def get_truncated_normal(
    mean: float = 0, std: float = 1, low: float = -1, upp: float = 1
) -> truncnorm.rvs:
    """
    Sample from a truncated normal distribution.
    This custom function wraps the scipy function to make it simpler to use.

    Parameters
    ----------
        mean: float
            Set the mean of the distribution
        std: float
            Set the standard deviation of the distribution
        low: float
            Set the lower bound of the truncated normal distribution
        upp: float
            Specify the upper bound of the truncated normal distribution

    Returns
    -------
        A truncated normal distribution with the given mean, standard deviation, lower and upper bounds
    """
    return truncnorm((low - mean) / std, (upp - mean) / std, loc=mean, scale=std)


def truncnorm_stats(
    data: np.array = None,
    mu: float = None,
    sigma: float = None,
    low: float = -1,
    upp: float = 1,
) -> tuple:
    """
    The truncnorm_stats function computes the mean and standard deviation of a truncated normal distribution.

    This can be calculated either directly from a sample of data, or from the mean and standard deviation of the
    associated normal distribution. If the data is not specified, then the mean and standard deviation must be supplied.
    Parameters
    ----------
        data: np.array
            Calculate the mean and standard deviation of the data
        mu: float
            Specify the mean of the associated normal distribution
        sigma: float
            Specify the standard deviation of the associated normal distribution
        low: float
            Specify the lower bound of the truncated normal distribution
        upp: float
            Specify the upper bound of the truncated normal distribution
    Returns
    -------
    tuple
        The mean and standard deviation of the truncated normal distribution
    """
    if data is None:
        if mu is None or sigma is None:
            raise ValueError("Must specify either data or mu and sigma")
    elif mu is None and sigma is None:
        mu = data.mean()
        sigma = data.std()
    else:
        assert mu is not None and sigma is not None
    za = (low - mu) / sigma
    zb = (upp - mu) / sigma
    mean = truncnorm.mean(za, zb, loc=mu, scale=sigma)
    std = truncnorm.std(za, zb, loc=mu, scale=sigma)
    return mean, std


def dist_generation(
    pl_mean: float,
    ev_mean: float,
    pl_std: float,
    ev_std: float,
    n: int = 1000,
    dist_type: str = "normal",
) -> tuple:
    # Generate a distribution from ISOPl and ISOEv means and stds
    """
    The dist_generation function generates a distribution of ISOPl and ISOEv values
    from the mean and standard deviation of each. The function takes in four parameters:
        pl_mean (float): The mean value for ISOPl.
        ev_mean (float): The mean value for ISOEv.
        pl_std (float):  The standard deviation for ISOPl.
        ev_std (float):  The standard deviation for ISOEv.

    Parameters
    ----------
        pl_mean: float
            Set the mean of the distribution
        ev_mean: float
            Set the mean of the ev distribution
        pl_std: float
            Set the standard deviation of the distribution
        ev_std: float
            Set the standard deviation of the ev distribution
        n: int
            Determine the number of samples to be generated
        dist_type: str
            Specify the type of distribution to generate
            Can be one of "normal" or "truncnorm". Default is "truncnorm"
    Returns
    -------
        A tuple of two arrays, one for each distribution
    """
    if dist_type == "normal":
        pl = np.random.normal(pl_mean, pl_std, n)
        ev = np.random.normal(ev_mean, ev_std, n)

    elif dist_type == "truncnorm":
        pl_mean, pl_std = truncnorm_stats(
            data=None, mu=pl_mean, sigma=pl_std, low=-1, upp=1
        )
        ev_mean, ev_std = truncnorm_stats(
            data=None, mu=ev_mean, sigma=ev_std, low=-1, upp=1
        )
        pl = get_truncated_normal(mean=pl_mean, std=pl_std, low=-1, upp=1).rvs(n)
        ev = get_truncated_normal(mean=ev_mean, std=ev_std, low=-1, upp=1).rvs(n)

    return pl, ev


def ssid_calc(
    test_x: np.array, test_y: np.array, target_x: np.array, target_y: np.array
) -> int:
    """
    The ssid_calc function takes in two sets of data, test_x and test_y, and compares them to a target set of data
    target_x and target_y. The function returns the SSID value as an integer between 0-100.

    Parameters
    ----------
        test_x: np.array
            Pass the x-values of the test data
        test_y: np.array
            Pass the y-values of the test data
        target_x: np.array
            Pass the x-values of the target data
        target_y: np.array
            Pass the y-values of the target data

    Returns
    -------
        The ssid value for the test and target distributions
    """
    P, D = ndtest.ks2d2s(test_x, test_y, target_x, target_y, extra=True)
    return int((1 - D) * 100)


def ssid(
    test_df: pd.DataFrame,
    pl_mean: float,
    ev_mean: float,
    pl_std: float,
    ev_std: float,
    n: int = 1000,
    dist_type: str = "truncnorm",
) -> int:
    """
    The ssid function takes in a dataframe of ISOPleasant & ISOEventful values to test, and the
    mean and standard deviation of the target pleasantness and eventfulness distributions.
    It then generates n samples from these distributions, calculates their SSID with respect to
    the target values, and returns this value.

    Parameters
    ----------
        test_df: pd.DataFrame
            Pass in the dataframe of the test distribution
        pl_mean: float
            Set the mean of the pleasantness distribution
        ev_mean: float
            Set the mean of the eventful distribution
        pl_std: float
            Set the standard deviation of the pleasantness distribution
        ev_std: float
            Define the standard deviation of the eventful distribution
        n: int
            Set the number of samples to be generated
        dist_type: str
            Specify the distribution type

    Returns
    -------
        A single number, the SSID

    """
    test_pl, test_ev = dist_generation(
        pl_mean, ev_mean, pl_std, ev_std, n=n, dist_type=dist_type
    )
    target_pl, target_ev = test_df["ISOPleasant"].values, test_df["ISOEventful"].values
    return ssid_calc(test_pl, test_ev, target_pl, target_ev)


# %%


def ssid_plot(
    test_data,
    pl_mean,
    ev_mean,
    pl_std,
    ev_std,
    title=None,
    dist_type="truncnorm",
    density_type="simple",
    incl_scatter=False,
    ax=None,
):
    from soundscapy.plotting import density

    test_data = test_data[["ISOPleasant", "ISOEventful"]].dropna()
    # osten = ssid(test_data, pl_mean, ev_mean, pl_std, ev_std, n=500)
    target_pl, target_ev = dist_generation(
        pl_mean, ev_mean, pl_std, ev_std, n=10000, dist_type=dist_type
    )
    target_df = pd.DataFrame({"ISOPleasant": target_pl, "ISOEventful": target_ev})
    osten = ssid_calc(
        test_data["ISOPleasant"].values,
        test_data["ISOEventful"].values,
        target_df["ISOPleasant"].values,
        target_df["ISOEventful"].values,
    )

    target_df["SSID"] = "Target"
    test_data["SSID"] = "Test"
    df = pd.concat((target_df, test_data))
    title = f"SPI = {osten} Östens" if title is None else title
    density(
        df,
        hue="SSID",
        density_type=density_type,
        incl_scatter=incl_scatter,
        title=title,
    )


# %%

if __name__ == "__main__":
    from soundscapy.databases import isd
    import matplotlib.pyplot as plt

    test_df = isd.load().query("LocationID == 'MonumentoGaribaldi'")
    x1 = test_df.ISOPleasant.dropna().values
    y1 = test_df.ISOEventful.dropna().values

    # mean_pl = 0.7
    # mean_ev = 0.3
    # std_pl = 0.3
    # std_ev = 0.2

    mean_pl = x1.mean()
    mean_ev = y1.mean()
    std_pl = x1.std()
    std_ev = y1.std()

    ssid(
        test_df[["ISOPleasant", "ISOEventful"]].dropna(),
        mean_pl,
        mean_ev,
        std_pl,
        std_ev,
        n=500,
        dist_type="truncnorm",
    )

    ssid_plot(
        test_df,
        mean_pl,
        mean_ev,
        std_pl,
        std_ev,
        dist_type="normal",
        density_type="full",
        incl_scatter=False,
    )
    plt.show()
