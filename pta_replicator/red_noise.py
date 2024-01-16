import numpy as np
from sim import SimulatedPulsar
from constants import DAY_IN_SEC, YEAR_IN_SEC
from astropy import units as u
from astropy.time import TimeDelta

def createfourierdesignmatrix_red(toas: np.ndarray, nmodes: int = 30,
                                  Tspan: float = None, logf: bool = False,
                                  fmin: float = None, fmax: float = None,
                                  pshift: bool = False, modes: np.ndarray = None) -> tuple:
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013

    Parameters
    ----------
    toas [array]: Vector of time series in seconds.
    nmodes [int]: Number of fourier coefficients to use.
    Tspan [float]: Option to us some other Tspan [s]
    logf [bool]: Use log frequency spacing.
    fmin [float]: Lower sampling frequency.
    fmax [float]: Upper sampling frequency.
    pshift [bool]: Option to add random phase shift.
    modes [array]: Option to provide explicit list or array of sampling frequencies.

    Returns
    -------
    F [array]: fourier design matrix, [NTOAs x 2 nfreqs].
    f [array]: Sampling frequencies, [2 nfreqs].
    """

    T = Tspan if Tspan is not None else toas.max() - toas.min()

    # define sampling frequencies
    if modes is not None:
        nmodes = len(modes)
        f = modes
    elif fmin is None and fmax is None and not logf:
        # make sure partially overlapping sets of modes
        # have identical frequencies
        f = 1.0 * np.arange(1, nmodes + 1) / T
    else:
        # more general case
        if fmin is None:
            fmin = 1 / T
        if fmax is None:
            fmax = nmodes / T
        if logf:
            f = np.logspace(np.log10(fmin), np.log10(fmax), nmodes)
        else:
            f = np.linspace(fmin, fmax, nmodes)

    # add random phase shift to basis functions
    ranphase = (np.random.uniform(0.0, 2 * np.pi, nmodes)
                if pshift else np.zeros(nmodes))

    Ffreqs = np.repeat(f, 2)

    N = len(toas)
    F = np.zeros((N, 2 * nmodes))

    # The sine/cosine modes
    F[:,::2] = np.sin(2*np.pi*toas[:,None]*f[None,:] +
                      ranphase[None,:])
    F[:,1::2] = np.cos(2*np.pi*toas[:,None]*f[None,:] +
                       ranphase[None,:])

    return F, Ffreqs

def add_rednoise(psr: SimulatedPulsar, A: float, gamma: float,
                 components: int = 30, seed: int = None,
                 modes: np.ndarray = None, Tspan: float = None):
    """Add red noise with P(f) = A^2 / (12 pi^2) (f * year)^-gamma,
    using `components` Fourier bases.
    Optionally take a pseudorandom-number-generator seed."""

    # nobs = len(psr.toas.table)

    fyr = 1 / YEAR_IN_SEC

    if seed is not None:
        np.random.seed(seed)
    if modes is not None:
        print('Must use linear spacing.')

    toas = np.array(psr.toas.table['tdbld'], dtype='float64') * DAY_IN_SEC #to sec
    Tspan = toas.max() - toas.min()
    F, freqs = createfourierdesignmatrix_red(toas, Tspan=Tspan, nmodes=components, modes=modes)
    prior = A**2 * (freqs/fyr)**(-gamma) / (12 * np.pi**2 * Tspan) * YEAR_IN_SEC**3
    y = np.sqrt(prior) * np.random.randn(freqs.size)
    dt = np.dot(F,y) * u.s
    psr.toas.adjust_TOAs(TimeDelta(dt.to('day')))
    psr.update_residuals()
