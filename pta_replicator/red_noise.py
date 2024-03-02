import numpy as np
from pta_replicator.simulate import SimulatedPulsar
from pta_replicator.constants import DAY_IN_SEC, YEAR_IN_SEC
from astropy import units as u
from astropy.time import TimeDelta
import ephem
import pta_replicator.spharmORFbasis as anis
from scipy import interpolate as interp


def extrap1d(interpolator):
    """
    Function to extend an interpolation function to an
    extrapolation function.
    :param interpolator: scipy interp1d object
    :returns ufunclike: extension of function to extrapolation
    """

    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]  # +(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]  # +(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return np.array(list(map(pointwise, np.array(xs))))

    return ufunclike


def create_fourier_design_matrix_red(toas: np.ndarray, nmodes: int = 30,
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


def add_red_noise(psr: SimulatedPulsar, log10_amplitude: float, spectral_index: float,
                  components: int = 30, seed: int = None,
                  modes: np.ndarray = None, Tspan: float = None):
    """Add red noise with P(f) = A^2 / (12 pi^2) (f * year)^-gamma,
    using `components` Fourier bases.
    Optionally take a pseudorandom-number-generator seed."""
    psr.update_added_signals('{}_red_noise'.format(psr.name), 
                             {'amplitude': log10_amplitude, 'spectral_index': spectral_index})
    A = 10**(log10_amplitude)
    gamma = spectral_index
    # nobs = len(psr.toas.table)

    fyr = 1 / YEAR_IN_SEC

    if seed is not None:
        np.random.seed(seed)
    if modes is not None:
        print('Must use linear spacing.')

    toas = np.array(psr.toas.table['tdbld'], dtype='float64') * DAY_IN_SEC #to sec
    Tspan = toas.max() - toas.min()
    F, freqs = create_fourier_design_matrix_red(toas, Tspan=Tspan, nmodes=components, modes=modes)
    prior = A**2 * (freqs/fyr)**(-gamma) / (12 * np.pi**2 * Tspan) * YEAR_IN_SEC**3
    y = np.sqrt(prior) * np.random.randn(freqs.size)
    dt = np.dot(F,y) * u.s
    psr.toas.adjust_TOAs(TimeDelta(dt.to('day')))
    psr.update_residuals()


def add_gwb(
    psrs: list,
    log10_amplitude: float,
    spectral_index: float,
    no_correlations: bool = False,
    seed: int = None,
    turnover: bool = False,
    clm: list = [np.sqrt(4.0 * np.pi)],
    lmax: int = 0,
    f0: float = 1e-9,
    beta: float = 1,
    power: float = 1,
    userSpec: np.ndarray = None,
    npts: int = 600,
    howml: int = 10,
):
    """
    Function to create GW-induced residuals from a stochastic GWB as defined
    in Chamberlin, Creighton, Demorest, et al. (2014).
    :param psrs: list of SimulatedPulsars to inject GWB into
    :param log10_amplitude: Amplitude of red noise in GW units
    :param spectral_index: Red noise power law spectral index
    :param no_correlations: Add red noise with no spatial correlations
    :param seed: Random number seed
    :param turnover: Produce spectrum with turnover at frequency f0
    :param clm: coefficients of spherical harmonic decomposition of GW power
    :param lmax: maximum multipole of GW power decomposition
    :param f0: Frequency of spectrum turnover
    :param beta: Spectral index of power spectram for f << f0
    :param power: Fudge factor for flatness of spectrum turnover
    :param userSpec: User-supplied characteristic strain spectrum
                     (first column is freqs, second is spectrum)
    :param npts: Number of points used in interpolation
    :param howml: Lowest frequency is 1/(howml * T)
    :returns: list of residuals for each pulsar
    """
    for psr in psrs:
        psr.update_added_signals('{}_gwb'.format(psr.name), 
                                 {'amplitude': log10_amplitude, 'spectral_index': spectral_index})
    Amp = 10**log10_amplitude
    gam = spectral_index
    if seed is not None:
        np.random.seed(seed)

    # number of pulsars
    Npulsars = len(psrs)

    # gw start and end times for entire data set
    start = float(np.min([psr.toas.first_MJD.value * 86400 for psr in psrs]) - 86400)
    stop = float(np.max([psr.toas.last_MJD.value * 86400 for psr in psrs]) + 86400)

    # duration of the signal
    dur = stop - start

    # get maximum number of points
    if npts is None:
        # default to cadence of 2 weeks
        npts = dur / (86400 * 14)

    # make a vector of evenly sampled data points
    ut = np.linspace(start, stop, npts)

    # time resolution in days
    dt = dur / npts

    # compute the overlap reduction function
    if no_correlations:
        ORF = np.diag(np.ones(Npulsars) * 2)
    else:
        psrlocs = np.zeros((Npulsars, 2))

        for ii in range(Npulsars):
            if "RAJ" and "DECJ" in psrs[ii].loc:
                psrlocs[ii] = float(psrs[ii].loc["RAJ"]*np.pi/12.0), float(psrs[ii].loc["DECJ"]*np.pi/180.0)
                ## Lookout for RAJ being in hours instead of degrees
                ## TODO: check for this in other places that use RAJ
            elif "ELONG" and "ELAT" in psrs[ii].loc:
                fac = 1.0
                #fac = 180.0 / np.pi
                # check for B name
                if "B" in psrs[ii].name:
                    epoch = "1950"
                else:
                    epoch = "2000"
                coords = ephem.Equatorial(
                    ephem.Ecliptic(str(psrs[ii].loc["ELONG"] * fac), str(psrs[ii].loc["ELAT"] * fac)), epoch=epoch
                )
                psrlocs[ii] = float(repr(coords.ra)), float(repr(coords.dec))

        psrlocs[:, 1] = np.pi / 2.0 - psrlocs[:, 1]
        anisbasis = np.array(anis.correlated_basis(psrlocs, lmax))
        ORF = sum(clm[kk] * anisbasis[kk] for kk in range(len(anisbasis)))
        ORF *= 2.0

    # Define frequencies spanning from DC to Nyquist.
    # This is a vector spanning these frequencies in increments of 1/(dur*howml).
    f = np.arange(0, 1 / (2 * dt), 1 / (dur * howml))
    f[0] = f[1]  # avoid divide by 0 warning
    Nf = len(f)

    # Use Cholesky transform to take 'square root' of ORF
    M = np.linalg.cholesky(ORF)

    # Create random frequency series from zero mean, unit variance, Gaussian distributions
    w = np.zeros((Npulsars, Nf), complex)
    for ll in range(Npulsars):
        w[ll, :] = np.random.randn(Nf) + 1j * np.random.randn(Nf)

    # strain amplitude
    if userSpec is None:

        f1yr = 1 / 3.16e7
        alpha = -0.5 * (gam - 3)
        hcf = Amp * (f / f1yr) ** (alpha)
        if turnover:
            si = alpha - beta
            hcf /= (1 + (f / f0) ** (power * si)) ** (1 / power)

    elif userSpec is not None:

        freqs = userSpec[:, 0]
        if len(userSpec[:, 0]) != len(freqs):
            raise ValueError("Number of supplied spectral points does not match number of frequencies!")
        else:
            fspec_in = interp.interp1d(np.log10(freqs), np.log10(userSpec[:, 1]), kind="linear")
            fspec_ex = extrap1d(fspec_in)
            hcf = 10.0 ** fspec_ex(np.log10(f))

    C = 1 / 96 / np.pi**2 * hcf**2 / f**3 * dur * howml

    # inject residuals in the frequency domain
    Res_f = np.dot(M, w)
    for ll in range(Npulsars):
        Res_f[ll] = Res_f[ll] * C ** (0.5)  # rescale by frequency dependent factor
        Res_f[ll, 0] = 0  # set DC bin to zero to avoid infinities
        Res_f[ll, -1] = 0  # set Nyquist bin to zero also

    # Now fill in bins after Nyquist (for fft data packing) and take inverse FT
    Res_f2 = np.zeros((Npulsars, 2 * Nf - 2), complex)
    Res_t = np.zeros((Npulsars, 2 * Nf - 2))
    Res_f2[:, 0:Nf] = Res_f[:, 0:Nf]
    Res_f2[:, Nf : (2 * Nf - 2)] = np.conj(Res_f[:, (Nf - 2) : 0 : -1])
    Res_t = np.real(np.fft.ifft(Res_f2) / dt)

    # shorten data and interpolate onto TOAs
    Res = np.zeros((Npulsars, npts))
    res_gw = []
    for ll in range(Npulsars):
        Res[ll, :] = Res_t[ll, 10 : (npts + 10)]
        f = interp.interp1d(ut, Res[ll, :], kind="linear")
        res_gw.append(f(psrs[ll].toas.get_mjds().value.astype(float) * 86400))

    # return res_gw
    ct = 0
    for psr in psrs:
        dt = res_gw[ct] / 86400.0 * u.day
        psr.toas.adjust_TOAs(TimeDelta(dt.to('day')))
        psr.update_residuals()
        ct += 1
