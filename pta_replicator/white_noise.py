import numpy as np
from sim import SimulatedPulsar
from astropy.time import TimeDelta

# def quantize(times, dt=1):
#     bins = np.arange(np.min(times), np.max(times) + dt, dt)
#     indices = np.digitize(times, bins)  # indices are labeled by "right edge"
#     counts = np.bincount(indices, minlength=len(bins) + 1)

#     bign, smalln = len(times), np.sum(counts > 0)

#     t = np.zeros(smalln, "d")
#     U = np.zeros((bign, smalln), "d")

#     j = 0
#     for i, c in enumerate(counts):
#         if c > 0:
#             U[indices == i, j] = 1
#             t[j] = np.mean(times[indices == i])
#             j = j + 1

#     return t, U

def quantize_fast(times, flags=None, dt=1.0):
    """
    Quantize TOAs into bins of width dt.
    Optionally take a list of TOA flags to use.

    Parameters
    ----------
    times : array
        The TOA times.
    flags : list
        A list of TOA flags to use
    dt : float
        The width of the bins in days.
    """
    isort = np.argsort(times)

    bucket_ref = [times[isort[0]]]
    bucket_ind = [[isort[0]]]

    for i in isort[1:]:
        if times[i] - bucket_ref[-1] < dt:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(times[i])
            bucket_ind.append([i])

    avetoas = np.array([np.mean(times[ind]) for ind in bucket_ind], "d")
    if flags is not None:
        aveflags = np.array([flags[ind[0]] for ind in bucket_ind])

    U = np.zeros((len(times), len(bucket_ind)), "d")
    for i, l in enumerate(bucket_ind):
        U[l, i] = 1

    if flags is not None:
        return avetoas, aveflags, U
    else:
        return avetoas, U

# def add_efac(psr: SimulatedPulsar, efac: float = 1.0,
#              flags: list = None, seed: int = None):
#     """Add nominal TOA errors, multiplied by `efac` factor.
#     Optionally take a pseudorandom-number-generator seed.
    
#     Parameters
#     ----------
#     psr : SimulatedPulsar
#         The pulsar to add noise to.
#     efac : float
#         The multiplicative factor for the TOA errors.
#     flags : list
#         A list of TOA flags to use
#     seed : int
#         The seed for the random number generator.
#     """

#     if seed is not None:
#         np.random.seed(seed)

#     # default efacvec
#     efacvec = np.zeros(psr.toas.ntoas)

#     # check that efac is scalar if flags is None
#     if flags is None:
#         if not np.isscalar(efac):
#             raise ValueError('ERROR: If flags is None, efac must be a scalar')
#         else:
#             efacvec = np.ones(psr.toas.ntoas) * efac

#     if flags is not None and not np.isscalar(efac):
#         if len(efac) == len(flags):
#             for ct, flag in enumerate(flags):
#                 ind = flag == np.array([f['f'] for f
#                                         in psr.toas.table['flags'].data])
#                 efacvec[ind] = efac[ct]

#     dt = efacvec * psr.toas.get_errors().to('s') * np.random.randn(psr.toas.ntoas)
#     psr.toas.adjust_TOAs(TimeDelta(dt.to('day')))
#     psr.update_residuals()

def add_measurement_noise(psr: SimulatedPulsar, efac: float = 1.0, equad: float = 0.0,
                          flags: list = None, seed: int = None, tnequad: bool = False):
    """
    Add nominal TOA errors added by EQUAD, and then multiplied by an EFAC factor.
    Optionally take a pseudorandom-number-generator seed.

    Parameters
    ----------
    psr : SimulatedPulsar
        The pulsar to add noise to.
    efac : float
        The multiplicative factor for the TOA errors.
    equad : float
        The additive factor for the TOA errors.
    flags : list
        A list of TOA flags to use
    seed : int
        The seed for the random number generator.
    tnquad : bool
        Whether to add measurment noise as
        EFAC * (TOA error + EQUAD) [default] or EFAC * TOA error + EQUAD.
    """

    if seed is not None:
        np.random.seed(seed)

    # default efacvec
    efacvec = np.zeros(psr.toas.ntoas)
    # default equadvec
    equadvec = np.zeros(psr.toas.ntoas)

    # check that efac is scalar if flags is None
    if flags is None:
        if not np.isscalar(efac) or not np.isscalar(equad):
            raise ValueError('ERROR: If flags is None, efac and equad must be a scalar')
        else:
            efacvec = np.ones(psr.toas.ntoas) * efac
            equadvec = np.ones(psr.toas.ntoas) * equad

    if (flags is not None and not np.isscalar(efac)) or (flags is not None and not np.isscalar(equad)):
        if len(efac) == len(flags) and len(equad) == len(flags):
            for ct, flag in enumerate(flags):
                ind = flag == np.array([f['f'] for f
                                        in psr.toas.table['flags'].data])
                efacvec[ind] = efac[ct]
                equadvec[ind] = equad[ct]
        else:
            raise ValueError('ERROR: flags must be same length as efac and equad')

    dt = efacvec * psr.toas.get_errors().to('s') * np.random.randn(psr.toas.ntoas)
    if tnequad:
        dt += equadvec * np.random.randn(psr.toas.ntoas)
    else:
        dt += efacvec * equadvec * np.random.randn(psr.toas.ntoas)

    psr.toas.adjust_TOAs(TimeDelta(dt.to('day')))
    psr.update_residuals()

def add_jitter(psr, ecorr, flags=None, coarsegrain=0.1, seed=None):
    """
    Add correlated quadrature noise of rms ecorr [s],
    with coarse-graining time coarsegrain [days].
    Optionally take a pseudorandom-number-generator seed.
    
    Parameters
    ----------
    psr : SimulatedPulsar
        The pulsar to add noise to.
    ecorr : float
        The rms of the jitter noise.
    flags : list
        A list of TOA flags to use
    coarsegrain : float
        The coarse-graining time in days.
    seed : int
        The seed for the random number generator.
    """

    if seed is not None:
        np.random.seed(seed)

    if flags is None:
        t, U = quantize_fast(np.array(psr.toas(), "d"), dt=coarsegrain)
    elif flags is not None:
        t, f, U = quantize_fast(np.array(psr.toas(), "d"), np.array(psr.flagvals(flags)), dt=coarsegrain)

    # default jitter value
    ecorrvec = np.zeros(len(t))

    # check that jitter is scalar if flags is None
    if flags is None:
        if not np.isscalar(ecorr):
            raise ValueError("ERROR: If flags is None, jitter must be a scalar")
        else:
            ecorrvec = np.ones(len(t)) * ecorr

    if flags is not None and not np.isscalar(ecorr):
        if len(ecorr) == len(flags):
            for ct, flag in enumerate(flags):
                ind = flag == np.array(f)
                ecorrvec[ind] = ecorr[ct]
        else:
            raise ValueError("ERROR: flags must be same length as jitter")

    psr.stoas[:] += (1 / day) * np.dot(U * ecorrvec, np.random.randn(U.shape[1]))
