import numpy as np
from pta_replicator.simulate import SimulatedPulsar
from astropy.time import TimeDelta
from astropy import units as u


def quantize_fast(times, flags=None, dt=1.0, min_toas=1):
    """
    Quantize TOAs into bins of width dt.
    Optionally take a list of TOA flags to use.

    The TOAs are grouped into epochs of width dt, optionally separated by backend flags.
    Epochs with fewer than `min_toas` TOAs are discarded, enterprise sets this to
    2 by default.

    Parameters
    ----------
    times : array
        The TOA times.
    flags : list
        A list of TOA flags to use
    dt : float
        The width of the bins in days.
    min_toas : int
        The minimum number of TOAs required for an epoch to be kept.
    """
    # If flags are provided, we need to create buckets for each backend
    if flags is not None:
        f = np.array(flags)
        # Create an index list for each unique backend
        b_idx = [np.where(f == b)[0] for b in np.unique(f)]
    else:
        # No backends -> treat as one backend
        b_idx = [np.arange(len(times))]

    # We need to compute:
    avg_toas = [] # list to store average TOAs for epoch
    avg_flags = [] # list to store flags for each epoch
    U = [] # list representing the U matrix

    for idx in b_idx:
        t = times[idx] # Extract the times for the current backend

        isort = np.argsort(t) # Sort by the time of arrival (in backend indices)

        bucket_ref = [t[isort[0]]] # list of reference times for each bucket
        bucket_ind = [[idx[isort[0]]]] # list of list of indices for each bucket

        # Loop through each time and either group it, or make a new bucket
        for i in isort[1:]:
            if t[i] - bucket_ref[-1] < dt:
                bucket_ind[-1].append(idx[i]) # Add the GLOBAL index to the current bucket
            else:
                bucket_ref.append(t[i])
                bucket_ind.append([idx[i]]) # Start a new bucket with the current global index

        # Add the average TOA and flag for each bucket to the lists
        avg_toas.extend([np.mean(times[ind]) for ind in bucket_ind])
        if flags is not None:
            # Add the first flag as the representative flag for the bucket
            avg_flags.extend([flags[ind[0]] for ind in bucket_ind])

        # Add the bucket indices to the U matrix list
        U.extend(bucket_ind)

    # After looping through all backends, we have lists of average TOAs, flags, and U indices
    # Need to convert the lists into arrays
    avg_toas = np.array(avg_toas, "d")
    if flags is not None:
        avg_flags = np.array(avg_flags)
    # Need to convert U indices into a matrix form
    U_mat = np.zeros((len(times), len(U)), "d") # (ntoa, n_buckets)
    for j, l in enumerate(U):
        U_mat[l, j] = 1

    # Remove 1-sized buckets as they don't get jitter
    keep = np.sum(U_mat, axis=0) >= min_toas
    avg_toas = avg_toas[keep]
    if flags is not None:
        avg_flags = avg_flags[keep]
    U_mat = U_mat[:, keep]

    if flags is not None:
        return avg_toas, avg_flags, U_mat
    else:
        return avg_toas, U_mat


def add_measurement_noise(psr: SimulatedPulsar, efac: float = 1.0,
                          log10_equad: float = None,
                          flagid: str = 'f', flags: list = None,
                          seed: int = None, tnequad: bool = False):
    """
    Add nominal TOA errors added by EQUAD, and then multiplied by an EFAC factor.
    Optionally take a pseudorandom-number-generator seed.

    Parameters
    ----------
    psr : SimulatedPulsar
        The pulsar to add noise to.
    efac : float or list
        The multiplicative factor for the TOA errors.
    log10_equad : float or list
        The additive factor for the TOA errors measured in log10(seconds).
    flagid : str
        The key for the flags in the TOA table.
    flags : list
        A list of TOA flags to use
    seed : int
        The seed for the random number generator.
    tnquad : bool
        Whether to add measurment noise as
        EFAC * (TOA error + EQUAD) [default] or EFAC * TOA error + EQUAD.
    """
    equad_str = 'tnequad' if tnequad else 't2equad'

    if log10_equad is not None:
        equad = 10**log10_equad
    else:
        equad = 0.0
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
                ind = flag == np.array([f[flagid] for f
                                        in psr.toas.table['flags'].data])
                efacvec[ind] = efac[ct]
                equadvec[ind] = equad[ct]
        else:
            raise ValueError('ERROR: flags must be same length as efac and log10_equad')

    dt = efacvec * psr.toas.get_errors().to('s') * np.random.randn(psr.toas.ntoas)
    if tnequad:
        dt += equadvec * np.random.randn(psr.toas.ntoas) * u.s
    else:
        dt += efacvec * equadvec * np.random.randn(psr.toas.ntoas) * u.s
        
    # update the added_signals dictionary
    if flags is None:
        psr.update_added_signals('{}_measurement_noise'.format(psr.name),
                                 {'efac': efac, 'log10_' + equad_str: log10_equad},
                                 dt)
    else:
        # update added_signals_time dictionary first
        psr.update_added_signals('{}_measurement_noise'.format(psr.name), {}, dt)
        # next update added_signals dictionary
        for i, flag in enumerate(flags):
            psr.update_added_signals('{}_{}_measurement_noise'.format(psr.name, flag),
                                     {'efac': efac[i], 'log10_' + equad_str: log10_equad[i]})

    psr.toas.adjust_TOAs(TimeDelta(dt.to('day')))
    psr.update_residuals()


def add_jitter(psr: SimulatedPulsar, log10_ecorr: float,
               flagid: str = 'f', flags: list = None,
               coarsegrain: float = 0.1, min_toas: int = 1, seed: int = None):
    """
    Add correlated quadrature noise of rms ecorr [s],
    with coarse-graining time coarsegrain [days].
    Optionally take a pseudorandom-number-generator seed.

    `min_toas` is the minimum number of TOAs required for an epoch to be kept. 
    It defaults to 1, but enterprise defaults it to 2.
    
    Parameters
    ----------
    psr : SimulatedPulsar
        The pulsar to add noise to.
    ecorr : float
        The rms of the jitter noise.
    flagid : str
        The key for the flags in the TOA table.
    flags : list
        A list of TOA flags to use
    coarsegrain : float
        The coarse-graining time in days.
    min_toas : int
        The minimum number of TOAs required for an epoch to be kept.
    seed : int
        The seed for the random number generator.
    """

    ecorr = 10**log10_ecorr

    if seed is not None:
        np.random.seed(seed)

    if flags is None:
        t, U = quantize_fast(psr.toas.get_mjds().value, dt=coarsegrain, min_toas=min_toas)
    elif flags is not None:
        t, f, U = quantize_fast(psr.toas.get_mjds().value,
                                np.array([f[flagid] for f in psr.toas.table['flags'].data]),
                                dt=coarsegrain, min_toas=min_toas)

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

    dt = u.s * np.dot(U * ecorrvec, np.random.randn(U.shape[1]))
    
    # update the added_signals dictionary
    if flags is None:
        psr.update_added_signals('{}_jitter'.format(psr.name),
                                 {'log10_ecorr': log10_ecorr},
                                 dt)
    else:
        # update added_signals_time dictionary first
        psr.update_added_signals('{}_jitter'.format(psr.name), {}, dt)
        # next update added_signal dictionary
        for i, flag in enumerate(flags):
            psr.update_added_signals('{}_{}_jitter'.format(psr.name, flag),
                                     {'log10_ecorr': log10_ecorr[i]})

    psr.toas.adjust_TOAs(TimeDelta(dt.to('day')))
    psr.update_residuals()
