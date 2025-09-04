import numpy as np
from pta_replicator.constants import SOLAR2S, MPC2S, KPC2S
import ephem
import astropy as ap
from astropy.time import TimeDelta
from astropy import units as u

from holodeck import utils, cosmo
from pta_replicator.red_noise import add_gwb

from numba import njit, prange

def add_cgw(
    psr,
    gwtheta,
    gwphi,
    mc,
    dist,
    fgw,
    phase0,
    psi,
    inc,
    pdist=1.0,
    pphase=None,
    psrTerm=True,
    evolve=True,
    phase_approx=False,
    tref=0,
    signal_name='cw',
):
    """
    Function to add CW residuals (adapted from libstempo.toasim)

    :param psr: pulsar object
    :param gwtheta: Polar angle of GW source in celestial coords [radians]
    :param gwphi: Azimuthal angle of GW source in celestial coords [radians]
    :param mc: Chirp mass of SMBHB [solar masses]
    :param dist: Luminosity distance to SMBHB [Mpc]
    :param fgw: Frequency of GW (twice the orbital frequency) [Hz]
    :param phase0: Initial Phase of GW source [radians]
    :param psi: Polarization of GW source [radians]
    :param inc: Inclination of GW source [radians]
    :param pdist: Pulsar distance to use other than those in psr [kpc]
    :param pphase: Use pulsar phase to determine distance [radian]
    :param psrTerm: Option to include pulsar term [boolean]
    :param evolve: Option to exclude evolution [boolean]
    :param tref: Fidicuial time at which initial parameters are referenced
    """

    # convert units
    mc *= SOLAR2S  # convert from solar masses to seconds
    dist *= MPC2S  # convert from Mpc to seconds

    # define initial orbital frequency
    w0 = np.pi * fgw
    phase0 /= 2  # orbital phase
    w053 = w0 ** (-5 / 3)

    # define variable for later use
    cosgwtheta, cosgwphi = np.cos(gwtheta), np.cos(gwphi)
    singwtheta, singwphi = np.sin(gwtheta), np.sin(gwphi)
    sin2psi, cos2psi = np.sin(2 * psi), np.cos(2 * psi)
    incfac1, incfac2 = 0.5 * (3 + np.cos(2 * inc)), 2 * np.cos(inc)

    # unit vectors to GW source
    m = np.array([singwphi, -cosgwphi, 0.0])
    n = np.array([-cosgwtheta * cosgwphi, -cosgwtheta * singwphi, singwtheta])
    omhat = np.array([-singwtheta * cosgwphi, -singwtheta * singwphi, -cosgwtheta])

    # various factors invloving GW parameters
    fac1 = 256 / 5 * mc ** (5 / 3) * w0 ** (8 / 3)
    fac2 = 1 / 32 / mc ** (5 / 3)
    fac3 = mc ** (5 / 3) / dist

    # pulsar location
    if "RAJ" and "DECJ" in psr.loc.keys():
        ptheta = np.pi / 2 - psr.loc["DECJ"]*np.pi/180.0
        pphi = psr.loc["RAJ"]*np.pi/12.0
    elif "ELONG" and "ELAT" in psr.loc.keys():
        fac = 1.0#180.0 / np.pi #no need in pint
        if "B" in psr.name:
            epoch = "1950"
        else:
            epoch = "2000"
        coords = ephem.Equatorial(ephem.Ecliptic(str(psr.loc["ELONG"] * fac), str(psr.loc["ELAT"] * fac)), epoch=epoch)

        ptheta = np.pi / 2 - float(repr(coords.dec))
        pphi = float(repr(coords.ra))

    # use definition from Sesana et al 2010 and Ellis et al 2012
    phat = np.array([np.sin(ptheta) * np.cos(pphi), np.sin(ptheta) * np.sin(pphi), np.cos(ptheta)])

    fplus = 0.5 * (np.dot(m, phat) ** 2 - np.dot(n, phat) ** 2) / (1 + np.dot(omhat, phat))
    fcross = (np.dot(m, phat) * np.dot(n, phat)) / (1 + np.dot(omhat, phat))
    cosMu = -np.dot(omhat, phat)

    # get values from pulsar object
    toas = psr.toas.get_mjds().value * 86400 - tref
    if pphase is not None:
        pd = pphase / (2 * np.pi * fgw * (1 - cosMu)) / KPC2S
    else:
        pd = pdist

    # convert units
    pd *= KPC2S  # convert from kpc to seconds

    # get pulsar time
    tp = toas - pd * (1 - cosMu)

    # evolution
    if evolve:

        # calculate time dependent frequency at earth and pulsar
        omega = w0 * (1 - fac1 * toas) ** (-3 / 8)
        omega_p = w0 * (1 - fac1 * tp) ** (-3 / 8)

        # calculate time dependent phase
        phase = phase0 + fac2 * (w053 - omega ** (-5 / 3))
        phase_p = phase0 + fac2 * (w053 - omega_p ** (-5 / 3))

    # use approximation that frequency does not evlolve over observation time
    elif phase_approx:

        # frequencies
        omega = w0
        omega_p = w0 * (1 + fac1 * pd * (1 - cosMu)) ** (-3 / 8)

        # phases
        phase = phase0 + omega * toas
        phase_p = phase0 + fac2 * (w053 - omega_p ** (-5 / 3)) + omega_p * toas

    # no evolution
    else:

        # monochromatic
        omega = w0
        omega_p = omega

        # phases
        phase = phase0 + omega * toas
        phase_p = phase0 + omega * tp

    # define time dependent coefficients
    At = np.sin(2 * phase) * incfac1
    Bt = np.cos(2 * phase) * incfac2
    At_p = np.sin(2 * phase_p) * incfac1
    Bt_p = np.cos(2 * phase_p) * incfac2

    # now define time dependent amplitudes
    alpha = fac3 / omega ** (1 / 3)
    alpha_p = fac3 / omega_p ** (1 / 3)

    # define rplus and rcross
    rplus = alpha * (At * cos2psi + Bt * sin2psi)
    rcross = alpha * (-At * sin2psi + Bt * cos2psi)
    rplus_p = alpha_p * (At_p * cos2psi + Bt_p * sin2psi)
    rcross_p = alpha_p * (-At_p * sin2psi + Bt_p * cos2psi)

    # residuals
    if psrTerm:
        res = fplus * (rplus_p - rplus) + fcross * (rcross_p - rcross)
    else:
        res = -fplus * rplus - fcross * rcross

    dt = res * u.s
    
    psr.update_added_signals('{}_'.format(psr.name)+signal_name,
                             {'gwtheta': gwtheta,
                              'gwphi':gwphi,
                              'mc':mc,
                              'dist':dist,
                              'fgw':fgw,
                              'phase0':phase0,
                              'psi':psi,
                              'inc':inc,
                              'pdist':pdist,
                              'pphase':pphase,
                              'psrTerm':psrTerm,
                              'evolve':evolve,
                              'phase_approx':phase_approx,
                              'tref':tref},
                             dt)
    
    psr.toas.adjust_TOAs(TimeDelta(dt.to('day')))
    psr.update_residuals()


def add_catalog_of_cws(psr, gwtheta_list, gwphi_list, mc_list, dist_list, fgw_list, phase0_list, psi_list, inc_list, pdist=1.0,
                       pphase=None, psrTerm=True, evolve=True, phase_approx=False, tref=0, chunk_size=10_000_000, signal_name='cw_catalog'):
    """
    Method to add a list of many SMBHBs more efficiently than by calling add_cgw multiple times. It takes the same input as add_cgw, except as lists.

    Parameters
    ----------
    psr : Pulsar object
    gwtheta_list : 1Darray
        List of polar angles of GW sources in celestial coords [radians]
    gwphi_list : 1Darray
        List of azimuthal angles of GW sources in celestial coords [radians]
    mc_list : 1Darray
        List of observed chirp masses of SMBHBs [solar masses] 
    dist_list : 1Darray
        List of luminosity distances to SMBHBs [Mpc]
    fgw_list : 1Darray
        List of observed frequencies of GW (twice the orbital frequency) [Hz]
    phase0_list : 1Darray
        List of initial phases of GW sources [radians]
    psi_list : 1Darray
        List of polarizations of GW sources [radians]
    inc_list : 1Darray
        List of inclinations of GW sources [radians]
    pdist : float
        Pulsar distance to use other than those in psr [kpc]
    pphase : float?
        Use pulsar phase to determine distance [radian]
    psrTerm : bool
        Option to include pulsar term [boolean]
    evolve : bool
        Option to exclude evolution [boolean]
    tref:  
        Fidicuial time at which initial parameters are referenced
    chunk_size : int
        default 10_000_000
    signal_name : str
        default 'cw_catalog'

    Returns
    -------
    None
    
    """

    # pulsar location
    if "RAJ" and "DECJ" in psr.loc.keys():
        ptheta = np.pi / 2 - psr.loc["DECJ"]*np.pi/180.0
        pphi = psr.loc["RAJ"]*np.pi/12.0
    elif "ELONG" and "ELAT" in psr.loc.keys():
        fac = 1.0#180.0 / np.pi #no need in pint
        if "B" in psr.name:
            epoch = "1950"
        else:
            epoch = "2000"
        coords = ephem.Equatorial(ephem.Ecliptic(str(psr.loc["ELONG"] * fac), str(psr.loc["ELAT"] * fac)), epoch=epoch)

        ptheta = np.pi / 2 - float(repr(coords.dec))
        pphi = float(repr(coords.ra))

    # use definition from Sesana et al 2010 and Ellis et al 2012
    phat = np.array([np.sin(ptheta)*np.cos(pphi), np.sin(ptheta)*np.sin(pphi),\
            np.cos(ptheta)])

    # get TOAs in seconds
    toas = psr.toas.get_mjds().value*86400 - tref

    #numba doesn't work with float128, so we need to convert phat and toas to float64
    #this should not be a problem, since we do not need such high precision in pulsar sky location or toas
    #for toas, note that this is only used to calculate the GW waveform and not to form the actual residuals
    if mc_list.size>1_000:
        #print("parallel")
        N_chunk = int(np.ceil(mc_list.size/chunk_size))
        print(N_chunk)
        for jjj in range(N_chunk):
            print(str(jjj) + " / " + str(N_chunk))
            idxs = range(jjj*chunk_size, min((jjj+1)*chunk_size,mc_list.size) )
            print(idxs)
            res = loop_over_CWs_parallel(phat.astype('float64'), toas.astype('float64'),
                                         gwtheta_list[idxs], gwphi_list[idxs], mc_list[idxs], dist_list[idxs],
                                         fgw_list[idxs], phase0_list[idxs], psi_list[idxs], inc_list[idxs],
                                         pdist=pdist, pphase=pphase, psrTerm=psrTerm, evolve=evolve, phase_approx=phase_approx)
            #add current batch to TOAs
            dt = res * u.s
            
            psr.update_added_signals('{}_'.format(psr.name)+signal_name,
                                     {'gwtheta_list': gwtheta_list,
                                      'gwphi_list':gwphi_list,
                                      'mc_list':mc_list,
                                      'dist_list':dist_list,
                                      'fgw_list':fgw_list,
                                      'phase0_list':phase0_list,
                                      'psi_list':psi_list,
                                      'inc_list':inc_list,
                                      'pdist':pdist,
                                      'pphase':pphase,
                                      'psrTerm':psrTerm,
                                      'evolve':evolve,
                                      'phase_approx':phase_approx,
                                      'tref':tref},
                                     dt)
            
            psr.toas.adjust_TOAs(TimeDelta(dt.to('day')))
            psr.update_residuals()
    else:
        res = loop_over_CWs(phat.astype('float64'), toas.astype('float64'), gwtheta_list, gwphi_list, mc_list, dist_list, fgw_list, phase0_list, psi_list, inc_list, pdist=pdist,
                            pphase=pphase, psrTerm=psrTerm, evolve=evolve, phase_approx=phase_approx)

        #End of loop over CW sources
        #Now add residual to TOAs
        dt = res * u.s
        
        psr.update_added_signals('{}_'.format(psr.name)+signal_name,
                                 {'gwtheta_list': gwtheta_list,
                                  'gwphi_list':gwphi_list,
                                  'mc_list':mc_list,
                                  'dist_list':dist_list,
                                  'fgw_list':fgw_list,
                                  'phase0_list':phase0_list,
                                  'psi_list':psi_list,
                                  'inc_list':inc_list,
                                  'pdist':pdist,
                                  'pphase':pphase,
                                  'psrTerm':psrTerm,
                                  'evolve':evolve,
                                  'phase_approx':phase_approx,
                                  'tref':tref},
                                 dt)
        
        psr.toas.adjust_TOAs(TimeDelta(dt.to('day')))
        psr.update_residuals()


@njit(fastmath=False, parallel=True)
def loop_over_CWs_parallel(phat, toas, gwtheta_list, gwphi_list, mc_list, dist_list, fgw_list, phase0_list, psi_list, inc_list, pdist=1.0,
                           pphase=None, psrTerm=True, evolve=True, phase_approx=False):
    """Efficient calculation of response from many CGWs in Numba - used by add_catalog_of_cws"""
    # set up array for residuals
    res = np.zeros((len(mc_list),toas.size))

    for iii in prange(len(mc_list)):
    #for gwtheta, gwphi, mc, dist, fgw, phase0, psi, inc in zip(gwtheta_list, gwphi_list, mc_list, dist_list, fgw_list, phase0_list, psi_list, inc_list):
        #if iii%1_000==0: print(iii)
        gwtheta = gwtheta_list[iii]
        gwphi = gwphi_list[iii]
        mc = mc_list[iii]
        dist = dist_list[iii]
        fgw = fgw_list[iii]
        phase0 = phase0_list[iii]
        psi = psi_list[iii]
        inc = inc_list[iii]
        # convert units
        mc *= SOLAR2S         # convert from solar masses to seconds
        dist *= MPC2S    # convert from Mpc to seconds

        # define initial orbital frequency
        w0 = np.pi * fgw
        phase0 /= 2 # orbital phase
        w053 = w0**(-5/3)

        # define variable for later use
        cosgwtheta, cosgwphi = np.cos(gwtheta), np.cos(gwphi)
        singwtheta, singwphi = np.sin(gwtheta), np.sin(gwphi)
        sin2psi, cos2psi = np.sin(2*psi), np.cos(2*psi)
        incfac1, incfac2 = 0.5*(3+np.cos(2*inc)), 2*np.cos(inc)

        # unit vectors to GW source
        m = np.array([singwphi, -cosgwphi, 0.0])
        n = np.array([-cosgwtheta*cosgwphi, -cosgwtheta*singwphi, singwtheta])
        omhat = np.array([-singwtheta*cosgwphi, -singwtheta*singwphi, -cosgwtheta])

        # various factors invloving GW parameters
        fac1 = 256/5 * mc**(5/3) * w0**(8/3)
        fac2 = 1/32/mc**(5/3)
        fac3 = mc**(5/3)/dist

        # get antenna patterns
        fplus = 0.5 * (np.dot(m, phat)**2 - np.dot(n, phat)**2) / (1+np.dot(omhat, phat))
        fcross = (np.dot(m, phat)*np.dot(n, phat)) / (1 + np.dot(omhat, phat))
        cosMu = -np.dot(omhat, phat)

        # get values from pulsar object
        if pphase is not None:
            pd = pphase/(2*np.pi*fgw*(1-cosMu)) / KPC2S
        else:
            pd = pdist

        # convert units
        pd *= KPC2S   # convert from kpc to seconds

        # get pulsar time
        tp = toas-pd*(1-cosMu)

        # evolution
        if evolve:

            # calculate time dependent frequency at earth and pulsar
            omega = w0 * (1 - fac1 * toas)**(-3/8)
            omega_p = w0 * (1 - fac1 * tp)**(-3/8)

            # calculate time dependent phase
            phase = phase0 + fac2 * (w053 - omega**(-5/3))
            phase_p = phase0 + fac2 * (w053 - omega_p**(-5/3))

        # use approximation that frequency does not evlolve over observation time
        elif phase_approx:

            # frequencies
            omega = w0 * np.ones(toas.size) #make sure omega is always an array and never a float (numba typing stuff)
            omega_p = w0 * (1 + fac1 * pd*(1-cosMu))**(-3/8) * np.ones(toas.size) #make sure omega_p is always an array and never a float (numba typing stuff)

            # phases
            phase = phase0 + omega * toas
            phase_p = phase0 + fac2 * (w053 - omega_p**(-5/3)) + omega_p*toas

        # no evolution
        else:

            # monochromatic
            omega = w0 * np.ones(toas.size) #make sure omega is always an array and never a float (numba typing stuff)
            omega_p = omega

            # phases
            phase = phase0 + omega * toas
            phase_p = phase0 + omega * tp


        # define time dependent coefficients
        At = np.sin(2*phase) * incfac1
        Bt = np.cos(2*phase) * incfac2
        At_p = np.sin(2*phase_p) * incfac1
        Bt_p = np.cos(2*phase_p) * incfac2

        # now define time dependent amplitudes
        alpha = fac3 / omega**(1/3)
        alpha_p = fac3 / omega_p**(1/3)

        # define rplus and rcross
        rplus = alpha * (At*cos2psi + Bt*sin2psi)
        rcross = alpha * (-At*sin2psi + Bt*cos2psi)
        rplus_p = alpha_p * (At_p*cos2psi + Bt_p*sin2psi)
        rcross_p = alpha_p * (-At_p*sin2psi + Bt_p*cos2psi)

        # residuals
        if psrTerm:
            #make sure we add zeros rather than NaNs in the rare occasion when the binary already merged and produces negative frequencies
            rrr = fplus*(rplus_p-rplus)+fcross*(rcross_p-rcross)
            res[iii,:] = np.where(np.isnan(rrr), 0.0, rrr)
        else:
            rrr = -fplus*rplus - fcross*rcross
            res[iii,:] = np.where(np.isnan(rrr), 0.0, rrr)

    return np.sum(res, axis=0)


@njit(fastmath=False, parallel=False)
def loop_over_CWs(phat, toas, gwtheta_list, gwphi_list, mc_list, dist_list, fgw_list, phase0_list, psi_list, inc_list, pdist=1.0,
                  pphase=None, psrTerm=True, evolve=True, phase_approx=False):
    """Efficient calculation of response from many CGWs in Numba (nonparallel version which can be more efficient for small number of CWs) - used by add_catalog_of_cws"""
    # set up array for residuals
    res = np.zeros(toas.size)

    for iii in range(len(mc_list)):
    #for gwtheta, gwphi, mc, dist, fgw, phase0, psi, inc in zip(gwtheta_list, gwphi_list, mc_list, dist_list, fgw_list, phase0_list, psi_list, inc_list):
        gwtheta = gwtheta_list[iii]
        gwphi = gwphi_list[iii]
        mc = mc_list[iii]
        dist = dist_list[iii]
        fgw = fgw_list[iii]
        phase0 = phase0_list[iii]
        psi = psi_list[iii]
        inc = inc_list[iii]
        # convert units
        mc *= SOLAR2S         # convert from solar masses to seconds
        dist *= MPC2S    # convert from Mpc to seconds

        # define initial orbital frequency
        w0 = np.pi * fgw
        phase0 /= 2 # orbital phase
        w053 = w0**(-5/3)

        # define variable for later use
        cosgwtheta, cosgwphi = np.cos(gwtheta), np.cos(gwphi)
        singwtheta, singwphi = np.sin(gwtheta), np.sin(gwphi)
        sin2psi, cos2psi = np.sin(2*psi), np.cos(2*psi)
        incfac1, incfac2 = 0.5*(3+np.cos(2*inc)), 2*np.cos(inc)

        # unit vectors to GW source
        m = np.array([singwphi, -cosgwphi, 0.0])
        n = np.array([-cosgwtheta*cosgwphi, -cosgwtheta*singwphi, singwtheta])
        omhat = np.array([-singwtheta*cosgwphi, -singwtheta*singwphi, -cosgwtheta])

        # various factors invloving GW parameters
        fac1 = 256/5 * mc**(5/3) * w0**(8/3)
        fac2 = 1/32/mc**(5/3)
        fac3 = mc**(5/3)/dist

        # get antenna patterns
        fplus = 0.5 * (np.dot(m, phat)**2 - np.dot(n, phat)**2) / (1+np.dot(omhat, phat))
        fcross = (np.dot(m, phat)*np.dot(n, phat)) / (1 + np.dot(omhat, phat))
        cosMu = -np.dot(omhat, phat)

        # get values from pulsar object
        if pphase is not None:
            pd = pphase/(2*np.pi*fgw*(1-cosMu)) / KPC2S
        else:
            pd = pdist

        # convert units
        pd *= KPC2S   # convert from kpc to seconds

        # get pulsar time
        tp = toas-pd*(1-cosMu)

        # evolution
        if evolve:

            # calculate time dependent frequency at earth and pulsar
            omega = w0 * (1 - fac1 * toas)**(-3/8)
            omega_p = w0 * (1 - fac1 * tp)**(-3/8)

            # calculate time dependent phase
            phase = phase0 + fac2 * (w053 - omega**(-5/3))
            phase_p = phase0 + fac2 * (w053 - omega_p**(-5/3))

        # use approximation that frequency does not evlolve over observation time
        elif phase_approx:

            # frequencies
            omega = w0 * np.ones(toas.size) #make sure omega is always an array and never a float (numba typing stuff)
            omega_p = w0 * (1 + fac1 * pd*(1-cosMu))**(-3/8) * np.ones(toas.size) #make sure omega_p is always an array and never a float (numba typing stuff)

            # phases
            phase = phase0 + omega * toas
            phase_p = phase0 + fac2 * (w053 - omega_p**(-5/3)) + omega_p*toas

        # no evolution
        else:

            # monochromatic
            omega = w0 * np.ones(toas.size) #make sure omega is always an array and never a float (numba typing stuff)
            omega_p = omega

            # phases
            phase = phase0 + omega * toas
            phase_p = phase0 + omega * tp


        # define time dependent coefficients
        At = np.sin(2*phase) * incfac1
        Bt = np.cos(2*phase) * incfac2
        At_p = np.sin(2*phase_p) * incfac1
        Bt_p = np.cos(2*phase_p) * incfac2

        # now define time dependent amplitudes
        alpha = fac3 / omega**(1/3)
        alpha_p = fac3 / omega_p**(1/3)

        # define rplus and rcross
        rplus = alpha * (At*cos2psi + Bt*sin2psi)
        rcross = alpha * (-At*sin2psi + Bt*cos2psi)
        rplus_p = alpha_p * (At_p*cos2psi + Bt_p*sin2psi)
        rcross_p = alpha_p * (-At_p*sin2psi + Bt_p*cos2psi)

        # residuals
        if psrTerm:
            #make sure we add zeros rather than NaNs in the rare occasion when the binary already merged and produces negative frequencies
            rrr = fplus*(rplus_p-rplus)+fcross*(rcross_p-rcross)
            res += np.where(np.isnan(rrr), 0.0, rrr)
        else:
            rrr = -fplus*rplus - fcross*rcross
            res += np.where(np.isnan(rrr), 0.0, rrr)

    return res


#@profile
def add_gwb_plus_outlier_cws(psrs, vals, weights, fobs, T_obs, outlier_per_bin=100, seed=None):
    """Function to create realistic datasets based on holodeck SMBHB populations by injecting loudest binaries individually and the rest as a GWB. Based on methods described in Becsy, Cornish, Kelley (2022)
    
    Parameters
    ----------
    psrs : array of Pulsar objects
        pulsars
    vals : array of 4 1Darrays
        [Mtots, Mrats, redZs, Fobs] for each BBH bin
        Mtots and Mrats use rest frame values
    weights : 1Darray
        number of binaries in each bin
    fobs : 1Darray
        Frequency bin edges [Hz]
    T_obs : float
        PTA observing duration in cgs units (s)
    outlier_per_bin : int
        Number of loudest MBHBs to treat individually per bin
    seed : int
        Random seed

    Returns
    -------
    f_centers : 1Darray
        Array of frequency bin centers [Hz] of length F
    free_spec : 1Darray
        GWB spectrum excluding outliers, hc in each frequency bin
    outlier_fo : 1Darray
        Observed frequency of each outlier [Hz]
        length N_outliers * F
    outlier_hs : 1Darray
        Observed hc^2 = h_s^2 * f/df of each outlier
        length N_outlier * F
    outlier_mc : 1Darray
        Observed chirp masses of outlier SMBHBs [solar masses]
    outlier_dl : 1Darray
        Luminosity distances to outlier SMBHBs [Mpc]
    random_gwthetas : 1Darray
        Randomly assigned polar angle of outliers in celestial coords [radians]  
    random_gwphis : 1Darray
        Randomly-assigned azimuthal angle of outliers in celestial coords [radians] 
    random_phases : 1Darray
        Randomly assigned initial Phase of outliers [radians] 
    random_psis : 1Darray
        Randomly size polarizations of outliers [radians]
    random_incs : 1Darray
        Radomly-assigned inclinations of outliers [radians]

    """
    PC = ap.constants.pc.cgs.value
    MSOL = ap.constants.M_sun.cgs.value
    
    f_centers = []
    for iii in range(fobs.size-1):
        f_centers.append((fobs[iii+1]+fobs[iii])/2)
        
    f_centers = np.array(f_centers)
    
    mc = utils.chirp_mass(*utils.m1m2_from_mtmr(vals[0], vals[1])) # rest frame
    rz = vals[2, :]
    frst = vals[3] * (1.0 + rz)
    #get comoving distance for h calculation
    dc = cosmo.z_to_dcom(rz)
    #and luminosity distance for outlier injections below
    dl = np.copy(dc) * (1.0 + rz) #cosmo.luminosity_distance(rz).cgs.value
    
    hs = utils.gw_strain_source(mc, dc, frst/2)
    #testing
    #speed_of_light = 299792458.0 #m/s
    #T_sun = 1.327124400e20 / speed_of_light**3
    #hs_mine = 8/np.sqrt(10) * (mc/MSOL*T_sun)**(5/3) * (np.pi*frst)**(2/3) / (dc/100.0) * speed_of_light
    #8/sqrt(10) * pi^2/3 * G^5/3 / c^4 * M_c^5/3* (2*f_orb_rest)^2/3 / dcom
    #print(hs[0], hs_mine[0])
    fo = vals[-1]

    #convert mc to observer frame since that's what we will need for outlier injection below
    mc = mc * (1.0 + rz) 
    
    freq_idxs = np.digitize(fo,fobs)
    
    free_spec = np.ones(fobs.shape[0]-1)*1e-100
    outlier_hs = np.zeros(free_spec.shape[0]*outlier_per_bin)
    outlier_fo = np.zeros(free_spec.shape[0]*outlier_per_bin)
    outlier_mc = np.zeros(free_spec.shape[0]*outlier_per_bin)
    outlier_dl = np.zeros(free_spec.shape[0]*outlier_per_bin)
    
    weighted_h_square = weights * hs**2 * fo * T_obs #apply weights and convert to characteristic strain
    for k in range(free_spec.shape[0]):
        bool_mask = (freq_idxs-1)==k
        weighted_h_square_bin = weighted_h_square[bool_mask]
        sort_idx = np.argsort(weighted_h_square_bin)[::-1]
        weighted_h_squared_bin_sorted = weighted_h_square_bin[sort_idx]
        fo_bin_sorted = fo[bool_mask][sort_idx] #Hz
        mc_bin_sorted = mc[bool_mask][sort_idx]/MSOL #solar mass
        dl_bin_sorted = dl[bool_mask][sort_idx]/PC/1e6 #Mpc
        
        if outlier_per_bin<weighted_h_squared_bin_sorted.shape[0]:
            outlier_limit = outlier_per_bin
        else:
            outlier_limit = weighted_h_squared_bin_sorted.shape[0]
        
        for j in range(outlier_limit):
            outlier_hs[outlier_per_bin*k+j] = weighted_h_squared_bin_sorted[j]
            outlier_fo[outlier_per_bin*k+j] = fo_bin_sorted[j]
            outlier_mc[outlier_per_bin*k+j] = mc_bin_sorted[j]
            outlier_dl[outlier_per_bin*k+j] = dl_bin_sorted[j]
        
        free_spec[k] += np.sum(weighted_h_squared_bin_sorted[outlier_per_bin:])
    
    FreeSpec = np.array([f_centers,np.sqrt(free_spec)]).T
    
    # print(FreeSpec)

    howml = 10
    
    add_gwb(psrs, None, None, userSpec=FreeSpec, howml=howml, seed=seed)
    
    #for pulsar in psrs:
    #    #print("GWB")
    #    #print(pulsar.name)
    #    #print(pulsar.residuals())
    #    #pulsar.fit()
        
    #filter out empty entries in outliers
    outlier_hs = outlier_hs[np.where(outlier_hs>0)]
    outlier_fo = outlier_fo[np.where(outlier_fo>0)]
    outlier_mc = outlier_mc[np.where(outlier_mc>0)]
    outlier_dl = outlier_dl[np.where(outlier_dl>0)]
        
    N_CW = outlier_hs.shape[0]
    
    random_gwthetas = np.arccos(np.random.uniform(low=-1.0, high=1.0, size=N_CW))
    random_gwphis = np.random.uniform(low=0.0, high=2*np.pi, size=N_CW)
    random_phases = np.random.uniform(low=0.0, high=2*np.pi, size=N_CW)
    random_psis = np.random.uniform(low=0.0, high=np.pi, size=N_CW)
    random_incs = np.arccos(np.random.uniform(low=-1.0, high=1.0, size=N_CW))
    
    for pulsar in psrs:
        add_catalog_of_cws(pulsar,
                           gwtheta_list=random_gwthetas, gwphi_list=random_gwphis, 
                           mc_list=outlier_mc, dist_list=outlier_dl, fgw_list=outlier_fo,
                           phase0_list=random_phases, psi_list=random_psis, inc_list=random_incs,
                           pdist=1.0, pphase=None, psrTerm=True, evolve=True,
                           phase_approx=False, tref=53000*86400)
    
    #for pulsar in psrs:
    #    #print("GWB+CW")
    #    #print(pulsar.residuals())
    #    #pulsar.fit()
    
    return f_centers, free_spec, outlier_fo, outlier_hs, outlier_mc, outlier_dl, random_gwthetas, random_gwphis, random_phases, random_psis, random_incs


def add_burst(psr, gwtheta, gwphi, waveform_plus, waveform_cross, psi=0.0, tref=0, remove_quad=False, signal_name='burst'):
    """
    Function to create GW-induced residuals from an arbitrary GW waveform assuming elliptical polarization.
    :param psr: pulsar object
    :param gwtheta: Polar angle of GW source in celestial coords [radians]
    :param gwphi: Azimuthal angle of GW source in celestial coords [radians]
    :param waveform_plus: Function defining the plus polarized waveform of the GW [function]
    :param waveform_cross: Function defining the cross polarized waveform of the GW [function]
    :param psi: Polarization angle [radians]. Mixes h+ and hx corresponding to rotation along the propagation direction (see eq. (7.24-25) in Maggiore Vol1, 2008).
    :param tref: Start time, such that gw_waveform gets t-tref as the time argument
    :param remove_quad: Fit out quadratic from residual if True to simulate f and fdot timing fit.
    :returns: Vector of induced residuals
    """

    # define variable for later use
    cosgwtheta, cosgwphi = np.cos(gwtheta), np.cos(gwphi)
    singwtheta, singwphi = np.sin(gwtheta), np.sin(gwphi)

    # unit vectors to GW source
    m = np.array([singwphi, -cosgwphi, 0.0])
    n = np.array([-cosgwtheta * cosgwphi, -cosgwtheta * singwphi, singwtheta])
    omhat = np.array([-singwtheta * cosgwphi, -singwtheta * singwphi, -cosgwtheta])

    # pulsar location
    if "RAJ" and "DECJ" in psr.loc.keys():
        ptheta = np.pi / 2 - psr.loc["DECJ"]*np.pi/180.0
        pphi = psr.loc["RAJ"]*np.pi/12.0
    elif "ELONG" and "ELAT" in psr.loc.keys():
        fac = 1.0#180.0 / np.pi #no need in pint
        if "B" in psr.name:
            epoch = "1950"
        else:
            epoch = "2000"
        coords = ephem.Equatorial(ephem.Ecliptic(str(psr.loc["ELONG"] * fac), str(psr.loc["ELAT"] * fac)), epoch=epoch)

        ptheta = np.pi / 2 - float(repr(coords.dec))
        pphi = float(repr(coords.ra))

    # use definition from Sesana et al 2010 and Ellis et al 2012
    phat = np.array([np.sin(ptheta) * np.cos(pphi), np.sin(ptheta) * np.sin(pphi), np.cos(ptheta)])

    fplus = 0.5 * (np.dot(m, phat) ** 2 - np.dot(n, phat) ** 2) / (1 + np.dot(omhat, phat))
    fcross = (np.dot(m, phat) * np.dot(n, phat)) / (1 + np.dot(omhat, phat))

    # get toas from pulsar object
    toas = psr.toas.get_mjds().value * 86400 - tref

    # define residuals: hplus and hcross
    hplus = waveform_plus(toas)
    hcross = waveform_cross(toas)

    #apply rotation by psi angle (see e.g. eq. (7.24-25) in Maggiore Vol1, 2008)
    rplus = hplus*np.cos(2*psi) - hcross*np.sin(2*psi)
    rcross = hplus*np.sin(2*psi) + hcross*np.cos(2*psi)

    # residuals
    res = -fplus*rplus - fcross*rcross

    if remove_quad:
        pp = np.polyfit(np.array(toas, dtype=np.double), np.array(res, dtype=np.double), 2)
        res = res - pp[0]*toas**2 -pp[1]*toas - pp[2]

    dt = res * u.s
    
    psr.update_added_signals('{}_'.format(psr.name)+signal_name,
                             {'gwtheta': gwtheta,
                              'gwphi':gwphi,
                              'waveform_plus':waveform_plus,
                              'waveform_cross':waveform_cross,
                              'psi':psi,
                              'tref':tref,
                              'remove_quad':remove_quad},
                             dt)
    
    psr.toas.adjust_TOAs(TimeDelta(dt.to('day')))
    psr.update_residuals()


def add_noise_transient(psr, waveform, tref=0, signal_name='noise_transient'):
    """
    Function to create incoherent residuals of arbitrary waveform in a given pulsar.
    :param psr: pulsar object
    :param waveform: Function defining the waveform of the glitch [function]
    :param tref: Start time, such that gw_waveform gets t-tref as the time argument
    :returns: Vector of induced residuals
    """

    # get toas from pulsar object
    toas = psr.toas.get_mjds().value * 86400 - tref

    # call waveform function
    res = waveform(toas)

    dt = res * u.s
    
    psr.update_added_signals('{}_'.format(psr.name)+signal_name,
                             {'waveform':waveform,
                              'tref':tref},
                             dt)
    
    psr.toas.adjust_TOAs(TimeDelta(dt.to('day')))
    psr.update_residuals()


def add_gw_memory(psr, strain, gwtheta, gwphi, bwm_pol, t0_mjd, signal_name='gw_memory'):
    """
    Function to add residuals due to a burst with memory (based on code from Jerry Sun)
    :param psr: pulsar object
    :param strain: Strain amplitude of burst
    :param gwtheta: Sky location of theta of source [radians]
    :param gwphi: Sky location of phi of source [radians]
    :param bwm_pol: Polarization angle [radians]
    :param t0_mjd: Burst epoch [MJD]
    """

    # define variable for later use
    cosgwtheta, cosgwphi = np.cos(gwtheta), np.cos(gwphi)
    singwtheta, singwphi = np.sin(gwtheta), np.sin(gwphi)

    # unit vectors to GW source
    m = np.array([singwphi, -cosgwphi, 0.0])
    n = np.array([-cosgwtheta * cosgwphi, -cosgwtheta * singwphi, singwtheta])
    omhat = np.array([-singwtheta * cosgwphi, -singwtheta * singwphi, -cosgwtheta])

    # pulsar location
    if "RAJ" and "DECJ" in psr.loc.keys():
        ptheta = np.pi / 2 - psr.loc["DECJ"]*np.pi/180.0
        pphi = psr.loc["RAJ"]*np.pi/12.0
    elif "ELONG" and "ELAT" in psr.loc.keys():
        fac = 1.0#180.0 / np.pi #no need in pint
        if "B" in psr.name:
            epoch = "1950"
        else:
            epoch = "2000"
        coords = ephem.Equatorial(ephem.Ecliptic(str(psr.loc["ELONG"] * fac), str(psr.loc["ELAT"] * fac)), epoch=epoch)

        ptheta = np.pi / 2 - float(repr(coords.dec))
        pphi = float(repr(coords.ra))

    # use definition from Sesana et al 2010 and Ellis et al 2012
    phat = np.array([np.sin(ptheta) * np.cos(pphi), np.sin(ptheta) * np.sin(pphi), np.cos(ptheta)])

    fplus = 0.5 * (np.dot(m, phat) ** 2 - np.dot(n, phat) ** 2) / (1 + np.dot(omhat, phat))
    fcross = (np.dot(m, phat) * np.dot(n, phat)) / (1 + np.dot(omhat, phat))
    #cosMu = -np.dot(omhat, phat)

    pol = np.cos(2 * bwm_pol) * fplus + np.sin(2*bwm_pol) * fcross

    #get the toas
    toas = psr.toas.get_mjds().value * 86400
    t0_sec = t0_mjd * 86400
    dt = np.zeros(len(toas))
    for toa_idx, toa in enumerate(toas):
        # print(type(toa))
        dt[toa_idx] = 0 if toa < t0_sec else (pol * strain * (toa - t0_sec))
    dt = dt*u.s
    
    psr.update_added_signals('{}_'.format(psr.name)+signal_name,
                             {'strain':strain,
                              'gwtheta': gwtheta,
                              'gwphi':gwphi,
                              'bwm_pol':bwm_pol,
                              't0_mjd':t0_mjd},
                             dt)
    
    psr.toas.adjust_TOAs(TimeDelta(dt.to('day')))
    psr.update_residuals()
