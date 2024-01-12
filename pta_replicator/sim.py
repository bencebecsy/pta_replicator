"""
Code to make simulated PTA datasets with PINT
Created by Bence Becsy, Jeff Hazboun, Aaron Johnson
With code adapted from libstempo (Michele Vallisneri)
"""
import glob, os
import numpy as np
import ephem
import astropy as ap
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.cosmology import Planck18 as cosmo

import scipy.constants as sc
from scipy import interpolate as interp
from numba import njit, prange

from holodeck import utils
from pint.residuals import Residuals
import pint.toa as toa
from pint import models
import pint.fitter
import spharmORFbasis as anis

DAY_IN_SEC = 86400
YEAR_IN_SEC = 365.25 * DAY_IN_SEC
DMk = 4.15e3  # Units MHz^2 cm^3 pc sec
SOLAR2S = sc.G / sc.c**3 * 1.98855e30
KPC2S = sc.parsec / sc.c * 1e3
MPC2S = sc.parsec / sc.c * 1e6

class SimulatedPulsar:
    """
    Class to hold properties of a simulated pulsar
    """
    def __init__(self, parfile, timfile, ephem='DE440'):
        if not os.path.isfile(parfile):
            raise FileNotFoundError("par file does not exist.")
        if not os.path.isfile(timfile):
            raise FileNotFoundError("tim file does not exist.")
        self.parfile = parfile
        self.timfile = timfile
        self.ephem = ephem
        self.model = models.get_model(parfile)
        self.toas = toa.get_TOAs(timfile, ephem=ephem)
        self.rs = Residuals(self.toas, self.model)
        self.name = self.model.PSR.value

        try:
            self.loc = {'RAJ': self.model.RAJ.value, 'DECJ': self.model.DECJ.value}
        except KeyError:
            self.loc = {'ELONG': self.model.ELONG.value, 'ELAT': self.model.ELAT.value}
        
    def update_residuals(self):
        """Method to take the current TOAs and model and update the residuals with them"""
        self.rs = Residuals(self.toas, self.model)

    def write_partim(self, outpar, outtim, tempo2=False):
        """Format for either PINT or Tempo2"""
        self.model.write_parfile(outpar)
        if tempo2:
            self.toas.write_TOA_file(outtim, format='Tempo2')
        else:
            self.toas.write_TOA_file(outtim)
    
    def fit(self):
        """Refit the timing model and update everything"""
        #self.f = pint.fitter.WLSFitter(self.toas, self.model)
        #self.f = pint.fitter.GLSFitter(self.toas, self.model)
        #self.f = pint.fitter.DownhillGLSFitter(self.toas, self.model)
        self.f = pint.fitter.Fitter.auto(self.toas, self.model)
        self.f.fit_toas()
        self.model = self.f.model
        self.update_residuals()


def load_pulsar(parfile, timfile, ephem='DE440'):
    


def load_psrs(par_dir, tim_dir, ephem='DE440', num_psrs=None):
    '''
    Loads a parfile and timfile and returns a pint.toas and pint.model object.
    '''
    unfiltered_pars = sorted(glob.glob(par_dir + "/*.par"))
    filtered_pars = [p for p in unfiltered_pars if ".t2" not in p]
    unfiltered_tims = sorted(glob.glob(tim_dir + "/*.tim"))
    combo_list = list(zip(filtered_pars, unfiltered_tims))
    psrs = []
    for par, tim in combo_list:
        if num_psrs:
            if len(psrs) >= num_psrs:
                break
        psrs.append(SimulatedPulsar(par, tim, ephem=ephem))
    return psrs


def make_ideal(psr, iterations=2):
    '''
    Takes a pint.toas and pint.model object and effectively zeros out the residuals.
    '''
    for ii in range(iterations):
        rs=Residuals(psr.toas, psr.model)
        psr.toas.adjust_TOAs(TimeDelta(-1.0*rs.time_resids))
    psr.update_residuals()


def createfourierdesignmatrix_red(toas, nmodes=30, Tspan=None,
                                  logf=False, fmin=None, fmax=None,
                                  pshift=False, modes=None):
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


def add_rednoise(psr, A, gamma, components=30,
                 seed=None, modes=None, Tspan=None):
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


def add_efac(psr, efac, flagid=None, flags=None, seed=None):
    """Add quadrature noise of rms `equad` [s].
       Optionally take a pseudorandom-number-generator seed."""

    if seed is not None:
        np.random.seed(seed)

    # default equadvec
    efacvec = np.zeros(psr.toas.ntoas)

    # check that equad is scalar if flags is None
    if flags is None:
        if not np.isscalar(efac):
            raise ValueError('ERROR: If flags is None, efac must be a scalar')
        else:
            efacvec = np.ones(psr.toas.ntoas) * efac

    if flags is not None and flagid is not None and not np.isscalar(efac):
        if len(efac) == len(flags):
            for ct, flag in enumerate(flags):
                ind = flag == np.array([f['f'] for f
                                        in psr.toas.table['flags'].data])
                efacvec[ind] = efac[ct]

    dt = efacvec * psr.toas.get_errors().to('s') * np.random.randn(psr.toas.ntoas)
    psr.toas.adjust_TOAs(TimeDelta(dt.to('day')))
    psr.update_residuals()

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
):
    """
    Function to add CW residuals (adapted from libstempo.toasim)

    :param psr: pulsar object
    :param gwtheta: Polar angle of GW source in celestial coords [radians]
    :param gwphi: Azimuthal angle of GW source in celestial coords [radians]
    :param mc: Chirp mass of SMBMB [solar masses]
    :param dist: Luminosity distance to SMBMB [Mpc]
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
        ptheta = np.pi / 2 - psr.loc["DECJ"]
        pphi = psr.loc["RAJ"]
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
    psr.toas.adjust_TOAs(TimeDelta(dt.to('day')))
    psr.update_residuals()

def add_catalog_of_cws(psr, gwtheta_list, gwphi_list, mc_list, dist_list, fgw_list, phase0_list, psi_list, inc_list, pdist=1.0,
                       pphase=None, psrTerm=True, evolve=True, phase_approx=False, tref=0, chunk_size=10_000_000):
    """
    Method to add a list of many SMBHBs more efficiently than by calling add_cgw multiple times. It takes the same input as add_cgw, except as lists.
    """

    # pulsar location
    if "RAJ" and "DECJ" in psr.loc.keys():
        ptheta = np.pi / 2 - psr.loc["DECJ"]
        pphi = psr.loc["RAJ"]
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
            psr.toas.adjust_TOAs(TimeDelta(dt.to('day')))
            psr.update_residuals()
    else:
        res = loop_over_CWs(phat.astype('float64'), toas.astype('float64'), gwtheta_list, gwphi_list, mc_list, dist_list, fgw_list, phase0_list, psi_list, inc_list, pdist=pdist,
                            pphase=pphase, psrTerm=psrTerm, evolve=evolve, phase_approx=phase_approx)

        #End of loop over CW sources
        #Now add residual to TOAs
        dt = res * u.s
        psr.toas.adjust_TOAs(TimeDelta(dt.to('day')))
        psr.update_residuals()

#@profile
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
def add_gwb_plus_outlier_cws(psrs, vals, weights, fobs, T_obs, outlier_per_bin=100, seed=1234):
    """Function to create realistic datasets based on holodeck SMBHB populations by injecting loudest binaries individually and the rest as a GWB. Based on methods described in Becsy, Cornish, Kelley (2022)"""
    PC = ap.constants.pc.cgs.value
    MSOL = ap.constants.M_sun.cgs.value
    
    f_centers = []
    for iii in range(fobs.size-1):
        f_centers.append((fobs[iii+1]+fobs[iii])/2)
        
    f_centers = np.array(f_centers)
    
    mc = utils.chirp_mass(*utils.m1m2_from_mtmr(vals[0], vals[1]))
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
    
    print(FreeSpec)

    howml = 10
    
    create_gwb(psrs, None, None, userSpec=FreeSpec, howml=howml, seed=seed)
    
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


def create_gwb(
    psrs,
    Amp,
    gam,
    noCorr=False,
    seed=None,
    turnover=False,
    clm=[np.sqrt(4.0 * np.pi)],
    lmax=0,
    f0=1e-9,
    beta=1,
    power=1,
    userSpec=None,
    npts=600,
    howml=10,
):
    """
    Function to create GW-induced residuals from a stochastic GWB as defined
    in Chamberlin, Creighton, Demorest, et al. (2014).
    :param toas_list: list of toas to inject GWB into
    :param Amp: Amplitude of red noise in GW units
    :param gam: Red noise power law spectral index
    :param noCorr: Add red noise with no spatial correlations
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
    if noCorr:
        ORF = np.diag(np.ones(Npulsars) * 2)
    else:
        psrlocs = np.zeros((Npulsars, 2))

        for ii in range(Npulsars):
            if "RAJ" and "DECJ" in psrs[ii].loc:
                psrlocs[ii] = np.double(psrs[ii]["RAJ"]), np.double(psrs[ii]["DECJ"])
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
        res_gw.append(f(psrs[ll].toas.get_mjds().value.astype(np.float) * 86400))

    # return res_gw
    ct = 0
    for psr in psrs:
        dt = res_gw[ct] / 86400.0 * u.day
        psr.toas.adjust_TOAs(TimeDelta(dt.to('day')))
        psr.update_residuals()
        ct += 1

def add_burst(psr, gwtheta, gwphi, waveform_plus, waveform_cross, psi=0.0, tref=0, remove_quad=False):
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
        ptheta = np.pi / 2 - psr.loc["DECJ"]
        pphi = psr.loc["RAJ"]
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
    psr.toas.adjust_TOAs(TimeDelta(dt.to('day')))
    psr.update_residuals()

def add_noise_transient(psr, waveform, tref=0):
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
    psr.toas.adjust_TOAs(TimeDelta(dt.to('day')))
    psr.update_residuals()
