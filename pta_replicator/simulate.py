"""
Code to make simulated PTA datasets with PINT
Created by Bence Becsy, Jeff Hazboun, Aaron Johnson
With code adapted from libstempo (Michele Vallisneri)

"""
import glob
import os
from dataclasses import dataclass
from astropy.time import TimeDelta

from pint.residuals import Residuals
import pint.toa as toa
from pint import models
import pint.fitter
import tqdm

from enterprise.pulsar import Pulsar


@dataclass
class SimulatedPulsar:
    """
    Class to hold properties of a simulated pulsar
    """
    ephem: str = 'DE440'
    model: models.TimingModel = None
    toas: toa.TOAs = None
    residuals: Residuals = None
    name: str = None
    loc: dict = None
    added_signals: dict = None
    added_signals_time: dict = None

    def __repr__(self):
        return f"SimulatedPulsar({self.name})"

    def update_residuals(self):
        """Method to take the current TOAs and model and update the residuals with them"""
        self.residuals = Residuals(self.toas, self.model)

    def fit(self, fitter='auto', **fitter_kwargs):
        """
        Refit the timing model and update everything

        Parameters
        ----------
        fitter : str
            Type of fitter to use [auto]
        fitter_kwargs :
            Kwargs to pass onto fit_toas. Can be useful to set parameters such as max_chi2_increase, min_lambda, etc.
        """
        if fitter == 'wls':
            self.f = pint.fitter.WLSFitter(self.toas, self.model)
        elif fitter == 'gls':
            self.f = pint.fitter.GLSFitter(self.toas, self.model)
        elif fitter == 'downhill':
            self.f = pint.fitter.DownhillGLSFitter(self.toas, self.model)
        elif fitter == 'auto':
            self.f = pint.fitter.Fitter.auto(self.toas, self.model)
        else:
            err = f"{fitter=} must be one of 'wls', 'gls', 'downhill' or 'auto'"
            raise ValueError(err)
        
        self.f.fit_toas(**fitter_kwargs)
        self.model = self.f.model
        self.update_residuals()

    def write_partim(self, outpar: str, outtim: str, tempo2: bool = False):
        """Format for either PINT or Tempo2"""
        self.model.write_parfile(outpar)
        if tempo2:
            self.toas.write_TOA_file(outtim, format='Tempo2')
        else:
            self.toas.write_TOA_file(outtim)

    def update_added_signals(self, signal_name, param_dict, dt=None):
        """
        Update the timing model with a new signal
        """
        if signal_name in self.added_signals:
            raise ValueError(f"{signal_name} already exists in the model.")
        self.added_signals[signal_name] = param_dict
        if dt is not None:
            self.added_signals_time[signal_name] = dt

    def to_enterprise(self, ephem='DE440'):
        """
        Convert to enterprise PintPulsar object
        """
        return Pulsar(self.toas, self.model, ephem=ephem, timing_package='pint')


def load_pulsar(parfile: str, timfile: str, ephem:str = 'DE440') -> SimulatedPulsar:
    """
    Load a SimulatedPulsar object from a par and tim file

    Parameters
    ----------
    parfile : str
        Path to par file
    timfile : str
        Path to tim file
    """
    if not os.path.isfile(parfile):
        raise FileNotFoundError("par file does not exist.")
    if not os.path.isfile(timfile):
        raise FileNotFoundError("tim file does not exist.")

    model = models.get_model(parfile)
    toas = toa.get_TOAs(timfile, ephem=ephem, planets=True)
    residuals = Residuals(toas, model)
    name = model.PSR.value

    if hasattr(model, 'RAJ') and hasattr(model, 'DECJ'):
        loc = {'RAJ': model.RAJ.value, 'DECJ': model.DECJ.value}
    elif hasattr(model, 'ELONG') and hasattr(model, 'ELAT'):
        loc = {'ELONG': model.ELONG.value, 'ELAT': model.ELAT.value}
    else:
        raise AttributeError("No pulsar location information (RAJ/DECJ or ELONG/ELAT) in parfile.")
    

    return SimulatedPulsar(ephem=ephem, model=model, toas=toas, residuals=residuals, name=name, loc=loc)


def load_from_directories(pardir: str, timdir: str, ephem:str = 'DE440', num_psrs: int = None, debug=False) -> list:
    """
    Takes a directory of par files and a directory of tim files and
    loads them into a list of SimulatedPulsar objects
    """
    if not os.path.isdir(pardir):
        raise FileNotFoundError("par directory does not exist.")
    if not os.path.isdir(timdir):
        raise FileNotFoundError("tim directory does not exist.")
    unfiltered_pars = sorted(glob.glob(pardir + "/*.par"))
    filtered_pars = [p for p in unfiltered_pars if ".t2" not in p]
    unfiltered_tims = sorted(glob.glob(timdir + "/*.tim"))
    combo_list = list(zip(filtered_pars, unfiltered_tims))
    psrs = []
    for par, tim in combo_list:
        if num_psrs:
            if len(psrs) >= num_psrs:
                break
        if debug: print(f"loading {par=}, {tim=}")
        psrs.append(load_pulsar(par, tim, ephem=ephem))
    return psrs


def make_ideal(psr: SimulatedPulsar, iterations: int = 2):
    """
    Takes a pint.TOAs and pint.TimingModel object and effectively zeros out the residuals.
    """
    for ii in range(iterations):
        residuals = Residuals(psr.toas, psr.model)
        psr.toas.adjust_TOAs(TimeDelta(-1.0*residuals.time_resids))
    psr.added_signals = {}
    psr.added_signals_time = {}
    psr.update_residuals()
