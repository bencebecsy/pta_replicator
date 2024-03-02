from pta_replicator.white_noise import add_measurement_noise
from pta_replicator.white_noise import add_jitter
from pta_replicator.red_noise import add_red_noise, add_gwb
from pta_replicator.deterministic import add_gw_memory, add_cgw, add_burst, add_noise_transient

from pta_replicator.simulate import load_pulsar, load_from_directories
from pta_replicator.simulate import make_ideal
import pint
import sys
import json
import numpy as np


pint.logging.setup(sink=sys.stderr, level="WARNING", usecolors=True)

pardir = '../test_partim_small/par'
timdir = '../test_partim_small/tim'

psrs = load_from_directories(pardir, timdir, num_psrs=3)  # Load 3 pulsars

for ii, psr in enumerate(psrs):
    ## make ideal
    make_ideal(psr)

#seed_wn = 12345
#for ii, psr in enumerate(psrs):
#    ## add measurement noise
#    add_measurement_noise(psr,
#                          efac = 1.0,
#                          log10_equad = np.log10(3e-7),
#                          seed = seed_wn + ii)

#seed_rn = 12345
#for ii, psr in enumerate(psrs):
#    #add rn
#    add_red_noise(psr,
#                  -15,
#                  4.2,
#                  components=30,
#                  Tspan=None,
#                  seed = seed_rn + ii)

add_gwb(psrs, -14, 4.33, seed=123456)

residuals = np.zeros((3,122))
for i in range(3):
    residuals[i,:] = psrs[i].residuals.resids_value

npzfile = np.load("libstempo_test_residuals.npz")
libstempo_residuals = npzfile['residuals']

#print(residuals)
#print(libstempo_residuals)
#print(libstempo_residuals-residuals)
#print(np.all(libstempo_residuals==residuals))
print(np.all(((residuals-libstempo_residuals)/np.sqrt(np.mean(residuals**2)))<1e-3))
