import scipy.constants as sc

DAY_IN_SEC = 86400
YEAR_IN_SEC = 365.25 * DAY_IN_SEC
DMk = 4.15e3  # Units MHz^2 cm^3 pc sec
SOLAR2S = sc.G / sc.c**3 * 1.98855e30
KPC2S = sc.parsec / sc.c * 1e3
MPC2S = sc.parsec / sc.c * 1e6
