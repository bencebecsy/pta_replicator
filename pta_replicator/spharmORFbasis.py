"""
Created by stevertaylor
Copyright (c) 2014 Stephen R. Taylor

Code contributions by Rutger van Haasteren (piccard), Justin Ellis (PAL/PAL2), and Chiara Mingarelli.

"Numbafied" by Aaron Johnson
"""

# from cmath import exp

from numba import njit
import numpy as np
from scipy import special as sp

NORM = 3.0 / (8 * np.pi)


@njit
def calczeta(phi1, phi2, theta1, theta2):
    """
    Calculate the angular separation between position (phi1, theta1) and
    (phi2, theta2)

    """

    zeta = 0.0

    if phi1 == phi2 and theta1 == theta2:
        zeta = 0.0
    else:
        argument = np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2) + np.cos(theta1) * np.cos(theta2)

        if argument < -1:
            zeta = np.pi
        elif argument > 1:
            zeta = 0.0
        else:
            zeta = np.arccos(argument)

    return zeta

"""
Following functions taken from Gair et. al (2014),
involving solutions of integrals to define the ORF for an
arbitrarily anisotropic GW background.
"""

@njit
def Fminus00(qq, mm, ll, zeta):
    """Taken from Gair et. al (2014)"""
    integrand = 0.0

    for ii in range(0, qq + 1):
        for jj in range(mm, ll + 1):

            integrand += (
                (2.0 ** (ii - jj) * (-1.0) ** (qq - ii + jj + mm))
                * (
                    sp.factorial(qq)
                    * sp.factorial(ll + jj)
                    * (2.0 ** (qq - ii + jj - mm + 1) - (1.0 + np.cos(zeta)) ** (qq - ii + jj - mm + 1))
                )
                / (
                    sp.factorial(ii)
                    * sp.factorial(qq - ii)
                    * sp.factorial(jj)
                    * sp.factorial(ll - jj)
                    * sp.factorial(jj - mm)
                    * (qq - ii + jj - mm + 1)
                )
            )

    return integrand


@njit
def Fminus01(qq, mm, ll, zeta):

    integrand = 0.0

    for ii in range(0, qq + 1):
        for jj in range(mm, ll + 1):

            integrand += (
                (2.0 ** (ii - jj) * (-1.0) ** (qq - ii + jj + mm))
                * (
                    sp.factorial(qq)
                    * sp.factorial(ll + jj)
                    * (2.0 ** (qq - ii + jj - mm + 2) - (1.0 + np.cos(zeta)) ** (qq - ii + jj - mm + 2))
                )
                / (
                    sp.factorial(ii)
                    * sp.factorial(qq - ii)
                    * sp.factorial(jj)
                    * sp.factorial(ll - jj)
                    * sp.factorial(jj - mm)
                    * (qq - ii + jj - mm + 2)
                )
            )

    return integrand


@njit
def Fplus01(qq, mm, ll, zeta):

    integrand = 0.0

    for ii in range(0, qq):
        for jj in range(mm, ll + 1):
            integrand += (
                (2.0 ** (ii - jj) * (-1.0) ** (ll + qq - ii + jj))
                * (
                    sp.factorial(qq)
                    * sp.factorial(ll + jj)
                    * (2.0 ** (qq - ii + jj - mm) - (1.0 - np.cos(zeta)) ** (qq - ii + jj - mm))
                )
                / (
                    sp.factorial(ii)
                    * sp.factorial(qq - ii)
                    * sp.factorial(jj)
                    * sp.factorial(ll - jj)
                    * sp.factorial(jj - mm)
                    * (qq - ii + jj - mm)
                )
            )

    if mm == ll:
        integrand += 0.0
    else:
        for jj in range(mm + 1, ll + 1):
            integrand += (
                (2.0 ** (qq - jj) * (-1.0) ** (ll + jj))
                * (sp.factorial(ll + jj) * (2.0 ** (jj - mm) - (1.0 - np.cos(zeta)) ** (jj - mm)))
                / (sp.factorial(jj) * sp.factorial(ll - jj) * sp.factorial(jj - mm) * (jj - mm))
            )

    integrand += ((-1.0) ** (ll + mm) * 2.0 ** (qq - mm) * sp.factorial(ll + mm) * np.log(2.0 / (1.0 - np.cos(zeta)))) / (
        1.0 * sp.factorial(mm) * sp.factorial(ll - mm)
    )

    return integrand


@njit
def Fplus00(qq, mm, ll, zeta):

    integrand = 0.0

    for ii in range(0, qq + 1):
        for jj in range(mm, ll + 1):

            integrand += (
                (2.0 ** (ii - jj) * (-1.0) ** (ll + qq - ii + jj))
                * (
                    sp.factorial(qq)
                    * sp.factorial(ll + jj)
                    * (2.0 ** (qq - ii + jj - mm + 1) - (1.0 - np.cos(zeta)) ** (qq - ii + jj - mm + 1))
                )
                / (
                    sp.factorial(ii)
                    * sp.factorial(qq - ii)
                    * sp.factorial(jj)
                    * sp.factorial(ll - jj)
                    * sp.factorial(jj - mm)
                    * (qq - ii + jj - mm + 1)
                )
            )

    return integrand


@njit
def arbORF(mm, ll, zeta):

    if mm == 0:

        if ll >= 0 and ll <= 2:

            delta = [1.0 + np.cos(zeta) / 3.0, -(1.0 + np.cos(zeta)) / 3.0, 2.0 * np.cos(zeta) / 15.0]

            if zeta == 0.0:
                return (
                    NORM
                    * 0.5
                    * np.sqrt((2.0 * ll + 1.0) * np.pi)
                    * (delta[ll] - (1.0 + np.cos(zeta)) * Fminus00(0, 0, ll, zeta))
                )
            else:
                return (
                    NORM
                    * 0.5
                    * np.sqrt((2.0 * ll + 1.0) * np.pi)
                    * (
                        delta[ll]
                        - (1.0 + np.cos(zeta)) * Fminus00(0, 0, ll, zeta)
                        - (1.0 - np.cos(zeta)) * Fplus01(1, 0, ll, zeta)
                    )
                )

        else:
            if zeta == 0.0:
                return NORM * 0.5 * np.sqrt((2.0 * ll + 1.0) * np.pi) * (-(1.0 + np.cos(zeta)) * Fminus00(0, 0, ll, zeta))
            else:
                return (
                    NORM
                    * 0.5
                    * np.sqrt((2.0 * ll + 1.0) * np.pi)
                    * (-(1.0 + np.cos(zeta)) * Fminus00(0, 0, ll, zeta) - (1.0 - np.cos(zeta)) * Fplus01(1, 0, ll, zeta))
                )

    elif mm == 1:

        if ll == 1 or ll == 2:

            delta = [2.0 * np.sin(zeta) / 3.0, -2.0 * np.sin(zeta) / 5.0]

            return (
                NORM
                * 0.25
                * np.sqrt((2.0 * ll + 1.0) * np.pi)
                * np.sqrt((1.0 * sp.factorial(ll - 1)) / (1.0 * sp.factorial(ll + 1)))
                * (
                    delta[ll - 1]
                    - ((1.0 + np.cos(zeta)) ** (3.0 / 2.0) / (1.0 - np.cos(zeta)) ** (1.0 / 2.0)) * Fminus00(1, 1, ll, zeta)
                    - ((1.0 - np.cos(zeta)) ** (3.0 / 2.0) / (1.0 + np.cos(zeta)) ** (1.0 / 2.0)) * Fplus01(2, 1, ll, zeta)
                )
            )

        else:

            return (
                NORM
                * 0.25
                * np.sqrt((2.0 * ll + 1.0) * np.pi)
                * np.sqrt((1.0 * sp.factorial(ll - 1)) / (1.0 * sp.factorial(ll + 1)))
                * (
                    -((1.0 + np.cos(zeta)) ** (3.0 / 2.0) / (1.0 - np.cos(zeta)) ** (1.0 / 2.0)) * Fminus00(1, 1, ll, zeta)
                    - ((1.0 - np.cos(zeta)) ** (3.0 / 2.0) / (1.0 + np.cos(zeta)) ** (1.0 / 2.0)) * Fplus01(2, 1, ll, zeta)
                )
            )

    else:

        return (
            -NORM
            * 0.25
            * np.sqrt((2.0 * ll + 1.0) * np.pi)
            * np.sqrt((1.0 * sp.factorial(ll - mm)) / (1.0 * sp.factorial(ll + mm)))
            * (
                ((1.0 + np.cos(zeta)) ** (mm / 2.0 + 1) / (1.0 - np.cos(zeta)) ** (mm / 2.0)) * Fminus00(mm, mm, ll, zeta)
                - ((1.0 + np.cos(zeta)) ** (mm / 2.0) / (1.0 - np.cos(zeta)) ** (mm / 2.0 - 1.0))
                * Fminus01(mm - 1, mm, ll, zeta)
                + ((1.0 - np.cos(zeta)) ** (mm / 2.0 + 1) / (1.0 + np.cos(zeta)) ** (mm / 2.0))
                * Fplus01(mm + 1, mm, ll, zeta)
                - ((1.0 - np.cos(zeta)) ** (mm / 2.0) / (1.0 + np.cos(zeta)) ** (mm / 2.0 - 1.0)) * Fplus00(mm, mm, ll, zeta)
            )
        )


@njit
def dlmk(l, m, k, theta1):
    """
    returns value of d^l_mk as defined in allen, ottewill 97.
    Called by Dlmk

    """

    if m >= k:

        factor = np.sqrt(sp.factorial(l - k) * sp.factorial(l + m) / sp.factorial(l + k) / sp.factorial(l - m))
        part2 = (np.cos(theta1 / 2)) ** (2 * l + k - m) * (-np.sin(theta1 / 2)) ** (m - k) / sp.factorial(m - k)
        part3 = sp.hyp2f1(m - l, -k - l, m - k + 1, -((np.tan(theta1 / 2)) ** 2))

        return factor * part2 * part3

    else:

        return (-1) ** (m - k) * dlmk(l, k, m, theta1)


@njit
def Dlmk(l, m, k, phi1, phi2, theta1, theta2):
    """
    returns value of D^l_mk as defined in allen, ottewill 97.

    """

    return (
        np.exp(complex(0.0, -m * phi1)) * dlmk(l, m, k, theta1) * np.exp(complex(0.0, -k * gamma(phi1, phi2, theta1, theta2)))
    )


@njit
def gamma(phi1, phi2, theta1, theta2):
    """
    calculate third rotation angle
    inputs are angles from 2 pulsars
    returns the angle.

    """

    if phi1 == phi2 and theta1 == theta2:
        gamma = 0
    else:
        gamma = np.arctan(
            np.sin(theta2) * np.sin(phi2 - phi1) / (np.cos(theta1) * np.sin(theta2) * np.cos(phi1 - phi2) - np.sin(theta1) * np.cos(theta2))
        )

    dummy_arg = (
        np.cos(gamma) * np.cos(theta1) * np.sin(theta2) * np.cos(phi1 - phi2)
        + np.sin(gamma) * np.sin(theta2) * np.sin(phi2 - phi1)
        - np.cos(gamma) * np.sin(theta1) * np.cos(theta2)
    )

    if dummy_arg >= 0:
        return gamma
    else:
        return np.pi + gamma


@njit
def arbCompFrame_ORF(mm, ll, zeta):

    if zeta == 0.0:

        if ll > 2:
            return 0.0
        elif ll == 2:
            if mm == 0:
                # pulsar-term doubling
                return 2 * 0.25 * NORM * (4.0 / 3) * (np.sqrt(np.pi / 5)) * np.cos(zeta)
            else:
                return 0.0
        elif ll == 1:
            if mm == 0:
                # pulsar-term doubling
                return -2 * 0.5 * NORM * (np.sqrt(np.pi / 3.0)) * (1.0 + np.cos(zeta))
            else:
                return 0.0
        elif ll == 0:
            # pulsar-term doubling
            return 2.0 * NORM * 0.25 * np.sqrt(np.pi * 4) * (1 + (np.cos(zeta) / 3.0))

    elif zeta == np.pi:

        if ll > 2:
            return 0.0
        elif ll == 2 and mm != 0:
            return 0.0
        elif ll == 1 and mm != 0:
            return 0.0
        else:
            return arbORF(mm, ll, zeta)

    else:

        return arbORF(mm, ll, zeta)


@njit
def rotated_Gamma_ml(m, l, phi1, phi2, theta1, theta2, gamma_ml):
    """
    This function takes any gamma in the computational frame and rotates it to the
    cosmic frame.

    """

    rotated_gamma = 0

    for ii in range(2 * l + 1):
        rotated_gamma += Dlmk(l, m, ii - l, phi1, phi2, theta1, theta2).conjugate() * gamma_ml[ii]

    return rotated_gamma


@njit
def real_rotated_Gammas(m, l, phi1, phi2, theta1, theta2, gamma_ml):
    """
    This function returns the real-valued form of the Overlap Reduction Functions,
    see Eqs 47 in Mingarelli et al, 2013.

    """

    if m > 0:
        ans = (1.0 / np.sqrt(2)) * (
            rotated_Gamma_ml(m, l, phi1, phi2, theta1, theta2, gamma_ml)
            + (-1) ** m * rotated_Gamma_ml(-m, l, phi1, phi2, theta1, theta2, gamma_ml)
        )
        return ans.real
    if m == 0:
        return rotated_Gamma_ml(0, l, phi1, phi2, theta1, theta2, gamma_ml).real
    if m < 0:
        ans = (1.0 / np.sqrt(2) / complex(0.0, 1)) * (
            rotated_Gamma_ml(-m, l, phi1, phi2, theta1, theta2, gamma_ml)
            - (-1) ** m * rotated_Gamma_ml(m, l, phi1, phi2, theta1, theta2, gamma_ml)
        )
        return ans.real


@njit
def correlated_basis(psr_locs, lmax):

    corr = []

    for ll in range(0, lmax + 1):

        mmodes = 2 * ll + 1  # Number of modes for this ll
        for mm in range(mmodes):
            corr.append(np.zeros((len(psr_locs), len(psr_locs))))

        for aa in range(len(psr_locs)):
            for bb in range(aa, len(psr_locs)):

                plus_gamma_ml = []  # this will hold the list of gammas
                # evaluated at a specific value of phi{1,2}, and theta{1,2}.
                neg_gamma_ml = []
                gamma_ml = []

                # Pre-calculate all the gammas so this gets done only once.
                # Need all the values to execute rotation codes.
                for mm in range(ll + 1):
                    zeta = calczeta(psr_locs[:, 0][aa], psr_locs[:, 0][bb], psr_locs[:, 1][aa], psr_locs[:, 1][bb])

                    intg_gamma = arbCompFrame_ORF(mm, ll, zeta)

                    # just (-1)^m Gamma_ml since this is in the computational frame
                    neg_intg_gamma = (-1) ** (mm) * intg_gamma

                    # all of the gammas from Gamma^-m_l --> Gamma ^m_l
                    plus_gamma_ml.append(intg_gamma)

                    # get the neg m values via complex conjugates
                    neg_gamma_ml.append(neg_intg_gamma)

                neg_gamma_ml = neg_gamma_ml[1:]  # this makes sure we don't have 0 twice
                rev_neg_gamma_ml = neg_gamma_ml[::-1]  # reverse direction of list, now runs from -m...0
                gamma_ml = rev_neg_gamma_ml + plus_gamma_ml

                mindex = len(corr) - mmodes
                for mm in range(mmodes):
                    m = mm - ll

                    corr[mindex + mm][aa, bb] = real_rotated_Gammas(
                        m, ll, psr_locs[:, 0][aa], psr_locs[:, 0][bb], psr_locs[:, 1][aa], psr_locs[:, 1][bb], gamma_ml
                    )

                    if aa != bb:
                        corr[mindex + mm][bb, aa] = corr[mindex + mm][aa, bb]

    return corr
