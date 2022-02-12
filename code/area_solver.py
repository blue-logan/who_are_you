"""
Author: Jessica O'Dell
Editor: Logan Blue
Date: 09/13/2019, 2/15/2020
This function does the translation of area curve to reflection 
coefficient series and vice versa. 
"""

#pylint: disable=bad-whitespace,invalid-name
#pylint: disable=trailing-whitespace


def areaSolver(r_N, A_0):
    """
    This function converts the series of reflection coefficients into the 
    estimated cross sectional area for a given tube series based on a given 
    starting cross sectional area. 
    """
    pos = len(r_N)-1
    A_list = []
    for pos in range(0, len(r_N)):
        r_k = r_N[pos]
        next_A = (A_0 * (r_k + 1)) / (1 - r_k)
        A_list.append(next_A)
        A_0 = next_A
    return A_list

def reflectionSolver(a_n):
    """
    This fucntion converts a series of cross sectional area into the corresponding
    reflection coefficient series. 
    """
    r_series = []
    for i in range(0, len(a_n) - 1):
        r_series.append((a_n[i+1] - a_n[i]) / (a_n[i+1] + a_n[i]))

    return r_series
