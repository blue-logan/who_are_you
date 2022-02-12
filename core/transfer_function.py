"""
Author: Logan Blue
Date: August 14, 2019

This script is intended to represent Rabiner et al's transfer function for
a lossless pipe series for evaluation in Guesswho
"""

import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import time
import pdb

#Pylint flags
#pylint: disable=trailing-whitespace,invalid-name,too-many-instance-attributes
#pylint: disable=too-many-arguments,no-else-return

class TF:
    """Object to calculate the transfer function of a certain lossless pipe series 
    at a give set of reflection coeffs"""

    def __init__(self, r_series=[], N=None, T=None, FS=44100, fft_steps=21, max_freq=22050):
        """Constructor for a transfer function object"""
        if len(r_series) > 0:
            if isinstance(r_series, list):
                self.r_series = np.array(r_series)
            else:
                self.r_series = r_series

        if N:
            self.N = N
        else:
            self.N = len(self.r_series)

        if T:
            self.T = T
        else:
            self.T = 1 / FS

        self.FS = FS
        self.freqs=list(range(0, max_freq, fft_steps))

        #set up object worker pool
        self.worker_pool = multiprocessing.Pool(multiprocessing.cpu_count() - 3)

    @staticmethod
    def transfer(inputs): 
        """Main function to calculate the transfer function for the theoretical lossless
        pipe series for a given omega value"""
        #unpack inputs
        omega, r_series, T, N = inputs

        def ze():
            """This function is a wrapper for my z->omega transition
             - z = exp(2*pi*j*omega*T)"""
            return np.exp(2 * np.pi * omega * T * 1j)

        def product_numerator():
            """Function to calculate the product segment of in the numerator of the
            lossless pipe series transfer function"""
            ans = 1
            for r in r_series:
                ans = ans * (1 + r) * ze_value**(-N/2)
            return ans

        #TODO: VALIDATE THAT THIS FUNCTION DOES MATH CORRECTLY....(only if we need to go 
        #back to cpu computation for some reason)
        def d_function_matrix():
            """Function to calculate the D_N(z) function using the matrix representation. 
            This will hopefully allow use to use the matrix optmizations done with python 
            to accelerate the computation of this step"""
            curr_matrix = [1, -1]
            for r in r_series:
                r_matrix = [[1, -r], [-r*ze_value**-1, ze_value**-1]] 
                curr_matrix = np.matmul(curr_matrix, r_matrix)
            return np.matmul(curr_matrix, [[1], [0]])[0]

        #Main equation of the transfer function
        ze_value = ze()

        top = product_numerator() 
        bot = d_function_matrix()
        return 20 * np.log10(abs(top / bot))

    def run(self, new_r=None):
        """This function will sweep through the complete omega range of a given configuration"""
        if new_r:
            if isinstance(new_r, list):
                self.r_series = np.array(new_r)
            else:
                self.r_series = new_r
            self.N = len(self.r_series)
        #check that we have r_series to run
        ans = []
        if len(self.r_series) > 0:
            tmp = [(self.freqs[i] * 2 * np.pi, self.r_series, self.T, self.N) \
                    for i in range(0, len(self.freqs))]
            ans = self.worker_pool.map(self.transfer, tmp) 

        return self.freqs, np.array(ans)
