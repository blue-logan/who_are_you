"""
Author: Logan Blue
Date: March 25, 2020

Revamping of the gpu code with new knownledge. 

These function are the gpu optimized versions of our transfer function/gradient
calculation step for guesswho in an attempt to not have to lock up the servers
for 30+ days.

NOTE: Currently this is a skeleton of the logic for the process. There are
still a lot of GPU level optimization that needs to be added to this code
before it will be truely performant.
"""

from numba import cuda, float64, complex128, int64
from math import log10, ceil
from cmath import sqrt, exp

R_STEP = 0.01
omega_len = 320
max_r_layers = 6
gamma = 1e-6

@cuda.jit('UniTuple(c16, 2)(f8, i8, i8)', device=True)
def z_value_calc(omega, N, FS):
    """Function to calculate the z^(-N/2) for a given omega"""
    #to work around pycuda's inability to take complex numbers to powers,
    #we need to move into polar coordinates and then keep track of real and
    z_val = exp(2 * 3.14159 * 1/FS * 1j * omega)
    #calc the numerator product for the high side
    power = N / 2

    #calc z^-N/2 once
    if power % 1 != 0:
        power = N // 2
        z_power = 1
        for x in range(0, power // 2):
            if x == 0:
                z_power = z_val ** 2
            else:
                z_power = z_power * z_val ** 2
        if power % 2 != 0:
            z_power = z_power * z_val
        z_power = z_power * sqrt(z_val)
    else:
        power = N // 2
        z_power = 1
        for x in range(0, power // 2):
            if x == 0:
                z_power = z_val ** 2
            else:
                z_power = z_power * z_val ** 2
        if power % 2 != 0:
            z_power = z_power * z_val
    return 1.0 / z_power, 1.0 / z_val

#@cuda.jit(device=True)
@cuda.jit('f8(f8[:,:], i8, i8, f8[:], c16, c16, f8[:], i8, b1)', device=True)
def calc_transfer(area, r_start, r_parallel, r_series, z_power, z_val, targets, w_id, upper):
    """Function to calculate the transfer fucntion"""
    number_final = 1
    for r_id in range(r_start, r_start + r_parallel):
        number_final = 1
        if r_id < r_series.size:            #check edge case
            for i in range(0, len(r_series)):
                if i == r_id:
                    if upper:
                        number_final = number_final * (1 + r_series[i] + R_STEP) * z_power
                    else:
                        number_final = number_final * (1 + r_series[i] - R_STEP) * z_power
                else:
                    number_final = number_final * (1 + r_series[i]) * z_power

            denom_final_0 = complex128(1)
            denom_final_1 = complex128(-1)
    
            #preform the square matrix multiplication for the denominator
            for i in range(0, len(r_series)):
                if i == r_id:
                    #this is the index of the r value that we are adjusting on
                    if upper:
                        r_loc = r_series[i] + R_STEP
                    else:
                        r_loc = r_series[i] - R_STEP
                else:
                    r_loc = r_series[i]
    
                #multiple denom_final with the newly constructed r_i matrix
                denom_tmp = denom_final_0  + denom_final_1 * -r_loc * z_val
                denom_final_1 = denom_final_0 * -r_loc + denom_final_1 * z_val
                denom_final_0 = denom_tmp
    
            #area[r_id][w_id] = abs(float64(targets[w_id]) - 20 * log10(abs(number_final / denom_final_0)))
            area_tmp = abs(float64(targets[w_id]) - 20 * log10(abs(number_final /denom_final_0)))
            area[r_id-r_start][w_id] = area_tmp

    return 20 * log10(abs(number_final / denom_final_0))

@cuda.jit(device=True)
def sum_simple(values):
    summed = 0
    for v in values:
        summed += v
    return summed

@cuda.jit('void(f8[:], f8[:,:], f8[:, :], f8[:, :], i8, i8, i8)', device=True)
def calc_gradients(grad, sub_matrix, high, low, r_start, r_parallel, r_size):
    """This function will subtract the 2-d area arrays from one another, sum the rows of this
    subtraction, and then calculate the per r gradient for the pass. 
    -   high: 2D array of area's found from calculating the TF area from truth based off increasing
            an r value.
    -   low: 2D array of area's found from calculating the TF area from truth based off decreasing
            an r value.
    -   sub_matrix: Pre-set up shared array for the results of the subtraction
    -   r_start: starting r for this block to calculated
    -   r_parallel: number of r values this block has to calculated
    """
    omega_id = cuda.threadIdx.x
    block = cuda.blockIdx.x
    #threadwise subtraction
    for r_id in range(0, r_parallel):
        sub_matrix[r_start + r_id][omega_id] = high[r_id][omega_id] - low[r_id][omega_id]

    #summing each row into a single value
    #threads 0, 1, 2, 3, ..., r_parallel - 1 will be used to sum r_start through r_start + r_parallel
    if block == 0 and omega_id < r_size:
        #grad[r_start + omega_id] = sum_reduce(sub_matrix[r_start + omega_id]) / (2 * R_STEP)
        grad[omega_id] = sum_simple(sub_matrix[omega_id]) / (2 * R_STEP)
    

#@cuda.jit('void(f8[:], f8[:], f8[:], f8[:], f8[:, :], i8, i8, f8[:])', nopython=True)
#removing nopython=True, gives error on linux
@cuda.jit('void(f8[:], f8[:], f8[:], f8[:], f8[:, :], i8, i8, f8[:])')
def grad_calc(r_series, omega_series, targets, fft_curve, sub_matrix, FS, MAX_ITER, gradients):
    """Main kernel for performing the gradient descent process on the gpu
        - r_series: r_coeff for the initial guess
        - omega_series: omega frequencies over which to evaluate the fit
        - targets: target amplitudes of each omega in the omega series (ground truth)
        - fft_curve: descents approximation of the targets according to the TF
        - MAX_ITER: maximum number of gradient descent steps we can talk before exiting
    """
    #set up shared memory fro the high and low calculations
    high_area = cuda.shared.array(shape=(max_r_layers, omega_len), dtype=float64)
    low_area = cuda.shared.array(shape=(max_r_layers, omega_len), dtype=float64)
    #gradients = cuda.shared.array(shape=50, dtype=float64)
    
    #get thread and block information
    w_id = cuda.threadIdx.x
    b_id = cuda.blockIdx.x
    #BLOCKS_PER_FRAME = cuda.gridDim
    BLOCKS_PER_FRAME = 8

    #Each thread will be responsible for ceil(r_series / 8) r's except for the last block
    r_parallel = int64(ceil(r_series.size / BLOCKS_PER_FRAME))
    r_start = int64(b_id * r_parallel)
    
    #calculate z value for each thread
    z_neg_n_2, z_neg = z_value_calc(omega_series[w_id], r_series.size, FS)

    #gradient descent looping section
    index = 0
    for index in range(0, MAX_ITER):
        #calculate Transfer Function output
        fft_curve[w_id] = calc_transfer(high_area, r_start, r_parallel, r_series, z_neg_n_2, \
                z_neg, targets, w_id, False)
        _ = calc_transfer(low_area, r_start, r_parallel, r_series, z_neg_n_2, z_neg, targets, \
                w_id, True)
        
        #find difference between high and low area
        cuda.syncthreads()
        calc_gradients(gradients, sub_matrix, high_area, low_area, r_start, r_parallel, r_series.size)
        
        #gradient found, step for next iteration
        if b_id == 0 and  w_id < gradients.size:
            new_val = r_series[w_id] + gamma * gradients[w_id]
            
            #enforce boundary conditions of r values
            if abs(new_val) <= 1.0:
                r_series[w_id] = new_val
            else:
                if new_val < 0:
                    r_series[w_id] = -0.99
                else:
                    r_series[w_id] = 0.99

        #set areas to zeros to see if that fixes issue
        #if b_id < max_r_layers:
        #    high_area[b_id][w_id] = 0.0
        #    low_area[b_id][w_id] = 0.0
        ##set sub_matrix to all zeros
        #for r_id in range(0, r_parallel):
        #    sub_matrix[r_start + r_id][w_id] = 0

        #make sure that all threads have adjusted the r_series before beginning next time through loop
        cuda.syncthreads()

