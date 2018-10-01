import argparse
import cv2
import logging
import uuid
import math
import numpy as np
import matplotlib.pyplot as plt

from constants.ultrasoundConstants import IMAGE_TYPE
from scipy.stats import skew

def __gaussian_filter(image):
    GAUSSIAN_KERNEL_SIZE_PAPER = 301
    CUTOFF_FREQ_PAPER = 30
    min_dim = np.min(image.shape[:2]) 
    
    if min_dim >= GAUSSIAN_KERNEL_SIZE_PAPER:
        ks = GAUSSIAN_KERNEL_SIZE_PAPER
    else:
        ks = min_dim

    sigma = 1 / (2 * np.pi * CUTOFF_FREQ_PAPER)

    blur = cv2.GaussianBlur(image, (ks, ks), sigma)

    return blur

def __linear_normalization(image):
    lbound05, ubound95 = np.percentile(image, (5, 95))
    print("5th Percentile: {}".format(lbound05))
    print("95th Percentile: {}".format(ubound95))

    return np.piecewise(image, [
        image <= lbound05, 
        image >= ubound95
    ], [
        lambda p: 0, 
        lambda p: 255, 
        lambda p: ((p - lbound05) * 255) / (ubound95 - lbound05)
    ])

def __enhance_hypoechoic_regions(image):
    SN = skew(image.flatten()) 
    z_a = 20
    z_c = np.mean(image)
    
    if SN <= 0:
        z_b = (z_a + z_c) / 2
    else:
        z_b = (z_a + z_c * (1-SN)) / 2

    print("z_a: {} | z_b: {} | z_c: {} | SN: {}".format(z_a, z_b, z_c, SN))

    return np.piecewise(image.astype(float), [
        image <= z_a,
        (image > z_a)&(image <= z_b),
        (image <= z_c)&(image > z_b)
    ], [
        1,
        lambda p: 1.0 - ((p - z_a)**2 / ((z_c - z_a) * (z_b - z_a))),
        lambda p: (p - z_c)**2 / ((z_c - z_a) * (z_c - z_b)),
        0
    ])
    
def __get_reference_point(image, max_iters=100, eps=2):

    plt.imshow(image)
    plt.colorbar()
    plt.show()

    # Watch out that this doesn't blow up if max_iters >>> 
    weight_dot_image_mat = np.empty((max_iters,) + image.shape)
    
    C = np.empty((max_iters, 2))

    # X_L, X_R, Y_T, Y_D
    WB = np.empty((max_iters, 4))
    
    M, N = image.shape
    row_ind, col_ind = np.indices(image.shape)

    weight_dot_image_mat[0] = np.multiply(np.ones(image.shape), image)

    C[0] = np.array([M // 2, N // 2])

    print("Initial Center: ({}, {})".format(C[0, 0], C[0, 1]))

    # THESE MIGHT BE THE WRONG INITIAL VALUES
    WB[0] = np.array([0.0, 0.0, 0.0 ,0.0])

    print("C shape: {}".format(C.shape))
    print("weight_dot_image_mat shape: {}".format(weight_dot_image_mat.shape))
    print("WB shape: {}".format(WB.shape))
    print("M: {}".format(M))
    print("N: {}".format(N))

    for it in range(1, max_iters):
        # compute C_i. Update C[i] <- C_i
        w_multi_prod = np.prod(weight_dot_image_mat[:it], axis=0)
        print("w_multi_prod shape: {}".format(w_multi_prod.shape))

        plt.imshow(w_multi_prod)
        plt.colorbar()
        plt.show()


        # Normalization constant
        nc = np.sum(w_multi_prod.flatten())
        print("nc: {}".format(nc))
        
        # This may be correct
        C_i_c = np.sum(np.sum(np.multiply(col_ind, w_multi_prod).flatten() / nc))
        C_i_r = np.sum(np.sum(np.multiply(row_ind, w_multi_prod).flatten() / nc))

        # THIS IS WRONG!!! 
        # C_i_c = np.sum(np.prod(np.multiply(w_multi_prod, col_ind), axis=0).flatten()  / nc) 
        # C_i_r = np.sum(np.prod(np.multiply(w_multi_prod, row_ind), axis=0).flatten() / nc) 

        print("C: ({}, {})".format(C_i_r, C_i_c))

        C[it] = np.array([C_i_r, C_i_c])

        if np.linalg.norm(C[it] - C[it-1]) < eps:
            print("Center change less than epsilon")
            return C[it]

        ## Update bounds
        # Update columns
        if C[it][1] - C[it-1][1] > 0:
            WB[it, 0] = C[it][1] - C[it-1][1]
            WB[it, 1] = WB[it-1, 1]
        else:
            WB[it, 0] = WB[it-1, 0]
            WB[it, 1] = C[it-1][1] - C[it][1]

        if C[it][0] - C[it-1][0] > 0:
            WB[it, 2] = C[it][0] - C[it-1][0]
            WB[it, 3] =  WB[it-1, 3]
        else: 
            WB[it, 2] = WB[it-1, 2]
            WB[it, 3] = C[it-1][0] - C[it][0]

        # BOUNDS ARE POSSIBLY GOING TO WORK
        print("New bounds: {}".format(WB[it]))

        # I THINK THIS IS WRONG!!!!  
        # Update weighting function
        row_weight_update = np.piecewise(row_ind, [
            (row_ind < WB[it, 2])|(row_ind > M - WB[it, 3]) 
        ], [
            0.0,
            lambda y: ((y - WB[2])*(M - WB[3] - y)) / ((M - WB[2] - WB[3]) / 2)**2
        ])

        col_weight_update = np.piecewise(col_ind, [
            (col_ind < WB[it, 0])|(row_ind > N - WB[it, 1]) 
        ], [
            0.0,
            lambda x: ((x - WB[0])*(N - WB[1] - x)) / ((N - WB[0] - WB[1]) / 2)**2
        ])

        weight_dot_image_mat[it] = np.multiply(image, np.multiply(row_weight_update, col_weight_update))
        
    return C[max_iters]

def get_ROI(image):
    blur = __gaussian_filter(image)
    normalized = __linear_normalization(blur)
    enhanced = __enhance_hypoechoic_regions(normalized)
    __get_reference_point(enhanced)

    return
    # return enhanced


if __name__ == "__main__":

    elephant = cv2.imread("../TestImages/CroppedImage.png", cv2.IMREAD_GRAYSCALE)

    get_ROI(elephant)

    # cv2.imshow("elephant", elephant)
    # cv2.waitKey(0)

    # # cv2.imshow("enhanced", cv2.applyColorMap(enhanced, cv2.COLORMAP_JET))
    # # cv2.waitKey(0)
    # plt.imshow(enhanced)
    # plt.show()