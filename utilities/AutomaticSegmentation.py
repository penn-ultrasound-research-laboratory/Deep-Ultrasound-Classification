import argparse
import cv2
import logging
import os
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
            lambda y: ((y - WB[it, 2])*(M - WB[it, 3] - y)) / ((M - WB[it, 2] - WB[it, 3]) / 2)**2
        ])

        col_weight_update = np.piecewise(col_ind, [
            (col_ind < WB[it, 0])|(row_ind > N - WB[it, 1]) 
        ], [
            0.0,
            lambda x: ((x - WB[it, 0])*(N - WB[it, 1] - x)) / ((N - WB[it, 0] - WB[it, 1]) / 2)**2
        ])

        weight_dot_image_mat[it] = np.multiply(image, np.multiply(row_weight_update, col_weight_update))
        
    return C[max_iters]


def get_ROI(image):
    blur = __gaussian_filter(image)
    normalized = __linear_normalization(blur)
    enhanced = __enhance_hypoechoic_regions(normalized)
    reference_point = __get_reference_point(enhanced)

    return reference_point, enhanced


def __unit_flat_kernel(p_diff):
    return 1.0 if p_diff <= 1.0 else 0.0


def __get_seed_point(image, rp, nd=12, h=12, eps=2, max_iters=100):
    
    ind = np.indices(image.shape)
    
    pre_cands = np.repeat(rp.reshape(1,2), nd, axis=0)

    ext = np.column_stack((
        np.cos(2*math.pi*np.arange(nd) / nd),
        np.sin(2*math.pi*np.arange(nd) / nd
    )))

    H_neg_sqrt = 1 / h

    v_msk = np.vectorize(__unit_flat_kernel)

    img_dot_ind = np.multiply(image, ind)
    
    pre_cands = pre_cands + h * ext
    post_cands = np.empty(pre_cands.shape)
    max_crit = np.empty(nd)
    
    for d in range(nd):
        p = pre_cands[d].reshape(2,1,1)
        for it in range(max_iters):

            K_h = v_msk(np.linalg.norm(H_neg_sqrt * (ind - p), axis=0))
            nc = np.sum(np.multiply(K_h, image).flatten())
            p_new = np.sum(np.multiply(K_h, img_dot_ind), axis=(1,2)) / nc
            
            if np.linalg.norm(p-p_new < eps):
                p = p_new
                break
            
            p = p_new
        
        max_crit[d] = nc
        post_cands[d] = p
        
    i = np.argmax(max_crit)
    max_p = post_cands[i, :]
    
    return max_p, post_cands


if __name__ == "__main__":

    n = len(os.listdir("TestImages/bank"))
    c = 3
    r = (n // 3) + 1

    fig = plt.figure(figsize=(15, 15))

    for i, f in enumerate(os.listdir("TestImages/bank")):
        
        img = cv2.imread("TestImages/bank/{}".format(f), cv2.IMREAD_GRAYSCALE)
        rp, enh = get_ROI(img)
        max_p, post_cands = __get_seed_point(enh, rp, nd=12, h=20, eps=1)
        
        fig.add_subplot(r, c, i+1)
        
        plt.imshow(img)
        plt.scatter(x=rp[1], y=rp[0], s=40, c='r')
    #     plt.scatter(x=post_cands[:,1], y=post_cands[:,0], s=20, c='g')
        plt.scatter(x=max_p[1], y=max_p[0], s=40, c='b')
        
    plt.show()

    # se = np.ones((16,16),np.uint8)
    # I_ro = cv2.erode(img, kernel = se, iterations = 1)
    # I_ro_c = cv2.bitwise_not(I_ro)

    # I_rc = cv2.bitwise_and(I_ro_c, I_ro_c, mask = cv2.bitwise_not(cv2.dilate(I_ro, kernel = se, iterations = 1)))

    # plt.imshow(I_rc)
    # plt.colorbar()
