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

def __seed_point_in_rectangle(seed_pt, rect):
    x,y,w,h = rect
    return (
        seed_pt[1] > x and 
        seed_pt[1] < x + w and
        seed_pt[0] > y and 
        seed_pt[0] < y + h
    )

def __rectangle_area(rect):
    return rect[2] * rect[3]

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

    # Watch out that this doesn't blow up if max_iters >>> 
    weight_dot_image_mat = np.empty((max_iters,) + image.shape)
    
    C = np.empty((max_iters, 2))

    # X_L, X_R, Y_T, Y_D
    WB = np.empty((max_iters, 4))
    
    M, N = image.shape
    ind = np.indices(image.shape)
    row_ind = ind[0]
    col_ind = ind[1]
        
    weight_dot_image_mat[0] = np.multiply(np.ones(image.shape), image)

    C[0] = np.array([M // 2, N // 2])

    # THESE MIGHT BE THE WRONG INITIAL VALUES
    WB[0] = np.array([0.0, 0.0, 0.0 ,0.0])

    for it in range(1, max_iters):
        # compute C_i. Update C[i] <- C_i
        w_multi_prod = np.prod(weight_dot_image_mat[:it], axis=0)

        # Normalization constant
        nc = np.sum(w_multi_prod.flatten())
        
        # This may be correct
        C_i_c = np.sum(np.sum(np.multiply(col_ind, w_multi_prod).flatten() / nc))
        C_i_r = np.sum(np.sum(np.multiply(row_ind, w_multi_prod).flatten() / nc))

        C[it] = np.array([C_i_r, C_i_c])

        if np.linalg.norm(C[it] - C[it-1]) < eps:
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

        # Update weighting function
        row_weight_update = np.piecewise(row_ind.astype(float), [
            (row_ind > M - WB[it, 3])|(row_ind < WB[it, 2]) 
        ], [
            0.0,
            lambda y: ((y - WB[it, 2])*(M - WB[it, 3] - y)) / (((M - WB[it, 3] - WB[it, 2]) / 2)**2)
        ])

        col_weight_update = np.piecewise(col_ind.astype(float), [
            (col_ind > N - WB[it, 1])|(col_ind < WB[it, 0]) 
        ], [
            0.0,
            lambda x: ((x - WB[it, 0])*(N - WB[it, 1] - x)) / (((N - WB[it, 1] - WB[it, 0]) / 2)**2)
        ])

        weight_dot_image_mat[it] = np.multiply(image, np.multiply(row_weight_update, col_weight_update))

    return C[max_iters]

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
        
    max_p = post_cands[np.argmax(max_crit), :]
    
    return max_p, post_cands

def __determine_roi(image, seed_pt, ks=(2,2)):

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ks)

    img_renorm = (image * 255).astype(np.uint8)

    I_rc = cv2.morphologyEx(
        img_renorm, 
        cv2.MORPH_CLOSE, 
        kernel)
       
    otsu_thresh, img_morph = cv2.threshold(I_rc, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    im2, contours, hierarchy = cv2.findContours(
        img_morph,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)    

    # Find all bounding rectangles of contours that contain the seed point
    br = [cv2.boundingRect(c) for c in contours]
    br = [r for r in br if __seed_point_in_rectangle(seed_pt, r)]

    # Destructure the minimum bounding rectangle of the minimum area contour containing seed point           
    x, y, w, h = min(br, key = __rectangle_area)
    
    M, N = image.shape
    x_exp = N // 20
    y_exp = M // 20

    return (
        max(x - x_exp, 0),
        max(y - y_exp, 0),
        min(x + w + x_exp, N),
        min(y + h + y_exp, M)
    )


def get_ROI(image):
    blur = __gaussian_filter(image)
    normalized = __linear_normalization(blur)
    enhanced = __enhance_hypoechoic_regions(normalized)
    ref_pt = __get_reference_point(enhanced)
    seed_pt, post_cands = __get_seed_point(enhanced, ref_pt)
    x, y, w, h = __determine_roi(enhanced, seed_pt)
    return image[y:y+h, x:x+w]


if __name__ == "__main__":

    for i, f in enumerate(os.listdir("../TestImages/bank")):
        
        img = cv2.imread("../TestImages/bank/{}".format(f), cv2.IMREAD_GRAYSCALE)
        roi = get_ROI(img)
               
        cv2.imshow("ROI", roi)
        cv2.waitKey(0)
