import argparse
import cv2
import logging
import os
import uuid
import math
import numpy as np
import scipy.stats as scp
import matplotlib.pyplot as plt

from src.constants.ultrasound import IMAGE_TYPE
from src.constants.automatic import (
    FIND_SEED_POINT_STOPPING_CRITERION,
    FIND_SEED_POINT_MAXIMUM_ITERATIONS,
    FIND_SEED_POINT_NUMBER_DIRECTIONS,
    FIND_SEED_POINT_RADIUS,
    GAUSSIAN_KERNEL_SIZE_PAPER,
    GAUSSIAN_CUTOFF_FREQ_PAPER,
    HYPOECHOIC_LOW_BOUNDARY
)

COL_START = 0
ROW_START = 1
WIDTH = 2
HEIGHT = 3

def __test_seed_point_in_rectangle(seed_pt, rectangle):
    x, y, w, h = rectangle
    return (
        seed_pt[1] > x and 
        seed_pt[1] < x + w and
        seed_pt[0] > y and 
        seed_pt[0] < y + h
    )

def __rectangle_area(rectangle):
    return rectangle[WIDTH] * rectangle[HEIGHT]

def __gaussian_filter(image):
    min_dim = np.min(image.shape[:2]) 
    
    # Xian et. al uses a large Gaussian kernel to blur the entire image. 
    # Their constant kernel size is probably related to their fixed input size.
    # I assume the minimum dimension of a variable sized input will be fine here. 

    kernel_size = min_dim if min_dim % 2 == 1 else min_dim - 1

    sigma = 1 / (2 * np.pi * GAUSSIAN_CUTOFF_FREQ_PAPER)

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

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
    skew = scp.skew(image.flatten()) 
    
    # Determine the HYPOECHOIC_HIGH_BOUNDARY
    z_c = np.mean(image)
    
    # Determine the HYPOECHOIC_MIDDLE_BOUNDARY
    if skew <= 0:
        z_b = (HYPOECHOIC_LOW_BOUNDARY + z_c) / 2
    else:
        z_b = (HYPOECHOIC_LOW_BOUNDARY + z_c * (1-skew)) / 2

    return np.piecewise(image.astype(float), [
        image <= HYPOECHOIC_LOW_BOUNDARY,
        (image > HYPOECHOIC_LOW_BOUNDARY)&(image <= z_b),
        (image <= z_c)&(image > z_b)
    ], [
        1,
        lambda p: 1.0 - ((p - HYPOECHOIC_LOW_BOUNDARY)**2 / ((z_c - HYPOECHOIC_LOW_BOUNDARY) * (z_b - HYPOECHOIC_LOW_BOUNDARY))),
        lambda p: (p - z_c)**2 / ((z_c - HYPOECHOIC_LOW_BOUNDARY) * (z_c - z_b)),
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

def __get_surrounding_circular_points(center_point, number_directions, radius):

    center_repeated = np.repeat(center_point.reshape(1,2), number_directions, axis=0)

    directed_extension = np.column_stack((
        np.cos(2*math.pi*np.arange(number_directions) / number_directions),
        np.sin(2*math.pi*np.arange(number_directions) / number_directions
    )))

    return center_repeated + radius * directed_extension

def __get_seed_point(
    image, 
    reference_point,
    number_directions=FIND_SEED_POINT_NUMBER_DIRECTIONS,
    radius=FIND_SEED_POINT_RADIUS,
    stopping_criterion=FIND_SEED_POINT_STOPPING_CRITERION,
    maximum_iterations=FIND_SEED_POINT_MAXIMUM_ITERATIONS):
    
    image_indices = np.indices(image.shape)

    H_neg_sqrt = 1 / radius

    flat_kernel_mask = np.vectorize(__unit_flat_kernel)

    # Dot product of the image with pixel indices
    image_dot_indices = np.multiply(image, image_indices)

    # Initial candidates for ROI search are circle surrounding reference reference point
    pre_search_candidates = __get_surrounding_circular_points(
        reference_point, 
        number_directions, 
        radius)

    post_search_candidates = np.empty(pre_search_candidates.shape)
    max_crit = np.empty(number_directions)
    
    for direction in range(number_directions):
        p = pre_search_candidates[direction].reshape(2, 1, 1)
        for it in range(maximum_iterations):

            K_h = flat_kernel_mask(np.linalg.norm(H_neg_sqrt * (image_indices - p), axis=0))
            nc = np.sum(np.multiply(K_h, image).flatten())
            p_new = np.sum(np.multiply(K_h, image_dot_indices), axis=(1,2)) / nc
            
            if np.linalg.norm(p-p_new < stopping_criterion):
                p = p_new
                break
            
            p = p_new
        
        max_crit[direction] = nc
        post_search_candidates[direction] = p
    
    # Point that maximizes the search criteria. Return as seed point
    seed_point = post_search_candidates[np.argmax(max_crit), :]
    
    return seed_point, post_search_candidates

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
    br = [r for r in br if __test_seed_point_in_rectangle(seed_pt, r)]

    # Destructure the minimum bounding rectangle of the minimum area contour containing seed point           
    x, y, w, h = min(br, key = __rectangle_area)
    
    M, N = image.shape
    x_exp = N // 20
    y_exp = M // 20

    x_start = max(x - x_exp, 0)
    x_end = min(x + w + x_exp, N)
    y_start = max(y - y_exp, 0)
    y_end = min(y + h + y_exp, M)

    # Return in standard (x, y, w, h) rectangle format
    return (
        x_start,
        y_start,
        x_end - x_start,
        y_end - y_start
    )

def get_ROI_debug(image):
    blur = __gaussian_filter(image)
    normalized = __linear_normalization(blur)
    enhanced = __enhance_hypoechoic_regions(normalized)
    ref_pt = __get_reference_point(enhanced)
    seed_pt, post_search_candidates = __get_seed_point(enhanced, ref_pt)
    roi_rect = __determine_roi(enhanced, seed_pt)

    return roi_rect, seed_pt


def get_ROI(image):
    blur = __gaussian_filter(image)
    normalized = __linear_normalization(blur)
    enhanced = __enhance_hypoechoic_regions(normalized)
    ref_pt = __get_reference_point(enhanced)
    seed_pt, post_search_candidates = __get_seed_point(enhanced, ref_pt)
    x, y, w, h = __determine_roi(enhanced, seed_pt)

    return image[y:y+h, x:x+w]

