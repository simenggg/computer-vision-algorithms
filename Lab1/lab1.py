""" CS4243 Lab 1: Template Matching
"""

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import math

##### Part 1: Image Preprossessing #####

def rgb2gray(img):
    """
    5 points
    Convert a colour image greyscale
    Use (R,G,B)=(0.299, 0.587, 0.114) as the weights for red, green and blue channels respectively
    :param img: numpy.ndarray (dtype: np.uint8)
    :return img_gray: numpy.ndarray (dtype:np.uint8)
    """
    if len(img.shape) != 3:
        print('RGB Image should have 3 channels')
        return
    
    """ Your code starts here """
    r_chennel = img[:, :, 0]
    g_channel = img[:, :, 1]
    b_channel = img[:, :, 2]

    height = img.shape[0]
    width = img.shape[1]

    img_gray = np.zeros((height, width))

    for i in range (0, height):
        for j in range (0, width):
            img_gray[i][j] = 0.229 * r_chennel[i][j] + 0.587 * g_channel[i][j] + 0.114 * b_channel[i][j]

    """ Your code ends here """
    return img_gray


def gray2grad(img):
    """
    5 points
    Estimate the gradient map from the grayscale images by convolving with Sobel filters (horizontal and vertical gradients) and Sobel-like filters (gradients oriented at 45 and 135 degrees)
    The coefficients of Sobel filters are provided in then code below.
    :param img: numpy.ndarray
    :return img_grad_h: horizontal gradient map. numpy.ndarray
    :return img_grad_v: vertical gradient map. numpy.ndarray
    :return img_grad_d1: diagonal gradient map 1. numpy.ndarray
    :return img_grad_d2: diagonal gradient map 2. numpy.ndarray
    """
    sobelh = np.array([[-1, 0, 1], 
                       [-2, 0, 2], 
                       [-1, 0, 1]], dtype = float)
    sobelv = np.array([[-1, -2, -1], 
                       [0, 0, 0], 
                       [1, 2, 1]], dtype = float)
    sobeld1 = np.array([[-2, -1, 0],
                        [-1, 0, 1],
                        [0,  1, 2]], dtype = float)
    sobeld2 = np.array([[0, -1, -2],
                        [1, 0, -1],
                        [2, 1, 0]], dtype = float)
    

    """ Your code starts here """
    height, width = img.shape

    img_grad_h = np.zeros((height, width), dtype=np.float64)
    img_grad_v = np.zeros((height, width), dtype=np.float64)
    img_grad_d1 = np.zeros((height, width), dtype=np.float64)
    img_grad_d2 = np.zeros((height, width), dtype=np.float64)

    """need padding first """
    pad_zeros_img = pad_zeros(img, 1, 1, 1, 1)

    sobelh_flipped = np.flip(sobelh, axis=(0,1))
    sobelv_flipped = np.flip(sobelv, axis=(0,1))
    sobeld1_flipped = np.flip(sobeld1, axis=(0,1))
    sobeld2_flipped = np.flip(sobeld2, axis=(0,1))

    for i in range(0, height):
        for j in range(0, width):
            region = pad_zeros_img[i:i+3, j:j+3]
            img_grad_h[i][j] = np.sum(region * sobelh_flipped)
            img_grad_v[i][j] = np.sum(region * sobelv_flipped)
            img_grad_d1[i][j] = np.sum(region * sobeld1_flipped)
            img_grad_d2[i][j] = np.sum(region * sobeld2_flipped)

    """ Your code ends here """
    return img_grad_h, img_grad_v, img_grad_d1, img_grad_d2

def pad_zeros(img, pad_height_bef, pad_height_aft, pad_width_bef, pad_width_aft):
    """
    5 points
    Add a border of zeros around the input images so that the output size will match the input size after a convolution or cross-correlation operation.
    e.g., given matrix [[1]] with pad_height_bef=1, pad_height_aft=2, pad_width_bef=3 and pad_width_aft=4, obtains:
    [[0 0 0 0 0 0 0 0]
    [0 0 0 1 0 0 0 0]
    [0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0]]
    :param img: numpy.ndarray
    :param pad_height_bef: int
    :param pad_height_aft: int
    :param pad_width_bef: int
    :param pad_width_aft: int
    :return img_pad: numpy.ndarray. dtype is the same as the input img. 
    """
    
    height, width = img.shape[:2] 
    new_height, new_width = (height + pad_height_bef + pad_height_aft), (width + pad_width_bef + pad_width_aft) 
    img_pad = np.zeros((new_height, new_width), dtype=img.dtype) if len(img.shape) == 2 else np.zeros((new_height, new_width, img.shape[2]), dtype=img.dtype)
    #add the data type here from the original code

    """ Your code starts here """
    img_pad[pad_height_bef:pad_height_bef + height, pad_width_bef:pad_width_bef + width] = img
    
    #for i in range (0, new_height): 
    #        for j in range (0, new_width):
    #            img_pad[i][j] = img[i - pad_height_bef][j - pad_width_bef]  

    """ Your code ends here """
    return img_pad




##### Part 2: Normalized Cross Correlation #####
def normalized_cross_correlation(img, template):
    """
    10 points.
    Implement the cross-correlation operation in a naive 4, 5 or 6 nested for-loops. 
    The loops should at least include the height and width of the output and height and width of the template.
    When it is 5 or 6 loops, the channel of the output and template may be included.
    :param img: numpy.ndarray.
    :param template: numpy.ndarray.
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    """ Your code starts here """
    if img.ndim == 2:         
        img = img[:, :, None] # (H, W, 1)

    if template.ndim == 2:
        template = template[:, :, None]

    img = img.astype(np.float64)
    template = template.astype(np.float64)
    response = np.zeros((Ho, Wo), dtype=np.float64)
    C = img.shape[2]

    for i in range(Ho):
        for j in range(Wo):
            P_sum2 = 0.0
            T_sum2 = 0.0
            P_dot_T = 0.0

            for c in range(C):
                for u in range(Hk):
                    for v in range(Wk):
                        P = img[i + u, j + v, c]
                        T = template[u, v, c]
                        
                        P_sum2 += P * P
                        T_sum2 += T * T
                        P_dot_T += P * T

                # OpenCV-style NCC
                # numerator = P_dot_T - (P_sum * T_sum) / N
                # denom_P = P_sum2 - (P_sum ** 2) / N
                # denom_T = T_sum2 - (T_sum ** 2) / N
                # not zero mean version!!!

            denominator = np.sqrt(P_sum2 * T_sum2)

            if denominator < 1e-12:
                response[i, j] = 0
            else:
                response[i, j] = P_dot_T / denominator
 
    """ Your code ends here """
    return response


def normalized_cross_correlation_fast(img, template):
    """
    10 points.
    Implement the cross correlation with 3 nested for-loops. 
    The for-loop over the template is replaced with the element-wise multiplication between the kernel and the image regions.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    """ Your code starts here """ 
    if img.ndim == 2:         
        img = img[:, :, None] # (H, W, 1)

    if template.ndim == 2:
        template = template[:, :, None]

    img = img.astype(np.float64)
    template = template.astype(np.float64)
    response = np.zeros((Ho, Wo), dtype=np.float64)
    C = img.shape[2]

    for i in range(Ho):
        for j in range(Wo):
            P_sum2 = 0.0
            T_sum2 = 0.0
            P_dot_T = 0.0

            for c in range(C):
                region = img[i:i+Hk, j:j+Wk, c]
                template_c = template[:, :, c]

                P_sum2 += np.sum(region ** 2)
                T_sum2 += np.sum(template_c ** 2)
                P_dot_T += np.sum(region * template_c)
  
            denominator = np.sqrt(P_sum2 * T_sum2)

            if denominator < 1e-12:
                response[i, j] = 0
            else:
                response[i, j] = P_dot_T / denominator
                   
    """ Your code ends here """
    return response




def normalized_cross_correlation_matrix(img, template):
    """
    10 points.
    Converts cross-correlation into a matrix multiplication operation to leverage optimized matrix operations.
    Please check the detailed instructions in the pdf file.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    """ Your code starts here """
    if img.ndim == 2:         
        img = img[:, :, None] # (H, W, 1)

    if template.ndim == 2:
        template = template[:, :, None]

    img = img.astype(np.float64)
    template = template.astype(np.float64)
    response = np.zeros((Ho, Wo), dtype=np.float64)
    C = img.shape[2]

    #reshapte template
    template_r = template[:, :, 0].reshape(-1, 1)
    template_g = template[:, :, 1].reshape(-1, 1)
    template_b = template[:, :, 2].reshape(-1, 1)
    template_reshapted = np.vstack((template_r, template_g, template_b)) if C == 3 else template_r

    #reshapte image
   
    img_reshaped_channels = []

    for c in range(C):
        img_reshaped_c = np.zeros((Ho * Wo, Hk * Wk), dtype=np.float64)
        for i in range (Ho):
            for j in range (Wo):
                patch = img[i:i+Hk, j:j+Wk, c]
                img_reshaped_c[i * Wo + j, :] = patch.reshape(1, -1) 
        img_reshaped_channels.append(img_reshaped_c)
                
        img_reshaped = np.hstack(img_reshaped_channels)

    template_norm = np.sqrt(np.sum(template_reshapted ** 2))
    patch_norms = np.linalg.norm(img_reshaped, axis=1, keepdims=True)
   
    response = np.dot(img_reshaped, template_reshapted) / (patch_norms * template_norm)
    response = response.reshape(Ho, Wo)

    """ Your code ends here """
    return response


##### Part 3: Non-maximum Suppression #####

def non_max_suppression(response, suppress_range, threshold=None):
    """
    10 points
    Implement the non-maximum suppression for translation symmetry detection
    The general approach for non-maximum suppression is as follows:
	1. Set a threshold τ; values in X<τ will not be considered.  Set X<τ to 0.  
    2. While there are non-zero values in X
        a. Find the global maximum in X and record the coordinates as a local maximum.
        b. Set a small window of size w×w points centered on the found maximum to 0.
	3. Return all recorded coordinates as the local maximum.
    :param response: numpy.ndarray, output from the normalized cross correlation
    :param suppress_range: a tuple of two ints (H_range, W_range). 
                           the points around the local maximum point within this range are set as 0. In this case, there are 2*H_range*2*W_range points including the local maxima are set to 0
    :param threshold: int, points with value less than the threshold are set to 0
    :return res: a sparse response map which has the same shape as response
    """
    
    """ Your code starts here """
    H_response, W_response = response.shape
    res = np.zeros((H_response, W_response), dtype=np.float64)

    for i in range(H_response):
        for j in range(W_response):
            if response[i, j] < threshold:
                res[i, j] = 0
    
    while True:
        max_val = np.max(response)
        if max_val == 0:
            break

    #while np.any(response):
        local_max = 0
        max_i = -1
        max_j = -1
        for i in range(H_response):
            for j in range(W_response):
                if response[i, j] > local_max:
                    local_max = response[i, j]
                    max_i = i
                    max_j = j
        res[max_i, max_j] = 255
        for m in range(max_i - suppress_range[0], max_i + suppress_range[0] + 1):
            for n in range(max_j - suppress_range[1], max_j + suppress_range[1] + 1):
                if m >= 0 and m < H_response and n >= 0 and n < W_response:
                    response[m, n] = 0
    
    #maybe not correct because there should only be one max within each grid (keep the brightest one)
    """ Your code ends here """
    return res

##### Part 4: Question And Answer #####
    
def normalized_cross_correlation_ms(img, template):
    """
    10 points
    Please implement mean-subtracted cross correlation which corresponds to OpenCV TM_CCOEFF_NORMED.
    For simplicty, use the "fast" version.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    """ Your code starts here """
    if img.ndim == 2:         
        img = img[:, :, None] # (H, W, 1)

    if template.ndim == 2:
        template = template[:, :, None]

    img = img.astype(np.float64)
    template = template.astype(np.float64)
    response = np.zeros((Ho, Wo), dtype=np.float64)
    C = img.shape[2]
   
    for i in range(Ho):
        for j in range(Wo):
            numerator = 0.0
            denom_P = 0.0
            denom_T = 0.0

            for c in range(C):
                template_c = template[:, :, c]  
                region = img[i:i+Hk, j:j+Wk, c]

                P_mean = np.mean(region)
                P_zero_mean = region - P_mean

                T_mean = np.mean(template_c)
                T_zero_mean = template_c - T_mean
                
                numerator += np.sum(P_zero_mean * T_zero_mean)
                denom_P += np.sum(P_zero_mean ** 2)
                denom_T += np.sum(T_zero_mean ** 2)

            
            denominator = np.sqrt(denom_P * denom_T)

            if denominator < 1e-12:
                response[i, j] = 0
            else:
                response[i, j] = numerator / denominator

    """ Your code ends here """
    return response




"""Helper functions: You should not have to touch the following functions.
"""
def read_img(filename):
    '''
    Read HxWxC image from the given filename
    :return img: numpy.ndarray, size (H, W, C) for RGB. The value is between [0, 255].
    '''
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def show_imgs(imgs, titles=None):
    '''
    Display a list of images in the notebook cell.
    :param imgs: a list of images or a single image
    '''
    if isinstance(imgs, list) and len(imgs) != 1:
        n = len(imgs)
        fig, axs = plt.subplots(1, n, figsize=(15,15))
        for i in range(n):
            axs[i].imshow(imgs[i], cmap='gray' if len(imgs[i].shape) == 2 else None)
            if titles is not None:
                axs[i].set_title(titles[i])
    else:
        img = imgs[0] if (isinstance(imgs, list) and len(imgs) == 1) else imgs
        plt.figure()
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)

def show_img_with_points(response, img_ori=None):
    '''
    Draw small red rectangles of size defined by rec_shape around the non-zero points in the image.
    Display the rectangles and the image with rectangles in the notebook cell.
    :param response: numpy.ndarray. The input response should be a very sparse image with most of points as 0.
                     The response map is from the non-maximum suppression.
    :param img_ori: numpy.ndarray. The original image where response is computed from
    :param rec_shape: a tuple of 2 ints. The size of the red rectangles.
    '''
    response = response.copy()
    if img_ori is not None:
        img_ori = img_ori.copy()

    xs, ys = response.nonzero()
    for x, y in zip(xs, ys):
        response = cv2.circle(response, (y, x), radius=0, color=(255, 0, 0), thickness=5)
        if img_ori is not None:
            img_ori = cv2.circle(img_ori, (y, x), radius=0, color=(255, 0, 0), thickness=5)
        
    if img_ori is not None:
        show_imgs([response, img_ori])
    else:
        show_imgs(response)


