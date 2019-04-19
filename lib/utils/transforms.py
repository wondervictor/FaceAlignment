# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Create by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import cv2
import torch
import scipy
import numpy as np


# TODO: remove
def fliplr(x):
    if x.ndim == 3:
        x = np.transpose(np.fliplr(np.transpose(x, (0, 2, 1))), (0, 2, 1))
    elif x.ndim == 4:
        for i in range(x.shape[0]):
            x[i] = np.transpose(np.fliplr(np.transpose(x[i], (0, 2, 1))), (0, 2, 1))
    return x.astype(float)


def flip_back(flip_output, dataset='mpii'):
    """
    flip output map
    """
    if dataset == 'mpii':
        matched_parts = (
            [0, ],   [1, 4],   [2, 3],
            [10, 15], [11, 14], [12, 13]
        )
    else:
        print('Not supported dataset: ' + dataset)

    # flip output horizontally
    flip_output = fliplr(flip_output.numpy())

    # Change left-right parts
    for pair in matched_parts:
        tmp = np.copy(flip_output[:, pair[0], :, :])
        flip_output[:, pair[0], :, :] = flip_output[:, pair[1], :, :]
        flip_output[:, pair[1], :, :] = tmp

    return torch.from_numpy(flip_output).float()


def shufflelr(x, width, dataset='mpii'):
    """
    flip coords
    """
    if dataset == 'aflw':
        matched_parts = (
            [1, 6],  [2, 5], [3, 4],
            [7, 12], [8, 11], [9, 10],
            [13, 15],
            [16, 18]
        )
    elif dataset == 'wflw':
        matched_parts = (
            [0, 32],  [1,  31], [2,  30], [3,  29], [4,  28], [5, 27], [6, 26], [7, 25], [8, 24], [9, 23], [10, 22],
            [11, 21], [12, 20], [13, 19], [14, 18], [15, 17],  # check
            [33, 46], [34, 45], [35, 44], [36, 43], [37, 42], [38, 50], [39, 49], [40, 48], [41, 47],  # elbrow
            [60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [65, 75], [66, 74], [67, 73],
            [55, 59], [56, 58],
            [76, 82], [77, 81], [78, 80], [87, 83], [86, 84],
            [88, 92], [89, 91], [95, 93], [96, 97]

        )
    elif dataset == '300w':
        matched_parts = (
            [1, 17], [2, 16], [3, 15], [4, 14], [5, 13], [6, 12], [7, 11], [8, 10],
            [18, 27], [19, 26], [20, 25], [21, 24], [22, 23],
            [32, 36], [33, 35],
            [37, 46], [38, 45], [39, 44], [40, 43], [41, 48], [42, 47],
            [49, 55], [50, 54], [51, 53], [62, 64], [61, 65], [68, 66], [59, 57], [60, 56]
        )
    elif dataset == 'cofw':
        matched_parts = (
            [1, 2], [5, 7], [3, 4], [6, 8], [9, 10], [11, 12], [13, 15], [17, 18], [14, 16], [19, 20], [23, 24]
        )
    else:
        # print('Not supported dataset: ' + dataset)
        raise NotImplemented()

    # Flip horizontal
    x[:, 0] = width - x[:, 0]

    # Change left-right parts
    for pair in matched_parts:
        tmp = x[pair[0]-1, :].clone()
        x[pair[0]-1, :] = x[pair[1]-1, :]
        x[pair[1]-1, :] = tmp

    return x


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def crop_v2(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(
        img, trans, (int(output_size[0]), int(output_size[1])),
        flags=cv2.INTER_LINEAR
    )

    return dst_img


def get_transform(center, scale, res, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1]/2
        t_mat[1, 2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform_pixel(pt, center, scale, res, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def transform_preds(coords, center, scale, res):
    # size = coords.size()
    # coords = coords.view(-1, coords.size(-1))
    # print(coords.size())
    for p in range(coords.size(0)):
        coords[p, 0:2] = transform_pixel(coords[p, 0:2], center, scale, res, 1, 0)
    return coords


def crop(img, center, scale, res, rot=0):
    center_new = center.clone()

    # Preprocessing for efficient cropping
    ht, wd = img.shape[0], img.shape[1]
    sf = scale * 200.0 / res[0]
    if sf < 2:
        sf = 1
    else:
        new_size = int(np.math.floor(max(ht, wd) / sf))
        new_ht = int(np.math.floor(ht / sf))
        new_wd = int(np.math.floor(wd / sf))
        if new_size < 2:
            return torch.zeros(res[0], res[1], img.shape[2]) \
                        if len(img.shape) > 2 else torch.zeros(res[0], res[1])
        else:
            img = scipy.misc.imresize(img, [new_ht, new_wd])  # (0-1)-->(0-255)
            center_new[0] = center_new[0] * 1.0 / sf
            center_new[1] = center_new[1] * 1.0 / sf
            scale = scale / sf

    # Upper left point
    ul = np.array(transform_pixel([0, 0], center_new, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform_pixel(res, center_new, scale, res, invert=1))

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]

    new_img = np.zeros(new_shape, dtype=np.float32)
    # new_img.fill(128)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    # new_img = im_to_torch(scipy.misc.imresize(new_img, res))
    new_img = scipy.misc.imresize(new_img, res)
    return new_img


def get_labelmap(img, pt, sigma, label_type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    # img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if label_type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    else:
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img


# def fliplr(x):
#     if x.ndim == 3:
#         x = np.# np.transpose(np.fliplr(np.transpose(x, (0, 2, 1))), (0, 2, 1))
#     elif x.ndim == 4:
#         for i in range(x.shape[0]):
#             x[i] = np.transpose(np.fliplr(np.transpose(x[i], (0, 2, 1))), (0, 2, 1))
#     return x.astype(float)

