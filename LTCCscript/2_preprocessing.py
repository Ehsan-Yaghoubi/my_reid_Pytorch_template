# This script receives the mask and key points of person images and removes
# some information (i.e., shape, head area, geometric information)
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import math
import random

def load_keypoints(json_dir, img_name, display=False):
    abs_name=img_name.split(".")[0]
    json_name = abs_name + ".json"
    try:
        with open(os.path.join(json_dir,json_name)) as json_file:
            data = json.load(json_file)
            _keypoints = data['bodies'][0]['joints']
    except FileNotFoundError:
        return None
    if display:
        display_keypoints(_keypoints, img_arry)
        cv2.imshow("img", img_arry)
        cv2.imshow("msk", orig_mask)
        cv2.waitKey(0)
    return _keypoints


def load_mask(masks_dir, img_name):
    _body_mask_array = cv2.imread(os.path.join(masks_dir,img_name), cv2.IMREAD_GRAYSCALE)
    return _body_mask_array

def load_img(image_dir, img_name):
    _img_array = cv2.imread(os.path.join(image_dir,img_name))
    return _img_array

def display_keypoints(keypoints, img_arry):
    """
    keypoint 0 : nose :             keypoints[0:3]
    keypoint 1 : neck :             keypoints[3:6]
    keypoint 2 : right shoulder :   keypoints[6:9]
    keypoint 3 : right elbow :      keypoints[9:12]
    keypoint 4 : right hand :       keypoints[12:15]
    keypoint 5 : left shoulder :    keypoints[15:18]
    keypoint 6 : left elbow :       keypoints[18:21]
    keypoint 7 : left hand :        keypoints[21:24]
    keypoint 8 : right hip :        keypoints[24:27]
    keypoint 9 : right knee :       keypoints[27:30]
    keypoint 10 : right foot :      keypoints[30:33]
    keypoint 11 : left hip :        keypoints[33:36]
    keypoint 12 : left knee :       keypoints[36:39]
    keypoint 13 : left foot :       keypoints[39:42]
    keypoint 14 : right eye :       keypoints[42:45]
    keypoint 15 : left eye :        keypoints[45:49]
    keypoint 16 : right ear :       keypoints[49:51]
    keypoint 17 : left ear :        keypoints[51:53]
    """
    for i in range(0,54,3):
        #round(255 * (keypoints[i + 2]))
        if (keypoints[i + 2])>=0.8:
            color = (0, 255, 0)
        elif  0.6<=(keypoints[i + 2])<0.8:
            color = (0, 125, 0)
        elif 0.3 < (keypoints[i + 2]) < 0.6:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)

        cv2.circle(img  = img_arry,
                   center = (round(keypoints[i]),round(keypoints[i+1])),
                   radius= 2,
                   color = color,
                   thickness = -1)
        if i==53:
            break

def get_head_mask(orig_mask, keypoints, display=False):
    # we can take the roi above the shoulders
    shoulder_r = keypoints[6:9]
    shoulder_l = keypoints[15:18]
    roi = [0,
           0,
           orig_mask.shape[0],
           round(max(shoulder_l[1], shoulder_r[1]))]  # x1, y1, x2, y2
    roi_img = np.copy(orig_mask[roi[1]:roi[3], roi[0]: roi[2]])  # use numpy slicing to get the image of the head
    head_mask = np.zeros(orig_mask.shape, np.uint8)  # make all the mask black
    head_mask[roi[1]:roi[3], roi[0]: roi[2]] = roi_img  # copy back only the roi

    kernel = np.ones((5,5), np.uint8)
    head_mask = cv2.dilate(head_mask, kernel)

    if display:
        cv2.imshow("new_mask", head_mask)
        cv2.waitKey(0)
    return head_mask



def get_mask_boarder(orig_mask, display=False):

    dilated_mask = mask_dilation_erosion(orig_mask, 'dilate', proportion = 0.03, iter = 1)
    eroded_mask = mask_dilation_erosion(orig_mask, 'erode', proportion = 0.05, iter = 3)

    #eroded_mask = ~eroded_mask
    eroded_mask = cv2.bitwise_not(eroded_mask)
    boarder_area = cv2.bitwise_and(dilated_mask, eroded_mask)
    if display:
        cv2.imshow("boarder_area", boarder_area)
        cv2.waitKey(0)
    return boarder_area

def get_final_mask(head_mask, mask_boarder, display=False):
    # assert head_mask.shape==mask_boarder.shape
    # assert head_mask.dtype==mask_boarder.dtype
    final_mask = cv2.bitwise_or(head_mask, mask_boarder)
    if display:
        cv2.imshow("final_mask", final_mask)
        cv2.waitKey(0)
    return final_mask

def do_inpainting(img, mask, display=False):
    radius = int(max(np.array(orig_mask.shape) * 0.03))
    inpainted_img = cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)
    if display:
        cv2.imshow("inpainted_img", inpainted_img)
        cv2.waitKey(0)
    return inpainted_img

def mask_dilation_erosion(mask, flag, proportion = 0.03, iter = 1):

    kernel_size = (np.array(mask.shape) * proportion).astype(int)
    kernel = np.ones(kernel_size, np.uint8)
    if flag == 'dilate':
        _mask = cv2.dilate(mask, kernel, iterations=iter)
    elif flag == 'erode':
        _mask = cv2.erode(orig_mask, kernel, iterations=iter)
    else:
        print("Specify the flag correctly")
        raise ValueError

    return _mask

def perturbed_mesh(row, column, desplay=False):
    # the idea has been taken from paper https://www.juew.org/publication/DocUNet.pdf
    mr = row
    mc = column

    xx = np.arange(mr - 1, -1, -1)
    yy = np.arange(0, mc, 1)

    # xx1 = np.random.randint(0,mr,(1,mr))[0]
    # yy1 =np.random.randint(0,mc,(1,mc))[0]

    [Y, X] = np.meshgrid(xx, yy)
    X_flatten = X.flatten('F')
    Y_flatten = Y.flatten('F')
    XY_mat = [X_flatten, Y_flatten]
    XY_mat_arr = np.asarray(XY_mat)
    ms = np.transpose(XY_mat_arr, (1, 0))

    perturbed_mesh = ms
    nv = np.random.randint(20) - 1
    for k in range(nv):
        # Choosing one vertex randomly
        vidx = np.random.randint(np.shape(ms)[0])
        vtex = ms[vidx, :]
        # Vector between all vertices and the selected one
        xv = perturbed_mesh - vtex
        # Random movement
        mv = (np.random.rand(1, 2) - 0.5) * 20
        hxv = np.zeros((np.shape(xv)[0], np.shape(xv)[1] + 1))
        hxv[:, :-1] = xv
        hmv = np.tile(np.append(mv, 0), (np.shape(xv)[0], 1))
        d = np.cross(hxv, hmv)
        d = np.absolute(d[:, 2])
        d = d / (np.linalg.norm(mv, ord=2))
        wt = d

        curve_type = np.random.rand(1)
        if curve_type > 0.3:
            alpha = np.random.rand(1) * 50 + 50
            wt = alpha / (wt + alpha)
        else:
            alpha = np.random.rand(1) + 1
            wt = 1 - (wt / 100) ** alpha
        msmv = mv * np.expand_dims(wt, axis=1)
        perturbed_mesh = perturbed_mesh + msmv
    if desplay:
        plt.scatter(perturbed_mesh[:, 0], perturbed_mesh[:, 1], c=np.arange(0, mr * mc))
        plt.show()
    return perturbed_mesh[:, 0], perturbed_mesh[:, 1]

def make_same_sizes (img1, img2, img2_mask):
    h1, w1 = img1.shape[0:2]
    h2, w2 = img2.shape[0:2]
    if h1>h2: # increase the height of img2
        img2_p = cv2.copyMakeBorder(img2,int((h1-h2)/2), int((h1-h2)/2), 0, 0 , borderType=cv2.BORDER_REPLICATE)
        img2_m_p = cv2.copyMakeBorder(img2_mask, int((h1 - h2) / 2), int((h1 - h2) / 2), 0, 0, borderType=cv2.BORDER_REPLICATE)
        img1_p = np.copy(img1)
    elif h2>h1: # increase the height of img1
        img2_p = np.copy(img2)
        img2_m_p = np.copy(img2_mask)
        img1_p = cv2.copyMakeBorder(img1, int((h2 - h1) / 2), int((h2 - h1) / 2), 0, 0, borderType=cv2.BORDER_REPLICATE)
    else:
        img1_p = np.copy(img1)
        img2_p = np.copy(img2)
        img2_m_p = np.copy(img2_mask)

    if w1>w2:# increase the width of img2
        img2_p = cv2.copyMakeBorder(img2_p, 0, 0, int((w1-w2)/2), int((w1-w2)/2) , borderType=cv2.BORDER_REPLICATE)
        img2_m_p = cv2.copyMakeBorder(img2_m_p, 0, 0, int((w1 - w2) / 2), int((w1 - w2) / 2),
                                    borderType=cv2.BORDER_REPLICATE)
    elif w2>w1:
        img1_p = cv2.copyMakeBorder(img1_p, 0, 0, int((w2-w1)/2), int((w2-w1)/2) , borderType=cv2.BORDER_REPLICATE)
    else:
        pass

    if img1_p.shape[0:2] == img2_p.shape[0:2] == img2_m_p.shape[0:2]:
        return img1_p, img2_p, img2_m_p
    else: # make them the same size
        h,w = img1_p.shape[0:2]
        img2_p = cv2.resize(img2_p, (w,h))
        img2_m_p = cv2.resize(img2_m_p, (w,h))
        return img1_p, img2_p, img2_m_p


def do_deformed_mesh(img, mask, display=False):
    # if display:
    #     cv2.imshow("mask", mask)
    #     cv2.waitKey(0)

    nw,nh = img.shape[0:2]

    body_area = cv2.bitwise_and(img, img, mask= mask)
    dw = nw // 2
    dh = nh // 2
    extended_body_area = cv2.copyMakeBorder(body_area, dh, dh, dw, dw, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    extended_body_area_mask = cv2.copyMakeBorder(mask, dh, dh, dw, dw, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # if display:
    #     cv2.imshow("extended_body_area_mask", extended_body_area_mask)
    #     cv2.waitKey(0)

    nw, nh = extended_body_area.shape[0:2]

    xs, ys = perturbed_mesh(nh,nw, False)  # the result is like np.meshgrid
    xs = xs.reshape(nh, nw).astype(np.float32)
    ys = ys.reshape(nh, nw).astype(np.float32)
    dst_1 = cv2.remap(extended_body_area, ys, xs, cv2.INTER_CUBIC)
    dst_2 = cv2.rotate(dst_1, cv2.ROTATE_90_CLOCKWISE)

    mask_dst_1 = cv2.remap(extended_body_area_mask, ys, xs, cv2.INTER_CUBIC)
    mask_dst_2 = cv2.rotate(mask_dst_1, cv2.ROTATE_90_CLOCKWISE)
    # if display:
    #     cv2.imshow("mask_dst_2", mask_dst_2)
    #     cv2.waitKey(0)

    # minimum rectangular contour
    dst_2_grey = cv2.cvtColor(dst_2, cv2.COLOR_BGR2GRAY)
    x,y,w,h = cv2.boundingRect(dst_2_grey)
    # if display:
    #     #show the rect on the image
    #     cv2.rectangle(dst_2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cropped_dst_2 = dst_2[y:y+h, x:x+w, :]
    cropped_mask_dst_2 = mask_dst_2[y:y+h, x:x+w]
    # if display:
    #     cv2.imshow("cropped_mask_dst_2", cropped_mask_dst_2)
    #     cv2.waitKey(0)

    inpanted_img = do_inpainting(img, mask, False)
    inpanted_img, cropped_dst_2, cropped_mask_dst_2 = make_same_sizes (inpanted_img, cropped_dst_2, cropped_mask_dst_2)

    # if display:
    #     cv2.imshow("cropped_mask_dst_2_resized", cropped_mask_dst_2)
    #     cv2.waitKey(0)

###################
    # cropped_dst_2_grey = cv2.cvtColor(cropped_dst_2,cv2.COLOR_BGR2GRAY)
    #
    # _, mask1 = cv2.threshold(cropped_dst_2_grey, 0, 255, cv2.THRESH_BINARY)
    # # contours, hierarchy = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # img = cv2.drawContours(cropped_dst_2, contours, -1, (0, 255, 0), thickness=cv2.FILLED)
###################

    kernel = np.ones((5, 5), np.uint8)
    mask2 = cv2.erode(cropped_mask_dst_2, kernel)

    # if display:
    #     cv2.imshow("cropped_mask_dst_2_eroded", mask2)
    #     cv2.waitKey(0)

    mask_3 = np.expand_dims(mask2, axis=2)
    mask4 = mask_3 * np.ones((1,1,3))
    mask4 = mask4>127
    np.copyto(inpanted_img, cropped_dst_2, casting='unsafe', where = mask4)

    # cropped_dst_2_mask_bin = [cropped_dst_2_grey!=0]
    # cropped_dst_2_mask_arr = np.array(cropped_dst_2_mask_bin[0]*255, dtype=np.uint8)
    # mask3D = np.dstack([mask] * 3)
    # dst_3 = cv2.bitwise_and(inpanted_img, inpanted_img, mask=thresh)


    if display:
        # cv2.imshow("cropped_dst_2", cropped_dst_2)
        cv2.imshow("inpanted_img", inpanted_img)
        cv2.waitKey(0)

    return inpanted_img

def do_scratch(img, min_prop, max_prop, display= False):
    if min_prop>= 1:
        raise ValueError("select a value LESS than 1")
    if max_prop<= 1:
        raise ValueError("select a value MORE than 1")

    h1, w1 = img.shape[0:2]
    w2 = random.randint(int(min_prop*w1), int(max_prop*w1))
    h2 = random.randint(int(min_prop*h1), int(max_prop*h1))
    resized_img = cv2.resize(img, (w2, h2))

    r1 = h1/w1
    r2 = h2/w2

    if r1>r2:
        add_to_length = (w2*r1)-h2
        resized_img = cv2.copyMakeBorder(resized_img, int(add_to_length/2), int(add_to_length/2), 0, 0, cv2.BORDER_REPLICATE)
        if display:
            cv2.imshow("resized_img", resized_img)
            cv2.waitKey(0)
        return resized_img
    elif r2>r1:
        add_to_width = (h2/r1)-w2
        resized_img = cv2.copyMakeBorder(resized_img, 0, 0, int(add_to_width/2), int(add_to_width/2), cv2.BORDER_REPLICATE)
        if display:
            cv2.imshow("resized_img", resized_img)
            cv2.waitKey(0)
        return resized_img
    else:
        if display:
            cv2.imshow("resized_img", resized_img)
            cv2.waitKey(0)
        return resized_img




def random_mosaic_tile(img, max_tiles, mask, display = False):
    h, w = img.shape[0:2]
    tiles = random.randint(2, max_tiles)
    wt = int(w/tiles)
    ht = int(h/tiles*2)
    tiled_img = np.zeros(img.shape, np.uint8)
    print(img.dtype)
    crop_hori = dict()
    crop_vert = dict()
    for i in range(0, tiles+1, 1):
        crop_hori['{}'.format(i)] = img[(ht * i):(ht * (i + 1)), :, :]
        crop_vert['{}'.format(i)] = img[:, (wt * i):(wt * (i + 1)), :]
    for j in range(0, tiles, 1): # create a tiled image by placing the crops horizontally and vertically
        if tiles>max_tiles/2:
            tiled_img[(ht * j):(ht * (j + 1)), :, :] = crop_hori['{}'.format(random.randint(0, tiles - 1))]
            tiled_img[:, (wt * j):(wt * (j + 1)), :] = crop_vert['{}'.format(random.randint(0, tiles - 1))]
        else:
            tiled_img[:, (wt * j):(wt * (j + 1)), :] = crop_vert['{}'.format(random.randint(0, tiles - 1))]
            tiled_img[(ht * j):(ht * (j + 1)), :, :] = crop_hori['{}'.format(random.randint(0, tiles - 1))]

    if display:
        cv2.imshow("tiled_img", tiled_img)
        cv2.waitKey(0)

def write_img(dir, img_array):
    if not os.path.isfile(os.path.join(dir, img_name)):
        cv2.imwrite(os.path.join(dir, img_name), img_array)
    else:
        print("file exists and did not over-writen")

if __name__ == '__main__':

    mask_dir = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/LTCC_ReID/LTCC_masks/query"
    image_dir = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/LTCC_ReID/LTCC_orig/query"
    json_files_dir = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/LTCC_ReID/LTCC_keypoints/query"
    dir_to_save_LTCC_noneID = "/media/socialab157/2cbae9f1-6394-4fa9-b963-5ef890eee044/B_DATASETS/Long_term_datasets/LTCC_ReID/LTCC_nonID/query_No_scratching"

    os.makedirs(dir_to_save_LTCC_noneID, exist_ok=True)

    images_names = os.listdir(mask_dir)
    for index, img_name in enumerate(images_names):

        img_arry = load_img(image_dir, img_name)
        orig_mask = load_mask(mask_dir, img_name)
        keypoints = load_keypoints(json_files_dir, img_name, False)
        if keypoints is None or orig_mask is None or img_arry is None:
            print("(img/mask/keypoint)data did not find: {}".format(img_name))
            continue

        head_mask = get_head_mask(orig_mask, keypoints, False)

        mask_boarder = get_mask_boarder(orig_mask, False)

        final_mask = get_final_mask(head_mask, mask_boarder, False)

        head_inpainted_img = do_inpainting(img_arry, final_mask, False)

        # mosaic_img = random_mosaic_tile(head_inpainted_img, 4, orig_mask, True) # it is done on the image

        deformed_img = do_deformed_mesh(head_inpainted_img, orig_mask, False) # can be done on mask and img

        scratched_img = do_scratch(deformed_img, 0.5, 1.5, False)  # can be done on mask and img


        write_img(dir_to_save_LTCC_noneID, deformed_img)

        if index%100 == 0:
            print("Save Generated Images to {}: {}/{}".format(dir_to_save_LTCC_noneID, index,len(images_names)))






