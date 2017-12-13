import matplotlib.pyplot as plt

from ImageClassifier import *
from Visualisation import *
from scipy.ndimage.measurements import label
import copy

saved_car_boxs = []


svc, X_scaler = get_classifier()

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size):
    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    window_list = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)

            # Scale features and make a prediction
            # test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_features = X_scaler.transform(np.hstack((spatial_features, hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                window_list.append(((xbox_left, ytop_draw + ystart),(xbox_left + win_draw, ytop_draw + win_draw + ystart)))
                # cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                #               (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

    return window_list

def get_classified_cars_boxs(image):
    cars_boxs = []
    scale = 1
    while scale <= 3:
        cars_boxs.extend(
            find_cars(image, int(image.shape[0] * .55), image.shape[0], scale, svc, X_scaler, orient, pix_per_cell,
                      cell_per_block, spatial_size))
        scale = scale + 0.5
    return cars_boxs

import collections

saved_car_boxs = collections.deque(maxlen=3)

def get_heat_map_boxs(cars_boxs, img,isVideo):
    global saved_car_boxs
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)

    if not isVideo:
        saved_car_boxs = collections.deque(maxlen=1)

    # Add heat to each box in box list
    saved_car_boxs.append(cars_boxs)

    boxs_list = np.array(list(saved_car_boxs))

    for boxs in boxs_list:
        heat = add_heat(heat, boxs)

    # Apply threshold to help remove false positives
    # if not isVideo:
    heat = apply_threshold(heat, 2)
    # else:
    #     heat = apply_threshold(heat, 1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    box_list = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        box_list.append(bbox)
    return box_list,heat


# image = mpimg.imread('test_images/test1.jpg')
#
# cars_boxs_list = get_cars_boxs(image)
#
# window_img = draw_boxes(image, cars_boxs_list, color=(0, 0, 255), thick=6)
#
# plt.imshow(window_img)
# plt.show()


