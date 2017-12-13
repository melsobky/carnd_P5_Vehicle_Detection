import PreProcessing as PreProcessing
import LaneFinding as LaneFinding
import Support as Support
import cv2
from moviepy.editor import VideoFileClip
import os.path
from VehicleDetector import *
import Visualisation
import numpy as np

import collections

def get_images_list(path):
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    return onlyfiles

def Process_Video_file(filename):
    vclip = VideoFileClip('test_videos/' + filename)

    processed_clip = vclip.fl_image(video_pipeline)

    processed_clip.write_videofile('output_videos/'+filename, audio=False)

def process_image_file(image_file,save_output_images=True):
    img = cv2.imread('test_images/' + image_file)
    file_name = image_file.split('.')[0]

    dst,unfiltered,heat_map,output = pipeline(img)

    if (save_output_images is True):
        if not dst is None:
            cv2.imwrite('output_images/' + file_name + '_0_undistorted.jpg', dst)
        if not unfiltered is None:
            cv2.imwrite('output_images/' + file_name + '_1_unfiltered.jpg', unfiltered)
        if not heat_map is None:
            heat_map = heat_map * 255
            cv2.imwrite('output_images/' + file_name + '_2_heat_map.jpg', heat_map)
        if not output is None:
            cv2.imwrite('output_images/' + file_name + '_3_output.jpg', output)


def video_pipeline(img):
    dst, binary, binary_warped, output = pipeline(img,isVideo=True)
    return output

smoothing_frame_counts = 10

left_fitx_buffer = collections.deque(maxlen=smoothing_frame_counts)
right_fitx_buffer = collections.deque(maxlen=smoothing_frame_counts)

lcurve_buffer = collections.deque(maxlen=smoothing_frame_counts)
rcurve_buffer = collections.deque(maxlen=smoothing_frame_counts)

def getSmoothedCurveData(left_fitx, right_fitx):
    global lcurve_buffer,rcurve_buffer
    lcurve_buffer.append(left_fitx)
    rcurve_buffer.append(right_fitx)

    lcurve_list = np.array(list(lcurve_buffer))
    rcurve_list = np.array(list(rcurve_buffer))

    lcurve_mean = np.mean(lcurve_list, axis=0)
    rcurve_mean = np.mean(rcurve_list, axis=0)

    return lcurve_mean, rcurve_mean

def getSmoothedLanesData(left_fitx, right_fitx):
    global left_fitx_buffer,right_fitx_buffer
    left_fitx_buffer.append(left_fitx)
    right_fitx_buffer.append(right_fitx)

    left_fitx_list = np.array(list(left_fitx_buffer))
    right_fitx_list = np.array(list(right_fitx_buffer))

    left_fitx_mean = np.mean(left_fitx_list, axis=0,dtype=np.int32)
    right_fitx_mean = np.mean(right_fitx_list, axis=0,dtype=np.int32)

    return left_fitx_mean, right_fitx_mean


def pipeline(img,isVideo=False):
    # Image Preprocessing
    undst,binary,binary_warped = PreProcessing.preprocess_image(img)

    # Lane Detection Code Start
    lanes, leftx, lefty, rightx, righty, ploty = LaneFinding.get_lane_lines(binary_warped,isVideo)

    lcurve, rcurve = Support.get_real_lanes_curvature(ploty, leftx, lefty, rightx, righty)

    output = draw_lane_area(undst,binary_warped,ploty,leftx, lefty, rightx, righty,isVideo)

    left_fit, right_fit, dummy = Support.fit_polylines(binary_warped.shape[0], leftx, lefty, rightx, righty, x_scale_factor=1, y_scale_factor=1)

    left_fitx, right_fitx = Support.get_polylines_points(ploty, left_fit, right_fit)

    if(isVideo is True):
        lcurve, rcurve = getSmoothedCurveData(lcurve, rcurve)
        left_fitx, right_fitx = getSmoothedLanesData(left_fitx, right_fitx)

    shiftFromLaneCenter_m, side = calculate_shift_from_lane_center(binary_warped,left_fitx, right_fitx)

    Font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    cv2.putText(output, 'curve = ' + str((lcurve + rcurve) / 2) + ' m', (10, 100), Font, 1, color, 2, cv2.LINE_AA)

    cv2.putText(output, 'Vehicle is ' + str(shiftFromLaneCenter_m) + ' (m) ' + side + ' of lane center', (10, 150), Font, 1,color, 2, cv2.LINE_AA)
    # Lane Detection Code End

    # Vehicle Detection Code Start
    cars_boxs = get_classified_cars_boxs(undst)
    classified_boxs = Visualisation.draw_boxes(undst, cars_boxs, color=(0, 0, 255), thick=6)
    filtered_boxs, heat_map = get_heat_map_boxs(cars_boxs,undst,isVideo)
    output = Visualisation.draw_boxes(output, filtered_boxs, color=(0, 0, 255), thick=6)
    # Vehicle Detection Code End

    return undst,classified_boxs,heat_map,output
    # return None,None,None,None,output
    # return undst,binary,binary_warped,None,None


def calculate_shift_from_lane_center(warped,left_fitx,right_fitx):
    LaneCenterx = (right_fitx[-1]+left_fitx[-1])/2

    shiftFromLaneCenter = warped.shape[1]/2 - LaneCenterx #Pixels

    shiftFromLaneCenter_m = shiftFromLaneCenter * Support.xm_per_pix #meters

    if shiftFromLaneCenter > 0:
        side = 'Right'
    else:
        side = 'Left'
    return abs(shiftFromLaneCenter_m),side


def draw_lane_area(undist, warped, ploty, leftx, lefty, rightx, righty, isVideo=False):
    left_fit, right_fit, dummy = Support.fit_polylines(warped.shape[0], leftx, lefty, rightx, righty, x_scale_factor=1, y_scale_factor=1)

    left_fitx, right_fitx = Support.get_polylines_points(ploty, left_fit, right_fit)

    if (isVideo is True):
        left_fitx, right_fitx = getSmoothedLanesData(left_fitx, right_fitx)
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    return result


image_files_names = get_images_list('test_images/')
Minv = PreProcessing.get_inv_transform_matrix()

# process_image_file('test1.jpg')
# process_image_file('test4.jpg')
Process_Video_file('project_video.mp4')
# Process_Video_file('test_video.mp4')
# Process_Video_file('test_video_1.mp4')
# Process_Video_file('test_video_2.mp4')
# Process_Video_file('test_video_3.mp4')
# Process_Video_file('test_video_4.mp4')

# for image_file in image_files_names:
#     process_image_file(image_file)

# car_image = cv2.imread('datasets/vehicles/GTI_Right/image0800.png')
# cv2.imwrite('output_images/car_image.jpg', car_image)
# dummy,car_image_hog_ch0 = get_hog_features(car_image[:,:,0],9,8,2,True,False)
# cv2.imwrite('output_images/car_image_hog_ch0.jpg', car_image_hog_ch0*255)
# dummy,car_image_hog_ch1 = get_hog_features(car_image[:,:,1],9,8,2,True,False)
# cv2.imwrite('output_images/car_image_hog_ch1.jpg', car_image_hog_ch1*255)
# dummy,car_image_hog_ch2 = get_hog_features(car_image[:,:,2],9,8,2,True,False)
# cv2.imwrite('output_images/car_image_hog_ch2.jpg', car_image_hog_ch2*255)
#
# notcar_image = cv2.imread('datasets/non-vehicles/GTI/image832.png')
# cv2.imwrite('output_images/notcar_image.jpg', notcar_image)
# dummy,notcar_image_hog_ch0 = get_hog_features(notcar_image[:,:,0],9,8,2,True,False)
# cv2.imwrite('output_images/notcar_image_hog_ch0.jpg', notcar_image_hog_ch0*255)
# dummy,notcar_image_hog_ch1 = get_hog_features(notcar_image[:,:,1],9,8,2,True,False)
# cv2.imwrite('output_images/notcar_image_hog_ch1.jpg', notcar_image_hog_ch1*255)
# dummy,notcar_image_hog_ch2 = get_hog_features(notcar_image[:,:,2],9,8,2,True,False)
# cv2.imwrite('output_images/notcar_image_hog_ch2.jpg', notcar_image_hog_ch2*255)