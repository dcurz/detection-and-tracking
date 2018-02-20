import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
from features_scan import *

def process_frame_subsample(image):
    
    color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 32    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    y_start_stop = [300, 720] # Min and max in y to search in slide_window()
    conv='RGB2HLS'
    
    hot_windows = []

    ystart = 380
    ystop = 550
    scale_one = 1.0
    cells_per_step = 1
    hot_windows.extend(find_cars(image, ystart, ystop, scale_one, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, cells_per_step))
    ystart = 380
    ystop = 600
    scale_two = 1.5
    cells_per_step = 3
    hot_windows.extend(find_cars(image, ystart, ystop, scale_two, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, cells_per_step))
    ystart = 350
    ystop = 600
    scale_three = 2.5
    cells_per_step = 3
    hot_windows.extend(find_cars(image, ystart, ystop, scale_three, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, cells_per_step))
#     ystart = 450
#     ystop = 720
#     scale_four = 4.0
#     hot_windows.extend(find_cars(image, ystart, ystop, scale_four, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))

    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    
    # Add heat to each box in box list
    heat = add_heat(heat,hot_windows)
    heat = apply_threshold(heat, 2)
    
    #shift down elements in array
    for i in range (len(heatmap_array)-1): 
        heatmap_array[i] = heatmap_array[i+1]
        
    #append "heat" to end of heatmap array
    heatmap_array[len(heatmap_array)-1] = heat
    
    all_heat = sum(heatmap_array)
        

    # Apply threshold to help remove false positives
    all_heat = apply_threshold(all_heat,7)

    # Visualize the heatmap when displaying    
    #all_heat = all_heat*30
    heatmap = np.clip(all_heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    
    return draw_img

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (255,140,0), 6)
    # Return the image
    return img
