#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML
%matplotlib inline

#grayscale an image
def grayscale(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)


#Applies the Canny transform
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


#Applies a Gaussian Noise kernel
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


#Applies an image mask
def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# Iterate over the output "lines" and draw lines
# Takes the result lines from Hough Transform as input, draw the final extended lane found
def draw_lines(img, lines, color=[255, 0, 0], thickness=10):  
    # Acquire current lane situation (x1, y1, x2, y2)
    global current_left_lane
    global current_right_lane

    # Get two of the found longest lines as dominant lines
    max_length_left_line, max_length_right_line = get_dominant_lines(img, lines)

    # Left one
    # Check whether the two dominant lines are reasonable, return a combine extended line if they are,
    # else return current line.
    nx1, ny1, nx2, ny2 = check_lane(img, current_left_lane, max_length_left_line)
    # To acquire a smoother change from the current line to the new one
    current_left_lane = x1, y1, x2, y2 = smoother(img, current_left_lane, nx1, ny1, nx2, ny2)
    # Draw the line
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    # Right one
    nx1, ny1, nx2, ny2 = check_lane(img, current_right_lane, max_length_right_line)
    current_right_lane = x1, y1, x2, y2 = smoother(img, current_right_lane, nx1, ny1, nx2, ny2)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
#     for line in lines:
#         for x1,y1,x2,y2 in line:
#             cv2.line(img,(x1,y1),(x2,y2),color,thickness)

# Get two of the found longest lines as dominant lines to estimate real lane location        
def get_dominant_lines(img, lines):
    # Image information
    xsize = img.shape[1]

    # Used for finding two dominant lane
    # Ps: [0] is the one smaller than [1]
    max_length_left = [0, 0]
    max_length_right = [0, 0]
    max_length_left_line = [[0, 0, 0, 0], [0, 0, 0, 0]]
    max_length_right_line = [[0, 0, 0, 0], [0, 0, 0, 0]]

    for line in lines:
        for x1, y1, x2, y2 in line:
            # Calculate length of the line
            length = ((y2-y1)**2 + (x2-x1)**2)**1/2
            
            # Calculate gradient of the line
            if x2 == x1:
                continue
            k = ((y2-y1)/(x2-x1))
            
            # Line gradient should line in this gradient threshold and
            # it should also lie in one of the half parts of the image
            thres_gradient = (0.5, 1.7) # pi/2 bis pi/3

            # left ones
            if (-thres_gradient[1] < k < -thres_gradient[0]) and (x1 < xsize/2):
                # Replace the shorter one
                if length > max_length_left[1]:
                    max_length_left = [max_length_left[1], length]
                    max_length_left_line = [max_length_left_line[1], [x1, y1, x2, y2]]
                elif length > max_length_left[0]:
                    max_length_left[0] = length
                    max_length_left_line[0] = x1, y1, x2, y2
            # right ones
            elif (thres_gradient[0] < k < thres_gradient[1]) and (x2 > xsize/2):
                # Replace the shorter one
                if length > max_length_right[1]:
                    max_length_right = [max_length_right[1], length]
                    max_length_right_line = [max_length_right_line[1], [x1, y1, x2, y2]]
                elif length > max_length_left[0]:
                    max_length_right[0] = length
                    max_length_right_line[0] = x1, y1, x2, y2

    return max_length_left_line, max_length_right_line

# check new line coordinate and compare it with the current one
# If new line is reasonable, return it, else return the current one
def check_lane(img, current_line, new_line):
    # Image information
    ysize = img.shape[0]
    xsize = img.shape[1]

    # New line positions
    nx1, ny1, nx2, ny2 = new_line[0]
    nx3, ny3, nx4, ny4 = new_line[1]
    cv2.line(img, (nx1, ny1), (nx2, ny2), [255, 0, 255], 2)
    cv2.line(img, (nx3, ny3), (nx4, ny4), [255, 0, 255], 2)

    # Turn two short lines into one long extended line
    if nx1 != 0:  # two lines found
        k, b = np.polyfit([nx1, nx2, nx3, nx4], [ny1, ny2, ny3, ny4], 1)
    else:  # no line or only one line found, continue current line
        return current_line
    # Calculate extended long line's position
    nx1, ny1, nx2, ny2 = int((ysize - b)/k), ysize, int((ysize//1.5 - b)/k), int(ysize/1.5)

    # If new position is not reasonable (out of image)
    if not (0 < nx1 < xsize) or not (0 < nx2 < xsize):
        return current_line

    # If new position is too far from the current ones
    dist_thres = 60
    if current_line != [0, 0, 0, 0] and \
            (abs(nx1 - current_line[0]) > dist_thres or abs(nx2 - current_line[2]) > dist_thres):
        return current_line

    return nx1, ny1, nx2, ny2

# This function will make the change of line position 
# smoother by setting a smoother coefficient
def smoother(img, current_line, x1, y1, x2, y2):
    # If new line stays the same as current_line, no need to move
    if current_line[0] == x1 and current_line[2] == x2:
        return current_line

    # Image information
    ysize = img.shape[0]

    # Smoother coefficient
    step = 10
    if current_line != [0, 0, 0, 0]:
        cx1 = current_line[0]
        cx2 = current_line[2]
        x1 = cx1 + (x1 - cx1)/step  # Change in a small step every time
        x2 = cx2 + (x2 - cx2)/step  # Change in a small step every time

    # Fit to find the new gradient k and b in order to draw a new long line
    k, b = np.polyfit([x1, x2], [y1, y2], 1)
    # Return a long line position
    return int((ysize - b)/k), ysize, int((ysize//1.5 - b)/k), int(ysize//1.5)


# Run Hough on edge detected image
#`img` should be the output of a Canny transform
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            min_line_len, max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Draw the lines on the edge image
#`img` is the output of the hough_lines(), An image with lines drawn on it. 
# Should be a blank image (all black) with lines drawn on it.
def weighted_img(img, initial_img, α=0.8, β=1., γ=0.0):
    return cv2.addWeighted(initial_img, α, img, β, γ) 


# Define function to process image
def process_image(image):
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # gaussian blur and edges detection
    kernel_size = 11
    blur_gray = gaussian_blur(gray, kernel_size)
    low_threshold = 30
    high_threshold = 50
    edges = canny(blur_gray, 30, 50)
    
    # define Region Masking
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(0.4*imshape[1], 0.6*imshape[0]), (0.6*imshape[1], 0.6*imshape[0]), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
      
    # Define the Hough transform parameters
    rho = 3 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 10     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 45 #minimum number of pixels making up a line
    max_line_gap = 35  # maximum gap in pixels between connectable line segments
    # Output "lines" is an array containing endpoints of detected line segments
    lines = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    
    
    # Iterate over the output "lines" and draw lines on a blank image
    lines_edges = weighted_img(lines, image, α=0.8, β=1., γ=0.0)
    
    return lines_edges


# Testing with images
test_image_list = os.listdir("test_images/")
# read images one by one
for imageFile in test_image_list:
    # 'current lane' variables are used for smoother lane location change
    # Since testing image, reset this every time
    current_left_lane = [0, 0, 0, 0]
    current_right_lane = [0, 0, 0, 0]
    # Reading image
    img = mpimg.imread("test_images/" + imageFile)
    # Process image
    image_output = process_image(img)
    # Save image
    mpimg.imsave("test_images_output/" + imageFile, image_output)
    
# Test on Videos
# Store current line position
current_left_lane = [0, 0, 0, 0]
current_right_lane = [0, 0, 0, 0]
white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

current_left_lane = [0, 0, 0, 0]
current_right_lane = [0, 0, 0, 0]
yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
#clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)

challenge_output = 'test_videos_output/challenge.mp4'
current_left_lane = [0, 0, 0, 0]
current_right_lane = [0, 0, 0, 0]
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
#clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)