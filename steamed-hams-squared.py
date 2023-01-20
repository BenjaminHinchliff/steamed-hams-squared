#!/usr/bin/env python

import cv2
import numpy as np
import scipy as sp

INPUT_FILE = "steamed-hams.webm"
OUTPUT_FILE = "steamed-hams-steamed-hams.mkv"
# used because it is fast and I can just re-encode it later
OUTPUT_CODEC = "mp4v"
TILE_RESOLUTION = (32, 24)
# 4:3 1080p
OUTPUT_RESOLUTION = (1440, 1080)

# Get the dominant color of an opencv image
def get_dominant_color(img):
    pixels = np.float32(img.reshape(-1, 3))

    # Old code from when I was testing using kmeans to find dominant colors
    # of frames instead of average colors, but it produced mediocre (possibly
    # worse) results with massively worse performance
    #
    # n_colors = 5
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    # flags = cv2.KMEANS_RANDOM_CENTERS
    # _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    # _, counts = np.unique(labels, return_counts=True)
    # dominant = palette[np.argmax(counts)]

    dominant = np.mean(pixels, axis=0)
    return dominant


cap = cv2.VideoCapture(INPUT_FILE)
if not cap.isOpened():
    raise SystemExit("Error opening video stream or file")

# dominant color of each frame
dominants = []
# dict to associate each color with a particular frame of video
frames = {}

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# first pass - find dominant colors of all frames
frame_num = 0
while cap.isOpened():
    print(
        f"\rGetting dominant color of frame {frame_num} of {frame_count} ({frame_num / frame_count:.0%})",
        end="",
    )

    ret, frame = cap.read()
    if not ret:
        break

    dominant = get_dominant_color(frame)
    dominants.append(dominant)

    # make frames square so aspect ratio of video is preserved
    _, height, _ = frame.shape
    square_frame = cv2.resize(frame, (height, height), interpolation=cv2.INTER_AREA)

    frames[dominant.tobytes()] = square_frame

    frame_num += 1

print()

# reset video for another pass
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# k-d tree to find nearest dominant color
dominants_tree = sp.spatial.KDTree(dominants)

# find the closest average frame color to a given pixel color
def closest_pixel_frame(pixel):
    # find index of closest dominant
    _, idx = dominants_tree.query(pixel)
    # get actual dominant value
    dominant = dominants[idx]
    # find that value in the dict
    return frames[dominant.tobytes()]


fourcc = cv2.VideoWriter_fourcc(*OUTPUT_CODEC)
video_out = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, OUTPUT_RESOLUTION)

# second pass - convert each frame to images by using numpy stacking functions
frame_num = 0
while cap.isOpened():
    print(
        f"\rConverting frame {frame_num} of {frame_count} ({frame_num / frame_count:.0%})",
        end="",
    )

    ret, frame = cap.read()
    if not ret:
        break

    # downscale source to tile resolution
    downscaled = cv2.resize(frame, TILE_RESOLUTION, interpolation=cv2.INTER_AREA)

    # approximate and stack images
    downscaled = np.vstack(
        [np.hstack([closest_pixel_frame(pixel) for pixel in row]) for row in downscaled]
    )

    # downscale the stacked images to the desired output resolution
    final = cv2.resize(downscaled, OUTPUT_RESOLUTION, interpolation=cv2.INTER_AREA)

    video_out.write(final)

    frame_num += 1

cap.release()
video_out.release()

cv2.destroyAllWindows()
