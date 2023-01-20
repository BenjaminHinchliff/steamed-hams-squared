#!/usr/bin/env python

import cv2
import numpy as np
import scipy as sp

INPUT_FILE = "steamed-hams.webm"
OUTPUT_FILE = "steamed-hams-steamed-hams.mkv"
OUTPUT_CODEC = "mp4v"
TILE_RESOLUTION = (32, 24)
OUTPUT_RESOLUTION = (1440, 1080)


def get_dominant_color(img):
    # n_colors = 5
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    # flags = cv2.KMEANS_RANDOM_CENTERS

    pixels = np.float32(img.reshape(-1, 3))

    # _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    # _, counts = np.unique(labels, return_counts=True)
    # dominant = palette[np.argmax(counts)]
    dominant = np.mean(pixels, axis=0)
    return dominant


cap = cv2.VideoCapture(INPUT_FILE)

# Check if camera opened successfully
if not cap.isOpened():
    raise SystemExit("Error opening video stream or file")

dominants = []
frames = {}

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

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

    _, height, _ = frame.shape
    square_frame = cv2.resize(frame, (height, height), interpolation=cv2.INTER_AREA)
    frames[dominant.tobytes()] = square_frame

    frame_num += 1

print()

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

dominants_tree = sp.spatial.KDTree(dominants)


def closest_pixel_frame(pixel):
    return frames[dominants[dominants_tree.query(pixel)[1]].tobytes()]


fourcc = cv2.VideoWriter_fourcc(*OUTPUT_CODEC)
video_out = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, OUTPUT_RESOLUTION)

frame_num = 0
while cap.isOpened():
    print(
        f"\rConverting frame {frame_num} of {frame_count} ({frame_num / frame_count:.0%})",
        end="",
    )

    ret, frame = cap.read()
    if not ret:
        break

    downscaled = cv2.resize(frame, TILE_RESOLUTION, interpolation=cv2.INTER_AREA)

    # convert and stack images
    downscaled = np.vstack(
        [np.hstack([closest_pixel_frame(pixel) for pixel in row]) for row in downscaled]
    )

    final = cv2.resize(downscaled, OUTPUT_RESOLUTION, interpolation=cv2.INTER_AREA)

    video_out.write(final)

    frame_num += 1

cap.release()
video_out.release()

cv2.destroyAllWindows()
