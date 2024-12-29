#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import shutil
import cv2
import numpy as np

###############################################################################
# Cartoonising Functions
###############################################################################

def colour_quantisation(img, k=8):
    """
    Reduce the number of colours in the image using k-means clustering.
    Lower `k` = fewer colours = more "cartoony".
    """
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centres = cv2.kmeans(
        data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    centres = np.uint8(centres)
    quantised = centres[labels.flatten()]
    return quantised.reshape(img.shape)

def cartoonise_clean(img_path, output_path):
    """
    Reads an image from img_path, applies colour quantisation + mild edge detection,
    and writes the cartoonised result to output_path.
    """
    # 1. Load image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not open or find the image at {img_path}")

    # 2. Colour quantisation to flatten colours (adjust k for more/fewer colours)
    quantised_img = colour_quantisation(img, k=8)

    # 3. Slightly blur the quantised image
    blurred_quantised = cv2.medianBlur(quantised_img, 3)

    # 4. Edge detection on the greyscale version of the original
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey_blur = cv2.medianBlur(grey, 5)
    edges = cv2.Canny(grey_blur, threshold1=80, threshold2=150)

    # 5. Morphological “opening” to remove small specks
    kernel = np.ones((3,3), np.uint8)
    edges_cleaned = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)

    # 6. Invert edges so we have black lines on a white background
    edges_inverted = cv2.bitwise_not(edges_cleaned)

    # 7. Convert edges to 3-channel
    edges_colour = cv2.cvtColor(edges_inverted, cv2.COLOR_GRAY2BGR)

    # 8. Combine edges with the quantised image
    cartoon = cv2.bitwise_and(blurred_quantised, edges_colour)

    # 9. Save result
    cv2.imwrite(output_path, cartoon)

###############################################################################
# Main Script: Splits data into train/test & creates "real" + "cartoon" subfolders
###############################################################################

def main(data_dir="data", train_ratio=0.8):
    """
    - data_dir: folder containing your images
    - train_ratio: fraction of images to use for training (0.8 = 80% train, 20% test)
    """

    # 1. Collect all image paths
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    all_images = [
        f for f in os.listdir(data_dir)
        if f.lower().endswith(valid_exts) and os.path.isfile(os.path.join(data_dir, f))
    ]

    if not all_images:
        print(f"No images found in {data_dir} with extensions {valid_exts}")
        return

    # Shuffle for random train/test split
    random.seed(42)  # for reproducibility; remove/modify for purely random
    random.shuffle(all_images)

    # 2. Determine split index
    split_index = int(len(all_images) * train_ratio)
    train_files = all_images[:split_index]
    test_files = all_images[split_index:]

    # 3. Create subfolders: train/real, train/cartoon, test/real, test/cartoon
    train_real_dir = os.path.join(data_dir, "train", "real")
    train_cartoon_dir = os.path.join(data_dir, "train", "cartoon")
    test_real_dir = os.path.join(data_dir, "test", "real")
    test_cartoon_dir = os.path.join(data_dir, "test", "cartoon")

    os.makedirs(train_real_dir, exist_ok=True)
    os.makedirs(train_cartoon_dir, exist_ok=True)
    os.makedirs(test_real_dir, exist_ok=True)
    os.makedirs(test_cartoon_dir, exist_ok=True)

    # 4. Helper function to process a batch of files (copy real, create cartoon)
    def process_batch(file_list, real_dir, cartoon_dir):
        for filename in file_list:
            src_path = os.path.join(data_dir, filename)
            
            # Copy original to "real" folder
            real_dest = os.path.join(real_dir, filename)
            shutil.copy2(src_path, real_dest)

            # Create cartoon version in "cartoon" folder
            cartoon_dest = os.path.join(cartoon_dir, filename)
            cartoonise_clean(real_dest, cartoon_dest)

    # 5. Process train set
    print(f"[INFO] Processing {len(train_files)} images for TRAIN set...")
    process_batch(train_files, train_real_dir, train_cartoon_dir)

    # 6. Process test set
    print(f"[INFO] Processing {len(test_files)} images for TEST set...")
    process_batch(test_files, test_real_dir, test_cartoon_dir)

    print("[INFO] Done! Folder structure created with real & cartoon subfolders.")
    print(f"Train: {len(train_files)} images, Test: {len(test_files)} images")

if __name__ == "__main__":
    # Default usage: just call main() with the default data directory = "data"
    main()
