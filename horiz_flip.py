#!/usr/bin/env python3
# script creates a copy of file data.csv, names copy data_with_flipped.csv
# reads image filenames and lane information from data.csv
# reads corresponding image, flips it, calculates new lane number
# and saves flipped image and appends its information into data_with_flipped.csv

import os
import json
import csv
from statistics import mean
import cv2
import numpy as np
from pathlib import Path
from shutil import copyfile

csv_input_filename = "data.csv"
csv_output_filename = "data_with_flipped.csv"

def run(csv_input_path, csv_output_path):
    with open(csv_output_path, "a") as csv_output_file:
        csv_writer = csv.writer(csv_output_file)

        with open(csv_input_path) as csv_input_file:
            csv_reader = csv.reader(csv_input_file, delimiter=',')
            line_count = 0

            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    img_input_filename = row[0]
                    nr_lanes = int(row[1])
                    lane = int(row[2])
                    print(f'{line_count} : {img_input_filename} : nr_lanes = {nr_lanes}, lane = {lane}.')
                    img = cv2.imread(img_input_filename)
                    img_flipped = cv2.flip(img, 1)
                    img_filename_split = img_input_filename.split(".")
                    img_output_filename = img_filename_split[0] + '_flipped' + '.' + img_filename_split[1]

                    csv_writer.writerow([Path(img_output_filename), str(nr_lanes), str(nr_lanes + 1 - lane)]) 
                    cv2.imwrite(img_output_filename, img_flipped)
                    line_count += 1

            print(f'Processed {line_count} lines.')


if __name__ == "__main__":    
    copyfile(csv_input_filename, csv_output_filename)
    run(csv_input_filename, csv_output_filename)