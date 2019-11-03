#!/usr/bin/env python3
# This script extracts val_data.csv file from
# filenames with the following template
# filename_NRLANES_LANENRFROMCENTERLINE
# e.g.
# image_044998-3-1.jpg
# Will become 3 lane road driving 
# on lane close to centerline


from glob import glob
import os

VAL_DATASET_PATH='val_data'
files=glob(os.path.join(VAL_DATASET_PATH,'*-*-*.jpg'))

f=open("val_data.csv", 'w')
f.write('img_path,nr_lanes,lane\n')
for path in files:
  filename = os.path.basename(path)
  parts=filename.replace('.jpg','').split('-')
  nr_lanes=parts[1]
  lane_nr=parts[2]
  f.write("%s,%s,%s\n" % (path, nr_lanes, lane_nr))
