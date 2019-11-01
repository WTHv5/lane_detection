#!/usr/bin/env python3

import os
import json
import csv
from statistics import mean
import cv2
import numpy as np

PATH = "./bdd100k/labels/"
INPUT = ["daytime_highway_bdd100k_labels_images_val.json","daytime_highway_bdd100k_labels_images_train.json"]
OUTPUT = "data.csv"


class Lane:

    def __init__(self, polygon, json_index, current):
        self.json_index = json_index
        self.current = current
        self.polygon = polygon
        self.y = mean(y for [x,y] in self.polygon)
        self.x = mean(x for [x,y] in self.polygon)


def run(path, type):
    json_file = open(path)
    data = json.load(json_file)
    
    out_file = open(OUTPUT, type)
    writer = csv.writer(out_file)

    for element in data:
        attributes = element["attributes"]
        
        # filter only daytime
        if attributes["timeofday"] != "daytime":
            continue
        
        # filter only highway
        if attributes["scene"] != "highway":
            continue
            
        image_name = element["name"]

        image_path = None
        if os.path.exists(os.path.join("bdd100k/images/100k/train", image_name)):
            image_path = os.path.join("bdd100k/images/100k/train", image_name)
        elif os.path.exists(os.path.join("bdd100k/images/100k/test", image_name)):
            image_path = os.path.join("bdd100k/images/100k/test", image_name)
        elif os.path.exists(os.path.join("bdd100k/images/100k/val", image_name)):
            image_path = os.path.join("bdd100k/images/100k/val", image_name)
        else:
            raise("unknown path of the file")
                
        totalLanes = 0
        currentLane = 0

        lanes = []
        show_images = False
        if show_images:
            img = cv2.imread(image_path)

        for label in element["labels"]:
            if label["category"] == "drivable area":
                totalLanes += 1
                lane = Lane(
                    polygon=label["poly2d"][0]["vertices"],
                    current=label["attributes"]["areaType"] == "direct",
                    json_index=totalLanes)
                lanes.append(lane)
                if show_images:
                    poly = np.asarray([label["poly2d"][0]["vertices"]], dtype='int32')
                    cv2.polylines(img, [poly], True, (0, 255, 0) if label["attributes"]["areaType"] == "direct" else (0, 0, 255), 7)
                    cv2.circle(img, (int(lane.x), int(lane.y)), 7, (255, 0, 0), 7)

                if label["attributes"]["areaType"] == "direct":
                    # can we assume that lanes ordered correctly? probably not!
                    currentLane = totalLanes

        lanes.sort(key=lambda l: l.x)
        # avoid cases where we do not have drivable maps
        if totalLanes != 0 and currentLane != 0:
            currentLaneByY = 1
            for lane in lanes:
                if not lane.current:
                    currentLaneByY += 1
                else:
                    break
            
            # currentLaneByY = next(i for i,v in enumerate(lanes) if v.current) + 1
            writer.writerow([image_path, totalLanes, currentLaneByY])
            if show_images:

                font                   = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,100)
                fontScale              = 1
                fontColor              = (255,255,255)
                lineType               = 2

                cv2.putText(img, "{}/{}/{}".format(totalLanes, currentLane, currentLaneByY),
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)

                cv2.imshow("window", img)
                cv2.waitKey(0)

    json_file.close()
    out_file.close()


if __name__ == "__main__":
    if os.path.exists(OUTPUT):
        os.remove(OUTPUT)
    
    run(os.path.join(PATH, INPUT[0]), "w")
    run(os.path.join(PATH, INPUT[1]), "a")
