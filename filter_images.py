#!/usr/bin/env python3

import os
import json
import csv
from statistics import mean

PATH = "./bdd100k/labels/"
INPUT = ["daytime_highway_bdd100k_labels_images_val.json","daytime_highway_bdd100k_labels_images_train.json"]
OUTPUT = "data.csv"


class Lane:
    def __init__(self, polygon, json_index, current):
        self.json_index = json_index
        self.current = current
        self.polygon = polygon
        self.y = mean(y for [x,y] in self.polygon)


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
        totalLanes = 0
        currentLane = 0
        lanes = []

        for label in element["labels"]:
            if label["category"] == "drivable area":
                totalLanes += 1
                lanes.append(Lane(
                    polygon=label["poly2d"][0]["vertices"],
                    current=label["attributes"]["areaType"] == "direct",
                    json_index=totalLanes))
                if label["attributes"]["areaType"] == "direct":
                    # can we assume that lanes ordered correctly? probably not!
                    currentLane = totalLanes

        # avoid cases where we do not have drivable maps
        if totalLanes != 0 and currentLane != 0:
            currentLaneByY = next(i for i,v in enumerate(lanes) if v.current)
            writer.writerow([image_name, totalLanes, currentLane, currentLaneByY])

    json_file.close()
    out_file.close()

if __name__ == "__main__":
    if os.path.exists(OUTPUT):
        os.remove(OUTPUT)

    run(os.path.join(PATH, INPUT[0]), "w")
    run(os.path.join(PATH, INPUT[1]), "a")