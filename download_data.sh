#!/bin/bash

wget https://www.dropbox.com/s/44twvk07syto4ua/bdd100k_labels_release.zip
unzip bdd100k_labels_release.zip

wget https://www.dropbox.com/s/p6wt442mwk6ggb0/bdd100k_images.zip
unzip bdd100k_images.zip

cat ./bdd100k/labels/bdd100k_labels_images_train.json | jq '[.[] | select(.attributes.scene | contains("highway")) ]'  | jq --compact-output '[.[] | select(.attributes.timeofday | contains("daytime")) ]' > ./bdd100k/labels/daytime_highway_bdd100k_labels_images_train.json
cat ./bdd100k/labels/bdd100k_labels_images_val.json | jq '[.[] | select(.attributes.scene | contains("highway")) ]'  | jq --compact-output '[.[] | select(.attributes.timeofday | contains("daytime")) ]' > ./bdd100k/labels/daytime_highway_bdd100k_labels_images_val.json