import json
import os
import sys

# Assume --dataset is the argument
dataset_name = None
if "--dataset" in sys.argv:
    dataset_index = sys.argv.index("--dataset") + 1
    dataset_name = sys.argv[dataset_index]

print("DATASET NAME CONST:", dataset_name)
restaurant_aspect_cate_list = ['essen', 'service', 'gesamteindruck', 'ambiente', 'preis']

POLARITY_MAPPING_POL_TO_TERM = {"negative": "schlecht", "neutral": "okay", "positive": "gut"}

cate_list = {
    dataset_name: restaurant_aspect_cate_list
}

task_data_list = {
    "tasd": [dataset_name]
}

force_words = {
    'tasd': {
        dataset_name: restaurant_aspect_cate_list + list(POLARITY_MAPPING_POL_TO_TERM.values()) + ['[SSEP]']
    }
}