import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=['.idea', '.git'],
    pythonpath=True,
)

# From RF-DETR developers (Roboflow)
import sys

from rfdetr import RFDETRMedium, RFDETRBase
import supervision as sv

import numpy as np
import polars
import os
import glob
import datetime as dt
import shutil
import itertools
import copy
import json

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

RANDOM_SEED = 1729
TEST_SIZE = 0.2

DATA_DIR = os.path.join(root, 'data', 'nfl-health-and-safety-helmet-assignment')
DATASET_DIR = os.path.join(DATA_DIR, 'nfl_helmet_image_dataset')

def move_images(DATA_DIR, ENDZONE_DIR, SIDELINE_DIR):
    os.makedirs(ENDZONE_DIR, exist_ok=True)
    os.makedirs(SIDELINE_DIR, exist_ok=True)
    # Create the directories. We will just have a train and test datasets.
    # First find the split...
    for view, directory in zip(['Endzone', 'Sideline'], [ENDZONE_DIR, SIDELINE_DIR]):
        print(f'Gathering the {view} view...')
        all_og_image_files = glob.glob(os.path.join(DATA_DIR, 'images', f'*{view}*.jpg'))
        # Create a train test split. Make it reproducible...
        train, test = train_test_split(all_og_image_files, test_size=TEST_SIZE, random_state=RANDOM_SEED)
        valid, test = train_test_split(test, test_size=0.5, random_state=RANDOM_SEED)
        print('Training images:', len(train))
        print('Validation images:', len(valid))
        print('Test images:', len(test))
        # Create the train and test directories inside...
        os.makedirs(os.path.join(directory, 'train'), exist_ok=True)
        os.makedirs(os.path.join(directory, 'valid'), exist_ok=True)
        os.makedirs(os.path.join(directory, 'test'), exist_ok=True)
        # Move the files to the sub-directory (/train or /test)
        for dirname, dataset in zip(['train', 'valid', 'test'], [train, valid, test]):
            for file in tqdm(dataset, desc=f'Moving {dirname} files'):
                shutil.move(file, os.path.join(directory, dirname, os.path.basename(file)))


def generate_annotations(DATASET_DIR, image_labels):
    # There is some info that will be the same across the 4 files
    base_annotations = {
        "info": {
            "year": "2025",
            "version": "1",
            "description": "Generated COCO annotations for NFL Helmet detection",
            "contributor": "mughil",
            "url": "",
            "date_created": "2025-10-10T00:00:00+00:00"
        },
        "licenses": [
            {
                "id": 1,
                "url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "name": "Public Domain"
            }
        ],
        "categories": [
            {
                "id": 0,
                "name": "Helmet",
                "supercategory": "none"
            }
        ]
    }
    for view, split in itertools.product(['Endzone', 'Sideline'], ['train', 'valid', 'test']):
        # Copy the base annotations...
        annotations = copy.copy(base_annotations)
        # Filter the dataframe according to the view and split,
        # and SORT according to the image file name.
        # This will come in handy later.
        dataset_view_split = image_labels.filter(
            polars.col('view').eq(view),
            polars.col('split').eq(split)
        ).sort('image', descending=False)
        # Create separate arrays for the images
        images = []
        bounding_boxes = []
        # For each row, keep track of the current image and its ID. As soon as
        # the image changes, update the ID, read it, and grab the
        # width and height...
        image = dataset_view_split[0, 'image']
        image_obj = cv2.imread(os.path.join(DATASET_DIR, view, split, image))
        width, height = image_obj.shape[:2]
        image_id = 0
        images.append({
            'id': image_id,
            'license': 1,
            'file_name': image,
            'width': width,
            'height': height,
            'date_captured': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        })
        # Iterate through rows, keeping track of the box IDs
        for box_id, row in tqdm(enumerate(dataset_view_split.iter_rows(named=True)),
                                desc=f'Generating annotations JSON for {view} ({split})',
                                total=dataset_view_split.shape[0]):
            # If we encounter a new image, then read it in, and append it to the image array.
            if image != row['image']:
                image = row['image']
                image_obj = cv2.imread(os.path.join(DATASET_DIR, view, split, image))
                width, height = image_obj.shape[:2]
                image_id += 1
                images.append({
                    'id': image_id,
                    'license': 1,
                    'file_name': image,
                    'width': width,
                    'height': height,
                    'date_captured': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                })
            # Get the coordinates
            x, y, width, height = row['left'], row['top'], row['width'], row['height']
            # Add a JSON dictionary to the bounding box array
            bounding_boxes.append({
                'id': box_id,
                'image_id': image_id,
                'category_id': 0,  # We only have one category
                'bbox': [x, y, width, height],
                'area': width * height,
                'segmentation': [],
                'iscrowd': 0,
            })
        # Both dictionaries are done. Add to the larger annotations JSON,
        # and save the file in the corresponding directory
        annotations['images'] = images
        annotations['annotations'] = bounding_boxes

        with open(os.path.join(DATASET_DIR, view, split, '_annotations.coco.json'), 'w') as f:
            json.dump(annotations, f, indent=4)

if  __name__ == '__main__':
    ENDZONE_DIR = os.path.join(DATASET_DIR, 'Endzone')
    SIDELINE_DIR = os.path.join(DATASET_DIR, 'Sideline')
    if len(glob.glob(os.path.join(DATASET_DIR, '**/*.jpg'), recursive=True)) == 0:
        move_images(DATA_DIR, ENDZONE_DIR, SIDELINE_DIR)

    # Read the image labels csv
    image_labels = polars.read_csv(os.path.join(DATA_DIR, 'image_labels.csv'))
    # Add columns that tell us what view it is, and whether it falls into the train or test set.
    structured_files = glob.glob(os.path.join(DATASET_DIR, '**/*.jpg'), recursive=True)
    filenames = [os.path.basename(file) for file in structured_files]
    dirnames = [os.path.basename(os.path.dirname(file)) for file in structured_files]
    get_train_test = lambda filename: dirnames[filenames.index(os.path.basename(filename))]
    image_labels = image_labels.with_columns(
        polars.when(polars.col('image').str.contains('Endzone'))
        .then(polars.lit('Endzone'))
        .otherwise(polars.lit('Sideline'))
        .alias('view'),
        polars.col('image').map_elements(get_train_test, return_dtype=polars.String).alias('split')
    )

    generate_annotations(DATASET_DIR, image_labels)

    # Create the model and train it!
    model = RFDETRBase()
    OUTPUT_PATH = os.path.join(DATASET_DIR, 'outputs')
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    model.train(
        dataset_dir=os.path.join(DATASET_DIR, 'Endzone'),
        epochs=2,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=OUTPUT_PATH,
        device='mps'
    )

