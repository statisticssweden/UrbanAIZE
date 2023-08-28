#!/usr/bin/python3

"""" Main annotation script. """

import os
import argparse
import geopandas as gpd
import json
import random
import numpy as np
import cv2
import time
from map_image import MapImage
from window_handler import MapWindow


# Randomly select a map image
def randomly_select(maps):
    if maps['images']:
        map = maps['images'].pop(random.randrange(len(maps['images'])))
        if 'area' in map:
            return MapImage(map)
        return randomly_select(maps)
    return None

# Load random image and create image window
def load_image(map, height):

    # Randomly load map image
    m = randomly_select(maps)
    if m is not None:
        img = m.load(cropped = True)

        # Create and return a map image window
        win = MapWindow(m.fname, img, m.points, m.corners, height = height)
        return win

    return None 

# Update map data file
def update_data(path, fname, corners):
    with open(path, 'r') as f:
        maps = json.load(f)
    for map in maps['images']:
        if map['fname'] == fname:
            map['corners'] = corners
            break
    with open(path, 'w') as f:
        json.dump(maps, f, indent = 2)

# Conditional main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data annotation pipeline.')
    parser.add_argument( '--path', '-p',
                         help = 'path to data folder',
                         default = './data/',
                         type = str )
    parser.add_argument( '--height',
                         help = 'fixed image height',
                         default = 1080,
                         type = int )
    parser.add_argument( '--samples', '-s',
                         help = 'number of samples',
                         default = 1000,
                         type = int )
    parser.add_argument( '--size', '-sz',
                         help = 'image sample size',
                         default = 256,
                         type = int )
    args = parser.parse_args()
    data_file = os.path.join(args.path, 'kartdata.json')

    # Read all data about all map images from JSON file
    try:
        with open(data_file, 'r+') as f:
            maps = json.load(f)

        # Load map image
        img = load_image(maps, args.height)
        # Main display loop
        while(True):

            # Saftey check
            if img is None:
                break

            # Display map image
            img.display()
        
            # Waiting for user input (and handle the input)
            key = cv2.waitKeyEx(1) & 0xFF
            if key == 27 or key == ord('Q') or key == ord('q'): 
                cv2.destroyAllWindows()
                break
            elif key == ord('M') or key == ord('m') or key == ord('N') or key == ord('n'):
                update_data(data_file, img.fname, img.corners)
                if key == ord('M') or key == ord('m'):
                    img.random_sample(args.path, samples = args.samples, sz = args.size)
                    img.close()
                    img = load_image(maps, args.height)
        else:
            img.key_handler(key)

    except FileNotFoundError as e:
        print(f"Error: could not load map images, ensure that that folder $args.path exist and that the the folder contains file 'kartdata.json'.")

