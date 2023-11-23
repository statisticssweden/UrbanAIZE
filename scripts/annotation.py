#!/usr/bin/python3
import os
import argparse
import json
import random
import numpy as np
import cv2
from map.image import Image
from map.utils import getModifiedName
from window.annotation import AnnotationWindow

'''
Main annotation script. 
'''

# Randomly select a map
def randomly_select(maps):
    if maps['images']:
        map = maps['images'].pop(random.randrange(len(maps['images'])))
        if 'area' in map and 'points' in map:
            return Image(map)
        return randomly_select(maps)
    return None

# Load random map image and create image window
def load_image(map, path, height, sz):

    # Randomly load map image
    map = randomly_select(maps)
    if map is not None:

        # Create and return an annotation window
        img = map.load(cropped = True)
        print("Loaded map image: {}".format(map.fname))
        name = getModifiedName(map.fname).split('.')[0]
        window = AnnotationWindow(name, path, img, map.points, map.corners, height = height, sz = sz)

        return map.fname, window

    return None 

# Update map data file
def update_data(data_file, fname, corners):
    with open(data_file, 'r') as f:
        maps = json.load(f)
    for map in maps['images']:
        if map['fname'] == fname:
            map['corners'] = corners
    with open(data_file, 'w', encoding = 'utf8') as f:
        json.dump(maps, f, indent = 2, ensure_ascii=False)

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
    parser.add_argument( '--size', '-sz',
                         help = 'sample image patch size',
                         default = 256,
                         type = int )
    args = parser.parse_args()
    data_file = os.path.join(args.path, 'kartdata.json')

    # Read all data about all map images from JSON file
    try:
        with open(data_file, 'r+') as f:
            maps = json.load(f)

        # Load map annotation window
        fname, window = load_image(maps, args.path, args.height, args.size)
        
        # Main display loop
        while(True):

            # Saftey check
            if window is None:
                break

            # Display map annotation window
            window.display()
        
            # Waiting for user input (and handle the input)
            key = cv2.waitKeyEx(1) & 0xFF
            if key == 27: 
                cv2.destroyAllWindows()
                break
            elif key == ord('M') or key == ord('m') or key == ord('N') or key == ord('n'):
                update_data(data_file, fname, window.offset)
                window.close()
                fname, window = load_image(maps, args.path, args.height, args.size)
            else:
                window.key_handler(key)

    except FileNotFoundError as e:
        print("Error: Could not open data file 'kartdata.json' as result of intial 'preperation.py'.")
        print("Run inital preperation and confirm that the file 'kartdata.json' is located in path: {}".format(args.path))
    except TypeError as e:
        print("No more map images to annotate... ")
        
        

