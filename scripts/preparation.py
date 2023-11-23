#!/usr/bin/python3
import argparse
import pandas as pd
import json
import os, os.path, shutil
from map.utils import getModifiedName, sr99Tort90
from map.image import Image
from map.area import detectMapArea, findBestSquare, squareness
from tqdm import tqdm
import cv2
import geopandas as gpd
import math
import numpy as np

''' 
Main preperation script. 
'''

# Check if map images already exist in collection
def existing_map_image(maps, img) -> bool:
    for map in maps['images']:
        if map.fname == img.fname:
            return True
    return False

# Load existing data file (if exist)
def load_data(data_file) -> dict:
    maps = { 'images': [] }
    if os.path.isfile(data_file):
        with open(data_file, 'r') as f:
            data = json.load(f)
        for map in data['images']:
            img = Image(map)
            if os.path.isfile(img.filepath()):
                print("Loaded map image: {}".format(img.fname))
                maps['images'].append(img)
    return maps
        
# Conditional main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preperation pipeline.')
    parser.add_argument( '--display', '-d', 
                         action = 'store_true', default=False,
                         help = 'display a result image')
    parser.add_argument( '--length', '-l',
                         help = 'map line length',
                         default = '2500',
                         type = int )
    parser.add_argument( '--path', '-p',
                         help = 'path to data folder',
                         default = './data/',
                         type = str )
    parser.add_argument( '--remove', '-r', 
                         action = 'store_true', default=False,
                         help = 'remove existing prepared folders')
    parser.add_argument( '--year', '-y',
                         help = 'year of interest',
                         default = '1980',
                         type = str )
    parser.add_argument( '--approach', '-a',
                         help = 'preperation approach (auto, manual, or single)',
                         default = 'single',
                         type = str )
    args = parser.parse_args()
    data_file = os.path.join(args.path, 'kartdata.json')
    directories = []
    for dir in ['data']:
        directories.append(os.path.join(args.path, dir))

    # Remove content of exesting directories
    if args.remove: 
        for dir in directories:
            if os.path.exists(dir) and args.remove:
                shutil.rmtree(dir)

    '''
    # Replace all the special characters in file and folder names (e.g. ÅÄÖ).
    for path, subdir, files in os.walk(args.path, topdown=False):
        for file in files:      
            os.rename( os.path.join(path, file), os.path.join(path, getModifiedName(file)))
        for folder in subdir:
            os.rename( os.path.join(path, folder), os.path.join(path, getModifiedName(folder)))
    '''
    
    # Create the structure of directories
    for dir in directories:
        if not os.path.exists(dir):
            os.mkdir(dir)

    # Load all map images and add new map images
    maps = load_data(data_file)
    for root, dirs, files in os.walk(args.path):
        if root in directories:
            continue
        for file in tqdm(files):
            img = Image(root, file)
            if img.valid() and not existing_map_image(maps, img):
                print("Added map image: {}".format(img.fname))
                maps['images'].append(img)

    # Open geodata database (from year 1980, default)
    try:
        fname = 'Tatorter_1980_2020.gpkg'
        gdf = gpd.read_file(os.path.join(args.path, fname), layer=f'To{args.year}_SR99TM')
        gdf['points'] = gdf.apply(lambda p: p['geometry'].exterior.coords, axis=1)
        print(gdf)
        for points in gdf['points']:
            if points:
                y, x = sr99Tort90(points[0][1], points[0][0]) # translate to rt90
                y = np.floor((y / 1000) / 5) * 5 # floor to nearest 5. if 6349.3492342934 -> 6345
                x = np.floor((x / 1000) / 5) * 5 # floor to nearest 5. if 1578.3492342984 -> 1575
                for map in maps['images']:
                    if map.checkRT90coordinates(x, y):
                        map.points = [sr99Tort90(p[1], p[0]) for p in points]
    except:
        print("Error: Could not open database file 'Tatorter_1980_2020.gpkg'.")
        print("Make sure the database file is located in path: {}".format(args.path))

    # Detect the map (square) image of interest
    for map in tqdm(maps['images']):

        # Load map image
        print(f"Map image: {map.fname}")
        img = map.load()

        # Run single map area detection
        if args.approach == 'single':
            try:
                area = detectMapArea(img = img, length = args.length)
                if area is not None:
                    if map.area is None:
                        map.area = area
                    else:
                        map.area = findBestSquare([area, map.area])
            except Exception as e:
                print(f"Error: {e}")

        # Run automatic map area detection
        elif args.approach == 'auto':
            for length in tqdm(range(2500, 3500, 100)):
                try:
                    area = detectMapArea(img = img, length = length)
                    if area is not None:
                        if map.area is None:
                            map.area = area
                        else:
                            map.area = findBestSquare([area, map.area])
                except Exception as e:
                    print(f"Error: {e}")

        # Print area and score
        if map.area is not None:
            print(f"Best area: {map.area}")
            print(f"Best score: {squareness(map.area)}")
                
        print("---------------------------")
                    

    # Save meta data to file
    with open(data_file, 'w+', encoding = 'utf8') as f:
        json.dump(maps, f, indent = 2, ensure_ascii=False)

    '''
    # Crop all map images of category 'Ekonomisk' 
    img = None
    for map in map_categories['Ekonomisk']:
        img = map.crop_image(folder = os.path.join(args.path, 'squares'), length = args.length, display = args.display)
        print(map.coordinates())
    
    # Display one resulting image
    if img is not None:
        scale_percent = 25 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow('Hough Lines',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    '''

    
