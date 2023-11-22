#!/usr/bin/python3
import argparse
import pandas as pd
import json
import os
import shutil
from map.utils import getModifiedName, sr99Tort90
from map.image import Image
from tqdm import tqdm
import cv2
import geopandas as gpd
import math
import numpy as np

''' 
Main preperation script. 
'''

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
    args = parser.parse_args()
    directories = []
    for dir in ['data']:
        directories.append(os.path.join(args.path, dir))

    # Remove content of exesting directories
    if args.remove: 
        for dir in directories:
            if os.path.exists(dir) and args.remove:
                shutil.rmtree(dir)
    
    # Replace all the special characters in file and folder names (e.g. ÅÄÖ).
    for path, subdir, files in os.walk(args.path, topdown=False):
        for file in files:      
            os.rename( os.path.join(path, file), os.path.join(path, getModifiedName(file)))
        for folder in subdir:
            os.rename( os.path.join(path, folder), os.path.join(path, getModifiedName(folder)))

    # Create the structure of directories
    for dir in directories:
        if not os.path.exists(dir):
            os.mkdir(dir)

    # Process all map images (including extracting map categories)
    maps = {
        'images': [],
    }
    for root, dirs, files in os.walk(args.path):
        if root in directories:
            continue
        for file in tqdm(files):
            img = Image(root, file)
            if not img.valid():
                continue
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
                    if map.check_coordinates(x, y):
                        map.points = [sr99Tort90(p[1], p[0]) for p in points]
    except:
        print("Error: Could not open database file 'Tatorter_1980_2020.gpkg', make sure the database file is located in path: {}".format(args.path) )

    # Detect the map (square) image of interest
    for map in tqdm(maps['images']):
        try:
            map.detect_map_area(length = args.length)
        except:
            pass

        # Save meta data to file
        with open(os.path.join(args.path, 'kartdata.json'), 'w+', encoding = 'utf8') as f:
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

    
