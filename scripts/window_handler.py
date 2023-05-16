#!/usr/bin/env python3
from logging import exception
import cv2
from cv2 import imshow
import numpy as np
import os
import sys
import random
import json

# --------------------
# General super class for handle a OpenCV windows
# ------------------------------------------------
class Window:
    def __init__(self, name, img, param = cv2.WINDOW_NORMAL) -> None:
        self.name = name
        self.img = img
        self.orimg = img.copy()
        
        # Create a named window
        cv2.namedWindow(self.fname, param)

    @property
    def fname(self) -> str:
        return self.name

    # Display the window
    def display(self) -> None:
        cv2.imshow(self.name, self.img)

# ------------------
# Class for handle map images
# --------------------------------
class MapWindow(Window):
    def __init__(self, name, img, points, offset, height) -> None:
        super().__init__(name, img)
        (self.h, self.w) = self.img.shape[:2]
        self.sc = height / self.h
        self.points = points

        # List of annotation windows
        self.windows = []
        
        # Move window and add mouse callback (spesific for this class)
        cv2.moveWindow(self.name, 0, 0)
        cv2.setMouseCallback(self.name, self.click, 0)

        # Get offset corner points and draw image points
        if offset is not None:
            self.offset = offset
        else:
            self.offset = {'x': 100, 'y': 100, 'w': self.w - 2 * 100, 'h': self.h - 2 * 100}
            self.get_corners()
        self.draw()

    @property
    def corners(self) -> dict:
        return self.offset

    # Handle keybord inputs
    def key_handler(self, key) -> None:
        if chr(key) in 'WwSsAaDdIiKkJjLl':
            if key == ord('W'):   self.offset['y'] = self.offset['y'] - 10
            elif key == ord('w'): self.offset['y'] = self.offset['y'] - 1
            elif key == ord('S'): self.offset['y'] = self.offset['y'] + 10
            elif key == ord('s'): self.offset['y'] = self.offset['y'] + 1
            elif key == ord('A'): self.offset['x'] = self.offset['x'] - 10
            elif key == ord('a'): self.offset['x'] = self.offset['x'] - 1
            elif key == ord('D'): self.offset['x'] = self.offset['x'] + 10
            elif key == ord('d'): self.offset['x'] = self.offset['x'] + 1
            elif key == ord('J'): self.offset['w'] = self.offset['w'] - 10
            elif key == ord('j'): self.offset['w'] = self.offset['w'] - 1
            elif key == ord('L'): self.offset['w'] = self.offset['w'] + 10
            elif key == ord('l'): self.offset['w'] = self.offset['w'] + 1
            elif key == ord('I'): self.offset['h'] = self.offset['h'] - 10
            elif key == ord('i'): self.offset['h'] = self.offset['h'] - 1
            elif key == ord('K'): self.offset['h'] = self.offset['h'] + 10
            elif key == ord('k'): self.offset['h'] = self.offset['h'] + 1
            self.draw()

    # Get map area corners
    def get_corners(self, th = 100.):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.001)

        # Detect top left corner
        gray = cv2.cvtColor(self.img[int(self.h * 0.01):int(self.h * 0.1), int(self.w * 0.01):int(self.w * 0.1)],cv2.COLOR_BGR2GRAY)
        dst = cv2.cornerHarris( np.float32(gray), 25, 17, 0.001)
        ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(dst))
        corners = cv2.cornerSubPix( gray, np.float32(centroids), (15,15), (-1,-1), criteria)
        corners = [(self.w * 0.01 + point[0], self.h * 0.01 + point[1]) for point in corners]
        distances = [ np.sqrt((point[0] - self.offset['x']) ** 2 + (point[1] - self.offset['y']) ** 2 ) for point in corners]
        distances = np.array(distances)
        idx = np.where(distances == np.min(distances))[0]
        if idx and distances[int(idx)] < th: 
            self.offset['x'], self.offset['y'] = int(corners[int(idx)][0]), int(corners[int(idx)][1])

        # Detect bottom right corner
        gray = cv2.cvtColor(self.img[int(self.h * 0.9):int(self.h * 0.99), int(self.w * 0.9):int(self.w * 0.99)],cv2.COLOR_BGR2GRAY)
        dst = cv2.cornerHarris( np.float32(gray), 25, 17, 0.001)
        ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(dst))
        corners = cv2.cornerSubPix( gray, np.float32(centroids), (15,15), (-1,-1), criteria)
        corners = [(self.w * 0.9 + point[0], self.h * 0.9 + point[1]) for point in corners]
        distances = [ np.sqrt((point[0] - (self.offset['x'] + self.offset['w'])) ** 2 + (point[1] - (self.offset['y'] + self.offset['h'])) ** 2 ) for point in corners]
        idx = np.where(distances == np.min(distances))[0]
        if idx and distances[int(idx)] < th: 
            self.offset['w'], self.offset['h'] = int(corners[int(idx)][0] - self.offset['x']), int(corners[int(idx)][1] - self.offset['y'])
    
    # Dispaly map image (and annotation sub images)
    def display(self) -> None:
        super().display()
        w, h = int(self.w * self.sc), int(self.h * self.sc)
        cv2.resizeWindow(self.name, (w, h))
        if self.windows:
            cv2.imshow("Sub image", self.windows[0])

    # Draw map  
    def draw(self) -> None:
        
        # Get polygon points
        points = []
        if self.points is not None:
            for point in self.points:
                x = point[0] * self.offset['w'] + self.offset['x']
                y = point[1] * self.offset['h'] + self.offset['y']
                points.append((x, y))

        # Draw polygons (weighted)
        self.img = self.orimg.copy()
        if points:        
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(self.img, [points], (0, 0, 255))
            self.img = cv2.addWeighted( self.img, 0.25, self.orimg, 0.75, 0.0)
            cv2.polylines( self.img, [points], False, (0, 0, 255), 10)

        # Draw helper corner offsets
        cv2.line( self.img, (self.offset['x'], self.offset['y']), (self.offset['x'] + 100, self.offset['y']), (0, 0, 255), 5)
        cv2.line( self.img, (self.offset['x'], self.offset['y']), (self.offset['x'], self.offset['y'] + 100), (0, 0, 255), 5)
        cv2.line( self.img, (self.offset['x'] + self.offset['w'], self.offset['y'] + self.offset['h']), (self.offset['x'] + self.offset['w'] - 100, self.offset['y'] + self.offset['h']), (0, 0, 255), 5)
        cv2.line( self.img, (self.offset['x'] + self.offset['w'], self.offset['y'] + self.offset['h']), (self.offset['x'] + self.offset['w'], self.offset['y'] + self.offset['h'] - 100), (0, 0, 255), 5)

    # Close (or destroy) map window
    def close(self):
        cv2.destroyWindow(self.fname)
 
    # Click handler
    def click(self, event, x, y, flags, params):
        pad, sz = 256, 512
        if event == cv2.EVENT_LBUTTONDOWN:
            subimg = self.img[y - pad: y + pad, x - pad: x + pad]
            self.windows.clear()
            self.windows.append(subimg)
            #cv2.imshow("Sub image", subimg)
            #cv2.wait

    # Random sample and save pairs of traning images
    def random_sample(self, path, samples = 1000, sz = 512, ratio = 0.05):

        # Get polygon points
        points = []
        if self.points is not None:
            for point in self.points:
                x = point[0] * self.offset['w'] + self.offset['x']
                y = point[1] * self.offset['h'] + self.offset['y']
                points.append((x, y))

        # Draw polygons (weighted)
        if points:        
            mask = np.zeros((self.h, self.w, 1), dtype = np.uint8)
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [points], (255))

            # Read meta data file
            meta_file = os.path.join(path, 'data', 'meta.json')
            try:
                with open(meta_file, 'r') as f:
                    data = json.load(f)
            except FileNotFoundError:
                if not os.path.exists(os.path.join(path, 'data')):
                    os.makedirs(os.path.join(path, 'data'))
                data = {'pairs': []}

            # Randomly sample image patches
            fname = self.fname.split('.')[0]
            counter = {'black': 0, 'white': 0}
            for _ in range(samples):
                x, y = random.randrange(self.offset['x'], self.offset['w'] - sz), random.randrange(self.offset['y'], self.offset['h'] - sz)
                sub_mask = mask[y: y + sz, x: x + sz]
                cntNoneZeros = cv2.countNonZero(sub_mask)
                if cntNoneZeros == 0:
                    if counter['black'] > samples * ratio:
                        continue
                    counter['black'] += 1
                elif cntNoneZeros == (sz * sz):
                    if counter['white'] > samples * ratio:
                        continue
                    counter['white'] += 1
                sub_img = self.orimg[y: y + sz, x: x + sz]

                # Write image patches to files
                fimg = fname + '_' + str(x) + '_' + str(y) + '_img.png'
                fmask = fname + '_' + str(x) + '_' + str(y) + '_mask.png'
                cv2.imwrite(os.path.join(path, 'data', fimg), sub_img)
                cv2.imwrite(os.path.join(path, 'data', fmask), sub_mask)

                # Add file names to meta data
                data['pairs'].append({'img': fimg, 'mask': fmask})

            # Write meta data file
            with open(meta_file, 'w') as f:
                data = json.dump(data, f, indent = 2)


    