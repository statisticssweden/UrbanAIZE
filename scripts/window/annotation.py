#!/usr/bin/env python3
from logging import exception
import cv2
import numpy as np
import os
import json

from .window import Window

# ------------------
# Sub window class for handle the annotation of map images
# ---------------------------------------------------------------
class AnnotationSubWindow(Window):
    def __init__(self, name, img, mask, x, y, sz, n) -> None:
        super().__init__("{} - {}".format(name, "Sub window"), img)
        self.fname = name
        self.mask = mask
        self.x, self.y = x, y
        self.sz, self.n = sz, n
        self.padding = (self.sz * self.n) // 2
        self.__data = {}

        # Add mouse click callback
        cv2.setMouseCallback(self.name, self.click_handler, 0)

        # Create list of square image patches
        self.squares = []
        px, py = 0, 0
        for i in range(n * n):
            self.squares.append(self.AnnotationSquare(px, py, sz))
            if (i + 1) % n == 0:
                px, py = 0, py + sz
            else:
                px, py = px + sz, py
        
        # Update the map image
        self.update()

    @property
    def data(self) -> dict:
        return self.__data
        
    # Dispaly map image (and annotation sub images)
    def display(self) -> None:
        super().display()
        cv2.resizeWindow(self.name, (self.sz * self.n, self.sz * self.n))
            
    # Update the map images  
    def update(self) -> None:
        self.reset()  # ...reset the map image to the original image

        # Get indices based on image mask
        idx = np.where(self.mask == 128)
        self.img[idx[0], idx[1], :] = (0, 0, 255)
        self.img = cv2.addWeighted( self.img, 0.25, self.orig_img, 0.75, 0.0)
        idx = np.where(self.mask == 255)
        self.img[idx[0], idx[1], :] = (0, 0, 255)
        self.img = cv2.addWeighted( self.img, 0.4, self.orig_img, 0.6, 0.0)

        # Draw grid pattern
        for square in self.squares:
                cv2.rectangle(self.img, square.top_left, square.bottom_right, (255, 255, 255), 1)

        # Draw clicked grids
        for square in self.squares:
            if square.clicked:
                cv2.rectangle(self.img, square.top_left, square.bottom_right, (0, 0, 255), 1)

        # Display the updated image
        self.display()

    # Clear content of all square images 
    def clear(self):
        for square in self.squares:
            square.clicked = False
        self.__data.clear()
        self.update()

    # -----------------
    # Window event handlers
    # ------------------------
    
    # Click handler
    def click_handler(self, event, x, y, flags, params) -> None:
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_MBUTTONDOWN:
            for square in self.squares:
                if square.contains(x, y):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        square.clicked = True
                        self.__store(square)
                    else:
                        square.clicked = False
                        self.__remove(square)
                    self.update()
                    break
                
    # -------------------
    # Private helper methods
    # ----------------------------                
                
    # Store image patches in data dictionary
    def __store(self, patch):
        name = "{}_{}".format(self.fname , patch.get_name(self.x, self.y, self.padding))
        if name not in self.__data:
            x, y, w, h = patch.rectangle
            self.__data[name] = {
                'image': self.orig_img[y:h, x:w],
                'labels': self.mask[y:h, x:w]        
            }
            print("Stored: {}".format(name))
        
    # Remove image patches from data dictionary
    def __remove(self, patch):
        try:
            name = "{}_{}".format(self.fname , patch.get_name(self.x, self.y, self.padding))
            del self.__data[name]
            print("Removed: {}".format(name))
        except KeyError as e:
            pass
             
    # ----------------------------------
    # Inner class for handle clickable squares (of a map image)
    # -----------------------------------------------------------
    class AnnotationSquare(object):
        def __init__(self, x, y, sz) -> None:
            self.x, self.y = x, y
            self.w, self.h = x + sz, y + sz
            self.__clicked = False
            
        @property
        def top_left(self) -> tuple:
            return (self.x, self.y)
        
        @property
        def bottom_right(self) -> tuple:
            return (self.w, self.h)

        @property
        def rectangle(self) -> tuple:
            return (self.x, self.y, self.w, self.h)
        
        @property
        def clicked(self) -> bool:
            return self.__clicked

        @clicked.setter
        def clicked(self, value) -> None:
            #print("clicked.setter", self.__clicked)
            self.__clicked = value
        
        # Check if coordinates is contianed within the square
        def contains(self, x, y):
            if x >= self.x and x < self.w and y >= self.y and y < self.h:
                return True
            return False

        # Get name including unique x- and y coordinates
        def get_name(self, x, y, padding):
            return "{}_{}".format((x - padding) + self.x, (y - padding) + self.y)
        
# ------------------
# Window class for handle the annotation of map images
# -----------------------------------------------------------
class AnnotationWindow(Window):
    def __init__(self, name, path, img, points, offset, height, thickness = 10, sz = 512, n = 3) -> None:
        super().__init__(name, img)
        self.path = path
        (self.h, self.w) = self.img.shape[:2]
        self.sc = height / self.h
        self.points = points
        self.thickness = thickness
        self.sz, self.n = sz, n

        # Annotation sub windows
        self.sub_window = None
        
        # Move window and add mouse click callback (specific for this class)
        cv2.moveWindow(self.name, 0, 0)
        cv2.setMouseCallback(self.name, self.click_handler, 0)

        # Get offset corner points
        if offset is not None:
            self.__offset = offset
        else:
            self.__offset = {'x': 100, 'y': 100, 'w': self.w - 2 * 100, 'h': self.h - 2 * 100}
            self.__get_corners()

        # Update the map image (to reflect offset corner points)
        self.update()

    @property
    def offset(self) -> dict:
        return self.__offset

    # Close (or destroy) map image window
    def close(self) -> None:
        super().close()
        if self.sub_window is not None:
            self.sub_window.close() 
    
    # Dispaly map image (and annotation sub images)
    def display(self) -> None:
        super().display()
        w, h = int(self.w * self.sc), int(self.h * self.sc)
        cv2.resizeWindow(self.name, (w, h))
        if self.sub_window is not None:
            self.sub_window.display() 
    
    # Update the map images  
    def update(self) -> None:
        self.reset()  # ...reset the map image to the original image

        # Get polygon points
        points = []
        if self.points is not None:
            for point in self.points:
                x = point[0] * self.offset['w'] + self.offset['x']
                y = point[1] * self.offset['h'] + self.offset['y']
                points.append((x, y))

        # Draw polygons (weighted)
        if points:        
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(self.img, [points], (0, 0, 255))
            self.img = cv2.addWeighted( self.img, 0.25, self.orig_img, 0.75, 0.0)
            cv2.polylines( self.img, [points], False, (0, 0, 255), self.thickness)

        # Draw helper corner offsets
        cv2.line( self.img, (self.offset['x'], self.offset['y']), (self.offset['x'] + 100, self.offset['y']), (0, 0, 255), 5)
        cv2.line( self.img, (self.offset['x'], self.offset['y']), (self.offset['x'], self.offset['y'] + 100), (0, 0, 255), 5)
        cv2.line( self.img, (self.offset['x'] + self.offset['w'], self.offset['y'] + self.offset['h']), (self.offset['x'] + self.offset['w'] - 100, self.offset['y'] + self.offset['h']), (0, 0, 255), 5)
        cv2.line( self.img, (self.offset['x'] + self.offset['w'], self.offset['y'] + self.offset['h']), (self.offset['x'] + self.offset['w'], self.offset['y'] + self.offset['h'] - 100), (0, 0, 255), 5)

    # -----------------
    # Window event handlers
    # ------------------------
    
    # Click handler
    def click_handler(self, event, x, y, flags, params):
        if self.sub_window is None:
            if event == cv2.EVENT_LBUTTONDOWN:

                # Get polygon points
                points = []
                if self.points is not None:
                    for point in self.points:
                        px = point[0] * self.offset['w'] + self.offset['x']
                        py = point[1] * self.offset['h'] + self.offset['y']
                        points.append((px, py))

                # Create image mask based on polygon points
                if points:
                    mask = np.zeros((self.h, self.w, 1), dtype = np.uint8)
                    points = np.array(points, dtype=np.int32)
                    cv2.fillPoly(mask, [points], (128))
                    cv2.polylines( mask, [points], False, (255), self.thickness)

                    # Create sub window 
                    padding = (self.sz * self.n) // 2
                    img = self.orig_img[y - padding: y + padding, x - padding: x + padding]
                    mask = mask[y - padding: y + padding, x - padding: x + padding]
                    self.sub_window = AnnotationSubWindow(self.name, img, mask, x, y, self.sz, self.n)
                    self.sub_window.display()
            
    # Handle keybord inputs
    def key_handler(self, key) -> None:
        if self.sub_window is not None:
            if chr(key) in 'SsRrQq':
                if key == ord('R') or key == ord('r'):
                    self.sub_window.clear()
                else:
                    if key == ord('S') or key == ord('s'):
                        self.__save(self.sub_window.data)
                    self.sub_window.close()
                    self.sub_window = None
        else:
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

                # Update the map image
                self.update()
    
    # -------------------
    # Private helper methods
    # ----------------------------
    
    # Get map area corners
    def __get_corners(self, th = 100.):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.001)

        # Detect top left corner
        gray = cv2.cvtColor(self.img[int(self.h * 0.01):int(self.h * 0.1), int(self.w * 0.01):int(self.w * 0.1)],cv2.COLOR_BGR2GRAY)
        dst = cv2.cornerHarris( np.float32(gray), 25, 17, 0.001)
        ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(dst))
        corners = cv2.cornerSubPix( gray, np.float32(centroids), (15,15), (-1,-1), criteria)
        corners = [(self.w * 0.01 + point[0], self.h * 0.01 + point[1]) for point in corners]
        distances = [ np.sqrt((point[0] - self.__ofset['x']) ** 2 + (point[1] - self.__ofset['y']) ** 2 ) for point in corners]
        distances = np.array(distances)
        idx = np.where(distances == np.min(distances))[0]
        if idx and distances[int(idx)] < th: 
            self.__ofset['x'], self.__ofset['y'] = int(corners[int(idx)][0]), int(corners[int(idx)][1])

        # Detect bottom right corner
        gray = cv2.cvtColor(self.img[int(self.h * 0.9):int(self.h * 0.99), int(self.w * 0.9):int(self.w * 0.99)],cv2.COLOR_BGR2GRAY)
        dst = cv2.cornerHarris( np.float32(gray), 25, 17, 0.001)
        ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(dst))
        corners = cv2.cornerSubPix( gray, np.float32(centroids), (15,15), (-1,-1), criteria)
        corners = [(self.w * 0.9 + point[0], self.h * 0.9 + point[1]) for point in corners]
        distances = [ np.sqrt((point[0] - (self.__ofset['x'] + self.__ofset['w'])) ** 2 + (point[1] - (self.__ofset['y'] + self.__ofset['h'])) ** 2 ) for point in corners]
        idx = np.where(distances == np.min(distances))[0]
        if idx and distances[int(idx)] < th: 
            self.__ofset['w'], self.__ofset['h'] = int(corners[int(idx)][0] - self.__ofset['x']), int(corners[int(idx)][1] - self.__ofset['y'])

    # Save imaged and meta data to files
    def __save(self, data):

        # Read meta data file
        fmeta = os.path.join(self.path, 'data', 'meta.json')
        try:
            with open(fmeta, 'r') as f:
                meta_data = json.load(f)
        except FileNotFoundError:
            if not os.path.exists(os.path.join(self.path, 'data')):
                os.makedirs(os.path.join(self.path, 'data'))
            meta_data = {
                'path': os.path.join(self.path, 'data'),
                'pairs': []
            }

        # Iterate data dictonary and save to disc
        for fname, data in data.items():

            # Write image patches to files
            fimage = "{}_image.png".format(fname)
            flabels = "{}_labels.png".format(fname)
            cv2.imwrite(os.path.join(meta_data['path'], fimage), data['image'])
            cv2.imwrite(os.path.join(meta_data['path'], flabels), data['labels'])

            # Add file names to meta data
            meta_data['pairs'].append({'image': fimage, 'labels': flabels})

        # Write meta data file
        with open(fmeta, 'w') as f:
            json.dump(meta_data, f, indent = 2)
