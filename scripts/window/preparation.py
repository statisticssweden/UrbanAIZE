#!/usr/bin/env python3
from logging import exception
import cv2
import numpy as np
import os
import json

from .window import Window
                                     
# ------------------
# Window class for handle the preparation of map images
# -----------------------------------------------------------
class PreparationWindow(Window):
    def __init__(self, name, img, area, height) -> None:
        super().__init__(name, img)
        (self.h, self.w) = self.img.shape[:2]
        self.sc = height / self.h
        if area is not None:
            self.__area = area
        else:
            self.__area = {
                "top_left": [0, 0],
                "bottom_left": [0, self.h],
                "top_right": [self.w, 0],
                "bottom_right": [self.w, self.h]    
            }

        # Preparation sub window
        self.sub_window = None
        
        # Move window and add mouse click callback (specific for this class)
        cv2.moveWindow(self.name, 0, 0)
        cv2.setMouseCallback(self.name, self.click_handler, 0)

        # Update the map image (to draw map area)
        self.update()

    @property
    def area(self) -> dict:
        return self.__area

    # Close (or destroy) map image window
    def close(self) -> None:
        super().close()
    
    # Display map image window (and sub image window)
    def display(self) -> None:
        super().display()
        w, h = int(self.w * self.sc), int(self.h * self.sc)
        cv2.resizeWindow(self.name, (w, h))
    
    # Update the map image  
    def update(self) -> None:
        self.reset()  # ...reset the map image to the original image
        
        # Draw line between area corner points 
        cv2.line( self.img, self.area['top_left'], self.area['top_right'], (0, 0, 255), 5)
        cv2.line( self.img, self.area['top_right'], self.area['bottom_right'], (0, 0, 255), 5)
        cv2.line( self.img, self.area['bottom_right'], self.area['bottom_left'], (0, 0, 255), 5)
        cv2.line( self.img, self.area['bottom_left'], self.area['top_left'], (0, 0, 255), 5)

    # -----------------
    # Window event handlers
    # ------------------------
    
    # Click handler
    def click_handler(self, event, x, y, flags, params):
        if self.sub_window is None:
            if event == cv2.EVENT_LBUTTONDOWN:
                point = [x, y]
                closest = { 'point': None, 'distance': float('inf') }
                for key, corner in self.__area.items():
                    dist = np.linalg.norm(np.array(corner) - np.array(point))
                    if dist < closest['distance']:
                        closest['distance'] = dist
                        closest['point'] = key
                self.__area[closest['point']] = point
                self.update()
