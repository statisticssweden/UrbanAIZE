import os
import json
import cv2
import numpy as np
from collections import defaultdict

from tqdm import tqdm

from .utils import getRT90

class Image(dict):
    '''
    This class is used to load map images and to further sort them into folders.

    @param path: path to image
    @param categories: Dict of categories. Keys should be Ekonomisk, Fastighet and Annat.
    '''
    def __init__(self, *args) -> None:
        if len(args) == 2:
            dict.__init__(self, path = args[0], fname = args[1])
            self['category'] = self.__get_category()
            if self['category'] == 'Ekonomisk':
                x, y = self.__get_coordinates()
                self['rt90'] = {'x': x, 'y': y}
        elif len(args) == 1 and type(args[0]) == dict:
            dict.__init__(self, path = args[0]['path'], fname = args[0]['fname'], category = args[0]['category'])
            if 'rt90' in args[0]:
                self['rt90'] = args[0]['rt90']
            if 'area' in args[0]:
                self['area'] = args[0]['area']
            if 'corners' in args[0]:
                self['corners'] = args[0]['corners']
            if 'points' in args[0]:
                self['points'] = args[0]['points']

    @property
    def area(self) -> dict:
        return self.get('area', None)

    @area.setter
    def area(self, rect) -> None:
        self['area'] = {}
        for key, value in rect.items():
            self['area'][key] = value

    @property
    def corners(self) -> dict:
        return self.get('corners', None)

    @property
    def fname(self) -> str:
        return self.get('fname', None)

    @property
    def path(self) -> str:
        return self.get('path', None)
    
    @property
    def points(self) -> list:
        if 'points' not in self:
            return None
        points = []
        for p in self.get('points'):
            x, y = p[1], p[0]
            x = (x - self['rt90']['x'] * 1000.) / 5000.
            y = (y - self['rt90']['y'] * 1000.) / 5000.
            y = 1. - y
            points.append((x, y))
        return points
    
    @points.setter
    def points(self, lst) -> None:
        self['points'] = []
        for p in lst:
            self['points'].append(p)
   
    # Get full name (i.e., path and file name)
    def filepath(self, xtra: str = "") -> str:
        return os.path.join(self.path, xtra, self.fname)

    # Load image
    def load(self, cropped = False) -> np.array:
        '''
        Loads image regardless format
        '''
        stream = open(self.filepath(), "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

        # Load cropeed and rotated map image
        if cropped and 'area' in self:

            # Get bounding area size
            src = np.array(list(self.area.values()), dtype = "float32")
            src = np.array(sorted(src, key=lambda x: x[1]))
            rect = cv2.minAreaRect(src)
            box = cv2.boxPoints(rect) # make a box with the rect
            box = np.int0(box)
            width, height = int(rect[1][0]), int(rect[1][1])

            # Warp image by perspective transformation
            dst = np.array([
                [0, 0], 
                [width, 0],
                [0, height],
                [width, height]
                ],
                dtype = "float32"
            )
            M = cv2.getPerspectiveTransform(src, dst)
            img = cv2.warpPerspective(img, M, (width, height))

        return img

    # Check valid image file extension
    def valid(self) -> bool:
        if self['category'] == 'Ekonomisk' or self['category'] == 'Fastighet': 
            for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']: # Possible file extension
                if ext in self.fname:
                    return True
        return False
            
    # Check if given coordinates are eqauls to the RT90 coordinates given by the file name 
    def checkRT90coordinates(self, x, y) -> bool:
        if self['category'] == 'Ekonomisk':
            _x, _y = self.__get_coordinates()
            return x == _x and y == _y
        return False
    
        
    # ------------------------
    # Private helper functions
    # ---------------------------

    # Get the map category based on the file name 
    def __get_category(self) -> str:
        '''
        Checks if filename can reveal the map category by looking at the last numbers.
        '''
        check = self.__get_name()
        if(check[-7:].isnumeric()):   # 7 ending digits means "Fastighetskarta"
            return 'Fastighet'
        elif(check[-5:].isnumeric()): # 5 ending digits means "Ekonomisk karta"
            return 'Ekonomisk'
        return 'Annat' 

    # Get RT90 coordinates
    def __get_coordinates(self) -> [int, int]:
        digits = self.__get_digits()
        try:
            if(digits.isnumeric()): # Could need "txt.isnumeric() and txt[0]=='1'" But shouldn't be a problem
                x = float(digits[2] + '.' + digits[4])
                y = float(digits[:2] + '.' + digits[3])
                return getRT90(x, y)
        except Exception as e:
            return None, None
    
    # Get file name without file extension
    def __get_name(self) -> str:
        for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']: # Possible file extension
            if ext in self.fname:
                return self.fname[:self.fname.find(ext)]
        return self.fname
    
    # Get last digits of file name (as string)
    def __get_digits(self, idx: int = 5) -> str:
        digits = self.__get_name()
        return digits[-idx:]

    # Draw groups of lines, each defined by point cooridnates
    def __draw_lines(self, img, lines, color = (0, 0, 255)) -> None:
        '''
        Draw P lines on an image.
        '''
        for line in lines:
            for rho, theta in line:
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a * rho, b*rho
                x1 = int(x0 + img.shape[1] * (-b))
                y1 = int(y0 + img.shape[0] * (a))
                x2 = int(x0 - img.shape[1] * (-b))
                y2 = int(y0 - img.shape[0] * (a))
                cv2.line(img, (x1,y1), (x2,y2), color, 1)



