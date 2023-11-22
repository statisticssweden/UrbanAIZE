import json
import cv2
import numpy as np
import os
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
            self['category'] = self.category()
            if self['category'] == 'Ekonomisk':
                x, y = self.get_coordinates()
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
    def path(self) -> str:
        return self.get('path', None)

    @property
    def fname(self) -> str:
        return self.get('fname', None)
    
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
    
    @property
    def area(self) -> dict:
        return self.get('area')

    @property
    def corners(self) -> dict:
        return self.get('corners', None)

    @points.setter
    def points(self, lst):
        self['points'] = []
        for p in lst:
            self['points'].append(p)
   
    # Get full name (i.e., path and file name)
    def filepath(self, xtra: str = "") -> str:
        return os.path.join(self.path, xtra, self.fname)
    
    # Get the map category based on the file name 
    def category(self) -> str:
        '''
        Checks if filename can reveal the map category by looking at the last numbers.
        '''
        check = self.__get_name()
        if(check[-7:].isnumeric()):   # 7 ending numbers means "Fastighetskarta"
            return 'Fastighet'
        elif(check[-5:].isnumeric()): # 5 ending numbers means "Ekonomisk karta"
            return 'Ekonomisk'
        return 'Annat' 


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
    
    def valid(self) -> bool:
        for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']: # Possible file extension
            if ext in self.fname:
                return True
        return False
    
    # Get RT90 coordinates
    def get_coordinates(self) -> [int, int]:
        digits = self.__get_digits()
        try:
            if(digits.isnumeric()): # Could need "txt.isnumeric() and txt[0]=='1'" But shouldn't be a problem
                x = float(digits[2] + '.' + digits[4])
                y = float(digits[:2] + '.' + digits[3])
                return getRT90(x, y)
        except Exception as e:
            return None, None
        
    # Check RT90 coordinates
    def check_coordinates(self, x, y) -> bool:
        if self['category'] == 'Ekonomisk':
            _x, _y = self.get_coordinates()
            return x == _x and y == _y
        return False
    
    # Detect the actual map area in the image
    def detect_map_area(self, votes: int = 400, length: int = 1000, gap: int = 10, display: bool = False) -> np.array:
        print(self.filepath())
        # Load original image and apply edge detection
        img = self.load()
        edges = self.__edge_detection(img)

        # Finds linear edges (longer than 800 pixels)
        #lines = cv2.HoughLinesP(edges, 1, np.pi / 180, votes, minLineLength = length, maxLineGap = gap)
        lines = cv2.HoughLines(edges, 1, np.pi/180, length)

        # Cluster line angles into 2 groups (vertical and horizontal)
        if lines is not None:
            segmented = self.__segment_by_angle_kmeans(lines)

            # Find the intersections of each vertical line with each horizontal line
            intersections = self.__segmented_intersections(segmented)
            if intersections:
                points = self.__filter_intersections(intersections, img.shape[0], img.shape[1])

                # Check if interesction points form an (almost) perfect square
                print(points)
                if self.__is_square(points):
                    self['area'] = {}
                    for key, value in points.items():
                        self['area'][key] = (int(value[0]), int(value[1])) 
        '''
        # Get bounding area size
        src = np.array(list(points.values()), dtype = "float32")
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
        cropped = cv2.warpPerspective(img, M, (width, height))

        # Save resutling cropped image
        cv2.imwrite(self.filepath(folder), cropped)

        # Return result for display purposes
        if display:
            self.__draw_lines(img, segmented[0], (0, 0, 255))
            self.__draw_lines(img, segmented[1], (0, 255, 0))
            for pt in points.values():
                cv2.circle(img, pt, 3, (255, 0, 0), -1)
            return img
        return None
        '''
        
    '''
    Private helper functions
    '''
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

    '''
    Private image processing functions
    '''
    # Get all image edges
    def __edge_detection(self, img: np.ndarray, sz: int = 5) -> np.ndarray:

        # Canny Edge Detection (with Gaussian blur for better detection)
        edges = cv2.Canny( image = cv2.GaussianBlur(img, (sz,sz), 0), 
                           threshold1 = 50, 
                           threshold2 = 230, 
                           apertureSize = 3, 
                           L2gradient = True )

        # Postprocess the detected egdes with dilation filter
        kernel = np.ones((sz, sz),np.float32)
        return cv2.dilate(edges, kernel, iterations=1)


    def __segment_by_angle_kmeans(self, lines, k=2, **kwargs):
        """Groups lines based on angle with k-means.

        Uses k-means on the coordinates of the angle on the unit circle 
        to segment `k` angles inside `lines`.
        """

        # Define criteria = (type, max_iter, epsilon)
        default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
        criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
        flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
        attempts = kwargs.get('attempts', 1)

        # returns angles in [0, pi] in radians
        angles = np.array([line[0][1] for line in lines])
        # multiply the angles by two and find coordinates of that angle
        pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                        for angle in angles], dtype=np.float32)

        # run kmeans on the coords
        labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
        labels = labels.reshape(-1)  # transpose to row vec

        # segment lines based on their kmeans label
        segmented = defaultdict(list)
        for i, line in enumerate(lines):
            segmented[labels[i]].append(line)
        segmented = list(segmented.values())
        return segmented
    
    def __intersection(self, line1, line2) -> list:
        '''
        Finds the intersection of two lines given in Hesse normal form.

        Returns closest integer pixel locations.
        See https://stackoverflow.com/a/383527/5087436
        '''
        rho1, theta1 = line1[0]
        rho2, theta2 = line2[0]
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [[x0, y0]]


    def __segmented_intersections(self, lines) -> list:
        '''
        Finds the intersections between groups of lines.
        '''
        intersections = []
        for i, group in enumerate(lines[:-1]):
            for next_group in lines[i+1:]:
                for line1 in group:
                    for line2 in next_group:
                        intersections.append(self.__intersection(line1, line2)) 

        return intersections
    
    def __filter_intersections(self, intersections, rows, cols) -> dict:
        '''
        Filter intersections points.
        '''
        intersections = [pt[0] for pt in intersections]
        intersections = [pt for pt in intersections if pt[0] > 0 and pt[0] < rows and pt[1] > 0 and pt[1] < cols]
        intersections = np.array(intersections)
        s = intersections.sum(axis = 1)
        d = np.diff(intersections, axis = 1)
        points = {
            "top_left": tuple(intersections[np.argmin(s)]),
            "bottom_left": tuple(intersections[np.argmax(d)]),
            "top_right": tuple(intersections[np.argmin(d)]),
            "bottom_right": tuple(intersections[np.argmax(s)])
        }
        return points
        
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


    def __draw_p_lines(self, img, lines, color = (0, 0, 255)) -> None:
        '''
        Draw P lines on an image.
        '''
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2:
                y1, y2 = -img.shape[0], img.shape[0]
            elif y1 == y2:
                x1, x2 = -img.shape[1], img.shape[1]
            else:
                k = (y2 - y1) / (x2 - x1)
                m = y1 -k * x1
                x1 = -img.shape[1]
                y1 = k * x1 + m
                x2 = -img.shape[1]
                y2 = k * x2 + m
            cv2.line(img, (x1, y1), (x2, y2), color, 1)

    def __is_square(self, points, th = 0.05) -> bool:
        '''
        Check if a region, defined by corner points, is a (almost) perfect square. 
        '''
        # Compare square sides
        side1 = np.sqrt( 
            (points['top_left'][0] - points['top_right'][0]) ** 2 
            +
            (points['top_left'][1] - points['top_right'][1]) ** 2 
        )
        side2 = np.sqrt( 
            (points['top_left'][0] - points['bottom_left'][0]) ** 2
            +
            (points['top_left'][1] - points['bottom_left'][1]) ** 2 
        )
        diff = 1.0 - side1 / side2 if side1 < side2 else 1.0 - side2 / side1
        if diff > th:
            return False
        
        # Compare square diagonals
        dia1 = np.sqrt( 
            (points['top_left'][0] - points['bottom_right'][0]) ** 2 
            +
            (points['top_left'][1] - points['bottom_right'][1]) ** 2 
        )
        dia2 = np.sqrt( 
            (points['top_right'][0] - points['bottom_left'][0]) ** 2 
            +
            (points['top_right'][1] - points['bottom_left'][1]) ** 2 
        )
        diff = 1.0 - dia1 / dia2 if dia1 < dia2 else 1.0 - dia2 / dia1
        if diff > th:
            return False
        
        return True
