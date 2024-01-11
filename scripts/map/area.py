import cv2
import numpy as np
from collections import defaultdict

# ----------------
# Main functions 
# --------------------

# Detect the map area of a map image
def detectMapArea(img: np.ndarray, length: int = 1000) -> np.array:
    
    # Detect edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = detectEgdes(gray, sz = 3, th1 = 30, th2 = 220)

    # Finds linear edges (longer than 'length' pixels)
    lines = cv2.HoughLines(edges, 1, np.pi/180, length)

    # Cluster line angles into 2 groups (vertical and horizontal)
    if lines is not None:
        groups = groupByAngleKmeans(lines)

        # Find the intersections of each vertical line with each horizontal line
        intersections = segmentIntersections(groups)
        if intersections:
            points = filterIntersections(intersections, img.shape[0], img.shape[1])

            # Check if interesction points form an (almost) perfect square
            if square(points):
                area = {}
                for key, value in points.items():
                    area[key] = (int(value[0]), int(value[1])) 
                return area

    return None


# Calcualte squareness score
def squareness(corners):

    # Calculate vectors between corners points 
    vectors = [
        np.array(corners['top_left']) - np.array(corners['top_right']),
        np.array(corners['top_right']) - np.array(corners['bottom_right']),
        np.array(corners['bottom_right']) - np.array(corners['bottom_left']),
        np.array(corners['bottom_left']) - np.array(corners['top_left']),
        np.array(corners['top_left']) - np.array(corners['bottom_right']),
        np.array(corners['bottom_left']) - np.array(corners['top_right'])
    ]


    # Calculate side and diagonal lengths
    side_lengths = [np.linalg.norm(vectors[i]) for i in range(4)]
    diagonal_lengths = [np.linalg.norm(vectors[i]) for i in range(4, 6)]

    # Calculate the sum of differences between sides
    side_diff_sum = sum(1. - abs(side_lengths[i] / side_lengths[(i + 1) % 4]) for i in range(4))

    # Calculate the sum differences between diagonals and diagonal of squares
    diagonal_diff_sum = sum([
        1. - abs(diagonal_lengths[0] / diagonal_lengths[1]),
        1. - abs(diagonal_lengths[0] / (side_lengths[2] * np.sqrt(2.))),
        1. - abs(diagonal_lengths[1] / (side_lengths[3] * np.sqrt(2.)))
    ]) 
        
    # Calculate angle differences between adjacent sides (should be close to 90 degrees)
    dot_products = [np.dot(vectors[i], vectors[(i + 1) % 4]) for i in range(4)]
    angle_diff_sum = sum(abs(dot_products[i]) for i in range(4))

    # Combine length and angle differences into a single score
    return abs(side_diff_sum + diagonal_diff_sum + angle_diff_sum)
    
    '''
    # Calculate vectors between corners points 
    vectors = [
        np.array(corners['top_left']) - np.array(corners['top_right']),
        np.array(corners['top_right']) - np.array(corners['bottom_right']),
        np.array(corners['bottom_right']) - np.array(corners['bottom_left']),
        np.array(corners['bottom_left']) - np.array(corners['top_left'])
    ]

    # Calculate the sum of squared differences of opposite sides
    side_lengths = [np.linalg.norm(vectors[i]) for i in range(4)]
    side_diff_sum = sum((side_lengths[i] - side_lengths[(i + 2) % 4]) ** 2 for i in range(4))
    
    # Check if angles between adjacent sides are close to 90 degrees
    dot_products = [np.dot(vectors[i], vectors[(i + 1) % 4]) for i in range(4)]
    angle_diff_sum = sum(abs(dot_products[i]) for i in range(4))

    # Combine side lengths and angle differences into a single score
    return side_diff_sum + angle_diff_sum
    '''

# Compare squares and find the best one (with respect to squareness)
def findBestSquare(squares):
    best = { 'square': None, 'score': float('inf') }
    for square in squares:
        score = squareness(square)
        if score < best['score']:
            best['square'] = square
            best['score'] = score
    return best['square']


# -------------------
# Support functions
# -----------------------
                    
# Detect all image edges.
def detectEgdes(img: np.ndarray, sz: int = -1, th1: int = 25, th2: int = 225) -> np.ndarray:
    img = cv2.GaussianBlur(img, (sz,sz), 0) if sz > 0 else img
    
    # Canny Edge Detection (with Gaussian blur for better detection)
    edges = cv2.Canny( image = img, 
                       threshold1 = th1, 
                       threshold2 = th2, 
                       apertureSize = 3, 
                       L2gradient = True )
    
    # Postprocess the detected egdes with dilation filter
    if sz > 0:
        kernel = cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (2 * sz + 1, 2 * sz + 1), (sz, sz))
        edges = cv2.dilate(edges, kernel)
    return edges

# Group lines based on angles (with k-means clustering).
def groupByAngleKmeans(lines, k=2, **kwargs) -> list:
    """
    Uses k-means on the coordinates of the angle on the unit circle 
    to segment 'k' angles inside 'lines'.
    """

    # Define criteria (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 1)

    # Angles in radians in range: [0, pi]
    angles = np.array([line[0][1] for line in lines])

    # Multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # Run k-means clustering on the line coordinates
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # ...transpose to row vec

    # Group lines based on their k-means label
    groups = defaultdict(list)
    for i, line in enumerate(lines):
        groups[labels[i]].append(line)
    groups = list(groups.values())
    return groups


#  Finds the intersection of two lines given in Hesse normal form.
def getIntersection(line1, line2) -> list:
    '''
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


# Finds the intersections between groups of lines.
def segmentIntersections(lines) -> list:
    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(getIntersection(line1, line2)) 
    return intersections


# Filter intersections points.
def filterIntersections(intersections, rows, cols) -> dict:
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


# Check if a region, defined by corner points, is a (almost) perfect square. 
def square(points, th = 0.05) -> bool:

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
