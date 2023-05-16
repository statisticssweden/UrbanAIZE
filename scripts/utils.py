import os

from sweden_crs_transformations.crs_projection import CrsProjection
from sweden_crs_transformations.crs_coordinate import CrsCoordinate

def getModifiedName(fname: str) -> str:
    '''
    Remove special characters from name of file or folder
    '''
    fname = fname.translate(str.maketrans("åäöÅÄÖ", "aaoAAO"))
    fname = fname.replace(' ', '')
    return fname

def getRT90(x: float, y: float) -> tuple:
	'''
    Converts the gridvalue to rt90 coordinate
    '''
	y = y * 50 + 6050 				
	x = x * 50 + 1200
	return (int(x), int(y))

def sr99Tort90(sr99y, sr99x) -> tuple:
    sr99: CrsCoordinate = CrsCoordinate.create_coordinate(
        CrsProjection.SWEREF_99_TM,
        sr99y,
        sr99x
    )
    rt90: CrsCoordinate = sr99.transform(CrsProjection.RT90_2_5_GON_V)
    x, y = rt90.get_longitude_x(), rt90.get_latitude_y()
    return (y, x)