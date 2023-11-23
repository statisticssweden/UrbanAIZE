import os
from sweden_crs_transformations.crs_projection import CrsProjection
from sweden_crs_transformations.crs_coordinate import CrsCoordinate

# Remove special characters from name of file or folder
def getModifiedName(fname: str) -> str:
    fname = fname.translate(str.maketrans("åäöÅÄÖ", "aaoAAO"))
    fname = fname.replace(' ', '')
    return fname

# Convert gridvalue to rt90 coordinates
def getRT90(x: float, y: float) -> tuple:
    y = y * 50 + 6050 				
    x = x * 50 + 1200
    return (int(x), int(y))

# Convert sr99 coordinates to rt90 coordinates
def sr99Tort90(sr99y, sr99x) -> tuple:
    sr99: CrsCoordinate = CrsCoordinate.create_coordinate(
        CrsProjection.SWEREF_99_TM,
        sr99y,
        sr99x
    )
    rt90: CrsCoordinate = sr99.transform(CrsProjection.RT90_2_5_GON_V)
    x, y = rt90.get_longitude_x(), rt90.get_latitude_y()
    return (y, x)
