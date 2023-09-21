#!/usr/bin/env python3
import cv2

# --------------------
# General super class for handle a OpenCV windows
# ------------------------------------------------
class Window:
    def __init__(self, name, img, param = cv2.WINDOW_NORMAL) -> None:
        self.name = name
        self.img = img
        self.org_img = img.copy()
        
        # Create a named window
        cv2.namedWindow(self.fname, param)

    @property
    def fname(self) -> str:
        return self.name    

    # Close (or destroy) map image window
    def close(self) -> None:
        cv2.destroyWindow(self.fname)

    # Display the map image window
    def display(self) -> None:
        cv2.imshow(self.fname, self.img)
    
    # Rest the image to the original image
    def reset(self) -> None:
        self.img = self.org_img.copy()
