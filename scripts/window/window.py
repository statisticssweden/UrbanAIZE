#!/usr/bin/env python3
import cv2

# --------------------
# General super class for handle a OpenCV windows
# ------------------------------------------------
class Window:
    def __init__(self, name, img, param = cv2.WINDOW_NORMAL) -> None:
        self.__name = name
        self.img = img
        self.orig_img = img.copy()
        
        # Create a named window
        cv2.namedWindow(self.name, param)

    @property
    def name(self) -> str:
        return self.__name    

    # Close (or destroy) map image window
    def close(self) -> None:
        cv2.destroyWindow(self.name)
        self.img = self.orig_img = None

    # Display the map image window
    def display(self) -> None:
        if self.img is not None:
            cv2.imshow(self.name, self.img)
    
    # Rest the image to the original image
    def reset(self) -> None:
        self.img = self.orig_img.copy()
