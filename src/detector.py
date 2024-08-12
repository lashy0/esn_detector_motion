import cv2
import torch
import numpy as np

from typing import List, Tuple, Dict


class MotionDetector:
    ALLOWED_ROI_ALGOS = ["inrange", "contours"]
    
    def __init__(
        self,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        min_area: int = 1000,
        roi_algo: str = "inrange",
        contours_coords: List[List[int]] = [[0, 0]],
        cnt_zones: int = 10
    ) -> None:
        if not isinstance(lower_bound, np.ndarray) or not isinstance(upper_bound, np.ndarray):
            raise TypeError("lower_bound and upper_bound must be of type np.ndarray")
        
        if lower_bound.shape != (3,) or upper_bound.shape != (3,):
            raise ValueError("lower_bound and upper_bound must have shape 3")
        
        if not isinstance(roi_algo, str):
            raise TypeError("roi_algo must be a string")
        
        if roi_algo not in self.ALLOWED_ROI_ALGOS:
            raise ValueError(f"roi_algo must be one of {self.ALLOWED_ROI_ALGOS}")
        
        self.roi_found = False
        self.cnt_frames = 0
        
        # Parametrs roi find
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.min_area = min_area
        self.roi_algo = roi_algo
        self.contours_coords = np.array(contours_coords, dtype=np.int32)
        self.cnt_zones = cnt_zones
        
        self.previous_frame = None
    
    def detect(self, frame: np.ndarray) -> bool:
        """
        Detect motion in the given frame.

        Args:
            frame (np.ndarray): Input frame for motion detection.

        Returns:
            bool: True if motion is detected, otherwize False.
        """
        if not self.roi_found:
            res = self.__find_roi(frame)
            if not res:
                return False
        
        res = False
        self.cnt_frames += 1
        
        # Код обработки
        
        return res
    
    def reset(self) -> None:
        """Reset the ROI found status."""
        self.roi_found = False
        self.cnt_frames = 0
    
    def is_roi_found(self) -> bool:
        """
        Check if the ROI has been found.

        Returns:
            bool: True if ROI is found, otherwise False.
        """
        return self.roi_found
    
    def __preprocess_frame(self, frame: np.ndarray) -> List[float]:
        """
        Preprocesses the input frame to extract motion features within predefined
        regions of interest (ROI).

        This method converts the input frame to grayscale, applies Gaussian blur
        to reduce noise, and then calculates the absolute difference between the
        current and the previous frame to detect motion. The resulting motion mask
        is used to evaluate the amount of motion within each ROI.

        Args:
            frame (np.ndarray): The current frame from the video stream.

        Returns:
            List[float]: A list of motion features, where each feature corresponds
            to the amount of motion detected within a specific ROI. If no motion is
            detected in a ROI, the feature value is 0.
        """
        features = []
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
        
        if self.previous_frame is None:
            self.previous_frame = gray_frame
            return features
        
        frame_diff = cv2.absdiff(self.previous_frame, gray_frame)
        
        _, motion_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        
        self.previous_frame = gray_frame
        
        for zone in self.zones:
            if "roi_mask" not in zone:
                features.append(0)
                continue
            
            x, y, w, h = zone["roi_rect"]
            
            roi = motion_mask[y : y + h, x : x + w]
            motion_value = np.sum(roi) / 255
            features.append(motion_value)
        
        return features
    
    def __find_roi(self, frame: np.ndarray) -> bool:
        """
        Indentifies and segments the ROI in the input frame based on the specified algo.
        
        This method processes the input frame to find areas of interest using
        one of two algorithms:
        - 'inrange': Applies color thresholding in the HSV color space to generate
        a mask based on specified lower and upper bounds.
        - 'contours': Uses predefined contour coordinates to create a mask that
        outlines the ROI.
        
        The nethod then processes the mask to refine it using dilation and erosion
        operations, finds contours, and selects the most significant one based om area.
        The selected contour is then divided into zones, and each zone mask and
        bounding rectangle are stored.

        Args:
            frame (np.ndarray): The input frame from which the ROI.

        Returns:
            bool: True if a valid ROI is found, False otherwise.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        if self.roi_algo == "inrange":
            mask = cv2.inRange(hsv, self.lower_bound, self.upper_bound)
        elif self.roi_algo == "contours":
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [self.contours_coords], -1, (255,), cv2.FILLED)
        else:
            raise ValueError(f"Unknown roi_algo: {self.roi_algo}")
        
        mask = cv2.dilate(mask, np.ones((3, 3)), iterations=2)
        mask = cv2.erode(mask, np.ones((3, 3)), iterations=1)
        
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        idx = self.__get_best_contour(contours)
        
        if (idx < 0) or (cv2.contourArea(contours[idx]) < self.min_area):
            return False
        
        if hierarchy is None:
            cv2.drawContours(mask, [contours[idx]], -1, (255,), cv2.FILLED)
        else:
            for i in range(0, len(contours)):
                if (i != idx) and (hierarchy[0][i][3] != idx):
                    cv2.drawContours(mask, [contours[i]], -1, (0), -1)
        
        # TODO: тут можно добавить апроксимацию контура
        contours = contours[idx]
        
        rect = cv2.boundingRect(contours)
        
        self.zones = self.__divide_mask(mask, rect, self.cnt_zones)
        
        for zone in self.zones:
            contours, _ = cv2.findContours(zone["mask"], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            idx = self.__get_best_contour(contours)
            if idx < 0:
                # TODO: подумать стоит тут создавать информацию о roi
                # или просто при работе с ним делать проверки на существование
                # значений roi_mask и roi_rect
                continue
            
            roi_rect = cv2.boundingRect(contours[idx])
            x, y, w, h = roi_rect
            roi_mask = mask[y : y + h, x : x + w]
            zone["roi_mask"] = roi_mask
            zone["roi_rect"] = roi_rect
        
            self.roi_found = True
        
        return self.roi_found
    
    def __get_best_contour(self, contours: List[np.ndarray]) -> int:
        """
        Get the index of the best contour based on the largest area.

        Args:
            contours (List[np.ndarray]): List of contours.

        Returns:
            int: Index of the best contour. Return -1 if no contours are found.
        """
        if len(contours) == 0:
            return -1
        
        areas = [cv2.contourArea(cnt) for cnt in contours]
        return int(np.argmax(areas))
    
    def __divide_mask(self, mask: np.ndarray, rect: Tuple[int], cnt_zones: int) -> List[Dict[str, np.ndarray]]:
        """
        Divide the mask within the given bounding rectangle into zones.

        Args:
            mask (np.ndarray): The mask to be divided.
            rect (Tuple[int]): Bounding rectangle (x, y, width, height).
            cnt_zones (int): Number of zones to divide into.

        Returns:
            List[Dict[str, np.ndarray]]: List of dictionaries with zone masks.
        """
        x, y, w, h = rect
        zones = [None] * cnt_zones
        
        mask_roi = mask[y : y + h, x : x + w]
        _, w_mask = mask_roi.shape[:2]
        col_w = w_mask // cnt_zones
        
        for i in range(cnt_zones):
            column = mask_roi[:, i * col_w : (i + 1) * col_w]
            zone = np.zeros_like(mask)
            zone[y : y + h, x + i * col_w : x + (i + 1) * col_w] = column
            
            zones[i] = {"mask": zone}
        
        return zones
