import cv2
import torch
import numpy as np

from typing import List, Tuple, Dict

from model import ESN
from utils import mse
from utils.buffer import FixedBuffer


class MotionDetector:
    ALLOWED_ROI_ALGOS = ["inrange", "contours"]
    
    def __init__(
        self,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        model: ESN,
        min_area: int = 1000,
        roi_algo: str = "inrange",
        contours_coords: List[List[int]] = [[0, 0]],
        cnt_zones: int = 10,
        size_data_history: int = 90,
        size_error_history: int = 50,
        size_anomaly_history: int = 10,
        score_threshold: float = 0.5,
        factor: float = 3.0
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
        
        # Parametrs preprocess frame
        self.previous_frame = None
        
        self.buffer_data = FixedBuffer(size_data_history)
        self.buffer_error = FixedBuffer(size_error_history)
        self.buffer_anomaly = FixedBuffer(size_anomaly_history)
        
        self.model = model
        self.model_train = False
        
        self.score_threshold = score_threshold
        self.adaptive_threshold = 0
        self.factor = factor
    
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
        data_zones = self.__preprocess_frame(frame)
        if data_zones:
            self.buffer_data.add(data_zones)
        
        if self.buffer_data.is_full():
            if not self.model_train:
                self.__train_esn()
                self.model_train = True
            else:
                res = self.__detect_anomaly()
        
        return res
    
    def reset(self) -> None:
        """Reset the ROI found status."""
        self.roi_found = False
        self.cnt_frames = 0
        
        self.model_train = False
        self.model.reset_reservoir()
    
    def is_roi_found(self) -> bool:
        """
        Check if the ROI has been found.

        Returns:
            bool: True if ROI is found, otherwise False.
        """
        return self.roi_found
    
    def __train_esn(self) -> None:
        """Train the ESN using the buffered data."""
        data = torch.tensor(self.buffer_data.get_buffer(), dtype=torch.float32)
        x = data[:-1]
        y = data[1:]
        self.model.fit(x, y, 1e-4)
    
    def __detect_anomaly(self) -> bool:
        """Detect anomaly in the data using the trained ESN.

        Returns:
            bool: True if an anomaly is detected, otherwise False.
        """
        x = torch.tensor(self.buffer_data.get_value(-2), dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.buffer_data.get_value(-1), dtype=torch.float32).unsqueeze(0)
        predict, _ = self.model(x)
        predict = predict.detach().numpy()
        
        error = mse(y, predict)
        self.buffer_error.add(error)
        
        self.model.update_redaut_lms(x, y, 0.01)
        
        if self.buffer_error.get_current_size() < self.buffer_error.size // 2:
            return False
        
        self.__update_treshold()
        anom = error > self.adaptive_threshold        
        self.buffer_anomaly.add(anom)
        
        score = np.mean(self.buffer_anomaly.get_buffer())
        
        return score > self.score_threshold
    
    def __update_treshold(self) -> float:
        errors = np.array(self.buffer_error.get_buffer())
        
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        self.adaptive_treshold = mean_error + self.factor * std_error
    
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
        
        for zone in self.zones:
            if "roi_mask" not in zone:
                features.append(0)
                continue
            
            x, y, w, h = zone["roi_rect"]
            mask = zone["roi_mask"]
            
            roi = frame[y : y + h, x : x + w]
            roi = cv2.bitwise_and(roi, roi, mask=mask)
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            if "prew" not in zone:
                features.append(0)
                continue
            
            diff = cv2.absdiff(roi, zone["prew"])
            diff = cv2.GaussianBlur(diff, (7, 7), 0)
            diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            
            flat = diff.flatten()
            non_zero_value = flat[np.nonzero(flat)]
            motion_value = np.median(non_zero_value) if non_zero_value.size > 0 else 0.0
            motion_value = motion_value / 255.0
            
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
