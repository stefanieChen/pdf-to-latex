"""Image enhancement pipeline for scanned documents."""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("pdf2latex.preprocessing.enhance")


class ImageEnhancer:
    """Enhance scanned document images: denoise, binarize, deskew."""

    def __init__(
        self,
        denoise_strength: int = 10,
        binary_block_size: int = 11,
        binary_constant: int = 2,
    ):
        """Initialize ImageEnhancer.

        Args:
            denoise_strength: Strength of Non-Local Means denoising.
            binary_block_size: Block size for adaptive thresholding (odd number).
            binary_constant: Constant subtracted from mean in adaptive threshold.
        """
        self.denoise_strength = denoise_strength
        self.binary_block_size = binary_block_size
        self.binary_constant = binary_constant

    def enhance(
        self,
        image: np.ndarray,
        denoise: bool = True,
        binarize: bool = False,
        deskew: bool = True,
    ) -> np.ndarray:
        """Run the full enhancement pipeline on an image.

        Args:
            image: Input image as numpy array (BGR).
            denoise: Whether to apply denoising.
            binarize: Whether to apply adaptive binarization.
            deskew: Whether to correct skew.

        Returns:
            Enhanced image as numpy array.
        """
        result = image.copy()

        if denoise:
            result = self.apply_denoise(result)

        if deskew:
            angle = self.detect_skew_angle(result)
            if abs(angle) > 0.5:
                result = self.apply_deskew(result, angle)
                logger.info("Deskewed by %.2f degrees", angle)

        if binarize:
            result = self.apply_binarize(result)

        return result

    def apply_denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply Non-Local Means denoising.

        Args:
            image: Input BGR image.

        Returns:
            Denoised image.
        """
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(
                image, None, self.denoise_strength, self.denoise_strength, 7, 21
            )
        return cv2.fastNlMeansDenoising(image, None, self.denoise_strength, 7, 21)

    def apply_binarize(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for binarization.

        Args:
            image: Input image (BGR or grayscale).

        Returns:
            Binary image.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.binary_block_size,
            self.binary_constant,
        )
        return binary

    def detect_skew_angle(self, image: np.ndarray) -> float:
        """Detect document skew angle using Hough Line Transform.

        Args:
            image: Input image (BGR or grayscale).

        Returns:
            Detected skew angle in degrees.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        if lines is None:
            return 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Only consider near-horizontal lines
            if abs(angle) < 45:
                angles.append(angle)

        if not angles:
            return 0.0

        return float(np.median(angles))

    def apply_deskew(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image to correct skew.

        Args:
            image: Input image.
            angle: Skew angle in degrees.

        Returns:
            Deskewed image.
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new bounding box
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2

        border_value = (255, 255, 255) if len(image.shape) == 3 else 255
        return cv2.warpAffine(image, rotation_matrix, (new_w, new_h), borderValue=border_value)

    def enhance_file(
        self,
        image_path: Path,
        output_path: Optional[Path] = None,
        **kwargs,
    ) -> Path:
        """Enhance an image file and save the result.

        Args:
            image_path: Path to input image.
            output_path: Path to save enhanced image. Defaults to overwriting input.
            **kwargs: Additional arguments passed to enhance().

        Returns:
            Path to the enhanced image.
        """
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        enhanced = self.enhance(image, **kwargs)

        if output_path is None:
            output_path = image_path
        output_path = Path(output_path)

        cv2.imwrite(str(output_path), enhanced)
        logger.debug("Enhanced image saved: %s", output_path)
        return output_path
