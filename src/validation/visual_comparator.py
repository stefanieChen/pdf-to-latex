"""Visual comparison between source and generated documents using SSIM."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("pdf2latex.validation.visual")


class VisualComparator:
    """Compare source and generated document images using structural similarity."""

    def __init__(self, ssim_threshold: float = 0.85):
        """Initialize VisualComparator.

        Args:
            ssim_threshold: Minimum SSIM score to consider a match acceptable.
        """
        self.ssim_threshold = ssim_threshold

    def compare(
        self,
        source_image: np.ndarray,
        generated_image: np.ndarray,
    ) -> Tuple[float, bool]:
        """Compare two images using SSIM.

        Args:
            source_image: Original document image (BGR or grayscale).
            generated_image: Generated document image (BGR or grayscale).

        Returns:
            Tuple of (ssim_score, passes_threshold).
        """
        # Convert to grayscale if needed
        if len(source_image.shape) == 3:
            source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
        else:
            source_gray = source_image

        if len(generated_image.shape) == 3:
            generated_gray = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)
        else:
            generated_gray = generated_image

        # Resize to same dimensions
        h = min(source_gray.shape[0], generated_gray.shape[0])
        w = min(source_gray.shape[1], generated_gray.shape[1])
        source_resized = cv2.resize(source_gray, (w, h))
        generated_resized = cv2.resize(generated_gray, (w, h))

        score = self._compute_ssim(source_resized, generated_resized)
        passes = score >= self.ssim_threshold

        logger.info("SSIM score: %.4f (threshold: %.2f, passes: %s)", score, self.ssim_threshold, passes)
        return score, passes

    def compare_files(
        self,
        source_path: Path,
        generated_path: Path,
    ) -> Tuple[float, bool]:
        """Compare two image files.

        Args:
            source_path: Path to source image.
            generated_path: Path to generated image.

        Returns:
            Tuple of (ssim_score, passes_threshold).
        """
        source = cv2.imread(str(source_path))
        generated = cv2.imread(str(generated_path))

        if source is None:
            raise ValueError(f"Cannot read source image: {source_path}")
        if generated is None:
            raise ValueError(f"Cannot read generated image: {generated_path}")

        return self.compare(source, generated)

    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute Structural Similarity Index between two grayscale images.

        Args:
            img1: First grayscale image.
            img2: Second grayscale image (same size as img1).

        Returns:
            SSIM score between -1 and 1.
        """
        try:
            from skimage.metrics import structural_similarity
            score = structural_similarity(img1, img2)
            return float(score)
        except ImportError:
            logger.warning("scikit-image not available, using OpenCV SSIM approximation")
            return self._compute_ssim_cv(img1, img2)

    @staticmethod
    def compare_images(image_a, image_b) -> float:
        """Compare two PIL Images and return an SSIM score.

        Convenience method used by the progressive DeTikZify quality gate.
        Accepts PIL.Image.Image or numpy arrays.

        Args:
            image_a: First image (PIL Image or numpy BGR/RGB array).
            image_b: Second image (PIL Image or numpy BGR/RGB array).

        Returns:
            SSIM score between -1 and 1.
        """
        def _to_gray(img) -> np.ndarray:
            # PIL Image
            try:
                from PIL import Image as PILImage
                if isinstance(img, PILImage.Image):
                    img = np.array(img.convert("L"))
                    return img
            except ImportError:
                pass
            # numpy array
            if len(img.shape) == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img

        g1 = _to_gray(image_a)
        g2 = _to_gray(image_b)
        h = min(g1.shape[0], g2.shape[0])
        w = min(g1.shape[1], g2.shape[1])
        g1 = cv2.resize(g1, (w, h))
        g2 = cv2.resize(g2, (w, h))
        return float(VisualComparator._compute_ssim_cv(g1, g2))

    @staticmethod
    def _compute_ssim_cv(img1: np.ndarray, img2: np.ndarray) -> float:
        """Approximate SSIM using OpenCV (fallback).

        Args:
            img1: First grayscale image.
            img2: Second grayscale image.

        Returns:
            Approximate SSIM score.
        """
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return float(ssim_map.mean())
