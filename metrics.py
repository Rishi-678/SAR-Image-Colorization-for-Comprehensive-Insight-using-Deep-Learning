import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def evaluate_metrics(original_path, colorized_path):
    original = cv2.imread("C:/Users/sanka/OneDrive/Desktop/Image-Colorization-DL/images/c2.jpg")
    colorized = cv2.imread("C:/Users/sanka/OneDrive/Desktop/Image-Colorization-DL/images/c2output.jpg")



    # Resize to the same size if needed
    if original.shape != colorized.shape:
        colorized = cv2.resize(colorized, (original.shape[1], original.shape[0]))

    # Convert to grayscale for SSIM
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    colorized_gray = cv2.cvtColor(colorized, cv2.COLOR_BGR2GRAY)

    ssim_val = ssim(original_gray, colorized_gray)
    mse_val = np.mean((original.astype("float") - colorized.astype("float")) ** 2)
    psnr_val = psnr(original, colorized)

    return ssim_val, mse_val, psnr_val

# Example usage:
original_image = "C:/Users/sanka/OneDrive/Desktop/Image-Colorization-DL/images/c2.jpg"
colorized_image ="C:/Users/sanka/OneDrive/Desktop/Image-Colorization-DL/images/c2output.jpg"

ssim_val, mse_val, psnr_val = evaluate_metrics(original_image, colorized_image)
print(f"SSIM: {ssim_val:.4f}")
print(f"MSE: {mse_val:.2f}")
print(f"PSNR: {psnr_val:.2f} dB")