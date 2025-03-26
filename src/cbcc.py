import cv2
print("hjbfdjbdf")
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA

# image = cv2.imread('datasets/trainA/n01496331_49.jpg')

# # # Display the image
# cv2.imshow('Image', image)
# cv2.waitKey(10000)
# cv2.destroyAllWindows()

# # Step 1: Color Balance-Guided Color Correction (CBCC)
def color_balance_guided_color_correction(image):
    # Convert to CIELab color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Apply box filtering to the L channel
    filtered_l = cv2.boxFilter(l_channel, -1, (1, 1))
    # filtered_l = cv2.GaussianBlur(l_channel, (3, 3), 0)
    # filtered_l = cv2.bilateralFilter(l_channel, d=9, sigmaColor=75, sigmaSpace=75)
    # Apply box filtering to the A and B channels

    # Calculate means of a and b channels
    mean_a = np.mean(a_channel)
    mean_b = np.mean(b_channel)

    # Update channels with lower average values
    if mean_a < mean_b:
        updated_a = a_channel + ((mean_b - mean_a) / (mean_b + mean_a)) * b_channel
        updated_b = b_channel
    else:
        updated_b = b_channel + ((mean_a - mean_b) / (mean_b + mean_a)) * a_channel
        updated_a = a_channel

    # Merge updated channels
    corrected_lab = cv2.merge([filtered_l, updated_a.astype(np.uint8), updated_b.astype(np.uint8)])

    # Convert back to RGB
    corrected_rgb = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
    return corrected_rgb

# # Step 2: Decomposition Using Mean Shift
def image_decomposition(image, alpha=0.5):
    """
    Decompose the image into foreground and background components.
    """
    # Normalize image to [0, 1]
    normalized_image = image.astype(np.float32) / 255.0

    # Compute separation critical value k
    max_intensity = np.max(normalized_image)
    k = alpha * normalized_image / max_intensity

    # Compute foreground and background
    foreground = (1 - k) * normalized_image
    background = k * normalized_image

    # Scale back to [0, 255]
    foreground = np.clip(foreground * 255, 0, 255).astype(np.uint8)
    background = np.clip(background * 255, 0, 255).astype(np.uint8)

    return foreground, background

# Step 3: Percentile Maximum-Based Contrast Enhancement
def percentile_maximum_contrast_enhancement(image):
    # Traverse RGB channels
    enhanced_channels = []
    for channel in cv2.split(image):
        channel = channel.astype(np.float32)

        # Compute percentiles
        v1 = np.percentile(channel, 0.1)
        v2 = np.percentile(channel, 99.5)

        # Thresholding
        channel[channel < v1] = v1
        channel[channel > v2] = v2

        # Normalize contrast
        enhanced_channel = (channel - np.min(channel)) / (np.max(channel) - np.min(channel))
        enhanced_channels.append(enhanced_channel)

    # Merge enhanced channels
    enhanced_image = cv2.merge(enhanced_channels)
    enhanced_image = (enhanced_image * 255).astype(np.uint8)
    return enhanced_image

# Step 4: Multilayer Transmission Map Estimated Dehazing
def multilayer_transmission_map_dehazing(image):
    # Estimate atmospheric light (A)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    A = np.max(gray)

    # Compute transmission map (t)
    block_size = 15
    h, w = image.shape[:2]
    t_map = np.zeros((h, w), dtype=np.float32)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            t_map[i:i+block_size, j:j+block_size] = 1 - np.min(block) / A

    # Dehaze using atmospheric scattering model
    J = np.zeros_like(image, dtype=np.float32)
    for c in range(3):  # RGB channels
        J[:, :, c] = (image[:, :, c] - A * (1 - t_map)) / t_map
        J[:, :, c][t_map == 0] = 0  # Avoid division by zero

    J = np.clip(J, 0, 255).astype(np.uint8)
    return J

# Step 5: Principal Component Analysis Fusion
def pca_fusion(foreground, background):
    # Flatten images for PCA
    fg_flat = foreground.reshape((-1, 3)).astype(np.float32)
    bg_flat = background.reshape((-1, 3)).astype(np.float32)

    # Compute covariance matrix
    cov_matrix = np.cov(fg_flat.T, bg_flat.T)

    # Solve for eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Normalize weights
    total = np.sum(eigenvalues)
    weights_fg = eigenvalues[0] / total
    weights_bg = eigenvalues[1] / total

    # Linearly combine images
    fused_image = (weights_fg * foreground + weights_bg * background).astype(np.uint8)
    return fused_image

# # Main Function
def pcfb_pipeline(input_image_path, output_image_path):
    # Load input image
    input_image = cv2.imread(input_image_path)

    # Step 1: Color Balance-Guided Color Correction
    corrected_image = color_balance_guided_color_correction(input_image)
    cv2.imshow('corrected_image', corrected_image)

    # Step 2: Decomposition Using Mean Shift
    background, foreground = image_decomposition(corrected_image)
    cv2.imshow('foreground', foreground)
    cv2.imshow('background', background)

    # Step 3: Percentile Maximum-Based Contrast Enhancement
    enhanced_foreground = percentile_maximum_contrast_enhancement(foreground)
    cv2.imshow('enhanced_foreground', enhanced_foreground)

    # Step 4: Multilayer Transmission Map Estimated Dehazing
    dehazed_background = multilayer_transmission_map_dehazing(background)

    cv2.imshow('dehazed_background', dehazed_background)

    # Step 5: Principal Component Analysis Fusion
    fused_image = pca_fusion(enhanced_foreground, dehazed_background)

    # Save the final enhanced image
    # image = cv2.imread('datasets/trainA/n01496331_49.jpg')

# # Display the image
    # cv2.imshow('inputImage', image)
    cv2.imshow('fused_image', fused_image)

    success =cv2.imwrite(output_image_path, fused_image)
    if not success:
        print("Failed to save the image.")
    cv2.waitKey(30000)
    cv2.destroyAllWindows()
    print(f"Enhanced image saved to {output_image_path}")

# Run the pipeline
if __name__ == "__main__":
    # input_image_path = "datasets/trainA/n01496331_921.jpg"  # Replace with your input image path
    input_image_path = "datasets/448.jpg" 
    output_image_path = "results/enhanced_image.jpg"  # Replace with your desired output path
    pcfb_pipeline(input_image_path, output_image_path)