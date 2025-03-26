# import cv2
# import numpy as np
# from sklearn.cluster import MeanShift

# def mean_shift_decomposition(image):
#     """
#     Decompose the input image into foreground and background components using Mean Shift clustering.

#     Parameters:
#         image (numpy.ndarray): Input RGB image (color-corrected image from Step 1).

#     Returns:
#         tuple: Foreground and background sub-images.
#     """
#     # Reshape the image for clustering (height * width, channels)
#     reshaped_image = image.reshape((-1, 3))
#     reshaped_image = np.float32(reshaped_image)

#     # Apply Mean Shift clustering
#     bandwidth = 20  # Adjust based on your dataset
#     ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
#     labels = ms.fit_predict(reshaped_image)

#     # Create masks for foreground and background
#     unique_labels = np.unique(labels)
#     foreground_mask = (labels == unique_labels[0]).reshape(image.shape[:2])
#     background_mask = (labels == unique_labels[1]).reshape(image.shape[:2])

#     # Extract foreground and background sub-images
#     foreground = image.copy()
#     background = image.copy()

#     foreground[~foreground_mask] = 0  # Set non-foreground pixels to black
#     background[~background_mask] = 0  # Set non-background pixels to black

#     return foreground, background

# # Example usage
# if __name__ == "__main__":
#     # Load the color-corrected image from Step 1
#     input_image_path = "results/enhanced_image.jpg"  # Replace with your path
#     output_foreground_path = "results/foreground_image.jpg"
#     output_background_path = "results/background_image.jpg"

#     # Read the color-corrected image
#     color_corrected_image = cv2.imread(input_image_path)
#     if color_corrected_image is None:
#         print(f"Error: Unable to load image at {input_image_path}")
#     else:
#         # Perform Mean Shift decomposition
#         foreground, background = mean_shift_decomposition(color_corrected_image)

#         # Save the results
#         cv2.imwrite(output_foreground_path, foreground)
#         cv2.imwrite(output_background_path, background)
#         print(f"Foreground saved to {output_foreground_path}")
#         print(f"Background saved to {output_background_path}")

#         # Display the results (optional)
#         cv2.imshow("Foreground", foreground)
#         cv2.imshow("Background", background)
#         cv2.waitKey(20000)
#         cv2.destroyAllWindows()



# import cv2
# import numpy as np

# def decompose_image(image, alpha=0.5):
#     """
#     Decompose the input image into foreground and background components.

#     Parameters:
#         image (numpy.ndarray): Input RGB image (color-corrected image).
#         alpha (float): Weighting factor for decomposition (default: 0.5).

#     Returns:
#         tuple: Foreground and background sub-images.
#     """
#     # Normalize the image to [0, 1] for computation
#     image_normalized = image.astype(np.float32) / 255.0

#     # Compute the separation critical value k
#     max_intensity = np.max(image_normalized)  # Maximum pixel intensity
#     k = alpha * image_normalized / max_intensity

#     # Compute foreground and background
#     foreground = (1 - k) * image_normalized
#     background = k * image_normalized

#     # Convert back to uint8 format
#     foreground = np.clip(foreground * 255, 0, 255).astype(np.uint8)
#     background = np.clip(background * 255, 0, 255).astype(np.uint8)

#     return foreground, background

# # Example usage
# if __name__ == "__main__":
#     # Load the color-corrected image from Step 1
#     input_image_path = "results/enhanced_image.jpg"  # Replace with your path
#     output_foreground_path = "results/foreground_image.jpg"
#     output_background_path = "results/background_image.jpg"

#     # Read the color-corrected image
#     color_corrected_image = cv2.imread(input_image_path)
#     if color_corrected_image is None:
#         print(f"Error: Unable to load image at {input_image_path}")
#     else:
#         # Perform decomposition
#         foreground, background = decompose_image(color_corrected_image, alpha=0.5)

#         # Save the results
#         cv2.imwrite(output_foreground_path, foreground)
#         cv2.imwrite(output_background_path, background)
#         print(f"Foreground saved to {output_foreground_path}")
#         print(f"Background saved to {output_background_path}")

#         # Display the results (optional)
#         cv2.imshow("Foreground", foreground)
#         cv2.imshow("Background", background)
#         cv2.waitKey(20000)  # Wait indefinitely until a key is pressed
#         cv2.destroyAllWindows()


# import cv2
# import numpy as np

# def decompose_image(image, alpha=0.6):
#     """
#     Decompose the input image into foreground and background components.

#     Parameters:
#         image (numpy.ndarray): Input RGB image (color-corrected image).
#         alpha (float): Weighting factor for decomposition (default: 0.6).

#     Returns:
#         tuple: Foreground and background sub-images with blacked-out regions.
#     """
#     # Normalize the image to [0, 1] for computation
#     image_normalized = image.astype(np.float32) / 255.0

#     # Compute the separation critical value k
#     max_intensity = np.max(image_normalized)  # Maximum pixel intensity
#     k = alpha * image_normalized / max_intensity

#     # Apply soft thresholding to k
#     k = np.clip(k, 0.1, 0.9)

#     # Create binary masks for foreground and background
#     foreground_mask = (1 - k) > 0.5  # Pixels where (1 - k) dominates
#     background_mask = k > 0.5       # Pixels where k dominates

#     # Convert masks to 3 channels (to match the image dimensions)
#     foreground_mask = np.stack([foreground_mask[:, :, 0]] * 3, axis=-1)
#     background_mask = np.stack([background_mask[:, :, 0]] * 3, axis=-1)

#     # Apply masks to set unwanted regions to black
#     foreground = image_normalized * foreground_mask
#     background = image_normalized * background_mask

#     # Convert back to uint8 format
#     foreground = np.clip(foreground * 255, 0, 255).astype(np.uint8)
#     background = np.clip(background * 255, 0, 255).astype(np.uint8)

#     return foreground, background

# # Example usage
# if __name__ == "__main__":
#     # Load the color-corrected image from Step 1
#     input_image_path = "results/enhanced_image.jpg"  # Replace with your path
#     output_foreground_path = "results/foreground_image_blackened.jpg"
#     output_background_path = "results/background_image_blackened.jpg"

#     # Read the color-corrected image
#     color_corrected_image = cv2.imread(input_image_path)
#     if color_corrected_image is None:
#         print(f"Error: Unable to load image at {input_image_path}")
#     else:
#         # Perform decomposition
#         foreground, background = decompose_image(color_corrected_image, alpha=0.6)

#         # Save the results
#         cv2.imwrite(output_foreground_path, foreground)
#         cv2.imwrite(output_background_path, background)
#         print(f"Foreground saved to {output_foreground_path}")
#         print(f"Background saved to {output_background_path}")

#         # Display the results (optional)
#         cv2.imshow("Foreground", foreground)
#         cv2.imshow("Background", background)
#         cv2.waitKey(20000)  # Wait indefinitely until a key is pressed
#         cv2.destroyAllWindows()

import cv2
import numpy as np
def decompose_image_adaptive(image, alpha=0.5):
    """
    Decompose the input image into foreground and background components using adaptive thresholding.

    Parameters:
        image (numpy.ndarray): Input RGB image (color-corrected image).
        alpha (float): Weighting factor for decomposition (default: 0.5).

    Returns:
        tuple: Foreground and background sub-images.
    """
    # Normalize the image to [0, 1] for computation
    image_normalized = image.astype(np.float32) / 255.0

    # Compute the separation critical value k
    max_intensity = np.max(image_normalized)
    k = alpha * image_normalized / max_intensity

    # Apply adaptive thresholding to k
    k_binary = np.zeros_like(k)
    for c in range(3):  # Iterate over each channel
        k_channel = k[:, :, c]
        _, binary_mask = cv2.threshold((k_channel * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k_binary[:, :, c] = binary_mask / 255.0

    # Compute foreground and background
    foreground = (1 - k_binary) * image_normalized
    background = k_binary * image_normalized

    # Convert back to uint8 format
    foreground = np.clip(foreground * 255, 0, 255).astype(np.uint8)
    background = np.clip(background * 255, 0, 255).astype(np.uint8)

    return foreground, background

if __name__ == "__main__":
    # Load the color-corrected image from Step 1
    input_image_path = "results/enhanced_image.jpg"  # Replace with your path
    output_foreground_path = "results/foreground_image_blackened.jpg"
    output_background_path = "results/background_image_blackened.jpg"

    # Read the color-corrected image
    color_corrected_image = cv2.imread(input_image_path)
    if color_corrected_image is None:
        print(f"Error: Unable to load image at {input_image_path}")
    else:
        # Perform decomposition
        foreground, background = decompose_image_adaptive(color_corrected_image, alpha=0.6)

        # Save the results
        cv2.imwrite(output_foreground_path, foreground)
        cv2.imwrite(output_background_path, background)
        print(f"Foreground saved to {output_foreground_path}")
        print(f"Background saved to {output_background_path}")

        # Display the results (optional)
        cv2.imshow("Foreground", foreground)
        cv2.imshow("Background", background)
        cv2.waitKey(20000)  # Wait indefinitely until a key is pressed
        cv2.destroyAllWindows()