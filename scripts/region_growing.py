import numpy as np

def region_growing(image, seed, threshold):
    """
    Basic 2D region growing algorithm skeleton.
    
    Args:
        image (np.ndarray): 2D array representing the image or feature map.
        seed (tuple): (y, x) coordinates of the initial seed point.
        threshold (float): Similarity threshold for including a pixel in the region.
        
    Returns:
        np.ndarray: Binary mask of the grown region.
    """
    # Assuming input is a 2D array
    if len(image.shape) > 2:
        image = np.mean(image, axis=2)
        
    height, width = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # List of pixels to check
    pixel_queue = [seed]
    
    # Set the seed pixel to visited and part of the region
    mask[seed[0], seed[1]] = 1
    
    # Intensity/value of the seed pixel
    seed_value = image[seed[0], seed[1]]
    
    # 4-connectivity neighbors
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while pixel_queue:
        current_pixel = pixel_queue.pop(0)
        y, x = current_pixel
        
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            
            # Check bounds
            if 0 <= ny < height and 0 <= nx < width:
                # Check if already visited
                if mask[ny, nx] == 0:
                    # Check similarity criterion
                    if abs(float(image[ny, nx]) - float(seed_value)) <= threshold:
                        mask[ny, nx] = 1
                        pixel_queue.append((ny, nx))
                        
    return mask

if __name__ == "__main__":
    # Example usage with dummy data
    print("Testing basic 2D region growing...")
    dummy_image = np.array([
        [10, 10, 10, 50, 50],
        [10, 12, 11, 50, 50],
        [10, 11, 10, 50, 50],
        [50, 50, 50, 50, 50],
        [50, 50, 50, 50, 50]
    ])
    
    seed_point = (1, 1) # Coordinates of value '12'
    thresh = 5
    
    result_mask = region_growing(dummy_image, seed_point, thresh)
    print("Original Image:\n", dummy_image)
    print("\nRegion Mask (Seed {}, Threshold {}):\n".format(seed_point, thresh), result_mask)
