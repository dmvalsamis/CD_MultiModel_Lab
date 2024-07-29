import numpy as np
import matplotlib.pyplot as plt

# Paths to the .npy files
file_path_cm = '/home/dvalsamis/Documents/data/sysu/SYSU_NPY/train_labels_NPY/3_cm.npy'
file_path_a = '/home/dvalsamis/Documents/data/sysu/SYSU_NPY/total_NPY/3_a.npy'
file_path_b = '/home/dvalsamis/Documents/data/sysu/SYSU_NPY/total_NPY/3_b.npy'

# Load the numpy arrays from the files
image_cm = np.load(file_path_cm)
image_a = np.load(file_path_a)
image_b = np.load(file_path_b)

# Calculate and print the distinct values in the change mask
distinct_values = np.unique(image_cm)
print("Distinct values in the change mask:", distinct_values)

# Create a figure to display the images
fig, ax = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

# Display the change mask
ax[0].imshow(image_cm, cmap='gray')  # Use a gray colormap for better clarity for masks
ax[0].set_title('Change Mask (0_cm)')
ax[0].axis('off')  # Hide axes

# Display the first image
ax[1].imshow(image_a)
ax[1].set_title('Image 0_a')
ax[1].axis('off')  # Hide axes

# Display the second image
ax[2].imshow(image_b)
ax[2].set_title('Image 0_b')
ax[2].axis('off')  # Hide axes

# Show the plot
plt.show()

import numpy as np




# Check if the files have different contents
are_different = not np.array_equal(image_a, image_b)

# Output the result
if are_different:
    print("The files contain different data.")
else:
    print("The files are identical.")

