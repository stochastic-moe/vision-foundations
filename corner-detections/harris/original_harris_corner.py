import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


# 1. Create a black RGB background image
width, height = 200, 200
bg_color = (100, 0, 60)
image = np.full((height, width, 3), bg_color, dtype=np.uint8)

# 2. Define coordinates
corner = (100, 100)      # Intersection point
line1_end = (199, 130) # End of first line
line2_end = (130, 199) # End of second line

# 3. Fill the region between lines
# The area is treated as a polygon with vertices: corner -> end1 -> end2
pts = np.array([corner, line1_end, (199, 199), line2_end, ], np.int32)
cv2.fillPoly(image, [pts], color=(0, 255, 0)) # Green fill

plt.imshow(image)
plt.axis('off') # Optional: hides the coordinate axes
plt.show(block=False)

# 4. Compute x and y derivatives using Sobel (use CV_64F to avoid overflow)
# ksize=3 or ksize=-1 (Scharr) are standard choices
grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

blur = np.zeros((width, height, 3, 360))

# 2. Define the angle (e.g., 45 degrees) and compute directional derivative 
for theta in range(360):
    theta_rad = np.deg2rad(theta)
    directional_derivative = grad_x * np.cos(theta_rad) - grad_y * np.sin(theta_rad)
    sum_square_diff = directional_derivative ** 2
    # Apply Gaussian Blur  (5, 5) is the kernel size, 0 tells OpenCV to calculate sigma based on kernel size
    blur[:, :, :, theta] = cv2.GaussianBlur(sum_square_diff, (5, 5), 0)

blur_max  = blur.max(axis=3, keepdims=True)
blur_norm = np.divide(blur, blur_max, out=np.zeros_like(blur, dtype=float), where=blur_max!=0)

blur_norm = blur_norm.max(axis=2)
    
print(blur_norm.shape)


# 3. Create the polar plot for corner point (100, 100)
theta = np.deg2rad(np.arange(0, 360))
r = blur_norm[100, 100, :]
print(r.shape)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta, r, color='b', linewidth=2)

# 4. Customize the plot
ax.set_title('Polar Plot (0 to 359 Degrees)', va='bottom')
ax.set_rlabel_position(-22.5)  # Move radial labels away from the plot line

plt.show(block=False)

# 3. Create the polar plot for flat point (90, 45)
theta = np.deg2rad(np.arange(0, 360))
r = blur_norm[90, 45, :]
print(r.shape)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta, r, color='b', linewidth=2)

# 4. Customize the plot
ax.set_title('Polar Plot (0 to 359 Degrees)', va='bottom')
ax.set_rlabel_position(-22.5)  # Move radial labels away from the plot line

# 3. Create the polar plot for flat point (150, 115)
theta = np.deg2rad(np.arange(0, 360))
r = blur_norm[115, 150, :]
print(r.shape)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta, r, color='b', linewidth=2)

# 4. Customize the plot
ax.set_title('Polar Plot (0 to 359 Degrees)', va='bottom')
ax.set_rlabel_position(-22.5)  # Move radial labels away from the plot line

plt.show()
