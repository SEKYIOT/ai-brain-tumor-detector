import numpy as np
import matplotlib.pyplot as plt

# 1. Create Real Brain MRI
size = 200
brain = np.zeros((size, size), dtype=np.uint8)
y, x = np.ogrid[:size, :size]
center = size // 2

# Brain tissue
brain_mask = (x - center)**2 + (y - center)**2 <= 80**2
brain[brain_mask] = 40

# Tumor (hidden)
tumor_mask = (x - 120)**2 + (y - 100)**2 <= 15**2
brain[tumor_mask] = 255

# 2. AI DETECTION - Find bright pixels automatically
tumor_pixels = np.where(brain > 200)
num_pixels = len(tumor_pixels[0])

# Find tumor center
tumor_y = int(np.mean(tumor_pixels[0]))
tumor_x = int(np.mean(tumor_pixels[1]))

# Calculate size in mm (1 pixel = 0.5mm fake)
diameter_mm = 15 * 2 * 0.5

print(f"=== AI REPORT ===")
print(f"Tumor FOUND at X={tumor_x}, Y={tumor_y}")
print(f"Pixels: {num_pixels}")
print(f"Diameter: {diameter_mm} mm")
print(f"ALERT: Abnormality detected!")

# 3. Draw with RED circle
fig, ax = plt.subplots()
ax.imshow(brain, cmap='gray')
ax.set_title(f'AI Detected Tumor - {diameter_mm}mm')

# Red circle around tumor - AI style
circle = plt.Circle((tumor_x, tumor_y), 20, color='red', fill=False, linewidth=2)
ax.add_patch(circle)

# Red X at center
ax.plot(tumor_x, tumor_y, 'r+', markersize=12, markeredgewidth=2)

ax.axis('off')
plt.savefig("ai_tumor_detected.png")
print("SAVED as ai_tumor_detected.png")
plt.show()