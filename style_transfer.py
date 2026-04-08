import numpy as np
import cv2
import matplotlib.pyplot as plt

# -----------------------------
# Load Images
# -----------------------------
content = cv2.imread("content.jpg")
style = cv2.imread("style2.jpg")

if content is None or style is None:
    raise ValueError("Check image paths!")

content = cv2.cvtColor(content, cv2.COLOR_BGR2RGB)
style = cv2.cvtColor(style, cv2.COLOR_BGR2RGB)

style = cv2.resize(style, (content.shape[1], content.shape[0]))

content = content.astype(np.float32)
style = style.astype(np.float32)

print("Images loaded")

# -----------------------------
# SAFE COLOR BIAS
# -----------------------------
style_mean = style.mean(axis=(0, 1))
content_mean = content.mean(axis=(0, 1))

content = content + (style_mean - content_mean) * 0.3
content = np.clip(content, 0, 255)

# -----------------------------
# AUTO MASK
# -----------------------------
gray = cv2.cvtColor(content.astype(np.uint8), cv2.COLOR_RGB2GRAY)

edges = cv2.Canny(gray, 100, 200)
edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros_like(gray, dtype=np.float32)
largest_contour = max(contours, key=cv2.contourArea)
cv2.drawContours(mask, [largest_contour], -1, 1, thickness=cv2.FILLED)

mask = cv2.GaussianBlur(mask, (31, 31), 0)
mask_3d = np.stack([mask] * 3, axis=2)

print("Mask created")

# -----------------------------
# SVD STYLE TRANSFER
# -----------------------------
k = 60
styled_channels = []

for i in range(3):
    C = content[:, :, i]
    S = style[:, :, i]

    Uc, Sc, Vc = np.linalg.svd(C, full_matrices=False)
    Us, Ss, Vs = np.linalg.svd(S, full_matrices=False)

    Sc[k:] = 0
    Ss[k:] = 0

    Ss = Ss / np.max(Ss) * np.max(Sc)
    Sigma_new = np.diag(Ss)

    styled_channel = Uc @ Sigma_new @ Vc
    styled_channel = cv2.normalize(styled_channel, None, 0, 255, cv2.NORM_MINMAX)

    styled_channels.append(styled_channel)

styled_img = np.stack(styled_channels, axis=2)
styled_img = np.clip(styled_img, 0, 255).astype(np.uint8)

print("SVD done")

# -----------------------------
# DIRECTIONAL TEXTURE
# -----------------------------
style_gray = cv2.cvtColor(style.astype(np.uint8), cv2.COLOR_RGB2GRAY)

grad_x = cv2.Sobel(style_gray, cv2.CV_32F, 1, 0, ksize=5)
grad_y = cv2.Sobel(style_gray, cv2.CV_32F, 0, 1, ksize=5)

texture = np.sqrt(grad_x**2 + grad_y**2)
texture = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX)
texture_3d = np.stack([texture] * 3, axis=2).astype(np.uint8)


# -----------------------------
# MULTI-SWIRL TRANSFORM
# -----------------------------
def multi_swirl(img, centers, strength=5, radius=250):
    h, w = img.shape[:2]
    y, x = np.indices((h, w), dtype=np.float32)

    dx = np.zeros_like(x)
    dy = np.zeros_like(y)

    for cx, cy in centers:
        x_c = x - cx
        y_c = y - cy
        r = np.sqrt(x_c**2 + y_c**2)

        theta = np.arctan2(y_c, x_c)
        factor = np.exp(-(r**2) / (radius**2))

        theta_new = theta + strength * factor

        dx += r * np.cos(theta_new) - x_c
        dy += r * np.sin(theta_new) - y_c

    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    return cv2.remap(
        img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )


# Choose multiple swirl centers
h, w = content.shape[:2]
centers = [(w // 2, h // 3), (w // 3, h // 2), (2 * w // 3, h // 2)]

# Apply swirl to texture
swirled_texture = multi_swirl(texture_3d, centers, strength=10, radius=300)

# Blend texture
enhanced_img = styled_img.astype(np.float32)
enhanced_img = 0.8 * enhanced_img + 0.2 * swirled_texture
enhanced_img = np.clip(enhanced_img, 0, 255).astype(np.uint8)

# Optional: swirl final image slightly
swirled = multi_swirl(enhanced_img, centers, strength=6, radius=350)
enhanced_img = mask_3d * swirled + (1 - mask_3d) * enhanced_img

print("Swirl applied")

# -----------------------------
# APPLY MASK
# -----------------------------
final_img = mask_3d * enhanced_img + (1 - mask_3d) * content
final_img = np.clip(final_img, 0, 255).astype(np.uint8)

print("Final image created")

# -----------------------------
# DISPLAY
# -----------------------------
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.title("Content")
plt.imshow(content.astype(np.uint8))
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("Style")
plt.imshow(style.astype(np.uint8))
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("Final Output")
plt.imshow(final_img)
plt.axis("off")

plt.show()
