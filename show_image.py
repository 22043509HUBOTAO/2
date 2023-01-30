import ImageTransform as it
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# test

image_file_path = './prediction/covid/1_000.png'
img = Image.open(image_file_path)

plt.imshow(img)
plt.show()

size = 224
mean = (0.5, 0.5, 0.5, 0.5)
std = (0.220, 0.220, 0.220, 0.220)

transform = it.ImageTransform(size, mean, std)
img_transformed = transform(img, phase="train")

img_transformed = img_transformed.numpy().transpose((1, 2, 0))
img_transformed = np.clip(img_transformed, 0, 1)
plt.imshow(img_transformed)
plt.show()


