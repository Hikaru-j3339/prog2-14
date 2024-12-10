import matplotlib.pyplot as plt
import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms


ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
)

print (f'num of datasets: {len(ds_train)}')

image, target = ds_train[0]
print(type(image), target)

plt.imshow(image)
plt.title(target)
plt.show()

image_tensor = transforms.functional.to_image(image)
image = transforms.functional.to_dtype(image_tensor,dtype=torch.float32, scale=True)
print(image_tensor.shape, image_tensor.dtype)
print(image.min(), image.max())