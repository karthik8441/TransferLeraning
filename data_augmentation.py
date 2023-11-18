import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from PIL import Image
import os

# Define the path to the directory containing the images
data_dir = './data/original/seg_images'

# Define the batch size and image dimensions
batch_size = 32
img_width, img_height = 640, 480

# Define the data augmentation transforms
data_transforms = transforms.Compose([
    transforms.RandomRotation(40),
    transforms.RandomResizedCrop(img_width, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the image dataset
image_dataset = ImageFolder(os.path.join(data_dir), transform=data_transforms)

# Save the augmented images to a new directory
save_dir = '../data/original/seg_augmented_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for i, (image, label) in enumerate(image_dataset):
    image_name = f"image_{i}.jpg"
    save_path = os.path.join(save_dir, image_name)
    image = image.permute(1, 2, 0).numpy()  # Convert tensor to numpy array
    image = (image * 0.5 + 0.5) * 255  # Denormalize the image
    image = image.astype('uint8')
    Image.fromarray(image).save(save_path)

# Split the dataset into training and testing sets
num_train = int(len(image_dataset) * 0.8)
num_test = len(image_dataset) - num_train
train_dataset, test_dataset = torch.utils.data.random_split(image_dataset, [num_train, num_test])

# Create data loaders for training and testing sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Print some information about the data
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of testing samples: {len(test_dataset)}")
print(f"Class labels: {image_dataset.classes}")


# Prepare data for model
class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target
    
    def __len__(self):
        return len(self.dataset)

train_data = MyDataset(train_dataset)
test_data = MyDataset(test_dataset)