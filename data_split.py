from imutils import paths
import numpy as np
import shutil
import os

BLOOD_DATASET_PATH = "../data/combined"
out_path = "../data/image_files"
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2

TRAIN_folder = os.path.join(out_path, "train")
VAL_folder = os.path.join(out_path, "val")
TEST_folder = os.path.join(out_path, "test")

def copy_images(imagePaths, folder):
    # check if the destination folder exists and if not create it
    if not os.path.exists(folder):
        os.makedirs(folder)
    # loop over the image paths
    for path in imagePaths:
        # grab image name and its label from the path and create a placeholder corresponding to the separate label folder
        imageName = path.split(os.path.sep)[-1]
        label = path.split(os.path.sep)[-2]
        labelFolder = os.path.join(folder, label)
        # check to see if the label folder exists and if not create it
        if not os.path.exists(labelFolder):
            os.makedirs(labelFolder)
        # construct the destination image path and copy the current image to it
        destination = os.path.join(labelFolder, imageName)
        print('------------copied---',destination)
        shutil.copy(path, destination)

print("[INFO] loading image paths...")
imagePaths = list(paths.list_images(BLOOD_DATASET_PATH))
np.random.shuffle(imagePaths)

# generate training, validation, and test paths
valPathsLen = int(len(imagePaths) * VAL_SPLIT)
testPathsLen = int(len(imagePaths) * TEST_SPLIT)
trainPathsLen = len(imagePaths) - valPathsLen - testPathsLen
trainPaths = imagePaths[:trainPathsLen]
valPaths = imagePaths[trainPathsLen:trainPathsLen+valPathsLen]
testPaths = imagePaths[trainPathsLen+valPathsLen:]

# copy the training, validation, and test images to their respective directories
print("[INFO] copying training, validation, and test images...")
copy_images(trainPaths, TRAIN_folder)
copy_images(valPaths, VAL_folder)
copy_images(testPaths, TEST_folder)
