import os
import numpy as np
from skimage import io
import pandas as pd
from scipy.ndimage import zoom
import glob


#### Images locations  ####
IMAGES_PATH = '/home/dvalsamis/Documents/data/LEVIR-CD/original_set/total/'
TRAIN_LABEL_PATH = '/home/dvalsamis/Documents/data/LEVIR-CD/original_set/train/label/'
TEST_LABEL_PATH = '/home/dvalsamis/Documents/data/LEVIR-CD/original_set/test/label/'

#### Save targets for the npy files ####
SAVE_IMAGES = '/home/dvalsamis/Documents/data/LEVIR-CD/Levir_NPY/total_NPY/'
SAVE_TRAIN_LABELS = '/home/dvalsamis/Documents/data/LEVIR-CD/Levir_NPY/train_labels_NPY/'
SAVE_TEST_LABELS = '/home/dvalsamis/Documents/data/LEVIR-CD/Levir_NPY/test_labels_NPY/'

images_set = pd.DataFrame(columns=['pair1','pair2','name'])
train_set = pd.DataFrame(columns=['pair1','pair2','change_mask'])
test_set = pd.DataFrame(columns=['pair1','pair2','change_mask'])

#Method to save the change masks to npy files
def save_labels(path,folder,prefix, target, index, train=False, test=False):
    cm = read_changemask(path+folder)
    if train:
        train_set.loc[index,'pair1'] = str(prefix)+"_a.npy"
        train_set.loc[index,'pair2'] = str(prefix)+"_b.npy"
        train_set.loc[index,'change_mask'] = str(prefix)+"_cm.npy"
    if test:
        test_set.loc[index,'pair1'] = str(prefix)+"_a.npy"
        test_set.loc[index,'pair2'] = str(prefix)+"_b.npy"
        test_set.loc[index,'change_mask'] = str(prefix)+"_cm.npy"
    np.save(target + str(prefix)+"_cm.npy",cm)




# final method that saves the images in a npy format
def save_image(path_to_images, prefix, suffix, path_to_target):
    """Save the image in npy format."""
    image_to_save = read_rgb_img(path_to_images)  # Use the modified function to read PNG
    np.save(os.path.join(path_to_target, f"{prefix}_{suffix}.npy"), image_to_save)
    return image_to_save

    
#Method to read the RGB bands of a given image, used ONLY for the image pairs    
def read_rgb_img(path):
    """Read RGB PNG image without normalization."""
    img_files = glob.glob(os.path.join(path, '*.png'))  # Adjust if naming convention differs
    if not img_files:
        raise FileNotFoundError(f"No image files found at {path}")
    
    img = io.imread(img_files[0])  # Reading the first image matching the pattern
    print("Image shape:", img.shape)
    return img


def adjust_shape(I, s):
    """Adjust shape of grayscale image I to s."""
    # crop if necesary  
    I = I[:s[0],:s[1]]
    si = I.shape

    # pad if necessary 
    p0 = max(0,s[0] - si[0])
    p1 = max(0,s[1] - si[1])

    return np.pad(I,((0,p0),(0,p1)),'edge')


def read_changemask(cm_path):
    """Read change mask from PNG files, convert values from [0, 255] to [0, 1]. Ensure the file exists."""
    cm_files = glob.glob(cm_path)  # Fetch all PNG files in directory
    if not cm_files:
        raise FileNotFoundError(f"No mask files found at {cm_path}")
    
    cm = io.imread(cm_files[0], as_gray=True)  # Safely read the first file, if it exists
    cm_normalized = (cm / 255.0).astype(int)  # Normalize to [0, 1] and convert to integer
    
    return cm_normalized


# Method to make image pairs with the CBMI images 
def make_image_pairs(loc):
    path1 = os.path.join(IMAGES_PATH, 'A')
    path2 = os.path.join(IMAGES_PATH, 'B')
    print("Path1: --->", path1)
    print("Path2: --->", path2)

    img1 = save_image(path1, loc, "a", SAVE_IMAGES)
    img2 = save_image(path2, loc, "b", SAVE_IMAGES)
    
    # Adjust shape handling to ensure consistency
    if img1.shape != img2.shape:
        print("correction on shapes")
        s1 = img1.shape
        s2 = img2.shape
        new_shape = (min(s1[0], s2[0]), min(s1[1], s2[1]))
        img1 = img1[:new_shape[0], :new_shape[1]]
        img2 = img2[:new_shape[0], :new_shape[1]]
        np.save(os.path.join(SAVE_IMAGES, f"{loc}_a.npy"), img1)
        np.save(os.path.join(SAVE_IMAGES, f"{loc}_b.npy"), img2)


path1 = os.path.join(IMAGES_PATH, 'A')
folder_list = [f for f in os.listdir(path1) if f.endswith('.txt')==False]


for i in range(10):
    make_image_pairs(i)
    images_set.loc[i,'name'] = str(folder_list[i])
    images_set.loc[i,'pair1'] = str(i)+'_a.npy'
    images_set.loc[i,'pair2'] = str(i)+'_b.npy'
    
print("DONE with image pairs!")
print("Saving dataframe")    
images_set.to_csv(SAVE_IMAGES + 'LEVIR_set.csv', index=False)

train_label_list = [f for f in os.listdir(TRAIN_LABEL_PATH ) if f.endswith('.txt')==False]
    
pos=0    
for i in range(len(folder_list)):
    for j in range(len(train_label_list)):
        if folder_list[i] == train_label_list[j]:
            save_labels(TRAIN_LABEL_PATH,train_label_list[j], i, SAVE_TRAIN_LABELS, pos, train=True)
            pos+=1
print("DONE with train set!")
print("Saving dataframe")    
train_set.to_csv(SAVE_TRAIN_LABELS + 'train_set.csv', index=False)
            
test_label_list = os.listdir(TEST_LABEL_PATH)

pos = 0
for i in range(len(folder_list)):
    for j in range(len(test_label_list)):
        if folder_list[i] == test_label_list[j]:
            save_labels(TEST_LABEL_PATH,test_label_list[j], i, SAVE_TEST_LABELS, pos, train=False, test=True)
            pos += 1

print("DONE with test set!")
print("Saving dataframe")    
test_set.to_csv(SAVE_TEST_LABELS + 'test_set.csv', index=False)

print("DONE")