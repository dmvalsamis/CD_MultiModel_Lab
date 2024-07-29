import os
import numpy as np
from skimage import io
import pandas as pd
from scipy.ndimage import zoom
import glob
from pandas import DataFrame



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
    image_to_save = read_single_png_image(path_to_images)
    np.save(path_to_target + str(prefix) +"_"+ str(suffix) + ".npy", image_to_save)
    
    return image_to_save
    


def adjust_shape(I, s):
    """Adjust shape of grayscale image I to s."""
    # crop if necesary  
    I = I[:s[0],:s[1]]
    si = I.shape

    # pad if necessary 
    p0 = max(0,s[0] - si[0])
    p1 = max(0,s[1] - si[1])

    return np.pad(I,((0,p0),(0,p1)),'edge')



def read_single_png_image(path):
    """Read a .png file, either directly from the path or from a directory."""
    if os.path.isdir(path):
        # If the path is a directory, find the first .png file in it
        files = [f for f in os.listdir(path) if f.endswith('.png')]
        if not files:
            raise FileNotFoundError("No PNG files found in the directory.")
        file_path = os.path.join(path, files[0])  # First .png file
    elif os.path.isfile(path) and path.endswith('.png'):
        # If the path is directly a .png file
        file_path = path
    else:
        raise FileNotFoundError("No PNG file found at the provided path.")

    print(f"Loading image: {file_path}")
    
    # Load the image using skimage.io.imread
    image = io.imread(file_path)
    
    return image


def read_changemask(cm_path):
    """Read change mask from PNG files, convert values from [0, 255] to [0, 1]. Ensure the file exists."""
    cm_files = glob.glob(cm_path)  # Fetch all PNG files in directory
    if not cm_files:
        raise FileNotFoundError(f"No mask files found at {cm_path}")
    
    cm = io.imread(cm_files[0], as_gray=True)  # Safely read the first file, if it exists
    cm_normalized = (cm / 255.0).astype(int)  # Normalize to [0, 1] and convert to integer
    
    return cm_normalized

def make_image_pairs(images_path, save_images):
    folder_a = os.path.join(images_path, 'A')
    folder_b = os.path.join(images_path, 'B')
    
    images_a = [f for f in os.listdir(folder_a) if f.endswith('.png')]
    images_b = set(os.listdir(folder_b))  # Use a set for quick lookup

    images_set = pd.DataFrame(columns=['name', 'pair1', 'pair2'])

    for i, filename in enumerate(images_a):
        if filename in images_b:  # Check if the corresponding image exists in folder 'B'
            path1 = os.path.join(folder_a, filename)
            path2 = os.path.join(folder_b, filename)
            print(f"Processing pair {filename}:")
            
            img1 = save_image(path1, i, "a", save_images)
            img2 = save_image(path2, i, "b", save_images)

            # Store information in the DataFrame
            images_set.loc[i] = {'name': filename, 'pair1': f"{i}_a.npy", 'pair2': f"{i}_b.npy"}

    print("Done with image pairs!")
    print("Saving dataframe")
    images_set.to_csv(os.path.join(save_images, 'Levir_set.csv'), index=False)

# Call the function
#make_image_pairs(IMAGES_PATH, SAVE_IMAGES)

path1 = os.path.join(IMAGES_PATH, 'A')
folder_list = [f for f in os.listdir(path1) if f.endswith('.txt')==False]

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