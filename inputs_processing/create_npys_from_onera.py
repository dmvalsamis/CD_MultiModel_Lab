import os
import numpy as np
from skimage import io
import pandas as pd
from scipy.ndimage import zoom

#### Images locations  ####
IMAGES_PATH = '/home/dvalsamis/Documents/data/CBMI/CBMI_0.3/CBMI_0.3/Dataset/CBMI_Images/'
TRAIN_LABEL_PATH = '/home/dvalsamis/Documents/data/CBMI/CBMI_0.3/CBMI_0.3/Dataset/CBMI_Train_Labels/'
TEST_LABEL_PATH = '/home/dvalsamis/Documents/data/CBMI/CBMI_0.3/CBMI_0.3/Dataset/CBMI_Test_Labels/'

#### Save targets for the npy files ####
SAVE_IMAGES = '/home/dvalsamis/Documents/data/CBMI/CBMI_0.3/CBMI_0.3/NPY_dataset/CBMI_Images_NPY/'
SAVE_TRAIN_LABELS = '/home/dvalsamis/Documents/data/CBMI/CBMI_0.3/CBMI_0.3/NPY_dataset/CBMI_Train_Labels_NPY/'
SAVE_TEST_LABELS = '/home/dvalsamis/Documents/data/CBMI/CBMI_0.3/CBMI_0.3/NPY_dataset/CBMI_Test_Labels_NPY/'

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
    image_to_save = read_sentinel_img_eq20(path_to_images)
    np.save(path_to_target + str(prefix) +"_"+ str(suffix) + ".npy", image_to_save)
    
    return image_to_save
    
#Method to read the RGB bands of a given image, used ONLY for the image pairs    
def read_sentinel_img(path):
    """Read cropped Sentinel-2 image: RGB bands."""
    im_name = os.listdir(path)[0][:-7]

    r = io.imread(path + im_name + "B04.tif")
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")
    vnir = io.imread(path + im_name + "B08.tif")

    I = np.stack((r,g,b,vnir),axis=2).astype('float')
    print("RGB image shape:", I.shape)

    return I

def adjust_shape(I, s):
    """Adjust shape of grayscale image I to s."""
    # crop if necesary  
    I = I[:s[0],:s[1]]
    si = I.shape

    # pad if necessary 
    p0 = max(0,s[0] - si[0])
    p1 = max(0,s[1] - si[1])

    return np.pad(I,((0,p0),(0,p1)),'edge')


def read_sentinel_img_eq20(path):
    """Read cropped Sentinel-2 image: bands with resolution less than or equals to 20m."""
    im_name = os.listdir(path)[0][:-7]
    print(im_name)

    r = io.imread(path+'/' + im_name + "B04.tif")
    s = r.shape

    ir1 = adjust_shape(zoom(io.imread(path +'/'+ im_name + "B05.tif"),2),s)
    ir2 = adjust_shape(zoom(io.imread(path +'/'+ im_name + "B06.tif"),2),s)
    ir3 = adjust_shape(zoom(io.imread(path +'/'+ im_name + "B07.tif"),2),s)
    nir2 = adjust_shape(zoom(io.imread(path +'/'+ im_name + "B8A.tif"),2),s)
    swir2 = adjust_shape(zoom(io.imread(path +'/'+ im_name + "B11.tif"),2),s)
    swir3 = adjust_shape(zoom(io.imread(path +'/'+ im_name + "B12.tif"),2),s)

    I = np.stack((ir1,ir2,ir3,nir2,swir2,swir3),axis=2).astype('float')    

    return I

def read_changemask(cm_path):
    cm = io.imread(cm_path + '/cm/cm.tif', as_gray=True) != 0
    return cm

# Method to make image pairs with the CBMI images 
def make_image_pairs(folder, loc):
    path1 = os.path.join(IMAGES_PATH, folder, 'img1_cropped')
    path2 = os.path.join(IMAGES_PATH, folder, 'img2_cropped')
    print("Path1: --->", path1)
    print("Path2: --->", path2)
    
    img1 = save_image(path1, loc, "a", SAVE_IMAGES)
    img2 = save_image(path2, loc, "b", SAVE_IMAGES)
    
    if img1.shape != img2.shape:
        print("correction on shapes")
        s1 = img1.shape
        s2 = img2.shape
        
        # Calculate the padding values to make the shapes equal
        pad_height = max(0, s1[0] - s2[0])
        pad_width = max(0, s1[1] - s2[1])
        
        # Pad img2 to match the shape of img1
        img2 = np.pad(img2, ((0, pad_height), (0, pad_width), (0, 0)), 'edge')
        print('Img2 after correction:', img2.shape)
        
        # Save the corrected img2
        np.save(os.path.join(SAVE_IMAGES, f"{loc}_b.npy"), img2)

   
folder_list = [f for f in os.listdir(IMAGES_PATH) if f.endswith('.txt')==False]


for i in range(len(folder_list)):
    make_image_pairs(folder_list[i], i)
    images_set.loc[i,'name'] = str(folder_list[i])
    images_set.loc[i,'pair1'] = str(i)+'_a.npy'
    images_set.loc[i,'pair2'] = str(i)+'_b.npy'
    
print("DONE with image pairs!")
print("Saving dataframe")    
images_set.to_csv(SAVE_IMAGES + 'CBMI_set.csv', index=False)

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