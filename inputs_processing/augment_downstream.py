import albumentations as A
import numpy as np
import pandas as pd

def CDaugment_patches(path_to_images, path_to_train, path_to_test, target_path_train, target_path_test, num_of_patches):
    
    dataset_train = pd.DataFrame(columns=['pair1','pair2','change_mask'])
    dataset_test = pd.DataFrame(columns=['pair1','pair2','change_mask'])
    
    transform = A.Compose(
    [A.RandomCrop(width=96, height=96),
     A.RandomRotate90(5),
     A.HorizontalFlip(p=0.5),
     A.VerticalFlip(p=1)
    ],
    additional_targets={'image0': 'image'}
    )
    
    #train
    train_setdf = pd.read_csv(path_to_train+'train_set.csv', dtype=str)
    #test
    test_setdf = pd.read_csv(path_to_test+'test_set.csv', dtype=str)
    
    ####################################train#############################################################
    pos = 0
    for index in range(len(train_setdf)):
        img1 = np.load(path_to_images+train_setdf['pair1'][index])
        img2 = np.load(path_to_images+train_setdf['pair2'][index])
        mask = np.load(path_to_train+train_setdf['change_mask'][index])
        
        #make 100 patches per image pair
        for i in range(num_of_patches):
            
            transformed = transform(image=img1, image0=img2, mask = mask)
            #left
            prefix_l = train_setdf['pair1'][index][:-4]
            np.save(target_path_train + prefix_l + '_l_'+ str(i) +'.npy',transformed['image'])
            #mask
            prefix_m = train_setdf['change_mask'][index][:-4]
            np.save(target_path_train + prefix_m +'_'+ str(i) +'.npy',transformed['mask'])
            #right
            prefix_r = train_setdf['pair2'][index][:-4]
            np.save(target_path_train + prefix_r + '_r_'+ str(i) + '.npy',transformed['image0'])
            #save in a dataframe
            dataset_train.loc[pos,'pair1'] = prefix_l + '_l_'+ str(i) +'.npy'
            dataset_train.loc[pos,'pair2'] =  prefix_r + '_r_'+ str(i) + '.npy'
            dataset_train.loc[pos,'change_mask'] = prefix_m +'_'+ str(i) +'.npy'
            pos += 1
            
    dataset_train.to_csv(target_path_train + 'dataset_train.csv', index=False)
    #################################test#####################################################
    pos = 0
    for index in range(len(test_setdf)):
        img1 = np.load(path_to_images+test_setdf['pair1'][index])
        img2 = np.load(path_to_images+test_setdf['pair2'][index])
        mask = np.load(path_to_test+test_setdf['change_mask'][index])
        
        #make 100 patches per image pair
        for i in range(num_of_patches):
            
            transformed = transform(image=img1, image0=img2, mask = mask)
            #left
            prefix_l = test_setdf['pair1'][index][:-4]
            np.save(target_path_test + prefix_l + '_l_'+ str(i) +'.npy',transformed['image'])
            #mask
            prefix_m = test_setdf['change_mask'][index][:-4]
            np.save(target_path_test + prefix_m +'_'+ str(i) +'.npy',transformed['mask'])
            #right
            prefix_r = test_setdf['pair2'][index][:-4]
            np.save(target_path_test + prefix_r + '_r_'+ str(i) + '.npy',transformed['image0'])
            #save in a dataframe
            dataset_test.loc[pos,'pair1'] = prefix_l + '_l_'+ str(i) +'.npy'
            dataset_test.loc[pos,'pair2'] =  prefix_r + '_r_'+ str(i) + '.npy'
            dataset_test.loc[pos,'change_mask'] = prefix_m +'_'+ str(i) +'.npy'
            pos += 1
            
    dataset_test.to_csv(target_path_test + 'dataset_test.csv', index=False)


# Paths and parameters
path_to_images = '/home/dvalsamis/Documents/data/sysu/SYSU_NPY/total_NPY/'
path_to_train = '/home/dvalsamis/Documents/data/sysu/SYSU_NPY/train_labels_NPY/'
path_to_test = '/home/dvalsamis/Documents/data/sysu/SYSU_NPY/test_labels_NPY/'
target_path_train = '/home/dvalsamis/Documents/data/sysu/SYSU_NPY/aug_train_data/'
target_path_test = '/home/dvalsamis/Documents/data/sysu/SYSU_NPY/aug_test_data/'
num_of_patches = 5

# Call the function
CDaugment_patches(path_to_images, path_to_train, path_to_test, target_path_train, target_path_test, num_of_patches)

print("Done")