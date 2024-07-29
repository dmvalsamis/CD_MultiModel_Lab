import albumentations as A
import numpy as np
import pandas as pd

def augment_patches(path_to_images, target_path,num_of_patches, onera=False):
    
    dataset = pd.DataFrame(columns=['pair1','pair2','overlap'])
    
    if onera == False:
        images_df = pd.read_csv(path_to_images + "s2mtcp_set.csv")
    else:
        images_df = pd.read_csv(path_to_images + "onera_set.csv")
        
    transform_overlap = A.Compose(
    [A.RandomCrop(width=96, height=96),
     A.RandomRotate90(p=0.5),
     A.HorizontalFlip(p=0.5),
     A.VerticalFlip(p=0.5)
    ],
    additional_targets={'image0': 'image'}
    )
    
    transform_non = A.Compose(
    [A.RandomCrop(width=96, height=96),
     A.RandomRotate90(p=0.5),
     A.HorizontalFlip(p=0.5),
     A.VerticalFlip(p=0.5)
    ]
    )
    pos = 0
    for index in range(len(images_df)):
        img1 = np.load(path_to_images+images_df['pair1'][index])
        img2 = np.load(path_to_images+images_df['pair2'][index])
        #make num_of_patches overlaping patches for this image pair
        for i in range(num_of_patches):
            #left
            transformed = transform_overlap(image=img1, image0=img2)
            prefix = images_df['pair1'][index][:-4]
            np.save( target_path + prefix + '_lo_'+ str(i) +'.npy',transformed['image'])
            #save in a dataframe
            dataset.loc[pos,'pair1'] = prefix + '_lo_'+ str(i) +'.npy'
            dataset.loc[pos,'overlap'] = 0
            #right
            prefix = images_df['pair2'][index][:-4]
            np.save( target_path + prefix + '_ro_'+ str(i) + '.npy',transformed['image0'])
            #save in a dataframe
            dataset.loc[pos,'pair2'] =  prefix + '_ro_'+ str(i) + '.npy'
            pos += 1
                
        #make num_of_patches non_overlaping patches for this image pair
        for i in range(num_of_patches):
            #left
            transformed = transform_non(image=img1)
            prefix = images_df['pair1'][index][:-4]
            np.save( target_path + prefix + '_ln_'+ str(i) +'.npy',transformed['image'])
            #save in a dataframe
            dataset.loc[pos,'pair1'] = prefix + '_ln_'+ str(i) +'.npy'
            dataset.loc[pos,'overlap'] = 1
            #right
            transformed = transform_non(image=img2)
            prefix = images_df['pair2'][index][:-4]
            np.save( target_path + prefix + '_rn_'+ str(i) + '.npy',transformed['image'])
            #save in a dataframe
            dataset.loc[pos,'pair2'] =  prefix + '_rn_'+ str(i) + '.npy'
            pos+=1
               
    dataset.to_csv(target_path + 'dataset.csv', index=False)