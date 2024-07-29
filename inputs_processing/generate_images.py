import numpy as np
from tensorflow import keras

class DataGenerator(keras.utils.Sequence):
    def __init__(self, path, df, x1_col, x2_col, y_col, batch_size=32, num_classes=None, shuffle=False, onera=False, norm=False):
        self.path = path
        self.batch_size = batch_size
        self.df = df
        self.indices = self.df.index.tolist()
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.x1_col = x1_col
        self.x2_col = x2_col
        self.y_col = y_col
        self.onera = onera
        self.norm = norm
        self.on_epoch_end()

    def __len__(self):
        #return len((self.indices) / self.batch_size) 
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        #print("Index:", index)
        batch = [self.indices[k] for k in index]
        #print("Batch", batch)
        X1, X2, y = self.get_data(batch)
        return [X1,X2],y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)
    
    def get_data(self, batch):
        X1 = np.ndarray(shape=(self.batch_size , 96,96,3))
        X2 = np.ndarray(shape=(self.batch_size , 96,96,3))
        y =  np.ndarray(shape=(self.batch_size ,))
        
        for i, id in enumerate(batch):
            #print("i", i, "id", id)
            x1_temp = np.load(self.path + self.x1_col[id])
            #print("loading image:", self.x1_col[id])
            x2_temp = np.load(self.path + self.x2_col[id])
            #print("loading image:", self.x2_col[id])
            if self.onera == False:
                X1[i,] = x1_temp[:,:,1:4]
                X2[i,] = x2_temp[:,:,1:4]
            else: 
                X1[i,] = x1_temp
                X2[i,] = x2_temp
            y[i] = self.y_col[id]

        if self.norm == True:
            X1 = (X1 - X1.mean()) / X1.std()
            X2 = (X2 - X2.mean()) / X2.std()
        return X1, X2, y

#keras.utils.to_categorical(y, num_classes=self.num_classes)

class CD_DataGenerator(keras.utils.Sequence):
    def __init__(self, path, df, x1_col, x2_col, y_col, batch_size=32, num_classes=None, shuffle=False, onera=False, norm=False):
        self.path = path
        self.batch_size = batch_size
        self.df = df
        self.indices = self.df.index.tolist()
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.x1_col = x1_col
        self.x2_col = x2_col
        self.y_col = y_col
        self.onera = onera
        self.norm = norm
        self.on_epoch_end()

    def __len__(self):
        #return len((self.indices) / self.batch_size) 
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]
        X1, X2, y = self.get_data(batch)
        return [X1,X2],y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)
    
    def get_data(self, batch):
        X1 = np.ndarray(shape=(self.batch_size , 96,96,3))
        X2 = np.ndarray(shape=(self.batch_size , 96,96,3))
        y =  np.ndarray(shape=(self.batch_size ,96,96,))
        
        for i, id in enumerate(batch):
            x1_temp = np.load(self.path + self.x1_col[id])
            x2_temp = np.load(self.path + self.x2_col[id])
            y_temp = np.load(self.path + self.y_col[id])
            
            
            if self.onera == False:
                X1[i,] = x1_temp[:,:,1:4]
                X2[i,] = x2_temp[:,:,1:4]
            else: 
                X1[i,] = x1_temp
                X2[i,] = x2_temp
            y[i,] = y_temp
        if self.norm == True:
            X1 = (X1 - X1.mean()) / X1.std()
            X2 = (X2 - X2.mean()) / X2.std()
        return X1, X2, keras.utils.to_categorical(y, num_classes=self.num_classes)
    
        
    
