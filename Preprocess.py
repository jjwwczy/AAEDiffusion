import numpy as np
import scipy.io as sio
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from operator import truediv
import torch

def splitTrainTestSet(X, y, testRatio, randomState=345):
    # Generate the indices
    indices = np.arange(X.shape[0])
    
    # Split the data and the indices
    X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(
        X, y, indices, test_size=testRatio, random_state=randomState, stratify=y)
    
    return X_train, X_test, y_train, y_test, train_index, test_index

def PerClassSplit(X, y, perclass, stratify, randomState=345):
    np.random.seed(randomState)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    train_indices = []
    test_indices = []

    for label in stratify:
        indexList = [i for i in range(len(y)) if y[i] == label]
        train_index = np.random.choice(indexList, min(perclass,len(indexList))//2, replace=False)
        
        train_indices.extend(train_index)
        for i in train_index:
            X_train.append(X[i])
            y_train.append(label)
        
        test_index = [i for i in indexList if i not in train_index]
        test_indices.extend(test_index)
        for i in test_index:
            X_test.append(X[i])
            y_test.append(label)

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test), np.array(train_indices), np.array(test_indices)

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

# def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
#     margin = int((windowSize - 1) / 2)
#     zeroPaddedX = padWithZeros(X, margin=margin)
#     # split patches
#     patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]), dtype=np.float32)
#     patchesLabels = np.zeros((X.shape[0] * X.shape[1]), dtype=np.float32)
#     patchIndex = 0
#     for r in range(margin, zeroPaddedX.shape[0] - margin):
#         for c in range(margin, zeroPaddedX.shape[1] - margin):
#             patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
#             patchesData[patchIndex, :, :, :] = patch
#             patchesLabels[patchIndex] = y[r - margin, c - margin]
#             patchIndex = patchIndex + 1
#     if removeZeroLabels:
#         patchesData = patchesData[patchesLabels > 0, :, :, :]
#         patchesLabels = patchesLabels[patchesLabels > 0]
#         patchesLabels -= 1
#     return patchesData, patchesLabels
import numpy as np

def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    
    # 计算有多少非零标签，如果需要移除零标签的话
    if removeZeroLabels:
        num_patches = np.count_nonzero(y)
    else:
        num_patches = X.shape[0] * X.shape[1]

    # 初始化patches
    patchesData = np.zeros((num_patches, windowSize, windowSize, X.shape[2]), dtype=np.float32)
    patchesLabels = np.zeros(num_patches, dtype=np.float32)

    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            if removeZeroLabels and y[r - margin, c - margin] <= 0:
                continue
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin] - 1 if removeZeroLabels else y[r - margin, c - margin]
            patchIndex += 1

    return patchesData, patchesLabels

def loadData(name):
    data_path = os.path.join(os.getcwd(), 'datasets')
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
    elif name == 'KSC':
        data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
    return data, labels
def feature_normalize(data):
    mu = torch.mean(data,dim=0)
    std = torch.std(data,dim=0)
    return torch.div((data - mu),std)
def L2_Norm(data):
    norm=np.linalg.norm(data, ord=2)
    return truediv(data,norm)
def feature_normalize2(data):
    mu = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return truediv((data - mu),std)
def MinMaxNormalize(data):
    min_val = data.min(-1, keepdims=True)
    max_val = data.max(-1, keepdims=True)

    # 防止分母为零
    eps = 1e-10
    normalized_data = (data - min_val) / (max_val - min_val + eps)

    normalized_data = 2 * normalized_data - 1

    # 检查是否有 nan 或 inf 值
    if np.any(np.isnan(normalized_data)) or np.any(np.isinf(normalized_data)):
        print("Warning: NaN or Inf found in normalized data")

    return normalized_data
def normalize_hyperspectral_data(data):
    """
    Normalize hyperspectral data to the range [-1, 1].

    Parameters:
    data (numpy array): The hyperspectral data with shape (batch_size, window_size-1, window_size-1, channels).

    Returns:
    numpy array: Normalized data.
    """
    # Find the min and max values along the channel axis
    min_val = data.min(axis=(1, 2, 3), keepdims=True)
    max_val = data.max(axis=(1, 2, 3), keepdims=True)

    # Avoid division by zero by setting denominators that are 0 to 1 (effectively no scaling for those elements)
    scale = np.where(max_val - min_val != 0, max_val - min_val, 1)

    # Normalize the data to [-1, 1]
    normalized_data = 2 * (data - min_val) / scale - 1

    return normalized_data
def Preprocess(XPath,yPath,dataset, Windowsize=25, Patch_channel=15):

    X, y = loadData(dataset)
    X, pca = applyPCA(X, numComponents=Patch_channel)

    X, y = createImageCubes(X, y, windowSize=Windowsize)
    # X=torch.FloatTensor(X).cuda()
    X=MinMaxNormalize(X)
    # X = X.cpu().tolist()
    np.save(XPath, X)
    np.save(yPath,y)
    return X.shape
# Datadir='./DataArray/'
# XPath = Datadir + 'X.npy'
# yPath = Datadir + 'y.npy'
# Preprocess(XPath, yPath, 'IP', 5, Patch_channel=15)
def Preprocess2D(XPath,yPath,dataset, Windowsize=25, Patch_channel=15):

    X, y = loadData(dataset)

    X, pca = applyPCA(X, numComponents=Patch_channel)

    X, y = createImageCubes(X, y, windowSize=Windowsize)
    X=X[:,1:,1:,:]
    # X=torch.FloatTensor(X).cuda()
    # X=X.reshape(X.shape[0],X.shape[-1],-1)
    X=normalize_hyperspectral_data(X)
    # X = X.cpu().tolist()
    # print(X.shape)
    # print(X.min(), X.max())  # 应该打印出范围在 [-1, 1] 内的值
    X=X.reshape(X.shape[0],X.shape[3],X.shape[1],X.shape[2])

    np.save(XPath, X)
    np.save(yPath,y)
    # class_labels = np.unique(y)
    # print("Loaded X :", X.shape, "Max", X.max(), "Min", X.min())
    # print("Labels :", class_labels)
    return 0

def Preprocess3D(XPath,yPath,dataset, Windowsize=25, Patch_channel=15):

    X, y = loadData(dataset)

    X, pca = applyPCA(X, numComponents=Patch_channel)

    X, y = createImageCubes(X, y, windowSize=Windowsize)
    X=X[:,1:,1:,:]
    # X=torch.FloatTensor(X).cuda()
    # X=X.reshape(X.shape[0],X.shape[-1],-1)
    X=normalize_hyperspectral_data(X)
    # X = X.cpu().tolist()
    # print(X.shape)
    # print(X.min(), X.max())  # 应该打印出范围在 [-1, 1] 内的值
    X=X.reshape(X.shape[0],X.shape[3],1,X.shape[1],X.shape[2])

    np.save(XPath, X)
    np.save(yPath,y)
    # class_labels = np.unique(y)
    # print("Loaded X :", X.shape, "Max", X.max(), "Min", X.min())
    # print("Labels :", class_labels)
    return 0
# Datadir='./DataArray/'
# XPath = Datadir + 'X.npy'
# yPath = Datadir + 'y.npy'
# Preprocess2D(XPath, yPath, 'IP', 65, Patch_channel=10)


def reshape_and_normalize_features(features):
    assert features.shape[1] == 2048, "输入特征的维度必须是4096"
    feature_maps = features.reshape(features.shape[0], 2, 32, 32)
    min_val = np.min(feature_maps)
    max_val = np.max(feature_maps)
    normalized_feature_maps = 2 * (feature_maps - min_val) / (max_val - min_val) - 1
    # normalized_feature_maps=feature_maps
    return normalized_feature_maps, min_val, max_val


def normalize_new_features(new_features, min_val, max_val):
    assert new_features.shape[1] == 2048, "输入特征的维度必须是4096"
    feature_maps = new_features.reshape(new_features.shape[0], 2, 32, 32)
    normalized_feature_maps = 2 * (feature_maps - min_val) / (max_val - min_val) - 1
    return normalized_feature_maps

def denormalize_feature_maps(normalized_feature_maps, min_val, max_val):
    feature_maps = (normalized_feature_maps + 1) / 2 * (max_val - min_val) + min_val
    # 重新reshape回 (batch_size, 4096)
    reshaped_feature_maps = feature_maps.reshape(normalized_feature_maps.shape[0], 2048)
    return reshaped_feature_maps