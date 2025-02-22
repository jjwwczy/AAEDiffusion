import numpy as np
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
from Preprocess import denormalize_feature_maps,normalize_new_features
from sklearn.decomposition import PCA
import argparse

dataset_names = ['IP','SA','PU']

#PU (610, 340, 103) IP (145, 145, 200)  SA (512, 217, 204)
parser = argparse.ArgumentParser(description="Run on"
                                             " various hyperspectral datasets")
parser.add_argument('--dataset', type=str, default='PU', choices=dataset_names,
                    help="Dataset to use.")
Datadir = './DataArray/'
minmaxPath=Datadir+'minmax.npy'
Name = ['fake_Xtrain','Xtrain']
labelname = ['fake_ytrain','ytrain']

args = parser.parse_args()
dataset=args.dataset
for i in range(len(Name)):
    try:
        Feature = np.load(Datadir + Name[i] + '.npy')
        # min_val=np.load(minmaxPath)[0]
        # max_val=np.load(minmaxPath)[1]
        ##########反归一化特征图######################
        # Feature=denormalize_feature_maps(Feature, min_val, max_val)


        Feature = Feature.reshape(Feature.shape[0], -1)
        pca = PCA(n_components=16, whiten=True)
        Feature=pca.fit_transform(Feature)
        print(Feature.shape)
        y = np.load(Datadir + labelname[i] + '.npy').astype(np.int32)
        print(set(y))

        # Run t-SNE
        tsne = TSNE(n_jobs=8)
        X_tsne = tsne.fit_transform(Feature, y)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab20')

        legend1 = ax.legend(*scatter.legend_elements(prop='colors',num=len(set(y))),
                            loc="best", title="Class index")
        ax.add_artist(legend1)
        plt.title(dataset+'_' +Name[i])
        plt.axis('off')  # Consider using plt.axis('tight') for tighter axis bounds
        # plt.tight_layout()
        plt.savefig(dataset+'_' + Name[i] + '.png',dpi=300)
        plt.close()
    except Exception as e:
        print(f"An error occurred with dataset {Name[i]}: {str(e)}")
    print()

####储存不同的原型####
import numpy as np

# Read the Prototypes.npy file
prototypes = np.load('DataArray/Map_Prototypes.npy')
pooled_prototypes = np.mean(prototypes, axis=(-2, -3))
pooled_prototypes.shape
np.save(dataset+'_pooled_prototypes.npy',pooled_prototypes)

####储存好真实原型后储存扩散生成的原型######

fakeXtrain = np.load('DataArray/fake_Xtrain.npy')
fakeytrain = np.load('DataArray/fake_ytrain.npy')
# Find the unique labels in fakeytrain
unique_labels = np.unique(fakeytrain)

# Initialize an empty list to store the prototypes
prototypes = []

# Iterate over the unique labels
for label in unique_labels:
    # Find the indices of the data points with the current label
    indices = np.where(fakeytrain == label)[0]
    
    # Get the corresponding data points from fakeXtrain
    data_points = fakeXtrain[indices]
    
    # Calculate the prototype for the current label
    prototype = np.mean(data_points, axis=0)
    
    # Append the prototype to the list
    prototypes.append(prototype)

# Convert the list of prototypes to a numpy array
prototypes = np.array(prototypes)
fake_prototypes = np.mean(prototypes, axis=(-2, -3))
fake_prototypes.shape

np.save(dataset+'_fake_prototypes.npy',fake_prototypes)
