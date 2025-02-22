import numpy as np
import torch
from DefinedModels import Dec_AAE, Enc_AAE, Discriminant
from torchvision import transforms
import numpy as np
from tqdm import tqdm

def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

class AEDataset(torch.utils.data.Dataset):#需要继承data.Dataset
    def __init__(self,Datapath,transform):
        # 1. Initialize file path or list of file names.
        self.Datalist=np.load(Datapath)
        self.transform=transform
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data
        Data=self.Datalist[index]
        Data = np.transpose(Data, (2, 0, 1))# 变为 (15, 25, 25)
        Data = np.expand_dims(Data, axis=0)# 变为 (1, 15, 25, 25)
        Data = torch.from_numpy(Data).float()
        return Data
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.Datalist)

def generate_(batch_size,dim):
    return torch.from_numpy(
        np.random.multivariate_normal(mean=np.zeros([dim ]), cov=np.diag(np.ones([dim])),
                                      size=batch_size)).type(torch.float)
def spectral_smoothness_loss(recon):
    diff = recon[:, :, 1:, :, :] - recon[:, :, :-1, :, :]
    smoothness_loss = torch.mean(diff ** 2)
    return smoothness_loss

def TrainAAE_patch(dataset,XPath, Patch_channel=15, windowSize=25, encoded_dim=128, batch_size=128):
    trans = transforms.Compose(transforms=[
        transforms.ToTensor(),
    ])
    Enc_patch = Enc_AAE(channel=Patch_channel, output_dim=encoded_dim, windowSize=windowSize).cuda()
    Dec_patch = Dec_AAE(channel=Patch_channel, windowSize=windowSize, input_dim=encoded_dim).cuda()
    discriminant = Discriminant(encoded_dim).cuda()

    # 训练数据集
    patch_data = AEDataset(XPath, trans)
    Patch_loader = torch.utils.data.DataLoader(dataset=patch_data, batch_size=batch_size, shuffle=True,num_workers=4, pin_memory=True)
    
    optim_enc = torch.optim.Adam(Enc_patch.parameters(), lr=1e-3, weight_decay=0.0005)
    optim_dec = torch.optim.Adam(Dec_patch.parameters(), lr=1e-3, weight_decay=0.0005)
    optim_enc_gen = torch.optim.SGD(Enc_patch.parameters(), lr=1e-4, weight_decay=0.000)
    optim_disc = torch.optim.SGD(discriminant.parameters(), lr=5e-5, weight_decay=0.000)
    
    criterion = torch.nn.MSELoss()
    epochs = 20
    alpha = 0.1  # 光谱平滑性约束的权重

    for epoch in range(epochs):
        rl = 0
        l_dis_loss = 0
        l_encl = 0
        print('Epoch No {}'.format(epoch))
        
        for i, data in enumerate(tqdm(Patch_loader)):
            # Reconstruction phase
            data = data.cuda().float()
            Enc_patch.train()
            Dec_patch.train()
            
            optim_dec.zero_grad()
            optim_enc.zero_grad()
            
            map, code = Enc_patch(data)
            recon = Dec_patch(code)
            loss = criterion(data, recon)
            # smoothness_loss = spectral_smoothness_loss(recon)
            total_loss = loss 
            
            total_loss.backward()
            optim_dec.step()
            optim_enc.step()
            
            # Regularization phase
            discriminant.train()
            Enc_patch.eval()
            
            gauss = generate_(batch_size, encoded_dim).cuda()
            _, code2 = Enc_patch(data)
            
            fake_pred = discriminant(gauss)
            true_pred = discriminant(code2)
            dis_loss = -(torch.mean(fake_pred) - torch.mean(true_pred))
            
            optim_disc.zero_grad()
            dis_loss.backward()
            optim_disc.step()
            
            Enc_patch.train()
            discriminant.eval()
            
            _, code3 = Enc_patch(data)
            true_pred2 = discriminant(code3)
            encl = -torch.mean(true_pred2)
            
            optim_enc_gen.zero_grad()
            encl.backward()
            optim_enc_gen.step()
            
            rl += total_loss.item()
            l_dis_loss += dis_loss.item()
            l_encl += encl.item()
        
        print('\nPatch Reconstruction Loss: {:.6f}, Discriminant Loss: {:.6f}, Regularization Loss: {:.6f}'.format(
            rl/len(Patch_loader), l_dis_loss/len(Patch_loader), l_encl/len(Patch_loader)))
    
    torch.save(Enc_patch.state_dict(), './models/{}_Enc_AAE.pth'.format(dataset))
    torch.save(Dec_patch.state_dict(), './models/{}_Dec_AAE.pth'.format(dataset))

    return 0

def SaveFeatures_AAE(dataset,XPath, Patch_channel=15, windowSize=25, encoded_dim=64, batch_size=128):

    # from Preprocess import feature_normalize2, L2_Norm

    trans = transforms.Compose([
        transforms.ToTensor(),
    ])

    Enc_patch = Enc_AAE(channel=Patch_channel, output_dim=encoded_dim, windowSize=windowSize).cuda()
    Enc_patch.load_state_dict(torch.load('./models/{}_Enc_AAE.pth'.format(dataset)))

    # 运行patchAE 的encoder
    patch_data = AEDataset(XPath, trans)
    Patch_loader = torch.utils.data.DataLoader(dataset=patch_data, batch_size=batch_size, shuffle=False,num_workers=4, pin_memory=True)
    
    Enc_patch.eval()
    Patch_Features = []

    print('Start saving patch features...')
    with torch.no_grad():
        for data in tqdm(Patch_loader):
            data = data.cuda().float()
            feature,_ = Enc_patch(data)
            Patch_Features.append(feature.cpu().numpy())
    
    # 将特征转换为NumPy数组
    Patch_Features = np.concatenate(Patch_Features, axis=0)
    
    # Patch_Features = feature_normalize2(Patch_Features)
    # Patch_Features = L2_Norm(Patch_Features)

    return Patch_Features

def DecodeFeatures_AAE(dataset,latent_features, Patch_channel=15, windowSize=25, encoded_dim=64):
    # Initialize and load the decoder model
    Dec_patch = Dec_AAE(channel=Patch_channel, input_dim=encoded_dim, windowSize=windowSize).cuda()
    Dec_patch.load_state_dict(torch.load('./models/{}_Dec_AAE.pth'.format(dataset)))
    
    # Ensure latent_features is a PyTorch tensor and move to GPU
    if not isinstance(latent_features, torch.Tensor):
        latent_features = torch.tensor(latent_features)
    latent_features = latent_features.float().cuda()
    
    # Decode the latent features
    Dec_patch.eval()
    with torch.no_grad():
        decoded_output = Dec_patch(latent_features)
    
    # Move decoded output back to CPU and convert to numpy array
    decoded_output = decoded_output.cpu().numpy()
    
    # As normalization was zero-mean and unit-std, the decoded output is already in the comparable range
    return decoded_output
