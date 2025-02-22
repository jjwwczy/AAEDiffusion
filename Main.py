


from dataclasses import dataclass
from typing import Union, Optional
import torch
import os
import argparse
from Preprocess import applyPCA,PerClassSplit,splitTrainTestSet,Preprocess,reshape_and_normalize_features,normalize_new_features,denormalize_feature_maps
from TrainAE import TrainAAE_patch,SaveFeatures_AAE,DecodeFeatures_AAE

import numpy as np
from TrainSVM import  TrainNN, TestNN,TrainSVM,TestSVM
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import time
import csv
from utils import reports
from diffusers import DDIMScheduler,DDPMScheduler,UNet3DConditionModel,UNet2DModel
from diffusers.utils.torch_utils import randn_tensor
from typing import List, Optional, Tuple, Union
from diffusers.optimization import get_cosine_schedule_with_warmup
from model import train_loop,  LabeledTimeSeriesDataset2D,LabeledDDPMPipeline,LabeledDDIMPipeline,unsupervised_train_loop
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch

# @dataclass
# class SequenceDataOutput:
#   sequences: List[torch.Tensor]
#   history: List[torch.Tensor]

if __name__ == '__main__':

    dataset_names = ['IP','SA','PU']
    #PU (610, 340, 103) IP (145, 145, 200)  SA (512, 217, 204)
    parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                                " various hyperspectral datasets")
    parser.add_argument('--dataset', type=str, default='PU', choices=dataset_names,
                        help="Dataset to use.")
    parser.add_argument('--PureFake',type=bool, default=False)###只使用伪造样本训练
    parser.add_argument('--train',type=int, default=1,choices=(-1,0,1))###-1直接使用MLP分类，0使用预训练好的模型，1重新训练模型
    parser.add_argument('--perclass', type=float, default=10) #真实数据使用量，会除以100
    parser.add_argument('--SamplesMode', type=str, default='balance',choices=('balance','even')) #balance模式下，将每类样本补足到相同数量，even模式下，每类补相同数量的伪造样本
    parser.add_argument('--AddSamplesNum', type=int, default=200) #'even'模式下生效。每类补伪造样本的数量。
    parser.add_argument('--SamplesNumScale', type=float, default=1) #'BALANCE'模式下生效。数量最多的类别乘的比例，保证每类真实样本和伪造样本的总数量一样
    parser.add_argument('--device', type=str, default="cuda:0", choices=("cuda:0","cpu"))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--Windowsize', type=int, default=25)####
    parser.add_argument('--Patch_channel', type=int, default=15)
    parser.add_argument('--RandomSeed', type=bool, default=False)#True 代表固定随机数， False代表不固定
    parser.add_argument('--train_AAE', type=bool, default=True)#True 代表重新训练AAE， False代表不训练


    #定义所有time变量为0
    TrainAAE_time=0
    TestNN_time=0
    TrainNN_time=0
    train_loop_time=0
    SaveFeatures_AAE_time=0
    generate_time=0


    args = parser.parse_args()
    # Setup Constants
    DIFFUSION_TIMESTEPS = 1000  # Number of Diffusion Steps during training. Note, this is much smaller than ordinary (usually ~1000)
    beta_start = 0.0001  # Default from paper
    beta_end = 0.02  # NOTE: This is different to accomodate the much shorter DIFFUSION_TIMESTEPS (Usually ~1000). For 1000 diffusion timesteps, use 0.02.
    clip_sample = True  # Used for better sample generation


    ##########加噪声的时间表，每一个时间步加的噪声由这个控制###########
    noise_scheduler = DDIMScheduler(num_train_timesteps=DIFFUSION_TIMESTEPS, beta_end=beta_end, clip_sample=clip_sample)


    if args.RandomSeed:
        randomState=345
    else:
        randomState=int(np.random.randint(1, high=1000))
    args.perclass=args.perclass/100
    print(args)
    output_units = 9 if (args.dataset == 'PU' or args.dataset == 'PC') else 16
    classes=output_units
    channels=args.Patch_channel
    Windowsize=args.Windowsize
    timesteps=Windowsize*Windowsize
    device=args.device

    ####################生成config参数，从教学代码里改的#############
    @dataclass
    class TrainingConfig:
        num_epochs = 200
        gradient_accumulation_steps = 1
        learning_rate = 3e-4
        lr_warmup_steps = 100
        save_image_epochs = 10
        save_model_epochs = 30
        mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
        output_dir = 'ButterflyTutorial'  # the model name locally and on the HF Hub
        push_to_hub = False  # whether to upload the saved model to the HF Hub
        hub_private_repo = False
        overwrite_output_dir = True  # overwrite the old model when re-running the notebook
        seed = 0
        inference_timesteps=50 #############在pipeline中采样的时间步数，对于DDIM而言，50相当于DDPM的1000步
        spectral_loss_weight=1.0
    config = TrainingConfig()

    Datadir='./DataArray/'
    XPath = Datadir + 'X.npy'
    yPath = Datadir + 'y.npy'

    Xshape=Preprocess(XPath, yPath,args.dataset, Windowsize, Patch_channel=channels)
    X=np.load(XPath)
    y=np.load(yPath).astype(int)
    class_labels = np.unique(y)
    class_to_idx = {label: np.where(y == label)[0] for label in class_labels}
    Prototypes=[]#########统计每一类样本的均值，用于比对采样出的伪造样本真实性。###############
    for idx, key in enumerate(class_to_idx.keys()): 
        Prototypes.append(X[class_to_idx[key]].mean(axis=0))
    ##保存Prototypes，用于后续的可视化分析
    np.save(Datadir + 'Patch_Prototypes.npy', np.array(Prototypes))


    AAEPath=Datadir+'AAE_Features.npy'

    diffMapPath=Datadir+'diffMap.npy'
    minmaxPath=Datadir+'minmax.npy'

    if args.train_AAE:
            # Unsupervised Training
        start_time = time.time()  # 记录开始时间
        TrainAAE_patch(args.dataset,XPath,Patch_channel=channels,windowSize=args.Windowsize,encoded_dim=2048,batch_size=128)
        end_time = time.time()  # 记录结束时间
        TrainAAE_time = end_time - start_time  # 计算执行时间

        start_time = time.time()  # 记录开始时间
        AAEFeatures=SaveFeatures_AAE(args.dataset,XPath,Patch_channel=channels,windowSize=args.Windowsize,encoded_dim=2048,batch_size=128)
        end_time = time.time()  # 记录结束时间
        SaveFeatures_AAE_time = end_time - start_time  # 计算执行时间
        np.save(AAEPath,AAEFeatures)###保留原生的latent特征，不进行归一化和标准化
        start_time = time.time()  # 记录开始时间
        normalized_feature_maps, min_val, max_val=reshape_and_normalize_features(np.array(AAEFeatures))
        end_time = time.time()  # 记录结束时间
        np.save(diffMapPath,normalized_feature_maps)
        #保存最大最小值，用于后续的反归一化
        np.save(minmaxPath,np.array([min_val,max_val]))

    ####把特征图作为X输入扩散模型，y保持不变，即为原始标签
    Xmap=np.load(diffMapPath)
    y=np.load(yPath).astype(int)

    stratify = np.arange(0, output_units, 1)
    #########################判断是每类选几个样本还是按百分比划分训练样本#################

    if args.perclass > 1:
        Xtrain, Xtest, ytrain, ytest,train_index, test_index = PerClassSplit(Xmap, y, int(args.perclass), stratify,randomState=randomState)
    else:
        Xtrain, Xtest, ytrain, ytest,train_index, test_index = splitTrainTestSet(Xmap, y, 1 - args.perclass,randomState=randomState)

    class_labels = np.unique(y)
    print("Class labels: ", class_labels)
    class_to_idx = {label: np.where(y == label)[0] for label in class_labels}
    Prototypes=[]#########统计每一类样本的均值，用于比对采样出的伪造样本真实性。###############
    for idx, key in enumerate(class_to_idx.keys()): 
        Prototypes.append(Xmap[class_to_idx[key]].mean(axis=0))
    ##保存Prototypes，用于后续的可视化分析
    np.save(Datadir + 'Map_Prototypes.npy', np.array(Prototypes))

    #找到ytrain中样本数量最多的类别，获取其数量MaxTrueNum
    MaxTrueNum = np.max([np.sum(ytrain == label) for label in class_labels])

    #根据ytrain中不同类别所占的比例，构造class_weights权重，引导模型更多地关注样本数量少的类别
    class_weights = torch.tensor([MaxTrueNum/np.sum(ytrain == label) for label in class_labels], dtype=torch.float32).to(device)

    #构造字典FakeNumDict，key为类别标签，value为待生成的伪造样本的数量
    FakeNumDict = {label: int(args.SamplesNumScale*MaxTrueNum) - np.sum(ytrain == label)+1 for label in class_labels}

    ###############构造数据集#####################
    dataset = LabeledTimeSeriesDataset2D(Xtrain,ytrain)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,num_workers=4, pin_memory=True)

    #############选择带标签输入的UNet2DModel#######################
    model = UNet2DModel (
        sample_size=32,  # 或者根据您的需要设置
        in_channels=2,
        out_channels=2,
        block_out_channels= (112, 224, 336, 448),  # 或者根据您的需要设置
        norm_num_groups=16,
        time_embedding_type='positional',  # 根据需要选择合适的时间嵌入类型
        # 其他参数保持默认或根据需要修改
        class_embed_type='timestep',  # 使用什么方式的类别嵌入
        num_class_embeds=output_units,  # 设置类别数量
    )


    print("Dataset size :", len(dataset), "\nSingle Sample shape: ", dataset[0][0].shape, "\nLabel: ", dataset[0][1])
    # 1. 数据预处理，包括读数据，生成数组文件等等
    #

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,weight_decay=0.0005)

    ################训练模型的学习率时间表，注意num_training_steps需要和config.num_epochs的数量对应#######################
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(dataloader)* config.num_epochs,
    )
    min_val=np.load(minmaxPath)[0]
    max_val=np.load(minmaxPath)[1]
    if args.train==1:
        ############################################################################
        model.train()
        # Supervised Training
        start_time = time.time()  # 记录开始时间
        model,noise_scheduler=train_loop(class_weights,config, model, noise_scheduler, optimizer, dataloader, lr_scheduler)
        end_time = time.time()  # 记录结束时间
        train_loop_time = end_time - start_time  # 计算执行时间

        ################训练结束，开始采样构造仿真样本####################################################
        fake_Xtrain=[]
        fake_ytrain=[]
        # Can try changing `num_inference_steps` here to check if we can diffuse a sample with fewer than DIFFUSION_TIMESTEPS steps

        for class_ids in FakeNumDict.keys():
            # Uncomment below to have determinstic sampling
            g = None
            # g = torch.Generator(device=device)
            # g.manual_seed(0)
            pipeline = LabeledDDIMPipeline(model, noise_scheduler)#######eta=0时是DDIM，eta=1时是DDPM
            pipeline = pipeline.to(device)

            # 假设 sample_batchsize 是您想要生成的样本数量
            # 假设 class_ids 包含了相应的类别标签
            # 确保 class_ids 的长度与 sample_batchsize 相匹配
            if args.SamplesMode=='balance':
                curBatchSize=FakeNumDict[class_ids]
            elif args.SamplesMode=='even':
                curBatchSize=args.AddSamplesNum
            pipe_output = pipeline(
                labels=[class_ids],  # 传递类别标签
                batch_size=curBatchSize,
                num_inference_steps=config.inference_timesteps,
                generator=g,
                return_dict=False  # 使用 return_dict=True 来获取结构化的输出
            )

            # 根据您的需求获取输出
            # 如果您的输出是图像序列，可以使用 pipe_output.images
            # 但由于您的数据是高光谱数据，可能需要直接使用 Tensor
            samples =  pipe_output[0]

            # latentFeature=denormalize_feature_maps(samples, min_val, max_val)
            # samples=DecodeFeatures_AAE(latentFeature,Patch_channel=channels,windowSize=args.Windowsize,encoded_dim=2048)
            #把samples从（curBatchSize,1,15,25,25）reshape成（curBatchSize,25,25，15）
            # samples = samples.reshape(curBatchSize, samples.shape[3], samples.shape[4], samples.shape[2])
            samplelist=samples.cpu().numpy().tolist()
            fake_Xtrain+=samplelist
            fake_ytrain+=[int(class_ids)]*curBatchSize

        print('伪造样本的形状： '+str(np.array(fake_Xtrain).shape))
        if args.PureFake:
            Xtrain=np.array(fake_Xtrain)
            ytrain=np.array(fake_ytrain)
        else:
            # X=np.load(XPath)
            # Xtrain=X[train_index]
            # Xtest=X[test_index]
            Xtrain=np.concatenate((Xtrain,np.array(fake_Xtrain)),axis=0)
            ytrain=np.concatenate((ytrain,np.array(fake_ytrain)),axis=0)
    ####把fake_Xtrain和fake_ytrain保存下来，用于后续的可视化分析
        np.save(Datadir + 'fake_Xtrain.npy', np.array(fake_Xtrain))
        np.save(Datadir + 'fake_ytrain.npy', np.array(fake_ytrain))
    elif args.train==0:############使用预训练好的模型##########
        # model.load_state_dict(torch.load(checkpoint_dir+checkpoint_name))
        pipeline = LabeledDDIMPipeline.from_pretrained(config.output_dir)

        ################训练结束，开始采样构造仿真样本####################################################
        fake_Xtrain=[]
        fake_ytrain=[]
        #记录仿真样本生成时间
        start_time = time.time()  # 记录开始时间
        # Can try changing `num_inference_steps` here to check if we can diffuse a sample with fewer than DIFFUSION_TIMESTEPS steps
        for class_ids in class_labels:
            # Uncomment below to have determinstic sampling
            g = None
            # g = torch.Generator(device=device)
            # g.manual_seed(0)
            pipeline = pipeline.to(device)
            # 假设 sample_batchsize 是您想要生成的样本数量
            # 假设 class_ids 包含了相应的类别标签
            # 确保 class_ids 的长度与 sample_batchsize 相匹配
            if args.SamplesMode=='balance':
                curBatchSize=FakeNumDict[class_ids]
            elif args.SamplesMode=='even':
                curBatchSize=args.AddSamplesNum
            pipe_output = pipeline(
                labels=[class_ids],  # 传递类别标签
                batch_size=curBatchSize,
                num_inference_steps=config.inference_timesteps,
                generator=g,
                return_dict=False  # 使用 return_dict=True 来获取结构化的输出
            )

            # 根据您的需求获取输出
            # 如果您的输出是图像序列，可以使用 pipe_output.images
            # 但由于您的数据是高光谱数据，可能需要直接使用 Tensor
            samples =  pipe_output[0]

            # latentFeature=denormalize_feature_maps(samples, min_val, max_val)
            # samples=DecodeFeatures_AAE(latentFeature,Patch_channel=channels,windowSize=args.Windowsize,encoded_dim=2048)
            # samples = samples.reshape(curBatchSize, samples.shape[3], samples.shape[4], samples.shape[2])

            samplelist=samples.cpu().numpy().tolist()


            fake_Xtrain+=samplelist
            fake_ytrain+=[int(class_ids)]*curBatchSize
        stop_time = time.time()  # 记录开始时间
        generate_time = stop_time - start_time  # 计算执行时间
        print('伪造样本的形状： '+str(np.array(fake_Xtrain).shape))

        if args.PureFake:
            Xtrain=np.array(fake_Xtrain)
            ytrain=np.array(fake_ytrain)
        else:
            # X=np.load(XPath)
            # Xtrain=X[train_index]
            # Xtest=X[test_index]
            Xtrain=np.concatenate((Xtrain,np.array(fake_Xtrain)),axis=0)
            ytrain=np.concatenate((ytrain,np.array(fake_ytrain)),axis=0)
        ####把fake_Xtrain和fake_ytrain保存下来，用于后续的可视化分析
        np.save(Datadir + 'fake_Xtrain.npy', np.array(fake_Xtrain))
        np.save(Datadir + 'fake_ytrain.npy', np.array(fake_ytrain))    
    elif args.train==-1:#######与模型无关，即直接使用MLP分类
        X=np.load(diffMapPath)
        y=np.load(yPath).astype(int)

        stratify = np.arange(0, output_units, 1)
        #########################判断是每类选几个样本还是按百分比划分训练样本#################
        if args.perclass > 1:
            Xtrain, Xtest, ytrain, ytest,train_index, test_index = PerClassSplit(X, y, int(args.perclass), stratify,randomState=randomState)
        else:
            Xtrain, Xtest, ytrain, ytest,train_index, test_index = splitTrainTestSet(X, y, 1 - args.perclass,randomState=randomState)

    X=np.load(diffMapPath)
    Xtest=X[test_index]
    print('用于训练的样本的形状： '+str(Xtrain.shape))


    Xtrain=Xtrain.reshape(Xtrain.shape[0],-1)
    Xtest=Xtest.reshape(Xtest.shape[0],-1)
    #对Xtrain和Xtest进行PCA降维
    pca = PCA(n_components=128, whiten=True)

    Xtrain=pca.fit_transform(Xtrain)
    Xtest=pca.transform(Xtest)
    print('PCA降维后的训练样本的形状： '+str(Xtrain.shape))
    print('PCA降维后的测试样本的形状： '+str(Xtest.shape))



    ytrain=ytrain###########为了让NN能正常计算loss，label必须从0开始
    ytest=ytest###########为了让NN能正常计算loss，label必须从0开始
    np.save(Datadir + 'Xtrain.npy', Xtrain)
    np.save(Datadir + 'ytrain.npy', ytrain)
    np.save(Datadir + 'Xtest.npy', Xtest)
    np.save(Datadir + 'ytest.npy', ytest)

    # print(True in np.isnan(Xtrain))###判断是否出现了空值
    # print(True in np.isnan(Xtest))

    # SVM_model=TrainSVM(Xtrain,ytrain)#####SVM分类，但是好像数据维度一高会导致cpu崩溃
    # joblib.dump(SVM_model, './models/SVM.model')
    # SVM_model=joblib.load('./models/SVM.model')
    # Predictions=TestSVM(Xtest,SVM_model)

    start_time = time.time()  # 记录开始时间
    ModelPath=TrainNN(n_features=Xtrain.shape[-1],n_classes=output_units,Datadir=Datadir)#####简单MLP分类###
    end_time = time.time()  # 记录结束时间
    TrainNN_time = end_time - start_time  # 计算执行时间


    start_time = time.time()  # 记录开始时间
    Predictions=TestNN(n_features=Xtrain.shape[-1],n_classes=output_units, ModelPath=ModelPath,Datadir=Datadir)
    end_time = time.time()  # 记录结束时间
    TestNN_time = end_time - start_time  # 计算执行时间

    classification = classification_report(ytest.astype(int), Predictions)
    print(classification)
    classification, confusion, oa, each_acc, aa, kappa = reports(Predictions, ytest.astype(int), args.dataset)

    traintime=TrainAAE_time+TrainNN_time+train_loop_time  

    testtime=TestNN_time+SaveFeatures_AAE_time+generate_time

    AAE_traintime=TrainAAE_time+TrainNN_time
    AAE_testtime=TestNN_time+SaveFeatures_AAE_time

    each_acc_str = ','.join(str(x) for x in each_acc)
    add_info=[args.dataset,args.perclass,args.train,args.Windowsize,args.SamplesMode,args.AddSamplesNum,MaxTrueNum,int(args.SamplesNumScale*MaxTrueNum)+1,oa,aa,kappa,traintime,testtime,AAE_traintime,AAE_testtime]+each_acc_str.split('[')[0].split(']')[0].split(',')
    csvFile = open("AAEUnet大修版实验结果.csv", "a",newline='')
    writer = csv.writer(csvFile)
    writer.writerow(add_info)
    csvFile.close()
