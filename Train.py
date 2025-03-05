import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from imports.ParametersManager import *
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import csv
from NAST_Net import  NastNet
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
MODELNAME = 'Nast on Nist 94'
MODELFILEDIR = './'
BatchSize = 1
LEARNINGRATE = 8*(1e-5)
epochNums = 180
SaveModelEveryNEpoch = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
if not os.path.exists(MODELFILEDIR):
    os.mkdir(MODELFILEDIR)
MODELFILEPATH = os.path.join(MODELFILEDIR, MODELNAME + '.pt')
ImagePath='./NIST2016_re/'
TrainDatasetIndex = './NIST2016_re/Train.csv'
TestDatasetIndex = './NIST2016_re/Test.csv'
if __name__ == "__main__":
    model = NastNet()
    model.cuda()
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    parManager = ParametersManager(device)
    if os.path.exists(MODELFILEPATH):
        parManager.loadFromFile(MODELFILEPATH)
        parManager.setModelParameters(model)
    else:
        print('===No pre-trained model found!===')
    print(parManager.TestACC)
    criterion=nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), LEARNINGRATE)
    TrainDataset = MyDataset(TrainDatasetIndex)
    TestDataset = MyDataset(TestDatasetIndex)

    print('Trainset size: {}'.format(len(TrainDataset)))
    print('Testset size: {}'.format(len(TestDataset)))
    TrainLoader = DataLoader(TrainDataset, num_workers=0, pin_memory=True, batch_size=BatchSize,
                             sampler=torch.utils.data.sampler.SubsetRandomSampler(range(len(TrainDataset))),drop_last=True)
    TestLoader = DataLoader(TestDataset, num_workers=0, pin_memory=True, batch_size=BatchSize,
                            sampler=torch.utils.data.sampler.SubsetRandomSampler(range(len(TestDataset))),drop_last=True)
    print('len(TrainLoader):{}'.format(len(TrainLoader)))
    maxauc=0.0
    TrainACC = []
    TestACC = []
    GlobalLoss = []
    for epoch in range(epochNums):
        print("Epoch {}---sumEpoch {}".format(epoch, parManager.EpochDone))
        epochAccuracy = []
        epochLoss = []
        model.train()
        for batch_id, (inputs, label) in enumerate(TrainLoader):
            if batch_id==120 or batch_id==240:
                LEARNINGRATE=LEARNINGRATE/2
            # torch.train()
            optimizer.zero_grad()
            output = model(inputs.cuda())
            loss = criterion(output, label.cuda())
            loss.backward()
            optimizer.step()

            epochAccuracy.append(pixel_level_auc(output, label.cuda(),Tag=False))
            epochLoss.append(loss.item())
            # print status
            if batch_id % (int(len(TrainLoader) / 10)) == 0:
                print("    Now processing step[{}/{}], Current Epoch accuracy：{:.2f}%，Loss：{:.8f}".format(batch_id,
                                                                                                          len(TrainLoader),
                                                                                                          np.mean(
                                                                                                              epochAccuracy) * 100,
                                                                                                          loss))
        TrainACC.append(np.mean(epochAccuracy))
        GlobalLoss.append(np.mean(epochLoss))
        localTestACC = []
        model.eval()  #
        for inputs, label in TestLoader:
            torch.no_grad()
            output = model(inputs.cuda())
            localTestACC.append(pixel_level_auc(output, label.cuda(),Tag=True))
        TestACC.append(np.mean(localTestACC))
        print("Current Epoch Done, Train accuracy: {:3f}%, Test accuracy: {:3f}%".format(TrainACC[-1] * 100,
                                                                                         TestACC[-1] * 100))
        parManager.oneEpochDone(LEARNINGRATE, TrainACC[-1], TestACC[-1], GlobalLoss[-1])
        if epoch == epochNums - 1 or epoch % SaveModelEveryNEpoch == 0 :
            parManager.loadModelParameters(model)
            parManager.saveToFile(MODELFILEPATH)
            if float(format(TestACC[-1] * 100))>maxauc:
                maxauc=float(format(TestACC[-1] * 100))

    #输出结果
