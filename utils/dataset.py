import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
from matplotlib.pyplot import imshow

class MyDataset(Dataset):
    def __init__(self, Path) -> None:
        with open(Path, 'r') as f:
            reader = csv.reader(f)
            self.index = []
            for i in reader:
                self.index.append(i)
        self.trans = transforms.ToTensor()

    def __getitem__(self, i):
        image = Image.open("{}{}".format(ImagePath, self.index[i][0]))
        image = self.trans(image)

        if self.index[i][1] != 'N':
            '''
            this part is to generate a mask for manipulated images
            '''
            mask = Image.open("{}{}".format(ImagePath, self.index[i][1]))

            mask = mask.convert("1")  # convert to 0-1 image with PIL api
            mask = self.trans(mask)
            mask = 1 - mask


        else:
            ''' 
            CA中背景是纯黑的
            0是黑的
            torch.ones(...) generates a totally white image which represent to a mask of NO manipulation 
            '''
            mask = torch.zeros((1, image.shape[1], image.shape[2]))
        return image, mask

    def __len__(self):
        return len(self.index)
