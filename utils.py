from torch.utils.data import Dataset, DataLoader
from glob import glob
from torchvision import transforms
import cv2
from .model import QAModel
import torch
import pickle as pkl
import os


class EvalDataset(Dataset):
    def __init__(self, glob_pattern):
        self.data = sorted(glob(glob_pattern))
        self.tsfm = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(512),
                                        transforms.RandomCrop(299),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        im = self.tsfm(cv2.imread(self.data[idx]))
        print(im.shape)
        return self.data[idx], im


def evaluate(batch_size, glob_pattern, save_file):
    m = QAModel().cuda()
    if not os.path.exists('model_wts.pt'):
        os.system("cat model_wts.part* > model_wts.pt")
    m.load_state_dict(torch.load('model_wts.pt'))
    dataset = EvalDataset(glob_pattern)
    data_loader = DataLoader(dataset, batch_size=batch_size)  # , collate_fn=my_collate)
    fundus_quality = dict()
    with torch.no_grad():
        for fname, im in data_loader:
            fundus_quality.update(zip(fname, m(im.cuda()).cpu().numpy()))
    pkl.dump(fundus_quality, open(save_file, 'wb'))
