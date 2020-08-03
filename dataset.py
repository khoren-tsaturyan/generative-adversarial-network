import os
from torch.utils.data import Dataset
import tarfile
from PIL import Image


class ImagesDataset(Dataset):
    def __init__(self, root, transform=None):
        super(ImagesDataset, self).__init__()
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.filenames = []        
        data_dir = os.path.join(self.root,'Images')
        if not os.path.exists(data_dir):
            self.extract_file()
        
        for name in os.listdir(data_dir):
            full_path = os.path.join(data_dir,name)
            self.filenames.append(full_path)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = Image.open(filename)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

    def extract_file(self):
        file_dir = os.path.join(self.root,'Images.tar.gz')
        with tarfile.open(file_dir,'r:gz') as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.filenames)
