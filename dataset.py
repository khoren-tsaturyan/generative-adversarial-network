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
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=self.root)

    def __len__(self):
        return len(self.filenames)
