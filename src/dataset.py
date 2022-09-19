
import cv2
from torch.utils.data import Dataset
class LFWDataset(Dataset):
    def __init__(self, data_list, transform):
        self.data = data_list
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = cv2.imread(self.data[index])
        image = cv2.resize(image, (64, 64))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        return image
