from PIL import Image
import torch.utils.data as data
import os

class DataLoader(data.Dataset):
    def __init__(self, path_A, path_B, transform=None):
        super(DataLoader, self).__init__()

        self.path_A = path_A
        self.path_B = path_B
        self.transform = transform
        self.image_A = [x for x in sorted(os.listdir(self.path_A))]
        self.image_B = [x for x in sorted(os.listdir(self.path_B))]

    def __getitem__(self, index):
        img_fn_A = os.path.join(self.path_A, self.image_A[index])
        img_A = Image.open(img_fn_A)

        img_fn_B = os.path.join(self.path_B, self.image_B[index])
        img_B = Image.open(img_fn_B)

        if self.transform is not None:
            input = self.transform(img_A)
            target = self.transform(img_B)

        return input, target

    def __len__(self):
        return len(self.image_B)
