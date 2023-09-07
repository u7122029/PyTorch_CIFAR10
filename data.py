import os
import zipfile

import lightning as lt
import requests
import torchvision.io
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from pathlib import Path
from PIL import Image


class ToRGB:
    def __init__(self):
        pass

    def __call__(self, sample):
        _input = sample
        return _input.repeat(3,1,1)


class MNISTData(lt.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.prepare_data()
        self.setup("whocares")

    def prepare_data(self):
        # This will be run on a single process
        # We download the datasets here if they don't exist yet.
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage):
        train_transform = T.Compose([T.Resize((32,32)),
                                     T.RandomHorizontalFlip(),
                                     T.RandomVerticalFlip(),
                                     T.ToTensor(),
                                     ToRGB()])
        test_transform = T.Compose([T.Resize((32,32)),
                                    T.ToTensor(),
                                    ToRGB()])

        self.train_ds = MNIST(
            self.data_dir,
            train=True,
            transform=train_transform
        )
        self.test_ds = MNIST(
            self.data_dir,
            train=False,
            transform=test_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=len(self.test_ds),
            num_workers=self.num_workers,
            shuffle=False
        )
    

class CIFAR10Data(lt.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.h_params = args
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)


    @staticmethod
    def download_weights():
        url = (
            "https://rutgers.box.com/shared/static/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip"
        )

        # Streaming, so we can iterate over the response.
        r = requests.get(url, stream=True)

        # Total size in Mebibyte
        total_size = int(r.headers.get("content-length", 0))
        block_size = 2 ** 20  # Mebibyte
        t = tqdm(total=total_size, unit="MiB", unit_scale=True)

        with open("state_dicts.zip", "wb") as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()

        if total_size != 0 and t.n != total_size:
            raise Exception("Error, something went wrong")

        print("Download successful. Unzipping file...")
        path_to_zip_file = os.path.join(os.getcwd(), "state_dicts.zip")
        directory_to_extract_to = os.path.join(os.getcwd(), "cifar10_models")
        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
            print("Unzip file successful!")

    def train_dataloader(self):
        transform = T.Compose(
            [
                #T.RandomCrop(32, padding=4),
                #T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=self.h_params.data_dir, train=True, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.h_params.batch_size,
            num_workers=self.h_params.num_workers,
            shuffle=True,
            #drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=self.h_params.data_dir, train=False, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.h_params.batch_size,
            num_workers=self.h_params.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class SyntheticDigitsData(Dataset):
    def __init__(self, root: str, train=True, transform=None):
        super().__init__()
        self.root = Path(root) / "synthetic_digits"
        self.train = train
        self.transform = transform

        self.imgs = []
        self.labels = []
        if self.train:
            self.root /= "imgs_train"
        else:
            self.root /= "imgs_valid"

        for digit in range(10):
            current_dir = self.root / str(digit)
            for img_idx in range(1000 if self.train else 200):
                filename = f"{digit}_{str(img_idx + 1000 if not self.train else img_idx).zfill(5)}.jpg"
                complete_path = current_dir / filename
                self.labels.append(digit)
                img = torchvision.io.read_image(str(complete_path))
                self.imgs.append(img)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        img = to_pil_image(img)
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.labels)




