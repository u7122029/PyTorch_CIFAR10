import torch
from torchvision.datasets import MNIST, SVHN
import torchvision.transforms as T
from torch.utils.data import DataLoader, ConcatDataset
from data import ToRGB, SyntheticDigitsData
import torch.nn as nn
import torch.optim as optim
from module import all_classifiers
import time
import shutil
from pathlib import Path

# Training file that uses pure PyTorch - for when Lightning is unable to notice that your device has a GPU.
TRAIN_TRANSFORMS = {
    "mnist": T.Compose([T.Resize((32, 32)),
                        T.RandomHorizontalFlip(),
                        T.RandomVerticalFlip(),
                        T.ToTensor(),
                        ToRGB()]),
    "svhn": T.Compose([T.ToTensor(),
                       T.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))]),
    "synth_digits": T.Compose([T.Resize((32,32)),
                               T.RandomRotation(degrees=(-45,45)),
                               T.ToTensor(),
                               T.Normalize((0.4368, 0.4158, 0.3904), (0.2815, 0.2754, 0.2803))])
}

TEST_TRANSFORMS = {
    "mnist": T.Compose([T.Resize((32, 32)),
                        T.ToTensor(),
                        ToRGB()]),
    "synth_digits": T.Compose([T.Resize((32,32)),
                               T.ToTensor(),
                               T.Normalize((0.4368, 0.4158, 0.3904), (0.2815, 0.2754, 0.2803))]),
    "svhn": T.Compose([T.ToTensor(),
                       T.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))])
}

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = None
        self.sum = None
        self.avg = None
        self.val = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@model for the specified values of model
    :param output:
    :param target:
    :param topk:
    :return:
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def test_model(test_dataloader, model, device=DEVICE):
    """
    :param test_dataloader: The test dataloader
    :param model: The model to run the test dataloader over
    :param device: The device
    :param ss_batch_func: The function to convert each batch from test_dataloader to suit the self-supervision task.
                            If None, then we test for classification accuracy.
    :return: The accuracy of the model on the test dataloader and the corresponding losses for each batch.
    """
    print("testing model")
    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct = []
    losses = AverageMeter()

    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            class_out = model(inputs)
            loss = criterion(class_out, labels)
            losses.update(loss.item(), inputs.size(0))

            _, predicted = class_out.max(1)
            correct.append(predicted.eq(labels).cpu())

    acc = torch.cat(correct).numpy().mean() * 100.0

    print(f"classification average: {acc:.4f}%")

    return acc, losses


def save_checkpoint(model, weights_path, is_best: bool):
    """
    Saves model weights to disk.
    :param model: The model.
    :param weights_path: The path to save the weights.
    :param is_best: Whether the current weights performed best
    :param task_name: The name of the task.
    :return: None.
    """
    Path(weights_path).mkdir(exist_ok=True,parents=True)
    torch.save(model.state_dict(), f"{weights_path}/checkpoint.pt")

    if is_best:
        shutil.copyfile(f"{weights_path}/checkpoint.pt", f"{weights_path}/best.pt")


def get_dataloaders_svhn(batch_size, custom_transform_train=None, custom_transform_test=None):
    transform_train = TRAIN_TRANSFORMS['svhn']
    transform_test = TEST_TRANSFORMS['svhn']

    if custom_transform_train:
        transform_train = custom_transform_train

    if custom_transform_test:
        transform_test = custom_transform_test

    dataset_train = SVHN("/ml_datasets", split="train", download=True, transform=transform_train)
    dataset_test = SVHN("/ml_datasets", split="test", download=True, transform=transform_test)
    dataset_extra = SVHN("/ml_datasets", split="extra", download=True, transform=transform_train)
    return (DataLoader(ConcatDataset([dataset_train, dataset_extra]), shuffle=True, batch_size=batch_size),
            DataLoader(dataset_test))


def get_dataloaders_mnist(batch_size, custom_transform_train=None, custom_transform_test=None):
    train_transform = TRAIN_TRANSFORMS["mnist"]
    test_transform = TEST_TRANSFORMS["mnist"]
    if custom_transform_train:
        train_transform = custom_transform_train

    if custom_transform_test:
        test_transform = custom_transform_test

    dataset_train = MNIST(root="/ml_datasets", train=True, transform=train_transform)
    dataset_test = MNIST(root="/ml_datasets", train=False, transform=test_transform)
    dl_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(dataset=dataset_test)
    return dl_train, dl_test


def get_dataloaders_synth_digits(batch_size=64, custom_transform_train=None, custom_transform_test=None):
    train_transform = TRAIN_TRANSFORMS["synth_digits"]
    test_transform = TEST_TRANSFORMS["synth_digits"]
    if custom_transform_train:
        train_transform = custom_transform_train

    if custom_transform_test:
        test_transform = custom_transform_test

    dataset_train = SyntheticDigitsData(root="/ml_datasets", train=True, transform=train_transform)
    dataset_test = SyntheticDigitsData(root="/ml_datasets", train=False, transform=test_transform)
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset_test)
    return train_dataloader, test_dataloader


def get_dataset(dataset_name: str, custom_transform_train=None, custom_transform_test=None, batch_size=64):
    if dataset_name == "cifar10":
        raise Exception("no cifar10 for now.")
    elif dataset_name == "mnist":
        return get_dataloaders_mnist(batch_size, custom_transform_train, custom_transform_test)
    elif dataset_name == "svhn":
        return get_dataloaders_svhn(batch_size, custom_transform_train, custom_transform_test)
    elif dataset_name == "synth_digits":
        return get_dataloaders_synth_digits(batch_size, custom_transform_train, custom_transform_test)

    raise Exception("Invalid dataset name.")


def main(dataset_name,
         model_name,
         lr=0.005,
         epochs=25,
         test_pretrained=False,
         pretrained_dset="svhn",
         batch_size=64,
         custom_transform_train=None,
         custom_transform_test=None):
    dl_train, dl_test = get_dataset(dataset_name, custom_transform_train, custom_transform_test,
                                    batch_size=batch_size)

    model = all_classifiers[model_name]
    model.to(DEVICE)

    if test_pretrained:
        state_dict = torch.load(f"checkpoints/{pretrained_dset}/{model_name}/best.pt")
        model.load_state_dict(state_dict)
        acc, losses = test_model(dl_test, model)
        print(f"test set acc: {acc}")
        return

    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            nesterov=True
        )
    best_acc = 0

    for epoch in range(epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        model.train()

        for i, (inp, labels) in enumerate(dl_train):
            end = time.time()
            inp = inp.to(DEVICE)
            labels = labels.to(DEVICE)

            output = model(inp)
            loss = criterion(output, labels)

            # measure accuracy and record loss
            prec1 = accuracy(output, labels, topk=(1,))[0] # prec1 will always be the same.
            losses.update(loss.item(), inp.size(0))
            top1.update(prec1.item(), inp.size(0))

            # compute gradient and do SGD step
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # measure elapsed time
            batch_time.update(time.time() - end)

            if i % 100 == 0:
                print(f"Epoch: [{epoch}][{i}/{len(dl_train)}]\t"
                      f"Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                      f"Loss_SSH {loss:.4f} (Running avg: {losses.avg:.4f})\t"
                      )

        # Test dataloader
        acc, losses = test_model(dl_test, model)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint(
            model,
            f"checkpoints/{dataset_name}/{model_name}",
            is_best
        )
    print(f"best test acc: {best_acc}")


if __name__ == "__main__":
    # Get mean and std
    # Data augmentation always comes first before normalisation in preprocessing.
    # Also
    #dataset_train = SyntheticDigitsData(root="/ml_datasets", train=True, transform=TRAIN_TRANSFORMS["synth_digits"])
    #dataset_test = SyntheticDigitsData(root="/ml_datasets", train=False, transform=TEST_TRANSFORMS["synth_digits"])
    #complete_dset = torch.stack([dataset_train[x][0] for x in range(len(dataset_train))] + [dataset_test[x][0] for x in range(len(dataset_test))])
    #print(complete_dset.shape)
    #mean = torch.mean(complete_dset, dim=[0,2,3])
    #std = torch.std(complete_dset, dim=[0,2,3])
    #print(mean, std)
    """main("mnist",
         "linear",
         test_pretrained=True,
         pretrained_dset="svhn"
         )"""
    main("mnist",
         "mobilenet_v2",
         test_pretrained=True,
         pretrained_dset="svhn"
         )
