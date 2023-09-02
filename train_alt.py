import torch
from torchvision.datasets import MNIST
import torchvision.transforms as T
from torch.utils.data import DataLoader
from data import ToRGB
import torch.nn as nn
import torch.optim as optim
from module import all_classifiers
import time
import shutil
from pathlib import Path

# Training file that uses pure PyTorch - for when Lightning is unable to notice that your device has a GPU.


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

train_transform = T.Compose([T.Resize((32,32)),
                                     T.RandomHorizontalFlip(),
                                     T.RandomVerticalFlip(),
                                     T.ToTensor(),
                                     ToRGB()])
test_transform = T.Compose([T.Resize((32,32)),
                            T.ToTensor(),
                            ToRGB()])


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


def main(model_name, lr=0.005, epochs=25, test_pretrained=False):
    dataset_train = MNIST(root="/ml_datasets", train=True, transform=train_transform)
    dataset_test = MNIST(root="/ml_datasets", train=False, transform=test_transform)
    dl_train = DataLoader(dataset=dataset_train,batch_size=64, shuffle=True)
    dl_test = DataLoader(dataset=dataset_test)

    model = all_classifiers[model_name]
    model.to(DEVICE)

    if test_pretrained:
        state_dict = torch.load(f"checkpoints/mnist/{model_name}/best.pt")
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
            f"checkpoints/mnist/{model_name}",
            is_best
        )
    print(f"best test acc: {best_acc}")


if __name__ == "__main__":
    main("googlenet")