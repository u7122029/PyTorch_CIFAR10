import lightning as pl
import torch
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torchmetrics import Accuracy

from cifar10_models.densenet import densenet121, densenet161, densenet169
from cifar10_models.googlenet import googlenet
from cifar10_models.inception import inception_v3
from cifar10_models.mobilenetv2 import mobilenet_v2
from cifar10_models.resnet import resnet18, resnet34, resnet50
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from cifar10_models.linear import linear
from cifar10_models.lenet5 import lenet5

all_classifiers = {
    "vgg11_bn": vgg11_bn(),
    "vgg13_bn": vgg13_bn(),
    "vgg16_bn": vgg16_bn(),
    "vgg19_bn": vgg19_bn(),
    "resnet18": resnet18(),
    "resnet34": resnet34(),
    "resnet50": resnet50(),
    "densenet121": densenet121(),
    "densenet161": densenet161(),
    "densenet169": densenet169(),
    "mobilenet_v2": mobilenet_v2(),
    "googlenet": googlenet(),
    "inception_v3": inception_v3(),
    "linear": linear(),
    "lenet5": lenet5()
}


class CIFAR10Module(pl.LightningModule):
    @property
    def hparams(self):
        return self._hparams

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy("multiclass", num_classes=10)

        self.model = all_classifiers[self.hparams["classifier"]]

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy * 100

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log_dict({"train_loss": loss, "train_acc": accuracy}, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "train_acc": accuracy}

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log_dict({"val_loss": loss, "val_acc": accuracy}, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "val_acc": accuracy}

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log_dict({"test_loss": loss, "test_acc": accuracy}, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "accuracy": accuracy}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams["learning_rate"],
            #weight_decay=self.hparams["weight_decay"],
            momentum=0.9,
            nesterov=True
        )
        return optimizer

    @hparams.setter
    def hparams(self, value):
        self._hparams = value
