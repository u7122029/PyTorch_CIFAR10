from cifar10_models.densenet import densenet121, densenet161, densenet169
from cifar10_models.googlenet import googlenet

if __name__ == "__main__":
    model = googlenet(pretrained=True)
    print(model)