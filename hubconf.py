from cifar10_models.densenet import densenet121, densenet161, densenet169
from cifar10_models.googlenet import googlenet
from cifar10_models.inception import inception_v3

if __name__ == "__main__":
    model = inception_v3(pretrained=True)
    print(model)