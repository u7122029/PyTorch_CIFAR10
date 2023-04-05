from cifar10_models.densenet import densenet121, densenet161, densenet169

if __name__ == "__main__":
    model = densenet161(pretrained=True)
    print(model)