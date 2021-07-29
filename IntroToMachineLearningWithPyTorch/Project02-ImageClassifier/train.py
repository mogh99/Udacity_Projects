import argparse
from types import new_class
import torch
import torch.nn as nn

import torchvision.models as models
from torchvision import datasets, models, transforms

'''
    Parse all the required options:
    1. Dataset folder path [-d, --dataset][string]. 
    (Note: The dataset folder should follow the same hierarchy) 
    Dataset
        ├───test
        │   └───Class1
        │       └───image1
        ├───train
        │   └───Class1
        │       └───image1
        └───valid
            └───Class1
                └───image1
    2. Display accuracies and losses for both the training and validation [-s, --show][boolean].
    3. Specify architecture name from torchvision.models [-m, --model][string].
    4. Set the training hyperparameters (learning rate, number of hidden units, and epochs).
    4.1 learning rate [-lr, --learning-rate][float]
    4.2 number of hidden units [-n, --hidden-nodes][integer]
    4.3 number of epochs [-e, --epoch][integer]
    5. Training with GPU [-c, --cuda][boolean]
    6. Save trained model [-p, --checkpoint][boolean]

'''
def argument_parser():
    parser = argparse.ArgumentParser(prog="Train Torchvision Model"
                                    ,description="Use your dataset to train a model \
                                                  from torchvision models list")

    #Dataset Path
    parser.add_argument("--dataset", nargs="?", required=True, type=str, 
                        help="The dataset folder", metavar="path", dest="dataset_path")
    #Display Accuracies and Losses
    parser.add_argument("--show", nargs="?", default=False, type=bool, 
                        help="Display accuracies and losses while training", metavar="bool", 
                        dest="show")
    #Architecture Name
    parser.add_argument("--model", nargs="?", default="resnet18", type=str,
                        choices=["resnet18", "resent34", "resent50", 
                                "alexnet", "vgg13", "vgg16", "vgg19", 
                                "googlenet", "mobilenet_v2", "mobilenet_v3_large",
                                "mobilenet_v3_small"],
                        help="Select the network architecture", 
                        metavar="network_name", dest="architecture")
    #Hyperparameters
    #Learning Rate
    parser.add_argument( "--learning-rate", nargs="?", default=0.01, type=float, 
                        help="Optimizer learning rate", metavar="lr", dest="lr")
    #Number of Hidden Units
    parser.add_argument("--hidden-nodes", nargs="+", metavar="nodes", dest="nodes",
                        help="Specify the hidden nodes channels as a list e.g. 128, 256\
                              will generate two hidden layers", default=[0])
    #Number of Epochs
    parser.add_argument("--epochs", nargs="?", default=5, type=int,
                        help="Number of epochs", metavar="epoch", dest="epochs")
    #Use Cuda
    parser.add_argument("--cuda", nargs="?", default=False, type=bool,
                        help="Use cuda", metavar="bool", dest="cuda")
    #Save Checkpoint
    parser.add_argument("--checkpoint", nargs="?", default=False, type=bool,
                        help="Save the model checkpoint", metavar="bool", 
                        dest="checkpoint")

    args = parser.parse_args()

    return args

def load_dataset(dataset_path):
    train_dir = dataset_path + '/train'
    valid_dir = dataset_path + '/valid'
    test_dir = dataset_path + '/test'

    new_size = (224,224)

    train_data_transforms = transforms.Compose([transforms.Resize(size=new_size),
                                                transforms.RandomRotation(30),
                                                transforms.RandomHorizontalFlip(0.5),
                                                transforms.RandomVerticalFlip(0.5),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])]) 
    other_data_transforms = transforms.Compose([transforms.Resize(size=new_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225])])
    print(20*"*"+5*" "+"Loading The Data"+5*" "+20*"*")

    train_dataset = datasets.ImageFolder(train_dir, train_data_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, other_data_transforms)
    test_dataset  = datasets.ImageFolder(test_dir, other_data_transforms)

    num_classes = len(train_dataset.classes)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    print(f"Number of Classes = {num_classes}")
    print(f"Train Size = {len(train_dataset)}")
    print(f"Validation Size = {len(valid_dataset)}")
    print(f"Test Size = {len(test_dataset)}")

    print(20*"*"+5*" "+"Data Loading Completed"+5*" "+20*"*")
    print("\n")

    return [train_loader, valid_loader, test_loader], num_classes 


#To be able to add the new classifier to the choosen network architecture.
last_layer_name = {"resnet18": "fc", "resent34": "fc", "resent50": "fc", 
                   "alexnet": "classifier", "vgg13": "classifier", "vgg16": "classifier", 
                   "vgg19": "classifier", "googlenet": "fc", "mobilenet_v2": "classifier", 
                   "mobilenet_v3_large": "classifier", "mobilenet_v3_small": "classifier"}
def model_build(architecture, hidden_nodes, num_classes):
    print(20*"*"+5*" "+"Build The Model"+5*" "+20*"*")

    model = eval("models."+architecture+"(pretrained=True)") #e.g. model = models.alexnet(pretrained=True)

    #Freeze all the pretrained parameters
    for param in model.parameters():
        param.requires_grad = False

    old_classifier = getattr(model, last_layer_name[architecture])

    new_classifier = classifier_build(old_classifier, hidden_nodes, num_classes)
    
    setattr(model, last_layer_name[architecture], new_classifier)

    print(model)

    print(20*"*"+5*" "+"Building The Model Completed"+5*" "+20*"*")
    print("\n")

    return model

def classifier_build(old_classifier, hidden_nodes, num_classes):
    #Some architectures have single element or multiple elements in the classifier layer
    #To solve this issue check the type of the old_classifier and act accordingly.
    in_features = 0
    if type(old_classifier) == list:
        in_features = old_classifier[0].in_features
    #Only one element at the classifier layer.
    else:
        in_features = old_classifier.in_features

    new_classifier = []

    hidden_nodes.insert(0, in_features)
    
    for index in range(0, len(hidden_nodes)):
        if hidden_nodes[index] != 0 and index < len(hidden_nodes):
            new_classifier.append(nn.Linear(hidden_nodes[index], hidden_nodes[index+1]))
            
    #new_classifier.append(nn.Linear(123,123))
    #new_classifier.append(nn.ReLU())
    #new_classifier.append(nn.Dropout(0.5))

    return nn.Sequential(*new_classifier)

def optimizer_criterion():
    pass

if __name__ == "__main__":
    args = argument_parser()

    data_loaders, num_classes = load_dataset(args.dataset_path)

    model = model_build(args.architecture, args.nodes, num_classes)