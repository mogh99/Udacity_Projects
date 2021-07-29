import argparse
import torch

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
    parser.add_argument("--dataset", nargs=1, required=True, type=str, 
                        help="The dataset folder", metavar="path", dest="dataset_path")
    #Display Accuracies and Losses
    parser.add_argument("--show", nargs=1, default=False, type=bool, 
                        help="Display accuracies and losses while training", metavar="bool", 
                        dest="show")
    #Architecture Name
    parser.add_argument("--model", nargs=1, default="resnet18", type=str,
                        choices=["resnet18", "resent34", "resent50", 
                                "alexnet", "vgg13", "vgg16", "vgg19", 
                                "googlenet", "mobilenet_v2", "mobilenet_v3_large",
                                "mobilenet_v3_small"],
                        help="Select the network architecture", 
                        metavar="network_name", dest="architecture")
    #Hyperparameters
    #Learning Rate
    parser.add_argument( "--learning-rate", nargs=1, default=0.01, type=float, 
                        help="Optimizer learning rate", metavar="lr", dest="lr")
    #Number of Hidden Units
    parser.add_argument("--hidden-nodes", nargs=1, metavar="nodes", dest="nodes")
    #Number of Epochs
    parser.add_argument("--epochs", nargs=1, default=5, type=int,
                        help="Number of epochs", metavar="epoch", dest="epochs")
    #Use Cuda
    parser.add_argument("--cuda", nargs=1, default=False, type=bool,
                        help="Use cuda", metavar="bool", dest="cuda")
    #Save Checkpoint
    parser.add_argument("--checkpoint", nargs=1, default=False, type=bool,
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

    train_dataset = datasets.ImageFolder(train_dir, train_data_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, other_data_transforms)
    test_dataset  = datasets.ImageFolder(test_dir, other_data_transforms)

    num_classes = len(train_dataset.classes)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    return [train_loader, valid_loader, test_loader], num_classes 

#To be able to add the new classifier to the choosen network architecture.
last_layer_name = {"resnet18": "fc", "resent34": "fc", "resent50": "fc", 
                   "alexnet": "classifier", "vgg13": "classifier", "vgg16": "classifier", 
                   "vgg19": "classifier", "googlenet": "fc", "mobilenet_v2": "classifier", 
                   "mobilenet_v3_large": "classifier", "mobilenet_v3_small": "classifier"}
def model_build(architecture, num_classes):
    model = eval("models."+architecture+"()") #e.g. models.alexnet()
    in_features = getattr(model, last_layer_name[architecture])[0].in_features
    classifier = classifier_build(in_features)

def classifier_build():
    pass

def optimizer():
    pass

def criterion():
    pass

if __name__ == "__main__":
    args = argument_parser()

    model = model_build()
