import argparse
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models

from torchvision import datasets, models, transforms

'''
    Parse all the required options:
    1. Dataset folder path [--dataset][string]. 
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
    2. Display accuracies and losses for both the training and validation [--show][boolean].
    3. Specify architecture name from torchvision.models [--model][string].
    4. Set the training hyperparameters (learning rate, number of hidden units, and epochs).
    4.1 learning rate [--learning-rate][float]
    4.2 number of hidden units [--hidden-nodes][integer]
    4.3 number of epochs [--epoch][integer]
    5. Training with GPU [--cuda][boolean]
    6. Save trained model [--checkpoint][boolean]
'''
def argument_parser():
    parser = argparse.ArgumentParser(prog="Train Torchvision Model"
                                    ,description="Use your dataset to train a model \
                                                  from torchvision models list")

    #Dataset Path
    parser.add_argument("--dataset", nargs="?", required=True, type=str, 
                        help="The dataset folder", metavar="path", dest="dataset_path")
    #Display Accuracies and Losses
    parser.add_argument("--show", dest="show", action='store_true',
                        help="Display accuracies and losses while training")
    #Architecture Name
    parser.add_argument("--model", nargs="?", default="resnet18", type=str,
                        choices=["resnet18", "resent34", "resent50", 
                                "alexnet", "vgg13", "vgg16", "vgg19", 
                                "googlenet", "mobilenet_v2", "mobilenet_v3_large",
                                "mobilenet_v3_small"],
                        help="Select the network architecture", 
                        dest="architecture")
    #Hyperparameters
    #Learning Rate
    parser.add_argument( "--learning-rate", nargs="?", default=0.01, type=float, 
                        help="Optimizer learning rate", metavar="lr", dest="lr")
    #Number of Hidden Units
    parser.add_argument("--hidden-nodes", nargs="+", metavar="nodes", dest="nodes", type=int,
                        help="Specify the hidden nodes channels as a list e.g. 128, 256\
                              will generate two hidden layers", default=[])
    #Number of Epochs
    parser.add_argument("--epochs", nargs="?", default=5, type=int,
                        help="Number of epochs", metavar="epochs", dest="epochs")
    #Use Cuda
    parser.add_argument("--cuda", default=False, help="Use cuda",
                        dest="cuda", action='store_true')
    #Save Checkpoint
    parser.add_argument("--checkpoint", default=False, help="Save the model checkpoint",
                        dest="checkpoint", action='store_true')

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

    return {"train":train_loader, "valid":valid_loader, "test":test_loader}, num_classes, train_dataset.class_to_idx


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
    hidden_nodes.append(num_classes)

    for index in range(0, len(hidden_nodes)):
        if index < (len(hidden_nodes) - 1):
            new_classifier.append(nn.Linear(hidden_nodes[index], hidden_nodes[index+1]))
            #Add ReLU and Dropout to all the layers except the last layer.
            if index < (len(hidden_nodes) - 2):
                new_classifier.append(nn.ReLU())
                new_classifier.append(nn.Dropout(0.5))

    return nn.Sequential(*new_classifier)

def optimizer_criterion(parameters, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    return optimizer, criterion
    
def train_validation(n_epochs, loaders, model, optimizer, criterion, using_cuda, show): 
    print(20*"*"+5*" "+"Train The Model"+5*" "+20*"*")
   
    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        valid_loss = 0.0
        train_correct = 0
        valid_correct = 0

        model.train()
        if using_cuda:
            model.cuda()
            
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if using_cuda:
                data, target = data.cuda(), target.cuda()

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # sum all the correct predictions to find the training accuracy
            _, predicted_labels = output.max(1)
            train_correct += (predicted_labels == target).float().sum().cpu().numpy()
            # update running training loss
            train_loss += ((1 / (batch_idx + 1)) * (loss.data.item() - train_loss))
            # print updates every 5 batches
            if (batch_idx + 1) % 5 == 0 and show:
                print(f"Epoch: {epoch} \tBatch Index: {batch_idx+1} \tTraining Loss: {train_loss}")

        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if using_cuda:
                data, target = data.cuda(), target.cuda()

            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # sum all the correct predictions to find the validation accuracy
            _, predicted_labels = output.max(1)
            valid_correct += (predicted_labels == target).float().sum().cpu().numpy()
            # update running validation loss 
            valid_loss += (loss.data.item() - valid_loss) / (batch_idx + 1)

        # print training/validation statistics
        if show:
            print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.6f}% \tValidation Loss: {:.6f} \tValidation Accuracy {:.6f}%'.format(
                epoch,
                train_loss,
                (train_correct/len(loaders["train"].dataset))*100,
                valid_loss,
                (valid_correct/len(loaders["valid"].dataset))*100
                ))

    print('Training Loss: {:.6f} \tTraining Accuracy: {:.6f}% \tValidation Loss: {:.6f} \tValidation Accuracy {:.6f}%'.format(
                train_loss,
                (train_correct/len(loaders["train"].dataset))*100,
                valid_loss,
                (valid_correct/len(loaders["valid"].dataset))*100
                ))

    print(20*"*"+5*" "+"Complete Training"+5*" "+20*"*")
    print("\n")
    
    return model

def test(loaders, model, criterion, using_cuda):
    print(20*"*"+5*" "+"Test The Model"+5*" "+20*"*")

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    # set the module to evaluation mode
    model.eval()
    if using_cuda:
        model.cuda()
        
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if using_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data.item() - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))
    
    print(20*"*"+5*" "+"Complete Testing"+5*" "+20*"*")
    print("\n")

def save_checkpoint(model, architecture, class_to_idx):
    print(20*"*"+5*" "+"Saving The Model"+5*" "+20*"*")

    checkpoint_path = "model.pth"

    #The name of the attribute to change the new_classifier with
    attribute_to_change = last_layer_name[architecture]

    new_classifier = getattr(model, attribute_to_change)

    checkpoint = {
    "state_dict": model.state_dict(),
    "model_architecutre": architecture,
    "attribute_to_change": attribute_to_change,
    "new_classifier": new_classifier,
    "class_to_idx": class_to_idx,
    }

    torch.save(checkpoint, checkpoint_path)

    print(20*"*"+5*" "+"Complete Saving"+5*" "+20*"*")
    print("\n")

def using_cuda(use_cuda):
    cuda_available = torch.cuda.is_available()
    using_cuda = cuda_available and use_cuda

    if cuda_available and using_cuda:
        print(20*"$"+" Using Cuda "+20*"$")
    else:
        print(20*"$"+" Using CPU "+20*"$")

    print("\n")
        
    return using_cuda     

if __name__ == "__main__":
    args = argument_parser()

    data_loaders, num_classes, class_to_idx = load_dataset(args.dataset_path)

    model = model_build(args.architecture, args.nodes, num_classes)

    optimizer, criterion = optimizer_criterion(model.parameters(), args.lr)

    using_cuda = using_cuda(args.cuda)

    model = train_validation(args.epochs, data_loaders, model, optimizer, criterion, using_cuda, args.show)

    if args.checkpoint:
        save_checkpoint(model, args.architecture, class_to_idx)

    test(data_loaders, model, criterion, using_cuda)

    