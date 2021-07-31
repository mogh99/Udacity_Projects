import argparse
import torch
import json
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

from PIL import Image
from torchvision import models, transforms

'''
    Parse all the required options:
    1. Image path [--image-path][string]
    2. Checkpoint path [--checkpoint][string]
    3. Top-K [-k][int]
    4. Use Cuda [--cuda][bool]
    5. JSON FILE [--category-names][string]
'''
def argument_parser():
    parser = argparse.ArgumentParser(prog="Train Torchvision Model"
                                    ,description="Use your dataset to train a model \
                                                  from torchvision models list")

    #Dataset Path
    parser.add_argument("--image-path", nargs="?", required=True, type=str, 
                        help="The image path", metavar="image_path", dest="image_path")
    #Checkpoint Path
    parser.add_argument("--checkpoint-path", nargs="?", required=True, type=str, 
                        help="The checkpoint path", metavar="checkpoint_path", dest="checkpoint_path")
    #Top-Ks
    parser.add_argument("--top-k", nargs="?", default=5, type=int,
                        help="Top-K", metavar="topk", dest="topk")
    # Use Cuda
    parser.add_argument("--cuda",help="Use cuda", 
                        dest="cuda", action='store_true')
    #JSON file category
    parser.add_argument("--category-names", nargs="?", type=str, default="cat_to_name.json",
                        help="The category names file path", metavar="category_names_path", dest="category_names_path")

    args = parser.parse_args()

    return args

def load_image(image):
    image = Image.open(image)
    data_transforms = transforms.Compose([transforms.Resize(size=(224,224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    
    transformed_image = data_transforms(image)
    transformed_image = transformed_image.unsqueeze(0)

    return image, transformed_image

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    model = eval("models."+checkpoint["model_architecutre"]+"(pretrained=True)") #e.g. model = models.alexnet(pretrained=True)

    setattr(model, checkpoint["attribute_to_change"], checkpoint["new_classifier"])

    model.load_state_dict(checkpoint["state_dict"])

    model.class_to_idx = checkpoint["class_to_idx"]

    return model


def predict(image, model, cat_to_name, topk, using_cuda):
    if using_cuda:
        device = torch.device("cuda")
        model = model.to(device)
        image = image.to(device)
    else:
        device = torch.device("cpu")
        model = model.to(device)
        image = image.to(device)

    model.eval()

    output = model(image)

    prob = nn.functional.softmax(output, dim=1)
    top_p, top_class = prob.topk(topk, dim=1)
    
    top_p, top_class = top_p.detach().cpu().numpy()[0], top_class.cpu().numpy()[0]
    
    fliped_idxes = {value: key for key, value in model.class_to_idx.items()}
    
    top_class_names = [cat_to_name[fliped_idxes[idx]] for idx in top_class]

    return top_class_names, top_p

def display_image_topk(image, top_class_names, top_p):
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle(f"Top-{len(top_p)} Class Probabilities")
    ax1.bar(top_class_names, top_p, color="green")
    ax2.imshow(image)

    plt.show()

def using_cuda(use_cuda):
    cuda_available = torch.cuda.is_available()
    using_cuda = cuda_available and use_cuda

    if cuda_available and using_cuda:
        print(20*"$"+" Using Cuda "+20*"$")
    else:
        print(20*"$"+" Using CPU "+20*"$")
        
    return using_cuda  

def read_category_names(file_path):
    with open(file_path, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name

if __name__ == "__main__":
    args = argument_parser()

    using_cuda = using_cuda(args.cuda)

    cat_to_name = read_category_names(args.category_names_path)

    model = load_checkpoint(args.checkpoint_path)
    
    image, transformed_image = load_image(args.image_path)

    top_class_names, top_p = predict(transformed_image, model, cat_to_name, args.topk, using_cuda)

    display_image_topk(image, top_class_names, top_p)