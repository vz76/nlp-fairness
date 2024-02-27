'''

This program aims to evaluate the interpretability of Inception, a pre-trained image classifier (DOI: 10.48550/arXiv.1512.00567)
We will then perform adversarial exploits of its classification by selectively modifying gradients of input Imagenets, through four variations of the fast gradient sign method (FGSM):
- untargeted (cheetah -> jaguar, delta P: -0.103)
- targeted (cheetah -> _)
- untargeted, iterative (cheetah -> jaguar, delta P: 0.116)
- targeted, iterative (cheetah -> _, delta P: 0.110)

We conclude that while all variants of FGSM exploit susceptibilities in Inception, iterative variants of adversarial inputs yield the greatest performance in perturbing model classification. 
'''

import torch
import torch.nn
import numpy as np
import requests, io, json
import matplotlib.pyplot as plt
from torch.autograd.gradcheck import zero_gradients
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable

# Pytorch Imagenet's pretrained mean / std for the model
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

def main():
    inceptionv3 = models.inception_v3(pretrained=True)
    inceptionv3.eval()

    img = Image.open("~/cheetah1.jpg")
    img_hyena = Image.open("~/hyena1.jpg")

    preprocess = transforms.Compose([
                    transforms.Resize((299,299)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
    
    image_tensor = preprocess(img)
    image_tensor = image_tensor.unsqueeze(0) # add batch dimension.  C X H X W ==> B X C X H X W

    image_tensor_hyena = preprocess(img_hyena)
    image_tensor_hyena = image_tensor_hyena.unsqueeze(0) # add batch dimension.  C X H X W ==> B X C X H X W

    #convert tensors into variables
    img_variable = Variable(image_tensor, requires_grad=True) 
    img_variable_hyena = Variable(image_tensor_hyena, requires_grad=True) 

    output = inceptionv3.forward(img_variable)
    label_idx = torch.max(output.data, 1)[1][0]

    output_hyena = inceptionv3.forward(img_variable_hyena)
    label_idx_hyena = torch.max(output_hyena.data, 1)[1][0]

    f = open("~/imagenet-simple-labels.json")
    labels = json.load(f)
    x_pred = labels[label_idx]

    x_pred_hyena = labels[label_idx_hyena]

    output_probs = F.softmax(output, dim=1)
    #x_pred_prob =  round((torch.max(output_probs.data, 1)[0][0]) * 100,4)
    x_pred_prob =  torch.max(output_probs.data, 1)[0][0]

    output_probs_hyena = F.softmax(output_hyena, dim=1)
    x_pred_prob_hyena =  torch.max(output_probs_hyena.data, 1)[0][0]

    eps = 0.05
    y_true = 293   #cheetah
    target = Variable(torch.LongTensor([y_true]), requires_grad=False)
    
    visualize(image_tensor, "Original Cheetah", x_pred, x_pred_prob)
    perturb_eps_0_05 = fgsm(inceptionv3, labels, output, target, eps, image_tensor, img_variable)
    visualize(perturb_eps_0_05[0], "Untargeted FGSM Cheetah, eps = 0.05", perturb_eps_0_05[1], perturb_eps_0_05[2])

    epsilons = [0.001, 0.01, 0.02, 0.03, 0.05, 0.1, 0.25, 0.5, 1]
    for e in epsilons:
        pert = fgsm(inceptionv3, labels, output, target, e, image_tensor, img_variable)
        visualize(pert[0], "Untargeted FGSM Cheetah, eps = "+str(e), pert[1], pert[2])

    visualize(image_tensor, "Original Cheetah", x_pred, x_pred_prob)

    y_target = 276   #cheetah
    targeted = Variable(torch.LongTensor([y_target]), requires_grad=False)
    epsilons = [0, 0.001, 0.01, 0.02, 0.03, 0.05, 0.1, 0.25, 0.5, 1]
    for e in epsilons:
        pert = targeted_fgsm(inceptionv3, labels, output, targeted, e, image_tensor, img_variable)
        visualize(pert[0], "Targeted FGSM Cheetah, eps ="+str(e), pert[1], pert[2])

    visualize(image_tensor, "Original Cheetah", x_pred, x_pred_prob)

    y_true = 293   #cheetah
    target = Variable(torch.LongTensor([y_true]), requires_grad=False)

    pert = iterative_fgsm(inceptionv3, labels, image_tensor, target, 0.20, 0.05, 10, image_tensor, img_variable)
    visualize(pert[0], "Untargeted Iterative FGSM Cheetah, a = 0.05, eps = 0.2, iter = 10", pert[1], pert[2])

    visualize(image_tensor, "Original Cheetah", x_pred, x_pred_prob)

    y_true = 276   #cheetah
    target = Variable(torch.LongTensor([y_true]), requires_grad=False)

    pert = targeted_iterative_fgsm(inceptionv3, labels, image_tensor, target, 0.20, 0.05, 10, image_tensor, img_variable)
    visualize(pert[0], "Targeted Iterative FGSM Cheetah, a = 0.05, eps = 0.2, iter = 10", pert[1], pert[2])

def visualize(x, title, y, prob):
    # Function: Visualize an input image tensor with Matplotlib
    # Args: (x: input image tensor) (title: name of the image) (y: predicted label) (prob: confidence score of particular label)
    
    x = x.squeeze(0)     #remove batch dimension # B X C H X W ==> C X H X W
    x = x.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy() #reverse of normalization op - "unnormalize"
    x = np.transpose( x , (1,2,0))   # C X H X W  ==>   H X W X C
    x = np.clip(x, 0, 1)
    figure, ax = plt.subplots(1,1, figsize=(18,8))
    ax.imshow(x)
    ax.set_title(title, fontsize=20)
    ax.text(0.5,-0.13, "Prediction: {}\n Probability: {}".format(y, prob), size=15, ha="center",
         transform=ax.transAxes)
    plt.show()

def fgsm(inceptionv3, labels, output, target, epsilon, image_tensor, img_variable):
  # Function: Perturb an input adversarially using untargeted, non-iterative FGSM
  # Args: (inceptionv3: pretrained image classificaiton model) (labels: list of integers mapped to Imagenet dataset's labels) (output: model classification for input image tensor) 
  #       (target: desired label to be produced) (epsilon: magnitude of adversarial perturbation) (image_tensor: input image tensor) (img_variable: wrapper variable for input image tensor)
  # Return: (perturb_tensor: adversarial output image tensor) (p_pred: output label from model for adversarial input) (p_pred_prob: confidence score of model for output label)
  loss = torch.nn.CrossEntropyLoss()(output, target)
  loss.backward(retain_graph=True)

  perturb_tensor = image_tensor + epsilon * img_variable.grad.data.sign()
  perturb_variable = Variable(perturb_tensor, requires_grad=True)
  perturb_output = inceptionv3.forward(perturb_variable)
  perturb_label = torch.max(perturb_output.data, 1)[1][0]
  #print(perturb_label)

  p_pred = labels[perturb_label]
  #print(p_pred)

  p_output_probs = F.softmax(perturb_output, dim=1)
  #x_pred_prob =  round((torch.max(output_probs.data, 1)[0][0]) * 100,4)
  p_pred_prob =  torch.max(p_output_probs.data, 1)[0][0]
  #print(p_pred_prob)

  return perturb_tensor, p_pred, p_pred_prob

def targeted_fgsm(inceptionv3, labels, output, target, epsilon, image_tensor, img_variable):
  # Function: Perturb an input adversarially using targeted, non-iterative FGSM
  # Args: (inceptionv3: pretrained image classificaiton model) (labels: list of integers mapped to Imagenet dataset's labels) (output: model classification for input image tensor) 
  #       (target: desired label to be produced) (epsilon: magnitude of adversarial perturbation) (image_tensor: input image tensor) (img_variable: wrapper variable for input image tensor)
  # Return: (perturb_tensor: adversarial output image tensor) (p_pred: output label from model for adversarial input) (p_pred_prob: confidence score of model for output label)
  loss = torch.nn.CrossEntropyLoss()(output, target)
  loss.backward(retain_graph=True)

  perturb_tensor = image_tensor - epsilon * img_variable.grad.data.sign()
  perturb_variable = Variable(perturb_tensor, requires_grad=True)
  perturb_output = inceptionv3.forward(perturb_variable)
  perturb_label = torch.max(perturb_output.data, 1)[1][0]   #get an index(class number) of a largest element
  #print(perturb_label)

  p_pred = labels[perturb_label]
  #print(p_pred)

  p_output_probs = F.softmax(perturb_output, dim=1)
  #x_pred_prob =  round((torch.max(output_probs.data, 1)[0][0]) * 100,4)
  p_pred_prob =  torch.max(p_output_probs.data, 1)[0][0]
  #print(p_pred_prob)

  return perturb_tensor, p_pred, p_pred_prob

def iterative_fgsm(inceptionv3, labels, tensor, target, epsilon, alpha, epochs, image_tensor, img_variable):
  # Function: Perturb an input adversarially using untargeted, iterative FGSM
  # Args: (inceptionv3: pretrained image classificaiton model) (labels: list of integers mapped to Imagenet dataset's labels) (tensor: input image tensor) (target: desired label to be produced)
  #       (epsilon: magnitude of adversarial perturbation) (alpha: hyperparameter for adjustment at each iteration) (epochs: cycles of iteration) (image_tensor: input image tensor) (img_variable: wrapper variable for input image tensor)
  # Return: (perturb_tensor: adversarial output image tensor) (p_pred: output label from model for adversarial input) (p_pred_prob: confidence score of model for output label)
  variable_clone = Variable(tensor.clone(), requires_grad=True)

  img_variable = Variable(tensor, requires_grad=True)
  for i in range(epochs):
    img_variable = Variable(img_variable, requires_grad=True)
    output = inceptionv3.forward(img_variable)

    loss = torch.nn.CrossEntropyLoss()(output, target)
    loss.backward(retain_graph=True)

    perturb_variable = img_variable + alpha * img_variable.grad.data.sign()
    perturb_variable = torch.clamp(perturb_variable, variable_clone - epsilon, variable_clone + epsilon)

    img_variable = perturb_variable

    inceptionv3.zero_grad()

  perturb_output = inceptionv3.forward(perturb_variable)
  perturb_label = torch.max(perturb_output.data, 1)[1][0]
  #print(perturb_label)

  p_pred = labels[perturb_label]
  #print(p_pred)
  
  p_output_probs = F.softmax(perturb_output, dim=1)
  #x_pred_prob =  round((torch.max(output_probs.data, 1)[0][0]) * 100,4)
  p_pred_prob =  torch.max(p_output_probs.data, 1)[0][0]
  #print(p_pred_prob)

  return perturb_variable.detach(), p_pred, p_pred_prob

def targeted_iterative_fgsm(inceptionv3, labels, tensor, target, epsilon, alpha, epochs, image_tensor, img_variable):
  # Function: Perturb an input adversarially using targeted, iterative FGSM
  # Args: (inceptionv3: pretrained image classificaiton model) (labels: list of integers mapped to Imagenet dataset's labels) (tensor: input image tensor) (target: desired label to be produced)
  #       (epsilon: magnitude of adversarial perturbation) (alpha: hyperparameter for adjustment at each iteration) (epochs: cycles of iteration) (image_tensor: input image tensor) (img_variable: wrapper variable for input image tensor)
  # Return: (perturb_tensor: adversarial output image tensor) (p_pred: output label from model for adversarial input) (p_pred_prob: confidence score of model for output label)
  variable_clone = Variable(tensor.clone(), requires_grad=True)

  img_variable = Variable(tensor, requires_grad=True)
  for i in range(epochs):
    img_variable = Variable(img_variable, requires_grad=True)
    output = inceptionv3.forward(img_variable)

    loss = torch.nn.CrossEntropyLoss()(output, target)
    loss.backward(retain_graph=True)

    perturb_variable = img_variable - alpha * img_variable.grad.data.sign()
    perturb_variable = torch.clamp(perturb_variable, variable_clone - epsilon, variable_clone + epsilon)

    img_variable = perturb_variable

    inceptionv3.zero_grad()

  perturb_output = inceptionv3.forward(perturb_variable)
  perturb_label = torch.max(perturb_output.data, 1)[1][0]
  #print(perturb_label)

  p_pred = labels[perturb_label]
  #print(p_pred)
  
  p_output_probs = F.softmax(perturb_output, dim=1)
  #x_pred_prob =  round((torch.max(output_probs.data, 1)[0][0]) * 100,4)
  p_pred_prob =  torch.max(p_output_probs.data, 1)[0][0]
  #print(p_pred_prob)

  return perturb_variable.detach(), p_pred, p_pred_prob
