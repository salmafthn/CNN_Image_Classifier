from model import Model
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import SGD
from itertools import product
import numpy as np
import os

# Hyperparameters grid to search over
learning_rates = [0.1]
batch_sizes = [64, 128, 256]
momentums = [0.9, 0.95]

# Combine all hyperparameter options
hyperparameter_combinations = list(product(learning_rates, batch_sizes, momentums))

best_acc = 0
best_params = None

# Define transformations for the training and test dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load MNIST datasets
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

for lr, batch_size, momentum in hyperparameter_combinations:
    print(f"Training with learning_rate={lr}, batch_size={batch_size}, momentum={momentum}")
    
    # Update DataLoader with new batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, optimizer, and loss function with new hyperparameters
    model = Model().to(device)
    sgd = SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_fn = nn.CrossEntropyLoss()
    
    all_epoch = 10  # You can adjust the number of epochs
    for current_epoch in range(all_epoch):
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            train_x = train_x.to(device)
            train_label = train_label.to(device)
            sgd.zero_grad()
            predict_y = model(train_x.float())
            loss = loss_fn(predict_y, train_label.long())
            loss.backward()
            sgd.step()
        
        # Evaluate on test set
        all_correct_num = 0
        all_sample_num = 0
        model.eval()
        
        for idx, (test_x, test_label) in enumerate(test_loader):
            test_x = test_x.to(device)
            test_label = test_label.to(device)
            predict_y = model(test_x.float()).detach()
            predict_y = torch.argmax(predict_y, dim=-1)
            current_correct_num = predict_y == test_label
            all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
        
        acc = all_correct_num / all_sample_num
        print(f'Accuracy: {acc:.3f}', flush=True)
        
        if acc > best_acc:
            best_acc = acc
            best_params = {'learning_rate': lr, 'batch_size': batch_size, 'momentum': momentum}
            # Save the best model
            if not os.path.exists('models'):
                os.mkdir('models')
            torch.save(model, f'models/best_model_{acc:.3f}.pkl')

print(f"Best Accuracy: {best_acc:.3f} with parameters: {best_params}")
