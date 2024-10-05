import torch
from model import Model 
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np

best_model_path = 'models/best_model_0.988.pkl' 
model = torch.load(best_model_path)
model.eval() 

test_dataset = MNIST(root='./test', train=False, transform=ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


all_correct_num = 0
all_sample_num = 0

with torch.no_grad():  
    for test_x, test_label in test_loader:
        test_x, test_label = test_x.to(device), test_label.to(device)
        predict_y = model(test_x.float()).detach()  
        predict_y = torch.argmax(predict_y, dim=-1) 
        
        current_correct_num = predict_y == test_label
        all_correct_num += np.sum(current_correct_num.cpu().numpy())
        all_sample_num += current_correct_num.shape[0]

accuracy = all_correct_num / all_sample_num
print(f"Test Accuracy: {accuracy:.3f}")
