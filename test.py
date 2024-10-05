import torch
from model import Model  # Sesuaikan dengan path model Anda
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np

# Load the best model
best_model_path = 'models/best_model_0.988.pkl'  # Path ke model yang disimpan
model = torch.load(best_model_path)
model.eval()  # Set model ke mode evaluasi

# Load test dataset
test_dataset = MNIST(root='./test', train=False, transform=ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


all_correct_num = 0
all_sample_num = 0

# Test the model
with torch.no_grad():  # Disable gradient computation for testing
    for test_x, test_label in test_loader:
        test_x, test_label = test_x.to(device), test_label.to(device)
        predict_y = model(test_x.float()).detach()  # Perform inference
        predict_y = torch.argmax(predict_y, dim=-1)  # Get the predicted class
        
        # Calculate number of correct predictions
        current_correct_num = predict_y == test_label
        all_correct_num += np.sum(current_correct_num.cpu().numpy())
        all_sample_num += current_correct_num.shape[0]

# Calculate and print accuracy
accuracy = all_correct_num / all_sample_num
print(f"Test Accuracy: {accuracy:.3f}")
