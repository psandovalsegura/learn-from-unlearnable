# import sys
# sys.path.append('../')

from datasets import construct_train_dataset

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from utils_projection import CustomTensorDataset
from torchvision.utils import make_grid, save_image
from constants import LINEAR_CKPT_DIR, DATA_SETUPS, NUM_CLASSES

parser = argparse.ArgumentParser(description='Train a linear model on CIFAR10')
parser.add_argument('poison_identifier', type=str, default=None, 
                    help='Poison identifier (from DATA_SETUPS in constants.py). Determines poison perturbations to test for linear separability.')
parser.add_argument('--epochs', type=int, default=20)
args = parser.parse_args()

dataset = construct_train_dataset(args.poison_identifier, normalize=False, transforms_key='test_transform')
print("loaded training data")

class LinearModel(torch.nn.Module):
    def __init__(self, image_shape, num_classes=10):
        super(LinearModel, self).__init__()
        input_dim = np.prod(image_shape)
        self.fc = torch.nn.Linear(input_dim, num_classes, bias=False)
        
    def forward(self, x):
        x = self.fc(x.view((x.shape[0], -1)))
        return x
    
    def project_weights(self, max_val=1.0):
        self.fc.weight.data = torch.clamp(self.fc.weight.data, -max_val, max_val)

def normalize_zero_one(delta):
    return (delta - torch.min(delta)) / (torch.max(delta) - torch.min(delta))

train_data = [x[0] for x in dataset]
label_data = [x[1] for x in dataset]
tensor_dataset = CustomTensorDataset((torch.stack(train_data), torch.LongTensor(label_data)))

image_shape = dataset[0][0].shape
num_classes = NUM_CLASSES[DATA_SETUPS[args.poison_identifier]['dataset_name']]
model = LinearModel(image_shape, num_classes=num_classes).cuda()
EPOCHS = args.epochs
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(EPOCHS*0.5),int(EPOCHS*0.75)])
trainloader = torch.utils.data.DataLoader(tensor_dataset, batch_size=1024)
for epoch in tqdm(range(EPOCHS), desc='training linear model'):
    correct = 0
    for batch_data, batch_target in trainloader:
        batch_data, batch_target = batch_data.float().cuda(), batch_target.cuda()
        output = model(batch_data)
        pred = output.max(1)[1]
        correct += pred.eq(batch_target).sum().item()
        loss = torch.nn.functional.cross_entropy(output, batch_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
        
    accuracy = correct/len(dataset)  

recovered_p = []
for i in range(10):
    p = model.fc.weight[i].reshape(3,32,32)
    normalized_p = normalize_zero_one(p)
    recovered_p.append(normalized_p)
recovered_p = torch.stack(recovered_p)
image_array = make_grid(recovered_p, nrow=10)

os.makedirs(LINEAR_CKPT_DIR, exist_ok=True)

# Save image visualization
save_image(image_array, f'{LINEAR_CKPT_DIR}/{args.poison_identifier}.png')

# Save the model
save_path = os.path.join(LINEAR_CKPT_DIR, args.poison_identifier+'.pt')
torch.save(model.state_dict(), save_path)
print(f'Saved linear model to {save_path}')

