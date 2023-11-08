import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import time


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
        
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class CnnDroneControlModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        #self.layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4)
        self.network = nn.Sequential(

            # (hierarchical feature extractor)

            # Layer 0: Input Layer
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, padding=0),
            nn.Tanh(),

            # Layer 1: Convolutional Layer
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, padding=0),
            nn.Tanh(),

            # Layer 2: MaxPooling Layer
            nn.MaxPool2d(kernel_size=2),

            # Layer 3: Convolutional Layer
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, padding=0),
            nn.Tanh(),

            # Layer 4: MaxPooling Layer
            nn.MaxPool2d(kernel_size=2),

            # Layer 5: Convolutional Layer
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, padding=0),
            nn.Tanh(),

            # Layer 6: MaxPooling Layer
            nn.MaxPool2d(kernel_size=2),

            # Layer 7: Convolutional Layer
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, padding=0),
            nn.Tanh(),

            # Layer 8: MaxPooling Layer
            nn.MaxPool2d(kernel_size=2), #32 * 3 * 3

            # Layer 9: Fully Connected Layer (general classifier)
            nn.Flatten(),
            nn.Linear(32 * 3 * 3, 200),
            nn.Tanh(),

            # Layer 10: Output Layer
            nn.Linear(200, 3),
            nn.Softmax(dim=1)
            )
        
    def forward(self, xb):
        return self.network(xb)



#Check for GPU & Load Model
device = torch.device('cpu') #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = to_device(CnnDroneControlModel(), device)
model.load_state_dict(torch.load('weights_and_biases.pth', map_location=torch.device('cpu')))



transform = transforms.Compose([
    transforms.Resize((101, 101)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return image.classes[preds[0].item()]



path = 'test.jpg'
image = Image.open(str(path))
image = transform(image)
image.classes = ['lc', 'sc', 'rc']

start_time = time.time()
prediction = predict_image(image, model)
end_time = time.time()
elapsed_time = end_time - start_time

print(prediction, '' , round(elapsed_time, 4))





