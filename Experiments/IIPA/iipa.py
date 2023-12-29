import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the ResNet50 model
iipa_model = models.resnet50()
iipa_model.fc = torch.nn.Linear(in_features=2048, out_features=1)

# Load the pre-trained weights
checkpoint = torch.load('Experiments\IIPA\model-resnet50.pth', map_location=device)
iipa_model.load_state_dict(checkpoint)

def getiipa(image_np, iipa_model):

    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(image_np.astype('uint8'), 'RGB')

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
    image = transform(image)

    # Move the input tensor to the same device as the model
    image = image.unsqueeze(0).to(device)
    iipa_model = iipa_model.to(device)

    # Predict IIPA score
    with torch.no_grad():
        iipa_model.eval()
        iipa_score = iipa_model(image).item()

    return iipa_score