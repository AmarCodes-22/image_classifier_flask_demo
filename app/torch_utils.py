import io

import torch
from torch import nn
from torchvision import transforms
from torchvision.models.resnet import ResNet, BasicBlock
from PIL import Image


class DogCatClassifier(ResNet):
    def __init__(self):
        super(DogCatClassifier, self).__init__(BasicBlock, [2,2,2,2], num_classes=2)


dog_cat_classifier = DogCatClassifier()

path_to_weights = 'app/dog_cat_classifier_weights.pt'  # production
# path_to_weights = 'dog_cat_classifier_weights.pt'  # development

dog_cat_classifier.load_state_dict(
    torch.load(path_to_weights, map_location=torch.device('cpu'))
)
dog_cat_classifier.eval()


def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    image = transform(image)

    return image.unsqueeze(0)


def get_prediction(image_tensor):
    index_to_labels = {
        0: 'cat',
        1: 'dog'
    }

    outputs = dog_cat_classifier(image_tensor)
    _, pred = torch.max(outputs.data, 1)

    return index_to_labels[int(pred.item())]
