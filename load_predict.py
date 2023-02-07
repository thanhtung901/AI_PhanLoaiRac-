import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision import models, transforms

class_id = ['cardboard','glass','metal', 'paper','plastic','trash']
class Predict():
    def __init__(self, class_id):
        self.class_id = class_id

    def predict_labels(self, output):
        max_id = np.argmax(output.detach().numpy())
        predicted_label = self.class_id[max_id]
        return predicted_label
predictor = Predict(class_id)
def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor()])
    return my_transforms(image_bytes).unsqueeze(0)
path = '.\\trained_resnet50.pth'
'''
 vgg16 val acc: 84%
 resnet50 val acc:  83%
 trained_resnet50 86,7%
'''
def predict(img):
    net = torchvision.models.resnet50(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 6)
    # net = torchvision.models.vgg16(pretrained=True)
    # net.classifier[6] = nn.Linear(in_features=4096, out_features=6)
    model = net
    model.load_state_dict(torch.load(path))
    model.eval()

    data_input = transform_image(img)

    output = model(data_input)
    label = predictor.predict_labels(output)
    return label

if __name__ == '__main__':
    img = Image.open(r"bia.jpg")
    img.show()
    name = predict(img)
    print(name)
