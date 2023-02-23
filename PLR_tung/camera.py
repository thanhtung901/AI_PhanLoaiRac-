import numpy as np
import torch
import torch.nn
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
import PIL
import cv2

# Let's preprocess the inputted frame
data_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor()

])

net = torchvision.models.resnet50(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 6)
path = '.\\best_model.pth'
model = net
model.load_state_dict(torch.load(path))
model.eval()


# Set the Webcam
def Webcam_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

class_id = ['bia','thuytinh','lon', 'giay','nhua','rac']
def argmax(model,inputs):
    output = model(inputs)
    max_id = np.argmax(output.detach().numpy())
    output_arr = output.detach().numpy()
    out_sorted = np.sort(output_arr)

    max1 = out_sorted[0][-1]
    max2 = out_sorted[0][-2]
    result = max1 - max2
    print('result', result)
    if result > 0.5:
        predicted_label = class_id[max_id]
    else:
        predicted_label = 'unknown'
    return result,predicted_label


def preprocess(image):
    image = PIL.Image.fromarray(image)  # Webcam frames are numpy array format
    # Therefore transform back to PIL image
    image = data_transforms(image)
    image = image.float()
    # image = Variable(image, requires_autograd=True)
    image = image.unsqueeze(0)  # I don't know for sure but Resnet-50 model seems to only
    # accpets 4-D Vector Tensor so we need to squeeze another
    return image  # dimension out of our 3-D vector Tensor


# Let's start the real-time classification process!

cap = cv2.VideoCapture(0)  # Set the webcam
Webcam_720p()
fps = 0
sequence = 0
result = ''
score = 0.0
while True:
    ret, frame = cap.read()  # Capture each frame

    if fps == 4:
        image = frame[100:450, 150:570]
        image_data = preprocess(image)
        score, result = argmax(model,image_data)
        fps = 0

    fps += 1
    cv2.putText(frame, '%s' % (result), (950, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(frame, '(score = %.5f)' % (score), (950, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # cv2.rectangle(frame, (400, 150), (900, 550), (250, 0, 0), 2)
    cv2.imshow("DETECTER", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyWindow()