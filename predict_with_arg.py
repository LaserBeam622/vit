import time
import base64

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

import datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from transformers import ViTModel, ViTFeatureExtractor, logging

logging.set_verbosity_error()

from torchvision import transforms

model_name = "google/vit-base-patch16-224"

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.vit = ViTModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels):
        outputs = self.vit(pixel_values=pixel_values)

        output = self.dropout(outputs.last_hidden_state[:,0])
        logits = self.classifier(output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        
        if loss is not None:
            return logits, loss.item()
        else:
            return logits, None
        
    def config(self):
        return self.vit.config

val_test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5], [0.5])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = 7
model = Net().to(device)
model.load_state_dict(torch.load("full_best_model.pth"))
model.eval()
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
class_name = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised', 'Neutral']

@app.route("/predict", methods = ['GET'])
def predict():

    base64url_img = request.args.get('query')
    base64_img = base64url_img.replace("-", "+").replace("_", "/").replace(",", "=")
    base64_to_img = base64.b64decode(base64_img)
    npimg = np.frombuffer(base64_to_img, dtype=np.uint8); 
    image = cv2.imdecode(npimg, 1)
    start = time.time()

    with torch.no_grad():
        x = val_test_transform(image)
        x = np.squeeze(np.array(x))

        for index, array in enumerate(x):
            x[index] = np.squeeze(array)

        x = torch.tensor(np.stack(feature_extractor(x)['pixel_values'], axis=0))
        x = x.to(device)

        prediction, _ = model(x, None)
        
        confidence = F.softmax(prediction.cpu().reshape(-1,), dim = 0)
        predicted_class = np.argmax(confidence)

        results = {class_name[i]: confidence[i] for i in range(len(confidence))}
        sort = sorted(results.items(), key=lambda item: item[1], reverse=True)
        more = ""

        for i in sort:
            more += i[0] + ": " + str(round(i[1].item() * 100, 2)) + "%;"

        more = more[:-1]

    end = time.time()
    est = datetime.timedelta(seconds=round(end - start, 2))
    total = str(est)
    est_time = ""

    if est.seconds >= 3600:
        est_time = str(int(total.split(":")[0])) + " hour(s)", str(int(total.split(":")[1])) + " minute(s)", str(float(total.split(":")[2])) + " second(s)"
    elif est.seconds >= 60:
        est_time = str(int(total.split(":")[1])) + " minute(s)", str(float(total.split(":")[2])) + " second(s)"
    else:
        est_time = str(float(total.split(":")[2])) + " second(s)"

    return jsonify({"prediction" : class_name[predicted_class], "confidence" : str(round(confidence[predicted_class].item() * 100, 2)) + "%", "estimated_time" : est_time, "more" : more})

if __name__ == "__main__":
    app.run()