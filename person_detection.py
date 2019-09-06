import torch
import torch.optim as optim
import my_model
from PIL import Image
import variables as var
import functions as func


labels = var.labels
model = my_model.Net()
model.eval()
optimizer = optim.SGD(model.parameters(), lr=var.lr, momentum=var.momentum)

checkpoint = torch.load('person_detection.pth')
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

transform = var.transform

image_name = ['IMG1.jpg', 'IMG2.jpg', 'IMG3.jpg']
for img in image_name:
    im = Image.open(img)
    # im.show()

    im_t = transform(im)
    func.imshow(im_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 'Unknown')

    outputs = model(torch.unsqueeze(im_t, 0))
    _, predicted = torch.max(outputs, 1)
    print('Image :', img, ' : ', labels[predicted])
