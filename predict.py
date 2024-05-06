import torch
from skimage import io

image_size = 100

model = torch.jit.load('models/04.pt')
model.eval()

with torch.no_grad():
    xss = torch.FloatTensor(1, image_size, image_size).cuda()
    xss[0] = torch.FloatTensor(io.imread('test/record.png', as_gray=True))
    output = model(xss)

    print(
        f'Weights:    {"  ".join([f'{i:<6}' for i in range(10)])}\n'
        f'            {"  ".join([f'{round(x, 2):<6}' for x in output.cpu().numpy().tolist()[0]])}\n\n'
        f'Prediction: {output.argmax(axis=1).cpu().numpy().item()}'
    )
