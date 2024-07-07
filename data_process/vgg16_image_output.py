import torch
import torchvision.models as models
from torchvision import transforms
from torchvision.models.vgg import VGG16_Weights
import data_process.vgg_funs as vgg_funs
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = vgg_funs.build_vgg().to(device)

output_file = 'F:\\datasets\\fruit360_2.csv'
input_path = 'F:\\datasets\\fruit360\\'

head = ",".join(['x' + str(x + 1) for x in range(1000)]) + ",label\n"
with open(output_file, 'w') as file:
    file.write(head)

dirs = os.listdir(input_path)
for tdir in dirs:
    label = tdir
    lns = []
    files = os.listdir(input_path + tdir)
    print(label, ":", len(files))
    for ff in files:
        fp = input_path + tdir + "\\" + ff
        # print(fp)
        image1 = Image.open(fp)
        scaler = transforms.Resize((100, 100))
        to_tensor = transforms.ToTensor()
        x = to_tensor(scaler(image1)).unsqueeze(0).to(device)
        # print(x.shape)
        y = vgg_funs.encode_img(model, x).cpu()
        s = vgg_funs.build_tensor_as_ln(y, label)
        # s = vgg_funs.encode_image_as_lnstr(model, x, label) + '\n'
        lns.append(s)
    with open(output_file, 'a') as file:
        file.writelines(lns)
    # break

# x = torch.rand(5,3,512,512)
image1 = Image.open(r"F:\datasets\fruit360\Apple Braeburn\23_100.jpg")
scaler = transforms.Resize((224, 224))
to_tensor = transforms.ToTensor()
x = to_tensor(scaler(image1)).unsqueeze(0).to(device)
print(x.shape)

# y = vgg_funs.encode_img(model,x)
# print(y,y.shape)
s = vgg_funs.encode_image_as_lnstr(model, x, "test")
print(s)
