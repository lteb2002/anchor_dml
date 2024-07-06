import torch
import torchvision.models as models
from torchvision import transforms
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.vgg import VGG16_Weights
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_vgg():
    model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    return model


def encode_img(vgg_model, img):
    return_layers = {
        'avgpool': 'avgpool',
    }
    # return_layers = {'28': 'out_layer17'}
    # model_with_multuple_layer = IntermediateLayerGetter(model.features, return_layers=return_layers)
    # y = model_with_multuple_layer(x)['out_layer17']
    mid_getter = MidGetter(vgg_model, return_layers=return_layers, keep_output=False)
    y, model_output = mid_getter(img)
    y = y['avgpool']
    y = torch.flatten(y, 1)
    # y = vgg_model.classifier[0](y)
    # y = vgg_model.classifier[1](y)
    # y = vgg_model.classifier[2](y)
    # y = vgg_model.classifier[3](y)
    y = vgg_model.classifier(y)
    return y


def build_tensor_as_ln(y, label):
    y = torch.round(y, decimals=4)
    s = ",".join(str(round(x, 4)) for x in y.squeeze().tolist())
    # print(s)
    return s + "," + label + "\n"


def encode_image_as_lnstr(vgg_model, img, label):
    y = encode_img(vgg_model, img)
    # print(y, y.shape)
    y = torch.round(y, decimals=4)
    s = ",".join(str(round(x, 4)) for x in y.squeeze().tolist())
    # print(s)
    return s + "," + label
