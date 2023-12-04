import torch
from torch import nn
from torchinfo import summary
from inf721_dataset import *

# Model definition
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


#dw
class DepthWiseConvolution(nn.Sequential):
    def __init__(self, in_fts, stride = 1):
        super(DepthWiseConvolution,self).__init__(
            nn.Conv2d(in_fts,in_fts,kernel_size=(3,3),stride=stride,padding=(1,1), groups=in_fts, bias=False),
            nn.BatchNorm2d(in_fts),
            nn.ReLU6(inplace=True))


#pw
class PointWiseConvolution(nn.Sequential):
    def __init__(self,in_fts,out_fts):
        super(PointWiseConvolution,self).__init__(
            nn.Conv2d(in_fts,out_fts,kernel_size=(1,1),bias=False),
            nn.BatchNorm2d(out_fts),
            nn.ReLU6(inplace=True))


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )



class Bottleneck(nn.Module):
    def __init__(self,inp, oup, stride, expand_ratio, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.stride = stride

        hidden_dim = int(round(inp*expand_ratio))
        layers = []
        self.use_res_connect = self.stride == 1 and inp == oup

        #pw
        if expand_ratio != 1:
            layers.append(PointWiseConvolution(inp,hidden_dim))

        #dw
        layers.extend([
            DepthWiseConvolution(hidden_dim,stride),
            #pw-linear
            nn.Conv2d(hidden_dim,oup,1,1,0,bias=False),
            nn.BatchNorm2d(oup)])

        self.conv = nn.Sequential(*layers)


    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class OurObjectDetectionNet(nn.Module):
    def __init__(self, bottleneckLayerDetail, inp = 3, num_classes=len(classes), width_mult=1.0, round_nearest=8):
        super(OurObjectDetectionNet, self).__init__()

        self.out = None

        bloco = Bottleneck
        inverted_residual_setting = bottleneckLayerDetail

        input_channel = 32
        last_channel = 1280

        input_channel = _make_divisible(input_channel*width_mult,round_nearest)
        self.last_channel = _make_divisible(last_channel*width_mult,round_nearest)

        #first layer
        features = [ConvBNReLU(inp, input_channel, stride=2)]

        #build layers
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c*width_mult,round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(bloco(input_channel,output_channel,stride = stride,expand_ratio=t))
                input_channel = output_channel


        #last layer
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))

        #make sequential
        self.features = nn.Sequential(*features)

        #classificador
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.last_channel, num_classes))

    def __forward_impl(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x,1).reshape(x.shape[0],-1)
        x = self.classifier(x)

        return x

    def forward(self, x):
        x = self.__forward_impl(x)
        return x


# Model declaration
bottleneckLayerDetail = [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
]

our_model = OurObjectDetectionNet(bottleneckLayerDetail)
summary(our_model, (1, 3, 224, 224), col_names=("input_size", "output_size",
                                                      "num_params", "kernel_size",
                                                      "mult_adds"))

# Saving/loading model for resume training
def save_model_for_resume(model, optim, path):
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict()
            }, path)

def load_model_for_resume(path, model, optim):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
