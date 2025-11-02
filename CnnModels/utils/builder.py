import math
import torch
import torch.nn as nn
from utils.options import args
import utils.conv_type


class Builder(object):
    def __init__(self, conv_layer, bn_layer, first_layer=None, nm_layers=None):
        self.conv_layer = conv_layer
        self.bn_layer = bn_layer
        self.first_layer = first_layer or conv_layer
        self.nm_layers = nm_layers or set()  # 存储应该使用NMConv的层名称
        
        # 如果指定了nm_layers，确保NMConv类型可用
        self.nm_conv_layer = getattr(utils.conv_type, "NMConv") if hasattr(utils.conv_type, "NMConv") else conv_layer

    def conv(self, kernel_size, in_planes, out_planes, stride=1, first_layer=False, bias=False, layer_name=None):
        # 根据layer_name决定使用哪种卷积层
        use_nm_conv = False
        if layer_name and self.nm_layers:
            # 精确匹配
            if layer_name in self.nm_layers:
                use_nm_conv = True
                print(f"Using NMConv for layer: {layer_name}")
            else:
                # 前缀匹配，检查是否有任何前缀匹配
                for prefix in self.nm_layers:
                    if layer_name.startswith(prefix):
                        use_nm_conv = True
                        print(f"Using NMConv for layer: {layer_name}")
                        break
        
        if use_nm_conv:
            conv_layer = self.nm_conv_layer
        else:
            conv_layer = self.first_layer if first_layer else self.conv_layer

        if first_layer:
            print(f"==> Building first layer with {args.first_layer_type}")

        if kernel_size == 3:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=bias,
            )
        elif kernel_size == 1:
            conv = conv_layer(
                in_planes, out_planes, kernel_size=1, stride=stride, bias=bias
            )
        elif kernel_size == 5:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=5,
                stride=stride,
                padding=2,
                bias=bias,
            )
        elif kernel_size == 7:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=7,
                stride=stride,
                padding=3,
                bias=bias,
            )
        else:
            return None

        # self._init_conv(conv)

        return conv

    def conv2d(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        return self.conv_layer(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )

    def conv3x3(self, in_planes, out_planes, stride=1, first_layer=False, layer_name=None):
        """3x3 convolution with padding"""
        c = self.conv(3, in_planes, out_planes, stride=stride, first_layer=first_layer, layer_name=layer_name)
        return c

    def conv1x1(self, in_planes, out_planes, stride=1, first_layer=False, layer_name=None):
        """1x1 convolution with padding"""
        c = self.conv(1, in_planes, out_planes, stride=stride, first_layer=first_layer, layer_name=layer_name)
        return c

    def conv1x1_fc(self, in_planes, out_planes, stride=1, first_layer=False):
        """full connect layer"""
        c = self.conv(1, in_planes, out_planes, stride=stride, first_layer=first_layer, bias=True)
        return c

    def conv7x7(self, in_planes, out_planes, stride=1, first_layer=False):
        """7x7 convolution with padding"""
        c = self.conv(7, in_planes, out_planes, stride=stride, first_layer=first_layer)
        return c

    def conv5x5(self, in_planes, out_planes, stride=1, first_layer=False):
        """5x5 convolution with padding"""
        c = self.conv(5, in_planes, out_planes, stride=stride, first_layer=first_layer)
        return c

    def batchnorm(self, planes, last_bn=False, first_layer=False):
        return self.bn_layer(planes)

    def activation(self):
        if args.nonlinearity == "relu":
            return (lambda: nn.ReLU(inplace=True))()
        else:
            raise ValueError(f"{args.nonlinearity} is not an initialization option!")

    def _init_conv(self, conv):
        if args.init == "signed_constant":

            fan = nn.init._calculate_correct_fan(conv.weight, args.mode)
            if args.scale_fan:
                fan = fan * (1 - args.prune_rate)
            gain = nn.init.calculate_gain(args.nonlinearity)
            std = gain / math.sqrt(fan)
            conv.weight.data = conv.weight.data.sign() * std

        elif args.init == "unsigned_constant":

            fan = nn.init._calculate_correct_fan(conv.weight, args.mode)
            if args.scale_fan:
                fan = fan * (1 - args.prune_rate)

            gain = nn.init.calculate_gain(args.nonlinearity)
            std = gain / math.sqrt(fan)
            conv.weight.data = torch.ones_like(conv.weight.data) * std

        elif args.init == "kaiming_normal":

            if args.scale_fan:
                fan = nn.init._calculate_correct_fan(conv.weight, args.mode)
                fan = fan * (1 - args.prune_rate)
                gain = nn.init.calculate_gain(args.nonlinearity)
                std = gain / math.sqrt(fan)
                with torch.no_grad():
                    conv.weight.data.normal_(0, std)
            else:
                nn.init.kaiming_normal_(
                    conv.weight, mode=args.mode, nonlinearity=args.nonlinearity
                )

        elif args.init == "standard":
            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))
        else:
            raise ValueError(f"{args.init} is not an initialization option!")


def get_builder():
    # 获取默认卷积层类型
    default_conv_type = args.conv_type
    
    # 解析nm_layers参数，确定哪些层使用NMConv
    nm_layers = set()
    if hasattr(args, 'nm_layers') and args.nm_layers:
        nm_layers_list = args.nm_layers.split(',')
        for layer in nm_layers_list:
            nm_layers.add(layer.strip())
        print("==> NMConv Layers: {}".format(list(nm_layers)))
        
        # 如果指定了nm_layers，则使用DenseConv作为默认卷积层
        print("==> Using selective layer replacement")
        print("==> Default Conv Type: DenseConv (overriding global setting)")
        conv_layer = getattr(utils.conv_type, "DenseConv")
    else:
        # 没有指定nm_layers，使用全局设置
        print("==> Conv Type: {}".format(default_conv_type))
        conv_layer = getattr(utils.conv_type, default_conv_type)
    
    print("==> BN Type: {}".format(args.bn_type))
    bn_layer = getattr(utils.conv_type, args.bn_type)

    first_layer = None
    
    builder = Builder(conv_layer=conv_layer, bn_layer=bn_layer, first_layer=first_layer, nm_layers=nm_layers)

    return builder

