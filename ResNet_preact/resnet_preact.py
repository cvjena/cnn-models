"""Generation script for residual networks."""

__author__      = "Marcel Simon"
__copyright__   = "BSD 2-clause"


import caffe
from caffe import layers as L, params as P

def bn_conv(bottom, nout, ks = 3, stride=1, pad = 0, learn = True, bn_inplace=False, is_first = False):
    if not is_first:
        bn = L.BatchNorm(bottom, param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)], in_place=bn_inplace)
        scale = L.Scale(bn,scale_param=dict(bias_term=True), param=[dict(lr_mult=1, decay_mult=1),dict(lr_mult=2, decay_mult=1)],in_place=True)
        relu = L.ReLU(scale, in_place=True)
        conv = L.Convolution(relu, kernel_size=ks, stride=stride,
            num_output=nout, pad=pad, bias_term=False, param = [dict(lr_mult=1, decay_mult=1)],
            weight_filler=dict(type="msra"),
            bias_filler=dict(type="constant",value=0))
        return bn, scale, relu, conv
    else:
        conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
            num_output=nout, pad=pad, bias_term=False, param = [dict(lr_mult=1, decay_mult=1)],
            weight_filler=dict(type="msra"),
            bias_filler=dict(type="constant",value=0))
        return conv,conv,conv,conv


def residual_standard_unit(n, nout, s, newdepth = False, is_first = False):
    """
    This creates the "standard unit" shown on the left side of Figure 5.
    """
    net = n.__dict__['tops']

    bottom = net[list(net.keys())[-1]] #find the last layer in netspec
    stride = 2 if newdepth else 1

    net[s + 'bn1'], net[s + 'scale1'], net[s + 'relu1'], net[s + 'conv1'] = bn_conv(bottom, ks = 3, stride = stride, nout = nout, pad = 1, bn_inplace=False, is_first=is_first)
    net[s + 'bn2'], net[s + 'scale2'], net[s + 'relu2'], net[s + 'conv2'] = bn_conv(net[s + 'conv1'], ks = 3, stride = 1, nout = nout, pad = 1)

    if newdepth:
        net[s + 'conv_expand'] = L.Convolution(net[s + 'relu1'], kernel_size=1, stride=stride,
            num_output=nout, pad=0, bias_term=False, param = [dict(lr_mult=1, decay_mult=1)],
            weight_filler=dict(type="msra"),
            bias_filler=dict(type="constant",value=0))
        net[s + 'sum'] = L.Eltwise(net[s + 'conv2'], net[s + 'conv_expand'])
    else:
        net[s + 'sum'] = L.Eltwise(net[s + 'conv2'], bottom)


def residual_bottleneck_unit(n, nout, s, newdepth = False, is_first = False):
    """
    This creates the "standard unit" shown on the left side of Figure 5.
    """
    net = n.__dict__['tops']

    bottom = net[list(net.keys())[-1]] #find the last layer in netspec
    stride = 2 if newdepth else 1

    net[s + 'bn1'], net[s + 'scale1'], net[s + 'relu1'], net[s + 'conv1'] = bn_conv(bottom, ks = 1, stride = 1, nout = nout, pad = 0, bn_inplace=False, is_first=is_first)
    net[s + 'bn2'], net[s + 'scale2'], net[s + 'relu2'], net[s + 'conv2'] = bn_conv(net[s + 'conv1'], ks = 3, stride = stride, nout = nout, pad = 1)
    net[s + 'bn3'], net[s + 'scale3'], net[s + 'relu3'], net[s + 'conv3'] = bn_conv(net[s + 'conv2'], ks = 1, stride = 1, nout = 4*nout, pad = 0)

    if newdepth or is_first:
        net[s + 'conv_expand'] = L.Convolution(net[s + 'relu1'], kernel_size=1, stride=stride,
            num_output=4*nout, pad=0, bias_term=False, param = [dict(lr_mult=1, decay_mult=1)],
            weight_filler=dict(type="msra"),
            bias_filler=dict(type="constant",value=0))
        net[s + 'sum'] = L.Eltwise(net[s + 'conv3'], net[s + 'conv_expand'])
    else:
        net[s + 'sum'] = L.Eltwise(net[s + 'conv3'], bottom)



def residual_net(total_depth, num_classes = 1000, acclayer = True):
    """
    Generates nets from "Deep Residual Learning for Image Recognition". Nets follow architectures outlined in Table 1.
    """
    # figure out network structure
    net_defs = {
        10:([1, 1, 1, 1], "standard"),
        18:([2, 2, 2, 2], "standard"),
        26:([3, 3, 3, 3], "standard"),
        #34:([4, 4, 4, 4], "standard"),
        34:([3, 4, 6, 3], "standard"),
        50:([3, 4, 6, 3], "bottleneck"),
        101:([3, 4, 23, 3], "bottleneck"),
        152:([3, 8, 36, 3], "bottleneck"),
    }
    assert total_depth in net_defs.keys(), "net of depth:{} not defined".format(total_depth)

    nunits_list, unit_type = net_defs[total_depth] # nunits_list a list of integers indicating the number of layers in each depth.
    base_filter_size = 64
    nouts = [base_filter_size* 2**i for i in range(4)]

    # setup the first couple of layers
    n = caffe.NetSpec()
    net = n.__dict__['tops']

    n.data, n.label = L.ImageData(batch_size=16,
                source="/home/simon/Datasets/ilsvrc12/train_images.txt",
                shuffle=True,new_height=256,new_width=256, ntop=2,
                transform_param=dict(crop_size=224, mirror=True))

    # The data mean
    n.data_bn = L.BatchNorm(n.data, param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)], in_place=False)
    n.data_scale = L.Scale(n.data_bn,scale_param=dict(bias_term=True), param=[dict(lr_mult=1, decay_mult=1),dict(lr_mult=2, decay_mult=1)], in_place=True)

    n.conv1 = L.Convolution(n.data_scale, kernel_size=7, stride=2,
            num_output=base_filter_size, pad=3, param = [dict(lr_mult=1, decay_mult=1),dict(lr_mult=2, decay_mult=1)],
            weight_filler=dict(type="msra",variance_norm=1),
            bias_filler=dict(type="constant",value=0))
    n.conv1_bn = L.BatchNorm(n.conv1, param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)], in_place=False)
    n.conv1_scale = L.Scale(n.conv1_bn,scale_param=dict(bias_term=True), param=[dict(lr_mult=1, decay_mult=1),dict(lr_mult=2, decay_mult=1)],in_place=True)
    n.conv1_relu = L.ReLU(n.conv1_scale, in_place=True)
    n.conv1_pool = L.Pooling(n.conv1_relu, stride = 2, kernel_size = 3)

    # make the convolutional body
    for nout, nunits in zip(nouts, nunits_list): # for each depth and nunits
        for unit in range(1, nunits + 1): # for each unit. Enumerate from 1.
            s = 'layer_' + str(nout) + '_' + str(unit) + '_' # layer name prefix
            if unit_type == "standard":
                residual_standard_unit(n, nout, s, newdepth = unit is 1 and nout > base_filter_size, is_first=unit is 1 and nout is base_filter_size)
            else:
                residual_bottleneck_unit(n, nout, s, newdepth = unit is 1 and nout > base_filter_size, is_first=unit is 1 and nout is base_filter_size)
            

    # add the end layers
    n.last_bn = L.BatchNorm(net[list(net.keys())[-1]], param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)], in_place=False)
    n.last_scale = L.Scale(n.last_bn,scale_param=dict(bias_term=True), param=[dict(lr_mult=1, decay_mult=1),dict(lr_mult=2, decay_mult=1)],in_place=True)
    n.last_relu = L.ReLU(n.last_scale, in_place=True)
    n.global_pool = L.Pooling(n.last_relu, pooling_param = dict(pool = 1, global_pooling = True))
    n.score = L.InnerProduct(n.global_pool, num_output = num_classes,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=1)])
    n.loss = L.SoftmaxWithLoss(n.score, n.label)
    if acclayer:
        n.accuracy = L.Accuracy(n.score, n.label, include=dict(phase=1))

    return n

