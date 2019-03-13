#!/usr/bin/env python
#coding: utf-8


def data_layer(layer_name, output_name1, output_name2, phase, crop_size=128, scale=0.0078125):
    context = '\
layer {\n\
  name: "%s"\n\
  type: "Data"\n\
  top: "%s"\n\
  top: "%s"\n\
  include {\n\
    phase: %s\n\
  }\n\
  transform_param {\n\
    crop_size: %d\n\
    scale: %.7f\n\
    mean_file: "@model_dir/mean.binaryproto"\n\
    mirror: %s\n\
  }\n\
  data_param {\n\
    source: "@data_dir/%s_data_db/"\n\
    backend: LMDB\n\
    batch_size: %s\n\
  }\n\
}\n\
' % (layer_name, output_name1, output_name2, phase, crop_size, scale, \
     "true" if phase == "TRAIN" else "false", \
     "train" if phase == "TRAIN" else "validation", \
     "@train_batch_size" if phase == "TRAIN" else "@test_batch_size")
    return context


def conv_layer(layer_name, input_name, output_name, num_output, kernel_size=3, stride=1, pad=1, w_lr_mult=1, w_decay_mult=1, b_lr_mult=2, b_decay_mult=0, initializer="xavier"):
    initializer_context = '\
      type: "xavier"\n\
' if initializer == "xavier" else\
'\
      type: "gaussian"\n\
      std: 0.01\n\
'

    context = '\
layer {\n\
  name: "%s"\n\
  type: "Convolution"\n\
  bottom: "%s"\n\
  top: "%s"\n\
  param {\n\
    lr_mult: %d\n\
    decay_mult: %d\n\
  }\n\
  param {\n\
    lr_mult: %d\n\
    decay_mult: %d\n\
  }\n\
  convolution_param {\n\
    num_output: %d\n\
    kernel_size: %d\n\
    stride: %d\n\
    pad: %d\n\
    weight_filler {\n%s\
    }\n\
    bias_filler {\n\
      type: "constant"\n\
      value: 0\n\
    }\n\
  }\n\
}\n\
' % (layer_name, input_name, output_name, w_lr_mult, w_decay_mult, b_lr_mult, b_decay_mult, num_output, kernel_size, stride, pad, initializer_context)
    return context


def relu_layer(layer_name, input_name, output_name):
    context = '\
layer {\n\
  name: "%s"\n\
  type: "PReLU"\n\
  bottom: "%s"\n\
  top: "%s"\n\
}\n\
' % (layer_name, input_name, output_name)
    return context


def eltwise_layer(layer_name, input_name1, input_name2, output_name):
    context = '\
layer {\n\
  name: "%s"\n\
  type: "Eltwise"\n\
  bottom: "%s"\n\
  bottom: "%s"\n\
  top: "%s"\n\
  eltwise_param {\n\
    operation: SUM\n\
  }\n\
}\n\
' % (layer_name, input_name1, input_name2, output_name)
    return context


def fc_layer(layer_name, input_name, output_name, num_output, w_lr_mult=1, w_decay_mult=1, b_lr_mult=2, b_decay_mult=0):
    context = '\
layer {\n\
  name: "%s"\n\
  type: "InnerProduct"\n\
  bottom: "%s"\n\
  top: "%s"\n\
  param {\n\
    lr_mult: %d\n\
    decay_mult: %d\n\
  }\n\
  param {\n\
    lr_mult: %d\n\
    decay_mult: %d\n\
  }\n\
  inner_product_param {\n\
    num_output: %d\n\
    weight_filler {\n\
      type: "xavier"\n\
    }\n\
    bias_filler {\n\
      type: "constant"\n\
      value: 0\n\
    }\n\
  }\n\
}\n\
' % (layer_name, input_name, output_name, w_lr_mult, w_decay_mult, b_lr_mult, b_decay_mult, num_output)
    return context


def resnet(file_name="deploy.prototxt", layer_num=100):
    cell_num = 4
    if layer_num == 20:
        block_type = 1
        block_num = [1, 2, 4, 1]
        #block_num = [2, 2, 2, 2]
    elif layer_num == 34:
        block_type = 1
        block_num = [2, 4, 7, 2]
    elif layer_num == 50:
        block_type = 1
        #block_num = [2, 6, 13, 2]
        block_num = [3, 4, 13, 3]
    elif layer_num == 64:
        block_type = 1
        block_num = [3, 8, 16, 3]
    elif layer_num == 100:
        block_type = 1
        #block_num = [3, 12, 30, 3]
        block_num = [4, 12, 28, 4]
    else:
        block_type = 1
        # 12
        block_num = [1, 1, 1, 1]
    num_output = 32

    network = 'name: "resnet_%d"\n' % layer_num
    # data layer
    layer_name = "data"
    output_name1 = layer_name
    output_name2 = "label000"
    output_name = layer_name
    network += data_layer(layer_name, output_name1, output_name2, "TRAIN")
    network += data_layer(layer_name, output_name1, output_name2, "TEST")
    for i in xrange(cell_num):
        num_output *= 2
        # ceil
        # conv layer
        layer_name = "conv%d_%d" % (i+1, 1)
        input_name = output_name
        output_name = layer_name
        network += conv_layer(layer_name, input_name, output_name, num_output, stride=2, b_lr_mult=2, initializer="xavier")
        # relu layer
        layer_name = "relu%d_%d" % (i+1, 1)
        input_name = output_name
        output_name = output_name
        network += relu_layer(layer_name, input_name, output_name)
        # block
        for j in xrange(block_num[i]):
            # type1
            if block_type == 1:
                for k in xrange(2):
                    # conv layer
                    layer_name = "conv%d_%d" % (i+1, 2*j+k+2)
                    input_name = output_name
                    output_name = layer_name
                    network += conv_layer(layer_name, input_name, output_name, num_output, stride=1, b_lr_mult=0, initializer="gaussian")
                    # relu layer
                    layer_name = "relu%d_%d" % (i+1, 2*j+k+2)
                    input_name = output_name
                    output_name = output_name
                    network += relu_layer(layer_name, input_name, output_name)
            # type2
            elif block_type == 2:
                for k in xrange(3):
                    # conv layer
                    layer_name = "conv%d_%d" % (i+1, 2*j+k+2)
                    input_name = output_name
                    output_name = layer_name
                    network += conv_layer(layer_name, input_name, output_name, num_output if k==2 else num_output/2, \
                                                                               kernel_size=(3 if k==1 else 1), \
                                                                               pad=(1 if k==1 else 0), \
                                                                               stride=1, b_lr_mult=0, initializer="gaussian")
                    # relu layer
                    layer_name = "relu%d_%d" % (i+1, 2*j+k+2)
                    input_name = output_name
                    output_name = output_name
                    network += relu_layer(layer_name, input_name, output_name)
            else:
                print "error"
            
            # eltwise_layer
            layer_name = "res%d_%d" % (i+1, 2*j+3)
            input_name1 = "%s%d_%d" % ("conv" if j == 0 else "res", i+1, 2*j+1)
            input_name2 = "conv%d_%d" % (i+1, 2*j+3)
            output_name = layer_name
            network += eltwise_layer(layer_name, input_name1, input_name2, output_name)
    # fc layer
    layer_name = "fc%d" % (cell_num+1)
    input_name = output_name
    output_name = layer_name
    network += fc_layer(layer_name, input_name, output_name, num_output)

    # write network to file
    open(file_name, "wb").write(network)


if __name__ == "__main__":
    layer_num = 20
    resnet(file_name="resnet_%d.prototxt" % layer_num, layer_num=layer_num)

