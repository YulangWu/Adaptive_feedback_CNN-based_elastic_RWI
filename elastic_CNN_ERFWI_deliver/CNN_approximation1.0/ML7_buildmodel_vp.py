from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ML7_parse_vp import Model
from ML7_parse_vp import model_generator
from util import *  # Import the common CNN functions

def create_generator_skip(generator_inputs, outputs_channels, ngf):
    filter_size = 3
    layers = []
    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = conv(generator_inputs, ngf, 1, filter_size)
        layers.append(output)

    layer_specs = [
        (ngf * 2, 2),  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        (ngf * 4, 2),  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        (ngf * 8, 2),  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        (ngf * 16, 1),  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        (ngf * 8, 1),  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        (ngf * 8, 1),  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        (ngf * 8, 1),  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        # (ngf * 8,1)  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]
    for encoder_layer, (out_channels, stride) in enumerate(layer_specs):
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(rectified, out_channels, stride, filter_size)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (ngf * 8, 1),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (ngf * 8, 1),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (ngf * 16, 1),  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        (ngf * 8, 1),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (ngf * 8, 2),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (ngf * 4, 2),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (ngf * 2, 2),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        # (ngf, 1),  # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, stride) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input_hidden = layers[-1]
            else:
                input_hidden = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input_hidden)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = deconv(rectified, out_channels, stride, filter_size)
            output = batchnorm(output)

            #if dropout > 0.0:
            #    output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input_hidden = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input_hidden)
        output = deconv(rectified, outputs_channels, 1, filter_size)
        output = tf.tanh(output)
        layers.append(output)

    for i in range(len(layers)):
        print(i, layers[i])
    return layers



def create_discriminator(input, ndf):
    n_layers = 7
    layers = []
    filter_size = 3

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = conv(input, ndf*8, 2, filter_size=filter_size)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = ndf * min(2 ** (i + 1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = conv(layers[-1], out_channels=out_channels, stride=stride, filter_size=filter_size)
            normalized = batchnorm(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = conv(rectified, out_channels=1, stride=1, filter_size=filter_size)
        output = tf.sigmoid(convolved)
        layers.append(output)

    for i in range(len(layers)):
        print(i, layers[i])

    return layers

def create_model(inputs, inputs2, inputs3, targets, nz, nx, batch, lr, beta1, ngf):
    inputs = tf.reshape(inputs, [batch, nx, nz, 1])
    inputs2 = tf.reshape(inputs2, [batch, nx, nz, 1])
    inputs3 = tf.reshape(inputs3, [batch, nx, nz, 1])
    targets = tf.reshape(targets, [batch, nx, nz, 1])
    inputs_combine = tf.concat([inputs,inputs2,inputs3],axis=3)

    with tf.variable_scope("generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        # same size CNN: outputs = create_generator(inputs, out_channels, ngf)
        outputs = create_generator_skip(inputs_combine, out_channels, ngf) # U-net:
        #outputs = create_generator_res(inputs, out_channels, ngf)

    with tf.name_scope("generator_loss"):
        L2_loss = tf.reduce_mean(tf.abs(outputs[-1] - targets)*tf.abs(outputs[-1] - targets))

    with tf.name_scope("generator_train"):
        gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        gen_optim = tf.train.AdamOptimizer(lr, beta1)
        gen_grads_and_vars = gen_optim.compute_gradients(L2_loss, var_list=gen_tvars)
        gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([L2_loss])
    #update_losses = L2_loss

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    return model_generator(
        inputs=inputs,
        inputs2=inputs2,
        targets=targets,
        outputs=outputs,
        L2_loss= L2_loss, #ema.average(L2_loss), #
        gen_grads_and_vars=gen_grads_and_vars,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )


#
# def gan_create_model(inputs, targets, nt, nx, compression, lr, beta1, ngf, ndf, gan_weight, l1_weight):
#     EPS = 1e-12
#     inputs = tf.reshape(inputs, [1, nt, nx, 1])
#     targets = tf.reshape(targets, [1, nt*compression, nx*compression, 1])
#
#     with tf.variable_scope("generator") as scope:
#         out_channels = int(targets.get_shape()[-1])
#         outputs = create_generator(inputs, out_channels, ngf)
#
#
#     # create two copies of discriminator, one for real pairs and one for fake pairs
#     # they share the same underlying variables
#     with tf.name_scope("real_discriminator"):
#         with tf.variable_scope("discriminator"):
#             # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
#             predict_real = create_discriminator(targets, ndf)
#
#     with tf.name_scope("fake_discriminator"):
#         with tf.variable_scope("discriminator", reuse=True):
#             # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
#             predict_fake = create_discriminator(outputs[-1], ndf)
#
#     with tf.name_scope("discriminator_loss"):
#         # minimizing -tf.log will try to get inputs to 1
#         # predict_real => 1
#         # predict_fake => 0
#         discrim_loss = tf.reduce_mean(-(tf.log(predict_real[-1] + EPS) + tf.log(1 - predict_fake[-1] + EPS)))
#
#     with tf.name_scope("discriminator_train"):
#         discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
#         discrim_optim = tf.train.AdamOptimizer(lr, beta1)
#         discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
#         discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)
#
#     with tf.name_scope("generator_loss"):
#         # predict_fake => 1
#         # abs(targets - outputs) => 0
#         gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake[-1] + EPS))
#         #gen_loss_GAN = tf.reduce_mean(tf.log(1 - predict_fake + EPS))
#         gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs[-1]))
#         gen_loss = gen_loss_GAN * gan_weight + gen_loss_L1 * l1_weight
#
#     with tf.name_scope("generator_train"):
#         with tf.control_dependencies([discrim_train]):
#             gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
#             gen_optim = tf.train.AdamOptimizer(lr, beta1)
#             gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
#             gen_train = gen_optim.apply_gradients(gen_grads_and_vars)
#     #     print(gen_grads_and_vars)
#
#     ema = tf.train.ExponentialMovingAverage(decay=0.99)
#     update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1]) #gen_loss_GAN, gen_loss_L1])
#
#     global_step = tf.contrib.framework.get_or_create_global_step()
#     incr_global_step = tf.assign(global_step, global_step + 1)
#
#     return Model(
#         predict_real=predict_real,
#         predict_fake=predict_fake,
#         discrim_loss=ema.average(discrim_loss),
#         gen_loss_GAN=ema.average(gen_loss_GAN),
#         gen_loss_L1=ema.average(gen_loss_L1),
#         inputs=inputs,
#         outputs=outputs,
#         targets=targets,
#         discrim_grads_and_vars=discrim_grads_and_vars,
#         gen_grads_and_vars=gen_grads_and_vars,
#         train=tf.group(update_losses, incr_global_step, gen_train, discrim_train)
#     )
