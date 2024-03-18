from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import collections


parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="train", choices=["train", "test", "export","real"])
parser.add_argument("--input_dir", default="_dataset", help="where to put seismic model related to data")
parser.add_argument("--output_dir", default="_outputs", help="where to put guessed initial model")
parser.add_argument("--parameter_dir", default="CNN_weights", help="where to put variables in tensorflow")
parser.add_argument("--seed", type=int)
parser.add_argument("--global_iteration_number", type=int, default=1, help="number of iteration of fwi<->gan")
parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, default=400, help="number of training epochs")
parser.add_argument("--display_freq", type=int, default=100, help="display progress every display_freq steps")
parser.add_argument("--output_freq", type=int, default=1, help="Store the output models every # epoch")
parser.add_argument("--store_weights_freq", type=int, default=400, help="Store the trained weights every # total steps")
parser.add_argument("--ngf", type=int, default=32, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=32, help="number of discriminator filters in first conv layer")
parser.add_argument("--batch", type=int, default=1, help="starting shot x-direction position from the left")
parser.add_argument("--nz", type=int, default=256, help="scale images to this size before cropping to 256x256")
parser.add_argument("--nx", type=int, default=256, help="scale images to this size before cropping to 256x256")
parser.add_argument("--nt", type=int, default=2048, help="scale images to this size before cropping to 256x256")
parser.add_argument("--stack_num", type=int, default=20, help="scale images to this size before cropping to 256x256")
parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--gan_weight", type=float, default=0, help="gan_weight")
parser.add_argument("--l1_weight", type=float, default=1, help="l1_weight")
parser.add_argument("--figs", type=bool, default=False, help="l1_weight")
parser.add_argument("--CNN_num",default='0', help="reload saved CNN weights")
parser.add_argument("--max_image", type=float, default=100000, help="maximum image")
parser.add_argument("--max_reflectivity", type=float, default=0.2, help="maximum reflectivity")
parser.add_argument("--max_vp", type=float, default=3000, help="maximum P-wave velocity")
parser.add_argument("--mean_vp", type=float, default=2000, help="mean P-wave velocity")
parser.add_argument("--max_vs", type=float, default=2000, help="maximum S-wave velocity")
parser.add_argument("--mean_vs", type=float, default=2000, help="mean S-wave velocity")
parser.add_argument("--max_rho", type=float, default=1000, help="maximum density")
parser.add_argument("--mean_rho", type=float, default=2000, help="mean density")

# export options
a = parser.parse_args()

if a.mode == 'train':
    a.input_dir = 'train' + a.input_dir
    a.output_dir = 'train' + a.output_dir

if a.mode == 'export':
    a.input_dir = 'train' + a.input_dir
    a.output_dir = 'train' + a.output_dir
    a.figs = True
    a.max_epochs = 1
    a.lr = 0.0

if a.mode == 'test':
    a.input_dir = a.mode + a.input_dir
    a.output_dir = a.mode + a.output_dir
    a.max_epochs = 1
    a.lr = 0.0
    a.figs = True

if a.mode == 'real':
    a.input_dir = a.mode + a.input_dir
    a.output_dir = a.mode + a.output_dir
    a.max_epochs = 1
    a.lr = 0.0
    a.figs = True

if a.input_dir is None or not os.path.exists(a.input_dir):
    raise Exception("input_dir does not exist")

if not os.path.exists(a.output_dir):
    try:
        os.makedirs(a.output_dir)
    except IOError:
        print("Output directory exists")


Model = collections.namedtuple("Model", "predict_real, predict_fake, discrim_loss, gen_loss_GAN, gen_loss_L1, "
                                        "inputs, targets, outputs, train, discrim_grads_and_vars, gen_grads_and_vars")

#for DispersionElimination.py use only
model_generator = collections.namedtuple("model_generator", "L2_loss, inputs, inputs2, targets, outputs, train, gen_grads_and_vars")