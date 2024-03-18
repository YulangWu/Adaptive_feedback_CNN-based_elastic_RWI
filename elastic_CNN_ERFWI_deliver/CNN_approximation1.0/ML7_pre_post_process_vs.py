from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import glob
import scipy as sp
import scipy.ndimage
import cmath

import glob
from random import shuffle
import random
import matplotlib.pyplot as plt


def taper(ntp,dir):
    #TAPER Summary of this function goes here
    #   Detailed explanation goes here
    tp = np.zeros(ntp)
    if dir == 1:
        for i in range(0,ntp):
                tp[i]=1-(np.cos((ntp-i)*np.pi/ntp)+1.)*0.5
    elif dir == 0:
        for i in range(0,ntp):
                tp[i]=(np.cos((ntp-i)*np.pi/ntp)+1.)*0.5
    else:
        raise("The direction of taper function is invalid!")
    return tp

def phase_shift(data,angle):
    data_shape = data.shape
    nt = data_shape[0]
    nx = data_shape[1]

    for i in range(nx):
        # plt.figure(3)
        # plt.plot(data[:,i])
        
        ## Fourier transform of real valued signal
        signalFFT = np.fft.rfft(data[:,i])
        signalPhase = np.angle(signalFFT)
        ## Phase Shift the signal +90 degrees
        
        newSignalFFT = signalFFT * cmath.rect( 1., np.pi*angle/180)

        ## Reverse Fourier transform
        newSignal = np.fft.irfft(newSignalFFT)
        data[:len(newSignal),i] = newSignal
        data[len(newSignal):,i] = 0

        # plt.plot(data[:,i])
        # plt.show()
    return data

def get_name(path):
    # get file name without directory and '.dat'
    fullname, _ = os.path.splitext(os.path.basename(path))
    return fullname

def get_list_files(input_dir,prefix):
    if input_dir is None or not os.path.exists(input_dir):
        raise Exception("input_dir does not exist")

    file_path = glob.glob(os.path.join(input_dir, prefix + "*.dat"))
    # print(file_path)
    # Count the number of files under the input directory:
    numFiles = len(file_path)
  
    file_list = []
    if len(file_path) == 0:
        raise Exception("input_dir contains no image files")

    return file_path


def should(iteration, freq):
    return freq > 0 and ((iteration + 1) % freq == 0)

def load_train_data(input_dir, prefix, nx, nz):
    file_list = get_list_files(input_dir, prefix)
    image_vz = np.zeros((len(file_list),nx*nz))
    image_vx = np.zeros((len(file_list),nx*nz))
    vp_init = np.zeros((len(file_list),nx*nz))
    vs_init = np.zeros((len(file_list),nx*nz))
    rho_init = np.zeros((len(file_list),nx*nz))
    vp_true = np.zeros((len(file_list),nx*nz))
    vs_true = np.zeros((len(file_list),nx*nz))
    rho_true = np.zeros((len(file_list),nx*nz))
    for i,filename in enumerate(file_list):
        print(" input: " + filename)
        data_set = np.loadtxt(filename)  
        len_data = len(data_set)

        image_vz[i,:] = data_set[0*int(len_data/8):1*int(len_data/8)] #picked data
        image_vx[i,:] = data_set[1*int(len_data/8):2*int(len_data/8)] #picked data
        vp_init[i,:] = data_set[2*int(len_data/8):3*int(len_data/8)] #picked data
        vs_init[i,:] = data_set[3*int(len_data/8):4*int(len_data/8)] #picked data
        rho_init[i,:] = data_set[4*int(len_data/8):5*int(len_data/8)] #picked data
        vp_true[i,:] = data_set[5*int(len_data/8):6*int(len_data/8)] #picked data
        vs_true[i,:] = data_set[6*int(len_data/8):7*int(len_data/8)] #picked data
        rho_true[i,:] = data_set[7*int(len_data/8):8*int(len_data/8)] #picked data


    data = {}
    data['image_vz'] = image_vz
    data['image_vx'] = image_vx
    data['vp_init'] = vp_init
    data['vs_init'] = vs_init
    data['rho_init'] = rho_init
    data['vp_true'] = vp_true
    data['vs_true'] = vs_true
    data['rho_true'] = rho_true

    return data

def load_data(input_dir, prefix, nx, nz):
    #     pair_data = load_stack_data(selected_files, stack_num, nx, nz)
    pair_data = load_train_data(input_dir, prefix, nx, nz)

    return pair_data

if __name__ == '__main__':
    
    from ML5_parse_vp import *
    print(a.input_dir)

    if os.path.exists(os.path.join(a.output_dir,'image_vz.npy')):
        print('load existing data')
        image_vz = np.load(os.path.join(a.output_dir,'image_vz.npy'))
        image_vx = np.load(os.path.join(a.output_dir,'image_vx.npy'))
        vp_init = np.load(os.path.join(a.output_dir,'vp_init.npy'))
        vs_init = np.load(os.path.join(a.output_dir,'vs_init.npy'))
        rho_init = np.load(os.path.join(a.output_dir,'rho_init.npy'))
        vp_true = np.load(os.path.join(a.output_dir,'vp_true.npy'))
        vs_true = np.load(os.path.join(a.output_dir,'vs_true.npy'))
        rho_true = np.load(os.path.join(a.output_dir,'rho_true.npy'))
    else:
        data_set = load_data(a.input_dir, 'CNN_train_dataset', a.nx, a.nz)
        print('save data')  
        image_vz = data_set['image_vz']
        image_vx = data_set['image_vx']
        vp_init = data_set['vp_init'] 
        vs_init = data_set['vs_init']  
        rho_init = data_set['rho_init']  
        vp_true = data_set['vp_true']  
        vs_true = data_set['vs_true']  
        rho_true = data_set['rho_true']  

        np.save(os.path.join(a.output_dir,'image_vz.npy'),image_vz)
        np.save(os.path.join(a.output_dir,'image_vx.npy'),image_vx)
        np.save(os.path.join(a.output_dir,'vp_init.npy'),vp_init)
        np.save(os.path.join(a.output_dir,'vs_init.npy'),vs_init)
        np.save(os.path.join(a.output_dir,'rho_init.npy'),rho_init)
        np.save(os.path.join(a.output_dir,'vp_true.npy'),vp_true)
        np.save(os.path.join(a.output_dir,'vs_true.npy'),vs_true)
        np.save(os.path.join(a.output_dir,'rho_true.npy'),rho_true)


    for i in range(len(image_vz)):
        plt.figure(101)
        plt.subplot(3,2,1)
        vp = image_vz[i,:].copy()
        vp.shape = a.nx,a.nz
        plt.imshow(np.transpose(vp),aspect='auto')
        plt.title('vz')

        plt.subplot(3,2,2)
        vp = image_vx[i,:].copy()
        vp.shape = a.nx,a.nz
        plt.imshow(np.transpose(vp),aspect='auto')
        plt.title('vx')

        plt.subplot(3,2,3)
        vp = vp_init[i,:].copy()
        vp.shape = a.nx,a.nz
        plt.imshow(np.transpose(vp),aspect='auto')
        plt.title('smooth vp')

        plt.subplot(3,2,4)
        vp = rho_init[i,:].copy()
        vp.shape = a.nx,a.nz
        plt.imshow(np.transpose(vp),aspect='auto')
        plt.title('smooth rho')

        plt.subplot(3,2,5)
        vp = vp_true[i,:].copy()
        vp.shape = a.nx,a.nz
        plt.imshow(np.transpose(vp),aspect='auto')
        plt.title('true vp')

        plt.subplot(3,2,6)
        vp = rho_true[i,:].copy()
        vp.shape = a.nx,a.nz
        plt.imshow(np.transpose(vp),aspect='auto')
        plt.title('true rho')

        plt.show()

