from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import time
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import subprocess

# self-defined functions
from ML7_parse_vs import *
from ML7_pre_post_process_vs import *
from ML7_buildmodel_vs import *

import shutil

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    a.parameter_dir = a.parameter_dir + '_vs'
    if not os.path.exists(a.parameter_dir):
        os.makedirs(a.parameter_dir)
    
    if a.mode != 'train':
        a.parameter_dir = a.parameter_dir + (a.CNN_num)
        print(a.parameter_dir)

    if tf.__version__.split('.')[0] != "1":
        raise Exception("Tensorflow version 1 required")

    if a.seed is None:
        a.seed = random.randint(0, 2 ** 31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    for k, v in a._get_kwargs():
        print(k, " = ", v)

    ##############################################################################
    # 1. Input train/test data:
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
        data_set = load_data(a.input_dir, 'CNN_'+ a.mode + '_dataset', a.nx, a.nz)
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

    image_vz /= a.max_image
    image_vx /= a.max_image

    vs_init = (vs_init - a.mean_vs) / a.max_vs
    vs_true = (vs_true - a.mean_vs) / a.max_vs


    # for i in range(len(image_vz)):
    #     plt.figure(101)
    #     plt.subplot(3,2,1)
    #     vp = image_vz[i,:].copy()
    #     vp.shape = a.nx,a.nz
    #     plt.imshow(np.transpose(vp),aspect='auto')
    #     plt.title('vz')
    #     plt.clim(-1,1)

    #     plt.subplot(3,2,2)
    #     vp = image_vx[i,:].copy()
    #     vp.shape = a.nx,a.nz
    #     plt.imshow(np.transpose(vp),aspect='auto')
    #     plt.title('vx')
    #     plt.clim(-1,1)

    #     plt.subplot(3,2,3)
    #     vp = vp_init[i,:].copy()
    #     vp.shape = a.nx,a.nz
    #     plt.imshow(np.transpose(vp),aspect='auto')
    #     plt.title('smooth vp')
    #     plt.clim(-1,1)

    #     plt.subplot(3,2,4)
    #     vp = rho_init[i,:].copy()
    #     vp.shape = a.nx,a.nz
    #     plt.imshow(np.transpose(vp),aspect='auto')
    #     plt.title('smooth rho')
    #     plt.clim(-1,1)

    #     plt.subplot(3,2,5)
    #     vp = vp_true[i,:].copy()
    #     vp.shape = a.nx,a.nz
    #     plt.imshow(np.transpose(vp),aspect='auto')
    #     plt.title('true vp')
    #     plt.clim(-1,1)

    #     plt.subplot(3,2,6)
    #     vp = rho_true[i,:].copy()
    #     vp.shape = a.nx,a.nz
    #     plt.imshow(np.transpose(vp),aspect='auto')
    #     plt.title('true rho')
    #     plt.clim(-1,1)

    #     plt.show()

    # prepare an index array for shuffling (shuffling index not data every time)
    index_arr = [i for i in range(len(image_vz))]
    print('Number of stacked data: {}'.format(len(image_vz))) 



    ##############################################################################
    # 2. Build CNN

    inputs = tf.placeholder(tf.float32, shape=(a.nx, a.nz)) #vz
    inputs2 = tf.placeholder(tf.float32, shape=(a.nx, a.nz)) #vp
    inputs3 = tf.placeholder(tf.float32, shape=(a.nx, a.nz)) #vx
    targets = tf.placeholder(tf.float32, shape=(a.nx, a.nz))

    #=============================================================================
    model_generator = create_model(inputs, inputs2, inputs3, targets, a.nz, a.nx, a.batch, a.lr, a.beta1, a.ngf)
    #=============================================================================


    ##############################################################################
    # 3. Execute CNN graph

    total_iterations = len(image_vz) * a.max_epochs #a.batch
    # loss_curve contains all L1 loss for each snapshot of each wavefield
    loss_curve = []

    # Define the epoch
    if a.mode == 'train' or a.mode == 'export':
        max_steps = 2 ** 32
        if a.max_epochs is not None:
            max_steps = total_iterations
            print("max_steps %d" % max_steps)
    else:
        max_steps = 1
        print("max_steps %d" % max_steps)

    # Run Tensorflow in the scope of tf.Session
    saver = tf.train.Saver(max_to_keep=1)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        # Initialize all variables in tensorflow
        sess.run(tf.global_variables_initializer())

        # Compute the total number of the variables in tensorflow
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v))
                                             for v in tf.trainable_variables()])
        print("parameter_count =", sess.run(parameter_count))

        # Reload the training results
        #if a.mode == 'test':
        if a.mode == 'train':
            ckpt = tf.train.get_checkpoint_state(a.parameter_dir + (a.CNN_num))
        else:
            ckpt = tf.train.get_checkpoint_state(a.parameter_dir)
            
        if ckpt and ckpt.model_checkpoint_path:
            print("loading model from checkpoint")
            saver.restore(sess, ckpt.model_checkpoint_path)

        # remove the directory of CNN weights saved at the previous iteration     
        if a.mode == "train":
            try:
                shutil.rmtree(a.parameter_dir + (a.CNN_num))
            except:
                print("No directory found")

        # calculating the time
        t_start = time.time()
        t_used = 0.0

        # 1st loop: (epoch loop)
        nitr = -1  # total iteration number
        for epoch in range(a.max_epochs):

            if a.mode == 'train' and a.max_epochs > 1:
                # shuffle the list of index every epoch
                shuffle(index_arr)

            # 2nd loop : each stacked data
            for index in index_arr:
                nitr += 1
                
                #  Run TensorFlow session:
                fetches = {"train": model_generator.train, "L2_loss": model_generator.L2_loss,
                            "outputs": model_generator.outputs}
                # fetches["inputs"] = model_generator.inputs
                # fetches["inputs2"] = model_generator.inputs2
                # fetches["inputs3"] = model_generator.inputs3
                # fetches["inputs4"] = model_generator.inputs4
                # fetches["inputs5"] = model_generator.inputs5
                # fetches["targets1"] = model_generator.targets1
                # fetches["targets2"] = model_generator.targets2
                # fetches["targets3"] = model_generator.targets3

                # Without this, CNN weights are not fully stored
                fetches["gen_grads_and_vars"] = model_generator.gen_grads_and_vars

                input_image_vz = image_vz[[index_arr[index]],:]
                input_image_vx = image_vx[[index_arr[index]],:]
                input_vs_init = vs_init[[index_arr[index]],:]
                input_vs_true = vs_true[[index_arr[index]],:]


                input_image_vz.shape = a.nx,a.nz
                input_image_vx.shape = a.nx,a.nz
                input_vs_init.shape = a.nx,a.nz
                input_vs_true.shape = a.nx,a.nz


                if a.mode == 'train' or a.mode == 'export':
                    results = sess.run(fetches,
                                    feed_dict={inputs: input_image_vz, inputs2: input_vs_init, inputs3: input_image_vx, targets: input_vs_true})
                else:
                    zero_data = np.zeros((a.nx,a.nz))
                    results = sess.run(fetches,
                                    feed_dict={inputs: input_image_vz, inputs2: input_vs_init, inputs3: input_image_vx, targets: zero_data})


                L2_loss = results["L2_loss"]
                loss_curve.append(L2_loss)

                if a.figs and (a.max_epochs == 1 or (epoch + 1) % a.display_freq == 0):
                    ###############################################################
                    #                          plot
                    ###############################################################
                    if index % int(len(input_image_vz)/int(len(input_image_vz))) == 0:
                        output_res = np.zeros((5,a.nx,a.nz))

                        output_res[0,:,:] = input_image_vz[:,:]*a.max_image 
                        output_res[1,:,:] = input_vs_init[:,:]*a.max_vs + a.mean_vs
                        output_res[2,:,:] = input_vs_true[:,:]*a.max_vs + a.mean_vs

                        output_res[3,:,:] = results['outputs'][-1][0,:,:,0]*a.max_vs + a.mean_vs
                        output_res[4,:,:] = input_image_vx[:,:]*a.max_image 
                        # plt.subplot(2,2,1)
                        # plt.imshow(np.transpose(output_res[2,:,:]),aspect='auto')
                        # # plt.set_cmap('seismic')
                        # plt.clim(1.5,4.5)
                        # plt.title('initial model')

                        # plt.subplot(2,2,2)
                        # plt.imshow(np.transpose(target_true_model[:,:]*a.max_vel + a.mean_vel),aspect='auto')
                        # # plt.set_cmap('seismic')
                        # plt.clim(1.5,4.5)
                        # plt.title('target model')

                        # plt.subplot(2,2,3)
                        # plt.imshow(np.transpose(output_res[4,:,:]),aspect='auto')
                        # # plt.set_cmap('seismic')
                        # plt.clim(1.5,4.5)
                        # plt.title('output model')

                        # plt.subplot(2,2,4)
                        # plt.imshow(np.transpose(output_res[4,:,:] - target_true_model[:,:]*a.max_vel - a.mean_vel),aspect='auto')
                        # plt.set_cmap('seismic')
                        # plt.clim(-.5,.5)
                        # plt.title('residual model' + str(index_arr[index]))

                        # plt.show()

                        output_res.shape = -1, 1
                        filename = os.path.join(a.output_dir, a.mode + str(index_arr[index]) + '_vs.dat')
                        np.savetxt(filename, output_res, fmt="%1.8f")

                if index_arr[index] == 0 and (epoch + 1) % a.display_freq == 0:
                    t_end = time.time()
                    t_used = t_end - t_start
                    t_period = t_used / (nitr+1) # average time per data
                    
                    t_remain = t_period * (total_iterations - nitr)
                    print('The running index = {}'.format(index_arr[index]))
                    print('epoch: %3.0f; loss = %17.14f \n (TIME min) used = %3.2f, period = %3.2f, remain = %3.2f' %
                          (epoch, loss_curve[-1], t_used / 60, t_period / 60, t_remain / 60))   #unit is minute

            # Store the CNN weights to the files every 'store_weights_freq' epoch
            if should(epoch, a.store_weights_freq) and a.mode =='train':
                if os.path.exists(a.parameter_dir):
                    print("saving training model at " + str(epoch) + "epoch")
                    saver.save(sess, a.parameter_dir + '/model.ckpt')
                    os.makedirs(a.parameter_dir + str(epoch))

                    subprocess.check_call("mv " + os.path.join(a.parameter_dir,"*") + " " + a.parameter_dir + str(epoch),
                                          shell=True)

            loss_filename = os.path.join(a.output_dir, 'L2_loss_' + a.mode + '_vs.dat')
            if os.path.exists(loss_filename):
                os.remove(loss_filename)
            np.savetxt(loss_filename, np.array(loss_curve), fmt="%17.8f")

        

main()

