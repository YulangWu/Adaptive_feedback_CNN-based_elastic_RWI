#! /bin/sh



#parameters ===========================================================================
nz=256
nx=256
num_threads=4 #number of threads to run RTM images
iteration=0    #The first iteration of the CNN-RFWI starts at 0 (th iteration)
iter=50 #The last iteration of the CNN-RFWI
num_samples=4 #The number of samples for CNN training
vel_dir='models/' #The velocity directory to store the predicted and smooth models
fortran_dir='fortran1.0/'
matlab_dir='matlab1.0/'
CNN_dir='CNN_approximation1.0/'
CNN_result_dir="CNN_results/"
CNN_max_epochs=1600
store_weights_freq=1600
CNN_last_epochs=`expr $CNN_max_epochs - 1`
CNN_max_epochs_vs=2800
store_weights_freq_vs=2800
CNN_last_epochs_vs=`expr $CNN_max_epochs_vs - 1`
postfix=""
sleep_time=60 #unit is second
sh_num_smooth_iteration=800
sh_filter_size=3
sh_water_depth=13 #Marmousi230z6500x
#remember to manually set total time step for Fortran
#======================================================================================
<<comment
comment

#0.1 prepare the true and starting velocity models 
#    in Marmousi folder to 1st iteration
echo "0.1 prepare the true and starting velocity models"
cd $matlab_dir$vel_dir
  matlab_filename=${iteration}"th_mig_"
  matlab -nodesktop -nosplash -noFigureWindows -r \
  "sh_num_smooth_iteration=$sh_num_smooth_iteration,\
  sh_filter_size=$sh_filter_size,\
  sh_nx=$nx,sh_nz=$nz,\
  sh_water_depth=$sh_water_depth;\
  prepare_corresponding_smooth_starting_mig_model;quit"
cd ../../

#0.2 prepare the velocity model for real RTM
#    in Marmousi folder
echo "0.2 prepare the velocity model for real RTM"

cd matlab1.0
  matlab_filename=${iteration}"th_mig_"  
  matlab -nodesktop -nosplash -noFigureWindows -r \
  "sh_nx=$nx,sh_nz=$nz,sh_vel_dir='$vel_dir';prepare_CNN_real_test_data_FORTRAN;quit"
cd ..

#0.3 To obtain real RTM images 
echo "0.3 To obtain real RTM images"

cd $fortran_dir
ifort -O2 migration0.f90 -o migration0.exe 
./migration0.exe

# 0.4 To prepare the CNN real data
echo "0.4 To prepare the CNN real data"
matlab -nodesktop -nosplash -noFigureWindows -r \
"sh_nx=$nx,sh_nz=$nz;preprocess_rtm_real_FORTRAN;quit"
rm -r given*
rm -r out*
mv real_dataset ../CNN_approximation1.0
cd ..



while [ ${iteration} -le $iter ]
do
  echo "This is the "${iteration}" th iteration 0th  step>>>>>>>>>>>>"
  if [ ${iteration} -gt 0 ]
  then
    cd $matlab_dir$vel_dir
    echo "enter into directory "$matlab_dir$vel_dir
    matlab -nodesktop -nosplash -noFigureWindows -r \
    "sh_iter='${iteration}',\
    sh_nx=$nx,sh_nz=$nz;\
    prepare_corresponding_smooth_prediction_model;\
    quit" #> 'temp.txt'
    cd ../../
  fi

  echo "This is the "${iteration}" th iteration 1th  step>>>>>>>>>>>>"
  #1. To do clustering and prepare training velocity models in fortran1.0 for nextth  step
    cd matlab1.0
      matlab_filename=${iteration}"th_mig_"

      matlab -nodesktop -nosplash -noFigureWindows -r \
      "sh_name='$matlab_filename',\
      sh_nx=$nx,sh_nz=$nz,\
      sh_vel_dir='$vel_dir',\
      num_samples=$num_samples,num_threads=$num_threads;\
      Binary_kmeans_model_morphology_leaves_only_FORTRAN;\
      num_samples=$num_samples,num_threads=$num_threads,\
      sh_num_smooth_iteration=$sh_num_smooth_iteration,\
      sh_filter_size=$sh_filter_size,\
      sh_nx=$nx,sh_nz=$nz,\
      sh_vel_dir='$vel_dir',\
      sh_water_depth=$sh_water_depth;\
      prepare_CNN_train_data_FORTRAN_vp_vs_rho;\
      quit" #> ${iteration}'matlab_res.txt'

      mv velocity velocity"_"$iteration

    cd ..
    echo "1. Finish training velocity models at "${iteration}"th iteration"

  echo "This is the "${iteration}" th iteration 2th  step>>>>>>>>>>>>"
  # #2. To obtain training RTM images 
    cd $fortran_dir
    ifort -O2 migration1.f90 -o migration1.exe 
    ifort -O2 migration2.f90 -o migration2.exe 
    ifort -O2 migration3.f90 -o migration3.exe 
    ifort -O2 migration4.f90 -o migration4.exe 

    nohup ./migration1.exe &
    nohup ./migration2.exe &
    nohup ./migration3.exe &
    nohup ./migration4.exe &

  echo "This is the "${iteration}" th iteration 3th  step>>>>>>>>>>>>"
  # 3. check the status of the four threads
    jobs -l > RTM_run.txt
    i=0
    while [ -s RTM_run.txt ]
    do
      jobs -l > RTM_run.txt
      sleep $sleep_time
    done
    echo "2 and 3. Finish RTM at "${iteration}"th iteration"
    rm RTM_run.txt

  echo "This is the "${iteration}" th iteration 4th  step>>>>>>>>>>>>"
  # 4. To prepare the CNN training data
    matlab -nodesktop -nosplash -noFigureWindows -r \
    "num_samples=$num_samples,num_threads=$num_threads,\
    sh_nx=$nx,sh_nz=$nz;\
    preprocess_rtm_train_FORTRAN;quit" #> 'temp.txt'
    rm -r given*
    rm -r out*
    cp -r train_dataset ../CNN_approximation1.0
    mv train_dataset ${iteration}"train_dataset"
    cd ..

  echo "This is the "${iteration}" th iteration 5th  step>>>>>>>>>>>>"
  # # 5. CNN training
    echo "5. CNN training at "${iteration}"th iteration"
    cd $CNN_dir 
    # #GPU 1 (slower)
    # nohup python ML7_vpFWI.py --max_epochs $CNN_max_epochs \
    # --store_weights_freq $store_weights_freq --CNN_num $CNN_last_epochs &
    # #GPU 0 (faster)
    # nohup python ML7_rhoFWI.py --max_epochs $CNN_max_epochs \
    # --store_weights_freq $store_weights_freq --CNN_num $CNN_last_epochs &

    #GPU 0 (faster)
    rm nohup*
    nohup python ML7_vsFWI.py --max_epochs $CNN_max_epochs_vs \
    --store_weights_freq $store_weights_freq_vs --CNN_num $CNN_last_epochs_vs \
     --nz $nz --nx $nx &

    #GPU 1 (slower)
    python ML7_vpFWI.py --max_epochs $CNN_max_epochs \
    --store_weights_freq $store_weights_freq --CNN_num $CNN_last_epochs \
     --nz $nz --nx $nx
    

    python ML7_rhoFWI.py --max_epochs $CNN_max_epochs \
    --store_weights_freq $store_weights_freq --CNN_num $CNN_last_epochs \
     --nz $nz --nx $nx

  echo "This is the "${iteration}" th iteration 6th  step>>>>>>>>>>>>"
  # 6. check the status of the four threads
    jobs -l > CNN_run.txt
    i=0
    while [ -s CNN_run.txt ]
    do
    jobs -l > CNN_run.txt
    sleep $sleep_time
    done
    echo "6. Finish CNN training at "${iteration}"th iteration"
    rm CNN_run.txt

  echo "This is the "${iteration}" th iteration 7th  step>>>>>>>>>>>>"
  # # 7. CNN prediction and export
    python ML7_vpFWI.py --mode real --CNN_num $CNN_last_epochs --nz $nz --nx $nx  #> 'temp.txt'
    python ML7_vpFWI.py --mode export --CNN_num $CNN_last_epochs --nz $nz --nx $nx   #> 'temp.txt'
    python ML7_vsFWI.py --mode real --CNN_num $CNN_last_epochs_vs --nz $nz --nx $nx  #> 'temp.txt'
    python ML7_vsFWI.py --mode export --CNN_num $CNN_last_epochs_vs --nz $nz --nx $nx   #> 'temp.txt'
    python ML7_rhoFWI.py --mode real --CNN_num $CNN_last_epochs --nz $nz --nx $nx  #> 'temp.txt'
    python ML7_rhoFWI.py --mode export --CNN_num $CNN_last_epochs --nz $nz --nx $nx  #> 'temp.txt'

  echo "This is the "${iteration}" th iteration 8th  step>>>>>>>>>>>>"
  # 8. clean the directory for next iteration
    echo ${iteration}"th clean the directory for next isteration" 
    mkdir ${iteration}${CNN_result_dir}
    rm -r train_dataset
    mv real_outputs ${iteration}${CNN_result_dir}
    mv train_outputs ${iteration}${CNN_result_dir}
    mv "CNN_weights_vp"$CNN_last_epochs ${iteration}${CNN_result_dir}
    mv "CNN_weights_vs"$CNN_last_epochs_vs ${iteration}${CNN_result_dir}
    mv "CNN_weights_rho"$CNN_last_epochs ${iteration}${CNN_result_dir}
    rm -r CNN*
    rm train_dataset
    
    cp -r ${iteration}${CNN_result_dir}"/CNN_weights_vp"$CNN_last_epochs .
    cp -r ${iteration}${CNN_result_dir}"/CNN_weights_vs"$CNN_last_epochs_vs .
    cp -r ${iteration}${CNN_result_dir}"/CNN_weights_rho"$CNN_last_epochs .

  echo "This is the "${iteration}" th iteration 9th  step>>>>>>>>>>>>"
  # 9. create the CNN-predicted velocity for the next iteration
    input_vp_filename=${iteration}${CNN_result_dir}"real_outputs/real0_vp.dat";
    input_vs_filename=${iteration}${CNN_result_dir}"real_outputs/real0_vs.dat";
    input_rho_filename=${iteration}${CNN_result_dir}"real_outputs/real0_rho.dat";
    next_iter=`expr ${iteration} + 1`
    output_vp_filename="../"$matlab_dir$vel_dir${next_iter}"th_true_vp"$postfix
    output_vs_filename="../"$matlab_dir$vel_dir${next_iter}"th_true_vs"$postfix
    output_rho_filename="../"$matlab_dir$vel_dir${next_iter}"th_true_rho"$postfix
    
    #notice: Undefined function or variable 'c-shell variable name' because the clear all is not commented!!!
    matlab -nodesktop -nosplash -noFigureWindows -r "sh_input_vp_filename='$input_vp_filename',\
    sh_nx=$nx,sh_nz=$nz,\
    sh_input_vs_filename='$input_vs_filename',sh_input_rho_filename='$input_rho_filename',\
    sh_output_vp_filename='$output_vp_filename',sh_output_vs_filename='$output_vs_filename',\
    sh_output_rho_filename='$output_rho_filename';output_CNN_predicted_models;quit"
  cd ..

  let iteration++
  echo "9. The "${iteration}"th iteration will be run now>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
done 




























