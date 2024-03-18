format short
%------------------ colorbar setting----------------------------
Ncolor=64;
lenwhite=0;
indexcolor=zeros(Ncolor*3/2-lenwhite/2,1);
for i=1:Ncolor*1.5-lenwhite/2
    indexcolor(i)=i/(Ncolor*1.5-lenwhite/2);
end
mycolor=zeros(Ncolor*3,3);
mycolor(1:Ncolor*2,1)=1;
mycolor(1+Ncolor:Ncolor*3,3)=1;
mycolor(Ncolor*1.5-lenwhite/2:Ncolor*1.5+lenwhite/2,2)=1;
mycolor(1:Ncolor*1.5-lenwhite/2,2)=indexcolor;
mycolor(1:Ncolor*1.5-lenwhite/2,3)=indexcolor;
mycolor(1+Ncolor*1.5+lenwhite/2:Ncolor*3,1)=flipud(indexcolor);
mycolor(1+Ncolor*1.5+lenwhite/2:Ncolor*3,2)=flipud(indexcolor);
mycolor=flipud(mycolor);
cvalue = 0.005;
%------------------ colorbar setting----------------------------

%---------------------------------------------------------
%  0 parameter setup
%---------------------------------------------------------
%input data info
nx = sh_nx;
nz = sh_nz;
num_smooth_iteration = sh_num_smooth_iteration;
filter_size = sh_filter_size;
water_depth = sh_water_depth;

%mode of data: for training or for test
mode = 'train' % 'test'
num_node = num_threads; %8 nodes, so evenly allocate velocity models to 8 folders
%shuffle the list of files
input_directory = 'velocity';
output_directory = '../fortran1.0/given_models';
for i = 1 : num_node
%     system(['rm -r ' output_directory num2str(i)]);
    system(['mkdir ' output_directory num2str(i)]);
end

Files_vp=dir([input_directory '/' 'new_' 'vp' 'model*.dat']); %file name
Files_vs=dir([input_directory '/' 'new_' 'vs' 'model*.dat']); %file name
Files_rho=dir([input_directory '/' 'new_' 'rho' 'model*.dat']); %file name

% %input starting model as initial model for all true model as default
% directory = 'right_BPmodel'; %'right_Sigsbee2A';
% filename = 'mig_BPbenchmark'; %'mig_10Sigsbee2A256x768'; %'true_Sigsbee2A256x768';
% smooth_model = dlmread([directory '/' filename '.dat']);
% smooth_model = reshape(smooth_model,1,nz*nx);
    
for k=1:length(Files_vp)
    FileNames_vp=Files_vp(k).name;
    FileNames_vs=Files_vs(k).name;
    FileNames_rho=Files_rho(k).name;

    len = length(FileNames_vp);
    name = ['train' FileNames_vp(4:len-4) '.dat'];
    
    
    if exist(name, 'file') ~= 0 %if file exists do not repeatedly create
%         disp([num2str(k) ' EXISTS===' FileNames_vp])
        continue;
    else
        disp([num2str(k) ' ' FileNames_vp])
        disp(name)
    end
    
    
    true_model = dlmread([input_directory '/' FileNames_vp]);
    true_model = reshape(true_model,nz,nx);
    true_model(1:water_depth,:) = 1.5;
    
    true_model_vs = dlmread([input_directory '/' FileNames_vs]);
    true_model_vs = reshape(true_model_vs,nz,nx);
    true_model_vs(1:water_depth,:) = 0.0;
    
    true_model_rho = dlmread([input_directory '/' FileNames_rho]);
    true_model_rho = reshape(true_model_rho,nz,nx);
    true_model_rho(1:water_depth,:) = 1.01;
    
    % create correponding smooth model if not use starting model as smooth
    % BE SURE that the smoothing paramters are the same as starting model!!
    smooth_model = true_model;
    for i = 1:num_smooth_iteration
        if i >= num_smooth_iteration-10
            smooth_model(1:water_depth,:) = true_model(1:water_depth,:);
        end
        smooth_model = imfilter(smooth_model, fspecial('gaussian',filter_size),'replicate','same');
    end
    smooth_model = reshape(smooth_model,1,nz*nx);
    
    smooth_model_vs = true_model_vs;
    for i = 1:num_smooth_iteration
        if i >= num_smooth_iteration-10
            smooth_model_vs(1:water_depth,:) = true_model_vs(1:water_depth,:);
        end
        smooth_model_vs = imfilter(smooth_model_vs, fspecial('gaussian',filter_size),'replicate','same');
    end
    smooth_model_vs = reshape(smooth_model_vs,1,nz*nx);

    % create correponding smooth model if not use starting model as smooth
    % BE SURE that the smoothing paramters are the same as starting model!!
    % 1. smoothing by filtering the true density model
    smooth_model_rho = true_model_rho;
    for i = 1:num_smooth_iteration
        if i >= num_smooth_iteration-10
            smooth_model_rho(1:water_depth,:) = true_model_rho(1:water_depth,:);
        end
        smooth_model_rho = imfilter(smooth_model_rho, fspecial('gaussian',filter_size),'replicate','same');
    end
    smooth_model_rho = reshape(smooth_model_rho,1,nz*nx);

    % 2. smoothing by Gardner's equation
    % smooth_model_rho = 0.31*sqrt(sqrt(smooth_model*1000));        
    




    
    true_model = reshape(true_model,1,nz*nx);
    true_model_vs = reshape(true_model_vs,1,nz*nx);
    true_model_rho = reshape(true_model_rho,1,nz*nx);
    % decimal output
%     disp(name)
    
    %vp
    if k/num_node ~= floor(k/num_node)
        fid=fopen([output_directory num2str(mod(k,num_node)+1) '/' num2str(floor(k/num_node+1)) 'true' 'vp' '.dat'],'wt');
    else
        fid=fopen([output_directory num2str(mod(k,num_node)+1) '/' num2str(floor(k/num_node)) 'true' 'vp' '.dat'],'wt');
    end
    
    disp(num2str(floor(k/num_node+1)))
    fprintf(fid,'%17.8f',true_model);
    fclose(fid);

    if k/num_node ~= floor(k/num_node)
        fid=fopen([output_directory num2str(mod(k,num_node)+1) '/' num2str(floor(k/num_node+1)) 'mig' 'vp' '.dat'],'wt');
    else
        fid=fopen([output_directory num2str(mod(k,num_node)+1) '/' num2str(floor(k/num_node)) 'mig' 'vp' '.dat'],'wt');
    end
    fprintf(fid,'%17.8f',smooth_model);
    fclose(fid);



    %vs
    if k/num_node ~= floor(k/num_node)
        fid=fopen([output_directory num2str(mod(k,num_node)+1) '/' num2str(floor(k/num_node+1)) 'true' 'vs' '.dat'],'wt');
    else
        fid=fopen([output_directory num2str(mod(k,num_node)+1) '/' num2str(floor(k/num_node)) 'true' 'vs' '.dat'],'wt');
    end
    
    disp(num2str(floor(k/num_node+1)))
    fprintf(fid,'%17.8f',true_model_vs);
    fclose(fid);

    if k/num_node ~= floor(k/num_node)
        fid=fopen([output_directory num2str(mod(k,num_node)+1) '/' num2str(floor(k/num_node+1)) 'mig' 'vs' '.dat'],'wt');
    else
        fid=fopen([output_directory num2str(mod(k,num_node)+1) '/' num2str(floor(k/num_node)) 'mig' 'vs' '.dat'],'wt');
    end
    fprintf(fid,'%17.8f',smooth_model_vs);
    fclose(fid);


    %rho
    if k/num_node ~= floor(k/num_node)
        fid=fopen([output_directory num2str(mod(k,num_node)+1) '/' num2str(floor(k/num_node+1)) 'true' 'rho' '.dat'],'wt');
    else
        fid=fopen([output_directory num2str(mod(k,num_node)+1) '/' num2str(floor(k/num_node)) 'true' 'rho' '.dat'],'wt');
    end
    
    disp(num2str(floor(k/num_node+1)))
    fprintf(fid,'%17.8f',true_model_rho);
    fclose(fid);

    if k/num_node ~= floor(k/num_node)
        fid=fopen([output_directory num2str(mod(k,num_node)+1) '/' num2str(floor(k/num_node+1)) 'mig' 'rho' '.dat'],'wt');
    else
        fid=fopen([output_directory num2str(mod(k,num_node)+1) '/' num2str(floor(k/num_node)) 'mig' 'rho' '.dat'],'wt');
    end
    fprintf(fid,'%17.8f',smooth_model_rho);
    fclose(fid);


    
    % binary output
%     fid = fopen(name,'w');
%     fwrite(fid,train_data,'float32');
%     fclose(fid);
end

clear all
close all
clc
% %---------------------------------------------------------
% %           figure 2   plot Pdata
% %---------------------------------------------------------
% Files_vp=dir([output_directory '/' 'train_model*.dat']); %file name
% for k=1:length(Files_vp)
%     FileNames_vp=Files_vp(k).name;
%     model = dlmread([output_directory '/' FileNames_vp]);
    
%     model_true = model(1:nz*nx);
%     model_true = reshape(model_true,nz,nx);
    
%     model_smooth = model(nz*nx+1:nz*nx*2);
%     model_smooth = reshape(model_smooth,nz,nx);
%     subplot(1,3,1)
%     imagesc(model_true);caxis([1.5 4.5]);
%     subplot(1,3,2)
%     imagesc(model_smooth);caxis([1.5 4.5]);
%     subplot(1,3,3)
%     reflectivity = (model_true - model_smooth)./model_smooth;
%     imagesc(reflectivity);caxis([-0.005 0.005])
%     drawnow;
%     pause(0.5);
% end

