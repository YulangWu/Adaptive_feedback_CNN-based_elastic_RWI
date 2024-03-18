% sh_num_smooth_iteration = 400;
% sh_filter_size = 3;
% sh_water_depth = 13;nx=256;

nx=sh_nx;
nz=sh_nz;
iter = num2str(0);
num_smooth_iteration = sh_num_smooth_iteration;
filter_size = sh_filter_size;
water_depth = sh_water_depth;
%input model
%vp = dlmread([iter 'th_iteration_FWI_model.dat']);
vp = dlmread(['Sigsbee_vp350x1.dat']);
vp = reshape(vp,nz,nx);

vs = dlmread(['Sigsbee_vs350x1.dat']);
vs = reshape(vs,nz,nx);

rho = dlmread(['Sigsbee_rho350x1.dat']);
rho = reshape(rho,nz,nx);

% % % % % vs = zeros(nz,nx);
% rho = 0.31*sqrt(sqrt(vp*1000));

low = 2;
high = 4;

figure(1);
subplot(2,3,1);imagesc(vp);caxis([1.5 4.5]);colormap('jet');
subplot(2,3,2);imagesc(vs);caxis([0 2.3]);colormap('jet');
subplot(2,3,3);imagesc(rho);caxis([1.5 2.5]);colormap('jet');

title('True model');
%output correct model:
vp = reshape(vp,1,nz*nx);
vs = reshape(vs,1,nz*nx);
rho = reshape(rho,1,nz*nx);

fid=fopen([iter 'th_true_vp.dat'],'wt');
fprintf(fid,'%17.8f',vp);
fclose(fid);

fid=fopen([iter 'th_true_vs.dat'],'wt');
fprintf(fid,'%17.8f',vs);
fclose(fid);

fid=fopen([iter 'th_true_rho.dat'],'wt');
fprintf(fid,'%17.8f',rho);
fclose(fid);

% ========================================
% create smooth vp model using slowness!
% ========================================
vp = reshape(vp,nz,nx);
vp_smooth = vp;
for i = 1:num_smooth_iteration
    if i >= num_smooth_iteration-10
        vp_smooth(1:water_depth,:) = vp(1:water_depth,:);
    end
    vp_smooth = imfilter(vp_smooth, fspecial('gaussian',filter_size),'replicate','same');
end
subplot(2,3,4);imagesc(vp_smooth);caxis([1.5 4.5]);colormap('jet');

vp_smooth=reshape(vp_smooth,1,nz*nx);
fid=fopen([iter 'th_mig_vp.dat'],'wt');
fprintf(fid,'%17.8f',vp_smooth);
fclose(fid);

% ========================================
% create smooth vs model using slowness!
% ========================================
vs = reshape(vs,nz,nx);
vs_smooth = vs;
for i = 1:num_smooth_iteration
    if i >= num_smooth_iteration-10
        vs_smooth(1:water_depth,:) = vs(1:water_depth,:);
    end
    vs_smooth = imfilter(vs_smooth, fspecial('gaussian',filter_size),'replicate','same');
end
subplot(2,3,5);imagesc(vs_smooth);caxis([0 2.3]);colormap('jet');

vs_smooth=reshape(vs_smooth,1,nz*nx);
fid=fopen([iter 'th_mig_vs.dat'],'wt');
fprintf(fid,'%17.8f',vs_smooth);
fclose(fid);



% ========================================
% create smooth rho model using slowness!
% ========================================
% % 1. smoothing by filtering the true density model
rho = reshape(rho,nz,nx);
rho_smooth = rho;
for i = 1:num_smooth_iteration
    if i >= num_smooth_iteration-10
        rho_smooth(1:water_depth,:) = rho(1:water_depth,:);
    end
    rho_smooth = imfilter(rho_smooth, fspecial('gaussian',filter_size),'replicate','same');
end
subplot(2,3,6);imagesc(rho_smooth);caxis([1.5 2.5]);colormap('jet');

rho_smooth=reshape(rho_smooth,1,nz*nx);

% 2. smoothing by Garner's equation
% rho_smooth = 0.31*sqrt(sqrt(vp_smooth*1000));

fid=fopen([iter 'th_mig_rho.dat'],'wt');
fprintf(fid,'%17.8f',rho_smooth);
fclose(fid);

% vp_smooth = reshape(vp_smooth,nz,nx);
% rho_smooth = reshape(rho_smooth,nz,nx);
% figure(2);plot(vp(:,100),'r');hold on;plot(vp_smooth(:,100));
% figure(3);plot(rho(:,100),'r');hold on;plot(rho_smooth(:,100));
% 
% clear all
% close all
% clc
