% clear all
% pack
% close all
% clc

nx=sh_nx;
nz=sh_nz;
low = 1.5; high = 4.5;num = 1;
% name = 'vs';low = .5; high = 2.5; num = 2;
% name = 'rho';low = 1.8; high = 2.4; num = 3;

file_title = '' %'_80_20'

iter = sh_iter;

%%% vp model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vp = dlmread([iter 'th_true_' ['vp' file_title] '.dat']);
vp = reshape(vp,nz,nx);


figure(1);imagesc(vp);
title('True model');caxis([low high]);
%output correct model:
vp = reshape(vp,1,nz*nx);

fid=fopen([iter 'th_true_' ['vp' file_title] '.dat'],'wt');
fprintf(fid,'%17.8f',vp);
fclose(fid);

vp = reshape(vp,nz,nx);
vp_smooth = vp;
num_smooth_iteration = 1;
for i = 1:num_smooth_iteration
    vp_smooth = imfilter(vp_smooth, fspecial('gaussian',3),'replicate','same');
end


vp_smooth=reshape(vp_smooth,1,nz*nx);
fid=fopen([iter 'th_mig_' ['vp' file_title] '.dat'],'wt');
fprintf(fid,'%17.8f',vp_smooth);
fclose(fid);


%%% vs model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vs = dlmread([iter 'th_true_' ['vs' file_title] '.dat']);
vs = reshape(vs,nz,nx);


figure(2);imagesc(vs);
title('True model');caxis([low high]);
%output correct model:
vs = reshape(vs,1,nz*nx);

fid=fopen([iter 'th_true_' ['vs' file_title] '.dat'],'wt');
fprintf(fid,'%17.8f',vs);
fclose(fid);

vs = reshape(vs,nz,nx);
vs_smooth = vs;
num_smooth_iteration = 1;
for i = 1:num_smooth_iteration
    vs_smooth = imfilter(vs_smooth, fspecial('gaussian',3),'replicate','same');
end


vs_smooth=reshape(vs_smooth,1,nz*nx);
fid=fopen([iter 'th_mig_' ['vs' file_title] '.dat'],'wt');
fprintf(fid,'%17.8f',vs_smooth);
fclose(fid);


















%%% rho model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rho = dlmread([iter 'th_true_' ['rho' file_title] '.dat']);
rho = reshape(rho,nz,nx);


figure(3);imagesc(rho);
title('True model');caxis([low high]);
%output correct model:
rho = reshape(rho,1,nz*nx);

fid=fopen([iter 'th_true_' ['rho' file_title] '.dat'],'wt');
fprintf(fid,'%17.8f',rho);
fclose(fid);

rho = reshape(rho,nz,nx);
rho_smooth = rho;
num_smooth_iteration = 1;
for i = 1:num_smooth_iteration
    rho_smooth = imfilter(rho_smooth, fspecial('gaussian',3),'replicate','same');
end


rho_smooth=reshape(rho_smooth,1,nz*nx);
fid=fopen([iter 'th_mig_' ['rho' file_title] '.dat'],'wt');
fprintf(fid,'%17.8f',rho_smooth);
fclose(fid);