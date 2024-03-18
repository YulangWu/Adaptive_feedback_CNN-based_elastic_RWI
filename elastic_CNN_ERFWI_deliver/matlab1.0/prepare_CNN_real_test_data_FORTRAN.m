
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
input_directory = sh_vel_dir;
output_directory = '../fortran1.0/given_models';
system(['mkdir ' output_directory num2str(0)]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
type = 'vp'


true_model = dlmread([input_directory  '0th_true_' type '.dat']);
smooth_model = dlmread([input_directory  '0th_mig_' type '.dat']);
smooth_model = reshape(smooth_model,1,nz*nx);
true_model = reshape(true_model,1,nz*nx);

name = input_directory;
disp(name)

fid=fopen([output_directory num2str(0) '/' num2str(0) 'true' type '.dat'],'wt');
disp(num2str(0))
fprintf(fid,'%17.8f',true_model);
fclose(fid);
    
fid=fopen([output_directory num2str(0) '/' num2str(0) 'mig' type '.dat'],'wt');
disp(num2str(0))
fprintf(fid,'%17.8f',smooth_model);
fclose(fid);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
type = 'vs'

true_model = dlmread([input_directory  '0th_true_' type '.dat']);
smooth_model = dlmread([input_directory  '0th_mig_' type '.dat']);
smooth_model = reshape(smooth_model,1,nz*nx);
true_model = reshape(true_model,1,nz*nx);

name = input_directory;
disp(name)

fid=fopen([output_directory num2str(0) '/' num2str(0) 'true' type '.dat'],'wt');
disp(num2str(0))
fprintf(fid,'%17.8f',true_model);
fclose(fid);
    
fid=fopen([output_directory num2str(0) '/' num2str(0) 'mig' type '.dat'],'wt');
disp(num2str(0))
fprintf(fid,'%17.8f',smooth_model);
fclose(fid);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
type = 'rho'

true_model = dlmread([input_directory  '0th_true_' type '.dat']);
smooth_model = dlmread([input_directory  '0th_mig_' type '.dat']);
smooth_model = reshape(smooth_model,1,nz*nx);
true_model = reshape(true_model,1,nz*nx);

name = input_directory;
disp(name)

fid=fopen([output_directory num2str(0) '/' num2str(0) 'true' type '.dat'],'wt');
disp(num2str(0))
fprintf(fid,'%17.8f',true_model);
fclose(fid);
    
fid=fopen([output_directory num2str(0) '/' num2str(0) 'mig' type '.dat'],'wt');
disp(num2str(0))
fprintf(fid,'%17.8f',smooth_model);
fclose(fid);

% 
% %---------------------------------------------------------
% %           figure 2   plot Pdata
% %---------------------------------------------------------
% Files=dir([output_directory '/' '*.dat']); %file name
% for k=1:length(Files)
%     FileNames=Files(k).name;
%     model = dlmread([output_directory '/' FileNames]);
%     
%     model_true = model(1:nz*nx);
%     model_true = reshape(model_true,nz,nx);
%     
%     model_smooth = model(nz*nx+1:nz*nx*2);
%     model_smooth = reshape(model_smooth,nz,nx);
%     subplot(1,3,1)
%     imagesc(model_true);caxis([1.5 4.5]);
%     subplot(1,3,2)
%     imagesc(model_smooth);caxis([1.5 4.5]);
%     subplot(1,3,3)
%     reflectivity = (model_true - model_smooth)./model_smooth;
%     imagesc(reflectivity);
%     drawnow;
%     pause(0.5);
% end

