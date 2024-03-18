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
nx=sh_nx;
nz=sh_nz;

%shuffle the list of files
mode = 'real'
input_directory = ['output'];%pysit result is here
output_directory = [mode '_dataset'];
system(['mkdir ' output_directory]);

name = ['CNN_' mode '_dataset0.dat'];

if exist([output_directory '/' name], 'file') ~= 0 %if file exists do not repeatedly create
    disp([num2str(0) ' EXISTS===' name])
else
    disp([num2str(0) ' ' name])
    disp(name)
end

figure(1)
%One purpose: replace rtm image (calculated by pysit) by preprocessed
%             image
v = dlmread([input_directory num2str(0) '/' 'Fortran_rtm' num2str(0) '.dat']);

v1 = v(1:nz*nx);
v1 = reshape(v1,nz,nx);
subplot(2,2,1);imagesc(v1);colormap(mycolor);caxis([-100000 100000]);
v1 = preprocess_rtm(v1);
subplot(2,2,2);imagesc(v1);colormap(mycolor);caxis([-100000 100000]);
v1 = reshape(v1,nz*nx,1);
v(1:nz*nx) = v1;

v1 = v(1+nz*nx:nz*nx+nz*nx);
v1 = reshape(v1,nz,nx);
subplot(2,2,3);imagesc(v1);colormap(mycolor);caxis([-100000 100000]);
v1 = preprocess_rtm(v1);
subplot(2,2,4);imagesc(v1);colormap(mycolor);caxis([-100000 100000]);
v1 = reshape(v1,nz*nx,1);
v(1+nz*nx:nz*nx+nz*nx) = v1;

% decimal output
disp(name)
disp(size(v)) 
fid=fopen([output_directory '/' name],'wt');
fprintf(fid,'%20.8f',v);
fclose(fid);

% binary output
%     fid = fopen(name,'w');
%     fwrite(fid,train_data,'float32');
%     fclose(fid);

