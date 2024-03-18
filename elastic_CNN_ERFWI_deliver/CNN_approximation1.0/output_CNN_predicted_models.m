
nx=sh_nx;
nz=sh_nz;

input_vp_filename = sh_input_vp_filename;
input_vs_filename = sh_input_vs_filename;
input_rho_filename = sh_input_rho_filename;
output_vp_filename = sh_output_vp_filename;
output_vs_filename = sh_output_vs_filename;
output_rho_filename = sh_output_rho_filename;

vp_data = dlmread(input_vp_filename);
vs_data = dlmread(input_vs_filename);
rho_data = dlmread(input_rho_filename);

CNN_output_vp= vp_data(1+nz*nx*3:nz*nx*4)/1000;
CNN_output_vs= vs_data(1+nz*nx*3:nz*nx*4)/1000;
CNN_output_rho= rho_data(1+nz*nx*3:nz*nx*4)/1000;
CNN_output_vp = reshape(CNN_output_vp,nz,nx);
CNN_output_vs = reshape(CNN_output_vs,nz,nx);
CNN_output_rho = reshape(CNN_output_rho,nz,nx);

vp = reshape(CNN_output_vp,1,nz*nx);
fid=fopen([output_vp_filename '.dat'],'wt');
fprintf(fid,'%17.8f',vp);
fclose(fid);

vs = reshape(CNN_output_vs,1,nz*nx);
fid=fopen([output_vs_filename '.dat'],'wt');
fprintf(fid,'%17.8f',vs);
fclose(fid);

rho = reshape(CNN_output_rho,1,nz*nx);
fid=fopen([output_rho_filename '.dat'],'wt');
fprintf(fid,'%17.8f',rho);
fclose(fid);
