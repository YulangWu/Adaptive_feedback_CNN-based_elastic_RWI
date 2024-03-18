clear all
close all
clc
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
cvalue = 0.01*256/2;
%---------------------------------------------------------------

nx=256;
nt=4000;
iter = 19;
err = zeros(1,iter+1);
figure(1)

a1 = dlmread('output_data/seis_data0.dat');
a1 = reshape(a1,nx,nt*2).';
a1 = a1(1:nt,:);
    
for i = iter : iter
    subplot(1,3,1)
    
    imagesc(a1);colormap(mycolor);caxis([-cvalue cvalue])
    title('Observed data')
    set(gca,'XAxisLocation','top'); 
    set(gca, 'XTick', [1 nx/3 nx/3*2 nx-6])            
    set(gca,'XTickLabel',{'0.0','1.1','2.1','3.2'}) 
    set(gca, 'YTick', [1 1000 2000 3000 4000])          
    set(gca,'YTickLabel',{'0.0','1.0','2.0','3.0','4.0'}) 
    xlabel('Position (km)')
    ylabel('Time (s)')

    subplot(1,3,2)
    a = dlmread(['output_data/seis_data' num2str(i) '.dat']);
    a = reshape(a,nx,nt*2).';
    a2 = a(1:nt,:);
    imagesc(a2);colormap(mycolor);caxis([-cvalue cvalue])
    title(['Data at ' num2str(iter) ' iteraion'])
    set(gca,'XAxisLocation','top'); 
    set(gca, 'XTick', [1 nx/3 nx/3*2 nx-6])            
    set(gca,'XTickLabel',{'0.0','1.1','2.1','3.2'}) 
    set(gca, 'YTick', [])          
    set(gca,'YTickLabel',{}) 
    xlabel('Position (km)')
    
    subplot(1,3,3)
    a = reshape(a,nx,nt*2).';
    imagesc(a1-a2);colormap(mycolor);caxis([-cvalue cvalue])
    err(i+1)=RMS_data(a1,a2);
    title('Data residuals')
    set(gca,'XAxisLocation','top'); 
    set(gca, 'XTick', [1 nx/3 nx/3*2 nx-6])            
    set(gca,'XTickLabel',{'0.0','1.1','2.1','3.2'}) 
    set(gca, 'YTick', [])          
    set(gca,'YTickLabel',{}) 
    xlabel('Position (km)')
    
    drawnow;pause(0.00)
end