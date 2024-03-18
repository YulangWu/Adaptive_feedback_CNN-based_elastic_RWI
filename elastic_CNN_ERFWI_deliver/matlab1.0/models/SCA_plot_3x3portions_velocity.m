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
cvalue = 0.001;
nz = 256;
nx = 256;
dh = 0.0125;
iter = 19;
offset = [80 160];
pos = [2000 6500 10000];
water_depth=0;
nzz = nz - water_depth;
vp_true1 = dlmread(['0th_true_' 'vp' '.dat']);
vp_true1 = reshape(vp_true1,nz,nx);vp_true1=vp_true1(water_depth+1:nz,:);

vp_true2 = dlmread(['0th_true_' 'vs' '.dat']);
vp_true2 = reshape(vp_true2,nz,nx);vp_true2=vp_true2(water_depth+1:nz,:);

vp_true3 = dlmread(['0th_true_' 'rho' '.dat']);
vp_true3 = reshape(vp_true3,nz,nx);vp_true3=vp_true3(water_depth+1:nz,:);

vp_init1 = dlmread(['0th_mig_' 'vp' '.dat']);
vp_init1 = reshape(vp_init1,nz,nx);vp_init1=vp_init1(water_depth+1:nz,:);

vp_init2 = dlmread(['0th_mig_' 'vs' '.dat']);
vp_init2 = reshape(vp_init2,nz,nx);vp_init2=vp_init2(water_depth+1:nz,:);

vp_init3 = dlmread(['0th_mig_' 'rho' '.dat']);
vp_init3 = reshape(vp_init3,nz,nx);vp_init3=vp_init3(water_depth+1:nz,:);

vp_init11 = dlmread(['0th_mig_' 'vp' '.dat']);
vp_init11 = reshape(vp_init11,nz,nx);vp_init11=vp_init11(water_depth+1:nz,:);

vp_init22 = dlmread(['0th_mig_' 'vs' '.dat']);
vp_init22 = reshape(vp_init22,nz,nx);vp_init22=vp_init22(water_depth+1:nz,:);

vp_init33 = dlmread(['0th_mig_' 'rho' '.dat']);
vp_init33 = reshape(vp_init33,nz,nx);vp_init33=vp_init33(water_depth+1:nz,:);

vp1_err = zeros(1,iter+1);
vp2_err = zeros(1,iter+1);
vp3_err = zeros(1,iter+1);
vp1_err(1)=RMS_data(vp_true1,vp_init1);
vp2_err(1)=RMS_data(vp_true2,vp_init2);
vp3_err(1)=RMS_data(vp_true3,vp_init3);

vp11_err = zeros(1,iter+1);
vp22_err = zeros(1,iter+1);
vp33_err = zeros(1,iter+1);
vp11_err(1)=RMS_data(vp_true1,vp_init11);
vp22_err(1)=RMS_data(vp_true2,vp_init22);
vp33_err(1)=RMS_data(vp_true3,vp_init33);

i = 6;
%------------------ Plot the velocity models--------------------
hfig1 = figure(1);
sh = 0.03;
sv = 0.03;
padding = 0.0;
margin = 0.1;

subaxis(5,3,1, 'sh', sh, 'sv', sv, 'padding', padding, 'margin', margin);
imagesc(vp_true1);caxis([1.5 3.0]);axis equal;xlim([1 nx]);ylim([1 nzz]);
%title(['RMS_data v_p = ' num2str(RMS_data(vp_true1,vp_true1))])
set(gca,'XAxisLocation','top');
set(gca, 'XTick', [1 nx/3 nx/3*2 nx])            
set(gca,'XTickLabel',{'0.0','1.1','2.1','3.2'}) 
set(gca, 'YTick', [1 nx/3 nx/3*2 nx])            
set(gca,'YTickLabel',{'0.0','1.1','2.1','3.2'}) 
xlabel('Distance (km)'); 
ylabel('Depth (km)');
h=text(19.2142857142858, 279.27380952381,['RMSE = ' num2str(RMS_data(vp_true1,vp_true1)*100, '%4.2f') '%'])
%set(h,'Rotation',90);
text(-93.906015037594, -66.797619047619,'a)') %get from the code of figure


subaxis(5,3,2, 'sh', sh, 'sv', sv, 'padding', padding, 'margin', margin);
imagesc(vp_true2);caxis([0 1.5]);axis equal;xlim([1 nx]);ylim([1 nzz]);
%title(['RMS_data v_p = ' num2str(RMS_data(vp_true1,vp_true1))])
set(gca,'XAxisLocation','top');
set(gca, 'XTick', [1 nx/3 nx/3*2 nx])            
set(gca,'XTickLabel',{'0.0','1.1','2.1','3.2'}) 
set(gca, 'YTick', [])            
set(gca,'YTickLabel',{''}) 
xlabel('Distance (km)'); 
h=text(19.2142857142858, 279.27380952381,['RMSE = ' num2str(RMS_data(vp_true2,vp_true2)*100, '%4.2f') '%'])
%set(h,'Rotation',90);
text(-22.2738095238095, -66.797619047619,'b)') %get from the code of figure

subaxis(5,3,3, 'sh', sh, 'sv', sv, 'padding', padding, 'margin', margin);
imagesc(vp_true3);caxis([1.9 2.35]);axis equal;xlim([1 nx]);ylim([1 nzz]);
%title(['RMS_data v_p = ' num2str(RMS_data(vp_true1,vp_true1))])
set(gca,'XAxisLocation','top');
set(gca, 'XTick', [1 nx/3 nx/3*2 nx])            
set(gca,'XTickLabel',{'0.0','1.1','2.1','3.2'}) 
set(gca, 'YTick', [])            
set(gca,'YTickLabel',{''}) 
xlabel('Distance (km)'); 
h=text(19.2142857142858, 279.27380952381,['RMSE = ' num2str(RMS_data(vp_true3,vp_true3)*100, '%4.2f') '%'])
%set(h,'Rotation',90);
text(-22.2738095238095, -66.797619047619,'c)') %get from the code of figure
h1=text(290.404761904762, 220.583333333333, 'True model');set(h1,'Rotation',90);

subaxis(5,3,4, 'sh', sh, 'sv', sv, 'padding', padding, 'margin', margin);
imagesc(vp_init1);caxis([1.5 3.0]);axis equal;xlim([1 nx]);ylim([1 nzz]);
%title(['RMS_data v_p = ' num2str(RMS_data(vp_true1,vp_init1))])
set(gca,'XAxisLocation','top');
set(gca, 'XTick', [])            
set(gca,'XTickLabel',{''}) 
set(gca, 'YTick', [1 nx/3 nx/3*2 nx])            
set(gca,'YTickLabel',{'0.0','1.1','2.1','3.2'}) 
ylabel('Depth (km)');
h=text(19.2142857142858, 279.27380952381,['RMSE = ' num2str(RMS_data(vp_true1,vp_init1)*100, '%4.2f') '%'])
%set(h,'Rotation',90);
text(-93.906015037594, -15.2969924812029,'d)') %get from the code of figure

subaxis(5,3,5, 'sh', sh, 'sv', sv, 'padding', padding, 'margin', margin);
imagesc(vp_init2);caxis([0 1.5]);axis equal;xlim([1 nx]);ylim([1 nzz]);
%title(['RMS_data v_p = ' num2str(RMS_data(vp_true1,vp_init1))])
set(gca,'XAxisLocation','top');
set(gca, 'XTick', [])            
set(gca,'XTickLabel',{''}) 
set(gca, 'YTick', [])            
set(gca,'YTickLabel',{''}) 
h=text(19.2142857142858, 279.27380952381,['RMSE = ' num2str(RMS_data(vp_true2,vp_init2)*100, '%4.2f') '%'])
%set(h,'Rotation',90);
text(-22.2738095238095, -15.2969924812029,'e)') %get from the code of figure

subaxis(5,3,6, 'sh', sh, 'sv', sv, 'padding', padding, 'margin', margin);
imagesc(vp_init3);caxis([1.9 2.35]);axis equal;xlim([1 nx]);ylim([1 nzz]);
%title(['RMS_data v_p = ' num2str(RMS_data(vp_true1,vp_init1))])
set(gca,'XAxisLocation','top');
set(gca, 'XTick', [])            
set(gca,'XTickLabel',{''}) 
set(gca, 'YTick', [])            
set(gca,'YTickLabel',{''}) 
h=text(19.2142857142858, 279.27380952381,['RMSE = ' num2str(RMS_data(vp_true3,vp_init3)*100, '%4.2f') '%'])
%set(h,'Rotation',90);
h1=text(290.40, 220.583333333333, 'Initial model');set(h1,'Rotation',90);
text(-22.2738095238095, -15.2969924812029,'f)') %get from the code of figure

vp1 = dlmread([num2str(iter) 'th_true_' 'vp' '.dat']);
vp1 = reshape(vp1,nz,nx);vp1=vp1(water_depth+1:nz,:);

vp2 = dlmread([num2str(iter) 'th_true_' 'vs' '.dat']);
vp2 = reshape(vp2,nz,nx);vp2=vp2(water_depth+1:nz,:);

vp3 = dlmread([num2str(iter) 'th_true_' 'rho' '.dat']);
vp3 = reshape(vp3,nz,nx);vp3=vp3(water_depth+1:nz,:);

vp11 = dlmread([num2str(iter) 'th_true_' 'vp' '.dat']);
vp11 = reshape(vp11,nz,nx);vp11=vp11(water_depth+1:nz,:);

vp22 = dlmread([num2str(iter) 'th_true_' 'vs' '.dat']);
vp22 = reshape(vp22,nz,nx);vp22=vp22(water_depth+1:nz,:);

vp33 = dlmread([num2str(iter) 'th_true_' 'rho' '.dat']);
vp33 = reshape(vp33,nz,nx);vp33=vp33(water_depth+1:nz,:);

vp1_err(i+1)=RMS_data(vp_true1,vp1);
vp2_err(i+1)=RMS_data(vp_true2,vp2);
vp3_err(i+1)=RMS_data(vp_true3,vp3);

vp11_err(i+1)=RMS_data(vp_true1,vp11);
vp22_err(i+1)=RMS_data(vp_true2,vp22);
vp33_err(i+1)=RMS_data(vp_true3,vp33);

subaxis(5,3,7, 'sh', sh, 'sv', sv, 'padding', padding, 'margin', margin);
imagesc(vp1);caxis([1.5 3.0]);axis equal;xlim([1 nx]);ylim([1 nzz]);
%title(['RMS_data = ' num2str(vp1_err(i+1))])
set(gca,'XAxisLocation','top');
set(gca, 'XTick', [])            
set(gca,'XTickLabel',{''}) 
set(gca, 'YTick', [1 nx/3 nx/3*2 nx])            
set(gca,'YTickLabel',{'0.0','1.1','2.1','3.2'}) 
ylabel('Depth (km)');
h=text(19.2142857142858, 279.27380952381,['RMSE = ' num2str(RMS_data(vp_true1,vp1)*100, '%4.2f') '%'])
%set(h,'Rotation',90);
text(-93.906015037594, -15.2969924812029,'g)') %get from the code of figure

c = colorbar('Location','SouthOutside');
colorTitleHandle = get(c,'XLabel');
titleString = 'P-wave velocity (km/s)';
set(colorTitleHandle ,'String',titleString);
set(c,'XTick',[1.5,3.0])
set(c,'XTickLabels',{'1.5','3.0'})
set(c,'Position',[0.110429623655202 0.385908332469779 0.224676759323522 0.0157170923379182]) %get from the code of figure
    
subaxis(5,3,8, 'sh', sh, 'sv', sv, 'padding', padding, 'margin', margin);
imagesc(vp2);caxis([0 1.5]);axis equal;xlim([1 nx]);ylim([1 nzz]);
%title(['RMS_data = ' num2str(vp1_err(i+1))])
set(gca,'XAxisLocation','top');
set(gca, 'XTick', [])            
set(gca,'XTickLabel',{''}) 
set(gca, 'YTick', [])            
set(gca,'YTickLabel',{''}) 
h=text(19.2142857142858, 279.27380952381,['RMSE = ' num2str(RMS_data(vp_true2,vp2)*100, '%4.2f') '%'])
%set(h,'Rotation',90);
text(-22.2738095238095, -15.2969924812029,'h)') %get from the code of figure

c = colorbar('Location','SouthOutside');
colorTitleHandle = get(c,'XLabel');
titleString = 'S-wave velocity (km/s)';
set(colorTitleHandle ,'String',titleString);
set(c,'XTick',[0,1.5])
set(c,'XTickLabels',{'0.0','1.5'})
set(c,'Position',[0.387025368336052 0.385908332469779 0.224676759323522 0.0157170923379182]) %get from the code of figure

subaxis(5,3,9, 'sh', sh, 'sv', sv, 'padding', padding, 'margin', margin);
imagesc(vp3);caxis([1.9 2.35]);axis equal;xlim([1 nx]);ylim([1 nzz]);
%title(['RMS_data = ' num2str(vp1_err(i+1))])
set(gca,'XAxisLocation','top');
set(gca, 'XTick', [])            
set(gca,'XTickLabel',{''}) 
set(gca, 'YTick', [])            
set(gca,'YTickLabel',{''}) 
h=text(19.2142857142858, 279.27380952381,['RMSE = ' num2str(RMS_data(vp_true3,vp3)*100, '%4.2f') '%'])
%set(h,'Rotation',90);
h1=text(290.40, 265.035433070866, 'CNN-output model');set(h1,'Rotation',90);
text(-22.2738095238095, -15.2969924812029,'i)') %get from the code of figure

% subaxis(5,3,10, 'sh', sh, 'sv', sv, 'padding', padding, 'margin', margin);
% imagesc(vp_init11);caxis([1.5 3.0]);axis equal;xlim([1 nx]);ylim([1 nzz]);
% %title(['RMS_data v_p = ' num2str(RMS_data(vp_true1,vp_init1))])
% set(gca,'XAxisLocation','top');
% set(gca, 'XTick', [])            
% set(gca,'XTickLabel',{''}) 
% set(gca, 'YTick', [1 nx/3 nx/3*2 nx])            
% set(gca,'YTickLabel',{'0.0','1.1','2.1','3.2'}) 
% ylabel('Depth (km)');
% h=text(19.2142857142858, 279.27380952381,['RMSE = ' num2str(RMS_data(vp_true1,vp_init11)*100, '%4.2f') '%'])
% %set(h,'Rotation',90);
% text(-93.906015037594, -15.2969924812029,'j)') %get from the code of figure
% 
% subaxis(5,3,11, 'sh', sh, 'sv', sv, 'padding', padding, 'margin', margin);
% imagesc(vp_init22);caxis([0 1.5]);axis equal;xlim([1 nx]);ylim([1 nzz]);
% %title(['RMS_data v_p = ' num2str(RMS_data(vp_true1,vp_init1))])
% set(gca,'XAxisLocation','top');
% set(gca, 'XTick', [])            
% set(gca,'XTickLabel',{''}) 
% set(gca, 'YTick', [])            
% set(gca,'YTickLabel',{''}) 
% h=text(19.2142857142858, 279.27380952381,['RMSE = ' num2str(RMS_data(vp_true2,vp_init22)*100, '%4.2f') '%'])
% %set(h,'Rotation',90);
% text(-22.2738095238095, -15.2969924812029,'k)') %get from the code of figure
% 
% subaxis(5,3,12, 'sh', sh, 'sv', sv, 'padding', padding, 'margin', margin);
% imagesc(vp_init33);caxis([1.9 2.35]);axis equal;xlim([1 nx]);ylim([1 nzz]);
% %title(['RMS_data v_p = ' num2str(RMS_data(vp_true1,vp_init1))])
% set(gca,'XAxisLocation','top');
% set(gca, 'XTick', [])            
% set(gca,'XTickLabel',{''}) 
% set(gca, 'YTick', [])            
% set(gca,'YTickLabel',{''}) 
% h=text(19.2142857142858, 279.27380952381,['RMSE = ' num2str(RMS_data(vp_true3,vp_init33)*100, '%4.2f') '%'])
% %set(h,'Rotation',90);
% h1=text(290.40, 220.583333333333, 'Initial model');set(h1,'Rotation',90);
% text(-22.2738095238095, -15.2969924812029,'l)') %get from the code of figure
% 







% 
% subaxis(5,3,13, 'sh', sh, 'sv', sv, 'padding', padding, 'margin', margin);
% imagesc(vp11);caxis([1.5 3.0]);axis equal;xlim([1 nx]);ylim([1 nzz]);
% %title(['RMS_data = ' num2str(vp1_err(i+1))])
% set(gca,'XAxisLocation','top');
% set(gca, 'XTick', [])            
% set(gca,'XTickLabel',{''}) 
% set(gca, 'YTick', [1 nx/3 nx/3*2 nx])            
% set(gca,'YTickLabel',{'0.0','1.1','2.1','3.2'}) 
% ylabel('Depth (km)');
% h=text(19.2142857142858, 279.27380952381,['RMSE = ' num2str(RMS_data(vp_true1,vp11)*100, '%4.2f') '%'])
% %set(h,'Rotation',90);
% text(-93.906015037594, -15.2969924812029,'m)') %get from the code of figure
% 
% subaxis(5,3,14, 'sh', sh, 'sv', sv, 'padding', padding, 'margin', margin);
% imagesc(vp22);caxis([1.5 3.0]);axis equal;xlim([1 nx]);ylim([1 nzz]);
% %title(['RMS_data = ' num2str(vp1_err(i+1))])
% set(gca,'XAxisLocation','top');
% set(gca, 'XTick', [])            
% set(gca,'XTickLabel',{''}) 
% set(gca, 'YTick', [])            
% set(gca,'YTickLabel',{''}) 
% h=text(19.2142857142858, 279.27380952381,['RMSE = ' num2str(RMS_data(vp_true2,vp22)*100, '%4.2f') '%'])
% %set(h,'Rotation',90);
% text(-22.2738095238095, -15.2969924812029,'n)') %get from the code of figure
% 
% subaxis(5,3,15, 'sh', sh, 'sv', sv, 'padding', padding, 'margin', margin);
% imagesc(vp33);caxis([1.5 3.0]);axis equal;xlim([1 nx]);ylim([1 nzz]);
% %title(['RMS_data = ' num2str(vp1_err(i+1))])
% set(gca,'XAxisLocation','top');
% set(gca, 'XTick', [])            
% set(gca,'XTickLabel',{''}) 
% set(gca, 'YTick', [])            
% set(gca,'YTickLabel',{''}) 
% h=text(19.2142857142858, 279.27380952381,['RMSE = ' num2str(RMS_data(vp_true3,vp33)*100, '%4.2f') '%'])
% %set(h,'Rotation',90);
% h1=text(290.40, 265.035433070866, 'CNN-output model');set(h1,'Rotation',90);
% h2=text(342.338582677165,78.3031496062993,'Synthetic test 2');set(h2,'Rotation',90);
% text(-22.2738095238095, -15.2969924812029,'o)') %get from the code of figure



c = colorbar('Location','SouthOutside');
colorTitleHandle = get(c,'XLabel');
titleString = 'Density (g/cc)';
set(colorTitleHandle ,'String',titleString);
set(c,'XTick',[1.9,2.3])
set(c,'XTickLabels',{'1.9','2.3'})
set(c,'Position',[0.663621113016902 0.385908332469779 0.224676759323522 0.0157170923379182]) %get from the code of figure


%------------------ Plot the velocity models--------------------
hfig1 = figure(2);
sh = 0.03;
sv = 0.03;
padding = 0.0;
margin = 0.1;
k=1;
subaxis(2,3,1, 'sh', sh, 'sv', sv, 'padding', padding, 'margin', margin);
depth=(0:1:nzz-1)*dh; % The depth (m) at each vertical grid lines
x_position = offset(k); %nx/4*1; % The horizontal position (unitless)
plot(vp_true1(:,x_position),depth,'k','LineWidth',1.5);hold on;
plot(vp_init1(:,x_position),depth,'b','LineWidth',1.5);hold on;
plot(vp1(:,x_position),depth,'r','LineWidth',1.5);hold off;
set(gca,'YDir','reverse')
xlabel('P-wave velocity (km/s)')
ylabel('Depth (km)')
set(gca,'ytick',[1 nz/4 nz/2 nz/4*3 nz]*dh) 
set(gca,'xtick',1.5:0.5:2.5) 
set(gca,'xticklabel',sprintf('%3.1f|',get(gca,'xtick')))
set(gca,'yticklabel',sprintf('%3.1f|',get(gca,'ytick')))
axis([1.5 2.7 1*dh (nzz)*dh]) 
set(gca,'XAxisLocation','top');
text(1.15015060240964, -0.55,'a)')
%title(i)

subaxis(2,3,2, 'sh', sh, 'sv', sv, 'padding', padding, 'margin', margin);
depth=(0:1:nzz-1)*dh; % The depth (m) at each vertical grid lines
x_position = offset(k); %nx/4*1; % The horizontal position (unitless)
plot(vp_true2(:,x_position),depth,'k','LineWidth',1.5);hold on;
plot(vp_init2(:,x_position),depth,'b','LineWidth',1.5);hold on;
plot(vp2(:,x_position),depth,'r','LineWidth',1.5);hold off;
set(gca,'YDir','reverse')
xlabel('S-wave velocity (km/s)')
set(gca,'ytick',[]*dh) 
set(gca,'xtick',0:0.5:1.0) 
set(gca,'xticklabel',sprintf('%3.1f|',get(gca,'xtick')))
set(gca,'yticklabel',sprintf('%3.1f|',get(gca,'ytick')))
axis([0.0 1.2 1*dh (nzz)*dh]) 

set(gca,'XAxisLocation','top');
text(-0.126506024096386, -0.55,'b)')

subaxis(2,3,3, 'sh', sh, 'sv', sv, 'padding', padding, 'margin', margin);
depth=(0:1:nzz-1)*dh; % The depth (m) at each vertical grid lines
x_position = offset(k); %nx/4*1; % The horizontal position (unitless)
plot(vp_true3(:,x_position),depth,'k','LineWidth',1.5);hold on;
plot(vp_init3(:,x_position),depth,'b','LineWidth',1.5);hold on;
plot(vp3(:,x_position),depth,'r','LineWidth',1.5);hold off;
set(gca,'YDir','reverse')
xlabel('Density (g/cc)')
% ylabel('Depth (km)')
set(gca,'ytick',[]*dh) 
set(gca,'xtick',1.9:0.1:2.2) 
set(gca,'xticklabel',sprintf('%3.1f|',get(gca,'xtick')))
set(gca,'yticklabel',sprintf('%3.1f|',get(gca,'ytick')))
axis([1.9 2.28 1*dh (nzz)*dh]) 

set(gca,'XAxisLocation','top');
text(1.85993975903614, -0.55,'c)')
h=text(2.32006024096386, 2.56742912371134,'Profile at 1.0 km');set(h,'Rotation',90);


k=2
subaxis(2,3,4, 'sh', sh, 'sv', sv, 'padding', padding, 'margin', margin);
depth=(0:1:nzz-1)*dh; % The depth (m) at each vertical grid lines
x_position = offset(k); %nx/4*1; % The horizontal position (unitless)
plot(vp_true1(:,x_position),depth,'k','LineWidth',1.5);hold on;
plot(vp_init11(:,x_position),depth,'b','LineWidth',1.5);hold on;
plot(vp11(:,x_position),depth,'r','LineWidth',1.5);hold off;
set(gca,'YDir','reverse')
% xlabel('Velocity (km/s)')
ylabel('Depth (km)')
set(gca,'ytick',[1 nz/4 nz/2 nz/4*3 nz]*dh) 
set(gca,'xtick',[]) 
set(gca,'xticklabel',sprintf('%3.1f|',get(gca,'xtick')))
set(gca,'yticklabel',sprintf('%3.1f|',get(gca,'ytick')))
axis([1.5 2.7 1*dh (nzz)*dh]) 

set(gca,'XAxisLocation','top');
%title(i)

subaxis(2,3,5, 'sh', sh, 'sv', sv, 'padding', padding, 'margin', margin);
depth=(0:1:nzz-1)*dh; % The depth (m) at each vertical grid lines
x_position = offset(k); %nx/4*1; % The horizontal position (unitless)
plot(vp_true2(:,x_position),depth,'k','LineWidth',1.5);hold on;
plot(vp_init22(:,x_position),depth,'b','LineWidth',1.5);hold on;
plot(vp22(:,x_position),depth,'r','LineWidth',1.5);hold off;
set(gca,'YDir','reverse')
% xlabel('Velocity (km/s)')
set(gca,'ytick',[]*dh)
set(gca,'xtick',[]) 
set(gca,'xticklabel',sprintf('%3.1f|',get(gca,'xtick')))
set(gca,'yticklabel',sprintf('%3.1f|',get(gca,'ytick')))
axis([0.0 1.2 1*dh (nzz)*dh]) 
set(gca,'XAxisLocation','top');

subaxis(2,3,6, 'sh', sh, 'sv', sv, 'padding', padding, 'margin', margin);
depth=(0:1:nzz-1)*dh; % The depth (m) at each vertical grid lines
x_position = offset(k); %nx/4*1; % The horizontal position (unitless)
plot(vp_true3(:,x_position),depth,'k','LineWidth',1.5);hold on;
plot(vp_init33(:,x_position),depth,'b','LineWidth',1.5);hold on;
plot(vp33(:,x_position),depth,'r','LineWidth',1.5);hold off;
set(gca,'YDir','reverse')
% xlabel('Velocity (km/s)')
% ylabel('Depth (km)')
set(gca,'ytick',[]*dh) 
set(gca,'xtick',[]) 
set(gca,'xticklabel',sprintf('%3.1f|',get(gca,'xtick')))
set(gca,'yticklabel',sprintf('%3.1f|',get(gca,'ytick')))
axis([1.9 2.28 1*dh (nzz)*dh]) 
set(gca,'XAxisLocation','top');
h=text(2.32006024096386, 2.56742912371134,'Profile at 2.0 km');set(h,'Rotation',90);

