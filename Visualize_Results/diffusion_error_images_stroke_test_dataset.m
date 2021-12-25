clc
clearvars

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% change this index for different undersampling patterns
% 1 ----> 12_DWI
% 2 ----> 30_DWI
% 3 ----> 60_DWI

sampling_idx = 2;


% change this scaling factor for changing the image size
fig_scale = 1.5;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nets_name = {'\textbf{1D-qDL}','\textbf{2D-CNN}', '\textbf{MESC-SD}'};

test_dir = {'test_1d','test_2d','test_3d'};

samplings = {'12_DWI','30_DWI','60_DWI'};

samplings_name = {'\textbf{(12 DWIs)}','\textbf{(30 DWIs)}','\textbf{(60 DWIs)}'};

data_path = './../Data/';

maps = {'dti_FA.nii.gz', 'dti_MD.nii.gz', 'dti_RD.nii.gz',...
    'dki_AK.nii.gz', 'dki_MK.nii.gz', 'dki_RK.nii.gz', ...
    'noddi_ODI.nii.gz', 'noddi_FISO.nii.gz', 'noddi_FICVF.nii.gz',...
    'smt_v_int.nii.gz', 'smt_lambda.nii.gz'};

maps_name = {'\textbf{FA}','\textbf{MD}', '\textbf{RD}',...
    '\textbf{AK}', '\textbf{MK}', '\textbf{RK}', ...
    '\textbf{ODI}', '$\mathbf{v_{iso}}$', '$\mathbf{v_{ic}}$',...
    '$\mathbf{v_{int}}$', '$\mathbf{\lambda}$'};




slice = 35;
x_crop = (25:125);
y_crop = (1:120);
n_x = length(y_crop);
n_y = length(x_crop);

bx_x = 52;
bx_y = 58;
x_box = (bx_y: bx_y + round(length(x_crop)/8));
y_box = (length(y_crop)-bx_x - round(length(y_crop)/8):length(y_crop)-bx_x);





nx = round(fig_scale*n_x);
ny = round(fig_scale*n_y);
box_pos = [bx_y bx_x round(length(x_crop)/8) round(length(y_crop)/8)];

ff = round(fig_scale*10);
ff_small = round(fig_scale*7);
offx = round(fig_scale*n_x/3);
offy = round(3*fig_scale*n_y/4);
offx_t = round(fig_scale*n_x/6);
offy_t = round(fig_scale*n_y/12);


blueColorMap = [linspace(1, 0, 124), zeros(1, 132)];
redColorMap = [zeros(1, 132), linspace(0, 1, 124)];
colorMap = [redColorMap; zeros(1, 256);blueColorMap]';


hfig = figure('unit','pixel','position',[1 1 11*ny+offy+offy_t 4*nx+offx+offx_t]);
mask_ref = niftiread([data_path 'Fully_Sampled/M_56_Strk/mask.nii.gz' ]);
mask_ref = single(mask_ref(x_crop,y_crop, slice));
for i = 0:10
    ax_cbar = axes('parent',hfig,'unit','pixel','position',[offy+i*ny 0 ny offx]);
    ax1 = axes('parent',hfig,'unit','pixel','position',[offy+i*ny 3*nx+offx ny nx]);
    ax2 = axes('parent',hfig,'unit','pixel','position',[offy+i*ny 2*nx+offx ny nx]);
    ax3 = axes('parent',hfig,'unit','pixel','position',[offy+i*ny 1*nx+offx ny nx]);
    ax4 = axes('parent',hfig,'unit','pixel','position',[offy+i*ny 0*nx+offx ny nx]);
    ax1_bx = axes('parent',hfig,'unit','pixel','position',[offy+i*ny 3*nx+offx round(ny/2.5) round(nx/2.5)]);
    ax2_bx = axes('parent',hfig,'unit','pixel','position',[offy+i*ny 2*nx+offx round(ny/2.5) round(nx/2.5)]);
    ax3_bx = axes('parent',hfig,'unit','pixel','position',[offy+i*ny 1*nx+offx round(ny/2.5) round(nx/2.5)]);
    ax4_bx = axes('parent',hfig,'unit','pixel','position',[offy+i*ny 0*nx+offx round(ny/2.5) round(nx/2.5)]);
    
    
    img0 = niftiread([data_path 'Fully_Sampled/M_56_Strk/' maps{i+1}]);
    img0 = img0(x_crop,y_crop, slice).*mask_ref;
    img0_bx = img0(x_box,y_box);
    M = max(img0(:));
    colorbar(ax_cbar,'south','fontsize',ff_small);caxis(ax_cbar,[-M M]);axis(ax_cbar,'off');colormap(ax_cbar,colorMap);
    
   
    img1 = niftiread([data_path samplings{sampling_idx} '/processed/M_56_Strk/Model_Fitting/'  maps{i+1}]);
    img1 = img1(x_crop,y_crop, slice).*mask_ref;
    img1_bx = img1(x_box,y_box);
    imagesc(3*(rot90(img1-img0)),'parent',ax1);caxis(ax1,[-M M]);colormap(ax1,colorMap);
    title(ax1,[maps_name{i+1} ' \textbf{err(3x)}'],'interpreter','latex','fontsize',ff,'fontweight','bold');
    set(ax1,'XTickLabel','');set(ax1,'YTickLabel','');
    imagesc(3*(rot90(img1_bx-img0_bx)),'parent',ax1_bx);caxis(ax1_bx,[-M M]);colormap(ax1_bx,colorMap);
    set(ax1_bx,'XTickLabel','');set(ax1_bx,'YTickLabel','');set(ax1_bx,'XColor','g');set(ax1_bx,'YColor','g');
    
    
    img2 = niftiread([data_path samplings{sampling_idx} '/' test_dir{1} '/M_56_Strk/' maps{i+1}]);
    img2 = img2(x_crop,y_crop, slice).*mask_ref;
    img2_bx = img2(x_box,y_box);
    imagesc(3*(rot90(img2-img0)),'parent',ax2);caxis(ax2,[-M M]);colormap(ax2,colorMap);
    set(ax2,'XTickLabel','');set(ax2,'YTickLabel','');
    imagesc(3*(rot90(img2_bx-img0_bx)),'parent',ax2_bx);caxis(ax2_bx,[-M M]);colormap(ax2_bx,colorMap);
    set(ax2_bx,'XTickLabel','');set(ax2_bx,'YTickLabel','');set(ax2_bx,'XColor','g');set(ax2_bx,'YColor','g');
    
    
    img3 = niftiread([data_path samplings{sampling_idx} '/' test_dir{2} '/M_56_Strk/' maps{i+1}]);
    img3 = img3(x_crop,y_crop, slice).*mask_ref;
    img3_bx = img3(x_box,y_box);
    imagesc(3*(rot90(img3-img0)),'parent',ax3);caxis(ax3,[-M M]);colormap(ax3,colorMap);
    set(ax3,'XTickLabel','');set(ax3,'YTickLabel','');
    imagesc(3*(rot90(img3_bx-img0_bx)),'parent',ax3_bx);caxis(ax3_bx,[-M M]);colormap(ax3_bx,colorMap);
    set(ax3_bx,'XTickLabel','');set(ax3_bx,'YTickLabel','');set(ax3_bx,'XColor','g');set(ax3_bx,'YColor','g');
    
    
    img4 = niftiread([data_path samplings{sampling_idx} '/' test_dir{3} '/M_56_Strk/' maps{i+1}]);
    img4 = img4(x_crop,y_crop, slice).*mask_ref;
    img4_bx = img4(x_box,y_box);
    imagesc(3*(rot90(img4-img0)),'parent',ax4);caxis(ax4,[-M M]);colormap(ax4,colorMap);
    set(ax4,'XTickLabel','');set(ax4,'YTickLabel','');
    imagesc(3*(rot90(img4_bx-img0_bx)),'parent',ax4_bx);caxis(ax4_bx,[-M M]);colormap(ax4_bx,colorMap);
    set(ax4_bx,'XTickLabel','');set(ax4_bx,'YTickLabel','');set(ax4_bx,'XColor','g');set(ax4_bx,'YColor','g');
    
      
end

ax_lb1 = axes('parent',hfig,'unit','pixel','position',[0 3*nx+offx offy nx]);
ax_lb2 = axes('parent',hfig,'unit','pixel','position',[0 2*nx+offx offy nx]);
ax_lb3 = axes('parent',hfig,'unit','pixel','position',[0 1*nx+offx offy nx]);
ax_lb4 = axes('parent',hfig,'unit','pixel','position',[0 0*nx+offx offy nx]);

back = ones(nx,offy);
imagesc(back,'parent',ax_lb1);caxis(ax_lb1,[0 1]);colormap(ax_lb1,'gray');text(ax_lb1,round(offy/2),round(nx/2),{'\textbf{MF}';samplings_name{sampling_idx}},'color','black','fontsize',ff,'HorizontalAlignment','center','Interpreter','latex');axis(ax_lb1,'off')
imagesc(back,'parent',ax_lb2);caxis(ax_lb2,[0 1]);colormap(ax_lb2,'gray');text(ax_lb2,round(offy/2),round(nx/2),{nets_name{1};samplings_name{sampling_idx}},'color','black','fontsize',ff,'HorizontalAlignment','center','Interpreter','latex');axis(ax_lb2,'off')
imagesc(back,'parent',ax_lb3);caxis(ax_lb3,[0 1]);colormap(ax_lb3,'gray');text(ax_lb3,round(offy/2),round(nx/2),{nets_name{2};samplings_name{sampling_idx}},'color','black','fontsize',ff,'HorizontalAlignment','center','Interpreter','latex');axis(ax_lb3,'off')
imagesc(back,'parent',ax_lb4);caxis(ax_lb4,[0 1]);colormap(ax_lb4,'gray');text(ax_lb4,round(offy/2),round(nx/2),{nets_name{3};samplings_name{sampling_idx}},'color','black','fontsize',ff,'HorizontalAlignment','center','Interpreter','latex');axis(ax_lb4,'off')















