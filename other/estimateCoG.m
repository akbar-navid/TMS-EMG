close all; 
clc;
clearvars; 
k = 3;
% CoG_gt = zeros(k,3);
% CoG_vae = zeros(k,3);
% CoG_shift = zeros(k,1);
p = 0;
    
% b = load(sprintf('025//Apr_13//X_Var_Ind_Mus_15_%d.mat',1)); 
% rt = b.X_Var_Ind_Mus_15; 
% r = rt;
% 
% b = load(sprintf('025//Apr_13//X_Var_Ind_Mus_15_%d.mat',2)); 
% rt = b.X_Var_Ind_Mus_15; 
% r = rt + r;

% for i = [1]
% for i = [17,27,33,47]
for i = [6,24,39,42]
% for i = [42]
    figure;
%     p = p +1; 
%     r = box_data;
%     r = box_matrix_E;
%     r = logical_mask;
%     r = box_field_E;
%     r = reshape(r,[64,64,64]);
% % 
%     b = load(sprintf('025//Mar_03//X_test_f1_%d.mat',i)); 
%     r = b.X_test_f1; 
    
%     b = load(sprintf('X_AE_dec_%d.mat',i)); 
%     r = b.X_AE_dec; 
    
%     b = load(sprintf('025//Mar_03//X_Conv_%d.mat',i)); 
%     r = b.X_Conv; 
    
%     b = load(sprintf('X_VAE_dec_%d.mat',i)); 
%     r = b.X_VAE_dec; 
    
%     b = load(sprintf('025//Mar_03//X_VAE_mod_%d.mat',i)); 
%     r = b.X_VAE_mod; 
% %     
    b = load(sprintf('025//Mar_03//X_Var_%d.mat',i)); 
    r = b.X_Var; 
%     
%     b = load(sprintf('025//Mar_03//X_Var_Ind_Mus_%d.mat',i)); 
%     r = b.X_Var_Ind_Mus; 
    
%     b = load(sprintf('025//Apr_13//X_Var_Ind_Mus_15_%d.mat',i)); 
%     r = b.X_Var_Ind_Mus_15; 

%     r = permute(r,[3 1 2]);
%     r = fliplr(r);
%     r = flipud(r);
%     r = rot90(r);
%     r = flip(r);
        
    [max_num, max_idx] = max(r(:));
    r_shape = size(r);
    [X,Y,Z] = ind2sub(r_shape,max_idx);

    % r = zone;
    % r = reshape(r,[230,190,180]);
    [x,y,z] = ind2sub(size(r),find(r>0));
    % r = reshape(r,[7866000,1]);
    r = reshape(r,[262144,1]);

    % ---Find COG---
    m = 0; Xc = 0; Yc = 0; Zc = 0; n = 0;
    for k = 1:262144
        if r(k) > 0
            m = m + 1;
            [xc,yc,zc] = ind2sub(r_shape,k);
            Xc = Xc + r(k) * xc;
            Yc = Yc + r(k) * yc;
            Zc = Zc + r(k) * zc;
            n = n + r(k);
        end
    end
    Xc = Xc / sum(r);
    Yc = Yc / sum(r);
    Zc = Zc / sum(r);
    
%     CoG_gt(p,:) = [Xc,Yc,Zc]
%     
%     CoG_vae(p,:) = [Xc,Yc,Zc]
%     CoG_shift(p) = vecnorm(CoG_gt(p,:) - CoG_vae(p,:), 2, 2)

%     q = cell(1,1);
%     q{1} = 'fit';
%     p = 0;

    k = find(r>0);
    r = r(k);
    sc = scatter3(x,y,z,80,r,'filled');
%     sc = surf(x,y,r);
%     view(r,[0 45 0]);

    dt1 = datatip(sc,Xc,Yc,Zc);
    sc.DataTipTemplate.DataTipRows(1).Label = 'CoG-fit: ';
    
%     direction = [0 1 1];
%     rotate(sc,direction,45);

    % row = dataTipTextRow('CoG-fit',q(1));
    % % row = dataTipTextRow('CoG-fit',statelabel);
    % sc.DataTipTemplate.DataTipRows(end+1) = row;

    % dt = datatip(sc,Xc,Yc,Zc,'SnapToDataVertex','off');

%     title(sprintf('GT, CoG-true=(%2.0f,%2.0f,%2.0f),Stim#%d',Xc,Yc,Zc,i));
%     title(sprintf('VAE-Recon, CoG-true=(%2.0f,%2.0f,%2.0f),Stim#%d',Xc,Yc,Zc,i));
%     title(sprintf('AE-Recon, CoG-true=(%2.0f,%2.0f,%2.0f),Stim#%d',Xc,Yc,Zc,i));
%     title(sprintf('Variational, CoG-true=(%2.0f,%2.0f,%2.0f),Stim#%d',Xc,Yc,Zc,i));

    % title('Ground Truth','FontSize',14);
%     grid on; box on;
%     view([396.2 -4.2]);
%     view([262.2 -34.9]);
    view([296.7 1.1]);
%     view([-51.3 -1.3]);
%     view([353 47]);
%     colorbar;
    caxis([0 1]);
    % title(sprintf('Sub-MD-Stim-Index-%d',i),'FontSize',14)
%     title('Sub-MD-Stim-Index','FontSize',14)
%     set(gca,'FontSize',20,'FontWeight','bold')
%     set(gca,'XColor', 'none','YColor','none','ZColor','none')

    set(gca,'visible','off')
% 
    set(gca, 'XDir','reverse');
%     set(gca, 'YDir','reverse');
    set(gca, 'ZDir','reverse')
    % zt = get(gca, 'YTick');
    % % set(gca, 'YTick',zt, 'YTickLabel',fliplr(zt))
    % 
    % % clear all; fliplr()
    % 
    % % subplot(2,2,1);
    % % pcolor(Weights_RMT_120_130_140)
    % % titl
    % % colormap('parula')
    % % k=[3,4,5,10];   

end   
% 
% %     s = q - t;    
% %     sim = find(s>0)