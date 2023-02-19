 close all; clc; 
%  clearvars;

% for i=2:12:47
for i=[110]
    
    figure; 

%     b = load(sprintf('X_test_nz_nstim_%d.mat',i)); 
%     r=b.X_test_nz_nstim; 
    r=X_Var;
    [x,y,z]=ind2sub(size(r),find(r>0));
    r=reshape(r,[262144,1]);
    k=find(r>0);
    r=r(k);
%     subplot(2,3,1);
    scatter3(x,y,z,80,r/55,'filled');
    grid on; box on;
%     view([353 47]);
    colorbar;
    title(sprintf('GT-Test (Trial = %d)',i),'FontSize',14)
    caxis([0 18.18e-3]);
    set(gca,'FontSize',20,'FontWeight','bold')
    
%     b = load('Y_test_f3.mat'); 
%     q = b.Y_test_f3(i,:);
% %     subplot(2,3,4);
%     bar(q);
%     ylim([0 1]);
%     set(gca,'FontSize',20,'FontWeight','bold');
% %     grid on;
% %     title('Normalized MEPs','FontSize',14);
  
%     b = load(sprintf('X_VAE_nz_nstim_Map_3d_bT_%d.mat',i));
%     r=b.X_VAE_nz_nstim_Map_3d_bT;        
%     [x,y,z]=ind2sub(size(r),find(r>0));
%     r=reshape(r,[262144,1]);
%     k=find(r>0);
%     r=r(k);
%     subplot(2,3,2);
%     scatter3(x,y,z,80,r/55,'filled');
%     grid on; box on;
%     view([353 47]);
%     colorbar;
%     title('VAE-Fwd=Rev-(NRMSE=0.218,CS=.9974)','FontSize',14);
%     caxis([0 18.18e-3]);
%     set(gca,'FontSize',20,'FontWeight','bold')
%     
%     b = load(sprintf('diff_X_VAE_nz_nstim_Map_3d_bT_%d.mat',i));
%     r=b.diff_X_VAE_nz_nstim_Map_3d_bT;    
%     [x,y,z]=ind2sub(size(r),find(r>0));
%     r=reshape(r,[262144,1]);
%     k=find(r>0);
%     r=r(k);
%     subplot(2,3,3);
%     scatter3(x,y,z,80,r/55,'filled');
%     grid on; box on;
%     view([353 47]);
%     colorbar;
%     title('VAE-Abs-Error','FontSize',14);
%     caxis([0 18.18e-3]);
%     set(gca,'FontSize',20,'FontWeight','bold')
%        
% 
%     b = load(sprintf('X_VAE_nz_nstim_Map_2d_pre_enc_bT_%d.mat',i));
%     r=b.X_VAE_nz_nstim_Map_2d_pre_enc_bT;    
%     [x,y,z]=ind2sub(size(r),find(r>0));
%     r=reshape(r,[262144,1]);
%     k=find(r>0);
%     r=r(k);
%     subplot(2,3,5);
%     scatter3(x,y,z,80,r/55,'filled');
%     grid on; box on;
%     view([353 47]);
%     colorbar;
%     title('VAE-Best-(NRMSE=0.184,CS=.999)','FontSize',14);
%     caxis([0 18.18e-3]);
%     set(gca,'FontSize',20,'FontWeight','bold')
%     
%     b = load(sprintf('diff_X_VAE_nz_nstim_Map_2d_pre_enc_bT_%d.mat',i));
%     r=b.diff_X_VAE_nz_nstim_Map_2d_pre_enc_bT;    
%     [x,y,z]=ind2sub(size(r),find(r>0));
%     r=reshape(r,[262144,1]);
%     k=find(r>0);
%     r=r(k);
%     subplot(2,3,6);
%     scatter3(x,y,z,80,r/55,'filled');
%     grid on; box on;
%     view([353 47]);
%     colorbar;
%     title('VAE-Abs-Error','FontSize',14);
%     caxis([0 18.18e-3]);
%     set(gca,'FontSize',20,'FontWeight','bold')
end       
