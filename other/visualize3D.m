 close all; clc; 
 clearvars;
 % view([307.3 5.6]); [latest]

% for i=2:12:47
for i=1:6
    
    figure; 

    b = load(sprintf('X_test_s3_f1_%d.mat',i)); 
    r=b.X_test_s3_f1; 
    [x,y,z]=ind2sub(size(r),find(r>0));
    r=reshape(r,[262144,1]);
    k=find(r>0);
    r=r(k);
    max(r)
    subplot(1,3,1);
    scatter3(x,y,z,40,r,'filled');
    grid on; box on;
    view([296.7 1.1]);
%     colorbar;
    title(sprintf('GT-%d)',i),'FontSize',14)
    caxis([0 1]);
%     colorbar;
    set(gca,'visible','off')
    set(gca, 'XDir','reverse');
    set(gca, 'ZDir','reverse')
    
%     b = load('Y_test_f3.mat'); 
%     q = b.Y_test_f3(i,:);
% %     subplot(2,3,4);
%     bar(q);
%     ylim([0 1]);
%     set(gca,'FontSize',20,'FontWeight','bold');
% %     grid on;
% %     title('Normalized MEPs','FontSize',14);
  
    b = load(sprintf('X_recon_s3_f1_%d.mat',i));
    r=b.X_recon_s3_f1;        
    [x,y,z]=ind2sub(size(r),find(r>0));
    r=reshape(r,[262144,1]);
    k=find(r>0);
    r=r(k);
    max(r)
    subplot(1,3,2);
    scatter3(x,y,z,40,r,'filled');
    grid on; box on;
    view([296.7 1.1]);
    title('Recon','FontSize',14);
    caxis([0 1]);
%     colorbar;
    set(gca,'visible','off')
    set(gca, 'XDir','reverse');
    set(gca, 'ZDir','reverse')
    
    b = load(sprintf('X_diff_s3_f1_%d.mat',i));
    r=b.X_diff_s3_f1;    
    [x,y,z]=ind2sub(size(r),find(r>0));
    r=reshape(r,[262144,1]);
    k=find(r>0);
    r=r(k);
    max(r)
    subplot(1,3,3);
    scatter3(x,y,z,40,r,'filled');
    grid on; box on;
    view([296.7 1.1]);
%     colorbar;
    title('Abs-Error','FontSize',14);
%     caxis([0 18.18e-3]);
    caxis([0 1]);
%     colorbar;
    set(gca,'visible','off')
    set(gca, 'XDir','reverse');
    set(gca, 'ZDir','reverse')
end       
