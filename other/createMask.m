close all; clf;

% Load Both Masks
d = load('ba4_mask.mat');   %Mask representing the 64-cube
crop_mask = reshape(d.mask,[230,190,180]); 
logical_mask = logical(crop_mask);

b = load('M1_mask.mat');    %Mask representing M1 
motor_mask = double(b.mask); 

% Convert Stims Using The Masks
for i=1:300   
    c = load(sprintf('Raw_Unnormalized_Stims_1//%d.mat',i));
    r = c.a; 
    
    % Below lines for RMT-110 only
    r = permute(r,[2 1 3]);
    r = reshape(r,[190,230,180]);
    r = permute(r,[2 1 3]);
    r = reshape(r,[230,190,180]);
    r = fliplr(r);

    zone = r(logical_mask); % Extracting the 64-cube
    zone = reshape(zone,[64,64,64]);    
    zone = zone.*motor_mask; % Applying the M1 mask
    save(sprintf('Masked_All_RMT//%d.mat',i+0),'zone');
end