function [fcGram,fIdx] = f_funConGram(sig,win,ds,brain_mask)
% used to calculate pixel-by-pixel functional connectivity over time
% Inputs:
%   sig - mesoscopic video
%   win - sliding FC window [win size, step size] 
%   ds - spatial downsampling kernel size
%   brain_mask - mask to define cortical exposure (makes calculation
%       quicker)
% Outputs:
%   fcGram - FC matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sig = f_downsample(sig,ds);
brain_mask = f_downsample(brain_mask,ds);

dim = size(sig);
sig = sig.*brain_mask;
sig = reshape(sig,[],dim(3));

nanIdx = isnan(sig(:,1));
sig = sig(~nanIdx,:)';

idx = 1:win(2):dim(3);
idx(idx-1+win(1) > dim(3)) = [];

fIdx = win(1)/2:win(2):dim(3)-win(1)/2;

fcGram = zeros(size(sig,2),size(sig,2),numel(idx));

for i = 1:numel(idx)
    fcGram(:,:,i) = corrcoef(sig(idx(i):idx(i)-1+win(1),:));
end