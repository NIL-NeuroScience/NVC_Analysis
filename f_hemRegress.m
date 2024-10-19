function [rMat,residual] = f_hemRegress(sig,reg,brain_mask)
% used to perform pixel-by-pixel linear regression between sig and reg
% matrices. m*X = Y, m = X \ Y
% Inputs:
%   sig (Y, regressand) - mesoscopic data video (3D matrix)
%   reg (X, regressor) - mesoscopic data video(s) (fourth dimension is used
%       for multiple regressors)
%   brain_mask - mask to define cortical exposure (makes calculation
%       quicker)
% Outputs:
%   rMat (m) - matrix of linear regression coefficients (cortical map(s))
%   residual - residual of linear regression predicition
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dim = size(sig);

sig = reshape(sig,[],dim(3));
reg = reshape(reg,dim(1)*dim(2),dim(3),[]);
nanIdx = isnan(brain_mask(:));

sig = sig(~nanIdx,:);
sig = sig';
reg = reg(~nanIdx,:,:);
reg = permute(reg,[2 3 1]);

N = size(sig,2);

rMat = zeros(N,size(reg,2));

for idx = 1:N
    rMat(idx,:) = reg(:,:,idx) \ sig(:,idx);
end

residual = squeeze(sum(reg.*permute(rMat,[3 2 1]),2));
residual = (sig-residual)';

nanMat = NaN(dim(1)*dim(2),dim(3));
nanMat(~nanIdx,:) = residual;
residual = reshape(nanMat,dim(1),dim(2),dim(3));

nanMat = NaN(dim(1)*dim(2),size(reg,2));
nanMat(~nanIdx,:) = rMat;
rMat = reshape(nanMat,dim(1),dim(2),[]);
