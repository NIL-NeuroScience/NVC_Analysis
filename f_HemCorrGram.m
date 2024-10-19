function [hemCorr] = f_HemCorrGram(Sig1,Sig2,win)
% used to calculate correlation over time between two mesoscopic videos 
% (3D Matrices) Sig1 and Sig2 
% Inputs:
%   Sig1 - first mesoscopic video
%   Sig2 - second mesoscopic video
%   win - sliding correlation window [win size, step size]
% Output:
%   hemCorr - pixel-by-pixel correlation over time (3D matrix)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dim = size(Sig1);

ind = 1:win(2):dim(3);
ind((ind+win(1)-1)>dim(3)) = [];

frames = size(ind,2);
tSig1 = zeros(dim(1)*frames,dim(2),win(1));
tSig2 = zeros(dim(1)*frames,dim(2),win(1));

for i = 1:frames
    tSig1(((i-1)*dim(1)+1):(i*dim(1)),:,:) = Sig1(:,:,ind(i):(ind(i)+win(1)-1));
    tSig2(((i-1)*dim(1)+1):(i*dim(1)),:,:) = Sig2(:,:,ind(i):(ind(i)+win(1)-1));
end

stds.tSig1 = std(tSig1,0,3);
stds.tSig2 = std(tSig2,0,3);

tSig1 = tSig1 - mean(tSig1,3);
tSig2 = tSig2 - mean(tSig2,3);

calc.cov = (1/win(1))*sum((tSig1.*tSig2),3);

hemCorr = calc.cov./(stds.tSig1.*stds.tSig2);
hemCorr = hemCorr';
hemCorr = reshape(hemCorr,[dim(2),dim(1),frames]);
hemCorr = permute(hemCorr,[2 1 3]);
