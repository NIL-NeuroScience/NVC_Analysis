function [hemCorr] = f_HemCorr(Sig1,Sig2)
% used to calculate the correlation (pixel-by-pixel) between two videos
% Inputs:
%   Sig1 - first mesoscopic video
%   Sig2 - second mesoscopic video
% Outputs:
%   hemCorr - pixel-by-pixel correlation map
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dim = size(Sig1);

stds.Sig1 = std(Sig1,0,3);
stds.Sig2 = std(Sig2,0,3);

Sig1 = Sig1 - mean(Sig1,3);
Sig2 = Sig2 - mean(Sig2,3);

calc.cov = (1/(dim(3)-1))*sum((Sig1.*Sig2),3);

hemCorr = calc.cov./(stds.Sig1.*stds.Sig2);
