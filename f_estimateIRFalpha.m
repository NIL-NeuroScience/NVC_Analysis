function [IRF,params,predicted] = f_estimateIRFalpha(rfp_HD,HbT,fs,l,brain_mask,ds)
% used to estimate the Impulse Response Function (IRF) between two signals
% Inputs:
%   rfp_HD - Ca2+ mesoscopic video
%   HbT - HbT mesoscopic video
%   fs - acquisition framerate
%   l - length of desired IRF
%   brain_mask - mask to define cortical exposure (makes calculation
%   ds - downsampling kernel size
% Outputs:
%   IRF - the IRF vector (length of l)
%   params - IRF parameters [t0,tau1,tau2,A,B]
%   predicted - convolution of rfp_HD with IRF (predicted HbT)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize design matrices
rfp_HD = f_downsample(rfp_HD,ds);
HbT = f_downsample(HbT,ds);
brain_mask = f_downsample(brain_mask,ds);

dim = size(rfp_HD);
N = sum(brain_mask,[1,2],'omitnan');

N_IRF = l*fs;

rfp_HD = rfp_HD.*brain_mask;
HbT = HbT.*brain_mask;
rfp_HD = reshape(rfp_HD,dim(1)*dim(2),dim(3));
HbT = reshape(HbT,dim(1)*dim(2),dim(3));
nanIdx = isnan(rfp_HD(:,1));
rfp_HD = rfp_HD(~nanIdx,:)';
HbT = HbT(~nanIdx,:)';
HbT = f_bpf(HbT,[0 0.5],10);

design1 = zeros(dim(3),N,N_IRF);
design2 = HbT(:);

for lIdx = 1:N_IRF
    design1(lIdx:end,:,lIdx) = rfp_HD(1:end-lIdx+1,:);
end

design1 = reshape(design1,[],N_IRF);

t0 = 0.1774;
tau1 = 0.4289;
tau2 = 0.4279;
A = -805.5;
B = 808.3;

fun = @(params)f_hrf_cost_func(params(1),params(2),params(3),params(4),params(5),fs,l,design1,design2);

options = optimset('MaxFunEvals',25000,'MaxIter',10000,'Display','iter');
params = fminsearch(fun,[t0(1), tau1(1), tau2(1), A(1), B(1)],options);
params(1) = abs(params(1));

IRF = f_alpha_IRF(params(1),params(2),params(3),params(4),params(5),fs,l);

% calc predicted HbT
predicted = sum(design1.*IRF',2);
predicted = reshape(predicted,dim(3),[]);
tmp = NaN(dim(1)*dim(2),dim(3));
tmp(~nanIdx,:) = predicted';
predicted = reshape(tmp,dim(1),dim(2),[]);

%% Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    function J = f_hrf_cost_func(t0,tau1,tau2,A,B,sr,hrf_l,X_mat,y)
    
        t0 = abs(t0);
        numHRF = floor(hrf_l*sr);
        tr = (((0:numHRF-1)/sr)-t0)';
        D = ((tr)./tau1).^3 .* exp(-(tr)./tau1);
        D(tr<0) = 0;
        C = ((tr)./tau2).^3 .* exp(-(tr)./tau2);
        C(tr<0) = 0;
        
        hrf = A*D + B*C;
        
        conv_result = X_mat*hrf;
        J = norm(y - conv_result)^2;
    
    end
    
    function IRF = f_alpha_IRF(t0,tau1,tau2,A,B,sr,hrf_l)
    
        numHRF = floor(hrf_l*sr);
        tr = (((0:numHRF-1)/sr)-t0)';
        D = ((tr)./tau1).^3 .* exp(-(tr)./tau1);
        D(tr<0) = 0;
        C = ((tr)./tau2).^3 .* exp(-(tr)./tau2);
        C(tr<0) = 0;
        
        IRF = A*D + B*C;
    
    end
end

