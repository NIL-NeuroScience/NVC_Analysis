function [perf,IRF,LS] = f_2xDeconvolve(signal,reg1,reg2,win,fs,brain_mask,ds_factor)

% process inputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ds.signal = f_downsample(signal,ds_factor);
ds.reg1 = f_downsample(reg1,ds_factor);
ds.reg2 = f_downsample(reg2,ds_factor);
ds.brain_mask = f_downsample(brain_mask,ds_factor);

dim = size(ds.signal);
T = dim(3);
nanIdx = isnan(ds.brain_mask);
N = sum(~nanIdx,[1 2]);

ds.signal = reshape(ds.signal,dim(1)*dim(2),[]);
ds.signal = ds.signal(~nanIdx,:)';
ds.reg1 = reshape(ds.reg1,dim(1)*dim(2),[]);
ds.reg1 = ds.reg1(~nanIdx,:)';
ds.reg2 = reshape(ds.reg2,dim(1)*dim(2),[]);
ds.reg2 = ds.reg2(~nanIdx,:)';

ds.signal = f_bpf(ds.signal,[0 0.5],fs);

% setup deconvolution

l_irf = fs*range(win)+1;
idx_irf = win(1)*fs:win(2)*fs;

i1 = abs(min([idx_irf;zeros(1,numel(idx_irf))],[],1))+1;
i2 = T-max([idx_irf;zeros(1,numel(idx_irf))],[],1);
i3 = max([idx_irf;zeros(1,numel(idx_irf))],[],1)+1;
i4 = T-i1+1;

Ca_mat = zeros(T,N,l_irf);
NE_mat = zeros(T,N,l_irf);

for v = 1:l_irf
    Ca_mat(i3(v):i4(v),:,v) = ds.reg1(i1(v):i2(v),:);
end
for v = 1:l_irf
    NE_mat(i3(v):i4(v),:,v) = ds.reg2(i1(v):i2(v),:);
end

Ca_mat = reshape(Ca_mat,[],l_irf);
NE_mat = reshape(NE_mat,[],l_irf);
HbT_mat = ds.signal(:);

design_matrix = [Ca_mat NE_mat];

IRF = design_matrix \ HbT_mat;
IRF = reshape(IRF,[],2);
IRF = IRF ./ sum(abs(IRF),1);

% predict HbT
clear ds

nanIdx = isnan(brain_mask);
dim = size(signal);
Ca_mat = reshape(reg1,[],dim(3));
Ca_mat = Ca_mat(~nanIdx,:)';
NE_mat = reshape(reg2,[],dim(3));
NE_mat = NE_mat(~nanIdx,:)';

N = size(Ca_mat,2);
irf_Idx = abs(-win(1)*fs+1:dim(3)-win(1)*fs);

for idx = 1:N
    tmp = conv(Ca_mat(:,idx),IRF(:,1));
    Ca_mat(:,idx) = tmp(irf_Idx);
    tmp = conv(NE_mat(:,idx),IRF(:,2));
    NE_mat(:,idx) = tmp(irf_Idx);
end

tmp_mat = nan(dim(1)*dim(2),T);
tmp_mat(~nanIdx,:) = Ca_mat';
Ca_mat = reshape(tmp_mat,dim(1),dim(2),T);
tmp_mat(~nanIdx,:) = NE_mat';
NE_mat = reshape(tmp_mat,dim(1),dim(2),T);

LS = f_hemRegress(signal,cat(4,Ca_mat,NE_mat),brain_mask);

pred_HbT = Ca_mat.*LS(:,:,1) + NE_mat.*LS(:,:,2);
perf = f_HemCorr(signal,pred_HbT);

% % process inputs
% 
% signal = f_downsample(signal,ds);
% reg1 = f_downsample(reg1,ds);
% reg2 = f_downsample(reg2,ds);
% ds_brain_mask = f_downsample(brain_mask,ds);
% 
% ds_signal = signal;
% 
% dim = size(signal);
% T = dim(3);
% nanIdx = isnan(ds_brain_mask);
% N = sum(~nanIdx,[1 2]);
% 
% signal = reshape(signal,dim(1)*dim(2),[]);
% signal = signal(~nanIdx,:)';
% reg1 = reshape(reg1,dim(1)*dim(2),[]);
% reg1 = reg1(~nanIdx,:)';
% reg2 = reshape(reg2,dim(1)*dim(2),[]);
% reg2 = reg2(~nanIdx,:)';
% 
% signal = f_lpf(signal,0.5,fs);
% 
% % setup deconvolution
% 
% l_irf = fs*range(win)+1;
% idx_irf = win(1)*fs:win(2)*fs;
% 
% i1 = abs(min([idx_irf;zeros(1,numel(idx_irf))],[],1))+1;
% i2 = T-max([idx_irf;zeros(1,numel(idx_irf))],[],1);
% i3 = max([idx_irf;zeros(1,numel(idx_irf))],[],1)+1;
% i4 = T-i1+1;
% 
% Ca_mat = zeros(T,N,l_irf);
% NE_mat = zeros(T,N,l_irf);
% 
% for v = 1:l_irf
%     Ca_mat(i3(v):i4(v),:,v) = reg1(i1(v):i2(v),:);
% end
% for v = 1:l_irf
%     NE_mat(i3(v):i4(v),:,v) = reg2(i1(v):i2(v),:);
% end
% 
% Ca_mat = reshape(Ca_mat,[],l_irf);
% NE_mat = reshape(NE_mat,[],l_irf);
% HbT_mat = signal(:);
% 
% design_matrix = [Ca_mat NE_mat];
% 
% IRF = design_matrix \ HbT_mat;
% IRF = reshape(IRF,[],2);
% IRF = IRF ./ sum(abs(IRF),1);
% 
% Ca_mat = Ca_mat * IRF(:,1);
% Ca_mat = reshape(Ca_mat,T,[]);
% 
% NE_mat = NE_mat * IRF(:,2);
% NE_mat = reshape(NE_mat,T,[]);
% 
% Ca_mat = Ca_mat';
% NE_mat = NE_mat';
% 
% tmp_mat = nan(dim(1)*dim(2),T);
% tmp_mat(~nanIdx,:) = Ca_mat;
% Ca_mat = reshape(tmp_mat,dim(1),dim(2),T);
% tmp_mat(~nanIdx,:) = NE_mat;
% NE_mat = reshape(tmp_mat,dim(1),dim(2),T);
% 
% LS = f_hemRegress(ds_signal,cat(4,Ca_mat,NE_mat),ds_brain_mask);
% 
% pred_HbT = Ca_mat.*LS(:,:,1) + NE_mat.*LS(:,:,2);
% perf = f_HemCorr(ds_signal,pred_HbT);
