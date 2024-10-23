%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                           NVC Analysis
%
% Version 0.1 BCR 07/20/2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rfp_HD - hemodynamic corrected Ca2+ fluorescence data
% gfp_HD - hemodynamic corrected GRAB fluorescence data
% HbO - oxygenated hemoglobin fluctuations
% HbR - deoxygenated hemoglobin fluctuations
% brain_mask - mask used to define cortical exposure
% parcellation - allen atlas parcellation structure
% fs - acquisition frequency (10 Hz)
% pupil - pupil diameter time course

HbT = HbO+HbR;

%% analysis settings %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

aSet = struct;
aSet.ds = 4; % downsampling kernel
aSet.IRF1.kernel = 7; % IRF1 kernel (s)
aSet.IRF1.corrGram_win = [15 1.5]; % sliding correlation window [win size, step size] (s)
aSet.IRF2.kernel = [-3 7]; % IRF2 kernel range (s)

%% spatially downsample the data

ds = struct;
ds.brain_mask = f_downsample(brain_mask,aSet.ds);
ds.rfp_HD = f_downsample(rfp_HD,aSet.ds);
ds.gfp_HD = f_downsample(gfp_HD,aSet.ds);
ds.HbT = f_downsample(HbT,aSet.ds);
ds.nanIdx = isnan(ds.brain_mask);
ds.dim = size(ds.rfp_HD);

%% Figure 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% estimate stationary IRF using downsampled data

IRF1 = struct;
[IRF1.IRF,IRF1.params,IRF1.pred] = f_estimateIRFalpha(ds.rfp_HD./std(ds.rfp_HD,0,3),ds.HbT./std(ds.HbT,0,3),fs,aSet.IRF1.kernel,ds.brain_mask,1); % estimate IRF for entire exposure

mask = f_downsample(parcellation,aSet.ds); % combine SSp-tr, SSp-ll, and SSp-ul allen ROIs
mask = sum(mask(:,:,4:6),3,'omitnan');
mask(mask==0) = NaN;

[IRF1.SM.IRF,IRF1.SM.params] = f_estimateIRFalpha(ds.rfp_HD./std(ds.rfp_HD,0,3),ds.HbT./std(ds.HbT,0,3),fs,aSet.IRF1.kernel,mask,1); % estimate IRF for medial SSp regions

% calculate stationary IRF accuracy (global)

dim = size(rfp_HD);
rfp = reshape(rfp_HD,[],dim(3));
nanIdx = isnan(brain_mask);
rfp = rfp(~nanIdx,:)';

IRF1.pred_full = zeros(dim(3)+fs*aSet.IRF1.kernel-1,size(rfp,2));
for idx = 1:size(rfp,2)
    IRF1.pred_full(:,idx) = conv(rfp(:,idx),IRF1.IRF);
end
IRF1.pred_full = IRF1.pred_full(1:dim(3),:);

tmp = NaN(dim(1)*dim(2),dim(3));
tmp(~nanIdx,:) = IRF1.pred_full';
IRF1.pred_full = reshape(tmp,dim(1),dim(2),[]);

% calculate stationary IRF accuracy (SSp)

IRF1.SM.pred_full = zeros(dim(3)+fs*aSet.IRF1.kernel-1,size(rfp,2));
for idx = 1:size(rfp,2)
    IRF1.SM.pred_full(:,idx) = conv(rfp(:,idx),IRF1.SM.IRF);
end
IRF1.SM.pred_full = IRF1.SM.pred_full(1:dim(3),:);

tmp = NaN(dim(1)*dim(2),dim(3));
tmp(~nanIdx,:) = IRF1.SM.pred_full';
IRF1.SM.pred_full = reshape(tmp,dim(1),dim(2),[]);

IRF1.perf_full = brain_mask.*f_HemCorr(HbT,IRF1.pred_full); % correlation map
IRF1.SM.perf_full = brain_mask.*f_HemCorr(HbT,IRF1.SM.pred_full); % correlation map

% 1x IRF performance over time
IRF1.corrGram = f_HemCorrGram(ds.HbT,IRF1.pred,aSet.IRF1.corrGram_win*fs);
IRF1.corrGram_full = f_HemCorrGram(HbT,IRF1.pred_full,aSet.IRF1.corrGram_win*fs);
IRF1.SM.corrGram_full = f_HemCorrGram(HbT,IRF1.SM.pred_full,aSet.IRF1.corrGram_win*fs);
IRF1.perf_dt = squeeze(mean(IRF1.corrGram,[1 2],'omitnan'));

% 1x IRF performance vs. arousal

IRF1.arousal_idx = aSet.IRF1.corrGram_win(1)*fs/2:aSet.IRF1.corrGram_win(2)*fs:ds.dim(3)-aSet.IRF1.corrGram_win(1)*fs/2;
IRF1.pupil = movmean(pupil,aSet.IRF1.corrGram_win*fs);
IRF1.pupil = IRF1.pupil(IRF1.arousal_idx)';

IRF1.GRAB = movmean(gfp_HD.*brain_mask,aSet.IRF1.corrGram_win*fs,3);
IRF1.GRAB = IRF1.GRAB(:,:,IRF1.arousal_idx);

%% plot figures for Figure 1

imAlpha = brain_mask;
imAlpha(isnan(imAlpha)) = 0;

% figure 1C
f = figure; 
plot(0:0.1:6.9,IRF1.IRF);
xlim([0 6.9]);
xlabel('Time (s)');
ylabel('a.u.');
title('Global IRF');
box off;

f = figure;
imagesc(IRF1.perf_full,'AlphaData',imAlpha);
colormap jet;
caxis([0 1]);
c = colorbar;
c.Label.String = 'r';
title('IRF Performance - Global');
axis off image;

% figure 1D
f = figure; 
plot(0:0.1:6.9,IRF1.SM.IRF);
xlim([0 6.9]);
xlabel('Time (s)');
ylabel('a.u.');
title('sm IRF');
box off;

f = figure;
imagesc(IRF1.SM.perf_full,'AlphaData',imAlpha);
colormap jet;
caxis([0 1]);
c = colorbar;
c.Label.String = 'r';
title('IRF Performance - SM');
axis off image;

% figure 1F
SM_IRF_perf = squeeze(mean(parcellation(:,:,3).*IRF1.SM.corrGram_full,[1 2],'omitnan'));
f = figure;hold on;
scatter(IRF1.pupil,SM_IRF_perf,50,'filled','MarkerFaceAlpha',0.2,'MarkerFaceColor',[0.7 0 0.7]);
mdl = fitlm(IRF1.pupil,SM_IRF_perf);
mdl = table2array(mdl.Coefficients);
plot([0.15 0.4],[0.15 0.4]*mdl(2,1)+mdl(1,1),'Color',[0.7 0 0.7],'LineWidth',2);
xlim([0.15 0.4]);
ylim([-1 1]);
xlabel('Pupil Diameter');
ylabel('IRF Performance');

NE = squeeze(mean(brain_mask.*IRF1.GRAB,[1 2],'omitnan'));
f = figure;hold on;
scatter(NE,SM_IRF_perf,50,'filled','MarkerFaceAlpha',0.2,'MarkerFaceColor',[0.4471 0.7529 0.0314]);
mdl = fitlm(NE,SM_IRF_perf);
mdl = table2array(mdl.Coefficients);
plot([-4 4],[-4 4]*mdl(2,1)+mdl(1,1),'Color',[0.4471 0.7529 0.0314],'LineWidth',2);
xlim([-4 4]);
ylim([-1 1]);
xlabel('Norepinephrine (\DeltaF/F)');
ylabel('IRF Performance');

%% Figure 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% normalized linear regression model

norm = struct;
LR = struct;
LR.lag = [9 1];

norm.low.rfp_HD = f_bpf(rfp_HD,[0 0.5],10,3);
norm.low.gfp_HD = f_bpf(gfp_HD,[0 0.5],10,3);
norm.low.HbT = f_bpf(HbT,[0 0.5],10,3);
norm.low.HbO = f_bpf(HbO,[0 0.5],10,3);
norm.low.HbR = f_bpf(HbR,[0 0.5],10,3);

norm.low.rfp_HD = norm.low.rfp_HD./std(norm.low.rfp_HD,0,3);
norm.low.gfp_HD = norm.low.gfp_HD./std(norm.low.gfp_HD,0,3);
norm.low.HbT = norm.low.HbT./std(norm.low.HbT,0,3);
norm.low.HbO = norm.low.HbO./std(norm.low.HbO,0,3);
norm.low.HbR = norm.low.HbR./std(norm.low.HbR,0,3);

LR.LS_full = f_hemRegress(norm.low.HbT(:,:,LR.lag(1)+1:end),cat(4,norm.low.rfp_HD(:,:,1:end-LR.lag(1)),norm.low.gfp_HD(:,:,LR.lag(1)+1-LR.lag(2):end-LR.lag(2))),brain_mask);
LR.perf_full = f_HemCorr(norm.low.HbT(:,:,1+LR.lag(1):end),LR.LS_full(:,:,1).*norm.low.rfp_HD(:,:,1:end-LR.lag(1))+LR.LS_full(:,:,2).*norm.low.gfp_HD(:,:,LR.lag(1)+1-LR.lag(2):end-LR.lag(2)));
LR.perf_full = LR.perf_full.*brain_mask;

LR.HbO.LS_full = f_hemRegress(norm.low.HbO(:,:,LR.lag(1)+1:end),cat(4,norm.low.rfp_HD(:,:,1:end-LR.lag(1)),norm.low.gfp_HD(:,:,LR.lag(1)+1-LR.lag(2):end-LR.lag(2))),brain_mask);
LR.HbO.perf_full = f_HemCorr(norm.low.HbO(:,:,1+LR.lag(1):end),LR.HbO.LS_full(:,:,1).*norm.low.rfp_HD(:,:,1:end-LR.lag(1))+LR.HbO.LS_full(:,:,2).*norm.low.gfp_HD(:,:,LR.lag(1)+1-LR.lag(2):end-LR.lag(2)));
LR.HbO.perf_full = LR.HbO.perf_full.*brain_mask;

LR.HbR.LS_full = f_hemRegress(norm.low.HbR(:,:,LR.lag(1)+1:end),cat(4,norm.low.rfp_HD(:,:,1:end-LR.lag(1)),norm.low.gfp_HD(:,:,LR.lag(1)+1-LR.lag(2):end-LR.lag(2))),brain_mask);
LR.HbR.perf_full = f_HemCorr(norm.low.HbR(:,:,1+LR.lag(1):end),LR.HbR.LS_full(:,:,1).*norm.low.rfp_HD(:,:,1:end-LR.lag(1))+LR.HbR.LS_full(:,:,2).*norm.low.gfp_HD(:,:,LR.lag(1)+1-LR.lag(2):end-LR.lag(2)));
LR.HbR.perf_full = LR.HbR.perf_full.*brain_mask;

% 2x IRF model

IRF2 = struct;
[IRF2.perf,IRF2.IRF,IRF2.LS] = f_2xDeconvolve(HbT./std(HbT,0,3),rfp_HD./std(rfp_HD,0,3),gfp_HD./std(gfp_HD,0,3),aSet.IRF2.kernel,fs,brain_mask,4);
[IRF2.HbO.perf,IRF2.HbO.IRF,IRF2.HbO.LS] = f_2xDeconvolve(HbO./std(HbO,0,3),rfp_HD./std(rfp_HD,0,3),gfp_HD./std(gfp_HD,0,3),aSet.IRF2.kernel,fs,brain_mask,4);
[IRF2.HbR.perf,IRF2.HbR.IRF,IRF2.HbR.LS] = f_2xDeconvolve(HbR./std(HbR,0,3),rfp_HD./std(rfp_HD,0,3),gfp_HD./std(gfp_HD,0,3),aSet.IRF2.kernel,fs,brain_mask,4);

%% plot figures for Figure 2

% figure 2B
f = figure;
imagesc(LR.LS_full(:,:,1),'AlphaData',imAlpha);
colormap cmpbbr;
caxis(0.65*[-1 1]);
c = colorbar;
title('Ca^2^+ LR Coefficient');
axis off image;

f = figure;
imagesc(LR.LS_full(:,:,2),'AlphaData',imAlpha);
colormap cmpbbr;
caxis(0.9*[-1 1]);
c = colorbar;
title('NE LR Coefficient');
axis off image;

% figure 2C
f = figure;
imagesc(LR.perf_full,'AlphaData',imAlpha);
colormap jet;
caxis([0 1]);
c = colorbar;
c.Label.String = 'r';
title('Linear Regression Model');
axis off image;

% figure 2D
f = figure;hold on;
scatter(IRF1.perf_full(1:10:end),LR.perf_full(1:10:end),50,'filled','MarkerFaceAlpha',0.05,'MarkerFaceColor',[230 163 0]/255);
plot([0 1],[0 1],'-k');
xlim([0 1]);
ylim([0 1]);
ylabel('LR Model (r)');
xlabel('IRF Performance');

% figure 2E
f = figure;hold on;
A = LR.LS_full(:,:,1);
scatter(IRF1.perf_full(1:10:end),A(1:10:end),50,'filled','MarkerFaceAlpha',0.05,'MarkerFaceColor',[0.8196 0.0784 0.0941]);
mdl = fitlm(IRF1.perf_full(1:10:end),A(1:10:end));
mdl = table2array(mdl.Coefficients);
plot([-0.5 0.8],[-0.5 0.8]*mdl(2,1)+mdl(1,1),'Color',[0.8196 0.0784 0.0941],'LineWidth',2);
xlim([-0.5 0.8]);
ylim([0 1]);
xlabel('IRF Performance');
ylabel('Ca^2^+ Coefficient (A)');

f = figure;hold on;
B = LR.LS_full(:,:,2);
scatter(IRF1.perf_full(1:10:end),B(1:10:end),50,'filled','MarkerFaceAlpha',0.05,'MarkerFaceColor',[0.4471 0.7529 0.0314]);
mdl = fitlm(IRF1.perf_full(1:10:end),B(1:10:end));
mdl = table2array(mdl.Coefficients);
plot([-0.5 0.8],[-0.5 0.8]*mdl(2,1)+mdl(1,1),'Color',[0.4471 0.7529 0.0314],'LineWidth',2);
xlim([-0.5 0.8]);
ylim([-1.5 0]);
xlabel('IRF Performance');
ylabel('NE Coefficient (A)');

% figure 2F
f = figure;
imagesc(IRF2.LS(:,:,1),'AlphaData',imAlpha);
colormap cmpbbr;
caxis(1.3*[-1 1]);
c = colorbar;
title('Ca^2^+ 2xIRF Coefficient');
axis off image;

f = figure;
imagesc(-IRF2.LS(:,:,2),'AlphaData',imAlpha);
colormap cmpbbr;
caxis(2*[-1 1]);
c = colorbar;
title('NE 2xIRF Coefficient');
axis off image;

% figure 2G
f = figure;
imagesc(IRF2.perf,'AlphaData',imAlpha);
colormap jet;
caxis([0 1]);
c = colorbar;
c.Label.String = 'r';
title('Double IRF Model');
axis off image;

% figure 2H
f = figure;hold on;
scatter(IRF1.perf_full(1:10:end),IRF2.perf(1:10:end),50,'filled','MarkerFaceAlpha',0.05,'MarkerFaceColor',[206 35 136]/255);
plot([0 1],[0 1],'-k');
xlim([0 1]);
ylim([0 1]);
ylabel('2x IRF Model (r)');
xlabel('IRF Performance');

%% Figure 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% downsample data
ds = struct;
ds.lag = 9;
ds.ds = 4;

ds.brain_mask = f_downsample(brain_mask,ds.ds);
ds.rfp = f_downsample(rfp_HD,ds.ds);
ds.rfp_low = f_bpf(f_downsample(rfp_HD,ds.ds),[0 0.5],fs,3);
ds.rfp_high = f_bpf(f_downsample(rfp_HD,ds.ds),[0.5 5],fs,3);
ds.HbT = f_bpf(f_downsample(HbT,ds.ds),[0 0.5],fs,3);

ds.rfp = ds.rfp(:,:,1:end-ds.lag);
ds.rfp_low = ds.rfp_low(:,:,1:end-ds.lag);
ds.rfp_high = ds.rfp_high(:,:,1:end-ds.lag);
ds.HbT = ds.HbT(:,:,ds.lag+1:end);
ds.nanIdx = isnan(ds.brain_mask);
ds.dim = size(ds.rfp_low);

%% calculate FC grams
FC_data = struct;

FC_data.fc_win = [300 30];

fcGram = struct;
fcGram.HbT = f_funConGram(ds.HbT,FC_data.fc_win,1,ds.brain_mask);
fcGram.Ca_low = f_funConGram(ds.rfp_low,FC_data.fc_win,1,ds.brain_mask);
fcGram.Ca_high = f_funConGram(ds.rfp_high,FC_data.fc_win,1,ds.brain_mask);
fcGram.Ca = f_funConGram(ds.rfp,FC_data.fc_win,1,ds.brain_mask);

%% fc correlation
fcIdx = tril(ones(size(fcGram.HbT(:,:,1)))).*(~diag(ones(1,size(fcGram.HbT,1))));
fcIdx = logical(fcIdx);

N = size(fcGram.HbT,3);
fcGram.rs.HbT = reshape(fcGram.HbT,[],N);
fcGram.rs.HbT = fcGram.rs.HbT(fcIdx(:),:);
fcGram.rs.Ca_low = reshape(fcGram.Ca_low,[],N);
fcGram.rs.Ca_low = fcGram.rs.Ca_low(fcIdx(:),:);
fcGram.rs.Ca_high = reshape(fcGram.Ca_high,[],N);
fcGram.rs.Ca_high = fcGram.rs.Ca_high(fcIdx(:),:);
fcGram.rs.Ca = reshape(fcGram.Ca,[],N);
fcGram.rs.Ca = fcGram.rs.Ca(fcIdx(:),:);

FC_data.corr.low = f_HemCorr(permute(fcGram.rs.HbT,[2 3 1]),permute(fcGram.rs.Ca_low,[2 3 1]));
FC_data.corr.high = f_HemCorr(permute(fcGram.rs.HbT,[2 3 1]),permute(fcGram.rs.Ca_high,[2 3 1]));
FC_data.corr.all = f_HemCorr(permute(fcGram.rs.HbT,[2 3 1]),permute(fcGram.rs.Ca,[2 3 1]));

%% allen region connectivity
mask = f_downsample(parcellation,ds.ds);
n = size(mask,3);

allen = struct;

allen.Ca_low = zeros(size(ds.rfp_low,3),n);
allen.Ca_high = allen.Ca_low;
allen.Ca = allen.Ca_low;
allen.HbT = allen.Ca_low;

for aIdx = 1:n
    allen.Ca_low(:,aIdx) = mean(ds.rfp_low.*mask(:,:,aIdx),[1 2],'omitnan');
    allen.Ca_high(:,aIdx) = mean(ds.rfp_high.*mask(:,:,aIdx),[1 2],'omitnan');
    allen.Ca(:,aIdx) = mean(ds.rfp.*mask(:,:,aIdx),[1 2],'omitnan');
    allen.HbT(:,aIdx) = mean(ds.HbT.*mask(:,:,aIdx),[1 2],'omitnan');
end

allen.fcGram.Ca_low = f_funConGram(permute(allen.Ca_low,[2 3 1]),FC_data.fc_win,1,ones(n,1));
allen.fcGram.Ca = f_funConGram(permute(allen.Ca,[2 3 1]),FC_data.fc_win,1,ones(n,1));
allen.fcGram.Ca_high = f_funConGram(permute(allen.Ca_high,[2 3 1]),FC_data.fc_win,1,ones(n,1));
allen.fcGram.HbT = f_funConGram(permute(allen.HbT,[2 3 1]),FC_data.fc_win,1,ones(n,1));

FC_data.NE = squeeze(mean(gfp_HD.*brain_mask,[1 2],'omitnan'));
FC_data.NE = FC_data.NE(FC_data.fc_win(1)/2:FC_data.fc_win(2):size(gfp_HD,3)-FC_data.fc_win(1)/2);
FC_data.NE(end) = [];

for y = 1:n
    for x = 1:n
        mdl = fitlm(FC_data.NE,squeeze(allen.fcGram.Ca_low(y,x,:)));
        mdl = table2array(mdl.Coefficients);
        allen.NE.Ca_low(y,x) = mdl(2,1);

        mdl = fitlm(FC_data.NE,squeeze(allen.fcGram.Ca(y,x,:)));
        mdl = table2array(mdl.Coefficients);
        allen.NE.Ca(y,x) = mdl(2,1);

        mdl = fitlm(FC_data.NE,squeeze(allen.fcGram.Ca_high(y,x,:)));
        mdl = table2array(mdl.Coefficients);
        allen.NE.Ca_high(y,x) = mdl(2,1);
        
        mdl = fitlm(FC_data.NE,squeeze(allen.fcGram.HbT(y,x,:)));
        mdl = table2array(mdl.Coefficients);
        allen.NE.HbT(y,x) = mdl(2,1);
    end
end
FC_data.allen = allen;

%% plot figures for Figure 3

% figure 3B
f = figure;hold on;
plot(15:3:582,squeeze(FC_data.allen.fcGram.Ca(2,12,:)),'Color',[0.819607843137255 0.0784313725490196 0.0941176470588235]);
plot(15:3:582,squeeze(FC_data.allen.fcGram.HbT(2,12,:)),'Color',[0.203921568627451 0.215686274509804 0.592156862745098]);
xlim([15 582]);
xlabel('Time (s)');
ylabel('r');
legend('r(Ca_M_O_s,Ca_V_I_S_p)','r(HbT_M_O_s,HbT_V_I_S_p)')

% figure 3C
f = figure;
plot(15:3:582,FC_data.corr.all,'Color',[0.870588235294118,0.576470588235294,0.019607843137255]);
xlim([15 582]);
xlabel('Time (s)');
ylabel('r');

% figure 3D
f = figure;hold on;
scatter(FC_data.NE,FC_data.corr.all,50,'filled','MarkerFaceAlpha',0.2,'MarkerFaceColor',[0.819607843137255 0.0784313725490196 0.0941176470588235]);
mdl = fitlm(FC_data.NE,FC_data.corr.all);
mdl = table2array(mdl.Coefficients);
plot([-4 4],[-4 4]*mdl(2,1)+mdl(1,1),'Color',[0.819607843137255 0.0784313725490196 0.0941176470588235],'LineWidth',2);
xlim([-4 4]);
ylim([0.2 1]);
xlabel('Norepinephrine (\DeltaF/F)');
ylabel('Correlation (R)');

% figure 3E
f = figure;
imagesc(FC_data.allen.NE.Ca);
colormap cmpbbr;
axis off image;
caxis(0.1*[-1 1]);
c = colorbar;
c.Label.String = 'R / \DeltaF/F (NE)';

% figure 3F
f = figure;
imagesc(FC_data.allen.NE.HbT);
colormap cmpbbr;
axis off image;
caxis(0.1*[-1 1]);
c = colorbar;
c.Label.String = 'R / \DeltaF/F (NE)';
