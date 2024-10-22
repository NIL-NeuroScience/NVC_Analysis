function [gfp_HD,rfp_HD,HbO,HbR,fs,brain_mask,parcellation,pupil,whisking,Acc] = f_loadNWB(nwb)
% used to load relevant data from NWB files downloaded from the following
% link: https://dandiarchive.org/dandiset/001211/draft
% Inputs:
%   nwb - either a pointer to the nwb file or the loaded nwb file
% Outputs:
%   gfp_HD - processed GRAB fluorescent widefield movie
%   rfp_HD - processed jRGECO fluorescent widefield movie
%   HbO - processed HbO widefield movie
%   HbR - processed HbR widefield movie
%   brain_mask - mask of cortical exposure
%   parcellation - allen atlas parcellation
%   pupil - pupil diameter behavioral readout (relative to total eye
%   diameter
%   whisking - whisking motion energy behavioral readout
%   Acc - accelerometer movement readout (V)
% Notes:
%   Variable parcellation consists of 12 masks in the order:
%   MOp, MOs, SSp-bfd, SSp-tr, SSp-ll, SSp-ul, SSp-un, VISpm, VISrl, VISam,
%   VISa, VISp
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isstring(nwb) || ischar(nwb)
    nwb = nwbRead(nwb,'ignorecache');
end

gfp_HD = nwb.acquisition.get('gfp_HD').data.load;
rfp_HD = nwb.acquisition.get('rfp_HD').data.load;
HbO = nwb.acquisition.get('HbO').data.load;
HbR = nwb.acquisition.get('HbR').data.load;

fs = nwb.acquisition.get('HbR').starting_time_rate;

%% extract brain_mask and parcellation
tmpROIs = nwb.processing.get('ophys').nwbdatainterface.get('ImageSegmentation').planesegmentation.get('PlaneSegmentation');
tmppixel_mask_index = tmpROIs.pixel_mask_index.data.load;
tmppixel_mask = tmpROIs.pixel_mask.data.load;

tmpsteps = [0;double(tmppixel_mask_index)];

masks = zeros(size(gfp_HD,1),size(gfp_HD,2),13);
for idx = 1:13
    tmpIdx = sub2ind([size(gfp_HD,1),size(gfp_HD,2)],tmppixel_mask.y(tmpsteps(idx)+1:tmpsteps(idx+1)),tmppixel_mask.x(tmpsteps(idx)+1:tmpsteps(idx+1)));
    tmpMask = zeros(size(gfp_HD,1),size(gfp_HD,2));
    tmpMask(tmpIdx) = 1;
    masks(:,:,idx) = tmpMask;
end

brain_mask = masks(:,:,1);
brain_mask(brain_mask==0) = NaN;
parcellation = masks(:,:,2:end);

%% extract behavioral readouts 
behavior = nwb.processing.get('behavior').nwbdatainterface.get('BehavioralTimeSeries');
pupil = behavior.timeseries.get('pupil').data.load;
whisking = behavior.timeseries.get('whisking').data.load;
Acc = behavior.timeseries.get('accelerometer').data.load;

clear tmp* idx nwb behavior masks