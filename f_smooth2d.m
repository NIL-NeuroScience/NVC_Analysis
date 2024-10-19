function im_out = f_smooth2d(im,smoothscale)  
% used to spatially smooth mesoscopic data videos
% Inputs:
%   im - mesoscopic data video (3D matrix)
%   smoothscale - size of smoothing kernel
% Ouput:
%   im_out - smoothed mesoscopic data video
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nrows = size(im,1);
ncols = size(im,2);

filtim = ((1:nrows-ceil((nrows+1)/2))'*ones(1,ncols)).^2+(ones(nrows,1)*1:ncols-ceil((ncols+1)/2)).^2;
filtim = exp(-smoothscale^2*filtim/max(filtim(:)));
tmp = fftshift(fftshift(fft(fft(squeeze(im(:,:,:)),[],1),[],2),2),1);

for fn = 1:size(tmp,3)
  tmp(:,:,fn) = squeeze(tmp(:,:,fn)).*filtim;
end

im_out = real(ifft(ifft(ifftshift(ifftshift(tmp,2),1),[],2),[],1));
