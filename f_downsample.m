function downsampled = f_downsample(sig,bin)
% used to spatially downsample images/videos
% Inputs:
%   sig - image/video to downsample
%   bin - downsampling kernel size
% Outputs:
%   downsampled - downsampled image/video
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if bin == 1
    downsampled = sig;
else   
    dim = size(sig);
    if size(dim,2) == 2
        dim(3) = 1;
    end
    downsampled = zeros(floor(dim(1)/bin),floor(dim(2)/bin),dim(3));
    
    for h = bin:bin:dim(1)
        for w = bin:bin:dim(2)
            downsampled(h/bin,w/bin,:) = mean(sig((h-bin+1):h,(w-bin+1):w,:),[1,2]);
        end
    end
end
