function filt = f_bpf(sig,fr,fs,dim)

idx = fr == [0 fs/2];
idx = ~idx;

if nargin > 3
    if dim == 3
        sig = permute(sig,[3 1 2]);
    end
end


if idx
    [r,a] = butter(6,fr(1)/(fs/2));
    filt = filtfilt(r,a,sig);
    high = sig - filt;
    [r,a] = butter(6,fr(2)/(fs/2));
    filt = filtfilt(r,a,high);
elseif ~idx(1)
    [r,a] = butter(6,fr(2)/(fs/2));
    filt = filtfilt(r,a,sig);
elseif ~idx(2)
    [r,a] = butter(6,fr(1)/(fs/2));
    filt = filtfilt(r,a,sig);
    filt = sig - filt;
else
    filt = sig;
end

if nargin > 3
    if dim == 3
        filt = permute(filt,[2 3 1]);
    end
end
