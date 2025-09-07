function [frmL,frmR,nFrm] = makeSpec(y)
[ss1,ff,tt] = spectrogram(y(:,1),hamming(round(16e-3*44100)),round(0.9*16e-3*44100),linspace(100,22000,30),44100);
frmL = 10*log10(real(ss1.*conj(ss1)));
frmR = [];
if size(y,2)>1
    [ss2,ff,tt] = spectrogram(y(:,2),hamming(round(16e-3*44100)),round(0.9*16e-3*44100),linspace(100,22000,30),44100);
    frmR = 10*log10(real(ss2.*conj(ss2)));
end
nFrm = size(frmL,2);
end