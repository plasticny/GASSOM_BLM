function [fRange] = compFreqRange(center_freq,bw_octave)
bw_Hz = 2^(bw_octave/2);
fRange = [center_freq/(bw_Hz) center_freq*bw_Hz];
end