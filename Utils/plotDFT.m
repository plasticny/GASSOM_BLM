function [P1,f] = plotDFT(x,plotFlag)
%DFT  compute the Discrete Fourier Transform of N samples
% calling:  [H,W] = dft (h, N)
%	h: input vector of finite length, L 
%	N:frequency bandwidth on   (-pi, pi) with condition:  N >= L
%   W:DFT bandwidth
%   H: Frequency response
fs = 44100;
L = length (x);
Y = fft(x);
P2 = mag2db(abs(Y/L));
P1 = P2(1:L/2+1);
f = fs*(0:(L/2))/L;

if plotFlag
    figure;
    plot(f,P1) 
    title('Single-Sided Spectrum of X(t)')
    xlabel('f (Hz)')
    ylabel('dB')
end