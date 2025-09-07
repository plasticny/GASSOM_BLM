function [filtered_signal,b,a] = btwFilter(signal,pass_band)
%     f_nyq = 44100/2;
%     [b,a] = butter(4,pass_band/f_nyq,'bandpass');
%     filtered_signal = filter(b,a,signal);
    fs = 44100;
    hf = design(fdesign.bandpass('N,F3dB1,F3dB2',4,pass_band(1),pass_band(2),fs));
    filtered_signal = filter(hf,signal);
    b = []; a = [];
end