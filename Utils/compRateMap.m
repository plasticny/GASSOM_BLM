function [rmL,rmR,nRM] = compRateMap(y)
global integration_dur rm_interval;
% integration_dur = 16; % in ms, default 8
% rm_interval = 1.6; % in ms, default 10
global n_channel;
% n_channel = 30;
lf = 100;
hf = 20000;
output_type = 'log';
rmL = makeRateMap_c(y(:,1),44100,lf,hf,n_channel,rm_interval,integration_dur,output_type);
rmR = [];
if size(y,2) == 2
    rmR = makeRateMap_c(y(:,2),44100,lf,hf,n_channel,rm_interval,integration_dur,output_type);
end
nRM = size(rmL,2);
end
