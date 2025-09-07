function [rmL,rmR,nRM] = makeRM(y)
if exist('rm_config.mat','file')
    load rm_config.mat;
else
    rm_int_dur = 8;
    rm_interval = 0.1;
end
% rm_int_dur = 8;
rmL = makeRateMap_c(y(:,1),44100,100,22000,30,rm_int_dur*rm_interval,rm_int_dur,'log');
rmR = [];
if size(y,2) == 2
    rmR = makeRateMap_c(y(:,2),44100,100,22000,30,rm_int_dur*rm_interval,rm_int_dur,'log');
end
nRM = size(rmL,2);
end
