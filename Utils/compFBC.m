function [fbs_rate,fbs] = compFBC(locs_true,locs_pred)
% compute Front-back Confusion Score for cell data
f = cellfun(@fberr,locs_true,locs_pred,'UniformOutput',true);
s = cellfun(@fbscore,locs_true,locs_pred,'UniformOutput',true);
fbs = double(f).*s;
fbs_rate = mean(fbs,1);
end

function f = fberr(x,y)
f = (x(2)~=y(2));
end

function s = fbscore(x,y)
if x(1)<y(1)
    s = (90-y(1))/(90-x(1));
else
    s = (y(1)+90)/(x(1)+90);
end
end