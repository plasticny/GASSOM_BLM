function [rand_bgn] = genRandBegin(length_list,inds,audio_len)
rand_bgn = zeros(length(inds),1);
% thresh_ind = find(tl>44100*3);
% tiLen = length(thresh_ind);
for ind = 1:length(inds)
    rand_bgn(ind,1) = randi(floor(length_list(inds(ind)))-audio_len);
end
end