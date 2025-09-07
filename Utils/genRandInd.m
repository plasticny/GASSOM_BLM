
function [inds] = genRandInd(ind_cell,n_iter)
tl = cell2mat(ind_cell);
thresh_ind = find(tl>44100*2);
tiLen = length(thresh_ind);
inds = thresh_ind(randi(tiLen,n_iter,1));
end