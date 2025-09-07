function [all_comb,all_comb_cell] = find_combination(N)
A = ones(N);
A2 = triu(A,1);
A3 = find(A2);
[A41,A42] = ind2sub(size(A2),A3);
all_comb = [A41,A42];
all_comb_cell = mat2cell(all_comb,ones(1,15))';
end