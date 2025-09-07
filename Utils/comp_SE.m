function square_error = comp_SE(c1,c2)
% compute square error
func_sq = @(x,y) sum((x-y).^2);
square_error = cellfun(func_sq,c1,c2,'UniformOutput',true);
end