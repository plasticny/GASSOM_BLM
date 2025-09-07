function plot2Ch(x,ax_handle)
if nargin<2, figure; ax_handle = gca; end
if isequal(size(x,1),1), x = x'; end
x = reshape(x,[],2);
plot(ax_handle,x);
end
