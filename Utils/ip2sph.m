function [out] = ip2sph(in)
% transfer interaural polar to spherical coordinate
out = zeros(size(in));
out(1,:) = rad2deg(atan2(tan(deg2rad(in(1,:))),cos(deg2rad(in(2,:)))));
out(2,:) = rad2deg(asin(cos(deg2rad(in(1,:))).*sin(deg2rad(in(2,:)))));
end