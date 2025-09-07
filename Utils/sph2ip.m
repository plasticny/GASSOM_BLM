function [out] = sph2ip(in)
% transfer spherical coordinate to interaural polar
out = zeros(size(in));
% n_angle = size(in,2);
% for i_angle = 1:n_angle
%     if in(1,i_angle) < 90
%         out(1,i_angle) = in(1,i_angle);
%     elseif in(1,i_angle) < 180
%         out(1,i_angle) = 180 - in(1,i_angle);
%         out(2,i_angle) = 180;
%     elseif in(1,i_angle) < 270
%         out(1,i_angle) = -(in(1,i_angle)-180);
%         out(2,i_angle) = 180;
%     else
%         out(1,i_angle) = -(360-in(1,i_angle));
%     end
% end
out(1,:) = rad2deg(asin(sin(deg2rad(in(1,:))).*cos(deg2rad(in(2,:)))));
out(2,:) = rad2deg(atan2(tan(deg2rad(in(2,:))),cos(deg2rad(in(1,:)))));
end