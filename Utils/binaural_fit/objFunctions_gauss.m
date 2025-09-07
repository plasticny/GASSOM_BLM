function [ rssOutput ] = objFunctions_gauss(params,x,y)
    
    a = params(1);
    b = params(2);
    c = params(3);
    d = params(4);
    e = params(5);
    f = params(6);
    g = params(7);
    h = params(8);
    i = params(9);
    j = params(10);
    k = params(11);
    l = params(12);
    T=x(end);
    rss1 = a.*(cos(2*pi*(x.*(1-x/(2*T)).*b+(x.^2/(2*T)).*c) + d)).*exp(-0.5*((x-i)./j).^2) + e - y(:,1);
    rss2 = f.*(cos(2*pi*(x.*(1-x/(2*T)).*b+(x.^2/(2*T)).*c) + g)).*exp(-0.5*((x-k)./l).^2) + h - y(:,2);
%     rss1 = a.*(cos(2*pi*x.*(2*b+c*(x-T))/2 + d)).*exp(-0.5*((x-i)./j).^2) + e - y(:,1);
%     rss2 = f.*(cos(2*pi*x.*(2*b+c*(x-T))/2 + g)).*exp(-0.5*((x-k)./l).^2) + h - y(:,2);
    rssOutput = [rss1; rss2]; 
end