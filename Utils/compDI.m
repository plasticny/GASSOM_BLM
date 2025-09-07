function [res] = compDI(GM)
n = GM.gsm{1}.n_subspace;
nsqrt = sqrt(n);
res = zeros(1,n);

for i=1:nsqrt
    for j=1:nsqrt
        bf = GM.gsm{1}.bases{1}(:,(i-1)*nsqrt+j);
        fit_res = [];
        if i>1
            tbf = GM.gsm{1}.bases{1}(:,(i-2)*nsqrt+j);
            p = polyfit(bf,tbf,1);
            sqe = sqrt(mean((polyval(p,bf)-tbf).^2));
            fit_res = [fit_res,sqe];
        end
        if j>1
            tbf = GM.gsm{1}.bases{1}(:,(i-1)*nsqrt+j-1);
            p = polyfit(bf,tbf,1);
            sqe = sqrt(mean((polyval(p,bf)-tbf).^2));
            fit_res = [fit_res,sqe];
        end
        if i<nsqrt
            tbf = GM.gsm{1}.bases{1}(:,(i)*nsqrt+j);
            p = polyfit(bf,tbf,1);
            sqe = sqrt(mean((polyval(p,bf)-tbf).^2));
        end
        if j<nsqrt
            tbf = GM.gsm{1}.bases{1}(:,(i-1)*nsqrt+j+1);
            p = polyfit(bf,tbf,1);
            sqe = sqrt(mean((polyval(p,bf)-tbf).^2));
            fit_res = [fit_res,sqe];
        end
        res((i-1)*nsqrt+j) = mean(fit_res);
    end
end