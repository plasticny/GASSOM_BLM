function [s,err]=fit_fm_together_with_gauss(d,sub,model,plot_flag)
    x = d(:,1);
    y = d(:,2);
    z = d(:,3);
    
    yu = max(y);
    yl = min(y);
    yr = (yu-yl);                                                                   % Range of â€˜yâ€™
    yz = y-yu+(yr/2);
    zx = x(yz .* circshift(yz,[1 0]) <= 0);                                         % Find zero-crossings
    per = 2*mean(diff(zx));                                                         % Estimate period
    ym = mean(y);                                                                   % Estimate offset
    [ ay,f0y ] = get_fundamental_freq( y );
    zu = max(z);
    zl = min(z);
    zr = (zu-zl);                                                                   % Range of â€˜yâ€™
    zz = z-zu+(zr/2);
    zx_z = x(zz .* circshift(zz,[1 0]) <= 0);                                         % Find zero-crossings
    per_z = 2*mean(diff(zx_z));                                                         % Estimate period
    zm = mean(z);
    [ az,f0z ] = get_fundamental_freq( z );
    [~,max_fund]=max([ay;az]);
    fund_freq=[f0y;f0z];
    
    [ e_freq ] = est_freq_bf( model,sub );
    f0=e_freq; %fund_freq(max_fund);
    
    T=x(end);
    [~,t_max_index_y]=max(abs(y));
    [~,t_min_index_y]=min(abs(y));
    [~,t_max_index_z]=max(abs(z));
    [~,t_min_index_z]=min(abs(z));
    ptp_y=x(t_max_index_y);
    ptp_z=x(t_max_index_z);
    init_sigma_y=abs(x(t_max_index_y)-x(t_min_index_y))/3;
    init_sigma_z=abs(x(t_max_index_z)-x(t_min_index_z))/3;
    
    fit = @(b,x)  b(1).*(cos(2*pi*(x.*(1-x/(2*T)).*b(2)+(x.^2/(2*T)).*b(3)) + b(4))).*exp(-0.5*((x-b(5))./b(6)).^2) + b(7);     % Function to fit
    lb=[0;100  ;100  ;-pi;-1;0;-pi;-1;0;T/6;0;T/6];
    ub=[1;20000  ;20000  ;pi ;1 ;1;pi ;1 ;T;inf;T;inf];
    all_errors=[];
    alls_s=[];
    all_yfit=[];
    all_zfit=[];
    for init_phase=0%-pi:pi/12:pi
        x0=[yr/2;f0;f0;    init_phase;ym;zr/2;   init_phase;zm;ptp_y;   init_sigma_y;ptp_z;init_sigma_z];
    %     x0=[yr/2;f0;f0;0  ;ym;zr/2;0  ;zm;T/2;init_sigma_y;T/2;init_sigma_z];
    %     fit = @(b,x)  b(1).*(cos(2*pi*x.*(2*b(2)+b(3)*(x-T))/2+ b(4))).*exp(-0.5*((x-b(5))./b(6)).^2) + b(7);     % Function to fit
    %     lb=[0   ;15000  ;-40*1e6 ;-pi;-1;0   ;-pi;-1;0    ;T;0    ;T];
    %     ub=[1   ;70000 ;10*1e6;pi ;1 ;1   ;pi ;1 ;T    ;inf  ;T    ;inf];
    %     x0=[yr/2;f0;-22.5*1e6;0  ;ym;zr/2;0  ;zm;ptp_y;init_sigma_y;ptp_z;init_sigma_z];
    %     s=lsqnonlin(@objFunctions_gauss,x0,lb,ub,[],x,[y z]);
        s=lsqnonlin(@objFunctions_gauss,x0,lb,ub,[],x,[y z]);
    %     s=fmincon(@(s) objFunctions_gauss_fmin(s,x,[y z]),x0,[0 -1 1 0 0 0 0 0 0 0 0 0],0,[],[],lb,ub);
    %     s=fmincon(@(s) objFunctions_gauss_fmin(s,x,[y z]),x0,[],[],[],[],[],[],@(s) nlcon(x));
        y_bar=fit(s([1 2 3 4 9  10 5]),x);
        z_bar=fit(s([6 2 3 7 11 12 8]),x);
        err1=sum((y-y_bar).^2)/length(y);
        err2=sum((z-z_bar).^2)/length(z);
        err=err1+err2;
        all_errors=[all_errors err];
        alls_s=[alls_s s];
        all_yfit=[all_yfit y_bar];
        all_zfit=[all_zfit z_bar];
    end
    H=[];
    [~,indx]=min(all_errors);
    err=all_errors(indx);
    s=alls_s(:,indx);
    y_bar=all_yfit(:,indx);
    z_bar=all_zfit(:,indx);
%     disp(['Amps:a1:' num2str(s(1)) 'a2:' num2str(s(6))]);
%     disp(['Freq:f1:' num2str(s(2)/1000) 'f2:' num2str(s(3)/1000)]);
%     disp(['Gauss_mean:m1:' num2str(s(9)*1e6) 'm2:' num2str(s(11)*1e6)]);
%     disp(['Gauss_std:s1:' num2str(s(10)*1e6) 's2:' num2str(s(12)*1e6)]);
    if(plot_flag==1)
%         subplot(311);hold off;
%         plot(x*1e6,y,'r', 'LineWidth',2); axis([0 400 -0.2 0.2]); ylabel('Amp'); xlabel('time(us)');
%         hold on;
%         plot(x*1e6,y_bar, '--k', 'LineWidth',2); axis([0 400 -0.2 0.2]);
%         legend('Left','Fitting');
%         subplot(312);hold off;
%         plot(x*1e6,z,'b', 'LineWidth',2); axis([0 400 -0.2 0.2]); ylabel('Amp'); xlabel('time(us)');
%         hold on;
%         plot(x*1e6,z_bar, '--k', 'LineWidth',2); axis([0 400 -0.2 0.2]);
%         legend('Right','Fitting');
%         subplot(313);
        h1=plot(x*1e6,y,'ro', 'LineWidth',2); axis([0 100 -0.5 0.5]);hold on;
        h2=plot(x*1e6,y_bar, 'r-', 'LineWidth',2); axis([0 100 -0.5 0.5]);
        ylabel('Amplitude'); xlabel('time(us)');
        hold on;
        h3=plot(x*1e6,z,'bo', 'LineWidth',2); axis([0 100 -0.5 0.5]);hold on;
        h4=plot(x*1e6,z_bar, 'b-', 'LineWidth',2); axis([0 100 -0.5 0.5]);
        H=[h1 h2 h3 h4];
        legend(H,'Left','Left-fit','Right','Right-fit','orientation','horizontal');
        grid on;
%         set(gca,'XTick',0:10:100);
%         set(gca,'XTickLabel',0:10:100);
        set(gca,'YTick',-0.5:0.1:0.5);
        set(gca,'YTickLabel',-0.5:0.1:0.5);
        set(gca,'Fontsize',14);
        drawnow;
%         pause;
        %saveas(gcf,['Fine_basis/basis_' num2str(sub) '.png']);
        print(figure(1),'-depsc',['Fine_basis/basis_' num2str(sub) '.eps']);
        hold off;
    end
%     file_name=[file_name '/Orthogonality/basis_' num2str(sub) '.png'];
%     hold off;
%     plot(x*1e6,y,'r', 'LineWidth',2); 
%     hold on;
%     plot(x*1e6,z,'b', 'LineWidth',2); axis([0 400 -0.2 0.2]); ylabel('Amp'); xlabel('time(us)');
%     legend('Left','Right');
%     set(gca,'fontsize', 16);
%     saveas(gcf,file_name);
%     hold off;
end

