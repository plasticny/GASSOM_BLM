classdef Binaural_Fit_expcos < handle
    properties
        Tuning_prop;
        Tuning_param_all;
        tuning_freq1;
        tuning_freq2;
        tuning_ILD;
        tuning_ITD;
        center_frequency;
        Fit_Error;
        aural_dominance;
        correlation_lr;
        
        Basis1;
        Basis2;
        fit;
        Left_Fit;
        Right_Fit;
        xx;
    end
    
    methods
        function this = Binaural_Fit_expcos(GM)
            win_size=GM.patch_len;

            T = win_size;
            this.fit = @(b,x)  b(1).*(cos(2*pi*(x.*(1-x/(2*T)).*b(2)+(x.^2/(2*T)).*b(3)) + b(4))).*exp(-0.5*((x-b(5))./b(6)).^2) + b(7);
            %% Select all bases
            this.Basis1=GM.gsm{1}.bases{1}(1:win_size,:);
            this.Basis2=GM.gsm{1}.bases{1}(win_size+1:end,:);
            Num_basis=GM.gsm{1}.n_subspace;
            Fs=GM.fs;
            t=0:1/Fs:(win_size-1)/Fs;

            this.Left_Fit=zeros(Num_basis,7);
            Left_Fit_Error=zeros(Num_basis,1);
            this.Right_Fit=zeros(Num_basis,7);
            Right_Fit_Error=zeros(Num_basis,1);
            this.Fit_Error=zeros(Num_basis,1);

            tuning_freq1=zeros(Num_basis,1);
            this.tuning_freq2=zeros(Num_basis,1);
            this.tuning_ILD=zeros(Num_basis,1);
            this.tuning_ITD=zeros(Num_basis,1);
            binocularity=zeros(Num_basis,1);
            this.aural_dominance=zeros(Num_basis,1);
            lr_energy_diff=zeros(Num_basis,1);
            Tuning_param_all=zeros(Num_basis,12);
            this.correlation_lr=zeros(Num_basis,2);
            upd_fit = textprogressbar(Num_basis);
            x=t';
            this.xx = x;
            for i=1:Num_basis
                y1=(this.Basis1(:,i));
                y2=(this.Basis2(:,i));
                [s,err]=fit_fm_together_with_gauss([x y1 y2],i,GM,false);
            %     pause;
                this.Tuning_param_all(i,:)=s;
                this.tuning_freq1(i)=s(2);
                this.tuning_freq2(i)=s(3);
                this.tuning_ILD(i)=20*log10(s(1)/s(6));
                central_freq=(s(2)+s(3))/2;
                this.tuning_ITD(i)=((s(4)-s(7))/(2*pi*central_freq))*1e6; % micro-sec; need unwrap
                binocularity(i)=sum(y1.^2)/sum(y2.^2);
            %     this.aural_dominance(i)=(s(1)-s(6))/(s(1)+s(6));
                this.aural_dominance(i)=(sum((abs(y2)).^2)-sum((abs(y1)).^2))/(sum((abs(y2)).^2)+sum((abs(y1)).^2));
                lr_energy_diff(i)=(sum((abs(y2)).^2)-sum((abs(y1)).^2));

                this.Left_Fit(i,:)=s([1 2 3 4 9 10 5]);
                this.Right_Fit(i,:)=s([6 2 3 7 11 12 8]);

                Left_Fit_Error(i) = sum((this.fit(this.Left_Fit(i,:),x)-y1).^2)/length(y1);
                Right_Fit_Error(i) = sum((this.fit(this.Right_Fit(i,:),x)-y2).^2)/length(y2);

                this.Fit_Error(i)=err;

                lr_corr=xcorr(y1,y2);
                [peaks_corr,loc_corr]=findpeaks(lr_corr);
                [sort_pk,sort_pk_index]=sort(peaks_corr,'descend');
                this.correlation_lr(i,1)=sort_pk(1);
                this.correlation_lr(i,2)=sort_pk_index(1);
                upd_fit(i);
            end


            this.center_frequency = (tuning_freq1+this.tuning_freq2)/2;
            Thresh=1;
            good_fittings=(Left_Fit_Error<Thresh)&(Right_Fit_Error<Thresh);   
            
            this.Tuning_prop{1}=Tuning_param_all;
            this.Tuning_prop{2}=this.tuning_freq1; 
            this.Tuning_prop{3}=this.tuning_freq2;
            this.Tuning_prop{4}=this.tuning_ILD;
            this.Tuning_prop{5}=this.tuning_ITD;
            this.Tuning_prop{6}=this.center_frequency/1000;
            this.Tuning_prop{7}=this.Fit_Error;
            this.Tuning_prop{8}=this.aural_dominance;
            this.Tuning_prop{9}=this.correlation_lr;
        end
       
        function [f] = plotFit(this)
            f = gcf;
            fig_ind = 1;
            for fit_ind = 100:100:400        
                left_fit1 = this.fit(this.Left_Fit(fit_ind,:),this.xx);
                right_fit1 = this.fit(this.Right_Fit(fit_ind,:),this.xx);
                subplot(4,2,fig_ind);
                plot(left_fit1);
                hold on
                plot(this.Basis1(:,fit_ind));
                subplot(4,2,fig_ind+4);
                plot(right_fit1);
                hold on
                plot(this.Basis2(:,fit_ind));
                fig_ind = fig_ind+1;
            end
        end
        
        function [fit_bf] = getFitBF(this,fit_ind)
            left_fit = this.fit(this.Left_Fit(fit_ind,:),this.xx);
            right_fit = this.fit(this.Right_Fit(fit_ind,:),this.xx);
            fit_bf = [left_fit,right_fit];
        end
        
        function [s,err]=fit_fm_together_with_gauss(this, d,sub,model)
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

        end


    end
end