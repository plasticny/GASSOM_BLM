classdef Binaural_Fit < handle
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
        function this = Binaural_Fit(GM)
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
    end
end
%%
% close all;
% freq_slope=(this.tuning_freq2-tuning_freq1)/(1000*(win_size*1000/Fs));
% % freq_slope=(this.tuning_freq2-tuning_freq1)/(1000*0.8);
% this.center_frequency=(tuning_freq1+this.tuning_freq2)/2;
% 
% figure(1); histn(this.tuning_ILD,-50,5,50); xlabel('ILD(dB)'); ylabel('Number of Basis(%)'); title('Interaural level difference (ILD)'); axis([-70 70 0 90]); set(gca,'fontsize', 16);
% figure(2); histn(this.center_frequency/1000,10,10,100); xlabel('Frequency (kHz)'); ylabel('Number of Basis(%)'); title('Frequency');axis([0 110 0 90]); set(gca,'fontsize', 16);
% figure(3); histn(freq_slope,-100,10,100); xlabel('m (kHz/ms)'); ylabel('Number of Basis(%)'); title('Rate of frequency change:(f_{2}-f_{1})/T');axis([-110 110 0 90]); set(gca,'fontsize', 16);
% figure(4); histn(this.tuning_ITD,-50,5,50); xlabel('ITD(us)'); ylabel('Number of Basis(%)'); title('Interaural time difference (ITD)');axis([-120 120 0 90]); set(gca,'fontsize', 16);
% figure(5); histn(tuning_freq1/1000,10,10,100); xlabel('Frequency (kHz)'); ylabel('Number of Basis(%)'); title('Frequency (f_1)');axis([0 110 0 90]); set(gca,'fontsize', 16);
% figure(6); histn(this.tuning_freq2/1000,10,10,100); xlabel('Frequency (kHz)'); ylabel('Number of Basis(%)'); title('Frequency (f_2)');axis([0 110 0 90]); set(gca,'fontsize', 16);
% 
% Tuning_param_all{1}=Tuning_param_all;
% Tuning_param_all{2}=tuning_freq1;
% Tuning_param_all{3}=this.tuning_freq2;
% Tuning_param_all{4}=this.tuning_ILD;
% Tuning_param_all{5}=this.tuning_ITD;
% Tuning_param_all{6}=this.center_frequency/1000;
% Tuning_param_all{7}=this.Fit_Error;
% Tuning_param_all{8}=this.aural_dominance;
% Tuning_param_all{9}=this.correlation_lr;
% saveas(figure(1),[file_name '\ILD.png']);
% % save([file_name '/Tuning_param_all.mat'],'Tuning_param_all');
% saveas(figure(2),[file_name '\CF.png']);
% saveas(figure(3),[file_name '\Slope.png']);
% saveas(figure(4),[file_name '\ITD.png']);
% saveas(figure(5),[file_name '\f_1.png']);
% saveas(figure(6),[file_name '\f_2.png']);
% 
% save([file_name '/Tuning_param_all_binaural.mat'],'Tuning_param_all');
% %%
% figure(7);
% center_freq=reshape(Tuning_param_all{6},topo);
% imagesc(center_freq'); colormap(jet); caxis([10 100]); colorbar;
% set(gca,'XTick',[]); set(gca,'YTick',[]);
% axis square;
% saveas(figure(7),[file_name '\CF_topo.png']);
% figure(8);
% itd_tuning=reshape(Tuning_param_all{5},topo);
% imagesc(itd_tuning'); colormap(jet); caxis([-100 100]); colorbar;
% set(gca,'XTick',[]); set(gca,'YTick',[]);
% axis square;
% saveas(figure(8),[file_name '\CF_itd.png']);
% figure(9);
% ild_tuning=reshape(Tuning_param_all{4},topo);
% imagesc(ild_tuning'); colormap(gray); caxis([-20 20]); colorbar;
% set(gca,'XTick',[]); set(gca,'YTick',[]);
% axis square;
% hold on;
% % [x_grid,y_grid]=meshgrid(1:20,1:20);
% % [C,h]=contour(x_grid,y_grid,ild_tuning',-100:20:100,'r');
% % clabel(C,h,[-100:10:100],'FontSize',12,'Color','b');
% % saveas(figure(9),[file_name '\CF_ild.png']);
% %%
% figure(10);
% histn(this.aural_dominance,-1,0.5,1); xlabel('Dominance'); ylabel('Number of Basis(%)'); title('Aural dominance'); axis([-1.5 1.5 0 50]); set(gca,'fontsize', 16);
% saveas(figure(10),[file_name '\AD.png']);
