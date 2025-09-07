% test3
clc; clear all; addpath(genpath(pwd));

%% Produced loaded SOFA if needed
sofaloaded = SOFALoader;
save('sofaloaded.mat', 'sofaloaded');
%%
load sofaloaded.mat;
% i_sub = 43;

fs = 44100;
patch_dur = 16*1e-3;
topo_space = [16 16];
max_iter = 5e4;
gm_param = {fs, patch_dur, topo_space, max_iter};
GM = GASSOM_Model(gm_param);

%%
for i_sub = 1:sofaloaded.cipic_subject_num
% for i_sub = 43:sofaloaded.cipic_subject_num
    GM.setHRTFandSubject('cipic',GM.env.sofa.cipic_subject_ind(i_sub));
    % GM.azim_list = GM.env.sofa.cipic_azimuths;
    sel_cipic_azim = [-80 -55 -40:10:40 55 80];
    GM.azim_list = [sel_cipic_azim, sel_cipic_azim+180];
    GM.elev_list = [0];
    GM.setEnvLocs;
    GM.setEnvTestLocs;
    
    GM.initGASSOM;
    GM.trainGASSOM_timit;
    
    [net,options] = createDNN(prod(topo_space),GM.env.locs_num,1);
    GM.netTrainParam.max_iter = 100;
    [XTrain,YTrain] = GM.env.genTrainGWN(GM.netTrainParam);
    XTrain = GM.getResponse(XTrain);
    model_no_gsm = trainNetwork(XTrain',categorical(YTrain'),net,options);
    
    save(['test3_result/cipic',int2str(GM.somTrainParam.subject),'_',...
        GM.inputType,'.mat'],'GM','model_no_gsm');
end
%% 
check_model
[ypred,ytrue,mae] = GM.testDNN(model_no_gsm);
%%
load test3_result/noname1.mat;
load sofaloaded.mat;
fbrate = zeros(sofaloaded.cipic_subject_num,size(sort_f,1));
s = 1;
for i_sub = 162
    load(['test3_result/cipic',int2str(i_sub),'_waveform.mat']);
    GM.env.sofa = sofaloaded;
    for i=1:size(sort_f,1)
        GM.netTestParam.hrtf = 'cipic';
        GM.netTestParam.subject = mod(sort_f(i,1),10000);
        GM.netTestParam.max_iter = 100;
        [XTest,YTest] = GM.env.genTestSingleHRTF(GM.netTestParam,[0;0],round(sort_f(i,1)/10000));

        ypred = GM.testDNN_XTest(model_no_gsm,XTest);
        fbrate(s,i) = sum(ypred>13)/numel(ypred);
    end
    s=s+1;
end
%%
% load fbrate;
load sofaloaded.mat;
load test3_result/noname1.mat;
% ind_count = [1:45 47:89];
ind_count = [43];
g = sort_f(:,2);
pvals = zeros(1,size(fbrate,2));
for i = 1:size(fbrate,2)
    pvals(i) = anova1(fbrate(ind_count,i),g(ind_count),'off');
end
%%
close all
[p,tbl,stat] = anova1(mean(fbrate(ind_count,:),2),g(ind_count));
figure(2);
xlabel('Cluster Index');
ylabel('Front-back Rate');
hold on;
plot(stat.means,'LineWidth',1.2,'LineStyle','-.','Marker','diamond','Color','r');
saveas(gcf,'test3_result/fbrate_anova1.fig');
saveas(gcf,'test3_result/fbrate_anova1.jpg');
%% ANOVAN (FAILED)
% load fbrate;
% load sofaloaded.mat;
% load ../HRTF_cluster/Brian_script/new_data/noname1.mat;
ind_count = [1:45 47:89];
fbrate2 = fbrate(ind_count,:);
gg = g(ind_count,1);
fbre = reshape(fbrate2,[],1);
g1 = repmat(gg,size(fbrate2,2),1);
g2 = repmat(1:size(fbrate2,2),size(fbrate2,1),1);
g2 = reshape(g2,[],1);
[p3,tbl3,stat3] = anovan(fbre,{g1,g2});

%% FOR BACKWARD
load test3_result/noname1.mat;
load sofaloaded.mat;
addpath(genpath(pwd));
bfrate = zeros(sofaloaded.cipic_subject_num,size(sort_b,1));
for i_sub = sofaloaded.cipic_subject_ind
    load(['test3_result/cipic',int2str(i_sub),'_waveform.mat']);
    GM.env.sofa = sofaloaded;
    for i=1:size(sort_b,1)
        GM.netTestParam.hrtf = 'cipic';
        GM.netTestParam.subject = mod(sort_b(i,1),10000);
        GM.netTestParam.max_iter = 10;
        [XTest,YTest] = GM.env.genTestSingleHRTF(GM.netTestParam,[180;0],round(sort_b(i,1)/10000));

        ypred = GM.testDNN_XTest(model_no_gsm,XTest);
        bfrate(i_sub,i) = sum(ypred<13)/numel(ypred);
    end
end
%%
close all;
gb = sort_b(ind_count2,2);
[p,tbl,stat] = anova1(mean(bfrate(ind_count2,:),2),gb);
xlabel('Cluster Index');
ylabel('Back-front Rate');
hold on;
plot(stat.means,'LineWidth',1.2,'LineStyle','-.','Marker','diamond','Color','r');
saveas(gcf,'test3_result/bfrate_anova1.fig');
saveas(gcf,'test3_result/bfrate_anova1.jpg');
%% FOLLOW PAPER
% FORWARD HRTFs 
fbc_all = [];
for i_sub = sofaloaded.cipic_subject_ind
    load(['test3_result/cipic',int2str(i_sub),'_waveform.mat']);
    GM.env.sofa = sofaloaded;
    hs = [3 60 8 10 48 152];
    lr = [1 2 2 1 2 1];
    fbc = zeros(1,5);
    for i=1:6
        GM.netTestParam.hrtf = 'cipic';
        GM.netTestParam.subject = hs(i);
        GM.netTestParam.max_iter = 100;
        [XTest,YTest] = GM.env.genTestSingleHRTF(GM.netTestParam,[0;0],lr(i));
        ypred = GM.testDNN_XTest(model_no_gsm,XTest);
        fbc(i) = sum(ypred>13)/numel(ypred);
    end
    fbc_all = [fbc_all;fbc];
end
%%
close all;
fbc1 = min(fbc_all(:,1:5),[],2);
fbc2 = fbc_all(:,5);
fbc = [fbc1 fbc2];
p = anova1(fbc);
xticklabels({'With selection','KEMAR'});
ylabel('Front-back confusion rate');
saveas(gcf,'test3_result/fbc_with_selection.fig');
saveas(gcf,'test3_result/fbc_with_selection.jpg');

%% BACKWARD Test
bfc_all = [];
for i_sub = sofaloaded.cipic_subject_ind
    load(['test3_result/cipic',int2str(i_sub),'_waveform.mat']);
    GM.env.sofa = sofaloaded;
    hs = [3 18 9 9 8 153];
    lr = [1 1 1 2 2 2];
    fbc = zeros(1,5);
    for i=1:length(hs)
        GM.netTestParam.hrtf = 'cipic';
        GM.netTestParam.subject = hs(i);
        GM.netTestParam.max_iter = 100;
        [XTest,YTest] = GM.env.genTestSingleHRTF(GM.netTestParam,[180;0],lr(i));
        ypred = GM.testDNN_XTest(model_no_gsm,XTest);
        bfc(i) = sum(ypred<13)/numel(ypred);
    end
    bfc_all = [bfc_all;bfc];
end
%%
close all;
bfc1 = min(bfc_all(:,1:5),[],2);
bfc2 = bfc_all(:,3);
bfc = [bfc1 bfc2];
p = anova1(bfc);
xticklabels({'With selection','KEMAR'});
ylabel('Back-front confusion rate');
saveas(gcf,'test3_result/bfc_with_selection.fig');
saveas(gcf,'test3_result/bfc_with_selection.jpg');