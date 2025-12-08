
classdef GASSOM_Model < handle
    %%%%%%% SIMPLE IS BEST %%%%%%
    properties
        gsm;
        env;
        net;
        current_test_net;
        note;

        max_iter;
        topo_space;
        
        patch_dur;
        patch_len;
        patch_stride_percent;
        
        inputType;
        binaural_only_flag;
        fs;
        azim_list;
        elev_list;
        locs_list;
        locs_num;
        
        somTrainParam;
        netTrainParam;
        netTestParam;

        % by peter, for avoiding changing the logic of patch_dur
        dim_patch;
    end
    
    methods
        % inputType: 'waveform' / 'cochleagram'
        function this = GASSOM_Model(PARAM, dim_patch, inputType, hrtf_database, hrtf_subject)
            assert(inputType == "waveform" || inputType == "cochleagram");

            if ~exist('temp_data','dir'), mkdir('temp_data'); end
            this.fs = PARAM{1};                        
            this.patch_dur = PARAM{2};
            this.topo_space = PARAM{3};
            this.max_iter = PARAM{4};  
            
            this.dim_patch = dim_patch;
            
            % default locations
%             this.azim_list = 0:10:350;
%             this.azim_list = -90:60:90;
            this.azim_list = [-90:10:90];
            this.elev_list = 0;
            this.locs_list = combvec(this.azim_list,this.elev_list);
            this.locs_num = length(this.locs_list);

            this.setHRTFandSubject(hrtf_database, hrtf_subject);
            
            % initialize GASSOMs                  
            this.inputType = inputType;
            this.binaural_only_flag = true;
            this.initEnv;
            this.loadEnvData;
            
            this.initGASSOM;
            
            this.somTrainParam.input_type = [];
            % this.somTrainParam.hrtf = 'cipic';
            % this.somTrainParam.subject = 9;
            this.somTrainParam.max_iter = this.max_iter;
            this.somTrainParam.locs_rand = randi(this.locs_num,this.max_iter,1);  
            this.somTrainParam.audio_dur = 200e-3;
            this.somTrainParam.audio_len = round(this.somTrainParam.audio_dur*this.fs);
            this.somTrainParam.audio_idx = genRandInd(this.env.timit_train(:,2),this.max_iter);
            this.somTrainParam.audio_bgn = genRandBegin(cell2mat(this.env.timit_train(:,2)),this.somTrainParam.audio_idx,this.somTrainParam.audio_len);
            
            % this.netTrainParam.hrtf = 'cipic'; 
            % this.netTrainParam.subject = 9;
            this.netTrainParam.max_iter = this.locs_num * 100;
            this.netTrainParam.locs_rand = randi(this.locs_num, this.netTrainParam.max_iter, 1);
            this.netTrainParam.audio_dur = this.somTrainParam.audio_dur;
            this.netTrainParam.audio_len = round(this.netTrainParam.audio_dur*this.fs);
            this.netTrainParam.audio_idx = genRandInd(this.env.timit_train(:,2),this.netTrainParam.max_iter);
            this.netTrainParam.audio_bgn = genRandBegin(cell2mat(this.env.timit_train(:,2)),this.netTrainParam.audio_idx,this.netTrainParam.audio_len);
            this.netTrainParam.gwn_seed = 1001;
            
            % this.netTestParam.hrtf = 'cipic'; 
            % this.netTestParam.subject = 9;
            this.netTestParam.max_iter = this.locs_num * 100;
            this.netTestParam.locs_rand = randi(this.locs_num, this.netTestParam.max_iter, 1);
            this.netTestParam.audio_dur = this.somTrainParam.audio_dur;
            this.netTestParam.audio_len = round(this.netTestParam.audio_dur*this.fs);            
            this.netTestParam.audio_idx = genRandInd(this.env.timit_test(:,2),this.netTestParam.max_iter);
            this.netTestParam.audio_bgn = genRandBegin(cell2mat(this.env.timit_test(:,2)),this.netTestParam.audio_idx,this.netTestParam.audio_len);            
            this.netTestParam.gwn_seed = 1002;
            this.netTestParam.pass_band = [];
            this.netTestParam.click_dur = 3e-3; % for click train exp.
            this.netTestParam.click_int = 10e-3;
            
            this.setEnvLocs;
        end      
        
        function initGASSOM(this)
            this.gsm = cell(3,1);
            % init_gsm_param = {[1 this.patch_len],this.topo_space,this.max_iter};   
            init_gsm_param = {[1 this.dim_patch],this.topo_space,this.max_iter};
            if this.inputType == "waveform"
                this.gsm{1} = GASSOM_Online_Waveform(init_gsm_param);
            elseif this.inputType == "cochleagram"
                this.gsm{1} = GASSOM_Online_Cochleagram(init_gsm_param);
                % this.gsm{1} = GASSOM_Batch(init_gsm_param);
            end
            this.gsm{2} = GASSOM_Online_S(init_gsm_param);
            this.gsm{3} = GASSOM_Online_S(init_gsm_param);
        end
        
        function setHRTFandSubject(this,h,s)
            this.somTrainParam.hrtf = h;
            this.netTrainParam.hrtf = h;
            this.netTestParam.hrtf = h;
            this.somTrainParam.subject = s;
            this.netTrainParam.subject = s;
            this.netTestParam.subject = s;

            if strcmp(this.somTrainParam.hrtf,'cipic')
                sel_cipic_azim = [-80 -55 -40:10:40 55 80];
                this.azim_list = [sel_cipic_azim, sel_cipic_azim+180];
                this.elev_list = [0];
            else
                this.azim_list = [-90:10:90];
                % this.azim_list = [this.azim_list, this.azim_list+180];
                this.elev_list = [0];
            end

            this.setEnvLocs;
            this.setEnvTestLocs;
        end
        
        function setDur(this,dur)
%             this.somTrainParam.audio_dur = dur;
%             this.somTrainParam.audio_len = round(this.fs*dur);
%             this.netTrainParam.audio_dur = dur;
%             this.netTrainParam.audio_len = round(this.fs*dur);
            this.netTestParam.audio_dur = dur;
            this.netTestParam.audio_len = round(this.fs*dur);
        end
        
        function loadEnvData(this)
            this.env.loadTimit;
            this.env.sofa = SOFALoader;
        end
        
        function initEnv(this)         
            this.env = environment({this.patch_dur});            
            this.patch_len = this.env.patch_len;
        end
        
        function setEnvLocs(this)
            this.locs_list = combvec(this.azim_list,this.elev_list);
            this.locs_num = size(this.locs_list,2);
            this.env.azim_list = this.azim_list;
            this.env.elev_list = this.elev_list;
            this.env.locs_list = this.locs_list;
            this.env.locs_num = this.locs_num;
            this.somTrainParam.locs_rand = randi(this.locs_num,this.max_iter,1);  
        end
        
        function setEnvTestLocs(this)
            this.env.test_locs_list = this.locs_list;
            this.env.test_locs_num = this.locs_num;
        end
        
        function delEnv(this)
            this.env.sofa = [];
            this.env.timit_train = [];
            this.env.timit_test = [];
        end
        
        function vs(this)
            this.gsm{1}.visualizeBases(1);
        end
        
        function loadSOFA(this)
            if exist('sofaloaded.mat','file')
                load sofaloaded.mat;
                this.env.sofa = sofaloaded;
            else
                this.env.sofa = SOFALoader;
            end
        end
                                
        function encodeGASSOM(this,XL,XR)
            % winner = this.gsm{1}.assomEncode([XL;XR]);
            % this.gsm{1}.updateBasis([XL;XR]);

            X = [XL;XR];
            this.gsm{1}.assomEncode(X);
            this.gsm{1}.updateBasis(X);  

            if ~this.binaural_only_flag
                this.gsm{2}.assomEncode(XL);
                this.gsm{2}.updateBasis(XL);
                this.gsm{3}.assomEncode(XR);
                this.gsm{3}.updateBasis(XR);
            end
        end

%         function [resp] = getResponse(this,X)    
%             if this.binaural_only_flag
%                 resp = this.gsm{1}.getResponse(X);
%             else
%                 resp_B1 = this.gsm{1}.getResponse(X);
%                 XL = X(1:this.patch_len,:);
%                 XR = X(this.patch_len+1:end,:);
%                 resp_L1 = this.gsm{2}.getResponse(XL);
%                 resp_R1 = this.gsm{3}.getResponse(XR);
%                 resp = [resp_B1;resp_L1;resp_R1];
%             end
%         end    

        function [resp] = getResponse(this,X) 
            resp = this.gsm{1}.getResponse(X);   
        end    
        
        function [predAve,pred] = predLocation(this,X,net_ind)
            X_resp = this.getResponse(X);
            Y_resp = this.net{net_ind}(X_resp);
            Y_resp_ave = mean(Y_resp,2);
            [~,pred] = max(Y_resp);
            [~,I] = max(Y_resp_ave);
            predAve = this.env.locs_list(:,I);
        end
        
              
        function plotWVD(this,ind)
            b = this.B1.getBasis(ind);
            bL = b(:,1);
            figure;
            [bL_wvd,f,t] = wvd(bL,this.fs);
            imagesc(t,f,bL_wvd); axis xy; colorbar; title(int2str(ind));
        end
        
        function [saveName] = saveModel(this)
            date_string = datestr(now,'yyyy-mm-dd-hh');
            saveName = ['Data/gsm_',this.somTrainParam.input_type,'_',this.somTrainParam.hrtf,'_',int2str(this.somTrainParam.subject),...
                '_net_',this.netTrainParam.hrtf,'_',int2str(this.netTrainParam.subject),'_',...
                date_string,'.mat'];
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Train GASSOM with Audio/Sweep etc...
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function trainGASSOM_timit(this)
            this.somTrainParam.input_type = 'timit';
            upd_som = textprogressbar(...
                this.max_iter, 'showremtime', true ...
            );
            for i_tg = 1:this.max_iter                
                [frmL,frmR] = this.env.genOneEpisode(this.somTrainParam,i_tg);
%                 [frmL,frmR] = this.env.genOneEpisodeGWN(this.somTrainParam,this.env.locs_list(:,this.somTrainParam.locs_rand(i_tg)));
                this.encodeGASSOM(frmL,frmR);
                upd_som(i_tg);
            end
        end

        function [winner_count] = trainGASSOM_timit2 (this, save_folder)
            this.somTrainParam.input_type = 'timit';
            
            % gsmMapShotVideoMaker = gassomMapVideoMaker(save_folder + '/timeshot', "waveform");
            % gsmMapShotVideoMaker.open();
            
            winner_count = zeros(this.gsm{1}.n_subspace,1);
            
            upd_som = textprogressbar(this.max_iter, 'showremtime', true);
            for i_tg = 1:this.max_iter                
                % the samples are z-score normalized in this step
                [frmL,frmR] = this.env.genOneEpisode(this.somTrainParam,i_tg);

                % frmL = frmL ./ vecnorm(frmL);
                % frmR = frmR ./ vecnorm(frmR);

                % frmL = frmL(:,1:53);
                % frmR = frmR(:,1:53);

                % chkIdx = 1:1:size(frmL,2);
                % j = chkIdx(randi(length(chkIdx)));

                % chkL = frmL(:,j);
                % chkR = frmR(:,j);

                this.encodeGASSOM(frmL,frmR);

                for iw = this.gsm{1}.winners
                    winner_count(iw) = winner_count(iw)+1;
                end

                % if i_tg == 1 || mod(i_tg, 1000) == 0
                %     gsmMapShotVideoMaker.addGassomMapFrame(this);
                % end

                upd_som(i_tg);
            end

            % gsmMapShotVideoMaker.close()
        end

        function trainGASSOM_timit3 (this)
            param = this.somTrainParam;
            assert(param.hrtf == "kemar" && param.subject == 0);

            param.input_type = "timit";

            tpb = textprogressbar(this.max_iter);
            for i = 1:this.max_iter
                y_all = this.env.timit_train{param.audio_idx(i),1};
                y = y_all(param.audio_bgn(i,1)+(1:param.audio_len));
                bi = this.env.sofa.spatMono(y,this.env.locs_list(:,param.locs_rand(i)),param.hrtf,param.subject);

                biL = bi(:,2);
                biR = bi(:,1);

                start_ind_ls = 1:this.env.patch_stride:length(biL)-this.env.patch_len;

                X = [];
                for j = 1:length(start_ind_ls)
                    start_ind = start_ind_ls(j);

                    frmL = biL(start_ind:start_ind + this.env.patch_len - 1);
                    frmR = biR(start_ind:start_ind + this.env.patch_len - 1);
    
                    x = [frmL;frmR];
                    x = x / norm(x);

                    X = [X x];
                end

                this.gsm{1}.assomEncode(X);
                this.gsm{1}.updateBasis(X);

                tpb(i);
            end            
        end
            
        function trainGASSOM_timit_Coch(this)
            this.somTrainParam.input_type = 'timit';
            upd_som = textprogressbar(this.max_iter);
            for i_tg = 1:this.max_iter
                [frmL,frmR,~] = this.env.genOneEpisodeCoch(this.somTrainParam,i_tg);
                single_len = size(frmL,1);
                rm = normalize([frmL;frmR]);
                frmL = rm(1:single_len,:);
                frmR = rm(single_len+1:end,:);
                this.encodeGASSOM(frmL,frmR);
                upd_som(i_tg);
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAIN GASSOM%%%%%%%%%%%%%%%%%%%%%%
        function trainGASSOM_cochleagram(this)
            this.somTrainParam.input_type = 'timit';
            upd_som = textprogressbar(this.max_iter);
            for i_tg = 1:this.max_iter
                [frmL,frmR,~] = this.env.genOneEpisodeCoch2(this.somTrainParam,i_tg);
                single_len = size(frmL,1);
                rm = normalize([frmL;frmR]);
                frmL = rm(1:single_len,:);
                frmR = rm(single_len+1:end,:);
                this.encodeGASSOM(frmL,frmR);
                upd_som(i_tg);
            end
        end

        function [winners] = trainGASSOM_cochleagram2 (this, chunk_size, dataset_path, save_folder)
            this.somTrainParam.input_type = 'timit';
            
            % gsmMapShotVideoMaker = gassomMapVideoMaker(save_folder + '/timeshot', "cochleagram", chunk_size);
            % gsmMapShotVideoMaker.open();
            
            chunk_shift = 1;
            
            winners = zeros(1,this.gsm{1}.n_subspace);
            
            disp("loading dataset");
            trainData = load(dataset_path);
            trainX = trainData.trainX;
            trainY = trainData.trainY;
            sampleSize = length(trainY);

            rng(49);
            
            upd_som = textprogressbar(this.max_iter);
            for i = 1:sampleSize
                frmL = trainX{i}{1};
                frmR = trainX{i}{2};

                nFrm = size(frmL, 2);
                assert(nFrm == size(frmR, 2));

                % X = [];
                % for j = 0:chunk_shift:nFrm-chunk_size
                %     chkL = frmL(:,j+(1:chunk_size));
                %     chkR = frmR(:,j+(1:chunk_size));

                %     % if mean(chkL, "all") <= -70 || mean(chkR, "all") <= -70
                %     %     continue
                %     % end

                %     % single_len = size(chkL,1);
                %     % rm = normalize([chkL;chkR]);
                %     % rm = [chkL;chkR];
                %     % rm = rm + abs(min(rm, [], "all"));
                %     % chkL = rm(1:single_len,:);
                %     % chkR = rm(single_len+1:end,:);

                %     x = [reshape(chkL, [], 1);reshape(chkR, [], 1)];
                %     x = x-ones(size(x,1),1)*mean(x,1);
                %     x = bsxfun(@rdivide, x, sqrt(sum(x.^2))+eps); 
                %     % x = x / norm(x);

                %     % assert(min(x, [], "all") >= 0);

                %     X = [X x];
                % end

                chkIdx = 0:chunk_shift:nFrm-chunk_size;
                j = chkIdx(randi(length(chkIdx)));

                frmL = frmL(:,j+(1:chunk_size));
                frmR = frmR(:,j+(1:chunk_size));

                % single_len = size(frmL,1);
                % rm = normalize([frmL;frmR]);
                % rm = rm + abs(min(rm, [], "all"));
                % frmL = rm(1:single_len,:);
                % frmR = rm(single_len+1:end,:);

                x = [reshape(frmL, [], 1);reshape(frmR, [], 1)];     
                x = x-ones(size(x,1),1)*mean(x,1);
                X = bsxfun(@rdivide, x, sqrt(sum(x.^2))+eps); 
                % X = x / norm(x);

                % assert(min(X, [], "all") >= 0);

                this.gsm{1}.assomEncode(X);
                this.gsm{1}.updateBasis(X); 
                for iw = this.gsm{1}.winners
                    winners(iw) = winners(iw)+1;    
                end

                % if i == 1 || mod(i, 3000) == 0
                %     gsmMapShotVideoMaker.addGassomMapFrame(this);
                % end

                upd_som(i);
            end

            % gsmMapShotVideoMaker.close()
        end
        
        function trainGASSOM_ratemap(this)
            this.somTrainParam.input_type = 'timit';
            upd_som = textprogressbar(this.max_iter);
            for i_tg = 1:this.max_iter
                [frmL,frmR,~] = this.env.genOneEpisodeRM(this.somTrainParam,i_tg);
                single_len = size(frmL,1);
                rm = normalize_data([frmL;frmR]);
                frmL = rm(1:single_len,:);
                frmR = rm(single_len+1:end,:);
                this.encodeGASSOM(frmL,frmR);
                upd_som(i_tg);
            end
        end
        function [winner_count] = trainGASSOM_ratemap_count(this)
            this.somTrainParam.input_type = 'timit';
            upd_som = textprogressbar(this.max_iter);
            winner_count = zeros(this.gsm{1}.n_subspace,1);
            for i_tg = 1:this.max_iter
                [frmL,frmR,~] = this.env.genOneEpisodeRM(this.somTrainParam,i_tg);
                single_len = size(frmL,1);
                rm = normalize([frmL;frmR]);
                frmL = rm(1:single_len,:);
                frmR = rm(single_len+1:end,:);
                this.encodeGASSOM(frmL,frmR);
                for iw = this.gsm{1}.winners
                    winner_count(iw) = winner_count(iw)+1;
                end
                upd_som(i_tg);
            end
        end        
        
        function [YTrue,YPred,mae] = testDNN_cochleagram(this,dnn_model)
            [XTest,~] = this.env.genTestGWNCoch2(this.netTestParam);
            YTrue = repmat((1:this.env.test_locs_num),this.netTestParam.max_iter,1);
            YPred = cell(size(XTest));
            mae = zeros(size(XTest));
            for i=1:numel(XTest)
                xt = this.getResponse(XTest{i});
                YPred{i} = double(classify(dnn_model,xt'));                
                mae(i) = 10*mean(abs(YPred{i}-YTrue(i)));
            end
        end        
        
        function [YTrue,YPred,mae] = testDNN_RM(this,dnn_model)
            [XTest,~] = this.env.genTestGWNRM(this.netTestParam);
            YTrue = repmat((1:this.env.test_locs_num),this.netTestParam.max_iter,1);
            YPred = cell(size(XTest));
            mae = zeros(size(XTest));
            for i=1:numel(XTest)
                xt = this.getResponse(XTest{i});
                YPred{i} = double(classify(dnn_model,xt'));                
                mae(i) = 10*mean(abs(YPred{i}-YTrue(i)));
            end
        end      
        
        function trainGASSOM_timit_spec(this)
            this.somTrainParam.input_type = 'timit';
            upd_som = textprogressbar(this.max_iter);
            
            patch_length = 10; patch_ove = 1;
            for i_tg = 1:this.max_iter
                rL = []; rR = [];
                [frmL,frmR,~] = this.env.genOneEpisodeSpec(this.somTrainParam,i_tg);
                for frmInd = 0:patch_ove:size(frmL,2)-patch_length
                    rmL = reshape(frmL(:,frmInd+(1:patch_length)),[],1);
                    rmR = reshape(frmR(:,frmInd+(1:patch_length)),[],1);
                    single_len = length(rmL);
                    rm = normalize([rmL;rmR]);
                    rL = [rL,rm(1:single_len,1)];
                    rR = [rR,rm(single_len+1:end,1)];
                end
                frmL = rL; frmR = rR;
                this.encodeGASSOM(frmL,frmR);
                upd_som(i_tg);
            end
        end
                
        function [winner_count] = trainGASSOM_timit_count(this)
            this.somTrainParam.input_type = 'timit';
            upd_som = textprogressbar(this.max_iter);
            winner_count = zeros(1,this.gsm{1}.n_subspace);
            for i_tg = 1:this.max_iter                
                [frmL,frmR] = this.env.genOneEpisode(this.somTrainParam,i_tg);
                this.encodeGASSOM(frmL,frmR);
                for iw = this.gsm{1}.winners
                    winner_count(iw) = winner_count(iw)+1;
                end
                upd_som(i_tg);
            end            
        end
        
        function trainGASSOM_sweep(this)
            this.somTrainParam.input_type = 'fm';
            upd_som = textprogressbar(this.max_iter);
            for i_tg = 1:this.max_iter
                [frmL,frmR] = this.env.genOneEpisodeFM(this.somTrainParam,i_tg);
                this.encodeGASSOM(frmL,frmR);
                upd_som(i_tg);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Train patternnet with timit and kemar
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [XTrain, YTrain] = trainNet_timit(this,net_ind,net_structure)
            % Train patternnet
            % varargin
            % netTrainParam
            if nargin<3, net_structure = [200 50]; end
            this.net{net_ind} = patternnet(net_structure);
            this.net{net_ind}.trainParam.epochs = 1000;
            [XTrain,YTrain] = this.env.genTrainAudio(this.netTrainParam);
            XTrain = this.getResponse(XTrain);
            YTrain = full(ind2vec(YTrain));
            this.net{net_ind} = train(this.net{net_ind},XTrain,YTrain);
        end
        
        function [YTest, YPred] = testNet_timit(this,net_ind)
            [XTest,YTest] = this.env.genTestAudio(this.netTestParam);            
            YPred = cell(size(XTest));
            disp('Data generated, predicting...');
            for i_test = 1:numel(XTest)
                YPred{i_test} = this.predLocation(XTest{i_test},net_ind);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Train patternnet with gaussian white noise and kemar
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function trainNet_gwn(this,net_ind,net_structure)
            if nargin<3, net_structure = [200 50]; end
            this.net{net_ind} = patternnet(net_structure);
            this.net{net_ind}.trainParam.epochs = 1000;
            [XTrain,YTrain] = this.env.genTrainGWN(this.netTrainParam);
            XTrain = this.getResponse(XTrain);
            YTrain = full(ind2vec(YTrain));
            this.net{net_ind} = train(this.net{net_ind},XTrain,YTrain);
        end

        
        function [YTest,YPred] = testNet_gwn(this,net_ind)
            [XTest,YTest] = this.env.genTestGWN(this.netTestParam);            
            YPred = cell(size(XTest));
            upd = textprogressbar(numel(XTest));
            for i_test = 1:numel(XTest)
                YPred{i_test} = this.predLocation(XTest{i_test},net_ind);
                upd(i_test);
            end
        end
        
        function [YTest,YPred] = testNet_ct(this,net_ind)
            [XTest,YTest] = this.env.genTestCT(this.netTestParam);            
            YPred = cell(size(XTest));
            upd = textprogressbar(numel(XTest));
            for i_test = 1:numel(XTest)
                YPred{i_test} = this.predLocation(XTest{i_test},net_ind);
                upd(i_test);
            end
        end
        
        % Selected basis functions
        function trainNet_gwn2(this,net_ind,net_structure,bf)
            if nargin<3, net_structure = [200 50]; end
            this.net{net_ind} = patternnet(net_structure);
            this.net{net_ind}.trainParam.epochs = 1000;
            [XTrain,YTrain] = this.env.genTrainGWN(this.netTrainParam);
            XTrain = this.getResponse(XTrain);
            XTrain = XTrain(bf,:);
            YTrain = full(ind2vec(YTrain));
            this.net{net_ind} = train(this.net{net_ind},XTrain,YTrain);
        end
        
        function [YTest,YPred] = testNet_gwn2(this,net_ind,bf)
            [XTest,YTest] = this.env.genTestGWN(this.netTestParam);            
            YPred = cell(size(XTest));
            upd = textprogressbar(numel(XTest));
            for i_test = 1:numel(XTest)
                YPred{i_test} = this.predLocation2(XTest{i_test},net_ind,bf);
                upd(i_test);
            end
        end
        
        function [predAve,pred] = predLocation2(this,X,net_ind,bf)
            X_resp = this.getResponse(X);
            X_resp = X_resp(bf,:);
            Y_resp = this.net{net_ind}(X_resp);
            Y_resp_ave = mean(Y_resp,2);
            [~,pred] = max(Y_resp);
            [~,I] = max(Y_resp_ave);
            predAve = this.env.locs_list(:,I);
        end
        
        function [YTest,YPred] = testNet_singleHRIR(this,net_ind,locs,LorR)
            [XTest,YTest] = this.env.genTestSingleHRTF(this.netTestParam,locs,LorR);            
            YPred = cell(size(XTest));
            upd = textprogressbar(numel(XTest));
            for i_test = 1:numel(XTest)
                YPred{i_test} = this.predLocation(XTest{i_test},net_ind);
                upd(i_test);
            end
        end
        
        % test with dnn
        function [YTrue,YPred,mae] = testDNN(this,dnn_model)
            [XTest,~] = this.env.genTestGWN(this.netTestParam);
            YTrue = repmat((1:this.env.test_locs_num),this.netTestParam.max_iter,1);
            YPred = zeros(size(XTest));
            mae = zeros(size(XTest));
            for i=1:numel(XTest)
                xt = this.getResponse(XTest{i});
                yp = predict(dnn_model,xt');
                [~,YPred(i)] = max(mean(yp,1));
                mae(i) = 10*mean(abs(YPred(i)-YTrue(i)));
            end
        end
        
        function [YTrue,YPred,mae] = testDNNCoch(this,dnn_model)
            [XTest,~] = this.env.genTestGWNCoch(this.netTestParam);
            YTrue = repmat((1:this.env.test_locs_num),this.netTestParam.max_iter,1);
            YPred = cell(size(XTest));
            mae = zeros(size(XTest));
            for i=1:numel(XTest)
                xt = this.getResponse(XTest{i});
                YPred{i} = double(classify(dnn_model,xt'));                
                mae(i) = 10*mean(abs(YPred{i}-YTrue(i)));
            end
        end
        
        function [YPred] = testDNN_XTest(this,dnn_model,XTest)                        
            YPred = zeros(size(XTest));           
            for i=1:numel(XTest)
                xt = this.getResponse(XTest{i});
                yp = predict(dnn_model,xt');
                [~,YPred(i)] = max(mean(yp,1));                
            end
        end
    end
end