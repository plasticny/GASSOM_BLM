
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
        function this = GASSOM_Model(PARAM, dim_patch, inputType)
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
            
            % initialize GASSOMs                  
            this.inputType = inputType;
            this.binaural_only_flag = true;
            this.initEnv;
            this.loadEnvData;
            
            this.initGASSOM;
            
            this.somTrainParam.input_type = [];
            this.somTrainParam.hrtf = 'kemar';
            % this.somTrainParam.hrtf = 'cipic';
            this.somTrainParam.subject = 0;
            % this.somTrainParam.subject = 3;
            this.somTrainParam.max_iter = this.max_iter;
            this.somTrainParam.locs_rand = randi(this.locs_num,this.max_iter,1);  
            this.somTrainParam.audio_dur = 200e-3;
            this.somTrainParam.audio_len = round(this.somTrainParam.audio_dur*this.fs);
%             this.somTrainParam.audio_idx = randi(length(this.env.timit_train),this.max_iter,1);
            this.somTrainParam.audio_idx = genRandInd(this.env.timit_train(:,2),this.max_iter);
            this.somTrainParam.audio_bgn = genRandBegin(cell2mat(this.env.timit_train(:,2)),this.somTrainParam.audio_idx,this.somTrainParam.audio_len);
            
%             if strcmp(this.somTrainParam.hrtf,'cipic')
% %                 this.azim_list = this.env.sofa.cipic_azimuths;
%                 this.azim_list = [-80 -55 -40 -30 -20 -10 0 10 20 30 40 55 80];
%                 this.elev_list = [0,180];
%                 this.locs_list = combvec(this.azim_list,this.elev_list);
%                 this.locs_num = length(this.locs_list);
%                 this.env.setLocs({this.azim_list,this.elev_list});
%             end
            
            this.netTrainParam.hrtf = 'kemar'; 
            this.netTrainParam.subject = 0;
            this.netTrainParam.max_iter = 1900;
            this.netTrainParam.audio_dur = this.somTrainParam.audio_dur;
            this.netTrainParam.audio_len = round(this.netTrainParam.audio_dur*this.fs);
            this.netTrainParam.audio_idx = randi(length(this.env.timit_train),this.netTrainParam.max_iter,1);
            this.netTrainParam.audio_bgn = genRandBegin(cell2mat(this.env.timit_train(:,2)),this.netTrainParam.audio_idx,this.netTrainParam.audio_len);
            this.netTrainParam.gwn_seed = 1001;
            
            this.netTestParam.hrtf = 'kemar'; 
            this.netTestParam.subject = 0;
            this.netTestParam.max_iter = 1900;
            this.netTestParam.audio_dur = this.somTrainParam.audio_dur;
            this.netTestParam.audio_len = round(this.netTestParam.audio_dur*this.fs);            
            this.netTestParam.audio_idx = randi(length(this.env.timit_test),this.netTestParam.max_iter,1);
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

            this.gsm{1}.assomEncode([XL;XR]);
            this.gsm{1}.updateBasis([XL;XR]);  

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

        function [winner_count] = trainGASSOM_timit2 (this)
            this.somTrainParam.input_type = 'timit';
            upd_som = textprogressbar(this.max_iter);

            gsmMapShotVideoMaker = gassomMapVideoMaker('temp_data/timeshot', "waveform");
            gsmMapShotVideoMaker.open();

            winner_count = zeros(this.gsm{1}.n_subspace,1);

            for i_tg = 1:this.max_iter                
                [frmL,frmR] = this.env.genOneEpisode(this.somTrainParam,i_tg);

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

                if i_tg == 1 || mod(i_tg, 1000) == 0
                    gsmMapShotVideoMaker.addGassomMapFrame(this);
                end

                upd_som(i_tg);
            end

            gsmMapShotVideoMaker.close()
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

        function [winners] = trainGASSOM_cochleagram_IOSR (this, chunk_size)
            % same as trainGASSOM_cochleagram, train gassom with cochleagram input
            % but use function from IOSR to get the cochleagram
            this.somTrainParam.input_type = 'timit';
            upd_som = textprogressbar(this.max_iter);

            gsmMapShotVideoMaker = gassomMapVideoMaker('chp4_result/timeshot', chunk_size);
            gsmMapShotVideoMaker.open();

            chunk_shift = 1;

            winners = zeros(1,this.gsm{1}.n_subspace);

            i_tg = 1;
            while i_tg <= this.max_iter
                [frmL,frmR,~] = this.env.genOneTrainEpisodeCochIOSR(this.somTrainParam,i_tg);

                nFrm = size(frmL, 2);
                assert(nFrm == size(frmR, 2));

                chkIdx = 0:chunk_shift:nFrm-chunk_size;
                j = chkIdx(randi(length(chkIdx)));

                XL = ones(128*chunk_size, 1);
                XR = ones(128*chunk_size, 1);

                frmL = frmL(:,j+(1:chunk_size));
                frmR = frmR(:,j+(1:chunk_size));

                single_len = size(frmL,1);
                rm = normalize([frmL;frmR]);
                frmL = rm(1:single_len,:);
                frmR = rm(single_len+1:end,:);

                XL(:,1) = reshape(frmL, [], 1);
                XR(:,1) = reshape(frmR, [], 1);

                this.encodeGASSOM(XL,XR);
                for iw = this.gsm{1}.winners
                    winners(iw) = winners(iw)+1;
                end

                if i_tg == 1 || mod(i_tg, 1000) == 0
                    gsmMapShotVideoMaker.addGassomMapFrame(this);
                end

                upd_som(i_tg);
                i_tg = i_tg + 1;
            end

            gsmMapShotVideoMaker.close()
        end

        function [winners] = trainGASSOM_cochleagram_IOSR2 (this, chunk_size)
            % same as trainGASSOM_cochleagram, train gassom with cochleagram input
            % but use function from IOSR to get the cochleagram
            this.somTrainParam.input_type = 'timit';
            upd_som = textprogressbar(this.max_iter);

            % gsmMapShotVideoMaker = gassomMapVideoMaker('temp_data/timeshot', "cochleagram", chunk_size);
            % gsmMapShotVideoMaker.open();

            chunk_shift = 1;

            winners = zeros(1,this.gsm{1}.n_subspace);

            for i_tg = 1:2:this.max_iter
                XL = [];
                XR = [];
                for i_it = 1:1:2
                    [frmL,frmR,~] = this.env.genOneTrainEpisodeCochIOSR(this.somTrainParam,i_tg);

                    nFrm = size(frmL, 2);
                    assert(nFrm == size(frmR, 2));

                    chkStart = 0:chunk_shift:nFrm-chunk_size;

                    xl = zeros(128*chunk_size, length(chkStart));
                    xr = zeros(128*chunk_size, length(chkStart));

                    for i_chk = 1:1:length(chkStart)
                        j = chkStart(i_chk);

                        chkL = frmL(:,j+(1:chunk_size));
                        chkR = frmR(:,j+(1:chunk_size));

                        single_len = size(chkL,1);
                        rm = [chkL; chkR];
                        rm = rm / norm(rm);
                        chkL = rm(1:single_len,:);
                        chkR = rm(single_len+1:end,:);

                        xl(:,i_chk) = reshape(chkL, [], 1);
                        xr(:,i_chk) = reshape(chkR, [], 1);
                    end

                    XL = [XL xl];
                    XR = [XR xr];
                end

                this.encodeGASSOM(XL,XR);
                for iw = this.gsm{1}.winners
                    winners(iw) = winners(iw)+1;
                end

                % if i_tg == 1 || mod(i_tg-1, 1000) == 0
                %     gsmMapShotVideoMaker.addGassomMapFrame(this);
                % end

                upd_som(i_tg);
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