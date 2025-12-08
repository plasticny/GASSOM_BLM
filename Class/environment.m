classdef environment < handle
    properties
        stiGenerator;  
        
        timit_train_num;
        timit_test_num;
        
        patch_len;
        patch_stride;
        fs;
        
        azim_list;
        elev_list;
        locs_list;
        locs_num;
        test_locs_list;
        test_locs_num;
    end

    properties(Transient)
        timit_train;
        timit_test;
        sofa;
    end
    
    methods
        function this = environment(PARAM)
            this.fs = 44100;
            this.patch_len = round(PARAM{1}*this.fs);
            this.patch_stride = round(this.patch_len*0.1);
            this.stiGenerator = stimuliGenerator;    
            if exist('sofaloaded.mat','file')
                load sofaloaded.mat;
                this.sofa = sofaloaded;
                clearvars sofaloaded;
            else
                this.sofa = SOFALoader;
            end
%             this.setTestLocs;
%             this.azim_list = 0:10:350;
%             this.elev_list = 0;
%             this.locs_list = combvec(this.azim_list,this.elev_list);
%             this.locs_num = length(this.locs_list);
        end           
        
        function setStride(this,stride_percent)
            this.patch_stride = round(this.patch_len*stride_percent);
        end
        
        function setLocs(this,param)
            if nargin<2
                this.azim_list = 0;
                this.elev_list = 0;                
            else
                this.azim_list = param{1};
                this.elev_list = param{2};
            end
            this.locs_list = combvec(this.azim_list,this.elev_list);
            this.locs_num = length(this.locs_list);
        end
        
        function setTestLocs(this,param)
            if nargin<2
                this.test_locs_list = this.locs_list;
                this.test_locs_num = this.locs_num;      
            else
                test_azim_list = param{1};
                test_elev_list = param{2};
                this.test_locs_list = combvec(test_azim_list,test_elev_list);
                this.test_locs_num = length(this.test_locs_list);
            end
        end
        
        function [frmL,frmR,nFrm] = genFrame(this,bi)
            frmL = []; frmR = []; nFrm = -1; %#ok<NASGU>
            biL = bi(:,2); biR = bi(:,1);
            frmIndex = bsxfun(@plus,(0:this.patch_stride:length(biL)-this.patch_len)',...
                1:this.patch_len);
            frmL = (biL(frmIndex))'; frmR = (biR(frmIndex))';            
            frm = normalize_data([frmL;frmR]); % z-score normalization
            frmL = frm(1:this.patch_len,:);
            frmR = frm(this.patch_len+1:end,:);
            nFrm = size(frm,2);
        end   
        
        
        %%% Generate one trainig episode
        function [frmL,frmR,nFrm] = genOneEpisode(this,param,ind)
            y_all = this.timit_train{param.audio_idx(ind),1};
            y = y_all(param.audio_bgn(ind,1)+(1:param.audio_len));
            bi = this.sofa.spatMono(y,this.locs_list(:,param.locs_rand(ind)),param.hrtf,param.subject);
            [frmL,frmR,nFrm] = this.genFrame(bi);
        end
        
        function [frmL, frmR, nFrm] = genOneEpisodeAngle(this,param,ind)
            y_all = this.timit_train{param.audio_idx(ind),1};
            y = y_all(param.audio_bgn(ind,1)+(1:param.audio_len));
            frmL = []; frmR = [];
            for i_tg = 1:this.locs_num
                bi = this.sofa.spatMono(y,this.locs_list(:,i_tg),...
                    param.hrtf,param.subject);
                [fL,fR,nFrm] = this.genFrame(bi);                
                frmL = [frmL, fL]; %#ok<*AGROW>
                frmR = [frmR, fR];
            end
            nFrm = size(frmL,2);
        end
        
        function[blockL, blockR, nFrm] = genOneEpisodeAngleRM(this,param,ind)
            y_all = this.timit_train{param.audio_idx(ind),1};
            y = y_all(param.audio_bgn(ind,1)+(1:param.audio_len));            
            global blockLength blockShift;
            blockL = []; blockR = [];
            for i_tg = 1:this.locs_num
                bi = this.sofa.spatMono(y,this.locs_list(:,i_tg),...
                    param.hrtf,param.subject);
                [frmL,frmR,nFrm] = compRateMap(bi);                                                
                for i=0:blockShift:nFrm-blockLength
                    bL = reshape(frmL(:,i+(1:blockLength)),[],1);
                    bR = reshape(frmR(:,i+(1:blockLength)),[],1);
                    blockL = [blockL,bL];
                    blockR = [blockR,bR];
                end                  
            end
            nFrm = size(blockL,2);
        end
        
        %%% generate one training episode coch
        function [frmL,frmR,nFrm] = genOneEpisodeCoch(this,param,ind)
            y_all = this.timit_train{param.audio_idx(ind),1};
            y = y_all(param.audio_bgn(ind,1)+(1:param.audio_len));
            bi = this.sofa.spatMono(y,this.locs_list(:,param.locs_rand(ind)),param.hrtf,param.subject);
            [frmL,frmR] = makeRM(bi);
            nFrm = size(frmL,2);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%% Gen One Episode %%%%%%%%%%%%%%%%%%%%%%%%
        function [blockL,blockR,nFrm] = genOneEpisodeCoch2(this,param,ind)
            y_all = this.timit_train{param.audio_idx(ind),1};
            y = y_all(param.audio_bgn(ind,1)+(1:param.audio_len));
            bi = this.sofa.spatMono(y,this.locs_list(:,param.locs_rand(ind)),param.hrtf,param.subject);
            gfb = gammatoneFilterBank([100 22000],param.nCh,44100);
            gmtL = gfb(bi(:,1));
            gmtR = gfb(bi(:,2));

            global patchLength patchStride;
            frmL = []; frmR = [];
            for i=0:patchStride:length(bi)-patchLength
                pL = pow2db(sum(gmtL(i+(1:patchLength),:).^2));
                pR = pow2db(sum(gmtR(i+(1:patchLength),:).^2));
                frmL = [frmL;pL]; frmR = [frmR;pR];
            end
            frmL = frmL';
            frmR = frmR';
            nFrm = size(frmL,2);

            global blockLength blockShift;
            blockL = []; blockR = [];
            for i=0:blockShift:nFrm-blockLength
                bL = reshape(frmL(:,i+(1:blockLength)),[],1);
                bR = reshape(frmR(:,i+(1:blockLength)),[],1);
                blockL = [blockL,bL];
                blockR = [blockR,bR];
            end
            nFrm = size(blockL,2);            
        end

        function [frmL, frmR, nFrm] = genOneTrainEpisodeCochIOSR (this, param, ind)
            y_all = this.timit_train{param.audio_idx(ind),1};
            y = y_all(param.audio_bgn(ind,1)+(1:param.audio_len));

            bi = this.sofa.spatMono(y,this.locs_list(:,param.locs_rand(ind)),param.hrtf,param.subject);
            
            frmL = audio2cochlIOSR(...
                bi(:,2), ...
                this.fs, 100, 20000, 128, ...
                8, 4 ...
            );
            frmR = audio2cochlIOSR(...
                bi(:,1), ...
                this.fs, 100, 20000, 128, ...
                8, 4 ...
            );
            nFrm = size(frmL, 2);

            assert(nFrm == size(frmR, 2));
        end

        function [frmL, frmR, nFrm] = genOneTestEpisodeCochIOSR (this, param, ind)
            y_all = this.timit_test{param.audio_idx(ind),1};
            y = y_all(param.audio_bgn(ind,1)+(1:param.audio_len));

            bi = this.sofa.spatMono(y,this.locs_list(:,param.locs_rand(ind)),param.hrtf,param.subject);
            
            frmL = audio2cochlIOSR(...
                bi(:,2), ...
                this.fs, 100, 20000, 128, ...
                8, 4 ...
            );
            frmR = audio2cochlIOSR(...
                bi(:,1), ...
                this.fs, 100, 20000, 128, ...
                8, 4 ...
            );
            nFrm = size(frmL, 2);

            assert(nFrm == size(frmR, 2));
        end

        function [blockL,blockR,nFrm] = genOneEpisodeRM(this,param,ind)
            y_all = this.timit_train{param.audio_idx(ind),1};
            y = y_all(param.audio_bgn(ind,1)+(1:param.audio_len));
            bi = this.sofa.spatMono(y,this.locs_list(:,param.locs_rand(ind)),param.hrtf,param.subject);
            [frmL,frmR,nFrm] = compRateMap(bi);
            global blockLength blockShift;
            blockL = []; blockR = [];
            for i=0:blockShift:nFrm-blockLength
                bL = reshape(frmL(:,i+(1:blockLength)),[],1);
                bR = reshape(frmR(:,i+(1:blockLength)),[],1);
                blockL = [blockL,bL];
                blockR = [blockR,bR];
            end
            nFrm = size(blockL,2);            
        end        
        
        %%% generate one training episode coch
        function [frmL,frmR,nFrm] = genOneEpisodeSpec(this,param,ind)
            y_all = this.timit_train{param.audio_idx(ind),1};
            y = y_all(param.audio_bgn(ind,1)+(1:param.audio_len));
            bi = this.sofa.spatMono(y,this.locs_list(:,param.locs_rand(ind)),param.hrtf,param.subject);
            [frmL,frmR,nFrm] = makeSpec(bi);
%             nFrm = size(frmL,2);
        end
        
        function [frmL,frmR,nFrm] = genOneEpisodeFM(this,param,ind)
            y = this.genStimuli('FM',200e-3);
            bi = this.sofa.spatMono(y,this.locs_list(:,param.locs_rand(ind)),param.hrtf,param.subject);
            [frmL,frmR,nFrm] = this.genFrame(bi);
        end
        
        function [frmL,frmR,nFrm] = genOneEpisodeGWN(this,param,loc)
            y = this.genStimuli('GWN',200e-3);
            bi = this.sofa.spatMono(y,loc,param.hrtf,param.subject);
            [frmL,frmR,nFrm] = this.genFrame(bi);
        end
        
        %%%
        function [frmL,frmR,nFrm] = spatIntMono(this,y,param,loc)
            bi = this.sofa.filterMono(y,loc,param.hrtf,param.subject);
            [frmL,frmR,nFrm] = this.genFrame(bi);
        end
        
        function [frmL,frmR,nFrm] = spatMono(this,y,param,loc)
            bi = this.sofa.spatMono(y,loc,param.hrtf,param.subject);
            [frmL,frmR,nFrm] = this.genFrame(bi);
        end
        function [bi] = spatMono_bi(this,y,param,loc)
            bi = this.sofa.spatMono(y,loc,param.hrtf,param.subject);            
        end        
        % Generate training data
        function [XTrain, YTrain] = genTrainAudio(this,netTrainParam)
            disp('Generating training data...');
            XTrain = []; YTrain = [];
            upd = textprogressbar(netTrainParam.max_iter);
            for i_iter = 1:netTrainParam.max_iter
                y_all = this.timit_train{netTrainParam.audio_idx(i_iter),1};
                y = y_all(netTrainParam.audio_bgn(i_iter)+(1:netTrainParam.audio_len));
                for i_tg = 1:this.locs_num
                    bi = this.sofa.spatMono(y,this.locs_list(:,i_tg),...
                        netTrainParam.hrtf,netTrainParam.subject);
                    [frmL,frmR,nFrm] = this.genFrame(bi);
                    resp = [frmL;frmR];
                    XTrain = [XTrain,resp]; %#ok<*AGROW>
                    YTrain = [YTrain,i_tg*ones(1,nFrm)];
                end
                upd(i_iter);
            end  
        end               
        
        function [XTest,YTest] = genTestAudio(this,netTestParam)
            disp('Generating testing data...');
            XTest = cell(netTestParam.max_iter,this.locs_num);
            YTest = cell(netTestParam.max_iter,this.locs_num);
            upd = textprogressbar(netTestParam.max_iter);
            for i_iter = 1:netTestParam.max_iter
                y_all = this.timit_test{netTestParam.audio_idx(i_iter),1};
                y = y_all(netTestParam.audio_bgn(i_iter)+(1:netTestParam.audio_len));
                for i_tg = 1:this.test_locs_num
                    bi = this.sofa.spatMono(y,this.test_locs_list(:,i_tg),...
                        netTestParam.hrtf,netTestParam.subject);
                    [frmL,frmR,~] = this.genFrame(bi);
                    XTest{i_iter,i_tg} = [frmL;frmR];
                    YTest{i_iter,i_tg} = this.locs_list(:,i_tg);
                end
                upd(i_iter);
            end
        end

        function [X, Y] = genGwn (...
            this, ...
            locs_list, sample_size, audio_len, fs, ...
            hrtf_dataset, hrtf_subject, ...
            gwn_seed, ...
            do_bandpass, lb, ub ...
        )
            rng(gwn_seed);
            this.stiGenerator.reset_gwn(gwn_seed);
            X = cell(sample_size, 2);
            Y = cell(sample_size, 1);

            tpb = textprogressbar(sample_size, "showremtime", true);
            for i_iter = 1:sample_size
                y = this.genStimuli('GWN',audio_len / fs);

                if do_bandpass
                    hf = design(fdesign.bandpass('N,F3dB1,F3dB2',4,lb,ub,fs));
                    y = filter(hf, y);
                end

                loc_idx = randi(length(locs_list));
                loc = locs_list(:, loc_idx);
                bi = this.sofa.spatMono(y, loc, hrtf_dataset, hrtf_subject);

                frmL = bi(:,2);
                frmR = bi(:,1);

                X{i_iter}{1} = frmL;
                X{i_iter}{2} = frmR;
                Y{i_iter} = loc_idx;

                tpb(i_iter);
            end
        end

        function [X, Y] = genGwnIosr (...
            this, ...
            locs_list, sample_size, audio_len, fs, ...
            hrtf_dataset, hrtf_subject, ...
            gwn_seed, ...
            do_bandpass, lb, ub ...
        )
            rng(gwn_seed);
            this.stiGenerator.reset_gwn(gwn_seed);
            X = cell(sample_size, 2);
            Y = cell(sample_size, 1);

            tpb = textprogressbar(sample_size, "showremtime", true);
            for i_iter = 1:sample_size
                y = this.genStimuli('GWN',audio_len / fs);

                if do_bandpass
                    hf = design(fdesign.bandpass('N,F3dB1,F3dB2',4,lb,ub,fs));
                    y = filter(hf, y);
                end

                loc_idx = randi(length(locs_list));
                loc = locs_list(:, loc_idx);
                bi = this.sofa.spatMono(y, loc, hrtf_dataset, hrtf_subject);

                frmL = audio2cochlIOSR(...
                    bi(:,2), ...
                    fs, 100, 20000, 128, ...
                    8, 4 ...
                );
                frmR = audio2cochlIOSR(...
                    bi(:,1), ...
                    fs, 100, 20000, 128, ...
                    8, 4 ...
                );
                nFrm = size(frmL, 2);

                assert(nFrm == size(frmR, 2));

                X{i_iter}{1} = frmL;
                X{i_iter}{2} = frmR;
                Y{i_iter} = loc_idx;

                tpb(i_iter);
            end
        end
        
        function [XTrain,YTrain] = genTrainGWN(this,netTrainParam)
            this.stiGenerator.reset_gwn(netTrainParam.gwn_seed);            
            XTrain = []; YTrain = [];
            upd = textprogressbar(netTrainParam.max_iter);
            for i_iter = 1:netTrainParam.max_iter
                y = this.genStimuli('GWN',netTrainParam.audio_len/this.fs);
                if isfield(netTrainParam,'pass_band')
                    if ~isempty(netTrainParam.pass_band)
                        y = btwFilter(y,netTrainParam.pass_band);
                    end
                end
                for i_tg = 1:this.locs_num
                    bi = this.sofa.spatMono(y,this.locs_list(:,i_tg),...
                        netTrainParam.hrtf, netTrainParam.subject);
                    [frmL,frmR,nFrm] = this.genFrame(bi);
                    resp = [frmL;frmR];
                    XTrain = [XTrain,resp];
                    YTrain = [YTrain,i_tg*ones(1,nFrm)];
                end
                upd(i_iter);
            end 
        end
        
        function [XTrain,YTrain] = genTrainGWNCoch(this,netTrainParam)
            this.stiGenerator.reset_gwn(netTrainParam.gwn_seed);            
            XTrain = []; YTrain = [];
            patch_length = 10; patch_ove = 1;
            upd = textprogressbar(netTrainParam.max_iter);
            for i_iter = 1:netTrainParam.max_iter                
                y = this.genStimuli('GWN',netTrainParam.audio_len/this.fs);
                for i_tg = 1:this.locs_num
                    bi = this.sofa.spatMono(y,this.locs_list(:,i_tg),...
                        netTrainParam.hrtf, netTrainParam.subject);
                    [frmL,frmR,nFrm] = makeRM(bi);
                    frNorm = normalize([frmL;frmR]);                    
                    XTrain = [XTrain,frNorm];
                    YTrain = [YTrain,i_tg*ones(1,size(frNorm,2))];
                end
                upd(i_iter);
            end 
        end 
        
        function [XTrain,YTrain] = genTrainGWNCoch2(this,netTrainParam)
            this.stiGenerator.reset_gwn(netTrainParam.gwn_seed);
            XTrain = []; YTrain = [];
            upd = textprogressbar(netTrainParam.max_iter);            
            gfb = gammatoneFilterBank([100 22000],netTrainParam.nCh,44100);
            blockL = []; blockR = [];
            global blockLength blockShift;
            global patchLength patchStride;
            for i_iter = 1:netTrainParam.max_iter
                y = this.genStimuli('GWN',netTrainParam.audio_len/this.fs);
                for i_tg = 1:this.locs_num
                    bi = this.sofa.spatMono(y,this.locs_list(:,i_tg),...
                        netTrainParam.hrtf, netTrainParam.subject);
                    gmtL = gfb(bi(:,1));
                    gmtR = gfb(bi(:,2));
                    frmL = []; frmR = [];
                    for i=0:patchStride:length(bi)-patchLength
                        pL = pow2db(sum(gmtL(i+(1:patchLength),:).^2));
                        pR = pow2db(sum(gmtR(i+(1:patchLength),:).^2));
                        frmL = [frmL;pL]; frmR = [frmR;pR];
                    end
                    frmL = frmL';
                    frmR = frmR';
                    nFrm = size(frmL,2);
                    for i=0:blockShift:nFrm-blockLength
                        bL = reshape(frmL(:,i+(1:blockLength)),[],1);
                        bR = reshape(frmR(:,i+(1:blockLength)),[],1);
                        blockL = [blockL,bL];
                        blockR = [blockR,bR];
                        YTrain = [YTrain,i_tg];
                    end
                end
                upd(i_iter);
            end
            XTrain = normalize_data([blockL;blockR]);
        end

%         for cochleagram
        function [XTest,YTest] = genTestGWNCoch2(this,netTestParam)
            XTest = cell(netTestParam.max_iter,this.locs_num);
            YTest = cell(netTestParam.max_iter,this.locs_num);
            upd = textprogressbar(netTestParam.max_iter);
            global blockLength blockShift;
            global patchLength patchStride n_channel;
            gfb = gammatoneFilterBank([100 22000],n_channel,44100);
            for i_iter = 1:netTestParam.max_iter
                y = this.genStimuli('GWN',netTestParam.audio_dur);
                if ~isempty(netTestParam.pass_band), y = btwFilter(y,netTestParam.pass_band); end
                for i_tg = 1:this.test_locs_num
                    bi = this.sofa.spatMono(y,this.test_locs_list(:,i_tg),...
                        netTestParam.hrtf,netTestParam.subject);
                    
                    gmtL = gfb(bi(:,1));
                    gmtR = gfb(bi(:,2));
                    frmL = []; frmR = [];
                    for i=0:patchStride:length(bi)-patchLength
                        pL = pow2db(sum(gmtL(i+(1:patchLength),:).^2));
                        pR = pow2db(sum(gmtR(i+(1:patchLength),:).^2));
                        frmL = [frmL;pL]; frmR = [frmR;pR];
                    end
                    frmL = frmL';
                    frmR = frmR';
                    nFrm = size(frmL,2);
                    blockL = []; blockR = [];
                    for i=0:blockShift:nFrm-blockLength
                        bL = reshape(frmL(:,i+(1:blockLength)),[],1);
                        bR = reshape(frmR(:,i+(1:blockLength)),[],1);
                        blockL = [blockL,bL];
                        blockR = [blockR,bR];                        
                    end
                    XTest{i_iter,i_tg} = normalize_data([blockL;blockR]);
                    YTest{i_iter,i_tg} = this.test_locs_list(:,i_tg);
                end
                upd(i_iter);
            end
        end         

        
        %%% FOR Rate Map
        function [XTrain,YTrain] = genTrainGWNRM(this,netTrainParam)
            this.stiGenerator.reset_gwn(netTrainParam.gwn_seed);
            XTrain = []; YTrain = [];
            upd = textprogressbar(netTrainParam.max_iter);            
            blockL = []; blockR = [];
            global blockLength blockShift;
            for i_iter = 1:netTrainParam.max_iter
                y = this.genStimuli('GWN',netTrainParam.audio_len/this.fs);
                for i_tg = 1:this.locs_num
                    bi = this.sofa.spatMono(y,this.locs_list(:,i_tg),...
                        netTrainParam.hrtf, netTrainParam.subject);
                    [frmL,frmR,nFrm] = compRateMap(bi);
                    for i=0:blockShift:nFrm-blockLength
                        bL = reshape(frmL(:,i+(1:blockLength)),[],1);
                        bR = reshape(frmR(:,i+(1:blockLength)),[],1);
                        blockL = [blockL,bL];
                        blockR = [blockR,bR];
                        YTrain = [YTrain,i_tg];
                    end
                end
                upd(i_iter);
            end
            XTrain = normalize_data([blockL;blockR]);
        end
        
%         for cochleagram
        function [XTest,YTest] = genTestGWNRM(this,netTestParam)
            XTest = cell(netTestParam.max_iter,this.locs_num);
            YTest = cell(netTestParam.max_iter,this.locs_num);
            upd = textprogressbar(netTestParam.max_iter);
            global blockLength blockShift;
            for i_iter = 1:netTestParam.max_iter
                y = this.genStimuli('GWN',netTestParam.audio_dur);
                if ~isempty(netTestParam.pass_band), y = btwFilter(y,netTestParam.pass_band); end
                for i_tg = 1:this.test_locs_num
                    bi = this.sofa.spatMono(y,this.test_locs_list(:,i_tg),...
                        netTestParam.hrtf,netTestParam.subject);
                    [frmL,frmR,nFrm] = compRateMap(bi);
                    blockL = []; blockR = [];
                    for i=0:blockShift:nFrm-blockLength
                        bL = reshape(frmL(:,i+(1:blockLength)),[],1);
                        bR = reshape(frmR(:,i+(1:blockLength)),[],1);
                        blockL = [blockL,bL];
                        blockR = [blockR,bR];                        
                    end
                    XTest{i_iter,i_tg} = normalize_data([blockL;blockR]);
                    YTest{i_iter,i_tg} = this.test_locs_list(:,i_tg);
                end
                upd(i_iter);
            end
        end           
% %         function [blockL,blockR,nFrm] = genOneEpisodeCoch2(this,param,ind)
% %             y_all = this.timit_train{param.audio_idx(ind),1};
% %             y = y_all(param.audio_bgn(ind,1)+(1:param.audio_len));
% %             bi = this.sofa.spatMono(y,this.locs_list(:,param.locs_rand(ind)),param.hrtf,param.subject);
% %             gfb = gammatoneFilterBank([100 22000],param.nCh,44100);
% %             gmtL = gfb(bi(:,1));
% %             gmtR = gfb(bi(:,2));
% %             patchLength = 20e-3*this.fs;
% %             patchStride = 10e-3*this.fs;
% %             frmL = []; frmR = [];
% %             for i=0:patchStride:length(bi)-patchLength
% %                 pL = pow2db(sum(gmtL(i+(1:patchLength),:).^2));
% %                 pR = pow2db(sum(gmtR(i+(1:patchLength),:).^2));
% %                 frmL = [frmL;pL]; frmR = [frmR;pR];
% %             end
% %             frmL = frmL';
% %             frmR = frmR';
% %             nFrm = size(frmL,2);
% %             blockLength = 10; blockShift = 1;
% %             blockL = []; blockR = [];
% %             for i=0:blockShift:nFrm-blockLength
% %                 bL = reshape(frmL(:,i+(1:blockLength)),[],1);
% %                 bR = reshape(frmR(:,i+(1:blockLength)),[],1);
% %                 blockL = [blockL,bL];
% %                 blockR = [blockR,bR];
% %             end
% %             nFrm = size(blockL,2);            
% %         end

% %         function [XTrain,YTrain] = genTrainGWNCoch(this,netTrainParam)
% %             this.stiGenerator.reset_gwn(netTrainParam.gwn_seed);            
% %             XTrain = []; YTrain = [];
% %             patch_length = 10; patch_ove = 1;
% %             upd = textprogressbar(netTrainParam.max_iter);
% %             for i_iter = 1:netTrainParam.max_iter                
% %                 y = this.genStimuli('GWN',netTrainParam.audio_len/this.fs);
% %                 for i_tg = 1:this.locs_num
% %                     bi = this.sofa.spatMono(y,this.locs_list(:,i_tg),...
% %                         netTrainParam.hrtf, netTrainParam.subject);
% %                     [frmL,frmR,nFrm] = makeRM(bi);
% %                     rL = []; rR = [];
% %                     for frmInd = 0:patch_ove:size(frmL,2)-patch_length
% %                         rmL = reshape(frmL(:,frmInd+(1:patch_length)),[],1);
% %                         rmR = reshape(frmR(:,frmInd+(1:patch_length)),[],1);
% %                         single_len = length(rmL);
% %                         rm = normalize([rmL;rmR]);
% %                         rL = [rL,rm(1:single_len,1)];
% %                         rR = [rR,rm(single_len+1:end,1)];
% %                     end
% %                     resp = [rL;rR];
% %                     XTrain = [XTrain,resp];
% %                     YTrain = [YTrain,i_tg*ones(1,size(resp,2))];
% %                 end
% %                 upd(i_iter);
% %             end 
% %         end     
        
%         function [XTrain,YTrain] = genTrainGWNSpec(this,netTrainParam)
%             this.stiGenerator.reset_gwn(netTrainParam.gwn_seed);            
%             XTrain = []; YTrain = [];
%             patch_length = 10; patch_ove = 1;
%             upd = textprogressbar(netTrainParam.max_iter);
%             for i_iter = 1:netTrainParam.max_iter                
%                 y = this.genStimuli('GWN',netTrainParam.audio_len/this.fs);
%                 for i_tg = 1:this.locs_num
%                     bi = this.sofa.spatMono(y,this.locs_list(:,i_tg),...
%                         netTrainParam.hrtf, netTrainParam.subject);
%                     [frmL,frmR,nFrm] = makeSpec(bi);
%                     rL = []; rR = [];
%                     for frmInd = 0:patch_ove:size(frmL,2)-patch_length
%                         rmL = reshape(frmL(:,frmInd+(1:patch_length)),[],1);
%                         rmR = reshape(frmR(:,frmInd+(1:patch_length)),[],1);
%                         single_len = length(rmL);
%                         rm = normalize([rmL;rmR]);
%                         rL = [rL,rm(1:single_len,1)];
%                         rR = [rR,rm(single_len+1:end,1)];
%                     end
%                     resp = [rL;rR];
%                     XTrain = [XTrain,resp];
%                     YTrain = [YTrain,i_tg*ones(1,size(resp,2))];
%                 end
%                 upd(i_iter);
%             end 
%         end 
        
        function [XTest,YTest] = genTestGWN(this,netTestParam)
            XTest = cell(netTestParam.max_iter,this.locs_num);
            YTest = cell(netTestParam.max_iter,this.locs_num);
            upd = textprogressbar(netTestParam.max_iter);
            for i_iter = 1:netTestParam.max_iter
                y = this.genStimuli('GWN',netTestParam.audio_dur);
                if ~isempty(netTestParam.pass_band), y = btwFilter(y,netTestParam.pass_band); end
                for i_tg = 1:this.test_locs_num
                    bi = this.sofa.spatMono(y,this.test_locs_list(:,i_tg),...
                        netTestParam.hrtf,netTestParam.subject);
                    [frmL,frmR,~] = this.genFrame(bi);
                    XTest{i_iter,i_tg} = [frmL;frmR];
                    YTest{i_iter,i_tg} = this.test_locs_list(:,i_tg);
                end
                upd(i_iter);
            end
        end        
        
%         function [XTest,YTest] = genTestGWNCoch(this,netTestParam)
%             XTest = cell(netTestParam.max_iter,this.locs_num);
%             YTest = cell(netTestParam.max_iter,this.locs_num);
%             upd = textprogressbar(netTestParam.max_iter);
%             for i_iter = 1:netTestParam.max_iter
%                 y = this.genStimuli('GWN',netTestParam.audio_dur);
%                 if ~isempty(netTestParam.pass_band), y = btwFilter(y,netTestParam.pass_band); end
%                 for i_tg = 1:this.test_locs_num
%                     bi = this.sofa.spatMono(y,this.test_locs_list(:,i_tg),...
%                         netTestParam.hrtf,netTestParam.subject);
%                     [frmL,frmR,nFrm] = makeRM(bi);
%                     rm = normalize([frmL;frmR]);                    
%                     XTest{i_iter,i_tg} = rm;
%                     YTest{i_iter,i_tg} = this.test_locs_list(:,i_tg);
%                 end
%                 upd(i_iter);
%             end
%         end 
        function [XTest,YTest] = genTestGWNCoch(this,netTestParam)
            XTest = cell(netTestParam.max_iter,this.locs_num);
            YTest = cell(netTestParam.max_iter,this.locs_num);
            upd = textprogressbar(netTestParam.max_iter);
            patch_length = 10; patch_ove = 1;
            for i_iter = 1:netTestParam.max_iter
                y = this.genStimuli('GWN',netTestParam.audio_dur);
                if ~isempty(netTestParam.pass_band), y = btwFilter(y,netTestParam.pass_band); end
                for i_tg = 1:this.test_locs_num
                    bi = this.sofa.spatMono(y,this.test_locs_list(:,i_tg),...
                        netTestParam.hrtf,netTestParam.subject);
                    [frmL,frmR,nFrm] = makeRM(bi);
                    rL = []; rR = [];
                    for frmInd = 0:patch_ove:size(frmL,2)-patch_length
                        rmL = reshape(frmL(:,frmInd+(1:patch_length)),[],1);
                        rmR = reshape(frmR(:,frmInd+(1:patch_length)),[],1);
                        single_len = length(rmL);
                        rm = normalize([rmL;rmR]);
                        rL = [rL,rm(1:single_len,1)];
                        rR = [rR,rm(single_len+1:end,1)];
                    end
                    XTest{i_iter,i_tg} = [rL;rR];
                    YTest{i_iter,i_tg} = this.test_locs_list(:,i_tg);
                end
                upd(i_iter);
            end
        end 
        
        function [XTest,YTest] = genTestGWNSpec(this,netTestParam)
            XTest = cell(netTestParam.max_iter,this.locs_num);
            YTest = cell(netTestParam.max_iter,this.locs_num);
            upd = textprogressbar(netTestParam.max_iter);
            patch_length = 10; patch_ove = 1;
            for i_iter = 1:netTestParam.max_iter
                y = this.genStimuli('GWN',netTestParam.audio_dur);
                if ~isempty(netTestParam.pass_band), y = btwFilter(y,netTestParam.pass_band); end
                for i_tg = 1:this.test_locs_num
                    bi = this.sofa.spatMono(y,this.test_locs_list(:,i_tg),...
                        netTestParam.hrtf,netTestParam.subject);
                    [frmL,frmR,nFrm] = makeSpec(bi);
                    rL = []; rR = [];
                    for frmInd = 0:patch_ove:size(frmL,2)-patch_length
                        rmL = reshape(frmL(:,frmInd+(1:patch_length)),[],1);
                        rmR = reshape(frmR(:,frmInd+(1:patch_length)),[],1);
                        single_len = length(rmL);
                        rm = normalize([rmL;rmR]);
                        rL = [rL,rm(1:single_len,1)];
                        rR = [rR,rm(single_len+1:end,1)];
                    end
                    XTest{i_iter,i_tg} = [rL;rR];
                    YTest{i_iter,i_tg} = this.test_locs_list(:,i_tg);
                end
                upd(i_iter);
            end
        end 
        
        function [XTest,YTest] = genTestCT(this, netTestParam)
            XTest = cell(netTestParam.max_iter,this.locs_num);
            YTest = cell(netTestParam.max_iter,this.locs_num);
            upd = textprogressbar(netTestParam.max_iter);
            for i_iter = 1:netTestParam.max_iter
                y_click = this.genStimuli('GWN',netTestParam.click_dur);
                y_int = zeros(round((netTestParam.click_int-netTestParam.click_dur)*this.fs),1);
                y_single = [y_click;y_int];
                y = repmat(y_single,floor(netTestParam.audio_dur/netTestParam.click_int),1);
                for i_tg = 1:this.test_locs_num
                    bi = this.sofa.spatMono(y,this.test_locs_list(:,i_tg),...
                        netTestParam.hrtf,netTestParam.subject);
                    [frmL,frmR,~] = this.genFrame(bi);
                    XTest{i_iter,i_tg} = [frmL;frmR];
                    YTest{i_iter,i_tg} = this.locs_list(:,i_tg);
                end
                upd(i_iter);
            end
        end
        
        function [XTest,YTest] = genTestSingleHRTF(this,netTestParam, fb_locs, LorR)
            if nargin<3
                fb_locs = combvec([0],[0 180]); LorR = 2; 
            elseif isempty(fb_locs)
                fb_locs = combvec([0],[0 180]);
            end
            fb_locs_num = size(fb_locs,2);
            XTest = cell(netTestParam.max_iter,fb_locs_num);
            YTest = cell(netTestParam.max_iter,fb_locs_num);
            upd = textprogressbar(netTestParam.max_iter);
            for i_iter = 1:netTestParam.max_iter
                y = this.genStimuli('GWN',netTestParam.audio_dur);
                if ~isempty(netTestParam.pass_band), y = btwFilter(y,netTestParam.pass_band); end
                for i_tg = 1:fb_locs_num
                    bi = this.sofa.spatMono(y,fb_locs(:,i_tg),...
                        netTestParam.hrtf,netTestParam.subject);
                    switch LorR
                        case 2
                            bi(:,2) = bi(:,1);
                        case 1
                            bi(:,1) = bi(:,2);
                        case 0
                    end
                    [frmL,frmR,~] = this.genFrame(bi);
                    XTest{i_iter,i_tg} = [frmL;frmR];
                    YTest{i_iter,i_tg} = fb_locs(:,i_tg);
                end
                upd(i_iter);
            end
        end    
        
        
        % Load TIMIT dataset(not required for other stimuli types)
        function loadTimit(this)
            timit_path = 'temp_data/timit_array.mat';
            if ~exist(timit_path,'file')
                load_TIMIT;
            end
            load(timit_path,'timit_train','timit_test');
            this.timit_train = timit_train; 
            this.timit_train_num = length(this.timit_train);
            this.timit_test = timit_test; %#ok<*PROP>
            this.timit_test_num = length(this.timit_test);
            if exist('sofaloaded.mat','file')
                load sofaloaded.mat;
                this.sofa = sofaloaded;
                clearvars sofaloaded;
            else
                this.sofa = SOFALoader;
            end
        end
        
        function [y] = genStimuli(this,sti_type,sti_dur)
            if nargin<2
                y = this.stiGenerator.genGWN(400e-3);            
            else
                switch(sti_type)
                    case('FM')
                        y = this.stiGenerator.genSweep(sti_dur);
                    case('GWN')
                        y = this.stiGenerator.genGWN(sti_dur);
                    case('FM_silent')
                        y = this.stiGenerator.genSweep(sti_dur,100);
                end
            end
        end
        
    end
end
