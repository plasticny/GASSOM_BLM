map_width = 10;
chunk_size = 10;

% save_folder = "chp4/cochleagram/result/" + map_width + "x" + map_width + "_sz" + chunk_size + "_sf1/";
% save_folder = "chp4/vecnorm/waveform/16x16_16ms/";

% gm = loadGassom([10, 10], 5e4, save_folder + "gsm.mat", 10, "kemar", 0);
% gm = loadWaveformGassom([16, 16], 5e4, save_folder + "gsm.mat", 16 / 1000);
% trained_dnn = load(save_folder + "dnn_1000_4000_lr_scheme.mat").trained_dnn;

generateTestSet(gm, 250, 1/6);
generateTestSet(gm, 2000, 1/6);
generateTestSet(gm, 4000, 1/6);
generateTestSet(gm, 250, 1/3);
generateTestSet(gm, 2000, 1/3);
generateTestSet(gm, 4000, 1/3);
generateTestSet(gm, 250, 1);
generateTestSet(gm, 2000, 1);
generateTestSet(gm, 4000, 1);
generateTestSet(gm, 250, 2);
generateTestSet(gm, 2000, 2);
generateTestSet(gm, 4000, 2);
% visualizeFilteredStimuli(gm);

%%%
% cochleagram
%%%
% [predicts_250_1_6, truths_250_1_6, resp_250_1_6, rms_250_1_6] = predictTestSet(gm, trained_dnn, "cache/dnnTestGwn_kemar_0_1900_bandpass_cf250_1_6.mat");
% [predicts_2000_1_6, truths_2000_1_6, resp_2000_1_6, rms_2000_1_6] = predictTestSet(gm, trained_dnn, "cache/dnnTestGwn_kemar_0_1900_bandpass_cf2000_1_6.mat");
% [predicts_4000_1_6, truths_4000_1_6, resp_4000_1_6, rms_4000_1_6] = predictTestSet(gm, trained_dnn, "cache/dnnTestGwn_kemar_0_1900_bandpass_cf4000_1_6.mat");
% [predicts_250_1_3, truths_250_1_3, resp_250_1_3, rms_250_1_3] = predictTestSet(gm, trained_dnn, "cache/dnnTestGwn_kemar_0_1900_bandpass_cf250_1_3.mat");
% [predicts_2000_1_3, truths_2000_1_3, resp_2000_1_3, rms_2000_1_3] = predictTestSet(gm, trained_dnn, "cache/dnnTestGwn_kemar_0_1900_bandpass_cf2000_1_3.mat");
% [predicts_4000_1_3, truths_4000_1_3, resp_4000_1_3, rms_4000_1_3] = predictTestSet(gm, trained_dnn, "cache/dnnTestGwn_kemar_0_1900_bandpass_cf4000_1_3.mat");
% [predicts_250_1, truths_250_1, resp_250_1, rms_250_1] = predictTestSet(gm, trained_dnn, "cache/dnnTestGwn_kemar_0_1900_bandpass_cf250_1.mat");
% [predicts_2000_1, truths_2000_1, resp_2000_1, rms_2000_1] = predictTestSet(gm, trained_dnn, "cache/dnnTestGwn_kemar_0_1900_bandpass_cf2000_1.mat");
% [predicts_4000_1, truths_4000_1, resp_4000_1, rms_4000_1] = predictTestSet(gm, trained_dnn, "cache/dnnTestGwn_kemar_0_1900_bandpass_cf4000_1.mat");
% [predicts_250_2, truths_250_2, resp_250_2, rms_250_2] = predictTestSet(gm, trained_dnn, "cache/dnnTestGwn_kemar_0_1900_bandpass_cf250_2.mat");
% [predicts_2000_2, truths_2000_2, resp_2000_2, rms_2000_2] = predictTestSet(gm, trained_dnn, "cache/dnnTestGwn_kemar_0_1900_bandpass_cf2000_2.mat");
% [predicts_4000_2, truths_4000_2, resp_4000_2, rms_4000_2] = predictTestSet(gm, trained_dnn, "cache/dnnTestGwn_kemar_0_1900_bandpass_cf4000_2.mat");

%%%
% waveform
%%%
% [predicts_250_1_6, truths_250_1_6, resp_250_1_6, rms_250_1_6] = predictTestSetWaveform(gm, trained_dnn, 250, 1/6, 16 / 1000);
% [predicts_2000_1_6, truths_2000_1_6, resp_2000_1_6, rms_2000_1_6] = predictTestSetWaveform(gm, trained_dnn, 2000, 1/6, 16 / 1000);
% [predicts_4000_1_6, truths_4000_1_6, resp_4000_1_6, rms_4000_1_6] = predictTestSetWaveform(gm, trained_dnn, 4000, 1/6, 16 / 1000);
% [predicts_250_1_3, truths_250_1_3, resp_250_1_3, rms_250_1_3] = predictTestSetWaveform(gm, trained_dnn, 250, 1/3, 16 / 1000);
% [predicts_2000_1_3, truths_2000_1_3, resp_2000_1_3, rms_2000_1_3] = predictTestSetWaveform(gm, trained_dnn, 2000, 1/3, 16 / 1000);
% [predicts_4000_1_3, truths_4000_1_3, resp_4000_1_3, rms_4000_1_3] = predictTestSetWaveform(gm, trained_dnn, 4000, 1/3, 16 / 1000);
% [predicts_250_1, truths_250_1, resp_250_1, rms_250_1] = predictTestSetWaveform(gm, trained_dnn, 250, 1, 16 / 1000);
% [predicts_2000_1, truths_2000_1, resp_2000_1, rms_2000_1] = predictTestSetWaveform(gm, trained_dnn, 2000, 1, 16 / 1000);
% [predicts_4000_1, truths_4000_1, resp_4000_1, rms_4000_1] = predictTestSetWaveform(gm, trained_dnn, 4000, 1, 16 / 1000);
% [predicts_250_2, truths_250_2, resp_250_2, rms_250_2] = predictTestSetWaveform(gm, trained_dnn, 250, 2, 16 / 1000);
% [predicts_2000_2, truths_2000_2, resp_2000_2, rms_2000_2] = predictTestSetWaveform(gm, trained_dnn, 2000, 2, 16 / 1000);
% [predicts_4000_2, truths_4000_2, resp_4000_2, rms_4000_2] = predictTestSetWaveform(gm, trained_dnn, 4000, 2, 16 / 1000);

% result_save_folder = "chp5/result/combine/";

%%%
% load saved result
%%%
% rms_data = load(result_save_folder + "rms.mat");
% rms_field_names = fieldnames(rms_data);
% for i  = 1:numel(rms_field_names)
%     eval(rms_field_names{i} + " = rms_data." + rms_field_names{i} + ";");
% end

% predicts_data = load(result_save_folder + "predicts.mat");
% predicts_field_names = fieldnames(predicts_data);
% for i = 1:numel(predicts_field_names)
%     eval(predicts_field_names{i} + " = predicts_data." + predicts_field_names{i} + ";");
% end
% truths_data = load(result_save_folder + "truths.mat");

% plots

rms_data = [
    rms_250_1_6, rms_2000_1_6, rms_4000_1_6;
    rms_250_1_3, rms_2000_1_3, rms_4000_1_3;
    rms_250_1, rms_2000_1, rms_4000_1;
    rms_250_2, rms_2000_2, rms_4000_2
];
figure;
bar(rms_data, "grouped");
title("Result of DNN trained with timit")
ylabel("RMS/deg");
xlabel("Bandwidth/octave")
xticklabels(["1/6" "1/3" "1" "2"])
legend({"CF:250Hz", "CF:2000Hz", "CF:4000Hz"});

% figure;
% confusionchart(truths_250_2, predicts_250_2);
% title("CF: 250Hz; bandwidth: 2 octave")
% figure;
% confusionchart(truths_2000_2, predicts_2000_2);
% title("CF: 2000Hz; bandwidth: 2 octave")
% figure;
% confusionchart(truths_4000_2, predicts_4000_2);
% title("CF: 4000Hz; bandwidth: 2 octave")

% save( ...
%     result_save_folder + "rms.mat", ...
%     "rms_250_1_6", "rms_250_1_3", "rms_250_1", "rms_250_2", ...
%     "rms_2000_1_6", "rms_2000_1_3", "rms_2000_1", "rms_2000_2", ...
%     "rms_4000_1_6", "rms_4000_1_3", "rms_4000_1", "rms_4000_2" ...
% )
% save( ...
%     result_save_folder + "predicts.mat", ...
%     "predicts_250_1_6", "predicts_250_1_6", "predicts_250_1", "predicts_250_2", ...
%     "predicts_2000_1_6", "predicts_2000_1_3", "predicts_2000_1", "predicts_2000_2", ...
%     "predicts_4000_1_6", "predicts_4000_1_3", "predicts_4000_1", "predicts_4000_2" ...
% )
% save( ...
%     result_save_folder + "truths.mat", ...
%     "truths_250_1_6", "truths_250_1_6", "truths_250_1", "truths_250_2", ...
%     "truths_2000_1_6", "truths_2000_1_3", "truths_2000_1", "truths_2000_2", ...
%     "truths_4000_1_6", "truths_4000_1_3", "truths_4000_1", "truths_4000_2" ...
% )
% save( ...
%     result_save_folder + "responses.mat", ...
%     "resp_250_1_6", "resp_250_1_6", "resp_250_1", "resp_250_2", ...
%     "resp_2000_1_6", "resp_2000_1_3", "resp_2000_1", "resp_2000_2", ...
%     "resp_4000_1_6", "resp_4000_1_3", "resp_4000_1", "resp_4000_2" ...
% )

function [gm] = loadGassom (topo_space, max_iter, gsm_path, chunk_size, hrtf_database, hrtf_subject)
    rng(49);

    % sofaloaded = SOFALoader;
    % save('sofaloaded.mat', 'sofaloaded');
    load temp_data/sofaloaded.mat;

    disp("init gassom start");
    fs = 44100;
    patch_dur = 0; % should be not neccessary in this chapter
    gm_param = {fs, patch_dur, topo_space, max_iter};
    gm = GASSOM_Model(gm_param, 128 * chunk_size * 2, "cochleagram", hrtf_database, hrtf_subject);
    gm.setHRTFandSubject(hrtf_database, hrtf_subject);
    disp("init gassom end");

    gm.gsm{1} = load(gsm_path).gsm;
end

function [gm] = loadWaveformGassom (topo_space, max_iter, gsm_path, patch_dur)
    rng(49);

    load temp_data/sofaloaded.mat;

    disp("init gassom start")
    fs = 44100;
    gm_param = {fs, patch_dur, topo_space, max_iter};
    gm = GASSOM_Model(gm_param, round(fs * patch_dur), "waveform");
    disp("init gassom end");

    gm.gsm{1} = load(gsm_path).gsm;
end

function [y] = filterStimuli (y, fs, cf, bw_oct)
    % apply rise-decay window
    window_size = fs * 20 / 1000;
    rise_factor = (cos(linspace(pi / 2, 0, window_size)) .^ 2)';
    decay_factor = flip(rise_factor);
    y(1:length(rise_factor)) = y(1:length(rise_factor)) .* rise_factor;
    y(end - length(decay_factor) + 1:end) = y(end - length(decay_factor) + 1:end) .* decay_factor;

    % butterworth
    factor = 2 ^ (bw_oct / 2);
    lb = cf / factor;
    ub = cf * factor;
    hf = design(fdesign.bandpass('N,F3dB1,F3dB2',4,lb,ub,fs));
    y = filter(hf, y);
end

function generateTestGwn (gm)
    gwn_seed = gm.netTestParam.gwn_seed;
    sample_size = gm.locs_num * 100;
    audio_len = round(200e-3 * gm.env.fs);
    fs = gm.env.fs;
    locs_list = gm.locs_list;

    rng(gwn_seed);
    gm.env.stiGenerator.reset_gwn(gwn_seed);
    X = cell(sample_size, 1);
    Y = cell(sample_size, 1);

    tpb = textprogressbar(sample_size, "showremtime", true);
    for i_iter = 1:sample_size
        gwn = gm.env.genStimuli('GWN',audio_len / fs);
        X{i_iter} = gwn;
        Y{i_iter} = randi(length(locs_list));
        tpb(i_iter);
    end

    save("chp5/cache/testSettings.mat", "X", "Y");
end

function generateTestSet (gm, cf, bw_oct)
    gwn = load("chp5/cache/testSettings.mat");
    gwnX = gwn.X;
    testX = cell(sample_size, 2);
    testY = gwn.Y;
    testSize = length(testY);

    gfb = gammatoneFilterBank([100 20000],128,44100);
    patchLength = floor(44100 * 8 / 1000);
    patchStride = floor(44100 * 4 / 1000);

    tpb = textprogressbar(testSize, 'showremtime', true);
    for i = 1:testSize
        gwn = gwnX{i};
        gt_loc_idx = testY{i};
        
        % filter the gwn and spatalize
        gwn = filterStimuli(gwn, gm.env.fs, cf, bw_oct);
        loc = gm.locs_list(:, gt_loc_idx);
        bi = gm.env.sofa.spatMono(gwn, loc, "kemar", 0);

        gmtL = gfb(bi(:,2));
        gmtR = gfb(bi(:,1));
        frmL = []; frmR = [];
        for j=0:patchStride:length(bi)-patchLength
            pL = pow2db(sum(gmtL(j+(1:patchLength),:).^2));
            pR = pow2db(sum(gmtR(j+(1:patchLength),:).^2));
            frmL = [frmL;pL]; frmR = [frmR;pR];
        end
        frmL = frmL';
        frmR = frmR';

        testX{i}{1} = frmL;
        testX{i}{2} = frmR;

        tpb(i);
    end

    save("cache/dnnTestGwn_kemar_0_1900_bandpass_cf" + cf + "_" + bw_oct + ".mat", "testX", "testY");
end

% function [predicts, truths, cumu_resp, err] = predictTestSet (gm, trained_dnn, dataset_path)
function [predicts, truths, cumu_resp, err] = predictTestSet (gm, trained_dnn, cf, bw_oct)
    chunk_size = 10;

    % test_data = load(dataset_path);
    % testX = test_data.testX;
    % testY = test_data.testY;
    % testSize = length(testY);

    testSetting = load("chp5/cache/testSettings.mat");
    testX = testSetting.X;
    testY = testSetting.Y;
    testSize = length(testY);

    predicts = zeros(testSize, 1);
    truths = zeros(testSize, 1);
    cumu_resp = zeros(gm.locs_num, prod(gm.topo_space));

    tpb = textprogressbar(testSize, 'showremtime', true);
    for i = 1:testSize
        gwn = testX{i};
        gt_loc_idx = testY{i};
        
        % filter the gwn and spatalize
        gwn = filterStimuli(gwn, gm.env.fs, cf, bw_oct);
        loc = gm.locs_list(:, gt_loc_idx);

        bi = gm.env.sofa.spatMono(gwn, loc, "kemar", 0);

        frmL = audio2cochlIOSR(...
            bi(:,2), ...
            gm.env.fs, 100, 20000, 128, ...
            8, 4 ...
        );
        frmR = audio2cochlIOSR(...
            bi(:,1), ...
            gm.env.fs, 100, 20000, 128, ...
            8, 4 ...
        );
        % frmL = testX{i}{1};
        % frmR = testX{i}{2};
        nFrm = size(frmL, 2);

        % get response from gassom
        nChk = length(0:1:nFrm-chunk_size);
        responses = zeros(nChk, prod(gm.topo_space));   
        for j = 1:size(responses,1)
            chkL = frmL(:,j:(chunk_size + j - 1));
            chkR = frmR(:,j:(chunk_size + j - 1));

            % single_len = size(chkL,1);
            % rm = normalize([chkL;chkR]);
            % rm = rm + abs(min(rm, [], "all"));
            % chkL = rm(1:single_len,:);
            % chkR = rm(single_len+1:end,:);

            x = [reshape(chkL, [], 1);reshape(chkR, [], 1)];
            x = x-ones(size(x,1),1)*mean(x,1);
            X = bsxfun(@rdivide, x, sqrt(sum(x.^2))+eps); 
            % x = x / norm(x);

            res = gm.getResponse(X);

            responses(j, :) = res;
            cumu_resp(gt_loc_idx, :) = cumu_resp(gt_loc_idx, :) + reshape(res, 1, []);
        end

        % predict response
        cum_predicted = zeros(1, length(gm.locs_list));
        for i_chk = 1:nChk
            res = responses(i_chk, :);
            predicted = predict(trained_dnn, res);
            cum_predicted = cum_predicted + predicted;
        end
        [~, p] = max(cum_predicted);

        predicts(i) = p;
        truths(i) = gt_loc_idx;

        tpb(i);
    end

    err = rms(truths, predicts);
end

function [predicts, truths, cumu_resp, err] = predictTestSetWaveform (...
    gm, trained_dnn, cf, bw_oct, patch_dur ...
)
    patch_len = round(gm.env.fs * patch_dur);
    patch_stride = round(patch_len * 0.1);    

    testSetting = load("chp5/cache/testSettings.mat");
    testX = testSetting.X;
    testY = testSetting.Y;
    testSize = length(testY);

    predicts = zeros(testSize, 1);
    truths = zeros(testSize, 1);
    cumu_resp = zeros(gm.locs_num, prod(gm.topo_space));

    tpb = textprogressbar(testSize, 'showremtime', true);
    for i = 1:testSize
        gwn = testX{i};
        gt_loc_idx = testY{i};
        
        % filter the gwn and spatalize
        gwn = filterStimuli(gwn, gm.env.fs, cf, bw_oct);
        loc = gm.locs_list(:, gt_loc_idx);
        bi = gm.env.sofa.spatMono(gwn, loc, "kemar", 0);
        frmL = bi(:, 2);
        frmR = bi(:, 1);

        % get response from gassom
        chkStart = 1:patch_stride:size(frmL, 1) - patch_len;
        nChk = length(chkStart);
        responses = zeros(nChk, prod(gm.topo_space));
        for j = 1:size(responses,1)
            s = chkStart(j);
            chkL = frmL(s:(s + patch_len - 1), 1);
            chkR = frmR(s:(s + patch_len - 1), 1);
            rm = normalize([chkL;chkR]);
            rm = rm / norm(rm);

            res = gm.getResponse(rm);
            % res = normalize(res);
            responses(j, :) = res;
            cumu_resp(gt_loc_idx, :) = cumu_resp(gt_loc_idx, :) + reshape(res, 1, []);
        end

        % predict response
        cum_predicted = zeros(1, length(gm.locs_list));
        for i_chk = 1:nChk
            res = responses(i_chk, :);
            % res = normalize(res, "range");
            predicted = predict(trained_dnn, res);
            cum_predicted = cum_predicted + predicted;
        end
        [~, p] = max(cum_predicted);

        predicts(i) = p;
        truths(i) = gt_loc_idx;

        tpb(i);
    end

    err = rms(truths, predicts);
end

function [err] = rms (truths, predicts)
    % truths and predicts are location index
    assert(length(truths) == length(predicts));
    truth_azimuths = truths * 10 - 100;
    predict_azimuth = predicts * 10 - 100;
    err = sqrt(sum((predict_azimuth - truth_azimuths) .^ 2) / length(truths));
end

function visualizeFilteredStimuli (gm)
    figure;
    plot(y);

    figure;
    subplot(4,3,1);
    plot(filterStimuli(y, gm.env.fs, 250, 1/6));
    title("CF:250Hz");
    ylabel("1/6 octave")
    subplot(4,3,2);
    plot(filterStimuli(y, gm.env.fs, 2000, 1/6));
    title("CF:2000Hz");
    subplot(4,3,3);
    plot(filterStimuli(y, gm.env.fs, 4000, 1/6));
    title("CF:4000Hz");
    subplot(4,3,4);
    plot(filterStimuli(y, gm.env.fs, 250, 1/3));
    ylabel("1/3 octave");
    subplot(4,3,5);
    plot(filterStimuli(y, gm.env.fs, 2000, 1/3));
    subplot(4,3,6);
    plot(filterStimuli(y, gm.env.fs, 4000, 1/3));
    subplot(4,3,7);
    plot(filterStimuli(y, gm.env.fs, 250, 1));
    ylabel("1 octave");
    subplot(4,3,8);
    plot(filterStimuli(y, gm.env.fs, 2000, 1));
    subplot(4,3,9);
    plot(filterStimuli(y, gm.env.fs, 4000, 1));
    subplot(4,3,10);
    plot(filterStimuli(y, gm.env.fs, 250, 2));
    ylabel("2 octave");
    subplot(4,3,11);
    plot(filterStimuli(y, gm.env.fs, 2000, 2));
    xlabel("data point")
    subplot(4,3,12);
    plot(filterStimuli(y, gm.env.fs, 4000, 2));
end
