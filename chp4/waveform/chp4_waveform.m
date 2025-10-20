function [gm] = initGassom (topo_space, max_iter, patch_dur)
    rng(49);

    % sofaloaded = SOFALoader;
    % save('sofaloaded.mat', 'sofaloaded');
    load temp_data/sofaloaded.mat;

    disp("init gassom start")
    fs = 44100;
    gm_param = {fs, patch_dur, topo_space, max_iter};
    gm = GASSOM_Model(gm_param, round(fs * patch_dur), "waveform");
    disp("init gassom end");
end

function [gm] = loadGassom (topo_space, max_iter, gsm_path, patch_dur)
    gm = initGassom(topo_space, max_iter, patch_dur);
    gm.gsm{1} = load(gsm_path).gsm;
end

function generateDnnSamples (gm)
    disp("generating training samples");
    [trainX, trainY] = gm.env.genGwn(...
        gm.locs_list, gm.netTrainParam.max_iter, ...
        gm.netTrainParam.audio_len, gm.env.fs, ...
        gm.netTrainParam.hrtf, gm.netTrainParam.subject, ...
        gm.netTrainParam.gwn_seed...
    );
    disp("generating testing samples");
    [testX, testY] = gm.env.genGwn(...
        gm.locs_list, gm.netTestParam.max_iter, ...
        gm.netTestParam.audio_len, gm.env.fs, ...
        gm.netTestParam.hrtf, gm.netTestParam.subject, ...
        gm.netTestParam.gwn_seed...
    );
    save("temp_data/dnnTrainGwnWaveform.mat", "trainX", "trainY");
    save("temp_data/dnnTestGwnWaveform.mat", "testX", "testY");
end

function [trained_dnn, train_info] = trainDNN (gm, patch_dur)
    trainSamples = load("temp_data/dnnTrainGwnWaveform.mat");
    assert(gm.netTrainParam.max_iter == size(trainSamples.trainX, 1));

    XTrain = zeros(gm.netTrainParam.max_iter, prod(gm.topo_space));
    YTrain = categorical(cell2mat(trainSamples.trainY));

    rng(49);

    patch_len = round(44100 * patch_dur);
    patch_stride = round(patch_len * 0.1);

    for i = 1:gm.netTrainParam.max_iter
        frmL = trainSamples.trainX{i}{1};
        frmR = trainSamples.trainX{i}{2};

        startPt = 0:patch_stride:size(frmL, 1)-patch_len;
        j = startPt(randi(length(startPt)));
        chkL = frmL(j+(1:patch_len),:);
        chkR = frmR(j+(1:patch_len),:);

        rm = normalize_data([chkL;chkR]);
        res = gm.getResponse(rm);
        res = normalize(res, "range");
        XTrain(i, :) = res;
    end

    [dnn, training_option] = createDNN(prod(gm.topo_space), gm.locs_num, 3);
    [trained_dnn, train_info] = trainNetwork(XTrain, YTrain, dnn, training_option);
end

function [mae, chunk_mae, azimuth_predicts, azimuth_truths, cumu_resp] = testModel (...
    gm, trained_dnn, patch_dur ...
)
    testSamples = load("temp_data/dnnTestGwnWaveform.mat");
    assert(gm.netTestParam.max_iter == size(testSamples.testX, 1));

    ae = zeros(gm.netTestParam.max_iter, 1);
    azimuth_predicts = zeros(gm.netTestParam.max_iter, 1);
    azimuth_truths = zeros(gm.netTestParam.max_iter, 1);

    cumu_resp = zeros(gm.locs_num, prod(gm.topo_space));
    loc_cnt = zeros(gm.locs_num, 1);

    chunk_ae = [];

    patch_len = round(44100 * patch_dur);
    patch_stride = round(patch_len * 0.1);

    tpb = textprogressbar(gm.netTestParam.max_iter, 'showremtime', true);
    for i_iter = 1:gm.netTestParam.max_iter
        YTest = testSamples.testY{i_iter};
        frmL = testSamples.testX{i_iter}{1};
        frmR = testSamples.testX{i_iter}{2};

        nChk = length(0:patch_stride:size(frmL, 1)-patch_len);

        azimuth_truth = gm.locs_list(1, YTest(1));

        % get response from gassom
        responses = zeros(nChk, prod(gm.topo_space));
        for i = 1:size(responses,1)
            offset = (i - 1) * patch_stride;
            chkL = frmL((1 + offset):(patch_len + offset), :);
            chkR = frmR((1 + offset):(patch_len + offset), :);
            rm = normalize([chkL;chkR]);

            res = gm.getResponse(rm);

            loc_idx = (azimuth_truth + 100) / 10;
            cumu_resp(loc_idx, :) = cumu_resp(loc_idx, :) + res';
            loc_cnt(loc_idx) = loc_cnt(loc_idx) + 1;

            res = normalize(res, "range");
            responses(i, :) = res;
        end

        % predict response and calculate mae
        cum_predicted = zeros(1, length(gm.locs_list));
        for i_chk = 1:nChk
            res = responses(i_chk, :);
            predicted = predict(trained_dnn, res);
            cum_predicted = cum_predicted + predicted;

            % chunk mae
            [~, cp] = max(predicted);
            chunk_azimuth_predict = gm.locs_list(1, cp);
            chunk_ae = [chunk_ae; abs(chunk_azimuth_predict - azimuth_truth)];
        end
        [~, yp] = max(cum_predicted);

        azimuth_predict = gm.locs_list(1, yp);

        ae(i_iter) = abs(azimuth_predict - azimuth_truth);
        azimuth_predicts(i_iter) = azimuth_predict;
        azimuth_truths(i_iter) = azimuth_truth;

        tpb(i_iter);
    end
        
    mae = mean(ae);
    chunk_mae = mean(chunk_ae);
    cumu_resp = cumu_resp ./ loc_cnt;
end

function visualizeGassomTrainingSample (gm, ind)
    y_all = gm.env.timit_train{gm.somTrainParam.audio_idx(ind),1};
    y = y_all(gm.somTrainParam.audio_bgn(ind,1)+(1:gm.somTrainParam.audio_len));
    [frmL,frmR] = gm.env.genOneEpisode(gm.somTrainParam,ind);

    loc = gm.env.locs_list(:, gm.somTrainParam.locs_rand(ind));

    figure;
    subplot(3, 1, 1);
    plot(y);
    title("timit segment waveform");
    subplot(3, 1, 2);
    plot(frmL);
    subplot(3, 1, 3);
    plot(frmR);
end

function visualizeDnnTrainingSample (gm, ind, chunk_size)
    trainSamples = load("temp_data/dnnTrainGwn.mat");

    disp(cell2mat(trainSamples.trainY));

    o = 256 * (ind - 1);
    frmL = trainSamples.trainX((1 + o):(128 + o), :);
    frmR = trainSamples.trainX((129 + o):(256 * ind), :);

    frmL = frmL(:,1:chunk_size);
    frmR = frmR(:,1:chunk_size);

    single_len = size(frmL,1);
    rm = normalize([frmL;frmR]);
    frmL = rm(1:single_len,:);
    frmR = rm(single_len+1:end,:);

    frmMax = max(rm,[],"all");
    frmMin = min(rm,[],"all");

    figure;
    subplot(2, 1, 1);
    imagesc(frmL);
    colormap('jet');
    set(gca, 'YDir', 'normal');
    caxis([frmMin frmMax]);
    subplot(2, 1, 2);
    imagesc(frmR);
    colormap('jet');
    set(gca, 'YDir', 'normal');
    caxis([frmMin frmMax]);
end

function visualizeMap (gm)
    r = gm.topo_space(1);
    c = gm.topo_space(2);

    figure;
    for i=1:r*c
        subplot(r,c,i);
        b = gm.gsm{1}.bases{1}(:,i);
        l = size(b, 1) / 2;
        plot(1:1:l, b(1:l,:));
        hold on;
        plot(l+1:1:l*2, b(l+1:end,:), "Color", "r");
        axis off;
    end

    figure;
    for i=1:r*c
        subplot(r,c,i);
        b = gm.gsm{1}.bases{2}(:,i);
        l = size(b, 1) / 2;
        plot(1:1:l, b(1:l,:));
        hold on;
        plot(l+1:1:l*2, b(l+1:end,:), "Color", "r");
        axis off;
    end
end

function visualizeBMT (norm_winners, save_folder)
    figure;
    subplot(1, 1, 1);
    bar(1:length(norm_winners), norm_winners, 'FaceColor', 'blue');
    title("normalized winner distribution");
    saveas(gca, save_folder + "/bmt_distri.png");

    figure;
    subplot(1, 1, 1);
    boxplot(norm_winners);
    saveas(gca, save_folder + "/bmt_box.png");

    bmt_std = std(norm_winners);
    disp("std of normalized winner counting");
    disp(bmt_std);
    save(save_folder + "/bmt_std.mat", "bmt_std");
end

function visualizeRespHeatMap (cumu_resp, map_size, save_folder)
    resp_save_folder = save_folder + "/map_response/";
    for i = 1:size(cumu_resp, 1)
        azimuth = i * 10 - 100;
        figure;
        heatmap(reshape(cumu_resp(i,:), map_size, map_size)');
        title(...
            sprintf("average gassom response on test set\n" + ...
            "(gassom map: " + map_size + "x" + map_size + ", azimuth: " + azimuth + ")")...
        );
        saveas(gca, resp_save_folder + i + "_azimuth_" + azimuth + ".png");
    end
end

map_size = 16;
patch_dur = 16 / 1000;
save_folder = "chp4/waveform/result/" + map_size + "x" + map_size + "/";

%%% train gassom
% gm = initGassom([map_size, map_size], 32000, patch_dur);
% gm = initGassom([5, 5], 24000);
% gm = loadGassom([map_size, map_size], 32000, save_folder + "gsm.mat", patch_dur);
% winners = gm.trainGASSOM_timit2();
% gsm = gm.gsm{1};
% save(save_folder + "gsm.mat", "gsm");
% visualizeMap(gm);

%%% calculate BMT
% norm_winners = load("chp4_result/10x10_sz5_sf1/norm_winners.mat").norm_winners;
% norm_winners = (winners.*100)./(gm.max_iter * 115);
% disp(std(norm_winners));
% save(save_folder + "winners.mat", "winners");
% save(save_folder + "norm_winners.mat", "norm_winners");
% visualizeBMT(norm_winners, save_folder);

%%% train dnn
% generateDnnSamples(gm);
% [trained_dnn, train_info] = trainDNN(gm, patch_dur);
% save(save_folder + "dnn.mat", "trained_dnn");
% save(save_folder + "dnn_train_info.mat", "train_info");

%%% test model
% trained_dnn = load(save_folder + "dnn.mat").trained_dnn;
% [mae, chunk_mae, azimuth_predicts, azimuth_truths, cumu_resp] = testModel(gm, trained_dnn, patch_dur);
% disp("mae");
% disp(mae);
% disp('chunk_mae');
% disp(chunk_mae);
% save(save_folder + "mae.mat", "mae");
% save(save_folder + "chunk_mae.mat", "chunk_mae");
% save(save_folder + "azimuth_predicts.mat", "azimuth_predicts");
% save(save_folder + "azimuth_truths.mat", "azimuth_truths");

% save(save_folder + "cumu_resp.mat", "cumu_resp");
cumu_resp = load(save_folder + "cumu_resp.mat").cumu_resp;
visualizeRespHeatMap(cumu_resp, map_size, save_folder);

%%% some visualization
% visualizeHtfs(gm, 1, 45);
% visualizeGassomTrainingSample(gm, 1);
% visualizeDnnTrainingSample(gm, 1, patch_dur);
