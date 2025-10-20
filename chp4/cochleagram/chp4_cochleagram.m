function [gm] = initGassom (topo_space, max_iter, chunk_size)
    rng(49);

    % sofaloaded = SOFALoader;
    % save('sofaloaded.mat', 'sofaloaded');
    load temp_data/sofaloaded.mat;

    disp("init gassom start")
    fs = 44100;
    patch_dur = 0; % should be not neccessary in this chapter
    gm_param = {fs, patch_dur, topo_space, max_iter};
    gm = GASSOM_Model(gm_param, 128 * chunk_size, "cochleagram");
    disp("init gassom end");
end

function [gm] = loadGassom (topo_space, max_iter, gsm_path, chunk_size)
    gm = initGassom(topo_space, max_iter, chunk_size);
    gm.gsm{1} = load(gsm_path).gsm;
end

function generateDnnSamples (gm)
    disp("generating training samples");
    [trainX, trainY] = gm.env.genGwnIosr(...
        gm.locs_list, gm.netTrainParam.max_iter, ...
        gm.netTrainParam.audio_len, gm.env.fs, ...
        gm.netTrainParam.hrtf, gm.netTrainParam.subject, ...
        gm.netTrainParam.gwn_seed...
    );
    % disp("generating validing samples");
    % [validX, validY] = gm.env.genGwnIosr(...
    %     gm.locs_list, 190, ...
    %     gm.netTrainParam.audio_len, gm.env.fs, ...
    %     gm.netTrainParam.hrtf, gm.netTrainParam.subject, ...
    %     1003 ...
    % );
    disp("generating testing samples");
    [testX, testY] = gm.env.genGwnIosr(...
        gm.locs_list, gm.netTestParam.max_iter, ...
        gm.netTestParam.audio_len, gm.env.fs, ...
        gm.netTestParam.hrtf, gm.netTestParam.subject, ...
        gm.netTestParam.gwn_seed...
    );
    save("temp_data/dnnTrainGwnCochl.mat", "trainX", "trainY");
    % save("temp_data/dnnValidGwn.mat", "validX", "validY");
    save("temp_data/dnnTestGwnCochl.mat", "testX", "testY");
end

function [trained_dnn, train_info] = trainDNN (gm, chunk_size)
    trainSamples = load("temp_data/dnnTrainGwnCochl.mat");

    assert(gm.netTrainParam.max_iter == size(trainSamples.trainX, 1) / 256);

    XTrain = zeros(gm.netTrainParam.max_iter, prod(gm.topo_space));
    YTrain = categorical(trainSamples.trainY);

    nFrm = size(trainSamples.trainX, 2);

    rng(49);

    for i = 1:gm.netTrainParam.max_iter
        o = 256 * (i - 1);
        frmL = trainSamples.trainX((1 + o):(128 + o), :);
        frmR = trainSamples.trainX((129 + o):(256 * i), :);

        chkIdx = 0:1:nFrm-chunk_size;
        j = chkIdx(randi(length(chkIdx)));
        chkL = frmL(:,j+(1:chunk_size));
        chkR = frmR(:,j+(1:chunk_size));

        single_len = size(chkL,1);
        rm = normalize([chkL;chkR]);
        chkL = rm(1:single_len,:);
        chkR = rm(single_len+1:end,:);

        x = [reshape(chkL, [], 1);reshape(chkR, [], 1)];

        res = gm.getResponse(x);
        XTrain(i, :) = res;
    end

    rng(49);

    [dnn, training_option] = createDNN(prod(gm.topo_space), gm.locs_num, 3);
    [trained_dnn, train_info] = trainNetwork(XTrain, YTrain, dnn, training_option);
end

function [mae, chunk_mae, azimuth_predicts, azimuth_truths, cumu_resp] = testModel (...
    gm, trained_dnn, chunk_size ...
)
    testSamples = load("temp_data/dnnTestGwnCochl.mat");
    assert(gm.netTestParam.max_iter == size(testSamples.testX, 1) / 256);

    ae = zeros(gm.netTestParam.max_iter, 1);
    azimuth_predicts = zeros(gm.netTestParam.max_iter, 1);
    azimuth_truths = zeros(gm.netTestParam.max_iter, 1);

    cumu_resp = zeros(gm.locs_num, prod(gm.topo_space));
    loc_cnt = zeros(gm.locs_num, 1);

    chunk_ae = [];

    nFrm = size(testSamples.testX, 2);

    tpb = textprogressbar(gm.netTestParam.max_iter, 'showremtime', true);
    for i_iter = 1:gm.netTestParam.max_iter
        YTest = testSamples.testY(i_iter);

        nChk = length(0:1:nFrm-chunk_size);

        o = 256 * (i_iter - 1);
        frmL = testSamples.testX((1 + o):(128 + o), :);
        frmR = testSamples.testX((129 + o):(256 * i_iter), :);
        azimuth_truth = gm.locs_list(1, YTest(1));

        % get response from gassom
        responses = zeros(nChk, prod(gm.topo_space));
        for i = 1:size(responses,1)
            chkL = frmL(:,i:(chunk_size + i - 1));
            chkR = frmR(:,i:(chunk_size + i - 1));

            single_len = size(chkL,1);
            rm = normalize([chkL;chkR]);
            chkL = rm(1:single_len,:);
            chkR = rm(single_len+1:end,:);

            x = [reshape(chkL, [], 1);reshape(chkR, [], 1)];
            res = gm.getResponse(x);
            loc_idx = (azimuth_truth + 100) / 10;
            cumu_resp(loc_idx, :) = cumu_resp(loc_idx, :) + res';
            loc_cnt(loc_idx) = loc_cnt(loc_idx) + 1;
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

function plotCochl (frm, cfs, title_str, cmin, cmax)
    assert(size(cfs,2) == 128)
    imagesc(frm);
    title(title_str);
    ylabel("frequency (hz)")
    xlabel("time frame")
    yticks([1 16 32 48 64 80 96 112 128]);
    yticklabels([cfs(1,1), cfs(1,16), cfs(1,32), cfs(1,48), cfs(1,64), cfs(1,80), cfs(1,96), cfs(1,112), cfs(1,128)]);
    colormap('jet');

    if nargin == 5
        caxis([cmin cmax]);
    end

    set(gca, 'YDir', 'normal');
end

function visualizeHtfs (gm, ind, azimuth)
    save_folder = 'chp4_result/hrtf_sample/' + string(azimuth) + '/';

    y = gm.env.timit_train{gm.somTrainParam.audio_idx(ind),1};

    loc = gm.env.locs_list(:, gm.somTrainParam.locs_rand(ind));
    loc(1) = azimuth;

    bi = gm.env.sofa.spatMono(y, loc, gm.somTrainParam.hrtf, gm.somTrainParam.subject);
    biL = bi(:,2);
    biR = bi(:,1);

    audiowrite(save_folder + 'origin.wav', y, gm.fs);
    audiowrite(save_folder + 'left' + string(azimuth) + '.wav', biL, gm.fs);
    audiowrite(save_folder + 'right' + string(azimuth) + '.wav', biR, gm.fs);

    %% visualize waveform

    time = ((1/gm.fs):(1/gm.fs):(length(y)/gm.fs));

    yMax = max([y;biL;biR], [], "all");
    yMin = min([y;biL;biR], [], "all");

    figure;
    subplot(3, 1, 1);
    plot(time, y);
    ylim([yMin yMax]);
    xlabel("time (s)");
    title("before hrtf with azimuth " + loc(1));

    subplot(3, 1, 2);
    plot(time, biL);
    title("left");
    xlabel("time (s)");
    ylim([yMin yMax]);
    
    subplot(3, 1, 3);
    plot(time, biR);
    title("right");
    xlabel("time (s)");
    ylim([yMin yMax]);

    saveas(gcf, save_folder + "waveform.png");

    %% viuslaize cochleagram

    [frm, ~] = audio2cochlIOSR(...
        y, ...
        gm.env.fs, 100, 20000, 128, ...
        8, 4 ...
    );

    [frmL, ~] = audio2cochlIOSR(...
        biL, ...
        gm.env.fs, 100, 20000, 128, ...
        8, 4 ...
    );

    [frmR, cfs] = audio2cochlIOSR(...
        biR, ...
        gm.env.fs, 100, 20000, 128, ...
        8, 4 ...
    );

    sample_len = size(frmL, 1);
    normalized = normalize([frmL; frmR; frm]);
    frmL = normalized(1:sample_len,:);
    frmR = normalized(sample_len+1:2*sample_len,:);
    frm = normalized(2*sample_len:end,:);

    frmMax = max(normalized,[],"all");
    frmMin = min(normalized,[],"all");
    % frmMax = max(frm, [], "all");
    % frmMin = min(frm, [], "all");

    figure;
    subplot(3, 1, 1);
    plotCochl(frm, cfs, "origin", frmMin, frmMax);
    subplot(3, 1, 2);
    plotCochl(frmL, cfs, "left", frmMin, frmMax);
    subplot(3, 1, 3);
    plotCochl(frmR, cfs, "right", frmMin, frmMax);

    saveas(gcf, save_folder + "cochleagram.png");
end

function visualizeGassomTrainingSample (gm, ind)
    y_all = gm.env.timit_train{gm.somTrainParam.audio_idx(ind),1};
    y = y_all(gm.somTrainParam.audio_bgn(ind,1)+(1:gm.somTrainParam.audio_len));

    [frm, cfs] = audio2cochlIOSR(...
        y, ...
        gm.env.fs, 100, 20000, 128, ...
        8, 4 ...
    );

    loc = gm.env.locs_list(:, gm.somTrainParam.locs_rand(ind));
    [frmL, frmR, ~] = gm.env.genOneTrainEpisodeCochIOSR(gm.somTrainParam, ind);

    % normalize
    sample_len = size(frmL, 1);
    nm = normalize([frmL; frmR; frm]);
    frmL = nm(1:sample_len,:);
    frmR = nm(sample_len+1:2*sample_len,:);
    frm = nm(2*sample_len:end,:);
    % nm = norm([frmL;frmR]);
    % frmL = frmL/nm;
    % frmR = frmR/nm;

    blkL = frmL(:,1:5);
    blkR = frmR(:,1:5);

    frmMax = max(nm,[],"all");
    frmMin = min(nm,[],"all");
    blkMax = max([blkL; blkR],[],"all");
    blkMin = min([blkL; blkR],[],"all");

    figure;
    subplot(1, 1, 1);
    bar(...
        1:19, ...
        histcounts(gm.somTrainParam.locs_rand, 'BinMethod', 'integers', 'BinLimits', [1 19]), ...
        'FaceColor', 'blue' ...
    );
    title("counting of every sound location");
    
    figure;
    subplot(1, 1, 1);
    plot(y);
    title("timit segment waveform");

    figure;
    subplot(3, 1, 1);
    plotCochl(frm, cfs, "normalized timit segment", frmMin, frmMax);
    subplot(3, 1, 2);
    plotCochl(frmL, cfs, "normalized left (azimuth " + loc(1) + ")", frmMin, frmMax);
    subplot(3, 1, 3);
    plotCochl(frmR, cfs, "normalized right (azimuth " + loc(1) + ")", frmMin, frmMax);
    
    figure;
    subplot(3, 1, 1);
    plotCochl(blkL, cfs, "a chunk (5 frames) of normalized left", blkMin, blkMax);
    subplot(3, 1, 2);
    plotCochl(blkR, cfs, "a chunk (5 frames) of normalized right", blkMin, blkMax);
    subplot(3, 1, 3);
    plotCochl([blkL blkR], cfs, "sample that feed into gassom", blkMin, blkMax);
end

function visualizeDnnTrainingSample (gm, ind, chunk_size)
    trainSamples = load("temp_data/dnnTrainGwn.mat");

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

function visualizeCochlMap (gm, chunk_size)
    figure;
    r = gm.topo_space(1);
    c = gm.topo_space(2);
    for i=1:r*c
        subplot(r,c,i);
        imagesc(reshape(gm.gsm{1}.bases{1}(:,i), [], chunk_size * 2));
        colormap('jet');
        axis off;
        set(gca, 'YDir', 'normal');
    end

    % figure;
    % r = gm.topo_space(1);
    % c = gm.topo_space(2);
    % for i=1:r*c
    %     subplot(r,c,i);
    %     imagesc(reshape(gm.gsm{1}.bases{2}(:,i), [], 10));
    %     colormap('jet');
    %     axis off;
    %     set(gca, 'YDir', 'normal');
    % end
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

function visualizeRespHeatMap (cumu_resp, map_width, chunk_size, save_folder)
    resp_save_folder = save_folder + "/map_response/";
    for i = 1:size(cumu_resp, 1)
        azimuth = i * 10 - 100;
        figure;
        heatmap(reshape(cumu_resp(i,:), map_width, map_width)');
        title(...
            sprintf("average gassom response on test set\n" + ...
            "(gassom map: " + map_width + "x" + map_width + ", chunk size: " + chunk_size + ", azimuth: " + azimuth + ")")...
        );
        saveas(gca, resp_save_folder + i + "_azimuth_" + azimuth + ".png");
    end
end

map_width = 8;
chunk_size = 5;
save_folder = "chp4/cochleagram/result/" + map_width + "x" + map_width + "_sz" + chunk_size + "_sf1/";

%%% train gassom
% gm = initGassom([20, 20], 5e4, chunk_size);
% gm = initGassom([5, 5], 24000);
gm = loadGassom([map_width, map_width], 5e4, save_folder + "gsm.mat", chunk_size);
% winners = gm.trainGASSOM_cochleagram_IOSR2(chunk_size);
% gsm = gm.gsm{1};
% save(save_folder + "gsm.mat", "gsm");
% visualizeCochlMap(gm, chunk_size);

%%% calculate BMT
% norm_winners = load("chp4_result/10x10_sz5_sf1/norm_winners.mat").norm_winners;
% norm_winners = (winners.*100)./gm.max_iter;
% disp(std(norm_winners));
% save(save_folder + "winners.mat", "winners");
% save(save_folder + "norm_winners.mat", "norm_winners");
% visualizeBMT(norm_winners, save_folder);

%%% train dnn
% generateDnnSamples(gm);
[trained_dnn, train_info] = trainDNN(gm, chunk_size);
% save(save_folder + "dnn.mat", "trained_dnn");
% save(save_folder + "dnn_train_info.mat", "train_info");

%%% test model
% trained_dnn = load(save_folder + "dnn.mat").trained_dnn;
[mae, chunk_mae, azimuth_predicts, azimuth_truths, cumu_resp] = testModel(gm, trained_dnn, chunk_size);
disp("mae");
disp(mae);
% disp('chunk_mae');
% disp(chunk_mae);
% save(save_folder + "mae.mat", "mae");
% save(save_folder + "chunk_mae.mat", "chunk_mae");
% save(save_folder + "azimuth_predicts.mat", "azimuth_predicts");
% save(save_folder + "azimuth_truths.mat", "azimuth_truths");
% save(save_folder + "cumu_resp.mat", "cumu_resp");

% cumu_resp = load(save_folder + "cumu_resp.mat").cumu_resp;
% visualizeRespHeatMap(cumu_resp, map_width, chunk_size, save_folder);

%%% some visualization
% visualizeHtfs(gm, 1, 45);
% visualizeGassomTrainingSample(gm, 1);
% visualizeDnnTrainingSample(gm, 1, chunk_size);
