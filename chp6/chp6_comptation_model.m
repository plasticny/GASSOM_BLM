function [gm] = initGassom (topo_space, max_iter, chunk_size, hrtf_database, hrtf_subject)
    rng(49);

    % sofaloaded = SOFALoader;
    % save('sofaloaded.mat', 'sofaloaded');
    load temp_data/sofaloaded.mat;

    disp("init gassom start");
    fs = 44100;
    patch_dur = 0; % should be not neccessary in this chapter
    gm_param = {fs, patch_dur, topo_space, max_iter};
    gm = GASSOM_Model(gm_param, 128 * chunk_size, "cochleagram");
    gm.setHRTFandSubject(hrtf_database, hrtf_subject);
    disp("init gassom end");
end

function [gm] = loadGassom (topo_space, max_iter, gsm_path, chunk_size, hrtf_database, hrtf_subject)
    gm = initGassom(topo_space, max_iter, chunk_size, hrtf_database, hrtf_subject);
    gm.gsm{1} = load(gsm_path).gsm;
end

function generateDnnTrainSamples (gm, hrtf, hrtf_subject)
    [trainX, trainY] = gm.env.genGwnIosr(...
        gm.locs_list, gm.locs_num * 100, ...
        gm.netTrainParam.audio_len, gm.env.fs, ...
        hrtf, hrtf_subject, ...
        gm.netTrainParam.gwn_seed...
    );
    save("chp6/cache/dnnTrainGwn_" + hrtf + "_" + hrtf_subject + ".mat", "trainX", "trainY");
end

function generateDnnTestSamples (gm, hrtf, hrtf_subject)
    [frontTestX, frontTestY] = gm.env.genGwnIosr(...
        gm.locs_list(:, 1:gm.locs_num / 2), gm.locs_num * 50, ...
        gm.netTestParam.audio_len, gm.env.fs, ...
        hrtf, hrtf_subject, ...
        gm.netTestParam.gwn_seed...
    );
    [backTestX, backTestY] = gm.env.genGwnIosr(...
        gm.locs_list(:, gm.locs_num / 2 + 1:end), gm.locs_num * 50, ...
        gm.netTestParam.audio_len, gm.env.fs, ...
        hrtf, hrtf_subject, ...
        gm.netTestParam.gwn_seed * 2 ...
    );
    save("chp6/cache/dnnTestGwn_" + hrtf + "_" + hrtf_subject + ".mat", "frontTestX", "frontTestY", "backTestX", "backTestY");
end

function [azimuth_predicts, azimuth_truths, cumu_resp] = testModel (...
    gm, trained_dnn, chunk_size, test_x, test_y ...
)
    azimuth_predicts = zeros(length(test_y), 1);
    azimuth_truths = zeros(length(test_y), 1);

    cumu_resp = zeros(gm.locs_num, prod(gm.topo_space));
    loc_cnt = zeros(gm.locs_num, 1);

    nFrm = size(test_x{1}{1}, 2);

    tpb = textprogressbar(length(test_y), 'showremtime', true);
    for i_iter = 1:length(test_y)
        YTest = test_y{i_iter};

        nChk = length(0:1:nFrm-chunk_size);

        frmL = test_x{i_iter}{1};
        frmR = test_x{i_iter}{2};
        azimuth_truth = YTest(1);

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
            cumu_resp(azimuth_truth, :) = cumu_resp(azimuth_truth, :) + res';
            loc_cnt(azimuth_truth) = loc_cnt(azimuth_truth) + 1;
            responses(i, :) = res;
        end

        % predict response
        cum_predicted = zeros(1, length(gm.locs_list));
        for i_chk = 1:nChk
            res = responses(i_chk, :);
            predicted = predict(trained_dnn, res);
            cum_predicted = cum_predicted + predicted;
        end
        [~, azimuth_predict] = max(cum_predicted);

        azimuth_predicts(i_iter) = azimuth_predict;
        azimuth_truths(i_iter) = azimuth_truth;

        tpb(i_iter);
    end
        
    cumu_resp = cumu_resp ./ loc_cnt;
end

function [front_confusion_rate, back_confustion_rate] = estimate_confusion_rate(gm, test_hrtf_database, test_hrtf_subject, save_folder)
    predicts = load(save_folder + "azimuth_predicts_" + test_hrtf_database + "_" + test_hrtf_subject + ".mat");
    front_pred = predicts.front_azimuth_predicts;
    back_pred = predicts.back_azimuth_predicts;
    front_confusion_rate = sum(front_pred >= gm.locs_num / 2 + 1) / length(front_pred);
    back_confustion_rate = sum(back_pred <= gm.locs_num / 2) / length(back_pred);
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
    save_folder = 'chp6/hrtf_sample/' + string(azimuth) + '/';

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

hrtf_database = "cipic";
map_width = 10;
chunk_size = 10;

% cipic_subjects = [...
%     48, 50, 51, 58, 59, ...
%     60, 119, 126, 127, 131, 133, ...
%     135, 137, 147, 148, 153, 155, 156, 158, ...
%     162, 163, 165
% ];
% cipic_subjects = [
%    3,  8,  9, 10, 11, 12, 15, 17, 18, 19, 
%   20, 21, 27, 28, 33, 40, 44, 61, 65, 124,
%   134, 152, 154, 165
% ];
cipic_subjects = [9];

for hrtf_subject = cipic_subjects
    % hrtf_subject = 9;
    disp("train comptational model for cipic subject " + hrtf_subject);

    save_folder = "chp6/comptational_model_10x10/subject_" + hrtf_subject + "/";
    % save_folder = "chp6/temp/";

    if ~exist(save_folder,'dir') mkdir(save_folder); end

    %%% train gassom
    gm = initGassom([map_width, map_width], 5e4, chunk_size, hrtf_database, hrtf_subject);
    winners = gm.trainGASSOM_cochleagram_IOSR(chunk_size, save_folder);
    gsm = gm.gsm{1};
    save(save_folder + "gsm.mat", "gsm");
    % visualizeCochlMap(gm, chunk_size);

    %%% calculate BMT
    % norm_winners = load("chp4_result/10x10_sz5_sf1/norm_winners.mat").norm_winners;
    norm_winners = (winners.*100)./gm.max_iter;
    disp(std(norm_winners));
    save(save_folder + "winners.mat", "winners");
    save(save_folder + "norm_winners.mat", "norm_winners");
    visualizeBMT(norm_winners, save_folder);
end

%%% train dnn
% generateDnnTrainSamples(gm, hrtf_database, hrtf_subject);
% gm = loadGassom([map_width, map_width], 5e4, save_folder + "gsm.mat", chunk_size, hrtf_database, hrtf_subject);
% [trained_dnn, train_info] = trainDNN(gm, chunk_size);
% save(save_folder + "dnn.mat", "trained_dnn");
% save(save_folder + "dnn_train_info.mat", "train_info");
% end


%%% test model
% test_hrtf_database = "cipic";
% test_hrtf_subject = 20;

% gm_test_hrtf = initGassom([map_width, map_width], 5e4, chunk_size, test_hrtf_database, test_hrtf_subject);
% generateDnnTestSamples(gm_test_hrtf, test_hrtf_database, test_hrtf_subject);

% hrtf_subject = 3;
% save_folder = "chp6/comptational_model/subject_" + hrtf_subject + "/";
% gm = loadGassom([map_width, map_width], 5e4, save_folder + "gsm.mat", chunk_size, hrtf_database, hrtf_subject);

% trained_dnn = load(save_folder + "dnn.mat").trained_dnn;
% testing_data = load("chp6/cache/dnnTestGwn_" + test_hrtf_database + "_" + test_hrtf_subject + ".mat");
% [front_azimuth_predicts, front_azimuth_truths, front_cumu_resp] = testModel(...
%     gm, trained_dnn, chunk_size, ...
%     testing_data.frontTestX, testing_data.frontTestY ...
% );
% [back_azimuth_predicts, back_azimuth_truths, back_cumu_resp] = testModel(...
%     gm, trained_dnn, chunk_size, ...
%     testing_data.backTestX, testing_data.backTestY ...
% );
% save(save_folder + "azimuth_predicts_" + test_hrtf_database + "_" + test_hrtf_subject + ".mat", "front_azimuth_predicts", "back_azimuth_predicts");
% save(save_folder + "azimuth_truths_" + test_hrtf_database + "_" + test_hrtf_subject + ".mat", "front_azimuth_truths", "back_azimuth_truths");
% save(save_folder + "cumu_resp_" + test_hrtf_database + "_" + test_hrtf_subject + ".mat", "front_cumu_resp", "back_cumu_resp");

% figure;
% confusionchart(front_azimuth_truths, front_azimuth_predicts);
% title("front confusion chart");
% figure;
% confusionchart(back_azimuth_truths, back_azimuth_predicts);
% title("back confusion chart");

% [front_confusion_rate, back_confusion_rate] = estimate_confusion_rate(gm, test_hrtf_database, test_hrtf_subject, save_folder);
% save(save_folder + "confusion_rate_" + test_hrtf_database + "_" + test_hrtf_subject + ".mat", "front_confusion_rate", "back_confusion_rate");

% cumu_resp = load(save_folder + "cumu_resp.mat").cumu_resp;
% visualizeRespHeatMap(cumu_resp, map_width, chunk_size, save_folder);


%%% some visualization
% visualizeHtfs(gm, 1, 45);
% visualizeGassomTrainingSample(gm, 1);
% visualizeDnnTrainingSample(gm, 1, chunk_size);
