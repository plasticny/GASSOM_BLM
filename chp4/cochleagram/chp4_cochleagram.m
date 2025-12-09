hrtf_database = "kemar";
hrtf_subject = 0;
map_width = 10;
chunk_size = 10;

% save_folder = "chp4/cochleagram/result/" + map_width + "x" + map_width + "_sz" + chunk_size + "_sf1/";
% save_folder = "/home/nycheung/desktop/shuto_gassom/GASSOM_BLM/chp4/cipic_360/cochleagram/10x10_10/8/";
save_folder = "chp4/kemar_online/";

% if ~exist(save_folder,'dir') mkdir(save_folder); end

%%% train gassom
gm = initGassom([map_width, map_width], 5e4, chunk_size, hrtf_database, hrtf_subject);
% gm = loadGassom([map_width, map_width], 5e4, save_folder + "gsm.mat", chunk_size, hrtf_database, hrtf_subject);
winners = gm.trainGASSOM_cochleagram2(chunk_size, "cache/somTrainSamples_kemar_0_19_50000.mat", save_folder);
% gsm = gm.gsm{1};
% save(save_folder + "gsm.mat", "gsm");
visualizeCochlMap(gm, chunk_size);
    
% checkResponse(gm, chunk_size, "cache/dnnTestGwn_kemar_0_1900.mat");
% checkResponse(gm, chunk_size, "cache/dnnTestGwn_kemar_0_1900_bandpass_cf250_2.mat");

%%% calculate BMT
% norm_winners = load("chp4_result/10x10_sz5_sf1/norm_winners.mat").norm_winners;
% norm_winners = (winners.*100)./gm.max_iter;
% disp(std(norm_winners));
% save(save_folder + "winners.mat", "winners");
% save(save_folder + "norm_winners.mat", "norm_winners");
% visualizeBMT(norm_winners, save_folder);

%%% train dnn
% generateDnnSamples(gm);
% [trained_dnn, ~] = trainDNN(gm, chunk_size, "cache/dnnTrainGwn_cipic_9_2600_bandpass_all_cf_bw.mat");
% [trained_dnn, ~] = trainDNN(gm, chunk_size, "cache/dnnTrainGwn_kemar_0_1900.mat");
% trained_dnn = trainDNNTimit(gm, chunk_size, "cache/dnnTrainTimit_cipic_9_2600.mat");
% save(save_folder + "dnn.mat", "trained_dnn");
% save(save_folder + "dnn_train_info.mat", "train_info");

%%% test model
% trained_dnn = load(save_folder + "dnn.mat").trained_dnn;
% [mae, predicts, truths, cumu_resp] = testModelTimit(gm, trained_dnn, chunk_size);
% [mae, azimuth_predicts, azimuth_truths, cumu_resp] = testModel(gm, trained_dnn, chunk_size, "cache/dnnTestGwn_kemar_0_1900.mat");
% disp("mae");
% disp(mae);
% figure;
% confusionchart(azimuth_truths, azimuth_predicts);

% disp(rms(truths, predicts));
% disp('chunk_mae');
% disp(chunk_mae);
% save(save_folder + "mae.mat", "mae");
% save(save_folder + "chunk_mae.mat", "chunk_mae");
% save(save_folder + "azimuth_predicts.mat", "azimuth_predicts");
% save(save_folder + "azimuth_truths.mat", "azimuth_truths");
% save(save_folder + "cumu_resp.mat", "cumu_resp");

% cumu_resp = load(save_folder + "cumu_resp.mat").cumu_resp;
% visualizeRespHeatMap(cumu_resp, map_width, chunk_size, save_folder);


function [gm] = initGassom (topo_space, max_iter, chunk_size, hrtf_database, hrtf_subject)
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
end

function [gm] = loadGassom (topo_space, max_iter, gsm_path, chunk_size, hrtf_database, hrtf_subject)
    gm = initGassom(topo_space, max_iter, chunk_size, hrtf_database, hrtf_subject);
    gm.gsm{1} = load(gsm_path).gsm;
end

function [trained_dnn, train_info] = trainDNN (gm, chunk_size, dataset)
    train_data = load(dataset);
    nFrm = size(train_data.trainX{1}{1}, 2);
    nChk = length(0:1:nFrm-chunk_size);
    % dnn_x = zeros(gm.netTrainParam.max_iter * nChk, prod(gm.topo_space));
    % dnn_y = zeros(gm.netTrainParam.max_iter * nChk, 1);
    % samples_size = gm.netTrainParam.max_iter;
    samples_size = length(train_data.trainY);
    dnn_x = zeros(samples_size, prod(gm.topo_space));
    dnn_y = zeros(samples_size, 1);

    rng(49);

    dnn_i = 1;

    tpb = textprogressbar(samples_size);
    for i = 1:samples_size
        frmL = train_data.trainX{i}{1};
        frmR = train_data.trainX{i}{2};

        % for j = 0:1:nFrm-chunk_size
        %     chkL = frmL(:,j+(1:chunk_size));
        %     chkR = frmR(:,j+(1:chunk_size));

        %     % single_len = size(chkL,1);
        %     % rm = normalize([chkL;chkR]);
        %     % rm = [chkL;chkR];
        %     % chkL = rm(1:single_len,:);
        %     % chkR = rm(single_len+1:end,:);

        %     x = [reshape(chkL, [], 1);reshape(chkR, [], 1)];
        %     x = x-ones(size(x,1),1)*mean(x,1);
        %     X = bsxfun(@rdivide, x, sqrt(sum(x.^2))+eps); 
        %     % x = x / norm(x);

        %     res = gm.getResponse(X);
            
        %     dnn_x(dnn_i,:) = res';
        %     dnn_y(dnn_i) = train_data.trainY{i};
        %     dnn_i = dnn_i + 1;
        %     % dnn_x = [dnn_x;res'];
        %     % dnn_y = [dnn_y;train_data.trainY{i}];
        % end

        chkIdx = 0:1:nFrm-chunk_size;
        
        j = chkIdx(randi(length(chkIdx)));
        chkL = frmL(:,j+(1:chunk_size));
        chkR = frmR(:,j+(1:chunk_size));

        % single_len = size(chkL,1);
        % rm = normalize([chkL;chkR]);
        % rm = [chkL;chkR];
        % chkL = rm(1:single_len,:);
        % chkR = rm(single_len+1:end,:);

        x = [reshape(chkL, [], 1);reshape(chkR, [], 1)];
        x = x-ones(size(x,1),1)*mean(x,1);
        X = bsxfun(@rdivide, x, sqrt(sum(x.^2))+eps); 
        % x = x / norm(x);

        res = gm.getResponse(X);
        % res = normalize(res);

        dnn_x(i, :) = res;
        dnn_y(i) = train_data.trainY{i};

        tpb(i);
    end

    dnn_y = categorical(dnn_y);

    rng(49);

    [dnn, training_option] = createDNN(prod(gm.topo_space), gm.locs_num, 3);
    [trained_dnn, train_info] = trainNetwork(dnn_x, dnn_y, dnn, training_option);
end

function [trained_dnn] = trainDNNTimit (gm, chunk_size, dataset_path)
    trainSamples = load(dataset_path);
    trainX = trainSamples.trainX;
    trainY = trainSamples.trainY;

    assert(gm.netTrainParam.max_iter == length(trainSamples.trainY));

    rng(49);

    sample_pre_x = 10;

    dnnX = zeros(gm.netTrainParam.max_iter * sample_pre_x, prod(gm.topo_space));
    dnnY = zeros(gm.netTrainParam.max_iter * sample_pre_x, 1);

    ind = 1;
    for i = 1:gm.netTrainParam.max_iter
        frmL = trainX{i}{1};
        frmR = trainX{i}{2};
        loc_idx = trainY{i};

        nFrm = size(frmL, 2);
        chkIdx = 0:1:nFrm-chunk_size;

        for j = 1:sample_pre_x
            chkStart = chkIdx(randi(length(chkIdx)));
            chkL = frmL(:,chkStart+(1:chunk_size));
            chkR = frmR(:,chkStart+(1:chunk_size));

            single_len = size(chkL,1);
            rm = normalize([chkL;chkR]);
            chkL = rm(1:single_len,:);
            chkR = rm(single_len+1:end,:);

            x = [reshape(chkL, [], 1);reshape(chkR, [], 1)];

            res = gm.getResponse(x);
            dnnX(ind, :) = res;
            dnnY(ind) = loc_idx;
            ind = ind + 1;
        end
    end

    dnnY = categorical(dnnY);

    rng(49);

    [dnn, training_option] = createDNN(prod(gm.topo_space), gm.locs_num, 3);
    [trained_dnn, ~] = trainNetwork(dnnX, dnnY, dnn, training_option);
end

function [mae, azimuth_predicts, azimuth_truths, cumu_resp] = testModel (...
    gm, trained_dnn, chunk_size, dataset_path ...
)
    test_data = load(dataset_path);
    testX = test_data.testX;
    testY = test_data.testY;
    % sample_size = gm.netTestParam.max_iter;
    sample_size = length(testY);
    % assert(sample_size == size(test_data.testX, 1));

    ae = zeros(sample_size, 1);
    azimuth_predicts = zeros(sample_size, 1);
    azimuth_truths = zeros(sample_size, 1);

    cumu_resp = zeros(gm.locs_num, prod(gm.topo_space));

    % chunk_ae = [];

    nFrm = size(testX{1}{1}, 2);

    tpb = textprogressbar(sample_size, 'showremtime', true);
    for i_iter = 1:sample_size
        YTest = testY{i_iter};

        nChk = length(0:1:nFrm-chunk_size);

        frmL = testX{i_iter}{1};
        frmR = testX{i_iter}{2};
        % azimuth_truth = gm.locs_list(1, YTest(1));

        % get response from gassom
        responses = zeros(nChk, prod(gm.topo_space));
        for i = 1:size(responses,1)
            chkL = frmL(:,i:(chunk_size + i - 1));
            chkR = frmR(:,i:(chunk_size + i - 1));

            % single_len = size(chkL,1);
            % rm = normalize([chkL;chkR]);
            % chkL = rm(1:single_len,:);
            % chkR = rm(single_len+1:end,:);

            x = [reshape(chkL, [], 1);reshape(chkR, [], 1)];
            x = x-ones(size(x,1),1)*mean(x,1);
            X = bsxfun(@rdivide, x, sqrt(sum(x.^2))+eps); 
            % x = x / norm(x);

            res = gm.getResponse(X);
            % res = normalize(res);

            % loc_idx = (azimuth_truth + 100) / 10;
            loc_idx = YTest(1);
            cumu_resp(loc_idx, :) = cumu_resp(loc_idx, :) + res';
            responses(i, :) = res;
        end

        % predict response and calculate mae
        cum_predicted = zeros(1, length(gm.locs_list));
        for i_chk = 1:nChk
            res = responses(i_chk, :);
            predicted = predict(trained_dnn, res);
            cum_predicted = cum_predicted + predicted;

            % chunk mae
            % [~, cp] = max(predicted);
            % chunk_azimuth_predict = gm.locs_list(1, cp);
            % chunk_ae = [chunk_ae; abs(chunk_azimuth_predict - azimuth_truth)];
        end
        [~, yp] = max(cum_predicted);

        % azimuth_predict = gm.locs_list(1, yp);
        % ae(i_iter) = abs(azimuth_predict - azimuth_truth);
        % azimuth_predicts(i_iter) = azimuth_predict;
        % azimuth_truths(i_iter) = azimuth_truth;
        ae(i_iter) = abs(yp - YTest(1));
        azimuth_predicts(i_iter) = yp;
        azimuth_truths(i_iter) = YTest(1);

        tpb(i_iter);
    end
        
    mae = mean(ae);
    % chunk_mae = mean(chunk_ae);
end

function [mae, predicts, truths, cumu_resp] = testModelTimit (...
    gm, trained_dnn, chunk_size ...
)
    testSamples = load("cache/dnnTestTimit_kemar_0.mat");
    assert(gm.netTestParam.max_iter == length(testSamples.testY));

    ae = zeros(gm.netTestParam.max_iter, 1);
    predicts = zeros(gm.netTestParam.max_iter, 1);
    truths = zeros(gm.netTestParam.max_iter, 1);

    cumu_resp = zeros(gm.locs_num, prod(gm.topo_space));

    nFrm = size(testSamples.testX{1}{1}, 2);

    tpb = textprogressbar(gm.netTestParam.max_iter, "showremtime", true);
    for i = 1:gm.netTestParam.max_iter
        truth_idx = testSamples.testY{i};

        nChk = length(0:1:nFrm-chunk_size);

        frmL = testSamples.testX{i}{1};
        frmR = testSamples.testX{i}{2};
        truth_azimuth = gm.locs_list(1, truth_idx);

        responses = zeros(nChk, prod(gm.topo_space));
        for j = 1:size(responses,1)
            chkL = frmL(:,j:(chunk_size + j - 1));
            chkR = frmR(:,j:(chunk_size + j - 1));

            single_len = size(chkL,1);
            rm = normalize([chkL;chkR]);
            chkL = rm(1:single_len,:);
            chkR = rm(single_len+1:end,:);

            x = [reshape(chkL, [], 1);reshape(chkR, [], 1)];
            res = gm.getResponse(x);
            loc_idx = (truth_azimuth + 100) / 10;
            cumu_resp(loc_idx, :) = cumu_resp(loc_idx, :) + res';
            responses(j, :) = res;
        end

        % predict response and calculate mae
        cum_predicted = zeros(1, length(gm.locs_list));
        for i_chk = 1:nChk
            res = responses(i_chk, :);
            predicted = predict(trained_dnn, res);
            cum_predicted = cum_predicted + predicted;
        end
        [~, yp] = max(cum_predicted);

        predict_azimuth = gm.locs_list(1, yp);
        % ae(i) = abs(predict_azimuth - truth_azimuth);
        % predicts(i) = predict_azimuth;
        % truths(i) = truth_azimuth;
        ae(i) = abs(yp - truth_idx);
        predicts(i) = yp;
        truths(i) = truth_idx;

        tpb(i);
    end

    mae = mean(ae);
end

function [cumu_resp] = checkResponse (gm, chunk_size, dataset_path)
    test_data = load(dataset_path);
    test_x = test_data.testX;
    test_y = test_data.testY;
    sample_size = length(test_y);

    tpb = textprogressbar(sample_size, "showremtime", true);
    cumu_resp = zeros(gm.locs_num, prod(gm.topo_space));
    for i = 1:sample_size
        frmL = test_x{i}{1};
        frmR = test_x{i}{2};
        loc_idx = test_y{i};

        nFrm = size(frmL, 2);
        nChk = length(0:1:nFrm-chunk_size);
        for j = 1:nChk
            chkL = frmL(:,j:(chunk_size + j - 1));
            chkR = frmR(:,j:(chunk_size + j - 1));

            % single_len = size(chkL, 1);
            % rm = [chkL;chkR];
            % rm = normalize([chkL;chkR]);
            % chkL = rm(1:single_len,:);
            % chkR = rm(single_len+1:end,:);

            x = [reshape(chkL, [], 1); reshape(chkR, [], 1)];
            x = x-ones(size(x,1),1)*mean(x,1);
            X = bsxfun(@rdivide, x, sqrt(sum(x.^2))+eps); 
            % x = x / norm(x);

            res = gm.getResponse(X);
            % res = normalize(res);

            cumu_resp(loc_idx, :) = cumu_resp(loc_idx, :) + res';
        end
        tpb(i);
    end

    for i = 1:19
        figure;
        imagesc(reshape(cumu_resp(i,:), 10, 10));
    end
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
    m = reshape(1:r*c, r, c)';

    for i=1:r*c
        subplot(r,c,i);
        [ir, ic] = ind2sub([r c], i);
        imagesc(reshape(gm.gsm{1}.bases{1}(:,m(ir,ic)), [], chunk_size * 2));
        colormap('jet');
        axis off;
        set(gca, 'YDir', 'normal');
    end

    % figure;
    % r = gm.topo_space(1);
    % c = gm.topo_space(2);
    % for i=1:r*c
    %     subplot(r,c,i);
    %     [ir, ic] = ind2sub([r c], i);
    %     b = reshape(gm.gsm{1}.bases{1}(:,m(ir,ic)), [], chunk_size * 2);
    %     if b(1) < 0
    %         b = -1 * b;
    %     end
    %     imagesc(b);
    %     colormap('jet');
    %     axis off;
    %     set(gca, 'YDir', 'normal');
    % end

    % figure;
    % img = [];
    % bs = gm.gsm{1}.bases{1}';
    % for i = 1:r*c
    %     b = bs(i, :);
    %     if b(1) < 0
    %         b = -1 * b;
    %     end
    %     img = [img;b];
    % end
    % imagesc(img);
    % colormap("jet");
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

function [err] = rms (truths, predicts)
    % truths and predicts are azimuth
    assert(length(truths) == length(predicts));
    err = sqrt(sum((predicts - truths) .^ 2) / length(truths));
end
