clc; clear all; addpath(genpath(pwd));

map_size = 10;
chunk_size = 10;
cipic_subjects = load("wenzel_cipic_subject/wenzel_cipic_subject.mat");

% % i_errs = [];
% % ni_errs = [];
for computation_subject_idx = 1:10
    % computation_subject_idx = 1;
    % computation_subject = computation_subjects(computation_subject_idx);
    computation_subject = cipic_subjects.individual_subjects(computation_subject_idx);

    disp("run wenzel on subject " + computation_subject);

    save_folder = "chp4/cipic_360/cochleagram/10x10_10/" + computation_subject + "/";
    gm = loadGassom([map_size, map_size], 5e4, save_folder + "gsm.mat", chunk_size, "cipic", computation_subject);
    trained_dnn = load(save_folder + "dnn_batch.mat").trained_dnn;

    [predicts, truths, cumu_resp] = testModel(gm, trained_dnn, chunk_size, "cache/dnnTestGwn_cipic_" + computation_subject + "_2600.mat");
    [i_errs, cm] = rms(truths, predicts);
    % figure;
    % bar([
    %     -170, -160, -150, -140, -125, ...
    %     -100, -80, -55, -40, -30, ...
    %     -20, -10, 0, 10, 20, ...
    %     30, 40, 55, 80, 100, ...
    %     125, 140, 150, 160, 170, 180], ...
    %     errs ...
    % );
    % ylim([-5 180]);
    % xlabel("Target Azimuth/deg");
    % ylabel("RMS/deg");

    ni_errs = [];
    for test_subject = cipic_subjects.non_individual_subjects(computation_subject_idx,:)
        disp(test_subject);
        [predicts, truths, cumu_resp] = testModel(gm, trained_dnn, chunk_size, "cache/dnnTestGwn_cipic_" + test_subject + "_2600.mat");
        [errs, cm] = rms(truths, predicts);
        % figure;               
        % bar([
        %     -170, -160, -150, -140, -125, ...
        %     -100, -80, -55, -40, -30, ...
        %     -20, -10, 0, 10, 20, ...
        %     30, 40, 55, 80, 100, ...
        %     125, 140, 150, 160, 170, 180], ...
        %     errs ...
        % );
        % ylim([-5 180]);
        % xlabel("Target Azimuth/deg");
        % ylabel("RMS/deg");
        ni_errs = [ni_errs, errs];
    end

    save(save_folder + "wenzel_result_batch.mat", "ni_errs", "i_errs");
end

plot_result()

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

function [azimuth_predicts, azimuth_truths, cumu_resp] = testModel (...
    gm, trained_dnn, chunk_size, dataset_path ...
)
    test_data = load(dataset_path);

    azimuth_predicts = zeros(gm.netTestParam.max_iter, 1);
    azimuth_truths = zeros(gm.netTestParam.max_iter, 1);
    cumu_resp = zeros(gm.locs_num, prod(gm.topo_space));

    nFrm = size(test_data.testX{1}{1}, 2);

    tpb = textprogressbar(gm.netTestParam.max_iter, 'showremtime', true);
    for i_iter = 1:gm.netTestParam.max_iter
        YTest = test_data.testY{i_iter};

        nChk = length(0:1:nFrm-chunk_size);

        frmL = test_data.testX{i_iter}{1};
        frmR = test_data.testX{i_iter}{2};

        % get response from gassom
        responses = zeros(nChk, prod(gm.topo_space));
        for i = 1:size(responses,1)
            chkL = frmL(:,i:(chunk_size + i - 1));
            chkR = frmR(:,i:(chunk_size + i - 1));

            x = [reshape(chkL, [], 1);reshape(chkR, [], 1)];
            x = x-ones(size(x,1),1)*mean(x,1);
            X = bsxfun(@rdivide, x, sqrt(sum(x.^2))+eps); 

            res = gm.getResponse(X);

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
        end
        [~, yp] = max(cum_predicted);

        azimuth_predicts(i_iter) = yp;
        azimuth_truths(i_iter) = YTest(1);

        tpb(i_iter);
    end

    for i = 1:length(azimuth_predicts)
        azimuth_predicts(i) = gm.locs_list(1,azimuth_predicts(i));
        azimuth_truths(i) = gm.locs_list(1,azimuth_truths(i));

        % tune the azimuth into the range of -180 to 170 deg
        if azimuth_predicts(i) > 180
            azimuth_predicts(i) = azimuth_predicts(i) - 360;
        end
        if azimuth_truths(i) > 180
            azimuth_truths(i) = azimuth_truths(i) - 360;
        end
    end
end

function [errs, cm] = rms (truths, predicts)
    assert(length(truths) == length(predicts));

    figure('visible','off');
    cm = confusionchart(truths, predicts);

    errs = zeros(length(cm.ClassLabels), 1);
    for i = 1:length(cm.ClassLabels)
        truth = cm.ClassLabels(i);
        if truth < 0
            truth = truth + 360;
        end

        err = 0;
        for j = 1:length(cm.ClassLabels)
            if i == j
                continue
            end
            if cm.NormalizedValues(i, j) == 0
                continue
            end
            predict = cm.ClassLabels(j);
            if predict < 0
                predict = predict + 360;
            end

            e = abs(predict - truth);
            e = min(e, 360 - e);

            err = err + e ^ 2 * cm.NormalizedValues(i, j);
        end
        errs(i) = sqrt(err / sum(cm.NormalizedValues(i, :)));
    end
end

function plot_result ()
    cipic_subjects = load("wenzel_cipic_subject/wenzel_cipic_subject.mat");

    i_errs = [];
    ni_errs = [];
    for computation_subject_idx = 1:10
        computation_subject = cipic_subjects.individual_subjects(computation_subject_idx);
        data = load("chp4/cipic_360/cochleagram/10x10_10/" + computation_subject + "/wenzel_result_batch.mat");
        i_errs = [i_errs, data.i_errs];
        ni_errs = [ni_errs, mean(data.ni_errs, 2)];
    end

    figure;
    boxplot(i_errs');
    ylim([-5, 180]);
    figure;
    boxplot(ni_errs');
    ylim([-5, 180]);
    hold on;
    plot(mean(ni_errs, 2));
end
