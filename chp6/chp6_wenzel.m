function [azimuth_predicts, azimuth_truths, cumu_resp] = testModel (...
    gm, trained_dnn, chunk_size, test_x, test_y ...
)
    azimuth_predicts = zeros(length(test_y), 1);
    azimuth_truths = zeros(length(test_y), 1);
    cumu_resp = zeros(gm.locs_num, prod(gm.topo_space));

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
            % res = normalize(res, "range");
            responses(i, :) = res;
            cumu_resp(azimuth_truth, :) = cumu_resp(azimuth_truth, :) + reshape(res, 1, []);
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

    for i = 1:length(azimuth_predicts)
        azimuth_predicts(i) = gm.locs_list(1,azimuth_predicts(i));
        azimuth_truths(i) = gm.locs_list(1,azimuth_truths(i));

        if azimuth_predicts(i) > 180
            azimuth_predicts(i) = azimuth_predicts(i) - 360;
        end
        if azimuth_truths(i) > 180
            azimuth_truths(i) = azimuth_truths(i) - 360;
        end
    end
end

function [azimuth_predicts, azimuth_truths, cumu_resp] = testModel2 (...
    gm, trained_dnn, chunk_size, test_x, test_y ...
)
    azimuth_predicts = zeros(length(test_y), 1);
    azimuth_truths = zeros(length(test_y), 1);
    cumu_resp = zeros(gm.locs_num, prod(gm.topo_space));

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
            res = normalize(res, "range");
            responses(i, :) = res;
            cumu_resp(azimuth_truth, :) = cumu_resp(azimuth_truth, :) + reshape(res, 1, []);
        end

        % predict response
        cum_predicted = zeros(1, length(gm.locs_list));
        for i_chk = 1:nChk
            res = responses(i_chk, :);
            predicted = predict(trained_dnn, res);
            disp(predicted);
            assert(false);
            cum_predicted = cum_predicted + predicted;
        end
        [~, azimuth_predict] = max(cum_predicted);

        azimuth_predicts(i_iter) = azimuth_predict;
        azimuth_truths(i_iter) = azimuth_truth;

        tpb(i_iter);
    end

    for i = 1:length(azimuth_predicts)
        azimuth_predicts(i) = gm.locs_list(1,azimuth_predicts(i));
        azimuth_truths(i) = gm.locs_list(1,azimuth_truths(i));

        if azimuth_predicts(i) > 180
            azimuth_predicts(i) = azimuth_predicts(i) - 360;
        end
        if azimuth_truths(i) > 180
            azimuth_truths(i) = azimuth_truths(i) - 360;
        end
    end
end

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

function [predicts, truths, cumu_resp] = test (gm, dnn, test_subject, chunk_size)
    test_data = load("chp6/cache/dnnTestGwn_cipic_" + test_subject + ".mat");
    [fp, ft, fr] = testModel(...
        gm, dnn, chunk_size, ...
        test_data.frontTestX, test_data.frontTestY ...
    );

    backTestY = test_data.backTestY;
    for i = 1:length(backTestY)
        backTestY{i} = backTestY{i} + gm.locs_num / 2;
    end
    [bp, bt, br] = testModel(...
        gm, dnn, chunk_size, ...
        test_data.backTestX, backTestY ...
    );

    predicts = [fp; bp];
    truths = [ft; bt];
    cumu_resp = fr + br;
end

function [predicts, truths, cumu_resp] = test2 (gm, dnn, test_subject, chunk_size)
    test_data = load("chp6/cache/dnnTestGwn_cipic_" + test_subject + ".mat");
    [fp, ft, fr] = testModel(...
        gm, dnn, chunk_size, ...
        test_data.frontTestX, test_data.frontTestY ...
    );

    backTestY = test_data.backTestY;
    for i = 1:length(backTestY)
        backTestY{i} = backTestY{i} + gm.locs_num / 2;
    end
    [bp, bt, br] = testModel(...
        gm, dnn, chunk_size, ...
        test_data.backTestX, backTestY ...
    );

    predicts = [fp; bp];
    truths = [ft; bt];
    cumu_resp = fr + br;
end

function [trained_dnn, train_info] = trainDNN (gm, chunk_size, hrtf, hrtf_subject)
    trainSamples = load("chp6/cache/dnnTrainGwn_" + hrtf + "_" + hrtf_subject + ".mat");

    XTrain = zeros(size(trainSamples.trainX, 1), prod(gm.topo_space));
    YTrain = categorical(cell2mat(trainSamples.trainY));

    nFrm = size(trainSamples.trainX{1}{1}, 2);

    rng(49);

    for i = 1:gm.netTrainParam.max_iter
        frmL = trainSamples.trainX{i}{1};
        frmR = trainSamples.trainX{i}{2};

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
        % res = normalize(res, "range");
        XTrain(i, :) = res;
    end

    rng(49);

    [dnn, training_option] = createDNN(prod(gm.topo_space), gm.locs_num, 2);
    [trained_dnn, train_info] = trainNetwork(XTrain, YTrain, dnn, training_option);
end

function [trained_dnn, train_info] = trainDNN2 (gm, chunk_size, hrtf, hrtf_subject)
    trainSamples = load("chp6/cache/dnnTrainGwn_" + hrtf + "_" + hrtf_subject + ".mat");

    
    XTrain = zeros(size(trainSamples.trainX, 1), prod(gm.topo_space));
    
    cos_sin_transform = [sind(gm.locs_list(1,:)); cosd(gm.locs_list(1,:))]';
    YTrain = zeros(length(trainSamples.trainY), 2);
    yTrainIdx = cell2mat(trainSamples.trainY);
    for idx = 1:length(yTrainIdx)
        YTrain(idx, :) = cos_sin_transform(idx, :);
    end

    nFrm = size(trainSamples.trainX{1}{1}, 2);

    rng(49);

    for i = 1:gm.netTrainParam.max_iter
        frmL = trainSamples.trainX{i}{1};
        frmR = trainSamples.trainX{i}{2};

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
        res = normalize(res, "range");
        XTrain(i, :) = res;
    end

    rng(49);

    [dnn, training_option] = createDNN(prod(gm.topo_space), gm.locs_num, 3);
    [trained_dnn, train_info] = trainNetwork(XTrain, YTrain, dnn, training_option);
end

function [errs, cm] = rms (truths, predicts)
    assert(length(truths) == length(predicts));

    figure;
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

function gassom_wenzel()
    map_size = 16;
    chunk_size = 10;

    computation_subjects = [9, 20, 40, 61, 65, 124, 134, 152, 154, 165];
    test_subjects = [
        [8, 15, 20, 44, 51, 65, 124, 131, 137, 155];
        [3, 18, 27, 48, 61, 126, 147, 153, 154, 163];
        [10, 12, 17, 27, 51, 61, 124, 131, 133, 163];
        [9, 12, 19, 33, 40, 58, 126, 131, 137, 158];
        [8, 11, 19, 27, 59, 137, 152, 153, 154, 165];
        [8, 20, 28, 58, 61, 65, 119, 126, 162, 163];
        [20, 44, 51, 65, 119, 126, 135, 137, 147, 153];
        [3, 18, 20, 27, 60, 61, 133, 134, 154, 156];
        [18, 21, 28, 33, 44, 48, 50, 126, 155, 162];
        [18, 21, 28, 44, 50, 65, 124, 127, 134, 135]
    ];

    for computation_subject_idx = 1:1
        % computation_subject_idx = 1;
        computation_subject = computation_subjects(computation_subject_idx);

        disp("run wenzel on subject " + computation_subject);

        save_folder = "chp6/comptational_model_16x16/subject_" + computation_subject + "/";
        % save_folder = "chp6/temp/";
        gm = loadGassom([map_size, map_size], 5e4, save_folder + "gsm.mat", chunk_size, "cipic", computation_subject);
        % trained_dnn = load(save_folder + "dnn.mat").trained_dnn;
        [trained_dnn, ~] = trainDNN(gm, chunk_size, "cipic", computation_subject);
        save(save_folder + "dnn_no_norm.mat", "trained_dnn");

        result_save_folder = "chp6/result/subject_" + computation_subject + "/individual/";
        if ~exist(result_save_folder,'dir')
            mkdir(result_save_folder);
        end
        [predicts, truths, cumu_resp] = test(gm, trained_dnn, computation_subject, chunk_size);
        [errs, cm] = rms(truths, predicts);
        saveas(gca, result_save_folder + "confusion.png");
        figure;
        bar([
            -170, -160, -150, -140, -125, ...
            -100, -80, -55, -40, -30, ...
            -20, -10, 0, 10, 20, ...
            30, 40, 55, 80, 100, ...
            125, 140, 150, 160, 170, 180], ...
            errs ...
        );
        ylim([-5 180]);
        xlabel("Target Azimuth/deg");
        ylabel("RMS/deg");
        saveas(gca, result_save_folder + "rms.png");
        figure;
        imagesc(cumu_resp);
        saveas(gca, result_save_folder + "response.png");
        save(result_save_folder + "result.mat", "predicts", "truths", "errs", "cumu_resp");

        for test_subject = test_subjects(computation_subject_idx,:)
            disp(test_subject);
            result_save_folder = "chp6/result/subject_" + computation_subject + "/subject_" + test_subject + "/";
            if ~exist(result_save_folder,'dir')
                mkdir(result_save_folder);
            end
            [predicts, truths, cumu_resp] = test(gm, trained_dnn, test_subject, chunk_size);
            [errs, cm] = rms(truths, predicts);
            saveas(gca, result_save_folder + "confusion.png");
            figure;
            bar([
                -170, -160, -150, -140, -125, ...
                -100, -80, -55, -40, -30, ...
                -20, -10, 0, 10, 20, ...
                30, 40, 55, 80, 100, ...
                125, 140, 150, 160, 170, 180], ...
                errs ...
            );
            ylim([-5 180]);
            xlabel("Target Azimuth/deg");
            ylabel("RMS/deg");
            saveas(gca, result_save_folder + "rms.png");
            figure;
            imagesc(cumu_resp);
            saveas(gca, result_save_folder + "response.png");
            save(result_save_folder + "result.mat", "predicts", "truths", "errs", "cumu_resp");
        end
    end
end

function plot_result ()
    computation_subjects = [9, 20, 40, 61, 65, 124, 134, 152, 154, 165];
    test_subjects = [
        [8, 15, 20, 44, 51, 65, 124, 131, 137, 155];
        [3, 18, 27, 48, 61, 126, 147, 153, 154, 163];
        [10, 12, 17, 27, 51, 61, 124, 131, 133, 163];
        [9, 12, 19, 33, 40, 58, 126, 131, 137, 158];
        [8, 11, 19, 27, 59, 137, 152, 153, 154, 165];
        [8, 20, 28, 58, 61, 65, 119, 126, 162, 163];
        [20, 44, 51, 65, 119, 126, 135, 137, 147, 153];
        [3, 18, 20, 27, 60, 61, 133, 134, 154, 156];
        [18, 21, 28, 33, 44, 48, 50, 126, 155, 162];
        [18, 21, 28, 44, 50, 65, 124, 127, 134, 135]
    ];
    x_labels = [
        -170, -160, -150, -140, -125, ...
        -100, -80, -55, -40, -30, ...
        -20, -10, 0, 10, 20, ...
        30, 40, 55, 80, 100, ...
        125, 140, 150, 160, 170, ...
        180
    ];

    % individual
    % individual_data = [];
    % for i = 1:10
    %     computation_subject = computation_subjects(i);
    %     tr = [];
    %     pr = [];
    %     for s = test_subjects(i, :)
    %         result_save_folder = "chp6/result/subject_" + computation_subject + "/individual/";
    %         mat = load(result_save_folder + "result.mat");
    %         tr = [tr; mat.truths];
    %         pr = [pr; mat.predicts];
    %     end
    %     errs = rms(tr, pr);
    %     individual_data = [individual_data, errs];
    % end
    % individual_mean = mean(individual_data, 2);
    % individual_data = individual_data';

    % non-individual
    non_individual_data = [];
    for i = 1:1
        computation_subject = computation_subjects(i);

        tr = [];
        pr = [];
        for s = test_subjects(i, :)
            result_save_folder = "chp6/result/subject_" + computation_subject + "/subject_" + s + "/";
            mat = load(result_save_folder + "result.mat");
            tr = [tr; mat.truths];
            pr = [pr; mat.predicts];
        end
        errs = rms(tr, pr);
        non_individual_data = [non_individual_data, errs];
    end
    bar(errs);
    assert(false);
    non_individual_mean = mean(non_individual_data, 2);
    non_individual_data = non_individual_data';

    figure;
    boxplot(individual_data);
    hold on;
    plot(individual_mean, 'o', 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'none');
    plot(individual_mean, '--r');
    hold off;
    ylim([-5 180]);
    xticklabels(x_labels)
    xlabel("Target Azimuth/deg");
    ylabel("RMS/deg");
    title("RMS error for localization of stimuli synthesized with individualized HRTF")

    figure;
    boxplot(non_individual_data);
    hold on;
    plot(non_individual_mean, 'o', 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'none');
    plot(non_individual_mean, '--r');
    hold off;
    ylim([-5 180]);
    xticklabels(x_labels)
    xlabel("Target Azimuth/deg");
    ylabel("RMS/deg");
    title("RMS error for localization of stimuli synthesized with non-individualized HRTF")
end

% gassom_wenzel();
plot_result();
