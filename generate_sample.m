clc; clear all; addpath(genpath(pwd));

gm = initGassom([10, 10], 5e4, 10, "kemar", 0);
% gm = loadGassom([10, 10], 5e4, "chp4/cipic/cochleagram/10x10_10/gsm.mat", 10, "cipic", 9);
% gm = loadWaveformGassom([16 16], 5e4, "chp4/vecnorm/waveform/16x16_16ms/gsm.mat", 16 / 1000, "kemar", 0);
% [trainX, trainY] = generateGassomTrainSamples(gm, "kemar", 0);
[trainX, trainY] = generateDnnTrainSamples(gm, "kemar", 0, true, 250 / (2 ^ (1)), 250 * (2 ^ (1)));
[testX, testY] = generateDnnTestSamples(gm, "kemar", 0, false);
% [testX, testY] = generateDnnWaveformTestSampless(gm, "kemar", 0, true, 125, 500);
% generateDnnTrainSamplesTimit(gm);

% gwn = load("cache/dnnTrainGwn_kemar_0_1900.mat");
% bandpassed = load("cache/dnnTrainGwn_kemar_0_1900_bandpass_all_cf_bw.mat");
% trainX = [bandpassed.trainX; gwn.trainX];
% trainY = [bandpassed.trainY; gwn.trainY];
% save("cache/dnnTrainGwn_kemar_0_1900_all.mat", "trainX", "trainY", "-v7.3");

% trainX = [];
% trainY = [];
% for cf = [250, 2000, 4000]
%     for bw = [1/6, 1/3, 1, 2]
%         disp(cf + " " + bw);

%         lb = cf / (2 ^ (bw/2));
%         ub = cf * (2 ^ (bw/2));

%         [x, y] = gm.env.genGwnToolbox(...
%             gm.locs_list, gm.locs_num * 100, ...
%             gm.netTrainParam.audio_len, gm.env.fs, ...
%             "cipic", 8, ...
%             gm.netTrainParam.gwn_seed, ...
%             true, lb, ub ...
%         );
%         trainX = [trainX; x];
%         trainY = [trainY; y];
%     end
% end
% disp("board band");
% [x, y] = generateDnnTrainSamples(gm, "cipic", 8, false, 250 / (2 ^ (1)), 250 * (2 ^ (1)));
% trainX = [trainX; x];
% trainY = [trainY; y];
% save("cache/dnnTrainGwn_cipic_8_2600_all.mat", "trainX", "trainY", "-v7.3");

% cipic_subjects = load("wenzel_cipic_subject/wenzel_cipic_subject.mat");
% for i = 1:10
%     s = cipic_subjects.individual_subjects(i);
%     if exist("cache/somTrainSamples_cipic_" + s + "_26_50000.mat", "file")
%         disp("skip " + s);
%         continue
%     end
%     disp("generating individual " + s);
%     gm = initGassom([10, 10], 5e4, 10, "cipic", s);
%     assert(gm.locs_num == 26);
%     generateGassomTrainSamples(gm, "cipic", s);
%     generateDnnTrainSamples(gm, "cipic", s, false);

%     for j = 1:10
%         s = cipic_subjects.non_individual_subjects(i,j);
%         if exist("cache/dnnTestGwn_cipic_" + s + "_2600.mat", "file")
%             disp("skip " + s);
%             continue
%         end
%         disp("generating non-individual " + s);
%         gm = initGassom([10, 10], 5e4, 10, "cipic", s);
%         assert(gm.locs_num == 26);
%         generateDnnTestSamples(gm, "cipic", s, false);
%     end
% end

function [gm] = initGassom (topo_space, max_iter, chunk_size, hrtf_database, hrtf_subject)
    rng(49);

    % sofaloaded = SOFALoader;
    % save('sofaloaded.mat', 'sofaloaded');
    load temp_data/sofaloaded.mat;

    disp("init gassom start");
    fs = 44100;
    patch_dur = 0; % should be not neccessary in this chapter
    gm_param = {fs, patch_dur, topo_space, max_iter};
    gm = GASSOM_Model(gm_param, 128 * chunk_size, "cochleagram", hrtf_database, hrtf_subject);
    gm.setHRTFandSubject(hrtf_database, hrtf_subject);
    disp("init gassom end");
end

function [gm] = loadGassom (topo_space, max_iter, gsm_path, chunk_size, hrtf_database, hrtf_subject)
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

    gm.gsm{1} = load(gsm_path).gsm;
end

function [gm] = loadWaveformGassom (topo_space, max_iter, gsm_path, patch_dur, hrtf_database, hrtf_subject)
    rng(49);

    load temp_data/sofaloaded.mat;

    disp("init gassom start")
    fs = 44100;
    gm_param = {fs, patch_dur, topo_space, max_iter};
    gm = GASSOM_Model(gm_param, round(fs * patch_dur), "waveform");
    gm.setHRTFandSubject(hrtf_database, hrtf_subject);
    disp("init gassom end");

    gm.gsm{1} = load(gsm_path).gsm;
end


function [trainX, trainY] = generateGassomTrainSamples (gm, hrtf, hrtf_subject)
    gm.somTrainParam.input_type = "timit";
    gm.somTrainParam.hrtf = hrtf;
    gm.somTrainParam.subject = hrtf_subject;

    sampleSize = gm.somTrainParam.max_iter;
    trainX = cell(sampleSize, 2);
    trainY = cell(sampleSize, 1);
    
    tpd = textprogressbar(sampleSize, "showremtime", true);
    for i = 1:sampleSize
        [frmL, frmR, ~] = gm.env.genOneEpisodeCoch2(gm.somTrainParam, i);
        trainX{i}{1} = frmL;
        trainX{i}{2} = frmR;
        trainY{i} = gm.somTrainParam.locs_rand(i);
        tpd(i);
    end

    save("cache/somTrainSamples_" + gm.somTrainParam.hrtf + "_" + gm.somTrainParam.subject + "_26_" + sampleSize + ".mat", "trainX", "trainY", "-v7.3");
end

function [trainX, trainY] = generateDnnTrainSamples (...
    gm, hrtf, hrtf_subject, ...
    do_bandpass, lb, ub ...
)
    if do_bandpass
        [trainX, trainY] = gm.env.genGwnToolbox(...
            gm.locs_list, gm.locs_num * 100, ...
            gm.netTrainParam.audio_len, gm.env.fs, ...
            hrtf, hrtf_subject, ...
            gm.netTrainParam.gwn_seed, ...
            true, lb, ub ...
        );
        save("cache/dnnTrainGwn_" + hrtf + "_" + hrtf_subject + "_" + length(trainY) + "_bandpass_" + lb + "_" + ub + ".mat", "trainX", "trainY");
    else
        [trainX, trainY] = gm.env.genGwnToolbox(...
            gm.locs_list, gm.locs_num * 100, ...
            gm.netTrainParam.audio_len, gm.env.fs, ...
            hrtf, hrtf_subject, ...
            gm.netTrainParam.gwn_seed, ...
            false ...
        );
        save("cache/dnnTrainGwn_" + hrtf + "_" + hrtf_subject + "_" + length(trainY) + ".mat", "trainX", "trainY");
    end
end

function generateDnnTrainSamplesTimit (gm)
    sample_size = gm.netTrainParam.max_iter;
    trainX = cell(sample_size, 2);
    trainY = cell(sample_size, 1);

    tpd = textprogressbar(sample_size, 'showremtime', true);
    for ind = 1:sample_size
        [frmL, frmR, ~] = gm.env.genOneEpisodeCoch2(gm.netTrainParam, ind);
        trainX{ind}{1} = frmL;
        trainX{ind}{2} = frmR;
        trainY{ind} = gm.netTrainParam.locs_rand(ind);
        tpd(ind);
    end

    save("cache/dnnTrainTimit_" + gm.netTrainParam.hrtf + "_" + gm.netTrainParam.subject + "_" + sample_size + ".mat", "trainX", "trainY");
end

function [testX, testY] = generateDnnTestSamples ( ...
    gm, hrtf, hrtf_subject, ...
    do_bandpass, lb, ub ...
)
    if do_bandpass
        [testX, testY] = gm.env.genGwnToolbox(...
            gm.locs_list, gm.locs_num * 100, ...
            gm.netTestParam.audio_len, gm.env.fs, ...
            hrtf, hrtf_subject, ...
            gm.netTestParam.gwn_seed, ...
            true, lb, ub ...
        );
        save("cache/dnnTestGwn_" + hrtf + "_" + hrtf_subject + "_" + length(testY) + "_bandpass_" + lb + "_" + ub + ".mat", "testX", "testY");
    else
        [testX, testY] = gm.env.genGwnToolbox(...
            gm.locs_list, gm.locs_num * 100, ...
            gm.netTestParam.audio_len, gm.env.fs, ...
            hrtf, hrtf_subject, ...
            gm.netTestParam.gwn_seed, ...
            false ...
        );
        save("cache/dnnTestGwn_" + hrtf + "_" + hrtf_subject + "_" + length(testY) + ".mat", "testX", "testY");
    end
end

function [testX, testY] = generateDnnWaveformTestSampless ( ...
    gm, hrtf, hrtf_subject, ...
    do_bandpass, lb, ub ...
)
    if do_bandpass
        [testX, testY] = gm.env.genGwn(...
            gm.locs_list, gm.locs_num * 100, ...
            gm.netTestParam.audio_len, gm.env.fs, ...
            hrtf, hrtf_subject, ...
            gm.netTestParam.gwn_seed, ...
            true, lb, ub ...
        );
        save("cache/dnnTestGwnWF_" + hrtf + "_" + hrtf_subject + "_" + length(testY) + "_bandpass_" + lb + "_" + ub + ".mat", "testX", "testY");
    else
        [testX, testY] = gm.env.genGwn(...
            gm.locs_list, gm.locs_num * 100, ...
            gm.netTestParam.audio_len, gm.env.fs, ...
            hrtf, hrtf_subject, ...
            gm.netTestParam.gwn_seed, ...
            false ...
        );
        save("cache/dnnTestGwnWF_" + hrtf + "_" + hrtf_subject + "_" + length(testY) + ".mat", "testX", "testY");
    end
end

function generateYostTestSample (gm, lb, ub, dur)

end
