function [gm] = initGassom (topo_space, max_iter)
    rng(1001);

    % sofaloaded = SOFALoader;
    % save('sofaloaded.mat', 'sofaloaded');
    load temp_data/sofaloaded.mat;

    disp("init gassom start")
    fs = 44100;
    patch_dur = 0; % should be not neccessary in this chapter
    % max_iter = 1000;
    gm_param = {fs, patch_dur, topo_space, max_iter};
    gm = GASSOM_Model(gm_param);
    disp("init gassom end")
end

function [gm] = loadGassom (topo_space, max_iter, gsm_path)
    gm = initGassom(topo_space, max_iter);
    gm.gsm{1} = load(gsm_path).gsm;
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

function visualizeHtfs (gm)
    y = gm.env.timit_train{1,1};

    loc = gm.env.locs_list(:,gm.somTrainParam.locs_rand(1));
    disp(loc);

    bi = gm.env.sofa.spatMono(y, loc, "kemar", "");
    biL = bi(:,1);
    biR = bi(:,2);

    figure;
    subplot(3, 1, 1);
    plot(y);
    title("before hrtf");
    subplot(3, 1, 2);
    plot(biL);
    title("left");
    subplot(3, 1, 3);
    plot(biR);
    title("right");
end

function visualizeGassomTrainingSample (gm, ind)
    y_all = gm.env.timit_train{gm.somTrainParam.audio_idx(ind),1};
    y = y_all(gm.somTrainParam.audio_bgn(ind,1)+(1:gm.somTrainParam.audio_len));

    [frm, cfs] = audio2cochlIOSR(...
        y, ...
        gm.env.fs, 100, 20000, 128, ...
        8, 4 ...
    );

    disp(gm.env.locs_list(:,gm.somTrainParam.locs_rand(ind)));

    [frmL, frmR, ~] = gm.env.genOneEpisodeCochIOSR(gm.somTrainParam, ind);

    % normalize
    sample_len = size(frmL, 1);
    normalized = normalize([frmL; frmR; frm]);
    frmL = normalized(1:sample_len,:);
    frmR = normalized(sample_len+1:2*sample_len,:);
    frm = normalized(2*sample_len:end,:);

    blkL = frmL(:,1:5);
    blkR = frmR(:,1:5);

    frmMax = max(normalized,[],"all");
    frmMin = min(normalized,[],"all");
    blkMax = max([blkL; blkR],[],"all");
    blkMin = min([blkL; blkR],[],"all");

    
    figure;
    subplot(1, 1, 1);
    plot(y);

    figure;
    subplot(3, 1, 1);
    plotCochl(frm, cfs, "timit segment", frmMin, frmMax);
    subplot(3, 1, 2);
    plotCochl(frmL, cfs, "normalized left", frmMin, frmMax);
    subplot(3, 1, 3);
    plotCochl(frmR, cfs, "normalized right", frmMin, frmMax);
    
    figure;
    subplot(3, 1, 1);
    plotCochl(blkL, cfs, "a chunk (5 frames) of normalized left", blkMin, blkMax);
    subplot(3, 1, 2);
    plotCochl(blkR, cfs, "a chunk (5 frames) of normalized right", blkMin, blkMax);
    subplot(3, 1, 3);
    plotCochl([blkL blkR], cfs, "sample that feed into gassom", blkMin, blkMax);
end

function visualizeDnnTrainingSample (gm)
    y = gm.env.genStimuli("GWN", gm.netTrainParam.audio_len / gm.env.fs);

    disp(mean(y))

    [frm, cfs] = audio2cochlIOSR(...
        y, ...
        gm.env.fs, 100, 20000, 128, ...
        8, 4 ...
    );

    bi = gm.env.sofa.spatMono(y, gm.locs_list(:, 1), gm.netTrainParam.hrtf, gm.netTrainParam.subject);
    frmL = audio2cochlIOSR(...
        bi(:,1), ...
        gm.env.fs, 100, 20000, 128, ...
        8, 4 ...
    );
    frmR = audio2cochlIOSR(...
        bi(:,2), ...
        gm.env.fs, 100, 20000, 128, ...
        8, 4 ...
    );

    sample_len = size(frmL, 1);
    normalized = normalize([frmL; frmR; frm]);
    frmL = normalized(1:sample_len,:);
    frmR = normalized(sample_len+1:2*sample_len,:);
    frm = normalized(2*sample_len:end,:);

    blkL = frmL(:,1:5);
    blkR = frmR(:,1:5);

    frmMax = max(normalized,[],"all");
    frmMin = min(normalized,[],"all");
    blkMax = max([blkL; blkR],[],"all");
    blkMin = min([blkL; blkR],[],"all");

    figure;
    subplot(1, 1, 1);
    plot(y);
    title("waveform gassian white noise");

    figure;
    subplot(3, 1, 1);
    plotCochl(frm, cfs, "normalized gassian while noise", frmMin, frmMax);
    subplot(3, 1, 2);
    plotCochl(frmL, cfs, "normalized left", frmMin, frmMax);
    subplot(3, 1, 3);
    plotCochl(frmR, cfs, "normalized right", frmMin, frmMax);

    figure;
    subplot(3, 1, 1);
    plotCochl(blkL, cfs, "a chunk (5 frames) of normalized left", blkMin, blkMax);
    subplot(3, 1, 2);
    plotCochl(blkR, cfs, "a chunk (5 frames) of normalized right", blkMin, blkMax);
    subplot(3, 1, 3);
    plotCochl([blkL blkR], cfs, "sample that feed into gassom", blkMin, blkMax);
end

function visualizeCochlMap (gm)
    figure;
    r = gm.topo_space(1);
    c = gm.topo_space(2);
    for i=1:r*c
        subplot(r,c,i);
        imagesc(reshape(gm.gsm{1}.bases{1}(:,i), [], 10));
        colormap('jet');
        axis off;
        set(gca, 'YDir', 'normal');
    end
end

% clc; clear all; addpath(genpath(pwd));

%%% train gassom
% gm = initGassom([10, 10], 5e4);
% gm.trainGASSOM_cochleagram_IOSR();
% visualizeCochlMap(gm);

% gm = loadGassom([10, 10], 5e4, "chp4_result/gsm_10x10_sz5_sf1_larger_lr.mat");
% dnn = createDNN(prod(gm.topo_space), gm.locs_num, 3);

visualizeHtfs(gm);
% visualizeGassomTrainingSample(gm, 300)
% visualizeDnnTrainingSample(gm);

% [x, y] = gm.env.genTrainGwnIosr(gm.locs_list, gm.netTrainParam.max_iter, gm.netTrainParam.audio_len, gm.env.fs, "kemar", "", gm.netTrainParam.gwn_seed);