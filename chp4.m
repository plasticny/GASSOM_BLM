function plotCochl (frm, cfs, title_str, cmin, cmax)
    assert(size(cfs,2) == 128)
    imagesc(frm);
    title(title_str);
    ylabel("frequency (hz)")
    xlabel("time frame")
    yticks([1 16 32 48 64 80 96 112 128]);
    yticklabels([cfs(1,1), cfs(1,16), cfs(1,32), cfs(1,48), cfs(1,64), cfs(1,80), cfs(1,96), cfs(1,112), cfs(1,128)]);
    colormap('jet');
    caxis([cmin cmax]);
    set(gca, 'YDir', 'normal');
end

function visualizeTrainingSample (gm, ind)
    y_all = gm.env.timit_train{gm.somTrainParam.audio_idx(ind),1};
    y = y_all(gm.somTrainParam.audio_bgn(ind,1)+(1:gm.somTrainParam.audio_len));

    [frm, cfs] = audio2cochlIOSR(...
        y, ...
        gm.env.fs, 100, 20000, 128, ...
        8, 4 ...
    );

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

clc; clear all; addpath(genpath(pwd));

rng(42);

% sofaloaded = SOFALoader;
% save('sofaloaded.mat', 'sofaloaded');
load temp_data/sofaloaded.mat;

fs = 44100;
patch_dur = 0; % should be not neccessary in this chapter
topo_space = [10 10];
max_iter = 5e4;
% max_iter = 1000;
gm_param = {fs, patch_dur, topo_space, max_iter};
GM = GASSOM_Model(gm_param);

GM.trainGASSOM_cochleagram_IOSR();

% visualizeTrainingSample(GM, 100)
visualizeCochlMap(GM);