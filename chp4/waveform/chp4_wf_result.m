function plot()
    result_folder = "chp4/waveform/result";

    meas = [...
        load(result_folder + "/8x8/mae.mat").mae, ...
        load(result_folder + "/10x10/mae.mat").mae, ...
        load(result_folder + "/16x16/mae.mat").mae, ...
        load(result_folder + "/20x20/mae.mat").mae
    ];
    stds = [...
        load(result_folder + "/8x8/bmt_std.mat").bmt_std, ...
        load(result_folder + "/10x10/bmt_std.mat").bmt_std, ...
        load(result_folder + "/16x16/bmt_std.mat").bmt_std, ...
        load(result_folder + "/20x20/bmt_std.mat").bmt_std
    ];
    % winners = [
    %     load(result_folder + "/8x8/norm_winners.mat").norm_winners; ...
    %     load(result_folder + "/10x10/norm_winners.mat").norm_winners; ...
    %     load(result_folder + "/16x16/norm_winners.mat").norm_winners; ...
    %     load(result_folder + "/20x20/norm_winners.mat").norm_winners
    % ];

    figure;
    bar(meas);
    title("MEA between map sizes");
    xticklabels({"8x8", "10x10", "16x16", "20x20"});
    xlabel("map size");
    ylabel("mae (deg)");
    ylim([0 max(meas) * 1.2]);

    figure;
    bar(stds);
    title("STD of BMT between map sizes");
    xticklabels({"8x8", "10x10", "16x16", "20x20"});
    xlabel("map size");
    ylabel("std");
    ylim([0 max(stds) * 1.2]);

    % figure;
    % boxplot(winners);
end

function visualizeClassification()
    predicts = load('chp4/waveform/result/20x20/azimuth_predicts.mat').azimuth_predicts;
    truths = load('chp4/waveform/result/20x20/azimuth_truths.mat').azimuth_truths;

    figure;
    confusionchart(truths, predicts);

    locations = {...
        '-90', '-80', '-70', '-60', '-50', '-40', '-30', '-20', '-10', ...
        '0', ...
        '10', '20', '30', '40', '50', '60', '70', '80', '90' ...
    };

    dict = containers.Map(locations, cell(size(locations)));

    for i = 1:1:length(predicts)
        p = predicts(i);
        t = truths(i);
        l = dict(num2str(t));
        dict(num2str(t)) = [l; abs(p - t)];
    end

    for loc = locations
        l = loc{1};
        disp(l + ": " + mean(dict(l)));
    end
end

plot();
% visualizeClassification()
