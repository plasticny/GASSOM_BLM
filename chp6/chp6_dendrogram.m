function check_cipic_position
    cipic_folder_path = '/home/nycheung/desktop/shuto_gassom/GASSOM_BLM/Dataset/cipic_sofa/';
    folder_strcut = dir(cipic_folder_path);

    % check by source position in the data
    eps = 10e-3;
    for i = 1:length(folder_strcut)
        name = folder_strcut(i).name;
        if endsWith(name, ".sofa")
            hrtf = SOFAload([cipic_folder_path name]);
            front_position = hrtf.SourcePosition(609, :);
            back_position = hrtf.SourcePosition(641, :);
            assert(abs(front_position(1)) < eps);
            assert(abs(front_position(2)) < eps);
            assert(abs(back_position(1) - 180) < eps);
            assert(abs(back_position(2)) < eps);
        end
    end

    % check by spatiaize audio and listen to it
    [y, sr] = audioread('/home/nycheung/desktop/shuto_gassom/GASSOM_BLM/chp6/cache/test.wav');
    hrtf = SOFAload('/home/nycheung/desktop/shuto_gassom/GASSOM_BLM/Dataset/cipic_sofa/subject_009.sofa');

    front = SOFAspat(y,hrtf,0,0);
    back = SOFAspat(y,hrtf,180,0);

    audiowrite('/home/nycheung/desktop/shuto_gassom/GASSOM_BLM/chp6/cache/test_front_1.wav', front(:,1), sr);
    audiowrite('/home/nycheung/desktop/shuto_gassom/GASSOM_BLM/chp6/cache/test_front_2.wav', front(:,2), sr);
    audiowrite('/home/nycheung/desktop/shuto_gassom/GASSOM_BLM/chp6/cache/test_back_1.wav', back(:,1), sr);
    audiowrite('/home/nycheung/desktop/shuto_gassom/GASSOM_BLM/chp6/cache/test_back_2.wav', back(:,2), sr);
end

function [y] = estimate_spectrum_v1 (x)
    y = fft(x);
    y = abs(y);
    y = y(1:end/2);
end

function [y] = estimate_spectrum_v2 (x)
    % from plotDFT
    L = length (x);
    Y = fft(x);
    P2 = mag2db(abs(Y/L));
    y = P2(1:L/2+1);
    % fs = 44100;
    % f = fs*(0:(L/2))/L;
end

function [y] = estimate_spectrum_v3 (x)
    % from SOFAplotHRTF
    y = 20*log10(abs(fft(x')));
    y = y(:,1:floor(size(y,2)/2));
    y = y';
end

function collect_cipic_spectrum
    cipic_folder_path = '/home/nycheung/desktop/shuto_gassom/GASSOM_BLM/Dataset/cipic_sofa/';
    folder_strcut = dir(cipic_folder_path);

    data = cell(45, 1);
    cnt = 1;
    for i = 1:length(folder_strcut)
        name = folder_strcut(i).name;
        if endsWith(name, ".sofa")
            hrtf = SOFAload([cipic_folder_path name]);
            ir = hrtf.Data.IR;
            front_ir = ir(609, :, :);
            back_ir = ir(641, :, :);
            data{cnt} = struct(...
                "subject_name", hrtf.GLOBAL_ListenerShortName, ...
                "front_ir_1", estimate_spectrum_v1(squeeze(front_ir(:, 1, :))), ...
                "front_ir_2", estimate_spectrum_v1(squeeze(front_ir(:, 2, :))), ...
                "back_ir_1", estimate_spectrum_v1(squeeze(back_ir(:, 1, :))), ...
                "back_ir_2", estimate_spectrum_v1(squeeze(back_ir(:, 2, :))) ...
            );
            cnt = cnt + 1;
        end
    end

    save("chp6/cache/cipic_spectrum.mat", "data");
end

function [mask, f] = get_freq_mask (freq_ranges)
    sr = 44100;
    L = 200;
    f = sr/L*(0:(L/2)-1);
    mask = zeros(size(f));

    assert(size(freq_ranges, 2) == 2);

    for freq_idx = 1:length(f)
        freq = f(freq_idx);
        for critical_range = freq_ranges'
            lower = critical_range(1);
            upper = critical_range(2);
            if (freq >= lower && freq <= upper)
                mask(freq_idx) = 1;
                break
            end
        end
    end

    mask = mask';
end

function extract_critical_bands_fasle
    % get frequency mask
    front_freq_ranges = [
        [3800, 8000]
        [13200, 16000]
        [150, 540]
        [1900, 2900]
        [3600, 5800]
        [8000, 16000]
    ];
    back_freq_ranges = [
        [10000, 13000]
        [720, 1700]
        [7400, 11100]
    ];
    [front_freq_mask, f] = get_freq_mask(front_freq_ranges);
    [back_freq_mask, ~] = get_freq_mask(back_freq_ranges);

    % extract
    spectrum_data = load("chp6/cache/cipic_spectrum.mat").data;
    subject_cnt = length(spectrum_data);

    front_extracted = zeros(subject_cnt * 2, length(nonzeros(front_freq_mask)));
    back_extracted = zeros(subject_cnt * 2, length(nonzeros(back_freq_mask)));
    % front_extracted = zeros(subject_cnt * 2, length(front_freq_mask));
    % back_extracted = zeros(subject_cnt * 2, length(back_freq_mask));

    for i = 1:length(spectrum_data)
        subject_data = spectrum_data{i};

        % disp(length(subject_data.front_ir_1));
        % disp(length(front_freq_mask));
        assert(length(subject_data.front_ir_1) == length(front_freq_mask));
        assert(length(subject_data.back_ir_1) == length(back_freq_mask));

        figure;
        subplot(2,1,1);
        plot(f, subject_data.front_ir_1);
        title("original hrtf (front)");
        subplot(2,1,2);
        plot(f, subject_data.front_ir_1 .* front_freq_mask);
        title("after remove non critical band (front)");
        assert(false);

        front_extracted(2 * i - 1, :) = subject_data.front_ir_1(front_freq_mask == 1);
        front_extracted(2 * i, :) = subject_data.front_ir_2(front_freq_mask == 1);
        back_extracted(2 * i - 1, :) = subject_data.back_ir_1(back_freq_mask == 1);
        back_extracted(2 * i, :) = subject_data.back_ir_2(back_freq_mask == 1);
        % front_extracted(2 * i - 1, :) = subject_data.front_ir_1;
        % front_extracted(2 * i, :) = subject_data.front_ir_2;
        % back_extracted(2 * i - 1, :) = subject_data.back_ir_1;
        % back_extracted(2 * i, :) = subject_data.back_ir_2;

    end

    save("chp6/cache/extracted_hrtfs.mat", "front_extracted", "back_extracted");
end

function [ret] = extract_mag (fft, freq)
    [mask, ~] = get_freq_mask(freq);
    ret = fft(mask == 1);
end

function [front_1_feature, front_2_feature, back_1_feature, back_2_feature] = extract_critical_bands (obj)
    f1_hz = [3800, 8000];
    f2_hz = [13200, 16000];
    f3_hz = [150, 540];
    f4_hz = [1900, 2900];
    f5_hz = [3600, 5800];
    b1_hz = [10000, 13000];

    ir = obj.Data.IR;

    front_ir = ir(609, :, :);
    front_ir_1 = squeeze(front_ir(:, 1, :));
    front_ir_2 = squeeze(front_ir(:, 2, :));
    front_fft_1 = estimate_spectrum_v1(front_ir_1);
    front_fft_2 = estimate_spectrum_v1(front_ir_2);

    back_ir = ir(641, :, :);
    back_ir_1 = squeeze(back_ir(:, 1, :));
    back_ir_2 = squeeze(back_ir(:, 1, :));
    back_fft_1 = estimate_spectrum_v1(back_ir_1);
    back_fft_2 = estimate_spectrum_v1(back_ir_2);

    front_1_feature = [...
        mean(extract_mag(front_fft_1, f1_hz)), ...
        mean(extract_mag(front_fft_1, f2_hz)), ...
        mean(extract_mag(front_fft_1, f3_hz) - extract_mag(back_fft_1, f3_hz)), ...
        mean(extract_mag(front_fft_1, f4_hz) - extract_mag(back_fft_1, f4_hz)), ...
        mean(extract_mag(front_fft_1, f5_hz) - extract_mag(back_fft_1, f5_hz)), ...
    ];
    front_2_feature = [...
        mean(extract_mag(front_fft_2, f1_hz)), ...
        mean(extract_mag(front_fft_2, f2_hz)), ...
        mean(extract_mag(front_fft_2, f3_hz) - extract_mag(back_fft_2, f3_hz)), ...
        mean(extract_mag(front_fft_2, f4_hz) - extract_mag(back_fft_2, f4_hz)), ...
        mean(extract_mag(front_fft_2, f5_hz) - extract_mag(back_fft_2, f5_hz)), ...
    ];
    back_1_feature = [
        mean(extract_mag(back_fft_1, b1_hz)), ...
        mean(extract_mag(front_fft_1, b2_hz) - extract_mag(back_fft_1, b2_hz)), ...
        mean(extract_mag(front_fft_1, b3_hz) - extract_mag(back_fft_1, b3_hz))
    ];
    back_2_feature = [
        mean(extract_mag(back_fft_2, b1_hz)), ...
        mean(extract_mag(front_fft_2, b2_hz) - extract_mag(back_fft_2, b2_hz)), ...
        mean(extract_mag(front_fft_2, b3_hz) - extract_mag(back_fft_2, b3_hz))
    ];
end

function plot_dendrogram (metric, front_thes, back_thes)
    data = load("chp6/cache/extracted_hrtfs.mat");

    % data = load("chp6/cache/cipic_spectrum.mat");
    % figure;
    % subplot(2, 1, 1);
    % plot(data.front_extracted(5, :));
    % subplot(2, 1, 2);
    % plot(data.front_extracted(6, :));

    Z = linkage(data.front_extracted, "average", metric);
    % T = cluster(Z,MaxClust=10);
    % cutoff = median([Z(end-6,3) Z(end-5,3)]);
    figure;
    dendrogram(Z, length(data.front_extracted), ColorThreshold = front_thes);
    % dendrogram(Z, length(data.front_extracted));
    title("front direction");
    set(gca,'xtick',[])

    Z = linkage(data.back_extracted, "average", metric);
    % T = cluster(Z,MaxClust=10);
    % cutoff = median([Z(end-6,3) Z(end-5,3)]);
    figure;
    dendrogram(Z, length(data.back_extracted), ColorThreshold = back_thes);
    % dendrogram(Z, length(data.back_extracted));
    title("back direction");
    set(gca,'xtick',[])

    % hrtf = SOFAload('/home/nycheung/desktop/shuto_gassom/GASSOM_BLM/Dataset/cipic_sofa/subject_003.sofa');
    % ir = hrtf.Data.IR(609, 1, :);
    % IR=squeeze(hrtf.Data.IR(609,1,:));
    % M=20*log10(abs(fft(IR')));
    % M=M(:,1:floor(size(M,2)/2)); 
    % y = plotDFT(ir, false);
    % plot(squeeze(f), squeeze(y));
    % y = fft(ir, 400);
end

% check_cipic_position();
% collect_cipic_spectrum();
% extract_critical_bands();
% plot_dendrogram("seuclidean", 0, 0);
% plot_dendrogram("cityblock", 34.5, 8.075);
% plot_dendrogram("chebychev", 0, 0);
plot_dendrogram("cosine", 0.1, 0.1);
