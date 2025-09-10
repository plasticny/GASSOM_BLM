function [frm, cfs] = audio2cochlIOSR (...
    x, ...
    fs, lower_freq, upper_freq, num_ch, ...
    frame_size, frame_shift...
)
    % frame_size & frame_shift: in ms
    cfs = makeErbCFs(lower_freq, upper_freq, num_ch);
    bm = gammatoneFast(x, cfs, fs);

    patchLength = floor(fs * frame_size / 1000);
    patchStride = floor(fs * frame_shift / 1000);

    frm = [];
    for i=0:patchStride:length(bm)-patchLength
        p = pow2db(sum(bm(i+(1:patchLength),:).^2));
        frm = [frm;p];
    end
    frm = frm';
end