%y_all = load("gwn_sample.mat").y;
%gfb = gammatoneFilterBank([100 20000],128,44100);
%gmt = gfb(y_all);

%patchLength = floor(44100 * 8 / 1000);
%patchStride = floor(44100 * 4 / 1000);

%frm = [];
%for i=0:patchStride:length(y_all)-patchLength
    %p = pow2db(sum(gmt(i+(1:patchLength),:).^2));
    %frm = [frm;p];
%end
%frm = frm';

%imagesc(frm);

tt_range = range(timit_toolbox, "all");
ti_range = range(timit_iosr, "all");
t_mae = mean(abs(timit_toolbox - timit_iosr), "all");

gt_range = range(gwn_toolbox, "all");
gi_range = range(gwn_iosr, "all");
g_mae = mean(abs(gwn_toolbox - gwn_iosr), "all");

disp("timit");
disp(tt_range);
disp(ti_range);
disp(t_mae);

disp("gwn");
disp(gt_range);
disp(gi_range);
disp(g_mae);

save("toolbox_isor_compare_result.mat", "tt_range", "ti_range", "t_mae", "gt_range", "gi_range", "g_mae");

figure;
subplot(2,1,1);
imagesc(timit_toolbox);
set(gca, 'YDir', 'normal');
colormap('jet');
title("toolbox");
axis("off");
subplot(2,1,2);
imagesc(timit_iosr);
set(gca, 'YDir', 'normal');
colormap('jet')
title("open source");
axis("off");
saveas(gcf, "timit.png");

figure;
subplot(2,1,1);
imagesc(gwn_toolbox);
set(gca, 'YDir', 'normal');
colormap('jet');
title("toolbox");
axis("off");
subplot(2,1,2);
imagesc(gwn_iosr);
set(gca, 'YDir', 'normal');
colormap('jet');
title("open source");
axis("off");
saveas(gcf, "gwn.png");
