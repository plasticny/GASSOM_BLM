% compute Front-back Confusion Score
function [FBC_score,FBC_rate] = calculate_FBC(locs_true,locs_pred)
locs_fb = (locs_pred(2,:)~=locs_true(2,:));
N_hat = size(locs_true,2);
N     = sum(locs_fb);
locs_true_fb = locs_true(1,locs_fb);
locs_pred_fb = locs_pred(1,locs_fb);
locs_pred_l  = (locs_true_fb<locs_pred_fb);
locs_pred_g  = not(locs_pred_l);
if ~isequal(sum(locs_pred_l),0)
    theta_l_max  = 90-locs_true_fb(locs_pred_l);
    theta_l      = locs_pred_fb(locs_pred_l)-locs_true_fb(locs_pred_l);
    w_l = 1-theta_l./theta_l_max;
else
    w_l = 0;
end

if ~isequal(sum(locs_pred_g),0)
    theta_g_max  = locs_true_fb(locs_pred_g)+90;
    theta_g      = locs_true_fb(locs_pred_g)-locs_pred_fb(locs_pred_g);
    w_g = 1-theta_g./theta_g_max;
else
    w_g = 0;
end
FBC_rate  = N/N_hat;
FBC_score = (sum(w_l)+sum(w_g))/N_hat;
end