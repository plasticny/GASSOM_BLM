function [h,h_fig] = plotBases(A, dim_patch, dim_topo, one_plot, h)

%%
% 
%  A        : the basis matrix([basis_length, number_of_basis])
%  dim_patch_single: Patch dimensions([nBin, length], nBin=1 for wave)
%  dim_topo : subspace topology([nrows, ncols])
%  bi_flag  : 1 -> binaural basis; 0 -> mono basis;
% 
patch_dist = 3;
n_row = dim_topo(1);
n_col = dim_topo(2);
dim_patch_single = dim_patch(1:2);
if isequal(dim_patch(3),1)
    bi_flag = false;
else
    bi_flag = true;
end

mycolormap = 'jet';
% if exist('mycm2.mat','file')
%     load('mycm2.mat','mycolormap');
% else
%     mycolormap = 'jet';
% end
%% normalize each patch
A = A./(ones(size(A,1),1)*max(abs(A)));
% A = (A-ones(size(A,1),1)*min(A)) ./ (ones(size(A,1),1)*(max(A)-min(A)));
max_abs_A = max(abs(A(:)));

%% plot
if bi_flag
    mon_len = round(size(A,1)/2);
    A1 = A(1:mon_len,:);
    A2 = A(mon_len+1:end,:);
%     if ~exist('figure_no','var')
%         h_fig = figure;
%     else
%         h_fig = figure(figure_no);
%     end
    h_fig = gcf;
    h = cell(1,prod(dim_topo));
    if dim_patch_single(1) == 1
        for idx = 1:prod(dim_topo)
            subplot(n_row,n_col, idx);
            if isempty(h{idx})
                h{idx} = plot(1:mon_len, [A1(:,idx),A2(:,idx)]);
            else
                set(h{idx},'YData',[A1(:,idx),A2(:,idx)]);
            end
            set(gca,'xTick',[],'YTick',[],'XLim',[0 mon_len],'YLim',[-max_abs_A max_abs_A]);
%             title(int2str(idx));
        end
    else
        if one_plot
            dim_patch_single1 =[dim_patch_single(1)+patch_dist*2 dim_patch_single(2)*2+patch_dist*3];
            I = ones(dim_patch_single1(1)*n_row-patch_dist*2, dim_patch_single1(2)*n_col-patch_dist*2);
            for idx = 1:prod(dim_topo)
                A_patch = flipud([reshape(A1(:,idx),dim_patch_single),ones(dim_patch_single(1),patch_dist),...
                    reshape(A2(:,idx),dim_patch_single)]);
                [i_col,i_row] = ind2sub([n_col,n_row],idx);
                roi_y = (i_row-1)*dim_patch_single1(1)+(1:dim_patch_single(1));
                roi_x = (i_col-1)*dim_patch_single1(2)+(1:(2*dim_patch_single(2)+patch_dist));
                I(roi_y,roi_x) = A_patch;
            end
%             if h==0
                h = imagesc(I,[-max_abs_A, max_abs_A]);
                axis equal tight off;
                colormap(mycolormap);
%             else
%                 set(h,'CData',I);
%                 set(gca,'CLim',[-max_abs_A max_abs_A]);
%                 colormap(mycolormap);
%             end
        else        
            for idx = 1:prod(dim_topo)
                A1_patch = reshape(A1(:,idx),dim_patch_single);
                A2_patch = reshape(A2(:,idx),dim_patch_single);
                A_patch = [A1_patch,A2_patch];
                subplot(n_row,n_col, idx);
                if isempty(h{idx})
                    h{idx} = imagesc(1:dim_patch_single(2), 1:dim_patch_single(1), A_patch);
                else
                    set(h{idx},'CData',A_patch);
                end
                set(gca,'CLim',[-max_abs_A max_abs_A]);
                axis equal tight off;
                colormap(mycolormap);
            end
        end
    end
    drawnow;
else
    mon_len = prod(dim_patch_single);
%     if ~exist('figure_no','var')
%         h_fig = figure;
%     else
%         h_fig = figure(figure_no);
%     end
    h_fig = gcf;
    if dim_patch_single(1) == 1
        for idx = 1:prod(dim_topo)
            subplot(n_row,n_col,idx);            
            if isempty(h{idx})
                h{idx} = plot(1:mon_len, [A(:,idx)]);
            else
                set(h{idx},'YData',[A(:,idx)]);
            end
            set(gca,'xTick',[],'YTick',[],'XLim',[0 mon_len],'YLim',[-max_abs_A max_abs_A]);
            title(int2str(idx));
        end
    else
        if one_plot
            patch_dist = 3;
            dim_patch_single1 =dim_patch_single+patch_dist;
            I = ones(dim_patch_single1(1)*n_row-patch_dist, dim_patch_single1(2)*n_col-patch_dist);
            for idx = 1:prod(dim_topo)
                A_patch = flipud(reshape(A(:,idx),dim_patch_single));
                [i_col,i_row] = ind2sub([n_col,n_row],idx);
                roi_y = (i_row-1)*dim_patch_single1(1)+(1:dim_patch_single(1));
                roi_x = (i_col-1)*dim_patch_single1(2)+(1:dim_patch_single(2));
                I(roi_y,roi_x) = A_patch;
            end
            if h==0
                h = imagesc(I,[-max_abs_A, max_abs_A]);
                axis equal tight off;
                colormap(mycolormap);
            else
                set(h,'CData',I);
                set(gca,'CLim',[-max_abs_A max_abs_A]);
                colormap(mycolormap);
            end
        else        
            for idx = 1:prod(dim_topo)
                A_patch = reshape(A(:,idx),dim_patch_single);
                subplot(n_row,n_col, idx);
                if isempty(h{idx})
                    h{idx} = imagesc(1:dim_patch_single(2), 1:dim_patch_single(1), A_patch);
                else
                    set(h{idx},'CData',A_patch);
                end
                set(gca,'CLim',[-max_abs_A max_abs_A]);
                axis equal tight off xy;
                colormap(mycolormap);
            end     
        end
    end
    drawnow;
end 
end


