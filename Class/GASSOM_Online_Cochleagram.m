classdef GASSOM_Online_Cochleagram < handle
    % GASSOM_ONLINE This is the GASSOM oniline algorithm with winner selectionwhich assumes
    % slowness in the data.  Please refer the example for usage.  
    % specific queries, tnc<at>connect<dot>ust<dot>hk
    
    %This could be run online, (i.e, X is one d-dimension column vector compricing the input, or things could be made faster by combining 
    %the inputs into mini batches of ~10 samples, X~[dxN])
    
    properties
        dim_patch_single;
        dim_patch;
        topo_subspace;
        max_iter;
        bi_flag;
        
        length_basis;
        n_subspace;
        n_basis;
        size_subspace;
        
        alpha_A;alpha_C;
        sigma_A;sigma_C;
        sigmaTrans;
        alphaTrans;
        bases;
        transProb;
        nodeProb;
        winCoef;
        winError;
        Proj;
        resi;
        coef;
        iter;
        sigma_n;
        sigma_w;
        updatecount;
        winners;
        tconst;
        tconst_n;
        winnerTrack;
        h_plot;
        h_fig;
    end
    
    methods
        function obj = GASSOM_Online_Cochleagram(PARAM)
            obj.dim_patch_single = PARAM{1}; % [dps 2] for Stereo
            obj.topo_subspace = PARAM{2};
            obj.max_iter = PARAM{3};
            
            %default
            obj.dim_patch = [obj.dim_patch_single 2];
            obj.n_subspace = prod(obj.topo_subspace);       
            obj.length_basis = prod(obj.dim_patch);           
            % obj.size_subspace = 2;
            obj.size_subspace = 1;
            obj.n_basis = obj.size_subspace * obj.n_subspace;                
            
            obj.sigmaTrans = 2.25;
            obj.alphaTrans = 0.4; % 0 -> no uniform
            obj.updatecount = 1;
            obj.sigma_n = 0.2;
            obj.sigma_w = 2;
            
            % obj.alpha_A = 10; % magnitude
            % obj.alpha_C = 1e-3;
            obj.alpha_A = 8e-4; % magnitude
            obj.alpha_C = 1e-5;

            % obj.sigma_A = 2;
            % obj.sigma_C = .1;
            % obj.sigma_A = 2;
            obj.sigma_A = 2;
            obj.sigma_C = .2;

            % obj.tconst = 10000;
            obj.tconst = 6250;
            % obj.tconst = 40000;
            
            obj.transProb =  genTransProbG(obj.topo_subspace,obj.sigmaTrans, obj.alphaTrans,0); 
            np = rand(obj.n_subspace,1);    
            obj.nodeProb = bsxfun(@rdivide,np,sum(np));

            %random initial bases                       
            % save_name = sprintf('temp_data/InitBases_%d_%d_B.mat',obj.length_basis,obj.n_subspace);
            % if ~exist(save_name,'file')
            A = randn(obj.length_basis, obj.size_subspace, obj.n_subspace);
            A = orthonormalize_subspace (A);
            obj.bases{1}= squeeze(A(:,1,:));
            % obj.bases{2}= squeeze(A(:,2,:));

            %     save(save_name,'A','np');
            % else
            %     load(save_name,'A','np');
            %     obj.bases{1}= squeeze(A(:,1,:)); obj.bases{2}= squeeze(A(:,2,:));
            %     obj.nodeProb = bsxfun(@rdivide,np,sum(np));    
            % end           
            obj.iter = 1;
            obj.updatecount = 0;
            %init visualization
            obj.h_plot = cell(1,2);  
            obj.h_fig = cell(1,2);
        end
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            Encode
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        function [winners] = assomEncode(this,X)          
            batch_size = size(X,2);

            this.coef{1} = this.bases{1}'*X; %[n_subspace batch_size]
            % this.coef{2} = this.bases{2}'*X;

            % this.Proj = this.coef{1}.^2 + this.coef{2}.^2; %P[n_subspace,batch_size]
            this.Proj = this.coef{1}.^2;
            this.Proj = this.Proj./max(this.Proj);

            % disp(max(this.Proj, [], "all"));
            assert(max(this.Proj, [], "all") <= 1);
            
            Perr = ones(size(this.Proj))-this.Proj;
            emissProb=exp(-this.Proj/(2*this.sigma_w^2)).*exp(-Perr/(2*this.sigma_n^2));      

            nodeprobTmp =zeros(this.n_subspace,batch_size);
            for i=1:batch_size
                nodeprobTmp(:,i) = (this.transProb'*this.nodeProb).* emissProb(:,i);
                this.nodeProb =  nodeprobTmp(:,i)./sum(nodeprobTmp(:,i));
            end       
            [~,this.winners] = max(nodeprobTmp);

            winners = this.winners;
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            get response
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
        function [resp] = getResponse(this,X,ind)
            if nargin<3
                coef1 = this.bases{1}'*X;
                % coef2 = this.bases{2}'*X;
                % resp = coef1.^2 + coef2.^2;
                resp = coef1.^2;
            else
                coef1 = this.bases{1}(:,ind)'*X;
                % coef2 = this.bases{2}(:,ind)'*X;
                % resp = coef1.^2 + coef2.^2;
                resp = coef1.^2;
            end
        end

        function [basis] = getBasis(this,ind,i_subspace)
            if nargin<3, i_subspace = 1; end
            basis_ind = this.bases{i_subspace}(:,ind);
            basis = reshape(basis_ind,[],2);
        end
        
        function [patch_L, patch_R] = getPatch(this,ind)
            basis_ind = reshape(this.bases{1}(:,ind),[],2);
            patch_L = reshape(basis_ind(:,1),this.dim_patch_single);
            patch_R = reshape(basis_ind(:,2),this.dim_patch_single);
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            updateBasis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%           
        
        function updateBasis(this,X)          
            alpha = (this.alpha_A*exp(-this.iter/this.tconst)+this.alpha_C);
            sigma_h = (this.sigma_A*exp(-this.iter/this.tconst)+this.sigma_C);           
           
            batch_size = size(X,2);
            [cj,ci] = ind2sub(this.topo_subspace,this.winners);   
            k = 1:this.n_subspace;
            [kj,ki] = ind2sub(this.topo_subspace,k);

            kj = repmat(kj',[1,batch_size] );
            ki = repmat(ki',[1,batch_size] );

            cj = repmat(cj, [this.n_subspace,1]);
            ci = repmat(ci, [this.n_subspace,1]);

            func_h = exp((-(ki-ci).^2-(kj-cj).^2)/(2*(sigma_h)^2)); %gaussian [n_subspace,batchsize]
            
            n_const = 1./(sqrt(this.Proj)+eps);
            weights = func_h.*n_const;
            w_c{1} =weights.*this.coef{1};
            % w_c{2} =weights.*this.coef{2};
            
            winput{1} = X*w_c{1}';
            % winput{2} = X*w_c{2}';
            
            % diff{1} =  winput{1}-bsxfun(@times,this.bases{1},sum(w_c{1}.*this.coef{1},2)')-bsxfun(@times,this.bases{2},sum(w_c{1}.*this.coef{2},2)');
            % diff{2} =  winput{2}-bsxfun(@times,this.bases{1},sum(w_c{2}.*this.coef{1},2)')-bsxfun(@times,this.bases{2},sum(w_c{2}.*this.coef{2},2)');
            diff{1} =  winput{1}-bsxfun(@times,this.bases{1},sum(w_c{1}.*this.coef{1},2)');
           
            Bases{1} = this.bases{1} +alpha*diff{1};
            % Bases{2} = this.bases{2} +alpha*diff{2};

            this.bases{1} = bsxfun(@rdivide, Bases{1}, sqrt(sum(Bases{1}.^2)));
            % Bases{2} = Bases{2} - bsxfun(@times,this.bases{1}, sum(this.bases{1}.*Bases{2}));
            % this.bases{2} = bsxfun(@rdivide, Bases{2}, sqrt(sum(Bases{2}.^2)));            
            
            this.iter = this.iter+1; 
            this.updatecount = this.updatecount+1;
        end
        
        function visualizeBases(this,iBase)
            if nargin<2
                for h = 1:this.size_subspace
                    [this.h_plot{h},this.h_fig{h}] = plotBases(this.bases{h},this.dim_patch,this.topo_subspace,false,this.h_plot{h});
                end      
            else
                [this.h_plot{iBase},this.h_fig{iBase}] = plotBases(this.bases{iBase},this.dim_patch,this.topo_subspace,false,this.h_plot{iBase});
            end            
        end
        
        function visualCoch(this) 
            this.h_plot = cell(1,2);
            plotBases(this.bases{1},this.dim_patch,this.topo_subspace,true,this.h_plot{1});
        end        
    end
    
end

