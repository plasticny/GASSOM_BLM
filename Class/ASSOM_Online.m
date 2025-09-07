classdef ASSOM_Online < handle
    %code which runs N simultanious sthreads of with shared bases
    properties

        %user defined
        dim_patch_single;
        topo_subspace;
        max_iter;
        
        length_basis;
        dim_patch;
        n_subspace;
        segment_length;
        n_basis;
        size_subspace;
        alpha_A;alpha_C;
        sigma_A;sigma_B;sigma_C;
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
        updatecount;
        winners;
        tconst;
        sigma_n;
        sigma_w;
        win_energy;
        
        h_plot;
        h_fig;
    end
    methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Function discription:
%%%     Constructor of ASSOMOnline class
%%% Parameter discription:
%%%     PARAM = {dim_patch_single,topo_subspace,max_iter};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function obj = ASSOM_Online(PARAM)
            obj.dim_patch_single = PARAM{1};
            obj.topo_subspace = PARAM{2};
            obj.max_iter = PARAM{3};
            obj.sigma_n = 0.2;
            obj.sigma_w = 2;
            
            obj.n_subspace = prod(obj.topo_subspace);
            obj.segment_length = prod(obj.dim_patch_single);
            obj.dim_patch = [obj.dim_patch_single 2];
            
            obj.length_basis = prod(obj.dim_patch);
            obj.size_subspace = 2;
            obj.n_basis = obj.size_subspace * obj.n_subspace;
            
            obj.alpha_A = 1;
            obj.alpha_C = 1e-3;
            obj.sigma_A = 2;
            obj.sigma_C = .1;
            obj.tconst = 8000;
            
%             obj.alpha_A = 8e-4;
%             obj.alpha_C = 1e-5;
%             obj.tconst = 10000;
%             obj.sigma_A = 2;
%             obj.sigma_B = 1000;
%             obj.sigma_C = 0.2;
            obj.sigmaTrans = 0.5;            
            obj.alphaTrans = 0.5;
            obj.updatecount = 1;
            obj.win_energy=[];
            
            %initialize
            %random initial bases
            A =randn(obj.length_basis, obj.size_subspace, obj.n_subspace);
            A = orthonormalize_subspace (A);
            obj.bases{1}= (squeeze(A(:,1,:))); obj.bases{2}= (squeeze(A(:,2,:)));
            obj.transProb =  (genTransProbG(obj.topo_subspace,obj.sigmaTrans, obj.alphaTrans));
            np = rand(obj.n_subspace,1);          
            obj.nodeProb = bsxfun(@rdivide,np,sum(np));
            obj.iter=1;
            
            obj.h_plot = cell(1,2);
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Function discription:
%%%     encode the input images with the best matched subspace
%%% Parameter discription:
%%%     imageBatch is the input images batch
%%% Returns
%%%     coef is the output coefficient matrix
%%%     error is the reconstructin error using current coefficients
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [winner] = assomEncode(this,X)
            
            [~ ,batch_size] = size(X);     
            this.winCoef = zeros(this.n_subspace,batch_size);

            this.coef{1} = this.bases{1}'*X; 
            this.coef{2} = this.bases{2}'*X;

            this.Proj  = this.coef{1}.^2 + this.coef{2}.^2; 

            [~,local_winners] = max(sum(this.Proj,2));

            winner = (local_winners);
            this.winners = repmat(local_winners,1,batch_size);
        end
        
        function [resp] = getResponse(this,X,ind)
            if nargin<3
                coef1 = this.bases{1}'*X;
                coef2 = this.bases{2}'*X;
                resp = coef1.^2 + coef2.^2;
            else
                coef1 = this.bases{1}(:,ind)'*X;
                coef2 = this.bases{2}(:,ind)'*X;
                resp = coef1.^2 + coef2.^2;
            end
        end
        
        function [basis] = getBasis(this,ind,i_subspace)
            if nargin<3, i_subspace = 1; end
            basis_ind = this.bases{i_subspace}(:,ind);
            basis = reshape(basis_ind,[],2);
        end
        
        function proj = getProj(this,X)
            coefs{1} = (this.bases{1})'*X; 
            coefs{2} = (this.bases{2})'*X;
            proj = coefs{1}.^2+coefs{2}.^2;
        end
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Function discription:
%%%     Update the bases
%%% Parameter discription:
%%%     coef is the coefficient
%%%     error is the reconstruction error
%%% Returns
%%%     coef is the output coefficient matrix
%%%     error is the reconstructin error using current coefficients
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function updateBasis(this,X)            
            
            alpha = (this.alpha_A*exp(-this.iter/this.tconst)+this.alpha_C);
            sigma_h = (this.sigma_A*exp(-this.iter/this.tconst)+this.sigma_C);
            [~, batch_size] = size(this.winners);
            [cj,ci] = ind2sub(this.topo_subspace,this.winners);   
            k = colon(1,this.n_subspace);
            [kj,ki] = ind2sub(this.topo_subspace,k);

            kj = repmat(kj',[1,batch_size] );
            ki = repmat(ki',[1,batch_size] );

            cj = repmat(cj, [this.n_subspace,1]);
            ci = repmat(ci, [this.n_subspace,1]);

            func_h = exp((-(ki-ci).^2-(kj-cj).^2)/(2*(sigma_h)^2));
            X_norm = repmat(sqrt(sum(X.^2)),[size(func_h,1) 1]);
            n_const = 1./((sqrt(this.Proj)+eps).*X_norm);
            weights = func_h.*n_const;
            w_c{1} =weights.*this.coef{1};
            w_c{2} =weights.*this.coef{2};
            winput{1} = X*w_c{1}';
            winput{2} = X*w_c{2}';
            diff{1} =  winput{1}-bsxfun(@times,this.bases{1},sum(w_c{1}.*this.coef{1},2)')-bsxfun(@times,this.bases{2},sum(w_c{1}.*this.coef{2},2)');
            diff{2} =  winput{2}-bsxfun(@times,this.bases{1},sum(w_c{2}.*this.coef{1},2)')-bsxfun(@times,this.bases{2},sum(w_c{2}.*this.coef{2},2)');
           
            Bases{1} = this.bases{1} +alpha*diff{1};
            Bases{2} = this.bases{2} +alpha*diff{2};	

            this.bases{1} = bsxfun(@rdivide, Bases{1}, sqrt(sum(Bases{1}.^2)));
            Bases{2} = Bases{2} - bsxfun(@times,this.bases{1}, sum(this.bases{1}.*Bases{2}));
            this.bases{2} = bsxfun(@rdivide, Bases{2}, sqrt(sum(Bases{2}.^2)));            
            
            this.iter = this.iter+1; 

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
        
        function setPARAM(obj,sigA,sigC,alphaA,alphaC,tconst)
            obj.sigma_A = sigA;
            obj.sigma_C = sigC;
            obj.alpha_A = alphaA;
            obj.alpha_C = alphaC;
            obj.tconst  = tconst;
        end
        
    end
end
