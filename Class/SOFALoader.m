classdef SOFALoader < handle
    properties
        cipic;
        cipic_subject_ind;
        cipic_subject_num;
        cipic_azimuths;
        cipic_elevations;
        
        listen;
        listen_subject_ind;
        listen_subject_num;
        
        kemar;
        root_path;
        
        cur_set;
        cur_sub;
        SOFA_API_path;
        
        int_al; % interpolation algorithm
    end
    
    methods(Static)
        function this = SOFALoader(load_now)
            this.root_path = 'Dataset';
            % this.SOFA_API_path = '../Tools/API_MO-master';
            % addpath(genpath(this.SOFA_API_path));
            this.cur_set = 'cipic';
            this.cur_sub = 3;
            
            this.cipic_subject_ind = [3 8 9 10 11 12 15 17 18 19 20 21 27 28 33 40 44 48 50 51 58 59 ...
                60 61 65 119 124 126 127 131 133 134 135 137 147 148 152 153 154 155 ...
                156 158 162 163 165];
            % this.cipic_subject_ind = [3 8 9 10 12];
            this.cipic_subject_num = numel(this.cipic_subject_ind);            
            this.cipic_azimuths   = [-80 -65 -55 -45:5:45 55 65 80];
            this.cipic_elevations = -45 + 5.625*(0:49);            
                        
            this.listen_subject_ind = [1002 1003 1004 1005 1006 1007 1008 1009 1012 1013 1014 1015 ...
                1016 1017 1018 1020 1021 1022 1023 1025 1026 1028 1029 1030 1031 1032 ...
                1033 1034 1037 1038 1039 1040 1041 1042 1043 1044 1045 1046 1047 1048 ...
                1049 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059];
            this.listen_subject_num = numel(this.listen_subject_ind);
            
            if nargin<1
                this.load_Raw_HRTF;
            end
            this.int_al = 'bilinear';
        end
    end
    
    methods
        
        function load_Raw_HRTF(this)
            % Load CIPIC
            SOFAstart('silent');
            this.cipic = cell(this.cipic_subject_num,1);
            for i = 1:this.cipic_subject_num
                full_path = fullfile(this.root_path, 'cipic_sofa',sprintf('subject_%03d.sofa',this.cipic_subject_ind(i)));
                this.cipic{i,1} = SOFAload(full_path);
            end                
            
            % % Load LISTEN
            % this.listen = cell(this.listen_subject_num,1);
            % for i = 1:this.listen_subject_num
            %     full_path = fullfile(this.root_path,'listen_sofa',sprintf('irc_%4d.sofa',this.listen_subject_ind(i)));
            %     this.listen{i,1} = SOFAload(full_path);
            % end
            
            % Load MIT KEMAR (normal pinna)
            full_path = fullfile(this.root_path,'kemar_sofa','mit_kemar_normal_pinna.sofa');
            this.kemar{1,1} = SOFAload(full_path);
        end
        
        function [hL,hR] = read_HRIR(this,azim,elev,dataset,subject)
            switch dataset
                case 'cipic'
                    sub_ind = find(this.cipic_subject_ind == subject);
                    [loc_ind,~,~,~] = SOFAfind(this.cipic{sub_ind,1},azim,elev);
                    hL = squeeze(this.cipic{sub_ind,1}.Data.IR(loc_ind,1,:));
                    hR = squeeze(this.cipic{sub_ind,1}.Data.IR(loc_ind,2,:));
                case 'listen'
                    sub_ind = find(this.listen_subject_ind == subject);
                    [loc_ind,~,~,~] = SOFAfind(this.listen,azim,elev);
                    hL = squeeze(this.listen{sub_ind,1}.Data.IR(loc_ind,1,:));
                    hR = squeeze(this.listen{sub_ind,1}.Data.IR(loc_ind,2,:));
                case 'kemar'
                    [loc_ind,~,~,~] = SOFAfind(this.kemar{1,1},azim,elev);
                    hL = squeeze(this.kemar{1,1}.Data.IR(loc_ind,1,:));
                    hR = squeeze(this.kemar{1,1}.Data.IR(loc_ind,2,:));
            end
        end    
        
        function [ind] = find_with_SOFA(this,azim,elev)
            [ind,~,~,~] = SOFAfind(this.cipic{1,1},azim,elev);
        end
              
        function [out] = spatMono(this,y,loc,dataset,subject)
            % one hrir mean diotic binaural signals
            if size(y,1) == 1, y = y'; end
            azim = loc(1); elev = loc(2);
            switch dataset
                case 'cipic'
                    sub_ind = this.cipic_subject_ind == subject;
                    out = SOFAspat(y,this.cipic{sub_ind,1},azim,elev);
                case 'listen'
                    sub_ind = this.listen_subject_ind == subject;
                    out = SOFAspat(y,this.listen{sub_ind,1},azim,elev);
                case 'kemar'
                    out = SOFAspat(y,this.kemar{1,1},azim,elev);
                otherwise
                    assert(false, "unexpected dataset " + dataset);
            end
            out = out(1:length(y),:);
        end
        
        function [out] = filterMono(this,y,loc,dataset,subject)
            if size(y,1) == 1, y = y'; end
%             [hL,hR] = this.read_HRIR(loc(1),loc(2),dataset,subject);
            [~,hL,hR] = this.intHRTF(loc,dataset,subject);
            yL = fftfilt(hL,y);
            yR = fftfilt(hR,y);
            out = [yL,yR];
        end
        
        function [ind] = findInd(this,targetPosition,dataset)
            targetPosition(1) = mod(targetPosition(1),360);
            switch dataset
                case 'cipic'
                    ind = find(abs(this.cipic{1,1}.SourcePosition(:,1)-targetPosition(1))<0.001 ...
                        & abs(this.cipic{1,1}.SourcePosition(:,2)-targetPosition(2))<0.001);
                case 'kemar'
                    ind = find(abs(this.kemar{1,1}.SourcePosition(:,1)-targetPosition(1))<0.001 ...
                        & abs(this.kemar{1,1}.SourcePosition(:,2)-targetPosition(2))<0.001);
            end
        end
        
        function [ih,hl,hr] = intHRTF(this,targetPosition,dataset,subject)
            if nargin<3, dataset = 'kemar'; subject = 0; end
            ind = this.findInd(targetPosition,dataset);
            if isempty(ind)
                if size(targetPosition,2) == 1, targetPosition = targetPosition'; end
                switch dataset
                    case 'cipic'
                        sub_ind = find(this.cipic_subject_ind == subject);
                        ih = interpolateHRTF(this.cipic{sub_ind,1}.Data.IR,this.cipic{sub_ind,1}.SourcePosition(:,1:2),...
                            targetPosition,'Algorithm',this.int_al);
                    case 'kemar'
                        ih = interpolateHRTF(this.kemar{1,1}.Data.IR, this.kemar{1,1}.SourcePosition(:,1:2),...
                            targetPosition,'Algorithm',this.int_al);
                end
                hl = squeeze(ih(:,1,:));
                hr = squeeze(ih(:,2,:));
            else
                switch dataset
                    case 'cipic'
                        sub_ind = find(this.cipic_subject_ind == subject);
                        hl = squeeze(this.cipic{sub_ind,1}.Data.IR(ind,1,:));
                        hr = squeeze(this.cipic{sub_ind,1}.Data.IR(ind,2,:));
                    case 'kemar'
                        hl = squeeze(this.kemar{1,1}.Data.IR(ind,1,:));
                        hr = squeeze(this.kemar{1,1}.Data.IR(ind,2,:));
                end
                ih = [hl, hr];
            end
        end
        
    end
end