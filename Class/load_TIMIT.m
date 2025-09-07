classdef load_TIMIT
    % Load timit dataset and convert to cell array
    %   default sampling frequency 44100Hz
    
    properties
        fs;
        
        train_path;
        test_path;
        train_list;
        train_num;
        test_list;
        test_num;
    end
    
    methods
        function obj = load_TIMIT(inputArg)
            if nargin>0
                obj.train_path = inputArg{1};
                obj.test_path = inputArg{2};
            else
                obj.train_path = "/Users/cst/Documents/Research/GASSOM/GASSOM_SSL-master/data/lisa/data/timit/raw/TIMIT/TRAIN";
                obj.test_path  = "/Users/cst/Documents/Research/GASSOM/GASSOM_SSL-master/data/lisa/data/timit/raw/TIMIT/TEST";
            end
            obj.fs = 44100;
            obj.train_list = listFiles(obj.train_path,'*.wav');
            obj.test_list  = listFiles(obj.test_path,'*.wav');
            obj.train_num  = numel(obj.train_list);
            obj.test_num   = numel(obj.test_list);
            obj.wav2cell();
        end
        
        function wav2cell(obj)
            % Convert wav to cell array
            timit_train = cell(obj.train_num,1);
            timit_test = cell(obj.test_num,1);
            for i_train = 1:obj.train_num
                [y,sr] = audioread(obj.train_list(i_train).name);
                if ~isequal(sr,obj.fs), y = resample(y,obj.fs,sr); end
                timit_train{i_train,1} = y;
                timit_train{i_train,2} = length(y);
            end
            for i_test = 1:obj.test_num
                [y,sr] = audioread(obj.test_list(i_test).name);
                if ~isequal(sr,obj.fs), y = resample(y,obj.fs,sr); end
                timit_test{i_test,1} = y;
                timit_test{i_test,2} = length(y);
            end
            if ~exist('temp_data','dir'), mkdir('temp_dat'); end
            save('temp_data/timit_array.mat','timit_train','timit_test','-v7.3');
        end
    end
end

