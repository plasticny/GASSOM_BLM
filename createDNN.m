function [net,options] = createDNN(input_size,output_size,net_no)
if nargin<3,net_no = 1; end
switch net_no
    case 1
        net = [featureInputLayer(input_size,'Normalization','zscore','Name','Input Layer');
            fullyConnectedLayer(200,'Name','fc1');
            dropoutLayer(0.2);
            reluLayer;    
            fullyConnectedLayer(output_size,'Name','fc end');
            softmaxLayer('Name','Softmax Layer');
            classificationLayer('Name','Output Layer');
            ];
    case 2
        net = [featureInputLayer(input_size,'Normalization','zscore','Name','Input Layer');
            fullyConnectedLayer(200,'Name','fc1');
            dropoutLayer(0.2);
            reluLayer;    
            fullyConnectedLayer(200,'Name','fc2');
            dropoutLayer(0.2);
            reluLayer;
            fullyConnectedLayer(200,'Name','fc3');
            dropoutLayer(0.2);
            reluLayer;
            fullyConnectedLayer(output_size,'Name','fc end');
            softmaxLayer('Name','Softmax Layer');
            classificationLayer('Name','Output Layer');
            ];
    case 3
        % 
        net = [
            featureInputLayer(input_size, 'Normalization','zscore', "Name", "Input Layer");
            fullyConnectedLayer(200, "Name", "fc1");
            reluLayer;
            dropoutLayer(0.2);
            fullyConnectedLayer(200, "Name", "fc2");
            reluLayer;
            dropoutLayer(0.2);
            fullyConnectedLayer(50, "Name", "fc3");
            reluLayer;
            dropoutLayer(0.2);
            fullyConnectedLayer(output_size, "Name", "fc ouput");
            softmaxLayer("Name", "Softmax layer");
            classificationLayer("Name", "output layer");
        ];
end

switch net_no
    case {1}
        % original 10 epochs
        options = trainingOptions('sgdm','InitialLearnRate',0.1,'LearnRateDropFactor',0.1,'ExecutionEnvironment','cpu',...
            'Shuffle','once','MaxEpochs',1);
    case {2, 3}
        % 40
        options = trainingOptions(...
            "sgdm",...
            "InitialLearnRate", 0.1,...
            "LearnRateSchedule", "piecewise", ...
            "LearnRateDropFactor", 0.1,...
            "LearnRateDropPeriod", 10,...
            "Shuffle", "every-epoch", ...
            "MaxEpochs", 20 ...
        );
        % "MiniBatchSize", 95, ...
end
end        