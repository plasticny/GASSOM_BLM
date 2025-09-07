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
end

options = trainingOptions('sgdm','InitialLearnRate',0.1,'LearnRateDropFactor',0.1,'ExecutionEnvironment','cpu',...
    'Shuffle','once','MaxEpochs',10);
end        