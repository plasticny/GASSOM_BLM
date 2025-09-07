function [ e_freq ] = est_freq_bf( model,sub )

Fs=model.fs;
Freqs=0.2:0.2:60;
Phi2=[0:pi/4:pi];
window=model.patch_len;
t=0:1/Fs:(window-1)/Fs;

Freq_response=zeros(length(Freqs),length(Phi2));

for p2=Phi2
    for fc=Freqs
        Left_signal=1*cos(2*pi*fc*100*t+p2);
        Right_signal=1*cos(2*pi*fc*100*t+p2);
%         norm_signal = normalize_data([Left_signal';Right_signal']);
        norm_signal = [Left_signal'; Right_signal'];
        [feature]=model.gsm{1}.getResponse(norm_signal,sub);
        Freq_response((fc==Freqs),(p2==Phi2))=feature;
    end
end
% normalize response
Response=Freq_response;
Norm_response=(Response)/(max(Response));

[~,indx]=max(Norm_response);
e_freq=Freqs(indx)*100;

end

