function [ amp,freq ] = get_fundamental_freq( X )
    Fs=16000;
    X_left = fft(X);     
    n = length(X);                         
    fshift = (-n/2:n/2-1)*(Fs/n);
    X_leftshift = fftshift(X_left);
%     subplot(211);
%     stem(fshift,abs(X_leftshift));
%     subplot(212);
%     stem(abs(X_leftshift));
%     pause;
    f=fshift((n/2+2):end);
    Xf=X_leftshift((n/2+2):end);
%     f=fshift(202:end);
%     Xf=X_leftshift(202:end);
    [~,index]=max(Xf);
    freq=f(index);
    amp=Xf(index);
end

