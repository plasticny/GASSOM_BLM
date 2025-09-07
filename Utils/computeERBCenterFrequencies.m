function [cfs] = computeERBCenterFrequencies(lf,hf,nf)
lfe = hz2erb(lf);
hfe = hz2erb(hf);
cfse = linspace(lfe,hfe,nf);
cfs = erb2hz(cfse);
end