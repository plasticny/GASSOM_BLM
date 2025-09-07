classdef stimuliGenerator < handle
    properties
        fs;
        gwn_seed;
        rand_gen_algorithm;
        rand_gwn_stream;
        
        fm_rand_stream;
        fm_length_seed;
        fm_param;
    end
    
    methods
        function this = stimuliGenerator
            this.fs = 44100;
            this.rand_gen_algorithm = 'twister';
            
            this.gwn_seed = 1;            
            this.rand_gwn_stream = RandStream(this.rand_gen_algorithm,'Seed',this.gwn_seed);
            this.reset_gwn;
            
            this.fm_length_seed = 10001;
            this.fm_rand_stream = RandStream(this.rand_gen_algorithm,'Seed',this.fm_length_seed);
            this.reset_fm;
            this.fm_param.silent_size = 100;
            this.fm_param.start_f = 100;
            this.fm_param.end_f = 8000;
            this.fm_param.method = 'linear';
        end
        
        %%% Reset Gaussian white noise random seed
        function reset_gwn(this,seed)
            if nargin<2
                reset(this.rand_gwn_stream, this.gwn_seed);
            else
                this.gwn_seed = seed;
                reset(this.rand_gwn_stream,this.gwn_seed);
            end
        end
        
        function [sti] = genGWN(this,sti_dur)
            sti = randn(this.rand_gwn_stream,[round(sti_dur*this.fs),1]);            
%             rampDuration = min(20e-3,sti_dur/10); 
%             rampTime = 1/this.fs:1/this.fs:rampDuration;
%             ramp = [0.5*(1+cos(2*pi*rampTime/(2*rampDuration)+pi)) ones(1,length(sti)-length(rampTime))];            
%             sti = sti.*ramp'; sti = sti.*flipud(ramp');
            
        end
        
        %%% Sweep sinewave
        function reset_fm(this,seed)
            if nargin>1, this.fm_length_seed = seed; end
            reset(this.fm_rand_stream,this.fm_length_seed);
        end
        
        function [sti] = genSweep(this,sti_dur)
            if sti_dur<0
                rand_part = rand(this.fm_rand_stream);
                sti_dur = (50+50*rand_part)*1e-3;
            end            
            n_sample = round(this.fs*sti_dur);
            t = (0:n_sample-1)/this.fs;
            sti = chirp(t,this.fm_param.start_f,t(end),this.fm_param.end_f,this.fm_param.method);
            
            % Add ramp
            rampDuration = min(20e-3,sti_dur/10); 
            rampTime = 1/this.fs:1/this.fs:rampDuration;
            ramp = [0.5*(1+cos(2*pi*rampTime/(2*rampDuration)+pi)) ones(1,length(sti)-length(rampTime))];            
            sti = sti.*ramp; sti = sti.*fliplr(ramp);
            sti = sti';
        end        
    end
end
