classdef gassomMapGifMaker < handle
    properties
        gifPath;
        firstFrameWritten;
    end

    methods
        function this = gassomMapGifMaker(PARAM)
            this.gifPath = PARAM{1};

            this.firstFrameWritten = false;
        end

        function addGassomMapFrame (gm)
            % param gm: an instance of GASSOM_Model
            if ~this.firstFrameWritten
                imwrite(im, this.gifPath, 'gif', 'LoopCount', Inf, 'DelayTime', 0.0625);
            else
                imwrite(im, this.gifPath, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
            end
        end
    end
end