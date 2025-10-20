classdef gassomMapVideoMaker < handle
    properties
        videoPath;
        frameRate;
        shotImgPath;

        videoWriters;

        chunk_size;

        inputType;
    end

    methods
        % chunk size is not needed if input type is waveform
        function this = gassomMapVideoMaker(video_path, inputType, chunk_size)
            assert(inputType == "waveform" || inputType == "cochleagram");
            assert(inputType == "waveform" || nargin == 3);
            this.inputType = inputType;

            this.videoPath = video_path;
            this.frameRate = 1;

            this.shotImgPath = "temp_data/tmp_shot.png";

            this.videoWriters = cell(2,1);
            this.videoWriters{1} = VideoWriter(this.videoPath + "_1");
            this.videoWriters{1}.FrameRate = this.frameRate;
            if this.inputType == "waveform"
                this.videoWriters{2} = VideoWriter(this.videoPath + "_2");
                this.videoWriters{2}.FrameRate = this.frameRate;
            end

            if this.inputType == "cochleagram"
                this.chunk_size = chunk_size;
            end
        end

        function open (this)
            open(this.videoWriters{1});
            if this.inputType == "waveform"
                open(this.videoWriters{2})
            end
        end

        function close (this)
            close(this.videoWriters{1});
            if this.inputType == "waveform"
                close(this.videoWriters{2});
            end
        end

        % param gm: an instance of GASSOM_Model
        function addGassomMapFrame (this, gm)
            if this.inputType == "cochleagram"
                this.saveShotImageCochl(gm);
                im = imread(this.shotImgPath);
                frame = im2frame(im);
                writeVideo(this.videoWriters{1}, frame)
            else
                for i=1:1:2
                    this.saveShotImageWaveform(gm, i)
                    im = imread(this.shotImgPath);
                    frame = im2frame(im);
                    writeVideo(this.videoWriters{i}, frame);
                end
            end
        end

        function saveShotImageCochl (this, gm)
            figure('visible','off');
            r = gm.topo_space(1);
            c = gm.topo_space(2);
            for i=1:r*c
                subplot(r,c,i);
                imagesc(reshape(gm.gsm{1}.bases{1}(:,i), [], this.chunk_size * 2));
                colormap('jet');
                axis off;
                set(gca, 'YDir', 'normal');
            end
            saveas(gcf, this.shotImgPath);
            close;
        end

        function saveShotImageWaveform (this, gm, base_number)
            figure('visible', 'off');
            r = gm.topo_space(1);
            c = gm.topo_space(2);
            for i = 1:r*c
                subplot(r,c,i);
                b = gm.gsm{1}.bases{base_number}(:,i);
                l = size(b, 1) / 2;
                plot(1:1:l, b(1:l,:));
                hold on;
                plot(l+1:1:l*2, b(l+1:end,:), "Color", "r");
            end
            saveas(gcf, this.shotImgPath);
            close;
        end
    end
end