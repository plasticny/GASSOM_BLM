classdef gassomMapVideoMaker < handle
    properties
        videoPath;
        frameRate;
        shotImgPath;

        videoWriter;

        chunk_size;
    end

    methods
        function this = gassomMapVideoMaker(video_path, chunk_size)
            this.videoPath = video_path;
            this.frameRate = 1;

            this.shotImgPath = "temp_data/tmp_shot.png";

            this.videoWriter = VideoWriter(this.videoPath);
            this.videoWriter.FrameRate = this.frameRate;            

            this.chunk_size = chunk_size;
        end

        function open (this)
            open(this.videoWriter);
        end

        function close (this)
            close(this.videoWriter);
        end

        function addGassomMapFrame (this, gm)
            % param gm: an instance of GASSOM_Model
            this.saveShotImage(gm);

            im = imread(this.shotImgPath);
            frame = im2frame(im);

            writeVideo(this.videoWriter, frame)
        end

        function saveShotImage (this, gm)
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
    end
end