function angle = pvaldeg(angle) % change angle range
    dtr = pi/180;
    angle = atan2(sin(angle*dtr),cos(angle*dtr))/dtr;
    if angle<-90
        angle = angle+360;
    end
end