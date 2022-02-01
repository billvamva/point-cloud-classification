function points = rotate_target(targetPoints)
    angle_lims = [-15 15];
    rot_y = roty((angle_lims(2)-angle_lims(1))*rand()+angle_lims(1));
    rot_x = rotx((angle_lims(2)-angle_lims(1))*rand()+angle_lims(1));
    points = targetPoints * rot_y' * rot_x';
end