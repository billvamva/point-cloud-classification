function [X,Y,Z] = generate_ground(min,max, height_range)
    Z = (height_range(2) - height_range(1)).*rand(max-min,max-min)+height_range(1);
    [X,Y] = ndgrid(min:1:max-1,min:1:max-1);
end

