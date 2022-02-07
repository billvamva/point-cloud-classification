clear all 
close all
clc
%%%Defining flight data
mess = 'Generating flight data'
flight_data = struct;
flight_data.trajectory = struct;
flight_data.trajectory.circle = struct;
flight_data.trajectory.diagonal = struct;
flight_data.trajectory.random =  struct;
flight_data.trajectory.procedural = struct;

show_trajs = false;
n_points = 32;

%%% Circle Variables
r=40;
height = 10;
parameters = [r,height,n_points];

%%% Defining points
z = -ones([1,n_points])*height;
theta=-pi:(2*pi/(n_points -1)):pi;
x=r*cos(theta);
y=r*sin(theta);
coordinates = [x(:),y(:),z(:)];

flight_data.trajectory.circle.coordinates = coordinates;
flight_data.trajectory.circle.angles = theta;
flight_data.trajectory.circle.parameters = parameters;
if show_trajs
    figure
    scatter3(x,y,z)
end
%Line variables
x_range = [-50 50];
y_range = [-50 50];
height = 20;
parameters = [x_range, y_range, height];

%Defining points
z = -ones([1,n_points])*height;
x = linspace(-50,50,n_points);
y = linspace(-50,50,n_points);
coordinates = [x(:),y(:),z(:)];

flight_data.trajectory.diagonal.coordinates = coordinates;
flight_data.trajectory.diagonal.parameters = [x_range, y_range, height];

if show_trajs
    figure
    scatter3(x,y,z)
end
%Random variables
radius_range = [10 20];
height_range = [5 10];
parameters = [radius_range,height_range];

%Defining points
theta = 2*pi*rand(1,n_points);
r = range(radius_range)*rand(1,n_points)+min(radius_range);
x = r.*cos(theta);
y = r.*sin(theta);
z = range(height_range)*rand(1,n_points)+min(height_range);
coordinates = [x(:),y(:),-z(:)];

flight_data.trajectory.random.coordinates = coordinates;
flight_data.trajectory.random.angles = theta;
flight_data.trajectory.random.parameters = parameters;
if show_trajs
    figure
    scatter3(x,y,z)
end

%Procedural variables
n_heights = 2;
n_radii = 2;
radius_range = [10 20];
height_range = [5 10];

assert(floor(n_points/n_heights) == n_points/n_heights)
assert(floor(n_points/(n_heights*n_radii))==n_points/(n_heights*n_radii))

heights = linspace(height_range(1),height_range(2),n_heights);
radii = linspace(radius_range(1),radius_range(2),n_radii);

parameters = [n_heights,radius_range,height_range];

x = [];
y = [];
z = [];
theta = [];
for i=1:length(heights)
    for j=1:length(radii)
        r = radii(j);
        z_part = -ones([1,n_points/(n_heights*n_radii)])*heights(i);
        theta_part=-pi:(2*pi/((n_points/(n_heights*n_radii)) -1)):pi;
        x_part=r*cos(theta_part);
        y_part=r*sin(theta_part);
        x = horzcat(x,x_part);
        y = horzcat(y,y_part);
        z = horzcat(z,z_part);
        theta = horzcat(theta,theta_part);
    end
end

coordinates = [x(:),y(:),z(:)];

flight_data.trajectory.procedural.coordinates = coordinates;
flight_data.trajectory.procedural.angles = theta;
flight_data.trajectory.procedural.parameters = parameters;
if show_trajs
    figure
    scatter3(x,y,z)
end
