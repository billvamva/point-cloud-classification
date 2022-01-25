%[status, result]=system('python3 test.py -p filepath')
clear all
close all
clc

color.Gray = 0.651*ones(1,3);
color.Green = [0.3922 0.8314 0.0745];
color.Red = [1 0 0];
n_points = 30;
height_range = [-.3 .3];
ground = (height_range(2) - height_range(1)).*rand(n_points,n_points)+height_range(1);
[X,Y] = ndgrid(0:1:n_points-1,0:1:n_points-1);

% [f,v,n] = stlread('Dataset_generation/car_0219.stl');
% x = f.Points(:,1);
% y = f.Points(:,2);
% z = f.Points(:,3);
% figure
% plot3(x,y,z)
scene = uavScenario("UpdateRate",12,"ReferenceLocation",[75 -46 0]);
addMesh(scene,"surface",{X,Y,ground},color.Gray)
% addMesh(scene,"custom",{f.Points,f.ConnectivityList},color.Green)
% addMesh(scene,"polygon",{[-2 -2; 2 -2 ; 2 2 ; -2 2],[0 5]},color.Green)
show3D(scene);
% 
% pc = pcread("Car_dataset/car_269.pcd");
% pcshow(pc)

% ting = 100;
% 
% for i=1:ting
%     pause(0.2)
%     progressbar(i/ting);
% end

%saveAsJSON(flight_data,'Cylinder_dataset/simulation_data.json')


% points = zeros(n_points,n_points,3);