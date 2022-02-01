%[status, result]=system('python3 test_gian.py -p filepath')
clear all
% close all
% clc
% addpath('Dataset_generation/')
% color.Gray = 0.651*ones(1,3);
% color.Green = [0.3922 0.8314 0.0745];
% color.Red = [1 0 0];
% n_points = 30;
% 
% [f,v,n] = stlread('Dataset_generation/Models/car_0219.stl');
% points = f.Points;
% scene = uavScenario("UpdateRate",12,"ReferenceLocation",[0 0 0]);
% 
% [X,Y,Z] = generate_ground(-20,20,[-.3 .3]);
% addMesh(scene,"surface",{X,Y,Z},color.Gray)
% %addMesh(scene,"polygon",{[-2 -2; 2 -2 ; 2 2 ; -2 2],[0 5]},color.Red)
% addMesh(scene,'custom',{points,f.ConnectivityList},color.Green)
% % removeMesh(scene,1)
% show3D(scene)
% for i=1:100
%     removeMesh(scene,2)
%     removeMesh(scene,1)
%     points = rotate_target(f.Points);
%     addMesh(scene,'custom',{points,f.ConnectivityList}, color.Green)
%     [X,Y,Z] = generate_ground(-20,20,[-.3 .3]);
%     addMesh(scene,"surface",{X,Y,Z},color.Gray)
%     show3D(scene)
%     pause(2)
% end

% Set up platform mesh. Add a rotation to orient the mesh to the UAV body frame.

% addMesh(scene,"custom",{f.Points,f.ConnectivityList},color.Green)
% addMesh(scene,"polygon",{[-2 -2; 2 -2 ; 2 2 ; -2 2],[0 5]},color.Green)
% show3D(scene);
% addMesh(scene,"polygon",{[-2 -2; 2 -2 ; 2 2 ; -2 2],[0 10]},color.Green);
% show3D(scene);
% 
pc = pcread("Dataset_generation/Block_dataset/block_256.pcd");
pcshow(pc)

% ting = 100;
% 
% for i=1:ting
%     pause(0.2)
%     progressbar(i/ting);
% end

%saveAsJSON(flight_data,'Cylinder_dataset/simulation_data.json')


% points = zeros(n_points,n_points,3);