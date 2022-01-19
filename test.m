%[status, result]=system('python3 test.py -p filepath')
color.Gray = 0.651*ones(1,3);
color.Green = [0.3922 0.8314 0.0745];
color.Red = [1 0 0];
[f,v,n] = stlread('Dataset generation/car_0219.stl');
x = f.Points(:,1);
y = f.Points(:,2);
z = f.Points(:,3);
% figure
% plot3(x,y,z)
scene = uavScenario("UpdateRate",12,"ReferenceLocation",[75 -46 0]);
addMesh(scene,"custom",{f.Points,f.ConnectivityList},color.Green)
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