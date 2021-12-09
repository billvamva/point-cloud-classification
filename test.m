%[status, result]=system('python3 test.py -p filepath')
% [f,v,n] = stlread('sofa_0689.stl');
% x = f.Points(:,1);
% y = f.Points(:,2);
% z = f.Points(:,3);
% figure
% % plot3(x,y,z)
% scene = uavScenario("UpdateRate",12,"ReferenceLocation",[75 -46 0]);
% addMesh(scene,"custom",{f.Points,f.ConnectivityList},color.Green)
% show3D(scene);
% 
% pc = pcread("Cylinder_dataset/cylinder_3.pcd");
% pcshow(pc)

% ting = 100;
% 
% for i=1:ting
%     pause(0.2)
%     progressbar(i/ting);
% end

saveAsJSON(flight_data,'Cylinder_dataset/simulation_data.json')