clear all
close all
clc

%Create the UAV scenario.
scene = uavScenario("UpdateRate",6,"ReferenceLocation",[75 -46 0]);

% Add a ground plane.
color.Gray = 0.651*ones(1,3);
color.Green = [0.3922 0.8314 0.0745];
color.Red = [1 0 0];
xterr = [0 450] + 5578445;
yterr = [0 330 ]-2976110;
%addMesh(scene,"polygon",{[-250 -150; 200 -150; 200 180; -250 180],[-4 0]},color.Gray)
%addCustomTerrain("ground","n00_e015_1arc_v3.dt2")
addMesh(scene,"terrain",{"ground",[5578444.315,5578444.315+100],[-2976110.7132,-2976110.7132+100]},color.Gray)

% % Load building polygons.
% load("buildingData.mat");
% 
% % Add sets of polygons as extruded meshes with varying heights from 10-30.
% addMesh(scene,"polygon",{buildingData{1}(1:4,:),[0 30]},color.Green)
% addMesh(scene,"polygon",{buildingData{2}(2:5,:),[0 30]},color.Green)
% addMesh(scene,"polygon",{buildingData{3}(2:10,:),[0 30]},color.Green)
% addMesh(scene,"polygon",{buildingData{4}(2:9,:),[0 30]},color.Green)
% addMesh(scene,"polygon",{buildingData{5}(1:end-1,:),[0 30]},color.Green)
% addMesh(scene,"polygon",{buildingData{6}(1:end-1,:),[0 15]},color.Red)
% addMesh(scene,"polygon",{buildingData{7}(1:end-1,:),[0 30]},color.Green)
% addMesh(scene,"polygon",{buildingData{8}(2:end-1,:),[0 10]},color.Green)
% addMesh(scene,"polygon",{buildingData{9}(1:end-1,:),[0 15]},color.Green)
% addMesh(scene,"polygon",{buildingData{10}(1:end-1,:),[0 30]},color.Green)
% addMesh(scene,"polygon",{buildingData{11}(1:end-2,:),[0 30]},color.Green)

% Show the scenario.
show3D(scene);
% xlim([-250 200])
% ylim([-150 180])
% zlim([0 50])



load("flightData.mat")

% Set up platform
plat = uavPlatform("UAV",scene,"ReferenceFrame","NED", ...
    "InitialPosition",position(:,:,1),"InitialOrientation",eul2quat(orientation(:,:,1)));

% Set up platform mesh. Add a rotation to orient the mesh to the UAV body frame.
updateMesh(plat,"quadrotor",{10},color.Red,[0 0 0],eul2quat([0 0 pi]))

lidarmodel = uavLidarPointCloudGenerator("AzimuthResolution",0.13671875,...
    "ElevationLimits",[-30 30],"AzimuthLimits",[-35 35],"ElevationResolution",0.141509434,...
    "MaxRange",90,"UpdateRate",6,"HasOrganizedOutput",true);

lidar = uavSensor("Lidar",plat,lidarmodel,"MountingLocation",[0,0,-1],"MountingAngle",[0 60 0 ]);

[ax,plotFrames] = show3D(scene);


xlim([-250 200])
ylim([-150 180])
zlim([0 50])
view([-110 30])
axis equal
hold on

traj = plot3(nan,nan,nan,"Color",[1 1 1],"LineWidth",2);
traj.XDataSource = "position(:,2,1:idx+1)";
traj.YDataSource = "position(:,1,1:idx+1)";
traj.ZDataSource = "-position(:,3,1:idx+1)";


colormap("jet")
pt = pointCloud(nan(1,1,3));
scatterplot = scatter3(nan,nan,nan,1,[0.3020 0.7451 0.9333],...
    "Parent",plotFrames.UAV.Lidar);
scatterplot.XDataSource = "reshape(pt.Location(:,:,1),[],1)";
scatterplot.YDataSource = "reshape(pt.Location(:,:,2),[],1)";
scatterplot.ZDataSource = "reshape(pt.Location(:,:,3),[],1)";
scatterplot.CDataSource = "reshape(pt.Location(:,:,3),[],1) - min(reshape(pt.Location(:,:,3),[],1))";
setup(scene)

player = pcplayer([-100 100],[-100 100],[0,20]);

for idx = 0:size(position, 3)-1
    [isupdated,lidarSampleTime, pt] = read(lidar);
    if isupdated
        % Use fast update to move platform visualization frames.
        show3D(scene,"Time",lidarSampleTime,"FastUpdate",true,"Parent",ax);
        % Refresh all plot data and visualize.
        refreshdata
        drawnow limitrate
    end
    % Advance scene simulation time and move platform.
    advance(scene);
    move(plat,[position(:,:,idx+1),zeros(1,6),eul2quat(orientation(:,:,idx+1)),zeros(1,3)])
    % Update all sensors in the scene.
    view(player,pt.Location(:,:,:))
    updateSensors(scene)
end
hold off