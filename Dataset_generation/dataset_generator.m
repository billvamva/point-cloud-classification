run("flight_data_generator.m");
addpath('../progressbar/');
addpath('../saveAsJSON/');
mess = 'Starting simulation'
scene = uavScenario("UpdateRate",12,"ReferenceLocation",[75 -46 0]);

% Add a ground plane.
color.Gray = 0.651*ones(1,3);
color.Green = [0.3922 0.8314 0.0745];
color.Red = [1 0 0];
%addMesh(scene,"polygon",{[-20 -20; 20 -20; 20 20; -20 20],[-2 0]},color.Gray)
[X,Y,Z] = generate_ground(-20,20,[-.3 .3]);
addMesh(scene,"surface",{X,Y,Z},color.Gray)

% Load custom stl model.
[f,v,n] = stlread('car.stl');
scale = 1;
translation = [0 0 0];
points = (f.Points*scale)+ translation;

y_mount_angle = 25;
rot = roty(-y_mount_angle);
trans = [0, 0, 0];
tform = rigid3d(rot,trans);

% Add sets of polygons as extruded meshes with varying heights from 10-30.
%addMesh(scene,"cylinder",{[0,0,3],[0,5]},color.Green)

%addMesh(scene,"custom",{points,f.ConnectivityList},color.Green)
addMesh(scene,"polygon",{[-2 -2; 2 -2 ; 2 2 ; -2 2],[0 5]},color.Red)

% Show the scenario.
show3D(scene);
xlim([-20 20])
ylim([-20 20])
zlim([-1 15])



% Set up platformb [-40 0 -10]
plat = uavPlatform("UAV",scene,"ReferenceFrame","NED", ...
    "InitialPosition",flight_data.trajectory.random.coordinates(1,:),"InitialOrientation",eul2quat([flight_data.trajectory.random.angles(1)+pi,0,pi]));

% Set up platform mesh. Add a rotation to orient the mesh to the UAV body frame.
updateMesh(plat,"quadrotor",{10},color.Red,[0 0 0],eul2quat([0 0 0]))

lidarmodel = uavLidarPointCloudGenerator("AzimuthResolution",0.13671875,...
    "ElevationLimits",[-30 30],"AzimuthLimits",[-35 35],"ElevationResolution",0.141509432,...
    "MaxRange",90,"UpdateRate",6,"HasOrganizedOutput",true,"HasNoise", true);

lidar = uavSensor("Lidar",plat,lidarmodel,"MountingLocation",[0,0,-3],"MountingAngle",[0 y_mount_angle 0]);

[ax,plotFrames] = show3D(scene);


xlim([-50 50])
ylim([-50 50])
zlim([-1 50])
view([-110 30])
axis equal
hold on
colormap("jet")
pt = pointCloud(nan(1,1,3));
scatterplot = scatter3(nan,nan,nan,1,[0.3020 0.7451 0.9333],...
    "Parent",plotFrames.UAV.Lidar);
scatterplot.XDataSource = "reshape(pt.Location(:,:,1),[],1)";
scatterplot.YDataSource = "reshape(pt.Location(:,:,2),[],1)";
scatterplot.ZDataSource = "reshape(pt.Location(:,:,3),[],1)";
scatterplot.CDataSource = "reshape(pt.Location(:,:,3),[],1) - min(reshape(pt.Location(:,:,3),[],1))";
setup(scene)

player = pcplayer([-50 50],[-50 50],[-10,30]);
for idx = 0:size(flight_data.trajectory.random.coordinates, 1)-1
    [isupdated,lidarSampleTime, pt] = read(lidar);
    if isupdated
        % Use fast update to move platform visualization frames.
        show3D(scene,"Time",lidarSampleTime,"FastUpdate",true,"Parent",ax);
        % Refresh all plot data and visualize.
        refreshdata
        drawnow limitrate
    end
    progressbar(idx/(size(flight_data.trajectory.random.coordinates, 1)-1));
%     % Advance scene simulation time and move platform.
    advance(scene);
    move(plat,[flight_data.trajectory.random.coordinates(idx+1,:),zeros(1,6),eul2quat([flight_data.trajectory.random.angles(idx+1)+pi,0,pi]),zeros(1,3)])
    ptCloudOut = pctransform(pt,tform);
    view(player,ptCloudOut)
    file_name = sprintf('Block_dataset/block_%d.pcd',idx+1);
    pcwrite(ptCloudOut,file_name,'PLYFormat','ascii')
    % Update all sensors in the scene.
    updateSensors(scene)
end
% hold off
mess = "Saving flight parameters"
saveAsJSON(flight_data.trajectory.random,'Block_dataset/simulation_data.json')
