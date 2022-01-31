run("flight_data_generator.m");
addpath('../progressbar/');
addpath('../saveAsJSON/');

mess = 'Starting simulation'


%Saving file and data json
file_template = 'surf_test/block_%d.pcd';
json_name = 'surf_test/simulation_data.json';


%Generate color identities for plotting
color.Gray = 0.651*ones(1,3);
color.Green = [0.3922 0.8314 0.0745];
color.Red = [1 0 0];

y_mount_angle = 25;

[scene,plat,lidar] = setup_scene(flight_data, y_mount_angle);

%Add ground plane
[X,Y,Z] = generate_ground(-20,20,[-.3 .3]);
addMesh(scene,"surface",{X,Y,Z},color.Gray)

% Load, scale, and position custom stl model.
[f,v,n] = stlread('Models/Cube.stl');
scale = 1;
translation = [0 0 0];
points = (f.Points*scale)+ translation;

%Camera transform for visualization of pc

rot = roty(-y_mount_angle);
trans = [0, 0, 0];
tform = rigid3d(rot,trans);

% Add initial target
points = rotate_target(f.Points);
addMesh(scene,"custom",{points,f.ConnectivityList},color.Green)

% Show the scenario.
show3D(scene);
xlim([-20 20])
ylim([-20 20])
zlim([-1 15])

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
        show3D(scene,"Time",lidarSampleTime,"Parent",ax,'FastUpdate',true);
        % Refresh all plot data and visualize.
        refreshdata
        drawnow limitrate
    end

    progressbar(idx/(size(flight_data.trajectory.random.coordinates, 1)-1));
%     % Advance scene simulation time and move platform.
    advance(scene);
    move(plat,[flight_data.trajectory.random.coordinates(idx+1,:),zeros(1,6), ...
        eul2quat([flight_data.trajectory.random.angles(idx+1)+pi,0,pi]),zeros(1,3)])
    ptCloudOut = pctransform(pt,tform);
    view(player,ptCloudOut)
    file_name = sprintf(file_template,idx+1);
    pcwrite(ptCloudOut,file_name,'PLYFormat','ascii')

    % Update all sensors in the scene.
    updateSensors(scene)

    %Update ground and target
    removeMesh(scene,2)
    removeMesh(scene,1)
    points = rotate_target(f.Points);
    addMesh(scene,'custom',{points,f.ConnectivityList}, color.Green)
    [X,Y,Z] = generate_ground(-20,20,[-.3 .3]);
    addMesh(scene,"surface",{X,Y,Z},color.Gray)
    
end
% hold off
mess = "Saving flight parameters"
saveAsJSON(flight_data.trajectory.random,json_name)
