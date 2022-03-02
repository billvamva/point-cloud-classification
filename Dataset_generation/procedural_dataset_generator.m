run("flight_data_generator.m");
addpath('../progressbar/');
addpath('../saveAsJSON/');

mess = 'Starting simulation'

color.Gray = 0.651*ones(1,3);
color.Green = [0.3922 0.8314 0.0745];
color.Red = [1 0 0];
y_mount_angle = 25;

[scene,plat,lidar] = setup_scene(flight_data, y_mount_angle, color);
setup(scene)
simulations = ["no_bcg","flat_bcg","rough_bcg"];

show3D(scene);
xlim([-20 20])
ylim([-20 20])
zlim([-1 15])

%Camera transform for visualization of pc

rot = roty(-y_mount_angle);
trans = [0, 0, 0];
tform = rigid3d(rot,trans);

[ax,plotFrames] = show3D(scene);
player = pcplayer([-50 50],[-50 50],[-10,30]);

for i = 1:length(simulations)

    models = load_models(simulations(i));
    if strcmp(simulations(i), 'flat_bcg')
        addMesh(scene,"polygon",{[-20 -20; 20 -20; 20 20; -20 20],[-2 0]},color.Gray)
    end

    for j = 1:length(models.model_files)
        model = models.model_objects.(models.model_files(j));
        target = model.Points;
        file_template = '%s_%s_dataset/%d_%.2g_%d.pcd';%target,bcg,vertical angle,horizontal angle,id number
        json_name = sprintf('%s_%s_dataset/simulation_data.json',models.model_files(j),simulations(i));%target,bcg
        for idx = 0:size(flight_data.trajectory.procedural.coordinates, 1)-1
            points = rotate_target(target);
            addMesh(scene,'custom',{points,model.ConnectivityList}, color.Green)

            if strcmp(simulations(i),'rough_bcg')
                [X,Y,Z] = generate_ground(-20,20,[-.3 .3]);
                addMesh(scene,"surface",{X,Y,Z},color.Gray)
            end

            [isupdated,lidarSampleTime, pt] = read(lidar);
            if isupdated
                show3D(scene,"Time",lidarSampleTime,"Parent",ax);
                % Refresh all plot data and visualize.
                refreshdata
                drawnow limitrate
            end
    
            progressbar(idx/(size(flight_data.trajectory.procedural.coordinates, 1)-1));
            advance(scene);
            move(plat,[flight_data.trajectory.procedural.coordinates(idx+1,:),zeros(1,6), ...
                eul2quat([flight_data.trajectory.procedural.angles(idx+1)+pi,0,pi]),zeros(1,3)])
            ptCloudOut = pctransform(pt,tform);
            view(player,ptCloudOut)
            file_name = sprintf(file_template,models.model_files(j),simulations(i),y_mount_angle, ...
                                flight_data.trajectory.procedural.angles(idx+1),idx+1);
            pcwrite(ptCloudOut,file_name,'PLYFormat','ascii')
            updateSensors(scene)
            
            
            if strcmp(simulations(i),'rough_bcg')
                removeMesh(scene,2)
                removeMesh(scene,1)
            elseif strcmp(simulations(i),'flat_bcg')
                removeMesh(scene,2)
            else
                removeMesh(scene,1)
            end
            
        end
        mess = "Saving flight parameters"
        saveAsJSON(flight_data,json_name)
    end
    
end