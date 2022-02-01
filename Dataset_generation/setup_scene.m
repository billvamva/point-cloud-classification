function [scene,plat,lidar] = setup_scene(flight_data, y_mount_angle,color)
    scene = uavScenario("UpdateRate",1,"ReferenceLocation",[0 0 0]);

    plat = uavPlatform("UAV",scene,"ReferenceFrame","NED", ...
    "InitialPosition",flight_data.trajectory.random.coordinates(1,:), ...
    "InitialOrientation",eul2quat([flight_data.trajectory.random.angles(1)+pi,0,pi]));

    updateMesh(plat,"quadrotor",{10},color.Red,[0 0 0],eul2quat([0 0 0]))

    lidarmodel = uavLidarPointCloudGenerator("AzimuthResolution",0.13671875,...
        "ElevationLimits",[-30 30],"AzimuthLimits",[-35 35],"ElevationResolution",0.141509432,...
        "MaxRange",90,"UpdateRate",1,"HasOrganizedOutput",true,"HasNoise", true);

    lidar = uavSensor("Lidar",plat,lidarmodel, ...
        "MountingLocation",[0,0,-3],"MountingAngle",[0 y_mount_angle 0]);

end