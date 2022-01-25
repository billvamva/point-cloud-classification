# import sys, getopt

# def main(argv):
#    inputfile = ''
#    outputfile = ''
#    try:
#       opts, args = getopt.getopt(argv,"hp:",[".bin file="])
#    except getopt.GetoptError:
#       print('test.py -p <inputfile>')
#       sys.exit(2)
#    for opt, arg in opts:
#       if opt == '-h':
#          print ('test.py -i <inputfile> -o <outputfile>')
#          sys.exit()
#       elif opt in ("-p", "--path"):
#          inputfile = arg
#    print(f'Path to pointcloud is {inputfile}')


# if __name__ == "__main__":
#    main(sys.argv[1:])
# #
import open3d as o3d
import numpy as np

print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("Simulation/Dataset_generation/Car_dataset/car_100.pcd")
# print(pcd)
# print(np.asarray(pcd.points).shape)
o3d.visualization.draw_geometries([pcd])
