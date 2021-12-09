<<<<<<< HEAD
import sys, getopt

def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hp:",[".bin file="])
   except getopt.GetoptError:
      print('test.py -p <inputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print ('test.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-p", "--path"):
         inputfile = arg
   print(f'Path to pointcloud is {inputfile}')


if __name__ == "__main__":
   main(sys.argv[1:])
=======
import numpy as np
import open3d as o3d
import sys

def plot_pcd(path):
    
    bin_pcd = np.fromfile(path , np.float32)

    # Reshape and drop reflection values
    points = bin_pcd.reshape((-1, 3))[:, 0:3]

    # Convert to Open3D point cloud
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    o3d.visualization.draw_geometries([o3d_pcd]) 

if __name__ == '__main__':
    
    path = sys.argv[1]

    plot_pcd(path)
>>>>>>> 5ed2bbc7c29dba6adb0c595a86474f19fd0fafbd
