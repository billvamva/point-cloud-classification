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