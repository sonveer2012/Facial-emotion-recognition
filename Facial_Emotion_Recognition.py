import tensorflow as tf

import NetworkCreator as NC
import FilePaths as FP
import Optimizer
import Operations
import PerformanceMeasures as PM

session = object
#the main function to execute the operations of the project.
def main():
    global session
    session = tf.Session()
    print("Creating network..")
    NC.create_network(FP.json_path)
    NC.define_cost()
    PM.initialize_performance_measures()
    Operations.train()

if __name__ == "main":
    main()







