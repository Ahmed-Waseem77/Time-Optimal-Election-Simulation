from TestingSuite import profile_with_increasing_nodes, profile_with_increasing_connectivity
from TimelyOptimal import TimelyOptimalNode
from Node import Node
import cProfile
import pstats
import pandas as pd
import matplotlib.pyplot as plt


# Driver code
if __name__ == "__main__":
    profile_with_increasing_connectivity('TimelyOptimal')
    profile_with_increasing_nodes('TimelyOptimal')
