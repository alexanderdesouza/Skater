"""Making LimeTabularExplainer Accessible"""
import os
import sys
lime_path = os.getcwd().split("Skater")[0] + "lime/"
sys.path.append(lime_path)
from lime.lime_tabular import LimeTabularExplainer
