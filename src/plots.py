import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend import Legend 
import matplotlib.patches as patches
from matplotlib import font_manager
from matplotlib import gridspec
import seaborn as sns
import scanpy as sc
from scipy.sparse import issparse
import yaml
from pathlib import Path
import warnings
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list
import nbformat
from collections import defaultdict
import plotly.graph_objects as go
import json

warnings.filterwarnings('ignore')



class Plotting:
    """
        Class to create atlas figures with config yml
    """

    def __init__(self, config_path, output_dir="figures"):
        self.config = self._load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_plotting_params()

    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        if isinstance(config_path, str):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        elif isinstance(config_path, dict):
            return config_path
        else:
            raise ValueError("Config must be a file path or dictionary")

    def _setup_plotting_params(self):
        """Set up matplotlib and scanpy plotting parameters"""
        cfg = self.config['plot_configs']['general']
        plt.rcParams['figure.dpi'] = cfg['dpi']
        plt.rcParams['savefig.dpi'] = cfg['dpi_save']
        plt.rcParams['legend.fontsize'] = cfg['legend_fontsize']
        plt.rcParams['axes.titlesize'] = cfg['title_fontsize']
        plt.rcParams['font.family'] = cfg["font_family"]
        sns.set_theme(style="white",font=cfg["font_family"])
        sc.settings.set_figure_params(
            dpi_save=cfg['dpi_save'],
            fontsize=cfg['legend_fontsize']
        )
