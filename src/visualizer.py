import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from logging_utils import log

class Visualizer:
    def __init__(self, figsize=(10, 6)):
        """Initialize the Visualizer with default figure size."""
        self.figsize = figsize

    def plot_numeric_scatter(self, df, numeric_list, color_label):
        """Create scatter plots for combinations of numeric features."""
        combo_list = []
        for i in range(len(numeric_list)):
            for j in range(i+1, len(numeric_list)):
                combo_list.append((numeric_list[i], numeric_list[j]))

        num_combos = len(combo_list)
        fig = plt.figure(figsize=(self.figsize[0], num_combos * self.figsize[1]))
        
        for i, (x, y) in enumerate(combo_list):
            ax = fig.add_subplot(num_combos, 1, i+1)
            sns.scatterplot(data=df, x=x, y=y, hue=color_label)
            ax.set_title(f'{x} vs {y}')
        
        fig.tight_layout()
        return fig

    def plot_categorical_counts(self, df, categorical_list, color_label):
        """Create count plots for categorical features."""
        num_feats = len(categorical_list)
        fig = plt.figure(figsize=(self.figsize[0], num_feats * self.figsize[1]))
        
        for i, var_name in enumerate(categorical_list):
            ax = fig.add_subplot(num_feats, 1, i+1)
            sns.countplot(data=df, x=var_name, hue=color_label)
            ax.set_title(var_name)
        
        fig.tight_layout()
        return fig

    def plot_numeric_histograms(self, df, numeric_tuples, color_label):
        """Create histograms for numeric features."""
        num_numerics = len(numeric_tuples)
        fig = plt.figure(figsize=(self.figsize[0], num_numerics * self.figsize[1]))
        
        for i, (numeric, bw) in enumerate(numeric_tuples):
            ax = fig.add_subplot(num_numerics, 1, i+1)
            sns.histplot(data=df, x=numeric, hue=color_label, binwidth=bw, kde=True)
            ax.set_title(numeric)
        
        fig.tight_layout()
        return fig

    def plot_category_heatmap(self, df, combo_list):
        """Create heatmaps for combinations of categorical features."""
        num_combos = len(combo_list)
        fig = plt.figure(figsize=(self.figsize[0], num_combos * self.figsize[1]))
        
        for i, (x, y) in enumerate(combo_list):
            gbdf = df.groupby([y, x])[x].size().unstack().fillna(0)
            ax = fig.add_subplot(num_combos, 1, i+1)
            sns.heatmap(gbdf.T, annot=True, fmt='g', cmap='coolwarm')
            ax.set_title(f'{x} vs {y}')
        
        fig.tight_layout()
        return fig

    def plot_category_group_counts(self, df, combo_list):
        """Create count plots for unique values in categorical group combinations."""
        df_dict = {}
        for tpl in combo_list:
            (y, x) = tpl
            cat_count_df = df.groupby([y, x])[x].size().unstack().fillna(0)
            unique_series = (cat_count_df > 0).sum(axis=1).reset_index(name='Count')['Count'].sort_values().astype(str)
            df_dict[tpl] = unique_series

        num_combos = len(combo_list)
        fig = plt.figure(figsize=(self.figsize[0], num_combos * self.figsize[1]))
        
        for i, (tpl, dfgb) in enumerate(df_dict.items()):
            (x, y) = tpl
            ax = fig.add_subplot(num_combos, 1, i+1)
            sns.countplot(x=dfgb)
            ax.set_title(f"Unique {y} per {x}")
        
        fig.tight_layout()
        return fig, df_dict

    def plot_correlation_matrix(self, df, columns):
        """Plot correlation matrix for specified columns."""
        corr_matrix = df[columns].corr()
        plt.figure(figsize=self.figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        return corr_matrix 