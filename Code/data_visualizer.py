"""Data visualization tools.

The principle of organizing this code is that I do not put any calculation here,
as this script is ONLY used for plotting. Any calculation involved should be
moved outside and run unit tests ideally. In another word, I am not worried
about the correctness of plotting as long as the numbers put in are correct.
"""
import os

from absl import logging
import collections
import numpy as np
import matplotlib
from matplotlib import gridspec
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
from tqdm import tqdm

import util


class AllenInstituteDataVisualizer(object):

  def __init__(self, session):
    self.session = session

  def plot_spikes_per_unit(
      self,
      stimulus_presentation_ids,
      unit_ids,
      trial_time_window=None,
      output_figure_path=None,
      show_figure=True):
    """Plots selected units."""
    stimulus_presentations = self.session._filter_owned_df(
        'stimulus_presentations', ids=stimulus_presentation_ids)
    units = self.session._filter_owned_df('units', ids=unit_ids)
    # display(units)

    spikes_table = self.session.trialwise_spike_times(
        stimulus_presentation_ids, unit_ids, trial_time_window)
    num_neurons = len(unit_ids)
    num_trials = len(stimulus_presentation_ids)

    plt.figure(figsize=(10, num_neurons * 0.3 + 2))
    for u, unit_id in enumerate(unit_ids):
      for s, stimulus_presentation_id in enumerate(stimulus_presentation_ids):

        spike_times = spikes_table[
            (spikes_table['unit_id'] == unit_id) &
            (spikes_table['stimulus_presentation_id'] ==
             stimulus_presentation_id)]
        spike_times = spike_times['time_since_stimulus_presentation_onset']
        y_values = np.zeros(spike_times.shape) + (
            s / num_trials + u)
        plt.plot(spike_times, y_values,
                 's', markersize=0.5,
                 color='k' if u % 2 == 0 else 'b')

      neuron_name = (units.loc[unit_id]['ecephys_structure_acronym'] +
                     str(unit_id))
      # neuron_name = areas[p] + str(unit)
      plt.text(2.55, u + 0.3, neuron_name,
               color='k' if u % 2 == 0 else 'b', fontsize=6)
    plt.xlabel('Time [s]')
    plt.grid(linestyle='dotted')
    plt.xlim(trial_time_window)

    if output_figure_path:
      plt.savefig(output_figure_path)
      print('Save figure to: ', output_figure_path)
    if show_figure:
      plt.show()
    plt.close()

  def plot_all_spikes_per_unit_per_probe(
      self,
      stimulus_presentation_ids,
      trial_time_window=None,
      output_figure_path=None,
      show_figure=True):
    """Plots selected units."""
    stimulus_presentations = self.session._filter_owned_df(
        'stimulus_presentations', ids=stimulus_presentation_ids)
    units = self.session.units
    num_trials = len(stimulus_presentation_ids)
    probes_list = ['probeA', 'probeB', 'probeC', 'probeD', 'probeE', 'probeF']

    plt.figure(figsize=(10*6, 10*6))
    for p, probe in enumerate(probes_list):
      ax = plt.subplot(1, len(probes_list), p + 1)
      if probe in units['probe_description'].values:
        probe_units = units[units['probe_description'] == probe]
      else:
        continue
      probe_unit_ids = probe_units.index.values
      spikes_table = self.session.trialwise_spike_times(
          stimulus_presentation_ids, probe_unit_ids, trial_time_window)

      for u, unit_id in enumerate(probe_unit_ids):
        for s, stimulus_presentation_id in enumerate(stimulus_presentation_ids):

          spike_times = spikes_table[
              (spikes_table['unit_id'] == unit_id) &
              (spikes_table['stimulus_presentation_id'] ==
               stimulus_presentation_id)]
          spike_times = spike_times['time_since_stimulus_presentation_onset']
          y_values = np.zeros(spike_times.shape) + (
              s / num_trials + u)

          area_color = util.color_by_brain_area(
              probe_units.loc[unit_id]['ecephys_structure_acronym'])
          plt.plot(spike_times, y_values,
                   's', markersize=0.5,
                   color=area_color if u % 2 == 0 else 'k')

        neuron_name = (units.loc[unit_id]['ecephys_structure_acronym'] +
                       str(unit_id))

        plt.text(2.55, u + 0.3, neuron_name,
                 color=area_color, fontsize=6)
      plt.xlim(trial_time_window)
      plt.grid(linestyle='dotted')
      title_name = [probe] + list(
          probe_units['ecephys_structure_acronym'].unique())
      print(title_name)
      plt.title(' '.join(title_name), fontsize=12)
      if p == 0:
        plt.xlabel('Time [s]')

    if output_figure_path:
      plt.savefig(output_figure_path)
      print('Save figure to: ', output_figure_path)
    if show_figure:
      plt.show()
    plt.close()

  def plot_psth_per_unit(
      self,
      time_bin_edges,
      stimulus_presentation_ids,
      unit_ids,
      smooth_sigma=0.8,
      output_figure_path=None,
      show_figure=True):
    """Plots PSTH.

    Args:
      spike_trains_data: The data structure is defined as
          `data_parser.MultiArraySpikeTrain`.
    """
    stimulus_presentations = self.session._filter_owned_df(
        'stimulus_presentations', ids=stimulus_presentation_ids)
    units = self.session._filter_owned_df('units', ids=unit_ids)

    histogram_table = self.session.unitwise_spike_histogram(
        time_bin_edges, stimulus_presentation_ids, unit_ids, smooth_sigma)
    num_neurons = len(unit_ids)
    num_trials = len(stimulus_presentation_ids)
    # Median for plot scaling.
    _amplitude = np.quantile(histogram_table.values, 0.98) * 1.5

    plt.figure(figsize=(10, num_neurons * 0.3 + 2))
    for u, unit_id in enumerate(unit_ids):

      # Use the left edges of time bins as the time line.
      if sum(histogram_table[:, u]) == 0:
        continue
      plt.plot(
          histogram_table['time_relative_to_stimulus_onset'],
          histogram_table[:, u] / _amplitude + u,
          # color=util.color_by_brain_area(ccf_structure),
          color='k',
          linewidth=0.4)

      neuron_name = (units.loc[unit_id]['ecephys_structure_acronym'] +
                     str(unit_id))
      plt.text(2.55, u, neuron_name,
               color='k' if u % 2 == 0 else 'b', fontsize=6)
      # plt.title(probe + ' ' + util.VISUAL_AREA_BY_PROBE[probe])
      plt.grid(linestyle='dotted')
      plt.xlim(np.min(histogram_table['time_relative_to_stimulus_onset']),
               np.max(histogram_table['time_relative_to_stimulus_onset']))
      plt.xlabel('Time (s)')
      plt.ylabel('Unit')
    title_name = units['ecephys_structure_acronym'].unique()
    plt.title(' '.join(title_name), fontsize=12)
    if output_figure_path:
      plt.savefig(output_figure_path)
      print('Save figure to: ', output_figure_path)
    if show_figure:
      plt.show()
    plt.close()

  def plot_all_psth_per_unit_per_probe(
      self,
      time_bin_edges,
      stimulus_presentation_ids,
      smooth_sigma=0.8,
      output_figure_path=None,
      show_figure=True):
    """Plots PSTH.

    Args:
      spike_trains_data: The data structure is defined as
          `data_parser.MultiArraySpikeTrain`.
    """
    stimulus_presentations = self.session._filter_owned_df(
        'stimulus_presentations', ids=stimulus_presentation_ids)
    units = self.session.units
    probes_list = ['probeA', 'probeB', 'probeC', 'probeD', 'probeE', 'probeF']

    plt.figure(figsize=(10*6, 10*6))
    for p, probe in enumerate(probes_list):
      ax = plt.subplot(1, len(probes_list), p + 1)
      if probe in units['probe_description'].values:
        probe_units = units[units['probe_description'] == probe]
      else:
        continue
      probe_unit_ids = probe_units.index.values
      histogram_table = self.session.unitwise_spike_histogram(
          time_bin_edges, stimulus_presentation_ids,
          probe_unit_ids, smooth_sigma)

      for u, probe_unit_id in enumerate(probe_unit_ids):
        # Use the left edges of time bins as the time line.
        if sum(histogram_table[:, u]) == 0:
          continue
        area_color = util.color_by_brain_area(
            probe_units.loc[probe_unit_id]['ecephys_structure_acronym'])
        plt.plot(
            histogram_table['time_relative_to_stimulus_onset'],
            histogram_table[:, u]*2 + u,
            color=area_color,
            linewidth=0.6)

        neuron_name = (
            probe_units.loc[probe_unit_id]['ecephys_structure_acronym'] +
            str(probe_unit_id))
        plt.text(2.55, u, neuron_name, color=area_color, fontsize=6)
      plt.grid(linestyle='dotted')
      plt.xlim(np.min(histogram_table['time_relative_to_stimulus_onset']),
               np.max(histogram_table['time_relative_to_stimulus_onset']))
      title_name = [probe] + list(
          probe_units['ecephys_structure_acronym'].unique())
      print(title_name)
      plt.title(' '.join(title_name), fontsize=12)
      if p == 0:
        plt.xlabel('Time (s)')

    if output_figure_path:
      plt.savefig(output_figure_path)
      print('Save figure to: ', output_figure_path)
    if show_figure:
      plt.show()
    plt.close()

  def plot_table(
      self,
      table,
      output_figure_path=None,
      show_figure=True):
    """Plot the rows from the `table`.

    Args:
      table: This can be pandas table or numpy matrix.
          row: data entries.
          col: data series.
    """
    num_rows, num_cols = table.shape
    if isinstance(table, pd.DataFrame):
      time_line = table.columns.values
      _median = np.median(table.values)
    else:
      time_line = np.arange(num_cols)
      _median = np.median(table)

    plt.figure(figsize=(10, num_rows * 0.3 + 2))
    for u in range(num_rows):
      if isinstance(table, pd.DataFrame):
        y_values = table.iloc[u] / 2
        row_name = (table.index.values[u])
      else:
        y_values = table[u]
        row_name = u
      plt.plot(time_line, y_values / _median + u,
          color='k' if u % 2 == 0 else 'b',
          linewidth=0.4)

      plt.text(np.max(time_line) + 0.05, u, row_name,
               color='k' if u % 2 == 0 else 'b', fontsize=6)

    plt.grid(linestyle='dotted')
    plt.xlim(np.min(time_line), np.max(time_line))
    # plt.xlabel('Time [s]')
    # plt.ylabel('Unit')

    if output_figure_path:
      plt.savefig(output_figure_path)
      print('Save figure to: ', output_figure_path)
    if show_figure:
      plt.show()
    plt.close()

  def plot_tables(
      self,
      tables,
      output_figure_path=None,
      show_figure=True):
    """Plot the rows from the `table`.

    Args:
      table: This can be pandas table or numpy matrix.
          row: data entries.
          col: data series.
    """
    num_tables = len(tables)
    table = tables[0]
    num_rows, num_cols = table.shape
    if isinstance(table, pd.DataFrame):
      x = table.columns.values
      _median = np.median(table.values)
    else:
      x = np.arange(num_cols)
      _median = np.median(table)
    # Check if every table matches.
    for table in tables:
      num_rows_, num_cols_ = table.shape
      if isinstance(table, pd.DataFrame):
        x_ = table.columns.values
      else:
        x_ = np.arange(num_cols)
      if (num_rows_ != num_rows or
          num_cols_ != num_cols or
          np.testing.assert_equal(x_, x)):
        raise ValueError('Table sizes do not match.')

    fig = plt.figure(figsize=(10, num_rows * 1.1 + 2))
    for t, table in enumerate(tables):
      if isinstance(table, pd.DataFrame):
        _median = np.median(table.values)
      else:
        _median = np.median(table)

      for u in range(num_rows):
        if isinstance(table, pd.DataFrame):
          # y_values = table.iloc[u] / _median / 3
          y_values = table.iloc[u] / 0.03 # / 200
          row_name = (table.index.values[u])
        else:
          y_values = table[u]
          row_name = u

        if t == 1:
          linestyle = '-.'
        else:
          linestyle = '-'
        plt.plot(x, y_values + u,
            linestyle=linestyle,
            color='k' if u % 2 == 0 else 'b',
            linewidth=0.4)

        plt.text(np.max(x) + 0.05, u, row_name,
                 color='k' if u % 2 == 0 else 'b', fontsize=6)

    plt.grid(linestyle='dotted')
    plt.xlim(np.min(x), np.max(x))
    # plt.xlim(0, 0.3)
    frame1 = plt.gca()
    frame1.axes.yaxis.set_ticklabels([])
    # plt.xlabel('Time [s]')
    # plt.ylabel('Unit')

    if output_figure_path:
      plt.savefig(output_figure_path)
      print('Save figure to: ', output_figure_path)
    if show_figure:
      plt.show()
    plt.close()

  def plot_trial_metric_per_unit_per_trial(
      self,
      stimulus_presentation_ids,
      unit_ids,
      trial_time_window,
      metric_type,
      output_figure_path=None,
      show_figure=True):
    """Plots Spike counts."""
    stimulus_presentations = self.session._filter_owned_df(
        'stimulus_presentations', ids=stimulus_presentation_ids)
    units = self.session._filter_owned_df('units', ids=unit_ids)
    # display(units)

    spikes_table = self.session.trialwise_spike_times(
        stimulus_presentation_ids, unit_ids, trial_time_window)
    num_neurons = len(unit_ids)
    num_trials = len(stimulus_presentation_ids)

    plt.figure(figsize=(10, num_neurons * 0.3 + 2))
    for u, unit_id in enumerate(unit_ids):
      metrics = np.zeros(num_trials)
      for s, stimulus_presentation_id in enumerate(stimulus_presentation_ids):

        spike_times = spikes_table[
            (spikes_table['unit_id'] == unit_id) &
            (spikes_table['stimulus_presentation_id'] ==
             stimulus_presentation_id)]
        spike_times = spike_times['time_since_stimulus_presentation_onset']
        if metric_type == 'count':
          metrics[s] = len(spike_times) if len(spike_times) != 0 else np.nan
          plot_normalizer = lambda x: x / 40
        elif metric_type == 'shift':
          metrics[s] = np.mean(spike_times)
          plot_normalizer = lambda x: (x - 1) * 5
        else:
          raise TypeError('Wrong type of metric_type.')

      y_values = plot_normalizer(metrics) + u
      plt.plot(range(num_trials), y_values,
               linewidth=0.4,
               color='k' if u % 2 == 0 else 'b')

      neuron_name = (units.loc[unit_id]['ecephys_structure_acronym'] +
                     str(unit_id))
      # neuron_name = areas[p] + str(unit)
      plt.text(num_trials-.45, u + 0.3, neuron_name,
               color='k' if u % 2 == 0 else 'b', fontsize=6)
    plt.xlabel('Time [s]')
    plt.grid(linestyle='dotted')
    plt.xlim(0, num_trials-1)

    if output_figure_path:
      plt.savefig(output_figure_path)
      print('Save figure to: ', output_figure_path)
    if show_figure:
      plt.show()
    plt.close()

  def plot_all_trial_metric_per_unit_per_probe(
      self,
      stimulus_presentation_ids,
      trial_time_window,
      metric_type,
      output_figure_path=None,
      show_figure=True):
    """Plots selected units."""
    stimulus_presentations = self.session._filter_owned_df(
        'stimulus_presentations', ids=stimulus_presentation_ids)
    units = self.session.units
    num_trials = len(stimulus_presentation_ids)
    probes_list = ['probeA', 'probeB', 'probeC', 'probeD', 'probeE', 'probeF']

    plt.figure(figsize=(10*6, 10*6))
    for p, probe in enumerate(probes_list):
      ax = plt.subplot(1, len(probes_list), p + 1)
      if probe in units['probe_description'].values:
        probe_units = units[units['probe_description'] == probe]
      else:
        continue
      probe_unit_ids = probe_units.index.values
      spikes_table = self.session.trialwise_spike_times(
          stimulus_presentation_ids, probe_unit_ids, trial_time_window)

      for u, unit_id in enumerate(probe_unit_ids):
        metrics = np.zeros(num_trials)
        for s, stimulus_presentation_id in enumerate(stimulus_presentation_ids):

          spike_times = spikes_table[
              (spikes_table['unit_id'] == unit_id) &
              (spikes_table['stimulus_presentation_id'] ==
               stimulus_presentation_id)]
          spike_times = spike_times['time_since_stimulus_presentation_onset']
          if metric_type == 'count':
            metrics[s] = len(spike_times) if len(spike_times) != 0 else np.nan
            plot_normalizer = lambda x: x / 40
          elif metric_type == 'shift':
            metrics[s] = np.mean(spike_times)
            plot_normalizer = lambda x: (x - 1) * 2
          else:
            raise TypeError('Wrong type of metric_type.')

        y_values = plot_normalizer(metrics) + u
        area_color = util.color_by_brain_area(
            probe_units.loc[unit_id]['ecephys_structure_acronym'])

        plt.plot(range(num_trials), y_values,
                 linewidth=0.4,
                 color=area_color)

        neuron_name = (probe_units.loc[unit_id]['ecephys_structure_acronym'] +
                       str(unit_id))

        plt.text(num_trials-.45, u + 0.3, neuron_name,
                 color=area_color, fontsize=6)
      plt.xlim(0, num_trials-1)
      plt.grid(linestyle='dotted')
      title_name = [probe] + list(
          probe_units['ecephys_structure_acronym'].unique())
      print(title_name)
      plt.title(' '.join(title_name), fontsize=12)
      if p == 0:
        plt.xlabel('Time [s]')

    if output_figure_path:
      plt.savefig(output_figure_path)
      print('Save figure to: ', output_figure_path)
    if show_figure:
      plt.show()
    plt.close()

  def plot_matrix_by_areas(
      self,
      matrix,
      fig_title='',
      output_figure_path=None,
      show_figure=True):
    """Plot the trial-to-trial variability correlation matrix.

    `units` should match the rows of `matrix`.
    """
    units = self.session._filter_owned_df('units', ids=matrix.index.values)
    units_left = self.session._filter_owned_df('units',
                                               ids=matrix.columns.values)

    # Areas labels
    num_areas = 5
    num_units = len(units)
    matrix_height, matrix_width = matrix.shape
    areas_names = ['Others', 'Visual cortex', 'Hippocampus', 'Midbrain',
                   'Thalamus']
    ################################### TOP ####################################
    # The order is the same as the color `color_list`.
    areas_string = units['ecephys_structure_acronym']
    areas_category = areas_string.unique()
    areas_category_values = np.zeros(len(areas_category))
    areas_category_values[[
        item in util.VISUAL_AREA for item in areas_category]] = 1
    areas_category_values[[
        item in util.HIPPOCAMPUS_AREA for item in areas_category]] = 2
    areas_category_values[[
        item in util.MIDBRAIN for item in areas_category]] = 3
    areas_category_values[[
        item in util.THALAMUS_AREA for item in areas_category]] = 4
    replace_dict = dict(zip(areas_category, areas_category_values))
    areas_labels = areas_string.replace(regex=replace_dict)

    # Probes labels. Top. 
    probes_string = units['probe_description']
    probes_category = probes_string.unique()
    probes_category_values = np.arange(len(probes_category)) % 2
    replace_dict = dict(zip(probes_category, probes_category_values))
    probes_labels = probes_string.replace(regex=replace_dict)

    # Preparing written labels in the plot
    areas_label_x = [np.where(areas_string.values == item)[0][0]
                     for item in areas_category]
    vis_label_x = [np.where(areas_string.values == item)[0][0]
                   for item in util.VISUAL_AREA if item in areas_category]
    vis_label = [item for item in util.VISUAL_AREA if item in areas_category]
    probes_label_x = [np.where(probes_string.values == item)[0][0]
                      for item in probes_category]
    probes_category_label = [x.replace('probe', '') for x in probes_category]

    ################################### LEFT ###################################
    # The order is the same as the color `color_list`.
    areas_string = units_left['ecephys_structure_acronym']
    areas_category = areas_string.unique()
    areas_category_values = np.zeros(len(areas_category))
    areas_category_values[[
        item in util.VISUAL_AREA for item in areas_category]] = 1
    areas_category_values[[
        item in util.HIPPOCAMPUS_AREA for item in areas_category]] = 2
    areas_category_values[[
        item in util.MIDBRAIN for item in areas_category]] = 3
    areas_category_values[[
        item in util.THALAMUS_AREA for item in areas_category]] = 4
    replace_dict = dict(zip(areas_category, areas_category_values))
    areas_labels_left = areas_string.replace(regex=replace_dict)

    # Probes labels. Top. 
    probes_string = units_left['probe_description']
    probes_category_left = probes_string.unique()
    probes_category_values = np.arange(len(probes_category_left)) % 2
    replace_dict = dict(zip(probes_category_left, probes_category_values))
    probes_labels_left = probes_string.replace(regex=replace_dict)

    # Preparing written labels in the plot
    areas_label_y = [np.where(areas_string.values == item)[0][0]
                     for item in areas_category]
    vis_label_y = [np.where(areas_string.values == item)[0][0]
                   for item in util.VISUAL_AREA if item in areas_category]
    vis_label_left = [item for item in util.VISUAL_AREA if item in areas_category]
    probes_label_y = [np.where(probes_string.values == item)[0][0]
                      for item in probes_category_left]
    probes_category_label_left = [
        x.replace('probe', '') for x in probes_category_left]

    # Colormap for the areas.
    from_list = matplotlib.colors.LinearSegmentedColormap.from_list
    color_list = [0] * 5
    color_list[0] = util.color_by_brain_area('other', 'rgby')
    color_list[1] = util.color_by_brain_area('VIS', 'rgby')
    color_list[2] = util.color_by_brain_area('HPF', 'rgby')
    color_list[3] = util.color_by_brain_area('MB', 'rgby')
    color_list[4] = util.color_by_brain_area('TH', 'rgby')
    areas_cmap = from_list(None, color_list, num_areas)

    # Begin the figure.
    fig = plt.figure(figsize=(12, 10))
    cbar_ax = fig.add_axes([.91, .2, .015, .25])
    areas_legend_ax = fig.add_axes([.91, .5, .015, .25])
    gs = gridspec.GridSpec(3, 3,
                           width_ratios=[0.5, 0.5, 16], 
                           height_ratios=[0.5, 0.5, 16], wspace=0.0, hspace=0.0)

    ################################### TOP ####################################
    #Top areas labels
    ax2 = plt.subplot(gs[2])
    seaborn.heatmap(
        areas_labels.values.reshape(1,-1),
        vmin=-0.5, vmax=num_areas-0.5,
        cmap=areas_cmap, cbar=True, cbar_ax=areas_legend_ax)
    ax2.set_ylabel('')
    ax2.set_xlabel('')
    plt.tick_params(top=False, bottom=False, left=False, right=False,
                    labelleft=False, labelbottom=False)
    colorbar = ax2.collections[0].colorbar
    colorbar.set_ticks([0, 1, 2, 3, 4])
    colorbar.set_ticklabels(areas_names)
    colorbar.ax.tick_params(labelsize=6)
    plt.text(1.01, 0, 'Brain areas', transform = ax2.transAxes)
    for i, probe in enumerate(vis_label):
      plt.text(vis_label_x[i], -0.5, vis_label[i])

    #Top probes labels
    ax5 = plt.subplot(gs[5])
    seaborn.heatmap(
        probes_labels.values.reshape(1,-1),
        vmin=-1, vmax=2, cmap=cm.Greys, cbar=False)
    ax5.set_ylabel('')
    ax5.set_xlabel('')
    plt.tick_params(top=False, bottom=False, left=False, right=False,
                    labelleft=False, labelbottom=False)
    for i, probe in enumerate(probes_category):
      plt.text(probes_label_x[i]+0.5, 0.5, probes_category_label[i],
               color='k' if i % 2 ==0 else 'w')
    plt.text(1.01, 0, 'Probes', transform = ax5.transAxes)

    ################################### LEFT ###################################
    # Left areas labels
    ax6 = plt.subplot(gs[6])
    seaborn.heatmap(areas_labels_left.values.reshape(-1,1),
                    vmin=-0.5, vmax=num_areas-0.5,
                    cmap=areas_cmap, cbar=False)
    ax6.set_ylabel('')
    ax6.set_xlabel('')
    plt.tick_params(top=False, bottom=False, left=False, right=False,
                    labelleft=False, labelbottom=False)
    # for i in range(len(areas_category)):
    #     plt.text(-2, areas_label_x[i]+1, areas_category[i], weight=20)
    for i, probe in enumerate(vis_label):
      plt.text(-0.7, vis_label_y[i], vis_label_left[i],
      rotation=-90, rotation_mode='anchor')

    # Left probes labels.
    ax7 = plt.subplot(gs[7])
    seaborn.heatmap(probes_labels_left.values.reshape(-1,1),
                vmin=-1, vmax=2, cmap=cm.Greys, cbar=False)
    ax7.set_ylabel('')
    ax7.set_xlabel('')
    plt.tick_params(top=False, bottom=False, left=False, right=False,
                    labelleft=False, labelbottom=False)
    for i in range(len(probes_category_left)):
        plt.text(0, probes_label_y[i], probes_category_label_left[i], 
                 color='k' if i % 2 ==0 else 'w',
                 rotation=-90, rotation_mode='anchor')

    # Main figure
    # corr_cmap = seaborn.diverging_palette(220, 10, as_cmap=True)
    corr_cmap = cm.RdYlGn
    ax = plt.subplot(gs[8])
    seaborn.heatmap(
        matrix,
        vmin=-0.6, vmax=0.6,
        cmap=corr_cmap, cbar=True, cbar_ax=cbar_ax)
    ax.hlines(probes_label_x, *ax.get_xlim(), color='blue', lw=1.2)
    ax.vlines(probes_label_x, *ax.get_ylim(), color='blue', lw=1.2)

    plt.tick_params(top=False, bottom=False, left=False, right=False,
                    labelleft=False, labelbottom=False)
    ax.set_ylabel('')
    ax.set_xlabel('')
    fig.suptitle(fig_title, y=0.95)

    if output_figure_path:
      plt.savefig(output_figure_path)
      print('Save figure to: ', output_figure_path)
    if show_figure:
      plt.show()
    plt.close()

  def plot_between_probes_metric(
      self,
      plot_dict,
      output_figure_path=None,
      show_figure=True):
    """Plot between across areas correlation.

    Args:
      sub_group_df: It shows the correlated, uncorrelated neurons.
    """
    prob_index = {'probeA':1, 'probeB':2, 'probeC':3,
                  'probeD':4, 'probeE':5, 'probeF':6}
    probe_to_area_index = {'probeA':'VISam', 'probeB':'VISpm', 'probeC':'VISp',
                           'probeD':'VISl', 'probeE':'VISal', 'probeF':'VISrl'}

    plt.figure(figsize=(15, 15))
    for probe_from, probe_to in list(plot_dict.keys()):
      row_id = prob_index[probe_from]
      col_id = prob_index[probe_to]

      ax = plt.subplot(6, 6, (row_id - 1)* 6 + col_id)

      y = plot_dict[(probe_from, probe_to)]['corr']
      if len(y) > 1:
        x = plot_dict[(probe_from, probe_to)]['corr'].index.values
        plt.plot(x, y, label='correlated group')
      else:
        plt.bar(-0.5, y, label='correlated group')

      y = plot_dict[(probe_from, probe_to)]['uncorr']
      if len(y) > 1:
        x = plot_dict[(probe_from, probe_to)]['uncorr'].index.values
        plt.plot(x, y, label='uncorrelated group')
      else:
        plt.bar(0.5, y, label='uncorrelated group')

      plt.ylim(-0.2, 1)
      # plt.ylim(-0.5, 1)
      plt.grid()

      if row_id != 1 or col_id != 2:
        ax.tick_params(top=False, bottom=False, left=False, right=False,
                    labelleft=False, labelbottom=False)
      if row_id == 1 and col_id == 2:
        plt.legend(bbox_to_anchor=(1, -0.5))
        plt.xlabel('Time [s]')
        plt.ylabel('Correlation')

      # Top rows probe labels
      if row_id == 1:
        plt.text(0.5, 1.1, probe_to_area_index[probe_to],
                 transform=ax.transAxes, horizontalalignment='center')
      # Right columns probe labels
      if col_id % 6 == 0:
        plt.text(1.1, 0.5, probe_to_area_index[probe_from],
                 transform = ax.transAxes, rotation=-90,
                 verticalalignment ='center')

    if output_figure_path:
      plt.savefig(output_figure_path)
      print('Save figure to: ', output_figure_path)
    if show_figure:
      plt.show()
    plt.close()