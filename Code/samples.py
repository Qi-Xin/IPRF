"""Data models."""
import os
import sys

import collections
from collections import defaultdict
from glob import glob
import io
import itertools
import numpy as np
import matplotlib
import matplotlib.pylab
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import networkx as nx
import pandas as pd
import pickle
import seaborn
import scipy
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

import smoothing_spline
import util
import hierarchical_sampling_model as hsm


class Samples(object):
  """Sample collector."""

  def __init__(
      self,
      num_areas=0,
      num_conditions=0,
      num_groups=0,
      num_trials=0,
      num_qs=0,
      session_id=0,
      probes=0):
    # Setup the samples collector.
    self.clear_samples_memory()

    self.num_areas = num_areas
    self.num_conditions = num_conditions
    self.num_groups = num_groups
    self.num_trials = num_trials
    self.num_qs = num_qs
    self.session_id = session_id
    self.probes = probes

    self.q_sample_accept_cnt_rc = np.zeros([num_trials, num_conditions])
    self.q_sample_cnt_rc = np.zeros([num_trials, num_conditions])
    self.f_sample_accept_cnt = 0
    self.f_sample_cnt = 0
    self.sub_group_df_c = None

  def clear_samples_memory(self):
    self.f_pop = []
    self.f_pop_beta = []
    self.q = []
    self.q_shift1 = []
    self.q_shift2 = []
    self.f_warp_sources = []
    self.f_warp_targets = []
    self.mu_cross_pop = []
    self.sigma_cross_pop = []
    self.rho_cross_pop_simple = []
    self.z = []
    self.p = []
    self.log_likelihood = []


  @property
  def memory_attributes(self):
    self._memory_attributes = [
        'f_pop', 'f_pop_beta', 'q', 'q_shift1', 'q_shift2',
        'f_warp_sources', 'f_warp_targets', 'mu_cross_pop', 'sigma_cross_pop',
        'rho_cross_pop_simple', 'z', 'p', 'log_likelihood']
    return self._memory_attributes


  @property
  def memory_parameters(self):
    self._memory_parameters = ['num_areas', 'num_conditions', 'num_groups',
        'num_qs', 'session_id', 'probes', 'q_sample_accept_cnt', 'q_sample_cnt',
        'sub_group_df_c', 'f_sample_accept_cnt', 'f_sample_cnt',
        'q_sample_accept_cnt_rc', 'q_sample_cnt_rc']

    return self._memory_parameters


  def save(self, file_path, verbose=False):
    """Saves the memory to file."""
    import pickle
    with open(file_path, 'wb') as f:
      pickle.dump(self, f)
      if verbose:
        print('Save file to:', file_path)


  def load(self, file_path, verbose=False):
    """Loads the memory from file."""
    with open(file_path, 'rb') as f:
      new_samples = pickle.load(f)
      # Copy everything from the loaded structure to myself.
      self.__dict__.update(new_samples.__dict__)
    if verbose:
      print('Load samples:', file_path)


  def load_batches(
      self,
      batches_dir,
      start_id=0,
      end_id=20,
      thin_step=0,
      reset=True,
      verbose=False):
    """Load samples in batches.

    Args:
      start_id:
      end_id: This has to be specified (maybe a large value).
    """
    prefix = '*_samples_batch*'
    batches_files = glob(os.path.join(batches_dir, prefix))
    print(f'Find {len(batches_files)} batches.')
    batch_ids = np.arange(start_id, end_id+1)

    if reset:
      self.clear_samples_memory()

    # for file_path in batches_files:
    for batch_id in batch_ids:
      # if batch_id < start_id or batch_id > batch_ids[end_id]:
      #   continue
      prefix = f'*_samples_batch{batch_id}_*'
      file_path = glob(os.path.join(batches_dir, prefix))
      if len(file_path) == 0:
        continue
      else:
        file_path = file_path[0]
      print(file_path)

      with open(file_path, 'rb') as f:
        new_samples = pickle.load(f)

      for key, item in new_samples.__dict__.items():
        if key in self.memory_attributes:
          setattr(self, key, getattr(self, key) + item[::thin_step])
        if key in self.memory_parameters:
          setattr(self, key, item)

    self.peek()


  def peek(self):
    """Get the samples sizes."""
    print_token = ''
    for attribute in self.memory_attributes:
      print_token += f'{attribute}:{len(getattr(self, attribute))}  '
    print(print_token)


  def plot_log_likelihood(self, head=100, tail=150):
    """Plots the trace of the log-likelihood."""
    iterations = np.arange(len(self.log_likelihood))

    gs_kw = dict(width_ratios=[1,1,1])
    fig, axs = plt.subplots(figsize=(18, 3), gridspec_kw=gs_kw, nrows=1, ncols=3)
    ax = fig.add_subplot(axs[0])
    plt.plot(iterations, self.log_likelihood)
    plt.xlabel('iteration')
    plt.ylabel('log-likelihood')
    plt.grid('on')

    ax = fig.add_subplot(axs[1])
    plt.plot(iterations[:head], self.log_likelihood[:head])
    plt.xlabel('iteration')
    plt.grid('on')

    ax = fig.add_subplot(axs[2])
    plt.plot(iterations[tail:], self.log_likelihood[tail:])
    plt.xlabel('iteration')
    plt.grid('on')
    plt.show()


  def plot_f_pop_CI_demo(
      self,
      c,
      a,
      spike_train_time_line,
      burn_in=0,
      end=None,
      step=1,
      dt=0.002,
      output_dir=None,
      show_plot=True):
    """Plots all f_pop together."""
    areas_names = ['V1', 'LM', 'AL']
    g = 0
    f_samples = np.stack(self.f_pop[burn_in:end:step], axis=0)
    print('f_samples.shape:', f_samples.shape)
    # print('f_pop accept ratio:', self.f_sample_cnt,
    #       self.f_sample_accept_cnt / self.f_sample_cnt)

    num_samples, num_conditions, num_areas, num_groups, num_bins = f_samples.shape
    colors = matplotlib.pylab.cm.jet(np.linspace(0, 1, num_samples))
    timeline = spike_train_time_line * 1000

    plt.figure()
    ax = plt.gca()

    f_pop = f_samples[:,c,a,g];
    f_pop = np.exp(f_pop) / dt
    # f_pop_center = np.mean(f_pop, axis=0)
    f_pop_center = np.quantile(f_pop, 0.5, axis=0)
    f_pop_up = np.quantile(f_pop, 0.975, axis=0)
    f_pop_dn = np.quantile(f_pop, 0.025, axis=0)
    plt.fill_between(timeline, f_pop_up, f_pop_dn,
                     facecolor='tab:grey', alpha=0.3, label='95% CI')
    plt.plot(timeline, f_pop_center, 'k', label='median')

    # ax.set_yticklabels([])
    # ax.set_xticklabels([])
    # ax.axis('off')

    # fig.tight_layout()
    if output_dir is not None:
      if hasattr(self, 'session_id'):
        session_id = self.session_id
      else:
        session_id = 0
      file_path = os.path.join(output_dir, f'{session_id}_f_pop.pdf')
      plt.savefig(file_path)

    if show_plot:
      plt.show()
    else:
      plt.close()

    return spike_train_time_line, f_pop_center, f_pop_dn, f_pop_up


  def plot_f_pop_CI(
      self,
      clist,
      spike_train_time_line,
      burn_in=0,
      end=None,
      step=1,
      dt=0.002,
      condition_ids=None,
      f_pop_cag_ref=None,
      f_peak1_ac_ref=None,
      f_peak2_ac_ref=None,
      show_legend=True,
      output_dir=None):
    """Plots all f_pop together."""
    areas_names = ['V1', 'LM', 'AL']
    f_samples = np.stack(self.f_pop[burn_in:end:step], axis=0)
    print('f_samples.shape:', f_samples.shape)
    # print('f_pop accept ratio:', self.f_sample_cnt,
    #       self.f_sample_accept_cnt / self.f_sample_cnt)

    num_samples, num_conditions, num_areas, num_groups, num_bins = f_samples.shape
    colors = matplotlib.pylab.cm.jet(np.linspace(0, 1, num_samples))
    timeline = spike_train_time_line * 1000

    num_block_rows = np.ceil(len(clist)/3).astype(int)
    num_block_cols = 3
    row_size = num_areas + 1
    col_size = num_groups + 1
    gs_kw = dict(width_ratios=[0.06,1,1,0.3] * num_block_cols,
                 height_ratios=[0.4,1,1,1] * num_block_rows)
    fig, axs = plt.subplots(figsize=(20, len(clist)*1.3), gridspec_kw=gs_kw,
        nrows=row_size * num_block_rows, ncols=col_size * num_block_cols)
    plt.subplots_adjust(left=0, right=0.99, top=1, bottom=0.03, hspace=0.15, wspace=0.3)

    for c_id, c in enumerate(clist):
      for a_id in range(num_areas+1):
        for g_id in range(num_groups+1):
          a = a_id - 1
          g = g_id - 1
          row = c_id // num_block_cols * row_size + a_id
          col = c_id % num_block_cols * col_size + g_id

          ax = fig.add_subplot(axs[row, col])
          # Set invisible at the margins.
          if a_id == 0 or g_id == 0:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.axis('off')
            continue

          f_pop = f_samples[:,c,a,g];
          f_pop = np.exp(f_pop) / dt

          if g != 2:
            # f_pop_mean = np.mean(f_pop, axis=0)
            f_pop_mean = np.quantile(f_pop, 0.5, axis=0)
            f_pop_up = np.quantile(f_pop, 0.975, axis=0)
            f_pop_dn = np.quantile(f_pop, 0.025, axis=0)
            plt.fill_between(timeline, f_pop_up, f_pop_dn,
                             facecolor='tab:grey', alpha=0.3, label='95% CI')
            plt.plot(timeline, f_pop_mean, 'k', label='median')
          elif g == 2:
            f_pop = f_pop.mean(axis=1)
            # f_pop_mean = np.mean(f_pop)
            f_pop_mean = np.quantile(f_pop, 0.5)
            f_pop_up = np.quantile(f_pop, 0.975) - f_pop_mean
            f_pop_dn = f_pop_mean - np.quantile(f_pop, 0.025)
            plt.errorbar([0], [f_pop_mean], yerr=[[f_pop_dn], [f_pop_up]],
                         fmt='+k', capsize=5, label='95% CI')

          if f_pop_cag_ref is not None:
            plt.plot(timeline, np.exp(f_pop_cag_ref[c,a,g]) / dt, color='g')
          if g == 0 and f_peak1_ac_ref is not None:
            plt.axvline(x=f_peak1_ac_ref[a,c] * 1000, linestyle='--', color='g')
          if g == 0 and f_peak2_ac_ref is not None:
            plt.axvline(x=f_peak2_ac_ref[a,c] * 1000, linestyle='--', color='g')
          if g == 2:
            pass
            # plt.ylim(0, np.max(np.exp(f_samples[:,c,a,1]) / dt))
            plt.ylim(f_pop_mean-0.1, f_pop_mean+0.1)
          # if g == 0:
          #   plt.ylim(0, 70)
          # elif g == 1:
          #   plt.ylim(0, 15)
          # elif g == 2:
          #   plt.ylim(0.2, 0.8)

          # plt.grid('on')
          # plt.title(f'{areas_names[a]}  g:{g}')
          if g == 0:
            plt.text(0.68, 0.83, f'{areas_names[a]}  pop',
              color='k', size=8, transform=ax.transAxes)
          elif g == 1:
            plt.text(0.68, 0.83, f'{areas_names[a]}  local-1',
              color='k', size=8, transform=ax.transAxes)
          elif g == 2:
            plt.text(0.06, 0.83, f'{areas_names[a]}  local-2',
              color='k', size=8, transform=ax.transAxes)

          if a == 0 and g == 0 and condition_ids is not None:
            plt.text(0.25, 1.1, f'condition={condition_ids[c]}',
                color='k', size=13, transform=ax.transAxes)
          elif  a == 0 and g == 0 and condition_ids is None:
            plt.text(0.25, 1.1, f'condition={c}',
                color='k', size=13, transform=ax.transAxes)

          if (c_id != len(clist)-num_block_cols) or a != num_areas - 1 or g != 0:
            ax.set_xticklabels([])
          if (show_legend and c_id == row_size*col_size-num_block_cols-1 and
              a == num_areas - 1 and g == 0):
            plt.xlabel('Time [ms]', fontsize=13)
            # plt.ylabel('Firing rate [spikes/sec]')
            fig.text(0.002, 0.05, 'Firing rate [spikes/sec]', 
                     va='bottom', rotation='vertical', fontsize=13)
          else:
            ax.set_xticklabels([])

          if (c_id == row_size*col_size-num_block_cols-1 and
              a == 2 and g == 1 and show_legend):
            plt.legend(bbox_to_anchor=(-0.1, -0.6), loc='lower left', ncol=3)
          if (c_id == row_size*col_size-num_block_cols-1 and
              a == 2 and g == 2 and show_legend):
            plt.legend(bbox_to_anchor=(-0.25, -0.6), loc='lower left', ncol=2)

    for c_id in range(len(clist), num_block_rows*num_block_cols-1):
      for a_id in range(num_areas+1):
        for g_id in range(num_groups+1):
          row = c_id // num_block_cols * row_size + a_id
          col = c_id % num_block_cols * col_size + g_id
          ax = fig.add_subplot(axs[row, col])
          plt.axis('off')

    if output_dir is not None:
      if hasattr(self, 'session_id'):
        session_id = self.session_id
      else:
        session_id = 0
      file_path = os.path.join(output_dir, f'{session_id}_f_pop.pdf')
      plt.savefig(file_path, bbox_inches='tight')
      print('Save figure:', file_path)
    plt.show()


  def plot_f_pop_overlap(
      self,
      clist,
      spike_train_time_line,
      burn_in=0,
      end=None,
      step=1,
      dt=0.002,
      output_dir=None):
    """Plots all f_pop together."""
    areas_names = ['V1', 'LM', 'AL']
    f_samples = np.stack(self.f_pop[burn_in:end:step], axis=0)
    print('f_samples.shape:', f_samples.shape)
    # print('f_pop accept ratio:', self.f_sample_cnt,
    #       self.f_sample_accept_cnt / self.f_sample_cnt)

    num_samples, num_conditions, num_areas, num_groups, num_bins = f_samples.shape
    colors = matplotlib.pylab.cm.jet(np.linspace(0, 1, num_samples))
    timeline = spike_train_time_line * 1000

    row_size = num_areas
    col_size = 2
    gs_kw = dict(width_ratios=[1]*col_size, height_ratios=[1]*num_areas)
    fig, axs = plt.subplots(figsize=(8, 5), gridspec_kw=gs_kw,
        nrows=row_size, ncols=col_size)
    plt.subplots_adjust(left=0, right=0.99, top=1, bottom=0.03, hspace=0.15,
        wspace=0.1)

    for a in range(num_areas):
      for g in range(col_size):
        ax = fig.add_subplot(axs[a, g])
        for c_id, c in enumerate(clist):
          f_pop = f_samples[:,c,a,g];
          f_pop = np.exp(f_pop) / dt
          # f_pop_mean = np.mean(f_pop, axis=0)
          f_pop_mean = np.quantile(f_pop, 0.5, axis=0)
          f_pop_up = np.quantile(f_pop, 0.975, axis=0)
          f_pop_dn = np.quantile(f_pop, 0.025, axis=0)
          # plt.fill_between(timeline, f_pop_up, f_pop_dn,
          #                  facecolor='tab:grey', alpha=0.3, label='95% CI')
          plt.plot(timeline, f_pop_mean, 'k', lw=1)
          if g == 0:
            plt.text(0.8, 0.85, f'{areas_names[a]} pop',
                  color='k', size=13, transform=ax.transAxes)
          elif g == 1:
            plt.text(0.75, 0.85, f'{areas_names[a]} local-1',
                  color='k', size=13, transform=ax.transAxes)

          if a == 2:
            plt.xlabel('Time [ms]')
            if g == 0:
              plt.ylabel('Firing rate [spikes/sec]')
          else:
            ax.set_xticklabels([])

    if output_dir is not None:
      if hasattr(self, 'session_id'):
        session_id = self.session_id
      else:
        session_id = 0
      file_path = os.path.join(output_dir, f'{session_id}_f_pop_overlap.pdf')
      plt.savefig(file_path, bbox_inches='tight')
      print('Save figure:', file_path)
    plt.show()


  def plot_f_pop_CI_g0(
      self,
      clist,
      spike_train_time_line,
      burn_in=0,
      end=None,
      step=1,
      dt=0.002,
      condition_ids=None,
      f_pop_cag_ref=None,
      f_peak1_ac_ref=None,
      f_peak2_ac_ref=None,
      output_dir=None):
    """Plots all f_pop together."""
    areas_names = ['V1', 'LM', 'AL']
    f_samples = np.stack(self.f_pop[burn_in:end:step], axis=0)
    print('f_samples.shape:', f_samples.shape)

    num_samples, num_conditions, num_areas, num_groups, num_bins = f_samples.shape
    colors = matplotlib.pylab.cm.jet(np.linspace(0, 1, num_samples))
    timeline = spike_train_time_line * 1000

    num_block_rows = len(clist)
    num_block_cols = 1
    row_size = 1
    col_size = num_areas + 1
    gs_kw = dict(width_ratios=[0.06,1,1,1],
                 height_ratios=[1] * num_block_rows)
    fig, axs = plt.subplots(figsize=(8, len(clist)*1.3), gridspec_kw=gs_kw,
        nrows=row_size * num_block_rows, ncols=col_size * num_block_cols)
    plt.subplots_adjust(left=0, right=0.99, top=1, bottom=0.03, hspace=0.2, wspace=0.3)

    for c_id, c in enumerate(clist):
      for a_id in range(num_areas+1):
        g_id = 1
        g = g_id - 1
        a = a_id - 1
        col = a_id
        row = c_id

        ax = fig.add_subplot(axs[row, col])
        # Set invisible at the margins.
        if a_id == 0:
          ax.set_yticklabels([])
          ax.set_xticklabels([])
          ax.axis('off')
          continue

        f_pop = f_samples[:,c,a,g];
        f_pop = np.exp(f_pop) / dt

        # f_pop_mean = np.mean(f_pop, axis=0)
        f_pop_mean = np.quantile(f_pop, 0.5, axis=0)
        f_pop_up = np.quantile(f_pop, 0.975, axis=0)
        f_pop_dn = np.quantile(f_pop, 0.025, axis=0)
        plt.fill_between(timeline, f_pop_up, f_pop_dn,
                         facecolor='tab:grey', alpha=0.3, label='95% CI')
        plt.plot(timeline, f_pop_mean, 'k', label='median')

        # plt.text(0.85, 0.83, f'{areas_names[a]}',
        #   color='k', size=10, transform=ax.transAxes)
        if c_id == 0:
          plt.title(f'{areas_names[a]}', fontsize=14)

        if a == 0 and g == 0 and condition_ids is not None:
          plt.text(0.6, 0.85, f'condition={condition_ids[c]}',
              color='k', size=8, transform=ax.transAxes)
        elif a == 0 and g == 0 and condition_ids is None:
          plt.text(0.88, 0.85, f'condition={c}',
              color='k', size=13, transform=ax.transAxes)

        if c_id == len(clist)-1 and a == 0:
          plt.xlabel('Time [ms]', fontsize=13)
          fig.text(0.005, 0.25, 'Firing rate [spikes/sec]', 
                   va='bottom', rotation='vertical', fontsize=13)
        elif c_id != len(clist)-1:
          ax.set_xticklabels([])

        if c_id == len(clist)-1 and a == 1:
          plt.legend(loc=(-0.1, -0.5), ncol=3)

    if output_dir is not None:
      if hasattr(self, 'session_id'):
        session_id = self.session_id
      else:
        session_id = 0
      file_path = os.path.join(output_dir, f'{session_id}_f_pop_2c_g0.pdf')
      plt.savefig(file_path, bbox_inches='tight')
      print('Save fig:', file_path)
    else:
      plt.show()


  def plot_f_pop_rainbow(
      self,
      clist,
      spike_train_time_line,
      burn_in=0,
      end=None,
      step=1,
      dt=0.002,
      condition_ids=None,
      f_pop_cag_ref=None,
      f_peak1_ac_ref=None,
      f_peak2_ac_ref=None,
      ylim=None):
    """Plots all f_pop together."""
    f_samples = np.stack(self.f_pop[burn_in:end:step], axis=0)
    print('f_samples.shape:', f_samples.shape)
    if self.f_sample_cnt != 0:
      print('accept ratio:', self.f_sample_accept_cnt / self.f_sample_cnt,
          self.f_sample_accept_cnt, self.f_sample_cnt)

    num_samples, num_conditions, num_areas, num_groups, num_bins = f_samples.shape
    colors = matplotlib.pylab.cm.jet(np.linspace(0, 1, num_samples))
    timeline = spike_train_time_line * 1000

    for c in clist:
      gs_kw = dict(width_ratios=[1]*num_groups, height_ratios=[1]*num_areas)
      fig, axs = plt.subplots(figsize=(12, 4.5), gridspec_kw=gs_kw,
          nrows=num_areas, ncols=num_groups)
      plt.subplots_adjust(left=None, right=None, hspace=0.1, wspace=0.2)

      for a in range(num_areas):
        for g in range(num_groups):
          ax = fig.add_subplot(axs[a,g])
          f_pop = f_samples[:,c,a,g];
          f_pop = np.exp(f_pop) / dt

          for i in range(num_samples):
            plt.plot(timeline, f_pop[i], color=colors[i], lw=0.5)
          if f_pop_cag_ref is not None:
            plt.plot(timeline, np.exp(f_pop_cag_ref[c,a,g]) / dt, color='g')

          plt.text(0.7, 0.9, f'a={a}  g={g}', fontsize=8, transform=ax.transAxes)
          plt.grid('on')
          if g == 0 and f_peak1_ac_ref is not None:
            plt.axvline(x=f_peak1_ac_ref[a,c] * 1000, linestyle='--', color='g')
          if g == 0 and f_peak2_ac_ref is not None:
            plt.axvline(x=f_peak2_ac_ref[a,c] * 1000, linestyle='--', color='g')
          if g == 2:
            plt.ylim(0, 3)

          if a == 0 and g ==0 and condition_ids is not None:
            plt.title(f'c {c}  {condition_ids[c]}', fontsize=14)
          elif a == 0 and g ==0:
            plt.title(f'c:{c}', fontsize=14)
          if a != num_areas - 1:
            ax.tick_params(labelbottom=False)
      plt.show()


  def plot_f_pop_baseline(
      self,
      clist,
      burn_in=0,
      end=None,
      step=1):
    """Plot the beta baseline."""
    areas_names = ['V1', 'LM', 'AL']
    sub_samples = self.f_pop_beta[burn_in:end:step]
    num_samples = len(sub_samples)
    num_conditions = len(clist)
    num_areas = len(self.probes)
    g = 0

    betas_cas = np.zeros([num_conditions, num_areas, num_samples])
    for i in range(num_samples):
      item = self.f_pop_beta[i]
      for c in clist:
        for a in range(num_areas):
          betas_cas[c,a,i] = item[(c,a,g,'baseline')]

    gs_kw = dict(width_ratios=[1]*num_areas, height_ratios=[1])
    fig, axs = plt.subplots(figsize=(16, 3), gridspec_kw=gs_kw,
        nrows=1, ncols=num_areas)
    plt.subplots_adjust(hspace=0.2)
    for a in range(num_areas):
      ax = fig.add_subplot(axs[a])
      plt.plot(betas_cas[:,a,:].T, 'k', lw=1)
      plt.title(f'{areas_names[a]}', fontsize=14)
    plt.show()



  def clear_q_sample_accept_cnt(
      self,
      num_trials=None,
      num_conditions=None):
    if num_trials is not None and num_conditions is not None:
      self.q_sample_accept_cnt_rc = np.zeros([num_trials, num_conditions])+1
      self.q_sample_cnt_rc = np.zeros([num_trials, num_conditions])+1
    else:
      self.q_sample_accept_cnt_rc = np.zeros_like(self.q_sample_accept_cnt_rc)+1
      self.q_sample_cnt_rc = np.zeros_like(self.q_sample_cnt_rc)+1


  @property
  def q_accept_ratio_mean(self):
    if (self.q_sample_cnt_rc > 0).any():
      return np.nanmean(self.q_sample_accept_cnt_rc / self.q_sample_cnt_rc)
    else:
      return 0


  @property
  def q_accept_ratio(self):
    if (self.q_sample_cnt_rc > 0).any():
      return self.q_sample_accept_cnt_rc / self.q_sample_cnt_rc
    else:
      return 0


  @property
  def f_cross_pop_accept_ratio(self):
    if self.f_sample_cnt > 0:
      return self.f_sample_accept_cnt / self.f_sample_cnt
    else:
      return 0


  def plot_q_arc(
      self,
      clist,
      option='q',
      burn_in=50,
      end=None,
      step=1,
      q_arc_ref=None,
      q_shift1_arc_ref=None,
      q_shift2_arc_ref=None,
      ylim=None,
      show_trace_mean=False,
      save_fig=False):
    """Plots q samples."""
    if option == 'q':
      q_samples = np.stack(self.q, axis=0)
      ylim = [-1, 1] if ylim is None else ylim
      if q_arc_ref is not None:
        q_true = q_arc_ref
    elif option == 'q_shift1':
      q_samples = np.stack(self.q_shift1, axis=0)
      ylim = [-0.05, 0.05] if ylim is None else ylim
      if q_shift1_arc_ref is not None:
        q_true = q_shift1_arc_ref
    elif option == 'q_shift2':
      q_samples = np.stack(self.q_shift2, axis=0)
      ylim = [-0.1, 0.1] if ylim is None else ylim
      if q_shift2_arc_ref is not None:
        q_true = q_shift2_arc_ref

    q_hat = q_samples[burn_in:end:step].mean(axis=0)
    num_areas, _, _ = q_hat.shape
    q_hat = q_hat.transpose(0,2,1).reshape(num_areas, -1)
    corr, p_value = scipy.stats.pearsonr(q_hat[0], q_hat[1])
    print(f'Estimated corr:{corr:.3f}  p-value:{p_value:.3e}')
    print('Accept ratio:', self.q_accept_ratio_mean,
          'min:', np.min(self.q_accept_ratio), 'max:', np.max(self.q_accept_ratio))

    num_samples, num_areas, num_qs, num_conditions = q_samples.shape
    num_rows, num_cols  = np.ceil(num_qs/8).astype(int), 8
    print('num_qs:', num_qs)
    print('q_samples.shape: ', q_samples.shape)

    colors = matplotlib.pylab.cm.jet(np.linspace(0, 1, len(clist)))
    for a in range(num_areas):
      # slicing has weird behavior, clist comes out first somehow.
      q_mean_s = np.mean(q_samples[burn_in:,a,:,clist],axis=(1,2))
      q_std_s = np.std(q_samples[burn_in:,a,:,clist], axis=(1,2))

      if show_trace_mean:
        print('sub q shape:', q_samples[burn_in:,a,:,clist].shape)
        print(f'Area {a}\n' +
            f'mean {q_mean_s}\nstd  {q_std_s}\nmean/std {q_mean_s/q_std_s}')
        plt.figure(figsize=[6,1.6])
        plt.plot(np.mean(q_samples[burn_in:None:10,a,:,clist], axis=2).T)
        plt.axhline(0, color='lightgrey', lw=1)
        plt.title('Trace of trials mean (per 10 step)')

      gs_kw = dict(width_ratios=[1]*num_cols, height_ratios=[1]*num_rows)
      fig, axs = plt.subplots(figsize=(18, num_rows*1.2), gridspec_kw=gs_kw,
          nrows=num_rows, ncols=num_cols)
      plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
      axs_array = axs.reshape(-1)

      for r in range(num_qs):
        ax = fig.add_subplot(axs_array[r])
        q = q_samples[burn_in:end:step,a,r,clist]
        plt.axhline(0, color='lightgrey', lw=1)
        for c_id, c in enumerate(clist):
          plt.plot(q[...,c_id], lw=0.8, c=colors[c_id])
        plt.ylim(ylim)
        # plt.grid(axis='y')
        plt.text(0.05, 0.85, f'# {r}  median {np.median(q):.3f}',
            color='k', size=8, transform=ax.transAxes)
        if (q_arc_ref is not None or
            q_shift1_arc_ref is not None or
            q_shift2_arc_ref is not None):
          plt.axhline(y=q_true[a,r,clist],
                      linestyle=':', color='g')
        if r != 8:
          ax.set_xticklabels([])
          ax.set_yticklabels([])
        else:
          ax.xaxis.set_tick_params(labelsize=7, rotation=-45)
          ax.yaxis.set_tick_params(labelsize=8)

      # Clean up the empty figures.
      for r in range(num_qs, num_rows*num_cols):
        ax = fig.add_subplot(axs_array[r])
        plt.axis('off')

      plt.show()
      if save_fig:
        output_figure_path = os.path.join(f'samples_{option}_area_1.pdf')
        plt.savefig(output_figure_path)
        plt.close()
        print('Save figure to: ', output_figure_path)


  def plot_q_arc_distribution(
      self,
      clist,
      burn_in=50,
      end=None,
      step=1,
      option='q',
      q_arc_ref=None,
      q_shift1_arc_ref=None,
      q_shift2_arc_ref=None,
      save_fig=False):
    """Plots q samples."""
    if option == 'q':
      q_samples = np.stack(self.q[burn_in:end:step], axis=0)
      xlim = [-1, 1]
      if q_arc_ref is not None:
        q_true = q_arc_ref
    elif option == 'q_shift1':
      q_samples = np.stack(self.q_shift1[burn_in:end:step], axis=0)
      xlim = [-0.08, 0.08]
      if q_shift1_arc_ref is not None:
        q_true = q_shift1_arc_ref
    elif option == 'q_shift2':
      q_samples = np.stack(self.q_shift2[burn_in:end:step], axis=0)
      xlim = [-0.12, 0.12]
      if q_shift2_arc_ref is not None:
        q_true = q_shift2_arc_ref

    # q_hat = q_samples[burn_in:end:step].mean(axis=0)
    # num_areas, _, _ = q_hat.shape
    # q_hat = q_hat.transpose(0,2,1).reshape(num_areas, -1)
    # corr, p_value = scipy.stats.pearsonr(q_hat[0], q_hat[1])
    # print(f'Estimated corr:{corr:.3f}  p-value:{p_value:.3e}')
    # print('Accept ratio:', self.q_sample_accept_cnt,
    #       '   Number of samples (for each trial):', self.q_sample_cnt)

    num_samples, num_areas, num_trials, num_conditions = q_samples.shape
    print('before q_samples.shape: ', q_samples.shape)
    q_samples = q_samples.transpose(1,2,3,0)
    print('after q_samples.shape: ', q_samples.shape)

    # Across all conditions.
    plt.figure(figsize=(3 * num_areas, 2))
    for a in range(num_areas):
      ax = plt.subplot(1, num_areas, a+1)
      data = q_samples[a,:,:,:].reshape(-1)
      CI_left = np.quantile(data, 0.025)
      CI_right = np.quantile(data, 0.975)
      median = np.quantile(data, 0.5)
      x, y = seaborn.distplot(
          data, bins=30, color='tab:gray').get_lines()[0].get_data()
      plt.axvline(x=CI_left, linestyle=':', color='k')
      plt.axvline(x=CI_right, linestyle=':', color='k', label='%95 CI')
      plt.axvline(x=median, color='g', label='mode')
      plt.xlim(xlim)
      plt.title(f'Area:{a}')
      plt.xlabel('Time [sec]')
    plt.show()

    # Across all trials.
    plt.figure(figsize=(3 * num_areas, 2))
    for a in range(num_areas):
      ax = plt.subplot(1, num_areas, a+1)
      data = q_samples[a,:,clist,:].reshape(-1)
      CI_left = np.quantile(data, 0.025)
      CI_right = np.quantile(data, 0.975)
      median = np.quantile(data, 0.5)
      x, y = seaborn.distplot(
          data, bins=30, color='tab:gray').get_lines()[0].get_data()
      plt.axvline(x=CI_left, linestyle=':', color='k')
      plt.axvline(x=CI_right, linestyle=':', color='k', label='%95 CI')
      plt.axvline(x=median, color='g', label='mode')
      plt.xlim(xlim)
      plt.title(f'Area:{a}')
      plt.xlabel('Time [sec]')
    plt.show()

    # For each condition for each trial.
    for c in clist:
      print('c:', c)
      for a in range(num_areas):
        plt.figure(figsize=(16, 3))
        for r in range(num_trials):
          data = q_samples[a,r,c,:]
          CI_left = np.quantile(data, 0.025)
          CI_right = np.quantile(data, 0.975)
          median = np.quantile(data, 0.5)
          ax = plt.subplot(2, 8, r+1)
          x, y = seaborn.distplot(
              data, bins=30, color='tab:gray').get_lines()[0].get_data()
          plt.axvline(x=CI_left, linestyle=':', color='k')
          plt.axvline(x=CI_right, linestyle=':', color='k', label='%95 CI')
          plt.axvline(x=median, color='g', label='mode')

          plt.xlim(xlim)
          plt.text(0.8, 0.85, f'r:{r}',
                   color='k', size=10, transform=ax.transAxes)
          if r != 8:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
          else:
            plt.xlabel('Time [sec]')
          if r == 0:
            plt.title(f'Area: {a} r:{r}')

        plt.tight_layout()
        plt.show()
        # if save_fig:
        #   output_figure_path = os.path.join(
        #       self.output_dir, f'samples_{option}_area_1.pdf')
        #   plt.savefig(output_figure_path)
        #   plt.close()
        #   print('Save figure to: ', output_figure_path)


  def plot_q_trials(
      self,
      clist,
      burn_in=50,
      end=None,
      step=5,
      figure_path=None):
    """Plots the gain agains the trials."""
    areas_names = ['V1', 'LM', 'AL']
    q_samples = np.stack(self.q[burn_in:end:step], axis=0)
    print('sliced sample size:', q_samples.shape)
    q_arc = np.quantile(q_samples, 0.5, axis=0)

    plt.figure(figsize=(3, 3 * self.num_conditions))
    for c in range(self.num_conditions):
      for a in range(self.num_areas):
        ax = plt.subplot(self.num_areas * self.num_conditions, 1,
                         c * self.num_areas + a + 1)
        plt.plot(q_arc[a,:,c],  color='grey' if c%2==0 else 'k')
        plt.text(1.1, 0.5, f'c={c}  {areas_names[a]}', transform=ax.transAxes)
        if a != self.num_areas-1:
          ax.set_xticklabels([])
      plt.tight_layout()
    if figure_path is not None:
      plt.savefig(figure_path)
      print('save figure:', figure_path)
    plt.show()


  def plot_q_q(
      self,
      i,j,
      burn_in=0,
      end=None,
      step=1,
      options=['q', 'q_shift1', 'q_shift2'],
      show_whole_panel=True):
    """Plots features again features."""
    num_qs = self.num_qs
    q_mean_0 = []
    q_mean_1 = []

    if 'q' in options:
      q_samples = np.stack(self.q[burn_in:end:step], axis=0)
      num_samples, num_areas, num_trials, num_conditions = q_samples.shape
      print('q shape:', num_samples, num_areas, num_trials, num_conditions)
      q_mean = q_samples.mean(axis=0)  # Posterior mean across samples.
      q_mean = q_mean.transpose(0,2,1).reshape(num_areas, -1)
      q_std = q_mean.std(axis=1).reshape(num_areas, -1)
      # q0_mean = q_mean / q_std
      q0_mean = q_mean
      q_mean_0.append(q0_mean[0])
      q_mean_1.append(q0_mean[1])
    if 'q_shift1' in options:
      q_samples = np.stack(self.q_shift1[burn_in:end:step], axis=0)
      num_samples, num_areas, num_trials, num_conditions = q_samples.shape
      print('q1 shape:', num_samples, num_areas, num_trials, num_conditions)
      q_mean = q_samples.mean(axis=0)  # Posterior mean across samples.
      q_mean = q_mean.transpose(0,2,1).reshape(num_areas, -1)
      q_std = q_mean.std(axis=1).reshape(num_areas, -1)
      # q1_mean = q_mean / q_std  # Normalize the qs.
      q1_mean = q_mean
      q_mean_0.append(q1_mean[0])
      q_mean_1.append(q1_mean[1])
    if 'q_shift2' in options:
      q_samples = np.stack(self.q_shift2[burn_in:end:step], axis=0)
      num_samples, num_areas, num_trials, num_conditions = q_samples.shape
      print('q2 shape:', num_samples, num_areas, num_trials, num_conditions)
      q_mean = q_samples.mean(axis=0)  # Posterior mean across samples.
      q_mean = q_mean.transpose(0,2,1).reshape(num_areas, -1)
      q_std = q_mean.std(axis=1).reshape(num_areas, -1)
      # q2_mean = q_mean / q_std
      q2_mean = q_mean
      q_mean_0.append(q2_mean[0])
      q_mean_1.append(q2_mean[1])

    q_mean_0 = np.array(q_mean_0)
    q_mean_1 = np.array(q_mean_1)
    corr = np.corrcoef(np.vstack((q_mean_0, q_mean_1)))

    # print(q_mean_0)
    # print(q_mean_1)
    corr, p_value = scipy.stats.pearsonr(q_mean_0[0], q_mean_1[0])
    # print('corr, p_value: ', corr, p_value)

    plt.figure(figsize=(4,4))
    ax = plt.subplot(111)
    plt.plot(q_mean_0[i]*1000, q_mean_1[j]*1000, '.')
    # plt.plot([0, 1], [0, 1], 'k:', transform=ax.transAxes)
    # plt.xlim(-0.02, 0.02); plt.ylim(-0.02, 0.02)
    plt.xlabel('V1 first peak shifting [ms]')
    plt.ylabel('AL first peak shifting [ms]')
    plt.text(0.1, 0.8, f'r={corr:.3f}  p={p_value:.3f}', transform=ax.transAxes)
    plt.show()

    if not show_whole_panel:
      return

    areas = ['V1', 'LM', 'AL']
    features = ['Gain', 'Peak-1', 'Peak-2']
    q_mean = np.vstack((q_mean_0, q_mean_1))
    size = self.num_areas * num_qs
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(size,size)

    for row in range(size):
      for col in range(size):

        if row == col:
          ax = fig.add_subplot(gs[row, col])
          plt.text(0, 0.2, areas[row // 3], ha='center', va='center')
          plt.text(0, -0.2, features[row % 3], ha='center', va='center')
          plt.xlim(-1, 1); plt.ylim(-1, 1)
          ax.set_yticklabels([])
          ax.set_xticklabels([])
          ax.axis('off')
          continue

        if row > col:
          continue
        # ax = plt.subplot(2*num_qs, 2*num_qs, row*2*num_qs + col+1)
        ax = fig.add_subplot(gs[row, col])
        plt.plot(q_mean[row], q_mean[col], '.')
        ax.yaxis.set_tick_params(labelsize=6)
        ax.xaxis.set_tick_params(labelsize=6)

        # if row != 0 or col != 1:
        #   ax.set_xticklabels([])
        #   ax.set_yticklabels([])

    plt.show()


  def plot_q_q_true(
      self,
      true_model,
      clist,
      rlist,
      burn_in=0,
      end=None,
      step=1,
      options=['q', 'q_shift1', 'q_shift2']):
    """Compare estimated q against true q.

    Args:
      clist: List.
      rlist: List.
    """
    num_qs = len(options)

    plt.figure(figsize=(num_qs * 4, 3))
    ind = 1
    if 'q' in options and len(self.q) > 1:
      q_samples = np.stack(self.q[burn_in:end:step], axis=0)
      num_samples, num_areas, num_trials, num_conditions = q_samples.shape
      print('q shape:', num_samples, num_areas, num_trials, num_conditions)
      # q_mean = q_samples.mean(axis=0)  # Posterior mean across samples.
      q_mean = np.quantile(q_samples, 0.5, axis=0)  # Median
      q_mean = q_mean[np.ix_(np.arange(num_areas),rlist,clist)]
      q_mean = q_mean.transpose(0,2,1).reshape(self.num_areas, -1)
      q0_true = true_model.q_arc[np.ix_(np.arange(num_areas),rlist,clist)]
      q0_true = q0_true.transpose(0,2,1).reshape(self.num_areas, -1)
      ax = plt.subplot(1, num_qs, ind); ind += 1
      plt.plot(q0_true.reshape(-1), q_mean.reshape(-1), '.')
      plt.plot([0, 1], [0, 1], 'k--', transform=ax.transAxes)
      plt.axis('square')
      plt.xlim(-1, 1); plt.ylim(-1, 1)
      plt.xlabel('True value'); plt.ylabel('Posterior mean')
      plt.title('Gain')
      # corr_true = np.corrcoef(q0_true)
      # print('True corr')
      # print(corr_true)
      # corr_hat = np.corrcoef(q_mean)
      # print('Estimated corr')
      # print(corr_hat)

    if 'q_shift1' in options and len(self.q_shift1) > 1:
      q_samples = np.stack(self.q_shift1[burn_in:end:step], axis=0)
      num_samples, num_areas, num_trials, num_conditions = q_samples.shape
      print('q1 shape:', num_samples, num_areas, num_trials, num_conditions)
      q_mean = q_samples.mean(axis=0)  # Posterior mean across samples.
      q_mean = q_mean[np.ix_(np.arange(num_areas),rlist,clist)]
      q1_mean = q_mean.transpose(0,2,1).reshape(-1)
      q1_true = true_model.q_shift1_arc[np.ix_(np.arange(num_areas),rlist,clist)]
      q1_true = q1_true.transpose(0,2,1).reshape(-1)
      ax = plt.subplot(1, num_qs, ind); ind += 1
      plt.plot(q1_true, q1_mean, '.')
      plt.plot([0, 1], [0, 1], 'k--', transform=ax.transAxes)
      plt.axis('square')
      plt.xlim(-0.05, 0.05); plt.ylim(-0.05, 0.05)
      plt.xlabel('True value'); plt.ylabel('Posterior mean')
      plt.title('Peak 1 deviation')

    if 'q_shift2' in options and len(self.q_shift2) > 1:
      q_samples = np.stack(self.q_shift2[burn_in:end:step], axis=0)
      num_samples, num_areas, num_trials, num_conditions = q_samples.shape
      print('q2 shape:', num_samples, num_areas, num_trials, num_conditions)
      q_mean = q_samples.mean(axis=0)  # Posterior mean across samples.
      q_mean = q_mean[np.ix_(np.arange(num_areas),rlist,clist)]
      q2_mean = q_mean.transpose(0,2,1).reshape(-1)
      q2_true = true_model.q_shift2_arc[np.ix_(np.arange(num_areas),rlist,clist)]
      q2_true = q2_true.transpose(0,2,1).reshape(-1)
      ax = plt.subplot(1, num_qs, ind)
      plt.plot(q2_true, q2_mean, '.')
      plt.plot([0, 1], [0, 1], 'k--', transform=ax.transAxes)
      plt.axis('square')
      plt.xlim(-0.1, 0.1); plt.ylim(-0.1, 0.1)
      plt.xlabel('True value'); plt.ylabel('Posterior mean')
      plt.title('Peak 2 deviation')


  def plot_correlation_traces(
      self,
      rho_type='partial',
      model_feature_type='BBS'):
    """Plot partial correlation graph.

    The partial correlation is calcualted using precision matirx.

    Args:
      plot_type: 'z', 'rho', 'corrcoef'
      distribution_type: 'CI', 'hist'
    """
    areas = ['V1', 'LM', 'AL']
    features = ['Gain', 'Peak-1', 'Peak-2']
    sigma_samples = np.stack(self.sigma_cross_pop, axis=0)
    num_samples, rho_size, _ = sigma_samples.shape
    print('sigma_samples.shape:', sigma_samples.shape)
    rho_samples_all = np.zeros(sigma_samples.shape)
    for s in range(num_samples):
      sigma = sigma_samples[s,:,:]
      if rho_type == 'partial':
        rho_samples_all[s] = util.partial_corr_from_cov(sigma)
      elif rho_type == 'marginal':
        rho_samples_all[s] = util.marginal_corr_from_cov(sigma)
    print('rho_samples.shape:', rho_samples_all.shape)

    gs_kw = dict(width_ratios=[1] * rho_size,
                 height_ratios=[1] * rho_size)
    fig, axs = plt.subplots(figsize=(19, 12), gridspec_kw=gs_kw,
        nrows=rho_size, ncols=rho_size)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)

    for row in range(rho_size):
      for col in range(rho_size):
        ax = fig.add_subplot(axs[row, col])
        if row > col:
          ax.set_xticks([])
          ax.set_yticks([])
          ax.axis('off')
          continue
        if row == col:
          plt.text(0, 0.25, areas[row // 3], ha='center', va='center')
          plt.text(0, -0.25, features[row % 3], ha='center', va='center')
          plt.xlim(-1, 1); plt.ylim(-1, 1)
          ax.set_xticks([])
          ax.set_yticks([])
          ax.axis('off')
          continue

        ax.set_xticks([])
        ax.set_yticks([])

        rho_data = rho_samples_all[:,row,col]
        plt.plot(rho_data, 'k', lw=0.5)
        plt.axhline(y=0, c='lightgrey', ls='-', lw=1)
        plt.ylim(-1, 1)

    plt.show()


  def plot_marginal_correlation(
      self,
      i=0, j=1,
      burn_in=0,
      end=None,
      step=1,
      plot_type='rho',
      true_model=None,
      model_feature_type=None,
      distribution_type='hist',
      output_dir=None):
    """Plots mu, sigma, rho.

    This is for the random design, where features such as q, q_shift1, q_shift2
    are treated as random generated from sigma. This is in contrast to the fixed
    design, where the features are fixed.

    Args:
      plot_type: 'z', 'rho', 'corrcoef'
      distribution_type: 'CI', 'hist', 'CI_hist'.
    """
    areas = ['V1', 'LM', 'AL']
    features = ['Gain', 'Peak-1', 'Peak-2']
    if plot_type in ['z', 'rho']:
      sigma_samples = np.stack(self.sigma_cross_pop, axis=0)
      num_samples, rho_size, _ = sigma_samples.shape
      print('sigma_samples.shape:', sigma_samples.shape)
      rho_samples_all = np.zeros(sigma_samples.shape)

      for s in range(num_samples):
        sigma = sigma_samples[s,:,:]
        rho_samples_all[s] = util.marginal_corr_from_cov(sigma)
      rho_samples = rho_samples_all[burn_in:end:step]
    elif plot_type in ['corrcoef']:
      qs = self.stack_q(0, None, 1, model_feature_type)
      num_samples, num_qs, _ = qs.shape
      rho_samples_all = np.zeros([num_samples, num_qs, num_qs])
      for s in range(num_samples):
        rho_samples_all[s] = np.corrcoef(qs[s])
      rho_samples = rho_samples_all[burn_in:end:step]
      print('rho_samples.shape:', rho_samples.shape)

    rho_data = rho_samples[:,i,j]
    z_data = util.fisher_transform(rho_data)

    if true_model is not None:
      sigma_true = true_model.sigma_cross_pop
      sigma_diag_sqrt = np.sqrt(np.diag(sigma_true))
      rho_true = sigma_true / sigma_diag_sqrt / sigma_diag_sqrt.reshape(-1, 1)
      rhoz_true = util.fisher_transform(rho_true)
      q_true = true_model.stack_q()
      q_true = q_true.transpose(0,2,1).reshape(
          true_model.num_areas*true_model.num_qs, -1)
      rho_true_fix = np.corrcoef(q_true)
      rhoz_true_hat = util.fisher_transform(rho_true_fix)
    else:
      rho_true = None

    # Small panel for (i,j) entry.
    if output_dir is not None:
      file_path = output_dir + 'margianl_corr.pdf'
    else:
      file_path = None
    # self.plot_rho_z_samples_single(rho_samples[:,i,j], file_path=file_path)
    # self.plot_rho_z_samples_two_stack(rho_samples[:,1,4], rho_samples[:,1,7],
    #     file_path=file_path)
    # self.plot_rho_z_samples(rho_samples[:,i,j], rho_samples_all[:,i,j], rho_true)
    # return None, None, None

    if distribution_type == 'hist':
      if true_model is None:
        rho_true = None
        rho_true_fix = None
      adj_mat = self.plot_rho(rho_samples, plot_type=plot_type,
          rho_true=rho_true, rho_true_fix=rho_true_fix,
          output_dir=output_dir, suffix='_marginal_corr')
      return adj_mat

    elif distribution_type == 'CI':
      if output_dir is not None:
        file_path = os.path.join(output_dir,
            f'{self.session_id}_marginal_corr_CI.pdf')
      else:
        file_path=None
      return self.plot_rho_CI(rho_samples, file_path=file_path)

    elif distribution_type == 'CI_hist':
      if output_dir is not None:
        file_path = os.path.join(output_dir,
            f'{self.session_id}_marginal_corr_CI_hist.pdf')
      else:
        file_path=None
      return self.plot_rho_CI_hist(rho_samples, file_path=file_path)


  def plot_marginal_correlation_stack(
      self,
      ij_list,
      sub_title_list,
      title,
      burn_in=0,
      end=None,
      step=1,
      plot_type='rho',
      true_model=None,
      model_feature_type=None,
      file_path=None):
    """Plots marginal correlations.

    Args:
      plot_type: 'z', 'rho', 'corrcoef'
    """
    areas = ['V1', 'LM', 'AL']
    features = ['Gain', 'Peak-1', 'Peak-2']
    if plot_type in ['z', 'rho']:
      sigma_samples = np.stack(self.sigma_cross_pop, axis=0)
      num_samples, rho_size, _ = sigma_samples.shape
      print('sigma_samples.shape:', sigma_samples.shape)
      rho_samples_all = np.zeros(sigma_samples.shape)

      for s in range(num_samples):
        sigma = sigma_samples[s,:,:]
        rho_samples_all[s] = util.marginal_corr_from_cov(sigma)
      rho_samples = rho_samples_all[burn_in:end:step]
    elif plot_type in ['corrcoef']:
      qs = self.stack_q(0, None, 1, model_feature_type)
      num_samples, num_qs, _ = qs.shape
      rho_samples_all = np.zeros([num_samples, num_qs, num_qs])
      for s in range(num_samples):
        rho_samples_all[s] = np.corrcoef(qs[s])
      rho_samples = rho_samples_all[burn_in:end:step]
      print('rho_samples.shape:', rho_samples.shape)

    stack_samples = []
    for i,j in ij_list:
      stack_samples.append(rho_samples[:,i,j])
      # z_data = util.fisher_transform(rho_data)
    self.plot_rho_z_samples_stack(stack_samples, sub_title_list, title,
        file_path=file_path)


  def plot_marginal_correlation_embedded(
      self,
      ij_pair,
      burn_in=0,
      end=None,
      step=1,
      plot_type='rho',
      true_model=None,
      model_feature_type=None,
      file_path=None):
    """Plots marginal correlations.

    Args:
      plot_type: 'z', 'rho', 'corrcoef'
    """
    areas = ['V1', 'LM', 'AL']
    features = ['Gain', 'Peak-1', 'Peak-2']
    if plot_type in ['z', 'rho']:
      sigma_samples = np.stack(self.sigma_cross_pop, axis=0)
      num_samples, rho_size, _ = sigma_samples.shape
      print('sigma_samples.shape:', sigma_samples.shape)
      rho_samples_all = np.zeros(sigma_samples.shape)

      for s in range(num_samples):
        sigma = sigma_samples[s,:,:]
        rho_samples_all[s] = util.marginal_corr_from_cov(sigma)
      rho_samples = rho_samples_all[burn_in:end:step]
    elif plot_type in ['corrcoef']:
      qs = self.stack_q(0, None, 1, model_feature_type)
      num_samples, num_qs, _ = qs.shape
      rho_samples_all = np.zeros([num_samples, num_qs, num_qs])
      for s in range(num_samples):
        rho_samples_all[s] = np.corrcoef(qs[s])
      rho_samples = rho_samples_all[burn_in:end:step]
      print('rho_samples.shape:', rho_samples.shape)

    i,j = ij_pair
    rho_data = rho_samples[:,i,j]

    gs_kw = dict(width_ratios=[1], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(1.5, 1), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
    ax = fig.add_subplot(axs)
    ax.tick_params(left=False, labelleft=False, bottom=True, labelbottom=True)
    x, y = seaborn.distplot(
        rho_data, bins=30, color='grey').get_lines()[0].get_data()
    center = np.quantile(rho_data, 0.5)
    CI_left = np.quantile(rho_data, 0.025)
    CI_right = np.quantile(rho_data, 0.975)
    err_left = center - CI_left
    err_right = CI_right - center
    gkde=scipy.stats.gaussian_kde(rho_data)
    x = np.linspace(CI_left, CI_right, 201)
    y = gkde.evaluate(x)
    mode = x[np.argmax(y)]

    plt.plot(1000, 0, 'k+', label='median')  # Fake point for median.
    plt.errorbar([center], [np.max(y)*1.5], xerr=[[err_left], [err_right]],
        fmt='+k', capsize=5, label='95% CI')
    plt.axvline(x=0, color='k', ls=':')
    plt.xlim(0, 1)
    plt.ylim(0, np.max(y)*2)

    print(f'CI_left {CI_left:.2f}, CI_right {CI_right:.2f}, mode {mode:.2f}')

    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('save figure:', file_path)
    plt.show()



  def plot_partial_correlation(
      self,
      i=0, j=1,
      burn_in=0,
      end=None,
      step=1,
      plot_type='rho',
      true_model=None,
      distribution_type='hist',
      model_feature_type='BBS',
      output_dir=None):
    """Plot partial correlation graph.

    The partial correlation is calcualted using precision matirx.

    Args:
      plot_type: 'z', 'rho', 'corrcoef'
      distribution_type: 'CI', 'hist'
    """
    areas = ['V1', 'LM', 'AL']
    features = ['Gain', 'First peak', 'Second peak']
    sigma_samples = np.stack(self.sigma_cross_pop, axis=0)
    num_samples, rho_size, _ = sigma_samples.shape
    print('sigma_samples.shape:', sigma_samples.shape)
    rho_samples_all = np.zeros(sigma_samples.shape)

    if plot_type in ['z', 'rho']:
      for s in range(num_samples):
        sigma = sigma_samples[s,:,:]
        rho_samples_all[s] = util.partial_corr_from_cov(sigma)
      rho_samples = rho_samples_all[burn_in:end:step]
      print('rho_samples.shape:', rho_samples.shape)

    elif plot_type in ['corrcoef']:
      qs = self.stack_q(0, None, 1, model_feature_type)
      num_samples, num_qs, _ = qs.shape
      rho_samples_all = np.zeros([num_samples, num_qs, num_qs])
      for s in range(num_samples):
        cov = np.cov(qs[s])
        rho_samples_all[s] = util.partial_corr_from_cov(cov)
      rho_samples = rho_samples_all[burn_in:end:step]
      print('rho_samples.shape:', rho_samples.shape)

    # Subsample for i,j entry.
    rho_data = rho_samples[:,i,j]
    z_data = util.fisher_transform(rho_data)
    # Get the probability > 0.
    # p_geq_0 = sum(rho_data >= 0) / len(rho_data)
    # print(f'P(rho > 0) = {p_geq_0:0.5f}')

    if true_model is not None:
      rho_true = util.partial_corr_from_cov(true_model.sigma_cross_pop)
      rhoz_true = util.fisher_transform(rho_true)
      q_true = true_model.stack_q()
      q_true = q_true.transpose(0,2,1).reshape(
          true_model.num_areas*true_model.num_qs, -1)
      cov = np.cov(q_true)
      rho_true_fix = util.partial_corr_from_cov(cov)
      rhoz_true_fix = util.fisher_transform(rho_true_fix)
    else:
      rho_true = None

    # Small panel for (i,j) entry.
    if output_dir is not None:
      file_path = output_dir + 'partial_corr.pdf'
    else:
      file_path = None
    # self.plot_rho_z_samples_single(rho_samples[:,i,j], file_path=file_path)
    # self.plot_rho_z_samples_two_stack(rho_samples[:,1,4], rho_samples[:,1,7],
    #     file_path=file_path)
    # self.plot_rho_z_samples(rho_samples[:,i,j], rho_samples_all[:,i,j], rho_true)

    # Show all pairs whole panel.
    if distribution_type == 'hist':
      if true_model is None:
        rho_true = None
        rho_true_fix = None
      return self.plot_rho(rho_samples, plot_type=plot_type,
          rho_true=rho_true, rho_true_fix=rho_true_fix,
          output_dir=output_dir, suffix='_partial_corr')

    elif distribution_type == 'CI':
      if output_dir is not None:
        file_path = os.path.join(output_dir,
            f'{self.session_id}_partial_corr_CI.pdf')
      else:
        file_path = None
      return self.plot_rho_CI(rho_samples, file_path=file_path)

    elif distribution_type == 'CI_hist':
      if output_dir is not None:
        file_path = os.path.join(output_dir,
            f'{self.session_id}_partial_corr_CI_hist.pdf')
      else:
        file_path = None
      return self.plot_rho_CI_hist(rho_samples, file_path=file_path)



  def plot_partial_correlation_stack(
      self,
      ij_list,
      sub_title_list,
      title,
      burn_in=0,
      end=None,
      step=1,
      plot_type='rho',
      true_model=None,
      model_feature_type='BBS',
      file_path=None):
    """Plot partial correlation graph.

    The partial correlation is calcualted using precision matirx.

    Args:
      plot_type: 'z', 'rho', 'corrcoef'
      distribution_type: 'CI', 'hist'
    """
    areas = ['V1', 'LM', 'AL']
    features = ['Gain', 'First peak', 'Second peak']
    sigma_samples = np.stack(self.sigma_cross_pop, axis=0)
    num_samples, rho_size, _ = sigma_samples.shape
    print('sigma_samples.shape:', sigma_samples.shape)
    rho_samples_all = np.zeros(sigma_samples.shape)

    if plot_type in ['z', 'rho']:
      for s in range(num_samples):
        sigma = sigma_samples[s,:,:]
        rho_samples_all[s] = util.partial_corr_from_cov(sigma)
      rho_samples = rho_samples_all[burn_in:end:step]
      print('rho_samples.shape:', rho_samples.shape)

    elif plot_type in ['corrcoef']:
      qs = self.stack_q(0, None, 1, model_feature_type)
      num_samples, num_qs, _ = qs.shape
      rho_samples_all = np.zeros([num_samples, num_qs, num_qs])
      for s in range(num_samples):
        cov = np.cov(qs[s])
        rho_samples_all[s] = util.partial_corr_from_cov(cov)
      rho_samples = rho_samples_all[burn_in:end:step]
      print('rho_samples.shape:', rho_samples.shape)

    stack_samples = []
    for i,j in ij_list:
      stack_samples.append(rho_samples[:,i,j])
      # z_data = util.fisher_transform(rho_data)
    self.plot_rho_z_samples_stack(stack_samples, sub_title_list, title,
        file_path=file_path)


  def plot_conditional_correlation_multiple(
      self,
      partial_indices,
      sub_titles,
      title,
      burn_in=0,
      end=None,
      step=1,
      output_dir=None):
    """Plot partial correlation graph.

    The partial correlation is calcualted using precision matirx.

    Args:
      partial_indices: lists of list.
    """
    num_subplots = len(partial_indices)
    gs_kw = dict(width_ratios=[1] * num_subplots)
    fig, axs = plt.subplots(figsize=(2.2*num_subplots, 1),
        gridspec_kw=gs_kw, nrows=1, ncols=num_subplots)
    axs = [axs] if num_subplots == 1 else axs
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.1)
    for col in range(num_subplots):
      sub_indices = partial_indices[col]

      # Extract data.
      sigma_samples = np.stack(self.sigma_cross_pop, axis=0)
      sigma_samples = sigma_samples[
          np.ix_(np.arange(sigma_samples.shape[0]), sub_indices, sub_indices)]
      num_samples, rho_size, _ = sigma_samples.shape
      rho_samples_all = np.zeros(sigma_samples.shape)
      for s in range(num_samples):
        sigma = sigma_samples[s,:,:]
        rho_samples_all[s] = util.partial_corr_from_cov(sigma)
      rho_samples = rho_samples_all[burn_in:end:step]
      if col == 0:
        print('sigma_samples.shape:', sigma_samples.shape)
        print('rho_samples.shape:', rho_samples.shape)

      # Subsample for i,j entry.
      i, j = 0, 1
      rho_data = rho_samples[:,i,j]

      ax = fig.add_subplot(axs[col])
      x, y = seaborn.distplot(
            rho_data, bins=25, color='tab:gray').get_lines()[0].get_data()
      center = np.quantile(rho_data, 0.5)
      CI_left = np.quantile(rho_data, 0.025)
      CI_right = np.quantile(rho_data, 0.975)
      err_left = center - CI_left
      err_right = CI_right - center
      plt.axvline(x=0, color='k', ls=':')
      plt.errorbar([center], [np.max(y)*1.5], xerr=[[err_left], [err_right]],
                   fmt='+k', capsize=5, label='95% CI')
      plt.xlim(-1.05, 1.05)
      plt.ylim(0, np.max(y)*2)
      plt.title(sub_titles[col],  y=0.95, fontsize=10)

      if col == 1:
        plt.plot(1000, 0, 'k+', label='median')  # Fake point for median.
        plt.legend(loc=(0, -0.6), ncol=2)

      if col == 0:
        ax.tick_params(left=False, labelleft=False, labelbottom=True,
                      bottom=True, top=False, labeltop=False)
        # plt.plot(1000, 0, 'k+', label='median')  # Fake point for median.
        # if center > 0:
        #   plt.legend(loc='center left')
        # else:
        #   plt.legend(loc='center right')
      else:
        ax.tick_params(left=False, labelleft=False, labelbottom=True,
                       bottom=True, top=False, labeltop=False)
      ax.tick_params(which='minor',
                     top=False, labeltop=False, bottom=True, labelbottom=True)
      ax.xaxis.set_minor_locator(MultipleLocator(0.2))
      if col == 0:
        plt.xlabel('Correlation')
        plt.text(0.5, 1.35, title,
                 ha='center', transform=ax.transAxes, fontsize=10)

    if output_dir is not None:
      file_title = title.replace(', ', '_').replace(' | ', '_')
      file_title = file_title.replace('cor(', '').replace(')', '')
      file_name = f'{self.session_id}_{file_title}_conditional_corr_CI.pdf'
      output_figure_path = os.path.join(output_dir, file_name)
      plt.savefig(output_figure_path, bbox_inches='tight')
      print('Save figure to: ', output_figure_path)
      # plt.close()
      plt.show()



  def plot_conditional_correlation_embedded(
      self,
      partial_index,
      burn_in=0,
      end=None,
      step=1,
      file_path=None):
    """Plot partial correlation graph.

    The partial correlation is calcualted using precision matirx.

    Args:
      partial_indices: lists of list.
    """
    sub_indices = partial_index
    # Extract data.
    sigma_samples = np.stack(self.sigma_cross_pop, axis=0)
    sigma_samples = sigma_samples[
        np.ix_(np.arange(sigma_samples.shape[0]), sub_indices, sub_indices)]
    num_samples, rho_size, _ = sigma_samples.shape
    rho_samples_all = np.zeros(sigma_samples.shape)
    for s in range(num_samples):
      sigma = sigma_samples[s,:,:]
      rho_samples_all[s] = util.partial_corr_from_cov(sigma)
    rho_samples = rho_samples_all[burn_in:end:step]
    print('sigma_samples.shape:', sigma_samples.shape)
    print('rho_samples.shape:', rho_samples.shape)
    # Subsample for i,j entry. Entries from 2 are conditions.
    i, j = 0, 1
    rho_data = rho_samples[:,i,j]

    gs_kw = dict(width_ratios=[1], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(1.5, 1), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
    ax = fig.add_subplot(axs)
    ax.tick_params(left=False, labelleft=False, bottom=True, labelbottom=True)
    x, y = seaborn.distplot(
          rho_data, bins=25, color='tab:gray').get_lines()[0].get_data()
    center = np.quantile(rho_data, 0.5)
    CI_left = np.quantile(rho_data, 0.025)
    CI_right = np.quantile(rho_data, 0.975)
    err_left = center - CI_left
    err_right = CI_right - center
    gkde=scipy.stats.gaussian_kde(rho_data)
    x = np.linspace(CI_left, CI_right, 201)
    y = gkde.evaluate(x)
    mode = x[np.argmax(y)]

    plt.axvline(x=0, color='k', ls=':')
    plt.errorbar([center], [np.max(y)*1.5], xerr=[[err_left], [err_right]],
                 fmt='+k', capsize=5, label='95% CI')
    plt.xlim(-1, 1)
    plt.ylim(0, np.max(y)*2)
    plt.plot(1000, 0, 'k+', label='median')  # Fake point for median.

    print(f'CI_left {CI_left:.2f}, CI_right {CI_right:.2f}, mode {mode:.2f}')

    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('Save figure to: ', file_path)
    plt.show()


  def plot_partial_correlation_featurewise(
      self,
      feature_id,
      burn_in=0,
      end=None,
      step=1,
      plot_type='rho',
      true_model=None,
      distribution_type='hist',
      model_feature_type='BSS',
      output_dir=None):
    """Plot feature-wise partial correlation."""
    if model_feature_type != 'BSS':
      print('Only support BSS currently.')
      return

    areas = ['V1', 'LM', 'AL']
    features = ['Gain', 'Peak-1', 'Peak-2']
    sigma_samples = np.stack(self.sigma_cross_pop, axis=0)
    num_samples, rho_size, _ = sigma_samples.shape

    # For B.
    rho_samples_all = np.zeros([num_samples, self.num_areas, self.num_areas])
    if model_feature_type == 'BSS':
      qid = [0,3,6]
    for s in range(num_samples):
      sub_sigma = sigma_samples[np.ix_([s], qid, qid)][0]
      theta = np.linalg.inv(sub_sigma)
      theta_diag_sqrt = np.sqrt(np.diag(theta))
      rho_samples_all[s] = - theta / np.outer(theta_diag_sqrt, theta_diag_sqrt)
    rho_samples_B = rho_samples_all[burn_in:end:step]

    # For q_shift1.
    rho_samples_all = np.zeros([num_samples, self.num_areas, self.num_areas])
    if model_feature_type == 'BSS':
      qid = [1,4,7]
    for s in range(num_samples):
      sub_sigma = sigma_samples[np.ix_([s], qid, qid)][0]
      theta = np.linalg.inv(sub_sigma)
      theta_diag_sqrt = np.sqrt(np.diag(theta))
      rho_samples_all[s] = - theta / np.outer(theta_diag_sqrt, theta_diag_sqrt)
    rho_samples_S1 = rho_samples_all[burn_in:end:step]

    # For q_shift2.
    rho_samples_all = np.zeros([num_samples, self.num_areas, self.num_areas])
    if model_feature_type == 'BSS':
      qid = [2,5,8]
    for s in range(num_samples):
      sub_sigma = sigma_samples[np.ix_([s], qid, qid)][0]
      theta = np.linalg.inv(sub_sigma)
      theta_diag_sqrt = np.sqrt(np.diag(theta))
      rho_samples_all[s] = - theta / np.outer(theta_diag_sqrt, theta_diag_sqrt)
    rho_samples_S2 = rho_samples_all[burn_in:end:step]

    # Begin plotting.
    if feature_id == 0:
      rho_samples = rho_samples_B
    elif feature_id == 1:
      rho_samples = rho_samples_S1
    elif feature_id == 2:
      rho_samples = rho_samples_S2
    print('rho_samples.shape:', rho_samples.shape)

    # ------------- CIs in matrix -------------
    gs_kw = dict(width_ratios=[1] * self.num_areas,
                 height_ratios=[1] * self.num_areas)
    fig, axs = plt.subplots(figsize=(8, 5), gridspec_kw=gs_kw,
        nrows=self.num_areas, ncols=self.num_areas, constrained_layout=True)
    # plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
    for row in range(self.num_areas):
      for col in range(self.num_areas):
        ax = fig.add_subplot(axs[row, col])
        if row > col:
          ax.set_xticks([])
          ax.set_yticks([])
          ax.axis('off')
          continue
        if row == col:
          plt.text(0, 0.25, areas[row], ha='center', va='center')
          plt.text(0, -0.25, features[feature_id], ha='center', va='center')
          plt.xlim(-1, 1); plt.ylim(-1, 1)
          ax.set_xticks([])
          ax.set_yticks([])
          ax.axis('off')
          continue

        rho_data = rho_samples[:,col,row]
        x, y = seaborn.distplot(
            rho_data, bins=25, color='tab:gray').get_lines()[0].get_data()
        # center = x[np.argmax(y)]
        center = np.quantile(rho_data, 0.5)
        CI_left = np.quantile(rho_data, 0.025)
        CI_right = np.quantile(rho_data, 0.975)
        plt.axvline(x=CI_left, linestyle=':', color='k')
        plt.axvline(x=CI_right, linestyle=':', color='k', label='%95 CI')
        plt.axvline(x=center, color='g', label='median')
        plt.xlim(-1.05, 1.05)
    plt.show()

    # ------------- CIs in a row for 3 areas -------------
    gs_kw = dict(width_ratios=[1] * self.num_areas)
    fig, axs = plt.subplots(figsize=(5, 1), gridspec_kw=gs_kw,
        nrows=1, ncols=self.num_areas)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
    for row in range(self.num_areas):
      for col in range(self.num_areas):
        if row >= col or row >= 2:
          continue
        rho_data = rho_samples[:,col,row]
        ax = fig.add_subplot(axs[col-1 + row])
        center = np.quantile(rho_data, 0.5)
        CI_left = np.quantile(rho_data, 0.025)
        CI_right = np.quantile(rho_data, 0.975)
        err_left = center - CI_left
        err_right = CI_right - center
        plt.axvline(x=0, color='k', ls=':')
        plt.errorbar([center], [0], xerr=[[err_left], [err_right]],
                     fmt='+k', capsize=5, label='95% CI')
        plt.xlim(-1.05, 1.05)
        ax.set_title(f'{areas[row]} - {areas[col]}', fontsize=10)

        if col == 1:
          ax.tick_params(left=False, labelleft=False, labelbottom=True,
                        bottom=True, top=False, labeltop=False)
        else:
          ax.tick_params(left=False, labelleft=False, labelbottom=False,
                         bottom=True, top=False, labeltop=False)
        ax.tick_params(which='minor',
                       top=False, labeltop=False, bottom=True, labelbottom=True)
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        if row == 0 and col == 2:
          plt.text(0.5, 1.3, f'{features[feature_id]}',
                   ha='center', transform=ax.transAxes, fontsize=12)
    if output_dir is not None:
      file_name = f'{self.session_id}_{features[feature_id]}_featurewise_partial_corr_CI.pdf'
      output_figure_path = os.path.join(output_dir, file_name)
      plt.savefig(output_figure_path, bbox_inches='tight')
      print('Save figure to: ', output_figure_path)
      # plt.close()
      plt.show()

    # ------------- CIs in separate figures -------------
    for row in range(self.num_areas):
      for col in range(self.num_areas):
        if row >= col:
          continue
        rho_data = rho_samples[:,col,row]
        plt.figure(figsize=(5,4))
        x, y = seaborn.distplot(
            rho_data, bins=25, color='tab:gray').get_lines()[0].get_data()
        center = np.quantile(rho_data, 0.5)
        CI_left = np.quantile(rho_data, 0.025)
        CI_right = np.quantile(rho_data, 0.975)
        plt.axvline(x=0, ymax=0.1, color='r')
        plt.axvline(x=CI_left, ymax=0.1, linestyle=':', color='k')
        plt.axvline(x=CI_right, ymax=0.1, linestyle=':', color='k', label='%95 CI')
        plt.axvline(x=center, ymax=0.1, color='g', label='median')

        plt.legend(loc='best')
        plt.xlim(-1, 1)
        title = (f'{areas[row]} {features[feature_id]} -- ' +
                    f'{areas[col]} {features[feature_id]}')
        # plt.title(title, fontsize=10)
        # plt.text(0.05, 0.7, f'median= {x[np.argmax(y)]:.2f}',
        #          transform=plt.gca().transAxes, fontsize=10)

        if output_dir is not None:
          file_name = f'{self.session_id}_' + title + '_featurewise_partial_corr.pdf'
          output_figure_path = os.path.join(output_dir, file_name)
          plt.savefig(output_figure_path)
          print('Save figure to: ', output_figure_path)
          plt.close()
        else:
          # plt.show()
          plt.close()


  def plot_rho_z_samples_single(
      self,
      rho_data,
      file_path=None):
    """Plots rho sample for verification."""
    gs_kw = dict(width_ratios=[1], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(4, 1.2), gridspec_kw=gs_kw, nrows=1, ncols=1)
    # plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.2)
    ax = axs
    ax.tick_params(left=False, labelleft=False, bottom=True, labelbottom=True)
    x, y = seaborn.distplot(
        rho_data, bins=25, color='grey').get_lines()[0].get_data()
    # center = x[np.argmax(y)]
    center = np.quantile(rho_data, 0.5)
    CI_left = np.quantile(rho_data, 0.025)
    CI_right = np.quantile(rho_data, 0.975)
    plt.axvline(x=CI_left, linestyle=':', color='k')
    plt.axvline(x=CI_right, linestyle=':', color='k', label='%95 CI')
    plt.axvline(x=center, color='k', ls='--', label='Median')

    plt.legend(loc='best', ncol=1)
    plt.xlim(-1, 1)
    plt.ylim(0, 1.5 * np.max(y))
    # plt.text(0.05, 0.7, f'median= {x[np.argmax(y)]:.2f}',
    #          transform=plt.gca().transAxes, fontsize=10)

    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('save figure:', file_path)
    plt.show()


  def plot_rho_z_samples_stack(
      self,
      rho_data_list,
      sub_title_list,
      fig_title,
      file_path=None):
    """Plots rho sample for verification."""
    num_stack = len(rho_data_list)
    gs_kw = dict(width_ratios=[1], height_ratios=[1]*num_stack)
    fig, axs = plt.subplots(figsize=(4, 3), gridspec_kw=gs_kw,
        nrows=num_stack, ncols=1)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)

    for r, rho_data in enumerate(rho_data_list):
      ax = fig.add_subplot(axs[r])
      ax.tick_params(left=False, labelleft=False, bottom=True, labelbottom=False,
          direction='in')
      if r == num_stack-1:
        ax.tick_params(bottom=True, labelbottom=True, direction='in')
      x, y = seaborn.distplot(
          rho_data, bins=30, color='grey').get_lines()[0].get_data()
      center = np.quantile(rho_data, 0.5)
      CI_left = np.quantile(rho_data, 0.025)
      CI_right = np.quantile(rho_data, 0.975)
      err_left = center - CI_left
      err_right = CI_right - center
      plt.plot(1000, 0, 'k+', label='median')  # Fake point for median.
      plt.errorbar([center], [np.max(y)*1.5], xerr=[[err_left], [err_right]],
          fmt='+k', capsize=5, label='95% CI')
      # plt.axvline(x=CI_left, linestyle=':', color='k')
      # plt.axvline(x=CI_right, linestyle=':', color='k', label='%95 CI')
      # plt.axvline(x=center, color='k', ls='--', label='Median')
      plt.axvline(x=0, color='k', ls=':')
      plt.xlim(-1, 1)
      # plt.xlim(0, 1)
      plt.ylim(0, np.max(y)*2)
      if center > 0:
        plt.text(0.05, 0.62, sub_title_list[r],
                 transform=plt.gca().transAxes, fontsize=10)
      else:
        plt.text(0.6, 0.62, sub_title_list[r],
                 transform=plt.gca().transAxes, fontsize=10)
      # if r == 0:
      #   if center > 0:
      #     plt.legend(loc='lower left', ncol=2, prop={'size': 8})
      #   else:
      #     plt.legend(loc='lower right', ncol=2, prop={'size': 8})
      if r == num_stack-1:
        plt.xlabel('Correlation')
      if r == 0:
        plt.title(fig_title, fontsize=16)

    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('save figure:', file_path)
    plt.show()


  def plot_rho_z_samples(
      self,
      rho_data,
      rho_samples_all=None,
      rho_true=None):
    """Plots rho sample for verification."""
    z_data = util.fisher_transform(rho_data)
    plt.figure(figsize=(12, 3))
    plt.subplot(131)
    x, y = seaborn.distplot(
        rho_data, bins=25, color='tab:gray').get_lines()[0].get_data()
    # center = x[np.argmax(y)]
    center = np.quantile(rho_data, 0.5)
    CI_left = np.quantile(rho_data, 0.025)
    CI_right = np.quantile(rho_data, 0.975)
    plt.axvline(x=CI_left, linestyle=':', color='k')
    plt.axvline(x=CI_right, linestyle=':', color='k', label='%95 CI')
    plt.axvline(x=center, color='g', label='median')

    if rho_true is not None:
      plt.axvline(x=rho_true, color='g', label='True corr')
    plt.legend(loc='best')
    plt.xlim(-1, 1)
    plt.text(0.05, 0.7, f'median= {x[np.argmax(y)]:.2f}',
             transform=plt.gca().transAxes, fontsize=10)

    plt.subplot(132)
    x, y = seaborn.distplot(
        z_data, bins=25, color='tab:gray').get_lines()[0].get_data()
    CI_left = np.quantile(z_data, 0.025)
    CI_right = np.quantile(z_data, 0.975)
    # center = x[np.argmax(y)]
    center = np.quantile(z_data, 0.5)

    plt.axvline(x=CI_left, linestyle=':', color='k')
    plt.axvline(x=CI_right, linestyle=':', color='k', label='%95 CI')
    plt.axvline(x=center, color='g', label='median')
    # plt.axvline(x=0, color='b', label='null hypothesis')
    # plt.legend(loc='upper left')
    # plt.xlim(-1.8, 1.8)
    plt.title('Z-transformed values')

    plt.subplot(133)
    if rho_true is not None:
      plt.axhline(y=rho_true, color='g', label='True value')
    if rho_samples_all is not None:
      plt.plot(rho_samples_all, '.-',lw=0.2)
    else:
      plt.plot(rho_data, '.-',lw=0.2)
    plt.ylim(-1.05, 1.05)
    plt.title('rho samples')

    plt.show()


  def plot_rho_CI(
      self,
      rho_samples,
      burn_in=0,
      end=None,
      step=1,
      file_path=None,
      verbose=False):
    """Calculate the CI coverage."""
    areas = ['V1', 'LM', 'AL']
    features = ['Gain', 'Peak-1', 'Peak-2']
    num_qs = 3
    num_samples, rho_size, _ = rho_samples.shape
    if verbose:
      print('rho_samples.shape:', rho_samples.shape)

    gs_kw = dict(width_ratios=[1] * rho_size,
                 height_ratios=[1] * rho_size)
    fig, axs = plt.subplots(figsize=(15, 8), gridspec_kw=gs_kw,
        nrows=rho_size, ncols=rho_size)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)

    confidence_intervals = np.zeros([rho_size, rho_size, 2])
    centers = np.zeros([rho_size, rho_size])
    pval_mat = np.zeros([rho_size, rho_size])
    # CI traps.
    for row in range(rho_size):
      for col in range(rho_size):
        ax = fig.add_subplot(axs[row, col])
        if row > col:
          ax.set_xticks([])
          ax.set_yticks([])
          ax.axis('off')
          continue
        if row == col:
          plt.text(0, 0.25, areas[row // 3], ha='center', va='center')
          plt.text(0, -0.25, features[row % 3], ha='center', va='center')
          plt.xlim(-1, 1); plt.ylim(-1, 1)
          ax.set_xticks([])
          ax.set_yticks([])
          ax.axis('off')
          continue

        sub_samples = rho_samples[:,row,col]
        CI_left = np.quantile(sub_samples, 0.025)
        CI_right = np.quantile(sub_samples, 0.975)
        confidence_intervals[[row, col], [col, row], :] = [CI_left, CI_right]
        # gkde=scipy.stats.gaussian_kde(sub_samples)
        # x = np.linspace(CI_left, CI_right, 201)
        # y = gkde.evaluate(x)
        # mode = x[np.argmax(y)]
        center = np.quantile(sub_samples, 0.5)
        centers[[row,col], [col,row]] = center
        err_left = center - CI_left
        err_right = CI_right - center
        samples_mean = np.mean(sub_samples)
        samples_std = np.std(sub_samples)
        alpha = scipy.stats.norm.cdf(0, loc=samples_mean, scale=samples_std)
        p_val = min(alpha, 1-alpha)  # Two tail test.
        pval_mat[[row, col], [col, row]] = p_val

        plt.axvline(x=0, color='k', ls=':')
        plt.errorbar([center], [0], xerr=[[err_left], [err_right]],
                     fmt='+k', capsize=5, label='95% CI')

        if row == 0 and col == 1:
          ax.tick_params(left=False, labelbottom=False, bottom=False,
                         top=True, labeltop=True)
          ax.set_xticks([-1, 0, 1])
          plt.plot([10], [0], '+k', label='median')
          plt.legend(loc=(-0.8, -1.2))
        else:
          ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(which='minor', direction='in',
                       top=False, labeltop=False, bottom=True, labelbottom=True)
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        plt.xlim(-1.05, 1.05)
        if 0 < CI_left:
          ax.set_facecolor('lightpink')
        if CI_right < 0:
          ax.set_facecolor('lightblue')

    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('save figure:', file_path)

    # If the CI is on the same side of the 0 line.
    significance = (np.sign(confidence_intervals[:,:,0]) ==
                    np.sign(confidence_intervals[:,:,1])) + 0
    # Mask out the significace edges.
    corr_mat = significance * centers
    return corr_mat, pval_mat


  def plot_rho_CI_hist(
      self,
      rho_samples,
      burn_in=0,
      end=None,
      step=1,
      file_path=None,
      verbose=False):
    """Calculate the CI coverage."""
    areas = ['V1', 'LM', 'AL']
    features = ['Gain', 'Peak-1', 'Peak-2']
    num_qs = 3
    num_samples, rho_size, _ = rho_samples.shape
    if verbose:
      print('rho_samples.shape:', rho_samples.shape)

    gs_kw = dict(width_ratios=[1] * rho_size,
                 height_ratios=[1] * rho_size)
    fig, axs = plt.subplots(figsize=(15, 8), gridspec_kw=gs_kw,
        nrows=rho_size, ncols=rho_size)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)

    confidence_intervals = np.zeros([rho_size, rho_size, 2])
    confidence_intervals2 = np.zeros([rho_size, rho_size, 2])
    centers = np.zeros([rho_size, rho_size])
    pval_mat = np.zeros([rho_size, rho_size])
    # CI traps.
    for row in range(rho_size):
      for col in range(rho_size):
        ax = fig.add_subplot(axs[row, col])
        if row > col:
          ax.set_xticks([])
          ax.set_yticks([])
          ax.axis('off')
          continue
        if row == col:
          plt.text(0, 0.25, areas[row // 3], ha='center', va='center')
          plt.text(0, -0.25, features[row % 3], ha='center', va='center')
          plt.xlim(-1, 1); plt.ylim(-1, 1)
          ax.set_xticks([])
          ax.set_yticks([])
          ax.axis('off')
          continue

        sub_samples = rho_samples[:,row,col]
        CI_left = np.quantile(sub_samples, 0.025)
        CI_right = np.quantile(sub_samples, 0.975)
        confidence_intervals[[row, col], [col, row], :] = [CI_left, CI_right]
        CI_left2 = np.quantile(sub_samples, 0.0025)
        CI_right2 = np.quantile(sub_samples, 0.9975)
        confidence_intervals2[[row, col], [col, row], :] = [CI_left2, CI_right2]
        # gkde=scipy.stats.gaussian_kde(sub_samples)
        # x = np.linspace(CI_left, CI_right, 201)
        # y = gkde.evaluate(x)
        # mode = x[np.argmax(y)]
        center = np.quantile(sub_samples, 0.5)
        centers[[row,col], [col,row]] = center
        err_left = center - CI_left
        err_right = CI_right - center
        err_left2 = center - CI_left2
        err_right2 = CI_right2 - center
        samples_mean = np.mean(sub_samples)
        samples_std = np.std(sub_samples)
        alpha = scipy.stats.norm.cdf(0, loc=samples_mean, scale=samples_std)
        p_val = min(alpha, 1-alpha)  # Two tail test.
        pval_mat[[row, col], [col, row]] = p_val

        # Before call get_lines, the plot has to be empty.
        # Other lines will contaminate the returns.
        x, y = seaborn.distplot(sub_samples, bins=25, color='tab:gray'
                             ).get_lines()[0].get_data()
        y_err_bar = max(y) * 1.5
        plt.errorbar([center], [y_err_bar], xerr=[[err_left], [err_right]],
                     fmt='+k', capsize=5, label='95% CI')
        # y_err_bar2 = max(y) * 1.8
        # plt.errorbar([center], [y_err_bar2], xerr=[[err_left2], [err_right2]],
        #              fmt='+k', capsize=5, label='99.5% CI')
        plt.axvline(x=0, color='k', ls=':')
        plt.ylim(0, 2*max(y))

        if row == 0 and col == 1:
          ax.tick_params(left=False, labelbottom=False, bottom=False,
                         top=True, labeltop=True)
          ax.set_xticks([-1, 0, 1])
          plt.plot([10], [0], '+k', label='median')
          plt.legend(loc=(-0.8, -1.2))
        else:
          ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(which='minor', direction='in',
                       top=False, labeltop=False, bottom=True, labelbottom=True)
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        plt.xlim(-1.05, 1.05)
        if 0 < CI_left:
          ax.set_facecolor('lightpink')
        if CI_right < 0:
          ax.set_facecolor('lightblue')

    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('save figure:', file_path)

    # If the CI is on the same side of the 0 line.
    significance = (np.sign(confidence_intervals[:,:,0]) ==
                    np.sign(confidence_intervals[:,:,1])) + 0
    significance2 = (np.sign(confidence_intervals2[:,:,0]) ==
                     np.sign(confidence_intervals2[:,:,1])) + 0
    # Mask out the significace edges.
    corr_mat = significance * centers
    corr_mat2 = significance2 * centers
    return corr_mat, corr_mat2, pval_mat


  def plot_rho(
      self,
      rho_samples,
      plot_type='rho',
      rho_true=None,
      rho_true_fix=None,
      show_whole_panel=True,
      output_dir=None,
      suffix='_rho'):
    """Plot pairwise distributions."""
    areas = ['V1', 'LM', 'AL']
    features = ['Gain', 'First peak shifting', 'Second peak shifting']
    num_qs = 3
    num_samples, rho_size, _ = rho_samples.shape

    # Generate all pairs of figures.
    for i in range(rho_size):
      for j in range(rho_size):
        if i >= j:
          continue
        # Small panel - rho values.
        plt.figure(figsize=(5,4))
        x, y = seaborn.distplot(rho_samples[:,i,j], bins=25, color='tab:gray'
            ).get_lines()[0].get_data()
        # center = x[np.argmax(y)]
        center = np.quantile(rho_samples[:,i,j], 0.5)
        CI_left = np.quantile(rho_samples[:,i,j], 0.025)
        CI_right = np.quantile(rho_samples[:,i,j], 0.975)
        plt.axvline(x=0, ymax=0.1, color='r')
        plt.axvline(x=CI_left, ymax=0.1, linestyle=':', color='k')
        plt.axvline(x=CI_right, ymax=0.1, linestyle=':', color='k', label='%95 CI')
        plt.axvline(x=center, ymax=0.1, color='g', label='median')

        # if true_model is not None:
        #   plt.axvline(x=rho_true, color='g', label='True corr')
        #   plt.axvline(x=rho_true_fix[i,j], color='g', ls='--', label='Corr of true q')
        plt.legend(loc='best')
        plt.xlim(-1, 1)
        title = (f'{areas[i // num_qs]} {features[i % num_qs]} -- ' +
                    f'{areas[j // num_qs]} {features[j % num_qs]}')
        # plt.title(title, fontsize=10)
        # plt.text(0.05, 0.7, f'median= {x[np.argmax(y)]:.2f}',
        #          transform=plt.gca().transAxes, fontsize=10)

        if output_dir and not show_whole_panel:
          file_name = f'{self.session_id}_' + title + suffix + '.pdf'
          output_figure_path = os.path.join(output_dir, file_name)
          plt.savefig(output_figure_path)
          # print('Save figure to: ', output_figure_path)
        plt.close()

    if rho_true is not None and rho_true_fix is not None:
      rhoz_true_hat = util.fisher_transform(rho_true_fix)

      # CI traps.
      CI_trap_hat = np.zeros([rho_size, rho_size])
      CI_trap_true = np.zeros([rho_size, rho_size])
      for row in range(rho_size):
        for col in range(rho_size):
          if row >= col:
            continue
          sub_samples = rho_samples[:,row,col]
          CI_left = np.quantile(sub_samples, 0.025)
          CI_right = np.quantile(sub_samples, 0.975)

          # If CI traps the true value.
          if (CI_left <= rho_true_fix[row,col] and
              CI_right >= rho_true_fix[row,col]):
            CI_trap_hat[row, col] = 1

          if CI_left <= rho_true[row,col] and CI_right >= rho_true[row,col]:
            CI_trap_true[row, col] = 1

      print('CI traps hat total, ratio:',
          np.sum(CI_trap_hat), np.sum(CI_trap_hat)/rho_size/(rho_size-1)*2)
      print('CI traps true total, ratio:',
          np.sum(CI_trap_true), np.sum(CI_trap_true)/rho_size/(rho_size-1)*2)

    # Whole panel plots.
    if show_whole_panel is False:
      return

    corr_mat = np.zeros([rho_size, rho_size])
    p_value_mat = np.ones([rho_size, rho_size])

    if plot_type == 'z':
      xlim = [-1.5, 1.5]
      xticks_labels = [-0.99, -0.95, -0.8, -0.5, 0, 0.5, 0.8, 0.95, 0.99]
      xticks = util.fisher_transform(xticks_labels)

    elif plot_type in ['rho', 'corrcoef']:
      xlim = [-1, 1]
      xticks = [-1, -0.5, 0, 0.5, 1]
      xticks_labels = xticks

    confidence_intervals = np.zeros([rho_size, rho_size, 2])
    centers = np.zeros([rho_size, rho_size])
    pval_mat = np.zeros([rho_size, rho_size])

    fig = plt.figure(figsize=(rho_size*2.5, rho_size*1.2))
    gs = fig.add_gridspec(rho_size, rho_size)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)

    for row in range(rho_size):
      for col in range(rho_size):
        if row == col:
          ax = fig.add_subplot(gs[row, col])
          plt.text(0, 0.2, areas[row // 3], ha='center', va='center')
          plt.text(0, -0.2, features[row % 3], ha='center', va='center')
          plt.xlim(-1, 1); plt.ylim(-1, 1)
          ax.set_yticklabels([])
          ax.set_xticklabels([])
          ax.axis('off')
          continue
        if row > col:
          continue

        sub_samples = rho_samples[:,row,col]
        if plot_type == 'z':
          sub_samples = util.fisher_transform(sub_samples)

        ax = fig.add_subplot(gs[row, col])

        if (row // self.num_qs + col // self.num_qs) % 2 == 0:
          color = 'tab:gray'
        else:
          color = 'tab:gray'
        x, y = seaborn.distplot(
            sub_samples, color=color, bins=30).get_lines()[0].get_data()

        CI_left = np.quantile(sub_samples, 0.025)
        CI_right = np.quantile(sub_samples, 0.975)
        # center = x[np.argmax(y)]
        mode = np.max(y)
        center = np.quantile(sub_samples, 0.5)
        samples_mean = np.mean(sub_samples)
        samples_std = np.std(sub_samples)
        alpha = scipy.stats.norm.cdf(0, loc=samples_mean, scale=samples_std)
        p_val = min(alpha, 1-alpha)  # Two tail test.

        corr_mat[row, col] = center
        p_value_mat[row, col] = scipy.stats.ttest_1samp(sub_samples, popmean=0)[1]
        confidence_intervals[[row, col], [col, row], :] = [CI_left, CI_right]
        centers[[row, col], [col, row]] = center
        pval_mat[[row, col], [col, row]] = p_val

        plt.ylim(0, mode*1.6)
        plt.axvline(x=CI_left, linestyle=':', color='k')
        plt.axvline(x=CI_right, linestyle=':', color='k', label='%95 CI')
        if rho_true is None and rho_true_fix is None:
          plt.axvline(x=center, color='g', label='median')
        # plt.axvline(np.mean(sub_samples), color='g', label='mean')
        # plt.axvline(x=0, color='b', label='null hypothesis')

        if rho_true is not None and rho_true_fix is not None:
          rhoz_true = util.fisher_transform(rho_true[row,col])
          if plot_type == 'z':
            plt.axvline(x=rhoz_true, color='g', label='True value')
            plt.axvline(x=rhoz_true_hat[row,col],
                        color='g', ls='--', label='True value')
          elif plot_type in ['rho', 'corrcoef']:
            plt.axvline(x=rho_true[row,col], color='g', label='True corr')
            plt.axvline(x=rho_true_fix[row,col],
                        color='g', ls='--', label='Corr of true q')

        ax.set_yticklabels([])
        if row != 0 or col != 1:
          # ax.set_xticklabels([])
          plt.xticks(xticks, [''] * len(xticks))
        if row == 0 and col == 1:
          plt.legend(loc=(-1,-2))
          plt.xticks(xticks, xticks_labels, rotation=-60)
        # ax.set_title(f'median= {center:.2f}', fontsize=8)
        plt.text(0.02, 0.85, f'm {center:.3f}', fontsize=8,
            transform=ax.transAxes)

        plt.xlim(xlim)

    if output_dir and show_whole_panel:
      file_name = f'{self.session_id}' + suffix + '.pdf'
      output_figure_path = os.path.join(output_dir, file_name)
      plt.savefig(output_figure_path)
      print('save figure:', output_figure_path)
    plt.show()

    # If the CI is on the same side of the 0 line.
    significance = (np.sign(confidence_intervals[:,:,0]) ==
                    np.sign(confidence_intervals[:,:,1])) + 0

    # Mask out the significace edges.
    corr_mat = significance * centers
    return corr_mat, pval_mat


  def plot_graph(
      self,
      adj_mat,
      adj_mat2=None,
      pval_mat=None,
      group_size=1,
      output_figure_path=None):
    """Plot the graph according to the adjacent matrix.

    Args:
      mat_type: 'corr', 'p_val'
      group_size: This has to be a divider of adj_mat size.
    """
    r1 = 5
    r2 = 1.5
    size = adj_mat.shape[0]
    graph = nx.from_numpy_matrix(adj_mat)
    positive_edges_weights = [(edge, w) for 
        (*edge, w) in graph.edges.data('weight') if w > 0]
    negative_edges_weights = [(edge, w) for 
        (*edge, w) in graph.edges.data('weight') if w < 0]

    # Construct nodes positions.
    if group_size == 1:
      positions = nx.circular_layout(graph)

    num_groups = np.ceil(size / group_size).astype(int)
    parent_degree = 2 * np.pi / num_groups
    child_degree = 2 * np.pi / group_size

    positions = [np.zeros(2) for _ in range(size)]
    labels = {}
    node_shapes = [0] * size
    counter = 0
    for i in range(num_groups):
      for j in range(group_size):
        positions[counter][0] = (r1 * np.cos(i*parent_degree + np.pi/2) +
                                 r2 * np.cos(j*child_degree + np.pi/3))
        positions[counter][1] = (r1 * np.sin(i*parent_degree + np.pi/2) +
                                 r2 * np.sin(j*child_degree + np.pi/3))

        if j % 3 == 0:
          labels[counter] = 'G'
        elif j % 3 == 1:
          labels[counter] = 'P1'
        elif j % 3 == 2:
          labels[counter] = 'P2'

        counter += 1

    plt.figure(figsize=(7, 5))

    nx.draw_networkx_nodes(
        graph, pos=positions, with_labels=True, node_color='lightgrey',
        node_size=800, alpha=1, node_shape='o', nodelist=[0,3,6])
    nx.draw_networkx_nodes(
        graph, pos=positions, with_labels=True, node_color='lightgrey',
        node_size=800, alpha=1, node_shape='^', nodelist=[1,4,7])
    nx.draw_networkx_nodes(
        graph, pos=positions, with_labels=True, node_color='lightgrey',
        node_size=800, alpha=1, node_shape='s', nodelist=[2,5,8])
    nx.draw_networkx_labels(graph, pos=positions, labels=labels)

    areas = ['V1', 'LM', 'AL']
    ax = plt.gca()
    plt.text(0.25, 0.8, areas[0], transform=ax.transAxes, fontsize=20)
    plt.text(0, 0.2, areas[1], transform=ax.transAxes, fontsize=20)
    plt.text(0.9, 0.2, areas[2], transform=ax.transAxes, fontsize=20)
    ax.axis('off')

    label_mark = [0, 0, 0]
    for edge, w in positive_edges_weights:
      if w > 0:
        style = 'solid'
        label = r'$\rho > 0$' if label_mark[0] == 0 else ''
        label_mark[0] = 1

      # if w > 0.8:
      #   style = 'solid'
      #   label = '1 > r > 0.8' if label_mark[0] == 0 else ''
      #   label_mark[0] = 1
      # elif w < 0.8 and w > 0.5:
      #   style = 'dashed'
      #   label = '0.5 < r < 0.8' if label_mark[1] == 0 else ''
      #   label_mark[1] = 1
      # else:
      #   style = 'dotted'
      #   label = '0 < r < -0.5' if label_mark[2] == 0 else ''
      #   label_mark[2] = 1
      if pval_mat is None and w > 0.4:
        style = 'solid'
        width = 4
      elif pval_mat is None:
        style = 'solid'
        width = 1.2
      row, col = edge
      if pval_mat is not None and pval_mat[row, col] < 0.005:
        style = 'solid'
        width = 4
      elif pval_mat is not None:
        style = 'solid'
        width = 1.2
      if adj_mat2 is not None and adj_mat2[row, col] != 0:
        style = 'solid'
        width = 4
      elif adj_mat2 is not None:
        style = 'solid'
        width = 1.2

      nx.draw_networkx_edges(
          graph, positions, edgelist=[edge], width=width,
          alpha=0.9, edge_color='r', style=style, label=label)

    label_mark = [0, 0, 0]
    for edge, w in negative_edges_weights:
      if w < 0:
        style = 'solid'
        label = r'$\rho < 0$' if label_mark[1] == 0 else ''
        label_mark[1] = 1
      # if w < -0.8:
      #   style = 'solid'
      #   label = '-1 < r < -0.8' if label_mark[0] == 0 else ''
      #   label_mark[0] = 1
      # elif w > -0.8 and w < -0.5:
      #   style = 'dashed'
      #   label = '-0.8 < r < -0.5' if label_mark[1] == 0 else ''
      #   label_mark[1] = 1
      # else:
      #   style = 'dotted'
      #   label = '0 > r > -0.5' if label_mark[2] == 0 else ''
      #   label_mark[2] = 1
      if pval_mat is None and w < -0.4:
        style = 'solid'
        width = 4
      elif pval_mat is None:
        style = 'solid'
        width = 1.2
      row, col = edge
      if pval_mat is not None and pval_mat[row, col] < 0.005:
        style = 'solid'
        width = 4
      elif pval_mat is not None:
        style = 'solid'
        width = 1.2
      if adj_mat2 is not None and adj_mat2[row, col] != 0:
        style = 'solid'
        width = 4
      elif adj_mat2 is not None:
        style = 'solid'
        width = 1.2

      nx.draw_networkx_edges(
          graph, positions, edgelist=[edge], width=width,
          alpha=0.9, edge_color='lightblue', style=style, label=label)

    plt.xlim(-10, 10)
    plt.ylim(-6, 8)
    plt.legend(loc=(0, 0.5))

    if output_figure_path is not None:
      plt.savefig(output_figure_path, bbox_inches='tight')
      print('save figure:', output_figure_path)
    plt.show()
    plt.close()


  def plot_graph_separate(
      self,
      adj_mat,
      output_figure_path=None):
    """Plot the graph according to the adjacent matrix.

    This only applies to the 3 areas BBS type of graph.

    Args:
      group_size: This has to be a divider of adj_mat size.
    """
    gs_kw = dict(width_ratios=[1,1,1], height_ratios=[1,1])
    fig, axs = plt.subplots(figsize=(8, 4), gridspec_kw=gs_kw,
        nrows=2, ncols=3)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)

    # ---------------------- G --------------------
    ax = fig.add_subplot(axs[0, 0])
    ax.tick_params(left=False, labelleft=False, labelbottom=False, bottom=False,
                   top=False, labeltop=False)

    graph = nx.from_numpy_matrix(adj_mat[np.ix_([0,3,6], [0,3,6])])
    positive_edges_weights = [(edge, w) for 
        (*edge, w) in graph.edges.data('weight') if w > 0]
    negative_edges_weights = [(edge, w) for 
        (*edge, w) in graph.edges.data('weight') if w < 0]
    labels = {0:'G', 1:'G', 2:'G'}
    r1, r2 = 5, 0
    parent_degree = 2 * np.pi / 3
    positions = [np.zeros(2) for _ in range(3)]
    for counter in range(3):
      positions[counter][0] = (r1 * np.cos(counter*parent_degree + np.pi/2))
      positions[counter][1] = (r1 * np.sin(counter*parent_degree + np.pi/2))

    nx.draw_networkx_labels(graph, pos=positions, labels=labels, font_size=8)
    nx.draw_networkx_nodes(
        graph, pos=positions, with_labels=True, node_color='lightgrey',
        node_size=200, alpha=1, node_shape='o', nodelist=[0,1,2])

    label_mark = [0, 0, 0]
    for edge, w in positive_edges_weights:
      if w > 0:
        style = 'solid'
        label = r'$\rho > 0$' if label_mark[0] == 0 else ''
        label_mark[0] = 1
      nx.draw_networkx_edges(
          graph, positions, edgelist=[edge], width=2,
          alpha=0.9, edge_color='r', style=style, label=label)

    label_mark = [0, 0, 0]
    for edge, w in negative_edges_weights:
      if w < 0:
        style = 'solid'
        label = r'$\rho < 0$' if label_mark[1] == 0 else ''
        label_mark[1] = 1
      nx.draw_networkx_edges(
          graph, positions, edgelist=[edge], width=2,
          alpha=0.9, edge_color='lightblue', style=style, label=label)
    plt.xlim(-7, 7)
    plt.ylim(-5, 6.5)
    ax.axis('off')
    plt.text(0.25, 0.8, 'V1', transform=ax.transAxes, fontsize=20)
    plt.text(0.1, 0, 'LM', transform=ax.transAxes, fontsize=20)
    plt.text(0.75, 0, 'AL', transform=ax.transAxes, fontsize=20)

    # ---------------------- P1 --------------------
    ax = fig.add_subplot(axs[0, 1])
    ax.tick_params(left=False, labelleft=False, labelbottom=False, bottom=False,
                   top=False, labeltop=False)

    graph = nx.from_numpy_matrix(adj_mat[np.ix_([1,4,7], [1,4,7])])
    positive_edges_weights = [(edge, w) for 
        (*edge, w) in graph.edges.data('weight') if w > 0]
    negative_edges_weights = [(edge, w) for 
        (*edge, w) in graph.edges.data('weight') if w < 0]
    labels = {0:'P1', 1:'P1', 2:'P1'}
    r1, r2 = 5, 0
    parent_degree = 2 * np.pi / 3
    positions = [np.zeros(2) for _ in range(3)]
    for counter in range(3):
      positions[counter][0] = (r1 * np.cos(counter*parent_degree + np.pi/2))
      positions[counter][1] = (r1 * np.sin(counter*parent_degree + np.pi/2))

    nx.draw_networkx_labels(graph, pos=positions, labels=labels, font_size=8)
    nx.draw_networkx_nodes(
        graph, pos=positions, with_labels=True, node_color='lightgrey',
        node_size=200, alpha=1, node_shape='^', nodelist=[0,1,2])

    label_mark = [0, 0, 0]
    for edge, w in positive_edges_weights:
      if w > 0:
        style = 'solid'
        label = r'$\rho > 0$' if label_mark[0] == 0 else ''
        label_mark[0] = 1
      nx.draw_networkx_edges(
          graph, positions, edgelist=[edge], width=2,
          alpha=0.9, edge_color='r', style=style, label=label)

    label_mark = [0, 0, 0]
    for edge, w in negative_edges_weights:
      if w < 0:
        style = 'solid'
        label = r'$\rho < 0$' if label_mark[1] == 0 else ''
        label_mark[1] = 1
      nx.draw_networkx_edges(
          graph, positions, edgelist=[edge], width=2,
          alpha=0.9, edge_color='lightblue', style=style, label=label)
    plt.xlim(-7, 7)
    plt.ylim(-5, 6.5)
    ax.axis('off')

    # ---------------------- P2 --------------------
    ax = fig.add_subplot(axs[0, 2])
    ax.tick_params(left=False, labelleft=False, labelbottom=False, bottom=False,
                   top=False, labeltop=False)

    graph = nx.from_numpy_matrix(adj_mat[np.ix_([2,5,8], [2,5,8])])
    positive_edges_weights = [(edge, w) for 
        (*edge, w) in graph.edges.data('weight') if w > 0]
    negative_edges_weights = [(edge, w) for 
        (*edge, w) in graph.edges.data('weight') if w < 0]
    labels = {0:'P2', 1:'P2', 2:'P2'}
    r1, r2 = 5, 0
    parent_degree = 2 * np.pi / 3
    positions = [np.zeros(2) for _ in range(3)]
    for counter in range(3):
      positions[counter][0] = (r1 * np.cos(counter*parent_degree + np.pi/2))
      positions[counter][1] = (r1 * np.sin(counter*parent_degree + np.pi/2))
    nx.draw_networkx_labels(graph, pos=positions, labels=labels, font_size=8)
    nx.draw_networkx_nodes(
        graph, pos=positions, with_labels=True, node_color='lightgrey',
        node_size=200, alpha=1, node_shape='s', nodelist=[0,1,2])
    label_mark = [0, 0, 0]
    for edge, w in positive_edges_weights:
      if w > 0:
        style = 'solid'
        label = r'$\rho > 0$' if label_mark[0] == 0 else ''
        label_mark[0] = 1
      nx.draw_networkx_edges(
          graph, positions, edgelist=[edge], width=2,
          alpha=0.9, edge_color='r', style=style, label=label)
    label_mark = [0, 0, 0]
    for edge, w in negative_edges_weights:
      if w < 0:
        style = 'solid'
        label = r'$\rho < 0$' if label_mark[1] == 0 else ''
        label_mark[1] = 1
      nx.draw_networkx_edges(
          graph, positions, edgelist=[edge], width=2,
          alpha=0.9, edge_color='lightblue', style=style, label=label)
    plt.xlim(-7, 7)
    plt.ylim(-5, 6.5)
    ax.axis('off')

    # ---------------------- G P1 --------------------
    ax = fig.add_subplot(axs[1, 0])
    ax.tick_params(left=False, labelleft=False, labelbottom=False, bottom=False,
                   top=False, labeltop=False)

    sub_adj = adj_mat[np.ix_([0,1,3,4,6,7], [0,1,3,4,6,7])]
    sub_adj[[0,0,2,2,4,4], [2,4,4,0,0,2]] = 0
    sub_adj[[1,1,3,3,5,5], [3,5,5,1,1,3]] = 0
    sub_adj[[0,1,2,3,4,5], [1,0,3,2,5,4]] = 0

    graph = nx.from_numpy_matrix(sub_adj)
    positive_edges_weights = [(edge, w) for 
        (*edge, w) in graph.edges.data('weight') if w > 0]
    negative_edges_weights = [(edge, w) for 
        (*edge, w) in graph.edges.data('weight') if w < 0]
    labels = {0:'G', 1:'P1', 2:'G', 3:'P1', 4:'G', 5:'P1'}
    r1, r2 = 5, 1.2
    parent_degree = 2 * np.pi / 3
    child_degree = 2 * np.pi / 2
    counter = 0
    positions = [np.zeros(2) for _ in range(6)]
    for i in range(3):
      for j in range(2):
        positions[counter][0] = (r1 * np.cos(i*parent_degree + np.pi/2) +
                                 r2 * np.cos((j+1)*child_degree + i*np.pi*2/3)) 
        positions[counter][1] = (r1 * np.sin(i*parent_degree + np.pi/2) +
                                 r2 * np.sin((j+1)*child_degree + i*np.pi*2/3))
        counter += 1
    nx.draw_networkx_labels(graph, pos=positions, labels=labels, font_size=8)
    nx.draw_networkx_nodes(
        graph, pos=positions, with_labels=True, node_color='lightgrey',
        node_size=200, alpha=1, node_shape='o', nodelist=[0,2,4])
    nx.draw_networkx_nodes(
        graph, pos=positions, with_labels=True, node_color='lightgrey',
        node_size=200, alpha=1, node_shape='^', nodelist=[1,3,5])

    label_mark = [0, 0, 0]
    for edge, w in positive_edges_weights:
      if w > 0:
        style = 'solid'
        label = r'$\rho > 0$' if label_mark[0] == 0 else ''
        label_mark[0] = 1
      nx.draw_networkx_edges(
          graph, positions, edgelist=[edge], width=2,
          alpha=0.9, edge_color='r', style=style, label=label)
    label_mark = [0, 0, 0]
    for edge, w in negative_edges_weights:
      if w < 0:
        style = 'solid'
        label = r'$\rho < 0$' if label_mark[1] == 0 else ''
        label_mark[1] = 1
      nx.draw_networkx_edges(
          graph, positions, edgelist=[edge], width=2,
          alpha=0.9, edge_color='lightblue', style=style, label=label)
    plt.xlim(-7, 7)
    plt.ylim(-5, 6.5)
    ax.axis('off')

    # ---------------------- G P2 --------------------
    ax = fig.add_subplot(axs[1, 1])
    ax.tick_params(left=False, labelleft=False, labelbottom=False, bottom=False,
                   top=False, labeltop=False)

    sub_adj = adj_mat[np.ix_([0,2,3,5,6,8], [0,2,3,5,6,8])]
    sub_adj[[0,0,2,2,4,4], [2,4,4,0,0,2]] = 0
    sub_adj[[1,1,3,3,5,5], [3,5,5,1,1,3]] = 0
    sub_adj[[0,1,2,3,4,5], [1,0,3,2,5,4]] = 0

    graph = nx.from_numpy_matrix(sub_adj)
    positive_edges_weights = [(edge, w) for 
        (*edge, w) in graph.edges.data('weight') if w > 0]
    negative_edges_weights = [(edge, w) for 
        (*edge, w) in graph.edges.data('weight') if w < 0]
    labels = {0:'G', 1:'P2', 2:'G', 3:'P2', 4:'G', 5:'P2'}
    r1, r2 = 5, 1.2
    parent_degree = 2 * np.pi / 3
    child_degree = 2 * np.pi / 2
    counter = 0
    positions = [np.zeros(2) for _ in range(6)]
    for i in range(3):
      for j in range(2):
        positions[counter][0] = (r1 * np.cos(i*parent_degree + np.pi/2) +
                                 r2 * np.cos((j+1)*child_degree + i*np.pi*2/3)) 
        positions[counter][1] = (r1 * np.sin(i*parent_degree + np.pi/2) +
                                 r2 * np.sin((j+1)*child_degree + i*np.pi*2/3))
        counter += 1
    nx.draw_networkx_labels(graph, pos=positions, labels=labels, font_size=8)
    nx.draw_networkx_nodes(
        graph, pos=positions, with_labels=True, node_color='lightgrey',
        node_size=200, alpha=1, node_shape='o', nodelist=[0,2,4])
    nx.draw_networkx_nodes(
        graph, pos=positions, with_labels=True, node_color='lightgrey',
        node_size=200, alpha=1, node_shape='s', nodelist=[1,3,5])

    label_mark = [0, 0, 0]
    for edge, w in positive_edges_weights:
      if w > 0:
        style = 'solid'
        label = r'$\rho > 0$' if label_mark[0] == 0 else ''
        label_mark[0] = 1
      nx.draw_networkx_edges(
          graph, positions, edgelist=[edge], width=2,
          alpha=0.9, edge_color='r', style=style, label=label)
    label_mark = [0, 0, 0]
    for edge, w in negative_edges_weights:
      if w < 0:
        style = 'solid'
        label = r'$\rho < 0$' if label_mark[1] == 0 else ''
        label_mark[1] = 1
      nx.draw_networkx_edges(
          graph, positions, edgelist=[edge], width=2,
          alpha=0.9, edge_color='lightblue', style=style, label=label)
    plt.xlim(-7, 7)
    plt.ylim(-5, 6.5)
    ax.axis('off')

    # ---------------------- P1  P2 --------------------
    ax = fig.add_subplot(axs[1, 2])
    ax.tick_params(left=False, labelleft=False, labelbottom=False, bottom=False,
                   top=False, labeltop=False)

    sub_adj = adj_mat[np.ix_([1,2,4,5,7,8], [1,2,4,5,7,8])]
    sub_adj[[0,0,2,2,4,4], [2,4,4,0,0,2]] = 0
    sub_adj[[1,1,3,3,5,5], [3,5,5,1,1,3]] = 0
    sub_adj[[0,1,2,3,4,5], [1,0,3,2,5,4]] = 0

    graph = nx.from_numpy_matrix(sub_adj)
    positive_edges_weights = [(edge, w) for 
        (*edge, w) in graph.edges.data('weight') if w > 0]
    negative_edges_weights = [(edge, w) for 
        (*edge, w) in graph.edges.data('weight') if w < 0]
    labels = {0:'P1', 1:'P2', 2:'P1', 3:'P2', 4:'P1', 5:'P2'}
    r1, r2 = 5, 1.2
    parent_degree = 2 * np.pi / 3
    child_degree = 2 * np.pi / 2
    counter = 0
    positions = [np.zeros(2) for _ in range(6)]
    for i in range(3):
      for j in range(2):
        positions[counter][0] = (r1 * np.cos(i*parent_degree + np.pi/2) +
                                 r2 * np.cos((j+1)*child_degree + i*np.pi*2/3)) 
        positions[counter][1] = (r1 * np.sin(i*parent_degree + np.pi/2) +
                                 r2 * np.sin((j+1)*child_degree + i*np.pi*2/3))
        counter += 1
    nx.draw_networkx_labels(graph, pos=positions, labels=labels, font_size=8)
    nx.draw_networkx_nodes(
        graph, pos=positions, with_labels=True, node_color='lightgrey',
        node_size=200, alpha=1, node_shape='^', nodelist=[0,2,4])
    nx.draw_networkx_nodes(
        graph, pos=positions, with_labels=True, node_color='lightgrey',
        node_size=200, alpha=1, node_shape='s', nodelist=[1,3,5])

    label_mark = [0, 0, 0]
    for edge, w in positive_edges_weights:
      if w > 0:
        style = 'solid'
        label = r'$\rho > 0$' if label_mark[0] == 0 else ''
        label_mark[0] = 1
      nx.draw_networkx_edges(
          graph, positions, edgelist=[edge], width=2,
          alpha=0.9, edge_color='r', style=style, label=label)
    label_mark = [0, 0, 0]
    for edge, w in negative_edges_weights:
      if w < 0:
        style = 'solid'
        label = r'$\rho < 0$' if label_mark[1] == 0 else ''
        label_mark[1] = 1
      nx.draw_networkx_edges(
          graph, positions, edgelist=[edge], width=2,
          alpha=0.9, edge_color='lightblue', style=style, label=label)
    plt.xlim(-7, 7)
    plt.ylim(-5, 6.5)
    ax.axis('off')

    if output_figure_path is not None:
      plt.savefig(output_figure_path, bbox_inches='tight')
      print('save figure:', output_figure_path)
    plt.show()
    plt.close()


  def plot_z_changes(
      self,
      burn_in=0,
      end=None,
      step=1,
      sub_group_df_c_true=None):
    """Plots z."""
    z_samples = self.z[burn_in:end:step]
    z_samples = np.stack(z_samples, axis=0)
    num_samples, num_conditions, num_neurons = z_samples.shape
    print(z_samples.shape)

    num_column = 10
    num_rows = np.ceil(num_conditions / num_column)

    plt.figure(figsize=(3 * num_column, 3 * num_rows))
    for c in range(num_conditions):
      plt.subplot(num_rows, num_column, c+1)
      z_diff = (np.diff(z_samples[:,c,:], axis=0) != 0) + 0
      z_diff = np.sum(z_diff, axis=1)
      plt.plot(z_diff, 'b', label='diff from last iter')
      if sub_group_df_c_true is not None:
        sub_group_df_c_true[c] = sub_group_df_c_true[c].reindex(
            self.sub_group_df_c[c].index)
        z_true = sub_group_df_c_true[c]['group_id'].values
        z_true_diff = ((z_samples[:,c,:] - z_true) != 0) + 0
        z_true_diff = np.sum(z_true_diff, axis=1)
        plt.plot(z_true_diff, 'g', label='diff from true')
      plt.title(f'Condition {c}')
    plt.legend()
    plt.show()


  def get_z_mode(
      self,
      burn_in=0,
      end=None,
      step=1,
      sub_group_df_c=None):
    """Gets the membership modes."""
    z_samples = self.z[burn_in:end:step]
    z_samples = np.stack(z_samples, axis=0)
    num_samples, num_conditions, num_neurons = z_samples.shape
    z_samples = z_samples.transpose(1,2,0)
    print('z_samples.shape', z_samples.shape)

    # Get the membership count mode for each neuron.
    units_template = sub_group_df_c[0].sort_values('group_id').index
    z_c = [pd.DataFrame() for c in range(num_conditions)]
    for c in range(num_conditions):
      z_c[c] = pd.DataFrame(z_samples[c], index=sub_group_df_c[c].index.values)
      z_c[c] = z_c[c].reindex(units_template)
      # NOTE that if there are multiple modes, it will return with multiple
      # columns. So only take the first one to avoid error in the following code.
      z_c[c] = z_c[c].mode(axis='columns', dropna=True)[0]
    return z_c


  def plot_z(
      self,
      clist,
      burn_in=0,
      end=None,
      step=1,
      sub_group_df_c=None,
      condition_ids=None,
      output_dir=None):
    """Plots z."""
    areas_names = ['V1', 'LM', 'AL']

    z_samples = self.z[burn_in:end:step]
    z_samples = np.stack(z_samples, axis=0)
    num_samples, num_conditions, num_neurons = z_samples.shape
    print('z_samples.shape', z_samples.shape)

    # Get the membership count mode for each neuron.
    probes = sub_group_df_c[0]['probe'].unique()
    z_c = self.get_z_mode(burn_in, end, step, sub_group_df_c)

    # Colormap for the areas.
    from_list = matplotlib.colors.LinearSegmentedColormap.from_list
    color_list = [0] * 3
    color_list[0] = 'k'
    color_list[1] = 'tab:blue'
    color_list[2] = 'grey'
    areas_cmap = from_list(None, color_list, self.num_groups)
    fig = plt.figure(figsize=(5 * self.num_areas, 0.2 * len(clist)))
    gs = gridspec.GridSpec(1, self.num_areas,
        width_ratios=[1,1,1], height_ratios=[1], wspace=0.03, hspace=0.1)

    membership_mat = [np.empty(0) for a, _ in enumerate(probes)]
    for a, probe in enumerate(probes):
      zlist = []
      # Setup template.
      sub_group_df = sub_group_df_c[clist[0]]
      sub_group_df = sub_group_df[sub_group_df['probe'] == probe]
      units_template = sub_group_df.sort_values('group_id').index

      for c in clist:
        z_df = z_c[c].loc[units_template.values]
        z = z_df.values.reshape(-1)
        zlist.append(z)
      membership_mat[a] = np.vstack(zlist)

      ax = plt.subplot(gs[0, a])
      if a != 1:
        seaborn.heatmap(np.vstack(zlist),
            vmin=-0.5, vmax=self.num_groups-0.5, cmap=areas_cmap, cbar=False)
        # colorbar = ax.collections[0].colorbar
        # colorbar.remove()
      elif a == 1:
        cbar_ax = fig.add_axes([.41, .06, .2, .03])
        seaborn.heatmap(np.vstack(zlist), ax=ax, cbar_ax=cbar_ax,
            vmin=-0.5, vmax=self.num_groups-0.5, cmap=areas_cmap,
            cbar_kws={'orientation': 'horizontal'})
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([0, 1, 2])
        colorbar.set_ticklabels(['cross-pop', 'local-pop-1', 'local-pop-2'])
        colorbar.ax.tick_params(labelsize=10, rotation=0)

      if a == 0 and condition_ids is not None:
        plt.yticks(np.arange(num_conditions)+0.5, condition_ids)
        ax.tick_params(top = False)
        ax.set_xticklabels([])
      elif a == 0 and condition_ids is None:
        plt.yticks(np.arange(num_conditions)+0.5, np.arange(num_conditions))
        ax.tick_params(top = False)
        ax.set_xticklabels([])
      else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(left=False, labelbottom = False, bottom=False,
                       top = False, labeltop=False)
      ax.set_ylim(len(clist), 0)

      if a == 0:
        plt.xlabel('Neuron index', fontsize=16)
        plt.ylabel('Condition index', fontsize=16)
      ax.set_title(areas_names[a], fontsize=16)

    if output_dir is not None:
      output_figure_path = os.path.join(output_dir, f'{self.session_id}_z.pdf')
      plt.savefig(output_figure_path, bbox_inches='tight')  # pad_inches=0,
      print('save figure:', output_figure_path)
    plt.show()
    plt.close()

    # Count membership portion.
    fig = plt.figure(figsize=(5 * self.num_areas, 0.3))
    gs = gridspec.GridSpec(1, self.num_areas,
        width_ratios=[1,1,1], height_ratios=[1], wspace=0.03, hspace=0.1)

    for a, probe in enumerate(probes):
      df = pd.DataFrame(membership_mat[a])
      membership_counts = df.nunique(axis=0).value_counts()[[1,2,3]]
      membership_portion = df.nunique(axis=0).value_counts()[[1,2,3]] / df.shape[1]
      ax = plt.subplot(gs[0, a])
      ax.tick_params(left=False, labelbottom=False, bottom=False,
                     top=False, labeltop=False)

      membership_portion = np.insert(membership_portion.values, 0, 0)
      membership_portion_cum = np.cumsum(membership_portion)
      for g in range(self.num_groups):
        plt.axvline(x=membership_portion_cum[g+1], color='k')
        if g == 0:
          plt.text(membership_portion_cum[g]+0.02, 0.2,
                   f'{membership_portion[g+1]*100:.1f}%', size=10, ha='left')
        elif g == 1:
          plt.text(membership_portion_cum[g]+0.02, 0.2,
                   f'{membership_portion[g+1]*100:.1f}%', size=10, ha='left')
        elif g == 2:
          plt.text(membership_portion_cum[g]-0.02, 0.2,
                   f'{membership_portion[g+1]*100:.1f}%', size=10, ha='right')

        if a == 0 and g == 0:
          plt.text(membership_portion_cum[g]+0.02, -0.7,
                   f'1 membership', size=10, ha='left')
        elif a == 0 and g == 1:
          plt.text(membership_portion_cum[g]+0.02, -0.7,
                   f'2 memberships', size=10, ha='left')
        elif a == 0 and g == 2:
          plt.text(membership_portion_cum[g]-0.2, -0.7,
                   f'3 memberships', size=10, ha='left')

      if a == 0:
        plt.ylabel(' ', fontsize=16)
        plt.yticks([0.5], [' '])
      else:
        plt.yticks([])
      plt.xlim(0, 1)

    if output_dir is not None:
      output_figure_path = os.path.join(output_dir, f'{self.session_id}_z_portion.pdf')
      plt.savefig(output_figure_path, bbox_inches='tight')  # pad_inches=0,
      print('save figure:', output_figure_path)
    plt.show()
    plt.close()


  def get_cross_pop_weight_category(
      self,
      burn_in=0,
      end=None,
      step=1,
      sub_group_df_c=None):
    """Gets the membership modes."""
    weight_threshold = [0.1, 0.9]

    z_samples = self.z[burn_in:end:step]
    z_samples = np.stack(z_samples, axis=0)
    num_samples, num_conditions, num_neurons = z_samples.shape
    z_samples = z_samples.transpose(1,2,0)
    print('z_samples.shape', z_samples.shape)

    # Get the membership count mode for each neuron.
    units_template = sub_group_df_c[0].sort_values('group_id').index
    z_c = [pd.DataFrame() for c in range(num_conditions)]
    for c in range(num_conditions):
      z_c[c] = pd.DataFrame(z_samples[c], index=sub_group_df_c[c].index.values)
      z_c[c] = z_c[c].reindex(units_template)
      # Count cross-pop membership only.
      z_ratio = (z_c[c] == 0).sum(axis=1) / num_samples
      z_ratio[z_ratio < weight_threshold[0]] = 0
      z_ratio[z_ratio >= weight_threshold[1]] = 1
      z_ratio[(z_ratio >= weight_threshold[0]) &
              (z_ratio < weight_threshold[1])] = 0.5
      z_c[c] = z_ratio
    return z_c


  def get_cross_pop_weight(
      self,
      burn_in=0,
      end=None,
      step=1,
      sub_group_df_c=None):
    """Gets the membership modes."""
    weight_threshold = [0.1, 0.9]

    z_samples = self.z[burn_in:end:step]
    z_samples = np.stack(z_samples, axis=0)
    num_samples, num_conditions, num_neurons = z_samples.shape
    z_samples = z_samples.transpose(1,2,0)
    print('z_samples.shape', z_samples.shape)

    # Get the membership count mode for each neuron.
    units_template = sub_group_df_c[0].sort_values('group_id').index
    z_c = [pd.DataFrame() for c in range(num_conditions)]
    for c in range(num_conditions):
      z_c[c] = pd.DataFrame(z_samples[c], index=sub_group_df_c[c].index.values)
      z_c[c] = z_c[c].reindex(units_template)
      # Count cross-pop membership only.
      z_ratio = (z_c[c] == 0).sum(axis=1) / num_samples
      z_c[c] = z_ratio
    return z_c


  def plot_z_cross_pop_weight(
      self,
      clist,
      burn_in=0,
      end=None,
      step=1,
      sub_group_df_c=None,
      condition_ids=None,
      output_dir=None):
    """Plots z."""
    areas_names = ['V1', 'LM', 'AL']

    z_samples = self.z[burn_in:end:step]
    z_samples = np.stack(z_samples, axis=0)
    num_samples, num_conditions, num_neurons = z_samples.shape

    probes = sub_group_df_c[0]['probe'].unique()
    z_c = self.get_cross_pop_weight_category(burn_in, end, step, sub_group_df_c)

    # Colormap for the areas.
    from_list = matplotlib.colors.LinearSegmentedColormap.from_list
    color_list = [0] * 3
    color_list[2] = 'tab:red'
    color_list[1] = 'tab:orange'
    color_list[0] = 'lightgrey'
    areas_cmap = from_list(None, color_list, self.num_groups)
    fig = plt.figure(figsize=(5 * self.num_areas, 0.2 * len(clist)))
    gs = gridspec.GridSpec(1, self.num_areas,
        width_ratios=[1,1,1], height_ratios=[1], wspace=0.03, hspace=0.1)

    for a, probe in enumerate(probes):
      zlist = []
      # Setup template.
      sub_group_df = sub_group_df_c[clist[0]]
      sub_group_df = sub_group_df[sub_group_df['probe'] == probe]
      units_template = sub_group_df.sort_values('group_id').index

      for c in clist:
        z_df = z_c[c].loc[units_template.values]
        z = z_df.values.reshape(-1)
        zlist.append(z)

      ax = plt.subplot(gs[0, a])
      if a != 1:
        seaborn.heatmap(np.vstack(zlist),
            vmin=0, vmax=1, cmap=areas_cmap, cbar=False)
        # colorbar = ax.collections[0].colorbar
        # colorbar.remove()
      elif a == 1:
        cbar_ax = fig.add_axes([.41, .06, .2, .03])
        seaborn.heatmap(np.vstack(zlist), ax=ax, cbar_ax=cbar_ax,
            vmin=0, vmax=1, cmap=areas_cmap,
            cbar_kws={'orientation': 'horizontal'})
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([1/6, 0.5, 5/6])
        colorbar.set_ticklabels(['low', 'medium', 'high'])
        colorbar.ax.tick_params(labelsize=10, rotation=0)

      if a == 0 and condition_ids is not None:
        plt.yticks(np.arange(num_conditions)+0.5, condition_ids)
        ax.tick_params(top=False, labelbottom=False, bottom=False)
        ax.set_xticklabels([])
      elif a == 0 and condition_ids is None:
        plt.yticks(np.arange(num_conditions)+0.5, np.arange(num_conditions))
        ax.tick_params(top=False, labelbottom=False, bottom=False)
        ax.set_xticklabels([])
      else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(left=False, labelbottom=False, bottom=False,
                       top=False, labeltop=False)
      ax.set_ylim(len(clist), 0)
      if a == 0:
        plt.xlabel('Neuron index', fontsize=16)
        plt.ylabel('Condition index', fontsize=16)
      ax.set_title(areas_names[a], fontsize=16)

    if output_dir is not None:
      output_figure_path = os.path.join(output_dir,
          f'{self.session_id}_z_cross_pop_weights.pdf')
      plt.savefig(output_figure_path, bbox_inches='tight')  # pad_inches=0,
      print('save figure:', output_figure_path)
    plt.show()


  def plot_z_portions(
      self,
      clist,
      burn_in=0,
      end=None,
      step=1,
      spike_counts_c=None,
      sub_group_df_c=None,
      condition_ids=None,
      output_dir=None):
    """Plots z."""
    areas_names = ['V1', 'LM', 'AL']
    z_samples = self.z[burn_in:end:step]
    z_samples = np.stack(z_samples, axis=0)
    num_samples, num_conditions, num_neurons = z_samples.shape
    probes = sub_group_df_c[0]['probe'].unique()

    z_c = self.get_z_mode(burn_in, end, step, sub_group_df_c)
    membership_mat = [np.empty(0) for a, _ in enumerate(probes)]
    spike_counts_portion = np.zeros((len(probes), num_conditions))
    for a, probe in enumerate(probes):
      zlist = []
      # Setup template.
      sub_group_df = sub_group_df_c[clist[0]]
      sub_group_df = sub_group_df[sub_group_df['probe'] == probe]
      units_template = sub_group_df.sort_values('group_id').index.values
      for c in clist:
        z_df = z_c[c].loc[units_template]
        z = z_df.values.reshape(-1)
        zlist.append(z)

        cross_pop_units = z_df[z_df == 0].index.values
        cross_pop_spike_counts = spike_counts_c[c].loc[cross_pop_units].sum().sum()
        area_spike_counts = spike_counts_c[c].loc[units_template].sum().sum()
        spike_counts_portion[a,c] = cross_pop_spike_counts / area_spike_counts
      membership_mat[a] = np.vstack(zlist)

    # -------------- Spike counts cross-pop portion. --------------
    fig = plt.figure(figsize=(5 * self.num_areas, 0.3))
    gs = gridspec.GridSpec(1, self.num_areas,
        width_ratios=[1,1,1], height_ratios=[1], wspace=0.03, hspace=0.1)

    for a, probe in enumerate(probes):
      df = pd.DataFrame(membership_mat[a])
      membership_portion = spike_counts_portion[a].mean()

      ax = plt.subplot(gs[0, a])
      ax.tick_params(left=False, labelbottom=False, bottom=False,
                     top=False, labeltop=False)
      membership_portion = np.array([0, membership_portion, 1-membership_portion])
      membership_portion_cum = np.cumsum(membership_portion)
      for g in range(2):
        plt.axvline(x=membership_portion_cum[g], color='k')
        if g == 0:
          plt.text(membership_portion_cum[g]+0.01, 0.2,
                   f'{membership_portion[g+1]*100:.1f}%', size=10, ha='left')
        elif g == 1:
          plt.text(membership_portion_cum[g]+0.01, 0.2,
                   f'{membership_portion[g+1]*100:.1f}%', size=10, ha='left')

        if a == 0 and g == 0:
          plt.text(membership_portion_cum[g]+0, -0.7,
                   f'pop', size=10, ha='left')
        elif a == 0 and g == 1:
          plt.text(membership_portion_cum[g]+0, -0.7,
                   f'non-pop', size=10, ha='left')
      if a == 0:
        plt.ylabel(' ', fontsize=16)
        plt.yticks([0.5], [' '])
        plt.title('Spike count portion (mean across conditions)',
            ha='left', x=0)
      else:
        plt.yticks([])
      plt.xlim(0, 1)

    if output_dir is not None:
      output_figure_path = os.path.join(output_dir,
          f'{self.session_id}_spk_cnt_pop_portion.pdf')
      plt.savefig(output_figure_path, bbox_inches='tight')  # pad_inches=0,
      print('save figure:', output_figure_path)
    plt.show()
    plt.close()

    # -------------- Avereage cross-pop portion. --------------
    fig = plt.figure(figsize=(5 * self.num_areas, 0.3))
    gs = gridspec.GridSpec(1, self.num_areas,
        width_ratios=[1,1,1], height_ratios=[1], wspace=0.03, hspace=0.1)

    for a, probe in enumerate(probes):
      df = pd.DataFrame(membership_mat[a])
      membership_portion = (membership_mat[a]==0).mean(axis=1).mean()

      ax = plt.subplot(gs[0, a])
      ax.tick_params(left=False, labelbottom=False, bottom=False,
                     top=False, labeltop=False)
      membership_portion = np.array([0, membership_portion, 1-membership_portion])
      membership_portion_cum = np.cumsum(membership_portion)
      for g in range(2):
        plt.axvline(x=membership_portion_cum[g], color='k')
        if g == 0:
          plt.text(membership_portion_cum[g]+0.01, 0.2,
                   f'{membership_portion[g+1]*100:.1f}%', size=10, ha='left')
        elif g == 1:
          plt.text(membership_portion_cum[g]+0.01, 0.2,
                   f'{membership_portion[g+1]*100:.1f}%', size=10, ha='left')

        if a == 0 and g == 0:
          plt.text(membership_portion_cum[g]+0, -0.7,
                   f'pop', size=10, ha='left')
        elif a == 0 and g == 1:
          plt.text(membership_portion_cum[g]+0, -0.7,
                   f'non-pop', size=10, ha='left')
      if a == 0:
        plt.ylabel(' ', fontsize=16)
        plt.yticks([0.5], [' '])
        plt.title('Neuron count portion (mean across conditions)',
            ha='left', x=0)
      else:
        plt.yticks([])
      plt.xlim(0, 1)

    if output_dir is not None:
      output_figure_path = os.path.join(output_dir,
          f'{self.session_id}_pop_portion.pdf')
      plt.savefig(output_figure_path, bbox_inches='tight')  # pad_inches=0,
      print('save figure:', output_figure_path)
    plt.show()
    plt.close()

    # -------------- Union pop portion. --------------
    fig = plt.figure(figsize=(5 * self.num_areas, 0.3))
    gs = gridspec.GridSpec(1, self.num_areas,
        width_ratios=[1,1,1], height_ratios=[1], wspace=0.03, hspace=0.1)

    for a, probe in enumerate(probes):
      df = pd.DataFrame(membership_mat[a])
      membership_portion = (membership_mat[a]==0).any(axis=0).mean()

      ax = plt.subplot(gs[0, a])
      ax.tick_params(left=False, labelbottom=False, bottom=False,
                     top=False, labeltop=False)
      membership_portion = np.array([0, membership_portion, 1-membership_portion])
      membership_portion_cum = np.cumsum(membership_portion)
      for g in range(2):
        plt.axvline(x=membership_portion_cum[g], color='k')
        if g == 0:
          plt.text(membership_portion_cum[g]+0.01, 0.2,
                   f'{membership_portion[g+1]*100:.1f}%', size=10, ha='left')
        elif g == 1:
          plt.text(membership_portion_cum[g]+0.01, 0.2,
                   f'{membership_portion[g+1]*100:.1f}%', size=10, ha='left')

        if a == 0 and g == 0:
          plt.text(membership_portion_cum[g]+0, -0.7,
                   f'pop', size=10, ha='left')
          plt.title('Neuron count portion (union across conditions)',
              ha='left', x=0)
        elif a == 0 and g == 1:
          plt.text(membership_portion_cum[g]+0, -0.7,
                   f'non-pop', size=10, ha='left')
      if a == 0:
        plt.ylabel(' ', fontsize=16)
        plt.yticks([0.5], [' '])
      else:
        plt.yticks([])
      plt.xlim(0, 1)

    if output_dir is not None:
      output_figure_path = os.path.join(output_dir,
          f'{self.session_id}_union_pop_portion.pdf')
      plt.savefig(output_figure_path, bbox_inches='tight')  # pad_inches=0,
      print('save figure:', output_figure_path)
    plt.show()
    plt.close()

    # -------------- Intersect pop portion. --------------
    fig = plt.figure(figsize=(5 * self.num_areas, 0.3))
    gs = gridspec.GridSpec(1, self.num_areas,
        width_ratios=[1,1,1], height_ratios=[1], wspace=0.03, hspace=0.1)

    for a, probe in enumerate(probes):
      df = pd.DataFrame(membership_mat[a])
      membership_portion = (membership_mat[a]==0).all(axis=0).mean()

      ax = plt.subplot(gs[0, a])
      ax.tick_params(left=False, labelbottom=False, bottom=False,
                     top=False, labeltop=False)
      membership_portion = np.array([0, membership_portion, 1-membership_portion])
      membership_portion_cum = np.cumsum(membership_portion)
      for g in range(2):
        plt.axvline(x=membership_portion_cum[g], color='k')
        if g == 0:
          plt.text(membership_portion_cum[g+1]+0.01, 0.2,
                   f'{membership_portion[g+1]*100:.1f}%', size=10, ha='left')
        elif g == 1:
          plt.text(0.8, 0.2,
                   f'{membership_portion[g+1]*100:.1f}%', size=10, ha='left')
        if a == 0 and g == 0:
          plt.text(membership_portion_cum[g]+0, -0.7,
                   f'pop', size=10, ha='left')
          plt.title('Neuron count portion (intersection across conditions)',
              ha='left', x=0)
        elif a == 0 and g == 1:
          plt.text(0.8, -0.7,
                   f'non-pop', size=10, ha='left')
      if a == 0:
        plt.ylabel(' ', fontsize=16)
        plt.yticks([0.5], [' '])
      else:
        plt.yticks([])
      plt.xlim(0, 1)

    if output_dir is not None:
      output_figure_path = os.path.join(output_dir,
          f'{self.session_id}_intersection_pop_portion.pdf')
      plt.savefig(output_figure_path, bbox_inches='tight')  # pad_inches=0,
      print('save figure:', output_figure_path)
    plt.show()
    plt.close()

    # -------------- Count each group portion portion. --------------
    fig = plt.figure(figsize=(5 * self.num_areas, 0.3))
    gs = gridspec.GridSpec(1, self.num_areas,
        width_ratios=[1,1,1], height_ratios=[1], wspace=0.03, hspace=0.1)

    for a, probe in enumerate(probes):
      df = pd.DataFrame(membership_mat[a])
      membership_counts = df.nunique(axis=0).value_counts()[[1,2,3]]
      membership_portion = df.nunique(axis=0).value_counts()[[1,2,3]] / df.shape[1]
      ax = plt.subplot(gs[0, a])
      ax.tick_params(left=False, labelbottom=False, bottom=False,
                     top=False, labeltop=False)

      membership_portion = np.insert(membership_portion.values, 0, 0)
      membership_portion_cum = np.cumsum(membership_portion)
      for g in range(self.num_groups):
        plt.axvline(x=membership_portion_cum[g+1], color='k')
        if g == 0:
          plt.text(membership_portion_cum[g]+0.01, 0.2,
                   f'{membership_portion[g+1]*100:.1f}%', size=10, ha='left')
        elif g == 1:
          plt.text(membership_portion_cum[g]+0.01, 0.2,
                   f'{membership_portion[g+1]*100:.1f}%', size=10, ha='left')
        elif g == 2:
          plt.text(membership_portion_cum[g]-0.01, 0.2,
                   f'{membership_portion[g+1]*100:.1f}%', size=10, ha='right')

        if a == 0 and g == 0:
          plt.text(membership_portion_cum[g]+0.01, -0.7,
                   f'1 membership', size=10, ha='left')
        elif a == 0 and g == 1:
          plt.text(membership_portion_cum[g]+0.01, -0.7,
                   f'2 memberships', size=10, ha='left')
        elif a == 0 and g == 2:
          plt.text(membership_portion_cum[g]-0.2, -0.7,
                   f'3 memberships', size=10, ha='left')

      if a == 0:
        plt.ylabel(' ', fontsize=16)
        plt.yticks([0.5], [' '])
      else:
        plt.yticks([])
      plt.xlim(0, 1)

    if output_dir is not None:
      output_figure_path = os.path.join(output_dir, f'{self.session_id}_z_portion.pdf')
      plt.savefig(output_figure_path, bbox_inches='tight')  # pad_inches=0,
      print('save figure:', output_figure_path)
    plt.show()
    plt.close()


  def plot_z_pop_probability_histogram(
      self,
      clist,
      burn_in=0,
      end=None,
      step=1,
      spike_counts_c=None,
      sub_group_df_c=None,
      condition_ids=None,
      output_dir=None):
    """Plots z."""
    areas_names = ['V1', 'LM', 'AL']
    z_samples = self.z[burn_in:end:step]
    z_samples = np.stack(z_samples, axis=0)
    num_samples, num_conditions, num_neurons = z_samples.shape
    probes = sub_group_df_c[0]['probe'].unique()
    z_c = self.get_cross_pop_weight(burn_in, end, step, sub_group_df_c)

    gs_kw = dict(width_ratios=[1,1,1], height_ratios=[1], wspace=0.03, hspace=0.1)
    fig, axs = plt.subplots(figsize=(5 * self.num_areas, 2), gridspec_kw=gs_kw,
        nrows=1, ncols=self.num_areas)

    for a, probe in enumerate(probes):
      zlist = []
      sub_group_df = sub_group_df_c[clist[0]]
      sub_group_df = sub_group_df[sub_group_df['probe'] == probe]
      units_template = sub_group_df.sort_values('group_id').index

      for c in clist:
        z_df = z_c[c].loc[units_template.values]
        z = z_df.values.reshape(-1)
        zlist.append(z)

      zmat = np.array(zlist)
      ax = fig.add_subplot(axs[a])
      seaborn.distplot(zmat.reshape(-1), bins=20, kde=False, color='grey',
          norm_hist=True)
      x,y = np.histogram(zmat.reshape(-1))
      print(x, x[1:].sum())
      print(y)
      ax.tick_params(left=True, labelleft=False, labelbottom=True, bottom=True,
                     top=False, labeltop=False)
      plt.ylim(0, 20)
      if a == 0:
        ax.tick_params(left=True, labelleft=True)
        plt.xlabel('\"pop\" neuron probability')
        plt.ylabel('Histogram probability')
      ax.set_title(areas_names[a], fontsize=16)

    if output_dir is not None:
      output_figure_path = os.path.join(output_dir,
          f'{self.session_id}_pop_posterior_histogram.pdf')
      plt.savefig(output_figure_path, bbox_inches='tight')  # pad_inches=0,
      print('save figure:', output_figure_path)
    plt.show()
    plt.close()


  def plot_z_pop_spk_weighted_probability_histogram(
      self,
      clist,
      burn_in=0,
      end=None,
      step=1,
      spike_counts_c=None,
      sub_group_df_c=None,
      condition_ids=None,
      output_dir=None):
    """Plots z."""
    areas_names = ['V1', 'LM', 'AL']
    z_samples = self.z[burn_in:end:step]
    z_samples = np.stack(z_samples, axis=0)
    num_samples, num_conditions, num_neurons = z_samples.shape
    probes = sub_group_df_c[0]['probe'].unique()
    z_c = self.get_cross_pop_weight(burn_in, end, step, sub_group_df_c)

    # ------------- spike prob ------------
    gs_kw = dict(width_ratios=[1,1,1], height_ratios=[1], wspace=0.2, hspace=0.1)
    fig, axs = plt.subplots(figsize=(5 * self.num_areas, 2), gridspec_kw=gs_kw,
        nrows=1, ncols=self.num_areas)

    for a, probe in enumerate(probes):
      zlist = []
      spk_cnt = []
      sub_group_df = sub_group_df_c[clist[0]]
      sub_group_df = sub_group_df[sub_group_df['probe'] == probe]
      units_template = sub_group_df.sort_values('group_id').index

      for c in clist:
        z_df = z_c[c].loc[units_template.values]
        z = z_df.values.reshape(-1)
        zlist.append(z)

        cnt_df = spike_counts_c[c].loc[units_template]
        cnt = cnt_df.sum(axis=1).values
        spk_cnt.append(cnt)

      zmat = np.array(zlist)
      spk_cnt_mat = np.array(spk_cnt)
      spk_weights = spk_cnt_mat / spk_cnt_mat.sum()
      zmat_weighted = zmat * spk_weights

      ax = fig.add_subplot(axs[a])
      seaborn.distplot(zmat_weighted.reshape(-1), bins=20, kde=False, color='grey',
          norm_hist=True)
      x,y = np.histogram(zmat_weighted.reshape(-1))
      print(x, x[1:].sum())
      print(y)
      ax.tick_params(left=True, labelleft=False, labelbottom=True, bottom=True,
                     top=False, labeltop=False)
      plt.ylim(0, 100)
      plt.xlim(0, 0.014)
      if a == 0:
        ax.tick_params(left=True, labelleft=True)
        plt.xlabel('\"pop\" neuron probability (spike count weighted)')
        plt.ylabel('Histogram probability')
      ax.set_title(areas_names[a], fontsize=16)

    if output_dir is not None:
      output_figure_path = os.path.join(output_dir,
          f'{self.session_id}_pop_posterior_spk_weighted_histogram.pdf')
      plt.savefig(output_figure_path, bbox_inches='tight')  # pad_inches=0,
      print('save figure:', output_figure_path)
    plt.show()
    plt.close()

    # ------------- spike count ------------
    gs_kw = dict(width_ratios=[1,1,1], height_ratios=[1], wspace=0.2, hspace=0.1)
    fig, axs = plt.subplots(figsize=(5 * self.num_areas, 2), gridspec_kw=gs_kw,
        nrows=1, ncols=self.num_areas)

    for a, probe in enumerate(probes):
      zlist = []
      spk_cnt = []
      sub_group_df = sub_group_df_c[clist[0]]
      sub_group_df = sub_group_df[sub_group_df['probe'] == probe]
      units_template = sub_group_df.sort_values('group_id').index

      for c in clist:
        z_df = z_c[c].loc[units_template.values]
        z = z_df.values.reshape(-1)
        zlist.append(z)

        cnt_df = spike_counts_c[c].loc[units_template]
        cnt = cnt_df.sum(axis=1).values
        spk_cnt.append(cnt)

      zmat = np.array(zlist)
      spk_cnt_mat = np.array(spk_cnt)
      spk_weights = spk_cnt_mat
      zmat_weighted = zmat * spk_weights

      ax = fig.add_subplot(axs[a])
      seaborn.distplot(zmat_weighted.reshape(-1), bins=20, kde=False, color='grey',
          norm_hist=True)
      x,y = np.histogram(zmat_weighted.reshape(-1))
      print(x, x[1:].sum())
      print(y)
      ax.tick_params(left=True, labelleft=False, labelbottom=True, bottom=True,
                     top=False, labeltop=False)
      # plt.ylim(0, 100)
      # plt.xlim(0, 0.014)
      if a == 0:
        ax.tick_params(left=True, labelleft=True)
        plt.xlabel('\"pop\" neuron probability (spike count weighted)')
        plt.ylabel('Histogram')
      ax.set_title(areas_names[a], fontsize=16)

    if output_dir is not None:
      output_figure_path = os.path.join(output_dir,
          f'{self.session_id}_pop_posterior_spk_count_histogram.pdf')
      plt.savefig(output_figure_path, bbox_inches='tight')  # pad_inches=0,
      print('save figure:', output_figure_path)
    plt.show()
    plt.close()


  def plot_z_spike_counts_vs_pop_counts(
      self,
      clist,
      burn_in=0,
      end=None,
      step=1,
      spike_counts_c=None,
      sub_group_df_c=None,
      condition_ids=None,
      output_dir=None):
    """Plots z."""
    areas_names = ['V1', 'LM', 'AL']
    z_samples = self.z[burn_in:end:step]
    z_samples = np.stack(z_samples, axis=0)
    num_samples, num_conditions, num_neurons = z_samples.shape
    probes = sub_group_df_c[0]['probe'].unique()

    z_c = self.get_z_mode(burn_in, end, step, sub_group_df_c)
    membership_mat = [np.empty(0) for a, _ in enumerate(probes)]
    spk_cnt_mat = [np.empty(0) for a, _ in enumerate(probes)]
    spike_counts_portion = np.zeros((len(probes), num_conditions))
    for a, probe in enumerate(probes):
      zlist = []
      spk_cnt = []
      # Setup template.
      sub_group_df = sub_group_df_c[clist[0]]
      sub_group_df = sub_group_df[sub_group_df['probe'] == probe]
      units_template = sub_group_df.sort_values('group_id').index.values
      for c in clist:
        z_df = z_c[c].loc[units_template]
        z = z_df.values.reshape(-1)
        zlist.append(z)

        cnt_df = spike_counts_c[c].loc[units_template]
        cnt = cnt_df.sum(axis=1).values
        spk_cnt.append(cnt)

        cross_pop_units = z_df[z_df == 0].index.values
        cross_pop_spike_counts = spike_counts_c[c].loc[cross_pop_units].sum().sum()
        area_spike_counts = spike_counts_c[c].loc[units_template].sum().sum()
        spike_counts_portion[a,c] = cross_pop_spike_counts / area_spike_counts

      membership_mat[a] = np.vstack(zlist)
      spk_cnt_mat[a] = np.vstack(spk_cnt)

    # -------------- Spike counts cross-pop portion. --------------
    fig = plt.figure(figsize=(3 * self.num_areas, 2.5))
    gs = gridspec.GridSpec(1, self.num_areas,
        width_ratios=[1,1,1], height_ratios=[1], wspace=0.3)

    for a, probe in enumerate(probes):
      df = pd.DataFrame(membership_mat[a])
      membership_portion = spike_counts_portion[a].mean()

      ax = plt.subplot(gs[0, a])
      ax.tick_params(left=False, labelbottom=True, bottom=False,
                     top=False, labeltop=False)

      cross_pop_cnt = (membership_mat[a] == 0).sum(axis=0)
      spk_cnt = spk_cnt_mat[a].sum(axis=0)
      plt.plot(cross_pop_cnt, spk_cnt, 'k.')
      ax.set_title(areas_names[a], fontsize=16)
      if a == 0:
        plt.xlabel('Number of pop')
        plt.ylabel('Spikes')

    if output_dir is not None:
      output_figure_path = os.path.join(output_dir,
          f'{self.session_id}_popcnt_vs_spkcnt.pdf')
      plt.savefig(output_figure_path, bbox_inches='tight')  # pad_inches=0,
      print('save figure:', output_figure_path)
    plt.show()
    plt.close()


  def plot_z_pop_histogram(
      self,
      clist,
      burn_in=0,
      end=None,
      step=1,
      spike_counts_c=None,
      sub_group_df_c=None,
      condition_ids=None,
      output_dir=None):
    """Plots z."""
    areas_names = ['V1', 'LM', 'AL']
    z_samples = self.z[burn_in:end:step]
    z_samples = np.stack(z_samples, axis=0)
    num_samples, num_conditions, num_neurons = z_samples.shape
    probes = sub_group_df_c[0]['probe'].unique()

    z_c = self.get_z_mode(burn_in, end, step, sub_group_df_c)
    membership_mat = [np.empty(0) for a, _ in enumerate(probes)]
    spk_cnt_mat = [np.empty(0) for a, _ in enumerate(probes)]
    spike_counts_portion = np.zeros((len(probes), num_conditions))
    for a, probe in enumerate(probes):
      zlist = []
      spk_cnt = []
      # Setup template.
      sub_group_df = sub_group_df_c[clist[0]]
      sub_group_df = sub_group_df[sub_group_df['probe'] == probe]
      units_template = sub_group_df.sort_values('group_id').index.values
      for c in clist:
        z_df = z_c[c].loc[units_template]
        z = z_df.values.reshape(-1)
        zlist.append(z)

        cnt_df = spike_counts_c[c].loc[units_template]
        cnt = cnt_df.sum(axis=1).values
        spk_cnt.append(cnt)

        cross_pop_units = z_df[z_df == 0].index.values
        cross_pop_spike_counts = spike_counts_c[c].loc[cross_pop_units].sum().sum()
        area_spike_counts = spike_counts_c[c].loc[units_template].sum().sum()
        spike_counts_portion[a,c] = cross_pop_spike_counts / area_spike_counts

      membership_mat[a] = np.vstack(zlist)
      spk_cnt_mat[a] = np.vstack(spk_cnt)

    # -------------- Spike counts cross-pop portion. --------------
    fig = plt.figure(figsize=(3 * self.num_areas, 2))
    gs = gridspec.GridSpec(1, self.num_areas,
        width_ratios=[1,1,1], height_ratios=[1], wspace=0.3)

    for a, probe in enumerate(probes):
      df = pd.DataFrame(membership_mat[a])
      membership_portion = spike_counts_portion[a].mean()

      ax = plt.subplot(gs[0, a])
      ax.tick_params(left=False, labelbottom=True, bottom=False,
                     top=False, labeltop=False)

      cross_pop_cnt = (membership_mat[a] == 0).sum(axis=0)
      seaborn.distplot(cross_pop_cnt, bins=13, color='grey', kde=False,
          norm_hist=True)
      ax.set_title(areas_names[a], fontsize=16)
      if a == 0:
        plt.xlabel('Number of pop')
        plt.ylabel('Probability')

    if output_dir is not None:
      output_figure_path = os.path.join(output_dir,
          f'{self.session_id}_pop_prob_histogram.pdf')
      plt.savefig(output_figure_path, bbox_inches='tight')  # pad_inches=0,
      print('save figure:', output_figure_path)
    plt.show()
    plt.close()

    # -------------- Spike counts cross-pop portion. --------------
    fig = plt.figure(figsize=(3 * self.num_areas, 2))
    gs = gridspec.GridSpec(2, self.num_areas,
        width_ratios=[1,1,1], height_ratios=[0.5, 1], wspace=0.3, hspace=0.15)

    for a, probe in enumerate(probes):
      df = pd.DataFrame(membership_mat[a])
      membership_portion = spike_counts_portion[a].mean()
      cross_pop_cnt = (membership_mat[a] == 0).sum(axis=0)
      count, bins = np.histogram(cross_pop_cnt, bins=np.arange(15))

      ax1 = plt.subplot(gs[0, a])
      # seaborn.distplot(cross_pop_cnt, bins=14, color='grey', kde=False,
      #     norm_hist=False)
      seaborn.barplot(bins[:-1], count, color='grey')
      plt.ylim(45, 70)
      plt.yticks([50,70], [50,70])
      ax1.spines['bottom'].set_visible(False)
      ax1.tick_params(labelbottom=False, bottom=False)
      ax1.set_title(areas_names[a], fontsize=16)

      ax2 = plt.subplot(gs[1, a])
      # seaborn.distplot(cross_pop_cnt, bins=5, color='grey', kde=False,
      #     norm_hist=False)
      seaborn.barplot(bins[:-1], count, color='grey')

      plt.ylim(0, 12)
      plt.yticks([0,5,10], [0,5,10])
      plt.xticks([0,5,10,13], [0,5,10,13])
      plt.text(0.5, 1.3, f'{len(cross_pop_cnt)} neurons', size=10,
                   rotation=0, transform=ax2.transAxes)
      ax2.spines['top'].set_visible(False)

      # Make broken ticks.
      d = .02  # how big to make the diagonal lines in axes coordinates
      kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
      ax1.plot((-d, +d), (-2*d, +2*d), **kwargs)        # top-left diagonal
      ax1.plot((1 - d, 1 + d), (-2*d, +2*d), **kwargs)  # top-right diagonal
      kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
      ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
      ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

      if a == 0:
        plt.xlabel('Number of counditions being pop')
        plt.ylabel('Neuron count')

    if output_dir is not None:
      output_figure_path = os.path.join(output_dir,
          f'{self.session_id}_pop_neuron_cnt_histogram_broken_ylim.pdf')
      plt.savefig(output_figure_path, bbox_inches='tight')  # pad_inches=0,
      print('save figure:', output_figure_path)
    plt.show()
    plt.close()


  def plot_p_gac(
      self,
      clist,
      burn_in=0,
      end=None,
      step=1,
      condition_ids=None,
      output_figure_path=None):
    """Plots p_gac."""
    from matplotlib.patches import Ellipse

    areas_names = ['V1', 'LM', 'AL']

    p_samples = self.p
    p_samples = np.stack(p_samples, axis=0)
    p_samples = p_samples[burn_in:end:step,:,:,:]
    num_samples, num_groups, num_areas, num_conditions = p_samples.shape
    print('p_samples.shape:', p_samples.shape)

    # Transform data from Dirichlet distribution to triangle coordinate.
    # cross-pop   (1,0,0) --> (-1, 0)
    # local-pop-1 (0,1,0) --> ( 1, 0)
    # local-pop-2 (0,0,1) --> ( 0, 1.72)
    M = np.array([[-1, 0], [1, 0], [0, 1.732]])

    num_block_rows = np.ceil(len(clist)/2).astype(int)
    num_block_cols = 2
    row_size = 1
    col_size = num_areas + 1
    gs_kw = dict(width_ratios=[0.3, 1,1,1] * num_block_cols,
                 height_ratios=[1] * num_block_rows)

    if len(clist) > 2:
      figure_height = len(clist)*1.1
    else:
      figure_height = 2

    fig, axs = plt.subplots(figsize=(15, figure_height), gridspec_kw=gs_kw,
        nrows=row_size * num_block_rows,
        ncols=col_size * num_block_cols)
    plt.subplots_adjust(left=None, right=None, hspace=0.2, wspace=0)

    for c_id, c in enumerate(clist):
      for a_id in range(self.num_areas + 1):
        a = a_id - 1
        row = c_id // num_block_cols * row_size
        col = c_id % num_block_cols * col_size + a_id
        if len(clist) > 2:
          ax = fig.add_subplot(axs[row, col])
        else:
          ax = fig.add_subplot(axs[col])

        if a_id == 0:
          ax.set_xticks([])
          ax.set_yticks([])
          ax.axis('off')
          continue
        if a_id == 1 and condition_ids is None:
          plt.text(-0.15, 0.9, f'condition={c}', size=10,
                   rotation=0, transform=ax.transAxes)
        elif a_id == 1 and condition_ids is not None:
          plt.text(-0.15, 0.9, f'condition={condition_ids[c]}', size=10,
                   rotation=0, transform=ax.transAxes)

        p_gac = p_samples[:,:,a,c]
        p_coor = p_gac @ M
        cov = np.cov(p_coor[:,0], p_coor[:,1])
        eigval, eigvec = np.linalg.eig(cov)
        eigval = np.sqrt(eigval)
        center_x, center_y = np.mean(p_coor[:,0]), np.mean(p_coor[:,1])
        angle = np.degrees(np.arctan2(*eigvec[:,0][::-1]))
        # This value is calculated using 2-df Chi-square distribution.
        # https://www.visiondummy.com/2014/04/
        # draw-error-ellipse-representing-covariance-matrix/
        squared_ratio = scipy.stats.chi2.ppf(0.95, df=2)
        CI95_ratio = np.sqrt(squared_ratio)

        ell = Ellipse(xy=(center_x, center_y), angle=angle,
                      width=2*eigval[0]*CI95_ratio,
                      height=2*eigval[1]*CI95_ratio,
                      fc=None, ec='k', fill=False)
        # ell.set_facecolor('none')
        ax.add_artist(ell)
        # Plot dots.
        # for s in range(num_samples):
        #   plt.plot(p_coor[s, 0], p_coor[s, 1], '.', c='k', markersize=0.2)
        plt.plot([center_x], [center_y], 'k.', label='mean')
        plt.plot([-1, 1, 0, -1], [0, 0, 1.73, 0], 'k')
        plt.xlim(-1.05, 1.05)
        plt.ylim(-0.05, 1.75)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        if c_id == 0:
          ax.set_title(areas_names[a], fontsize=16)

        if c_id == 0 and a_id == 1:
          plt.text(-0.04, 0.1, 'pop', size=10,
                   rotation=58, transform=ax.transAxes)
          plt.text(0.79, 0.13, 'local-1', size=10,
                   rotation=-58, transform=ax.transAxes)
          plt.text(0.58, 0.9, 'local-2', size=10,
                   rotation=0, transform=ax.transAxes)
        if c_id == 1 and a_id == 2:
          # Fake circle for the CI eclipse.
          plt.scatter([1000], [1000], s=80, facecolors='none', edgecolors='k', 
                      lw=0.1, label='95% credible region')
          plt.legend(loc=(-0.4, 1.05), ncol=2)

    # Clean the rest.
    for c_id in range(len(clist), num_block_cols*num_block_rows):
      for a_id in range(self.num_areas + 1):
        row = c_id // num_block_cols * row_size
        col = c_id % num_block_cols * col_size + a_id
        ax = fig.add_subplot(axs[row, col])
        ax.axis('off')

    if output_figure_path is not None:
      plt.savefig(output_figure_path, bbox_inches='tight')
      print('save figure:', output_figure_path)
    plt.show()
    plt.close()


  def plot_p_gac_dots(
      self,
      clist,
      burn_in=0,
      end=None,
      step=1):
    areas_names = ['V1', 'LM', 'AL']

    p_samples = self.p
    p_samples = np.stack(p_samples, axis=0)
    p_samples = p_samples[burn_in:end:step,:,:,:]
    num_samples, num_groups, num_areas, num_conditions = p_samples.shape
    print('p_samples.shape:', p_samples.shape)

    # Transform data from Dirichlet distribution to triangle coordinate.
    # cross-pop   (1,0,0) --> (-1, 0)
    # local-pop-1 (0,1,0) --> ( 1, 0)
    # local-pop-2 (0,0,1) --> ( 0, 1.72)
    M = np.array([[-1, 0], [1, 0], [0, 1.732]])
    colors = matplotlib.pylab.cm.jet(np.linspace(0, 1, num_samples))
    for c in clist:
      print('condition:', c)
      plt.figure(figsize=(4 * self.num_areas, 2.5))
      for a in range(self.num_areas):

        ax = plt.subplot(1, num_areas, a+1)
        p_gac = p_samples[:,:,a,c]
        p_coor = p_gac @ M
        for s in range(num_samples):
          plt.plot(p_coor[s, 0], p_coor[s, 1], '.', c=colors[s], ms=1)
        plt.plot([-1, 1, 0, -1], [0, 0, 1.73, 0], 'k')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title(f'Area {a}')
        # print('p_gac mean:', p_gac.mean(axis=0))
      plt.show()


  def get_lambda_garc(
      self,
      clist,
      spike_train_time_line,
      model_feature_type=None,
      burn_in=0,
      end=None,
      step=1,
      samples_aligned='end',
      dt=0.002,
      lambda_garc_ref=None,
      verbose=False):
    """Gets estimated intensities."""
    if samples_aligned == 'end' and model_feature_type in ['BSS']:
      total_num_samples = len(self.log_likelihood)

      sample_len = len(self.f_warp_sources)
      start = burn_in - (total_num_samples - sample_len)
      sources = self.f_warp_sources[start:end:step]

      sample_len = len(self.f_warp_targets)
      start = burn_in - (total_num_samples - sample_len)
      targets = self.f_warp_targets[start:end:step]

      sample_len = len(self.f_pop)
      start = burn_in - (total_num_samples - sample_len)
      f_pop = self.f_pop[start:end:step]

      sample_len = len(self.q)
      start = burn_in - (total_num_samples - sample_len)
      q = self.q[start:end:step]

      sources = np.stack(sources, axis=0)
      targets = np.stack(targets, axis=0)
      print('sources.shape:', sources.shape)
      print('targets.shape:', targets.shape)

    elif samples_aligned == 'end' and model_feature_type in ['B']:
      total_num_samples = len(self.log_likelihood)

      sample_len = len(self.f_pop)
      start = burn_in - (total_num_samples - sample_len)
      f_pop = self.f_pop[start:end:step]

      sample_len = len(self.q)
      start = burn_in - (total_num_samples - sample_len)
      q = self.q[start:end:step]

    else:
      sources = self.f_warp_sources[burn_in:end:step]
      targets = self.f_warp_targets[burn_in:end:step]
      f_pop = self.f_pop[burn_in:end:step]
      q = self.q[burn_in:end:step]

    f_pop = np.stack(f_pop, axis=0)
    q = np.stack(q, axis=0)
    num_bins = len(spike_train_time_line)
    num_samples, num_areas, num_trials, num_conditions = q.shape
    lambda_garc = np.zeros([self.num_groups, num_areas, num_trials,
                            num_conditions, num_bins])
    targets_arc = np.median(targets, axis=0)
    print('f_pop.shape:', f_pop.shape)
    print('q.shape:', q.shape)
    print('lambda_garc.shape:', lambda_garc.shape)
    print('targets_arc.shape', targets_arc.shape)

    for c in clist:
      for a in range(num_areas):
        for r in range(num_trials):
          for g in range(self.num_groups):
            if verbose:
              plt.figure(figsize=(16, 3))
            f_samples = np.zeros([num_samples, num_bins])
            for s in range(num_samples):
              log_lambda = f_pop[s,c,a,g]
              if g == 0 and model_feature_type in ['BSS']:
                # NOTE: in python, += is difference from x = x + val. The former
                # one DOES modify the refernece.
                log_lambda = log_lambda + q[s,a,r,c]
                f_samples[s] = hsm.HierarchicalSamplingModel.linear_time_warping(
                    spike_train_time_line, log_lambda,
                    sources[s,a,r,c], targets[s,a,r,c], verbose=False)
              elif g == 0 and model_feature_type in ['B']:
                log_lambda = log_lambda + q[s,a,r,c]
                f_samples[s] = log_lambda
              else:
                f_samples[s] = log_lambda

            # Get the mode of the posterior.
            f_samples = np.exp(f_samples)
            lambda_garc[g,a,r,c] = np.quantile(f_samples, 0.5, axis=0)
            # lambda_garc[g,a,r,c] = np.mean(f_samples, axis=0)

    return lambda_garc, targets_arc


  def get_peaks_initiations_arcs(
      self,
      clist,
      spike_train_time_line,
      burn_in=0,
      end=None,
      step=1,
      g=0,
      dt=0.002,
      initial_time_window=0.025,
      change_point_threshold=0.25,
      fit_type='refine',
      verbose=False):
    """Calculates the lead lag relations from samples.

    Args:
      fit_type: 'refine' search the peaks near the shifted landmarks.
                'simple' use the landmarks in target without searching for peaks.
      initial_time_window: The initial firing rate.
      change_point_threshold: As the name implies.
    """
    sources = self.f_warp_sources[burn_in:end:step]
    targets = self.f_warp_targets[burn_in:end:step]
    f_pop = self.f_pop[burn_in:end:step]
    sources = np.stack(sources, axis=0)
    targets = np.stack(targets, axis=0)
    f_pop = np.stack(f_pop, axis=0)
    # The initial firing rate.
    initial_index = np.where(spike_train_time_line < initial_time_window)[0][-1]
    # Change point at 25%.

    print('sources.shape:', sources.shape)
    print('targets.shape:', targets.shape)
    print('f_pop.shape:', f_pop.shape)
    num_samples, num_areas, num_trials, num_conditions, num_anchors = sources.shape

    peak1_acrs = np.zeros((num_areas, num_conditions, num_trials, num_samples))
    peak2_acrs = np.zeros((num_areas, num_conditions, num_trials, num_samples))
    initiation_acrs = np.zeros((num_areas, num_conditions, num_trials, num_samples))

    for c in clist:
      for s in range(num_samples):
        for r in range(num_trials):
          if verbose:
            plt.figure(figsize=(6, 3))

          for a in range(num_areas):
            if fit_type == 'brutal':
              log_lambda = f_pop[s,c,a,g]
              f0 = hsm.HierarchicalSamplingModel.linear_time_warping(
                  spike_train_time_line, log_lambda,
                  sources[s,a,r,c,:], targets[s,a,r,c,:], verbose=False)
              peak1_acrs[a,c,r,s] = hsm.HierarchicalSamplingModel.find_peak(
                  [0.03, 0.15], spike_train_time_line, f0)
              peak2_acrs[a,c,r,s] = hsm.HierarchicalSamplingModel.find_peak(
                  [0.15, 0.4], spike_train_time_line, f0)
            elif fit_type == 'simple':
              peak1_acrs[a,c,r,s] = targets[s,a,r,c,1]
              peak2_acrs[a,c,r,s] = targets[s,a,r,c,4]
            elif fit_type == 'refine':
              # I only take out the refine method from `get_peaks_arcs`.
              log_lambda = f_pop[s,c,a,g]
              f0 = hsm.HierarchicalSamplingModel.linear_time_warping(
                  spike_train_time_line, log_lambda,
                  sources[s,a,r,c,:], targets[s,a,r,c,:], verbose=False)
              peak1_ = targets[s,a,r,c,1]
              # Not early than 20 ms.
              time_left = np.clip(peak1_ - 0.03,
                  a_min=initial_time_window + dt, a_max=None)
              time_right = peak1_ + 0.03
              peak1_acrs[a,c,r,s] = hsm.HierarchicalSamplingModel.find_peak(
                  [time_left, time_right], spike_train_time_line, f0)
              peak2_ = targets[s,a,r,c,4]
              peak2_acrs[a,c,r,s] = hsm.HierarchicalSamplingModel.find_peak(
                  [peak2_-0.02, peak2_+0.02], spike_train_time_line, f0)

              # Find the raising point.
              initial_fr = np.median(np.exp(f0[:initial_index]) / dt)
              peak_index = int(peak1_acrs[a,c,r,s] / dt)
              peak1_fr = np.exp(f0[peak_index]) / dt
              change_point_fr = ((1 - change_point_threshold) * initial_fr +
                  change_point_threshold * peak1_fr)
              initiation_acrs[a,c,r,s] = np.interp(change_point_fr,
                  np.exp(f0[initial_index:peak_index]) / dt,
                  spike_train_time_line[initial_index:peak_index])

            if verbose and r == 12:
              plt.plot(spike_train_time_line, np.exp(f0) / dt, label=a, lw=0.3)
              peak_index = int(peak1_acrs[a,c,r,s] / dt)
              plt.plot([peak1_acrs[a,c,r,s]], [peak1_fr], 'r+')
              peak_index = int(peak2_acrs[a,c,r,s] / dt)
              plt.plot([peak2_acrs[a,c,r,s]], [np.exp(f0[peak_index]) / dt], 'r+')
              plt.plot([0, spike_train_time_line[initial_index]],
                       [initial_fr, initial_fr], 'k', lw=1)
              plt.plot([initiation_acrs[a,c,r,s]], [change_point_fr], 'g+')

              if a == 0:
                plt.axvline(x=time_left, ymin=0, ymax=0.33, linestyle=':', color='k')
                plt.axvline(x=time_right, ymin=0, ymax=0.33, linestyle=':', color='k')
              elif a == 1:
                plt.axvline(x=time_left, ymin=0.33, ymax=0.67, linestyle=':', color='k')
                plt.axvline(x=time_right, ymin=0.33, ymax=0.67, linestyle=':', color='k')
              elif a == 2:
                plt.axvline(x=time_left, ymin=0.67, ymax=1, linestyle=':', color='k')
                plt.axvline(x=time_right, ymin=0.67, ymax=1, linestyle=':', color='k')

          if verbose and r == 12:
            plt.legend()
            plt.title(f'trial {r}')

    return peak1_acrs, peak2_acrs, initiation_acrs


  def plot_lead_lag_CI(
      self,
      clist,
      spike_train_time_line,
      fit_type='simple',
      burn_in=0,
      end=None,
      step=1,
      output_dir=None,
      verbose=False):
    """Calculates the lead lag relations from samples."""
    areas_names = ['V1', 'LM', 'AL']
    dt = 0.002

    # peak1_acrs, peak2_acrs = self.get_peaks_arcs(clist, spike_train_time_line,
    #   burn_in, end, step, fit_type=fit_type, verbose=False)
    peak1_acrs, peak2_acrs, initiation_acrs = self.get_peaks_initiations_arcs(
        clist, spike_train_time_line, burn_in, end, step,
        change_point_threshold=0.5, fit_type=fit_type, verbose=False)

    # peak1_a = peak1_acrs[:,clist,:,:].reshape(self.num_areas, -1)
    # peak2_a = peak2_acrs[:,clist,:,:].reshape(self.num_areas, -1)
    # peak1_a = peak1_acrs[:,clist,:,:].mean(axis=2).reshape(self.num_areas, -1)
    # peak2_a = peak2_acrs[:,clist,:,:].mean(axis=2).reshape(self.num_areas, -1)
    peak1_a = peak1_acrs[:,clist,:,:].mean(axis=2).mean(axis=1)
    peak2_a = peak2_acrs[:,clist,:,:].mean(axis=2).mean(axis=1)
    initiation_a = initiation_acrs[:,clist,:,:].mean(axis=2).mean(axis=1)
    peak1_a = peak1_a * 1000
    peak2_a = peak2_a * 1000
    initiation_a = initiation_a * 1000
    print('peak1_a.shape', peak1_a.shape)

    # --------------- Peak-1 v.s Peak-2 self difference. ---------------
    gs_kw = dict(height_ratios=[1] * self.num_areas)
    fig, axs = plt.subplots(figsize=(4, 3), gridspec_kw=gs_kw,
        nrows=self.num_areas, ncols=1)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
    for a in range(self.num_areas):
      ax = fig.add_subplot(axs[a])
      if a == 2:
        ax.tick_params(left=False, labelleft=False, labelbottom=True, bottom=True,
                       top = False, labeltop=False, direction='in')
      else:
        ax.tick_params(left=False, labelleft=False, labelbottom=False, bottom=True,
                       top = False, labeltop=False, direction='in')
      ax.tick_params(which='minor', direction='in',
                     top=False, labeltop=False, bottom=True, labelbottom=True)
      lead_lag_self = peak2_a[a,:] - peak1_a[a,:]
      x, y = seaborn.distplot(lead_lag_self, color='grey', bins=30, kde=True
          ).get_lines()[0].get_data()
      CI_left = np.quantile(lead_lag_self, 0.025)
      CI_right = np.quantile(lead_lag_self, 0.975)
      center = np.quantile(lead_lag_self, 0.5)
      err_left = center - CI_left
      err_right = CI_right - center
      gkde=scipy.stats.gaussian_kde(lead_lag_self)
      x = np.linspace(CI_left, CI_right, 201)
      y = gkde.evaluate(x)
      mode = x[np.argmax(y)]
      print(f'CI_left {CI_left:.2f}, CI_right {CI_right:.2f}, mode {mode:.2f}')

      plt.errorbar([center], [np.max(y)*1.5], xerr=[[err_left], [err_right]],
                   fmt='+k', capsize=5, label='95% CI')
      # plt.axvline(x=CI_left, linestyle=':', color='k')
      # plt.axvline(x=CI_right, linestyle=':', color='k', label='%95 CI')
      # plt.axvline(x=center, color='g', label='mode')
      ax.xaxis.set_minor_locator(MultipleLocator(1))
      plt.xlim(130, 170)
      plt.ylim(0, 2*np.max(y))
      if a == 0:
        plt.title(f'Peak-2 - Peak-1 time lag', fontsize=16)
      if a == 2:
        plt.plot(1000, 0, 'k+', label='median')  # Fake point for plotting.
        plt.legend(loc='center right')
        plt.xlabel('Time lag [ms]')
      plt.text(0.05, 0.7, f'{areas_names[a]}', transform=ax.transAxes)
    if output_dir:
      figure_path = os.path.join(output_dir, 
          f'{self.session_id}_lead_lag_peak2_peak1.pdf')
      plt.savefig(figure_path) # bbox_inches='tight'
      print('save figure:', figure_path)
    plt.show()

    # --------------- First peak lead-lag. ---------------
    gs_kw = dict(height_ratios=[1] * self.num_areas)
    fig, axs = plt.subplots(figsize=(4, 3), gridspec_kw=gs_kw,
        nrows=self.num_areas, ncols=1)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
    for row in range(self.num_areas-1):
      for col in range(self.num_areas):
        if row >= col:
          continue
        lead_lag = peak1_a[col,:] - peak1_a[row,:]
        if areas_names[row] == 'LM' and areas_names[col] == 'AL':
          # This is required by Rob on Sep 1, 2021. 
          lead_lag = -lead_lag
        ax = fig.add_subplot(axs[col-1 + row])
        if row == 1:
          ax.tick_params(left=False, labelleft=False, labelbottom=True, bottom=True,
                         top = False, labeltop=False, direction='in')
        else:
          ax.tick_params(left=False, labelleft=False, labelbottom=False, bottom=True,
                         top = False, labeltop=False, direction='in')
        ax.tick_params(which='minor', direction='in',
                       top=False, labeltop=False, bottom=True, labelbottom=True)
        x, y = seaborn.distplot(lead_lag, color='grey', bins=30, kde=True
            ).get_lines()[0].get_data()
        CI_left = np.quantile(lead_lag, 0.025)
        CI_right = np.quantile(lead_lag, 0.975)
        center = np.quantile(lead_lag, 0.5)
        # center = x[np.argmax(y)]
        err_left = center - CI_left
        err_right = CI_right - center
        gkde=scipy.stats.gaussian_kde(lead_lag)
        x = np.linspace(CI_left, CI_right, 201)
        y = gkde.evaluate(x)
        mode = x[np.argmax(y)]
        print(f'CI_left {CI_left:.2f}, CI_right {CI_right:.2f}, mode {mode:.2f}')
        plt.errorbar([center], [np.max(y)*1.5], xerr=[[err_left], [err_right]],
                     fmt='+k', capsize=5, label='95% CI')
        plt.axvline(x=0, linestyle=':', color='k')
        # plt.axvline(x=CI_left, linestyle=':', color='k')
        # plt.axvline(x=CI_right, linestyle=':', color='k', label='%95 CI')
        # plt.axvline(x=center, color='g', label='mode')
        plt.xlim(-15, 25)
        plt.ylim(0, np.max(y)*2)
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        if col == 1:
          plt.title(f'Peak-1 time lag', fontsize=16)
        if row == 1:
          plt.plot(1000, 0, 'k+', label='median')  # Fake point for plotting.
          # plt.legend(loc='center right')
          plt.xlabel('Time lag [ms]')
        if areas_names[row] == 'LM' and areas_names[col] == 'AL':
          plt.text(0.05, 0.7, f'{areas_names[col]} leads {areas_names[row]}',
              transform=ax.transAxes)
        else:
          plt.text(0.05, 0.7, f'{areas_names[row]} leads {areas_names[col]}',
              transform=ax.transAxes)
    if output_dir:
      figure_path = os.path.join(output_dir,
          f'{self.session_id}_lead_lag_peak1.pdf')
      plt.savefig(figure_path)
      print('save figure:', figure_path)
    plt.show()

    # --------------- Second peak lead-lag. ---------------
    gs_kw = dict(height_ratios=[1] * self.num_areas)
    fig, axs = plt.subplots(figsize=(4, 3), gridspec_kw=gs_kw,
        nrows=self.num_areas, ncols=1)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
    for row in range(self.num_areas-1):
      for col in range(self.num_areas):
        if row >= col:
          continue
        lead_lag = peak2_a[col,:] - peak2_a[row,:]
        if areas_names[row] == 'LM' and areas_names[col] == 'AL':
          # This is required by Rob on Sep 1, 2021. 
          lead_lag = -lead_lag
        ax = fig.add_subplot(axs[col-1 + row])
        if row == 1:
          ax.tick_params(left=False, labelleft=False, labelbottom=True, bottom=True,
                         top = False, labeltop=False, direction='in')
        else:
          ax.tick_params(left=False, labelleft=False, labelbottom=False, bottom=True,
                         top = False, labeltop=False, direction='in')
        ax.tick_params(which='minor', direction='in',
                       top=False, labeltop=False, bottom=True, labelbottom=True)
        x, y = seaborn.distplot(lead_lag, color='grey', bins=30, kde=True
            ).get_lines()[0].get_data()
        CI_left = np.quantile(lead_lag, 0.025)
        CI_right = np.quantile(lead_lag, 0.975)
        center = np.quantile(lead_lag, 0.5)
        # center = x[np.argmax(y)]
        err_left = center - CI_left
        err_right = CI_right - center
        gkde=scipy.stats.gaussian_kde(lead_lag)
        x = np.linspace(CI_left, CI_right, 201)
        y = gkde.evaluate(x)
        mode = x[np.argmax(y)]
        print(f'CI_left {CI_left:.2f}, CI_right {CI_right:.2f}, mode {mode:.2f}')
        plt.errorbar([center], [np.max(y)*1.5], xerr=[[err_left], [err_right]],
                     fmt='+k', capsize=5, label='95% CI')
        plt.axvline(x=0, linestyle=':', color='k')
        # plt.axvline(x=CI_left, linestyle=':', color='k')
        # plt.axvline(x=CI_right, linestyle=':', color='k', label='%95 CI')
        # plt.axvline(x=center, color='g', label='mode')
        plt.xlim(-15, 25)
        plt.ylim(0, np.max(y)*2)
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        if col == 1:
          plt.title(f'Peak-2 time lag', fontsize=16)
        if row == 1:
          plt.plot(1000, 0, 'k+', label='median')  # Fake point for plotting.
          plt.legend(loc='center right')
          plt.xlabel('Time lag [ms]')
        if areas_names[row] == 'LM' and areas_names[col] == 'AL':
          plt.text(0.05, 0.7, f'{areas_names[col]} leads {areas_names[row]}',
              transform=ax.transAxes)
        else:
          plt.text(0.05, 0.7, f'{areas_names[row]} leads {areas_names[col]}',
              transform=ax.transAxes)
    if output_dir:
      figure_path = os.path.join(output_dir,
          f'{self.session_id}_lead_lag_peak2.pdf')
      plt.savefig(figure_path)
      print('save figure:', figure_path)
    plt.show()

    # --------------- Initiation lead-lag. ---------------
    gs_kw = dict(height_ratios=[1] * self.num_areas)
    fig, axs = plt.subplots(figsize=(4, 3), gridspec_kw=gs_kw,
        nrows=self.num_areas, ncols=1)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
    for row in range(self.num_areas-1):
      for col in range(self.num_areas):
        if row >= col:
          continue
        lead_lag = initiation_a[col,:] - initiation_a[row,:]
        if areas_names[row] == 'LM' and areas_names[col] == 'AL':
          # This is required by Rob on Sep 1, 2021. 
          lead_lag = -lead_lag
        ax = fig.add_subplot(axs[col-1 + row])
        if row == 1:
          ax.tick_params(left=False, labelleft=False, labelbottom=True, bottom=True,
                         top = False, labeltop=False, direction='in')
        else:
          ax.tick_params(left=False, labelleft=False, labelbottom=False, bottom=True,
                         top = False, labeltop=False, direction='in')
        ax.tick_params(which='minor', direction='in',
                       top=False, labeltop=False, bottom=True, labelbottom=True)
        x, y = seaborn.distplot(lead_lag, color='grey', bins=30, kde=True
            ).get_lines()[0].get_data()
        CI_left = np.quantile(lead_lag, 0.025)
        CI_right = np.quantile(lead_lag, 0.975)
        center = np.quantile(lead_lag, 0.5)
        # center = x[np.argmax(y)]
        gkde=scipy.stats.gaussian_kde(lead_lag)
        x = np.linspace(CI_left, CI_right, 201)
        y = gkde.evaluate(x)
        mode = x[np.argmax(y)]
        print(f'CI_left {CI_left:.2f}, CI_right {CI_right:.2f}, mode {mode:.2f}')
        err_left = center - CI_left
        err_right = CI_right - center
        plt.errorbar([center], [np.max(y)*1.5], xerr=[[err_left], [err_right]],
                     fmt='+k', capsize=5, label='95% CI')
        plt.axvline(x=0, linestyle=':', color='k')
        # plt.axvline(x=CI_left, linestyle=':', color='k')
        # plt.axvline(x=CI_right, linestyle=':', color='k', label='%95 CI')
        # plt.axvline(x=center, color='g', label='mode')
        plt.xlim(-15, 25)
        plt.ylim(0, np.max(y)*2)
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        if col == 1:
          plt.title(f'Initiation time lag', fontsize=16)
        if row == 1:
          plt.plot(1000, 0, 'k+', label='median')  # Fake point for plotting.
          plt.legend(loc='center right')
          plt.xlabel('Time lag [ms]')
        if areas_names[row] == 'LM' and areas_names[col] == 'AL':
          plt.text(0.05, 0.7, f'{areas_names[col]} leads {areas_names[row]}',
              transform=ax.transAxes)
        else:
          plt.text(0.05, 0.7, f'{areas_names[row]} leads {areas_names[col]}',
              transform=ax.transAxes)
    if output_dir:
      figure_path = os.path.join(output_dir,
          f'{self.session_id}_lead_lag_initiation.pdf')
      plt.savefig(figure_path)
      print('save figure:', figure_path)
    plt.show()

    # --------------- (X-P2-Y-P2)-(X-P1-Y-P1) ---------------
    area_pairs = [(0,1),(0,2),(1,2)]

    gs_kw = dict(height_ratios=[1] * self.num_areas)
    fig, axs = plt.subplots(figsize=(4, 3), gridspec_kw=gs_kw,
        nrows=self.num_areas, ncols=1)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
    for a, (a_0, a_1) in enumerate(area_pairs):
      ax = fig.add_subplot(axs[a])
      if a == 2:
        ax.tick_params(left=False, labelleft=False, labelbottom=True, bottom=True,
                       top = False, labeltop=False, direction='in')
      else:
        ax.tick_params(left=False, labelleft=False, labelbottom=False, bottom=True,
                       top = False, labeltop=False, direction='in')
      ax.tick_params(which='minor', direction='in',
                     top=False, labeltop=False, bottom=True, labelbottom=True)

      lead_lag = (peak2_a[a_0,:]-peak2_a[a_1,:]) - (peak1_a[a_0,:]-peak1_a[a_1,:])
      x, y = seaborn.distplot(lead_lag, color='grey', bins=30, kde=True
          ).get_lines()[0].get_data()
      CI_left = np.quantile(lead_lag, 0.025)
      CI_right = np.quantile(lead_lag, 0.975)
      center = np.quantile(lead_lag, 0.5)
      # center = x[np.argmax(y)]
      err_left = center - CI_left
      err_right = CI_right - center
      gkde=scipy.stats.gaussian_kde(lead_lag)
      x = np.linspace(CI_left, CI_right, 201)
      y = gkde.evaluate(x)
      mode = x[np.argmax(y)]
      print(f'CI_left {CI_left:.2f}, CI_right {CI_right:.2f}, mode {mode:.2f}')
      plt.errorbar([center], [np.max(y)*1.5], xerr=[[err_left], [err_right]],
                   fmt='+k', capsize=5, label='95% CI')
      # plt.axvline(x=CI_left, linestyle=':', color='k')
      # plt.axvline(x=CI_right, linestyle=':', color='k', label='%95 CI')
      # plt.axvline(x=center, color='g', label='mode')
      ax.xaxis.set_minor_locator(MultipleLocator(1))
      plt.xlim(-15, 25)
      plt.ylim(0, np.max(y)*2.8)
      plt.axvline(x=0, linestyle=':', color='k')
      if a == 0:
        plt.title(f'(X-P2 - Y-P2) - (X-P1 - Y-P1)', fontsize=16)
      if a == 2:
        plt.plot(1000, 0, 'k+', label='median')  # Fake point for plotting.
        plt.legend(loc='center right')
        plt.xlabel('Time lag [ms]')
      plt.text(0.02, 0.8, f'({areas_names[a_0]}-P2 - {areas_names[a_1]}-P2) - '+
          f'({areas_names[a_0]}-P1 - {areas_names[a_1]}-P1)',
          transform=ax.transAxes, fontsize=8)
    if output_dir:
      figure_path = os.path.join(output_dir, 
          f'{self.session_id}_lead_lag_XP2_YP2_XP1_YP1.pdf')
      plt.savefig(figure_path)
      print('save figure:', figure_path)
    plt.show()


  def plot_lead_lag(
      self,
      clist,
      spike_train_time_line,
      fit_type='simple',
      burn_in=0,
      end=None,
      step=1,
      output_dir=None,
      verbose=False):
    """Calculates the lead lag relations from samples."""
    areas_names = ['V1', 'LM', 'AL']
    dt = 0.002

    peak1_acrs, peak2_acrs, initiation_acrs = self.get_peaks_initiations_arcs(
      clist, spike_train_time_line, burn_in, end, step,
      fit_type=fit_type, verbose=False)

    # peak1_a = peak1_acrs[:,clist,:,:].reshape(self.num_areas, -1)
    # peak2_a = peak2_acrs[:,clist,:,:].reshape(self.num_areas, -1)
    # peak1_a = peak1_acrs[:,clist,:,:].mean(axis=2).reshape(self.num_areas, -1)
    # peak2_a = peak2_acrs[:,clist,:,:].mean(axis=2).reshape(self.num_areas, -1)
    peak1_a = peak1_acrs[:,clist,:,:].mean(axis=2).mean(axis=1)
    peak2_a = peak2_acrs[:,clist,:,:].mean(axis=2).mean(axis=1)
    peak1_a = peak1_a * 1000
    peak2_a = peak2_a * 1000
    print('peak1_a.shape', peak1_a.shape)

    # First peak lead-lag.
    fig = plt.figure(figsize=(4 * self.num_areas, 2 * self.num_areas))
    for row in range(self.num_areas):
      for col in range(self.num_areas):
        if row == 1 and col ==1:
          handles, labels = plot_ax.get_legend_handles_labels()
          fig.legend(handles, labels, loc=(0.22, 0.3))
        if row >= col:
          continue
        lead_lag1 = peak1_a[col,:] - peak1_a[row,:]
        ax = plt.subplot(self.num_areas, self.num_areas, row * self.num_areas + col + 1)
        ax.tick_params(left=False, labelbottom=True, bottom=True,
                       top = False, labeltop=False)
        ax.tick_params(which='minor',
                       top=False, labeltop=False, bottom=True, labelbottom=True)
        x, y = seaborn.distplot(lead_lag1, bins=30, kde=True).get_lines()[0].get_data()
        left_CI = np.quantile(lead_lag1, 0.025)
        right_CI = np.quantile(lead_lag1, 0.975)
        center = np.quantile(lead_lag1, 0.5)
        # center = x[np.argmax(y)]
        plt.axvline(x=left_CI, linestyle=':', color='k')
        plt.axvline(x=right_CI, linestyle=':', color='k', label='%95 CI')
        plt.axvline(x=center, color='g', label='mode')
        plt.xticks(np.arange(-100, 100, 5))
        plt.xlim(-10, 25)
        plt.xlabel('Time difference [ms]')
        plt.title(f'Peak1 {areas_names[col]} - {areas_names[row]}')
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        if row == 0 and col == 1:
          plot_ax = ax
    plt.tight_layout()
    plt.show()

    # Plot all pairs. First peak lead-lag.
    for row in range(self.num_areas):
      for col in range(self.num_areas):
        if row >= col:
          continue
        fig = plt.figure(figsize=(5, 4))
        ax = plt.gca()
        lead_lag1 = peak1_a[col,:] - peak1_a[row,:]
        ax.tick_params(left=False, labelbottom=True, bottom=True,
                       top = False, labeltop=False)
        ax.tick_params(which="minor",
                       top=False, labeltop=False, bottom=True, labelbottom=True)
        x, y = seaborn.distplot(lead_lag1, bins=30, kde=True,
            color='tab:grey').get_lines()[0].get_data()
        left_CI = np.quantile(lead_lag1, 0.025)
        right_CI = np.quantile(lead_lag1, 0.975)
        center = np.quantile(lead_lag1, 0.5)
        # center = x[np.argmax(y)]
        plt.axvline(x=left_CI, ymax=0.1, linestyle=':', color='k', lw=2)
        plt.axvline(x=right_CI, ymax=0.1, linestyle=':', color='k', label='%95 CI', lw=2)
        plt.axvline(x=center, ymax=0.1, color='g', label='median', lw=2)
        plt.axvline(x=0, ymax=0.1, color='r')
        plt.xticks(np.arange(-100, 100, 5))
        plt.xlim(-10, 30)
        plt.xlabel('Time difference [ms]', fontsize=16)
        # plt.title(f'Peak1 {areas_names[col]} - {areas_names[row]}')
        plt.legend(fontsize=12)
        ax.xaxis.set_minor_locator(MultipleLocator(1))

        if output_dir is not None:
          figure_path = os.path.join(output_dir,
              f'peak-1_{areas_names[col]}_{areas_names[row]}_lead_lag.pdf')
          plt.savefig(figure_path)
          print('save figure:', figure_path)
          plt.close()
        else:
          plt.show()


    # Second peak lead-lag.
    fig = plt.figure(figsize=(4 * self.num_areas, 2 * self.num_areas))
    for row in range(self.num_areas):
      for col in range(self.num_areas):
        if row == 1 and col ==1:
          handles, labels = plot_ax.get_legend_handles_labels()
          fig.legend(handles, labels, loc=(0.22, 0.3))
        if row >= col:
          continue
        lead_lag2 = peak2_a[col,:] - peak2_a[row,:]
        ax = plt.subplot(self.num_areas, self.num_areas, row * self.num_areas + col + 1)
        x, y = seaborn.distplot(lead_lag2, bins=30, kde=True).get_lines()[0].get_data()
        left_CI = np.quantile(lead_lag2, 0.025)
        right_CI = np.quantile(lead_lag2, 0.975)
        center = np.quantile(lead_lag2, 0.5)
        # center = x[np.argmax(y)]
        plt.axvline(x=left_CI, linestyle=':', color='k')
        plt.axvline(x=right_CI, linestyle=':', color='k', label='%95 CI')
        plt.axvline(x=center, color='g', label='mode')
        plt.xlim(-15, 25)
        plt.title(f'Peak2 {areas_names[col]} - {areas_names[row]}')
        if row == 0 and col == 1:
          plot_ax = ax
    plt.tight_layout()
    plt.show()

    if verbose:
      for r in range(num_trials):
        peak2_a = peak2_acrs[:,clist,r,:].reshape(self.num_areas, -1)

        plt.figure(figsize=(2 * self.num_areas, 2 * self.num_areas))
        for row in range(self.num_areas):
          for col in range(self.num_areas):
            if row >= col:
              continue
            lead_lag2 = peak2_a[col,:] - peak2_a[row,:]
            plt.subplot(self.num_areas, self.num_areas, row * self.num_areas + col + 1)
            seaborn.distplot(lead_lag2, bins=40, kde=True)
            plt.xlim(-0.12, 0.12)
            plt.title(f'r:{r}  Peak2 {areas_names[col]} - {areas_names[row]}')
        plt.tight_layout()
        plt.show()


  def plot_peaks(
      self,
      clist,
      spike_train_time_line,
      burn_in=0,
      end=None,
      step=1,
      fit_type='simple',
      verbose=False,
      output_dir=None):
    """Calculates the lead lag relations from samples.

    Args:
      fit_type: 'brutal', 'simple', 'refine'.
    """
    areas_names = ['V1', 'LM', 'AL']
    dt = 0.002

    peak1_acrs, peak2_acrs, initiation_acrs = self.get_peaks_initiations_arcs(
      clist, spike_train_time_line, burn_in, end, step,
      fit_type=fit_type, verbose=False)

    peak1_a = peak1_acrs[:,clist,:,:].reshape(self.num_areas, -1)
    peak2_a = peak2_acrs[:,clist,:,:].reshape(self.num_areas, -1)

    # First peak marginal distribution.
    fig = plt.figure(figsize=(3 * self.num_areas, 3))
    for a in range(self.num_areas):
      ax = plt.subplot(1, self.num_areas, a + 1)
      x, y = seaborn.distplot(peak1_a[a], bins=22, kde=True).get_lines()[0].get_data()
      CI_left = np.nanquantile(peak1_a[a], 0.025)
      CI_right = np.nanquantile(peak1_a[a], 0.975)
      mode = x[np.argmax(y)]

      plt.axvline(x=CI_left, linestyle=':', color='k')
      plt.axvline(x=CI_right, linestyle=':', color='k', label='%95 CI')
      plt.axvline(x=mode, color='g', label='mode')
      plt.title(f'Peak1 {areas_names[a]}')
      plt.xlim(0, 0.15)
      plt.xlabel('Time [sec]')
      # if a == 0:
      #   plt.legend()
    plt.tight_layout()
    if output_dir:
      if hasattr(self, 'session_id'):
        session_id = self.session_id
      else:
        session_id = 0
      figure_path = os.path.join(output_dir, 
          f'{session_id}_model_peak1_distribution.pdf')
      plt.savefig(figure_path)
      print('save figure:', figure_path)
    plt.show()

    # Second peak marginal distribution.
    fig = plt.figure(figsize=(3 * self.num_areas, 3))
    for a in range(self.num_areas):
      ax = plt.subplot(1, self.num_areas, a + 1)
      x, y = seaborn.distplot(peak2_a[a], bins=22, kde=True).get_lines()[0].get_data()
      CI_left = np.nanquantile(peak2_a[a], 0.025)
      CI_right = np.nanquantile(peak2_a[a], 0.975)
      mode = x[np.argmax(y)]

      plt.axvline(x=CI_left, linestyle=':', color='k')
      plt.axvline(x=CI_right, linestyle=':', color='k', label='%95 CI')
      plt.axvline(x=mode, color='g', label='mode')
      plt.title(f'Peak1 {areas_names[a]}')
      plt.xlim(0.1, 0.4)
      plt.xlabel('Time [sec]')
      # if a == 0:
      #   plt.legend()
    plt.tight_layout()
    if output_dir:
      if hasattr(self, 'session_id'):
        session_id = self.session_id
      else:
        session_id = 0
      figure_path = os.path.join(output_dir, 
          f'{session_id}_model_peak2_distribution.pdf')
      plt.savefig(figure_path)
      print('save figure:', figure_path)
    plt.show()

    # Centers + CIs.
    centers = np.zeros(self.num_areas)
    CIs = np.zeros([self.num_areas, 2])
    # First peak only CI.
    fig = plt.figure(figsize=(3, 3))
    for a in range(self.num_areas):
      data = peak1_a[a].reshape(-1)
      data = data[~np.isnan(data)]
      gkde=scipy.stats.gaussian_kde(data)
      CI_left = np.nanquantile(peak1_a[a], 0.025)
      CI_right = np.nanquantile(peak1_a[a], 0.975)
      x = np.linspace(CI_left, CI_right, 201)
      y = gkde.evaluate(x)
      # center = x[np.argmax(y)]  # mode.
      center = np.nanquantile(peak1_a[a], 0.5)  # median.
      centers[a] = center
      CIs[a] = [CI_left, CI_right]
      plt.plot([CI_left, CI_right], [a, a], 'k')
      plt.plot(center, a, 'kx')
      plt.text(center, a+0.2, f'{center}')
      plt.title(f'Peak1 {areas_names[a]}')
      plt.xlim(0, 0.15)
      plt.xlabel('Time [sec]')
      # if a == 0:
      #   plt.legend()
    plt.tight_layout()
    if output_dir:
      if hasattr(self, 'session_id'):
        session_id = self.session_id
      else:
        session_id = 0
      figure_path = os.path.join(output_dir, 
          f'{session_id}_model_peak1_median_CI.pdf')
      plt.savefig(figure_path)
      print('save figure:', figure_path)
    plt.show()

    return centers, CIs


  def plot_trials_peaks_dots(
      self,
      clist,
      spike_train_time_line,
      burn_in=0,
      end=None,
      step=1,
      fit_type='refine',
      verbose=False,
      output_dir=None):
    """Calculates the lead lag relations from samples.

    Args:
      fit_type: 'brutal', 'simple', 'refine'.
    """
    areas_names = ['V1', 'LM', 'AL']
    dt = 0.002

    q_shift1 = np.stack(self.q_shift1[burn_in:end:step], axis=0)
    q_shift1_median = np.quantile(q_shift1, 0.5, axis=0)
    q_shift1_median = q_shift1_median.transpose(0,2,1).reshape(self.num_areas, -1)
    q_shift1_median = q_shift1_median * 1000

    q_shift2 = np.stack(self.q_shift2[burn_in:end:step], axis=0)
    q_shift2_median = np.quantile(q_shift2, 0.5, axis=0) * 1000
    q_shift2_median = q_shift2_median.transpose(0,2,1).reshape(self.num_areas, -1)
    q_shift2_median = q_shift2_median * 1000

    peak1_acrs, peak2_acrs, initiation_acrs = self.get_peaks_initiations_arcs(
        clist, spike_train_time_line, burn_in, end, step,
        fit_type=fit_type, verbose=False)
    peak1_acr = np.quantile(peak1_acrs[:,clist,:,:], 0.5, axis=3)
    peak1_ar = peak1_acr.reshape(self.num_areas, -1) * 1000
    peak1_ac = np.mean(peak1_acr, axis=2, keepdims=True) * 1000
    peak1_a = np.mean(peak1_ar, axis=1, keepdims=True)
    peak2_acr = np.quantile(peak2_acrs[:,clist,:,:], 0.5, axis=3)
    peak2_ar = peak2_acr.reshape(self.num_areas, -1) * 1000
    peak2_ac = np.mean(peak2_acr, axis=2, keepdims=True) * 1000
    peak2_a = np.mean(peak2_ar, axis=1, keepdims=True)

    q_shift1 = q_shift1_median + peak1_a
    q_shift2 = q_shift2_median + peak2_a
    print('peak1_acrs.shape:', peak1_acrs.shape)
    print('peak1_ar.shape:', peak1_ar.shape)
    print('peak1_ac.shape', peak1_ac.shape)
    print('peak1_a.shape', peak1_a.shape)
    print('q_shift1_median.shape', q_shift1_median.shape)

    # # First peak marginal distribution.
    # fig = plt.figure(figsize=(3 * self.num_areas, 2.5 * self.num_areas))
    # for row in range(self.num_areas):
    #   for col in range(self.num_areas):
    #     if row >= col:
    #       continue
    #     ax = plt.subplot(self.num_areas, self.num_areas, row * self.num_areas + col + 1)
    #     plt.plot(peak1_ar[row], peak1_ar[col], '.')
    #     plt.plot([0, 150], [0, 150], ':k')
    #     plt.xlabel(f'{areas_names[col]}  post-stim time [ms]', fontsize=16)
    #     plt.ylabel(f'{areas_names[row]}  post-stim time [ms]', fontsize=16)
    # plt.tight_layout()
    # plt.show()

    # # Second peak marginal distribution.
    # fig = plt.figure(figsize=(3 * self.num_areas, 2.5 * self.num_areas))
    # for row in range(self.num_areas):
    #   for col in range(self.num_areas):
    #     if row >= col:
    #       continue
    #     ax = plt.subplot(self.num_areas, self.num_areas, row * self.num_areas + col + 1)
    #     plt.plot(peak2_ar[col], peak2_ar[row], '.')
    #     plt.plot([120, 500], [120, 500], ':k')
    #     plt.xlabel(f'{areas_names[col]}  post-stim time [ms]', fontsize=16)
    #     plt.ylabel(f'{areas_names[row]}  post-stim time [ms]', fontsize=16)
    # plt.tight_layout()
    # plt.show()

    # Demo P1. Peaks position searched by curves.
    # col, row = 1, 2
    # features_1, features_2 = peak1_ar[col], peak1_ar[row]
    # corr = scipy.stats.pearsonr(features_1, features_2)
    # plt.figure(figsize=(3.5, 3))
    # ax = plt.gca()
    # ax.tick_params(left=True, labelbottom=True, bottom=True,
    #                top=False, labeltop=False)
    # plt.plot(features_1, features_2, '.k')
    # plt.plot([0, 500], [0, 500], ls='--', c='lightgrey')
    # plt.axis('equal')
    # plt.axis('square')
    # plt.xlabel(f'{areas_names[col]}-Peak-1 time [ms]', fontsize=12)
    # plt.ylabel(f'{areas_names[row]}-Peak-1 time [ms]', fontsize=12)
    # ax.xaxis.set_tick_params(labelsize=10)
    # ax.yaxis.set_tick_params(labelsize=10)
    # # ax.set_title('Statistical model', fontsize=16)
    # plt.text(0.7, 0.92, fr'$\rho$={corr[0]:.2f}', transform=ax.transAxes)
    # plt.xticks(np.arange(0, 1000, 50))
    # plt.yticks(np.arange(0, 1000, 50))
    # plt.xlim(0, 150)
    # plt.ylim(0, 150)
    # plt.tight_layout()
    # if output_dir is not None:
    #   output_figure_path = os.path.join(output_dir,
    #       f'{self.session_id}_{areas_names[col]}_{areas_names[row]}_' +
    #       'peak1_model_posterior.pdf')
    #   plt.savefig(output_figure_path, bbox_inches='tight')  # pad_inches=0,
    #   print('save figure:', output_figure_path)
    # plt.show()

    # Demo P1. peak-1 position aligned by condition center.
    # col, row = 1, 2
    # features_1, features_2 = q_shift1[col], q_shift1[row]
    # corr = scipy.stats.pearsonr(features_1, features_2)
    # plt.figure(figsize=(3.5, 3))
    # ax = plt.gca()
    # ax.tick_params(left=True, labelbottom=True, bottom=True,
    #                top=False, labeltop=False)
    # plt.plot(features_1, features_2, '.k')
    # plt.plot([0, 500], [0, 500], ls='--', c='lightgrey')
    # plt.axis('equal')
    # plt.axis('square')
    # plt.xlabel(f'{areas_names[col]}-Peak-1 time [ms]', fontsize=12)
    # plt.ylabel(f'{areas_names[row]}-Peak-1 time [ms]', fontsize=12)
    # ax.xaxis.set_tick_params(labelsize=10)
    # ax.yaxis.set_tick_params(labelsize=10)
    # # ax.set_title('Statistical model', fontsize=16)
    # plt.text(0.7, 0.92, fr'$\rho$={corr[0]:.2f}', transform=ax.transAxes)
    # plt.xticks(np.arange(-500, 500, 50))
    # plt.yticks(np.arange(-500, 500, 50))
    # plt.xlim(25, 100)
    # plt.ylim(25, 100)
    # plt.tight_layout()
    # if output_dir is not None:
    #   output_figure_path = os.path.join(output_dir,
    #       f'{self.session_id}_{areas_names[col]}_{areas_names[row]}_' +
    #       'peak1_model_posterior.pdf')
    #   plt.savefig(output_figure_path, bbox_inches='tight')  # pad_inches=0,
    #   print('save figure:', output_figure_path)
    # plt.show()

    # Demo P2.
    col, row = 1, 0
    features_1, features_2 = peak2_ar[col], peak2_ar[row]
    corr = scipy.stats.pearsonr(features_1, features_2)
    plt.figure(figsize=(3.5, 3))
    ax = plt.gca()
    ax.tick_params(left=True, labelbottom=True, bottom=True,
                   top=False, labeltop=False)
    features_1 = features_1[features_1 < 350]
    features_2 = features_2[features_2 < 350]
    plt.plot(features_1, features_2, '.k')
    plt.plot([0, 500], [0, 500], ls='--', c='lightgrey')
    plt.axis('equal')
    plt.axis('square')
    plt.xlabel(f'{areas_names[col]}-Peak-2 time [ms]', fontsize=12)
    plt.ylabel(f'{areas_names[row]}-Peak-2 time [ms]', fontsize=12)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    # ax.set_title('Statistical model', fontsize=16)
    plt.text(0.7, 0.92, fr'$\rho$={corr[0]:.2f}', transform=ax.transAxes)
    plt.xticks(np.arange(0, 1000, 100))
    plt.yticks(np.arange(0, 1000, 100))
    plt.xlim(100, 400)
    plt.ylim(100, 400)
    plt.tight_layout()
    if output_dir is not None:
      output_figure_path = os.path.join(output_dir,
          f'{self.session_id}_{areas_names[col]}_{areas_names[row]}_' +
          'peak2_model_posterior.pdf')
      plt.savefig(output_figure_path)  # bbox_inches='tight' pad_inches=0,
      print('save figure:', output_figure_path)
    plt.show()


  def plot_trials_lead_lag_dots(
      self,
      clist,
      spike_train_time_line,
      burn_in=0,
      end=None,
      step=1,
      fit_type='refine',
      verbose=False,
      output_dir=None):
    """Calculates the lead lag relations from samples.

    Args:
      fit_type: 'brutal', 'simple', 'refine'.
    """
    areas_names = ['V1', 'LM', 'AL']
    dt = 0.002

    q_shift1 = np.stack(self.q_shift1[burn_in:end:step], axis=0)
    q_shift1_median = np.quantile(q_shift1, 0.5, axis=0)
    q_shift1_median = q_shift1_median.transpose(0,2,1).reshape(self.num_areas, -1)
    q_shift1_median = q_shift1_median * 1000

    q_shift2 = np.stack(self.q_shift2[burn_in:end:step], axis=0)
    q_shift2_median = np.quantile(q_shift2, 0.5, axis=0) * 1000
    q_shift2_median = q_shift2_median.transpose(0,2,1).reshape(self.num_areas, -1)
    q_shift2_median = q_shift2_median * 1000

    peak1_acrs, peak2_acrs, initiation_acrs = self.get_peaks_initiations_arcs(
        clist, spike_train_time_line, burn_in, end, step,
        fit_type=fit_type, verbose=False)
    peak1_acr = np.quantile(peak1_acrs[:,clist,:,:], 0.5, axis=3)
    peak1_ar = peak1_acr.reshape(self.num_areas, -1) * 1000
    peak1_ac = np.mean(peak1_acr, axis=2, keepdims=True) * 1000
    peak1_a = np.mean(peak1_ar, axis=1, keepdims=True)
    peak2_acr = np.quantile(peak2_acrs[:,clist,:,:], 0.5, axis=3)
    peak2_ar = peak2_acr.reshape(self.num_areas, -1) * 1000
    peak2_ac = np.mean(peak2_acr, axis=2, keepdims=True) * 1000
    peak2_a = np.mean(peak2_ar, axis=1, keepdims=True)

    q_shift1 = q_shift1_median + peak1_a
    q_shift2 = q_shift2_median + peak2_a
    print('peak1_acrs.shape:', peak1_acrs.shape)
    print('peak1_ar.shape:', peak1_ar.shape)
    print('peak1_ac.shape', peak1_ac.shape)
    print('peak1_a.shape', peak1_a.shape)
    print('q_shift1_median.shape', q_shift1_median.shape)

    # # First peak marginal distribution.
    # fig = plt.figure(figsize=(3 * self.num_areas, 2.5 * self.num_areas))
    # for row in range(self.num_areas):
    #   for col in range(self.num_areas):
    #     if row >= col:
    #       continue
    #     ax = plt.subplot(self.num_areas, self.num_areas, row * self.num_areas + col + 1)
    #     plt.plot(peak1_ar[row], peak1_ar[col], '.')
    #     plt.plot([0, 150], [0, 150], ':k')
    #     plt.xlabel(f'{areas_names[col]}  post-stim time [ms]', fontsize=16)
    #     plt.ylabel(f'{areas_names[row]}  post-stim time [ms]', fontsize=16)
    # plt.tight_layout()
    # plt.show()

    # # Second peak marginal distribution.
    # fig = plt.figure(figsize=(3 * self.num_areas, 2.5 * self.num_areas))
    # for row in range(self.num_areas):
    #   for col in range(self.num_areas):
    #     if row >= col:
    #       continue
    #     ax = plt.subplot(self.num_areas, self.num_areas, row * self.num_areas + col + 1)
    #     plt.plot(peak2_ar[col], peak2_ar[row], '.')
    #     plt.plot([120, 500], [120, 500], ':k')
    #     plt.xlabel(f'{areas_names[col]}  post-stim time [ms]', fontsize=16)
    #     plt.ylabel(f'{areas_names[row]}  post-stim time [ms]', fontsize=16)
    # plt.tight_layout()
    # plt.show()

    # Demo P1. Peaks position searched by curves.
    # col, row = 1, 2
    # features_1, features_2 = peak1_ar[col], peak1_ar[row]
    # corr = scipy.stats.pearsonr(features_1, features_2)
    # plt.figure(figsize=(3.5, 3))
    # ax = plt.gca()
    # ax.tick_params(left=True, labelbottom=True, bottom=True,
    #                top=False, labeltop=False)
    # plt.plot(features_1, features_2, '.k')
    # plt.plot([0, 500], [0, 500], ls='--', c='lightgrey')
    # plt.axis('equal')
    # plt.axis('square')
    # plt.xlabel(f'{areas_names[col]}-Peak-1 time [ms]', fontsize=12)
    # plt.ylabel(f'{areas_names[row]}-Peak-1 time [ms]', fontsize=12)
    # ax.xaxis.set_tick_params(labelsize=10)
    # ax.yaxis.set_tick_params(labelsize=10)
    # # ax.set_title('Statistical model', fontsize=16)
    # plt.text(0.7, 0.92, fr'$\rho$={corr[0]:.2f}', transform=ax.transAxes)
    # plt.xticks(np.arange(0, 1000, 50))
    # plt.yticks(np.arange(0, 1000, 50))
    # plt.xlim(0, 150)
    # plt.ylim(0, 150)
    # plt.tight_layout()
    # if output_dir is not None:
    #   output_figure_path = os.path.join(output_dir,
    #       f'{self.session_id}_{areas_names[col]}_{areas_names[row]}_' +
    #       'peak1_model_posterior.pdf')
    #   plt.savefig(output_figure_path, bbox_inches='tight')  # pad_inches=0,
    #   print('save figure:', output_figure_path)
    # plt.show()

    # Demo P2.
    col, row = 1, 0
    features_1, features_2 = peak2_ar[col], peak2_ar[row]


    plt.figure(figsize=(3.5, 3))
    ax = plt.gca()
    ax.tick_params(left=True, labelbottom=True, bottom=True,
                   top=False, labeltop=False)
    features_1 = features_1[features_1 < 350]
    features_2 = features_2[features_2 < 350]
    leads = features_1 - features_2
    seaborn.distplot(leads, bins=30, color='grey')

    # plt.plot([0, 500], [0, 500], ls='--', c='lightgrey')
    # plt.axis('equal')
    # plt.axis('square')
    # plt.xlabel(f'{areas_names[col]}-Peak-2 time [ms]', fontsize=12)
    # plt.ylabel(f'{areas_names[row]}-Peak-2 time [ms]', fontsize=12)
    # ax.xaxis.set_tick_params(labelsize=10)
    # ax.yaxis.set_tick_params(labelsize=10)
    # # ax.set_title('Statistical model', fontsize=16)
    # plt.text(0.7, 0.92, fr'$\rho$={corr[0]:.2f}', transform=ax.transAxes)
    # plt.xticks(np.arange(0, 1000, 100))
    # plt.yticks(np.arange(0, 1000, 100))
    # plt.xlim(100, 400)
    # plt.ylim(100, 400)

    plt.tight_layout()
    if output_dir is not None:
      output_figure_path = os.path.join(output_dir,
          f'{self.session_id}_{areas_names[col]}_{areas_names[row]}_' +
          'peak2_model_posterior_lead_lag_dots.pdf')
      plt.savefig(output_figure_path)  # bbox_inches='tight' pad_inches=0,
      print('save figure:', output_figure_path)
    plt.show()


  def plot_trials_peaks_dots_by_conditions(
      self,
      clist,
      spike_train_time_line,
      burn_in=0,
      end=None,
      step=1,
      fit_type='refine',
      verbose=False,
      output_dir=None):
    """Calculates the lead lag relations from samples.

    Args:
      fit_type: 'brutal', 'simple', 'refine'.
    """
    areas_names = ['V1', 'LM', 'AL']

    peak1_acrs, peak2_acrs, initiation_acrs = self.get_peaks_initiations_arcs(
        clist, spike_train_time_line, burn_in, end, step,
        fit_type=fit_type, verbose=False)
    peak1_ar = np.quantile(peak1_acrs[:,clist,:,:], 0.5, axis=3
        ).reshape(self.num_areas, -1) * 1000
    peak2_ar = np.quantile(peak2_acrs[:,clist,:,:], 0.5, axis=3
        ).reshape(self.num_areas, -1) * 1000
    print('peak1_acrs.shape:', peak1_acrs.shape)
    print('peak1_ar.shape:', peak1_ar.shape)

    # Conditioning on LM peak-2.
    features_v1, features_lm, features_al = peak2_ar[0], peak2_ar[1], peak2_ar[2]

    num_groups = 3
    quantiles = (np.arange(num_groups+1)) / num_groups
    thrs = np.quantile(features_lm, quantiles)
    print('LM peak-2 quantiles: ', thrs)
    group_ids = [0] * num_groups
    for g in range(num_groups):
      group_ids[g] = (features_lm >= thrs[g]) & (features_lm < thrs[g+1])
      corr = scipy.stats.pearsonr(features_al[group_ids[g]],
                                  features_v1[group_ids[g]])
      print(f'corr: {corr[0]:.3f}\tp-val: {corr[1]:.3e}')

    colors = matplotlib.pylab.cm.jet(np.linspace(0.1, 0.9, num_groups))
    plt.figure(figsize=(3.5, 3.5))
    ax = plt.gca()
    ax.tick_params(left=True, labelbottom=True, bottom=True,
                   top=False, labeltop=False)
    plt.plot(features_al, features_v1, '.', c='lightgrey')
    for g in range(num_groups):
      plt.plot(features_al[group_ids[g]], features_v1[group_ids[g]], '.',
          c=colors[g],
          label=f'{quantiles[g]*100:.0f} - {quantiles[g+1]*100:.0f}%')

    plt.plot([0, 500], [0, 500], ls='--', c='lightgrey')
    plt.axis('equal')
    plt.axis('square')
    plt.xlabel(f'V1-Peak-2 time [ms]', fontsize=12)
    plt.ylabel(f'AL-Peak-2 time [ms]', fontsize=12)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    # ax.set_title('Statistical model', fontsize=16)
    ax.set_title('Statistical model', fontsize=16)
    plt.xticks(np.arange(0, 1000, 100))
    plt.yticks(np.arange(0, 1000, 100))
    plt.xlim(100, 400)
    plt.ylim(100, 400)
    plt.legend()
    plt.tight_layout()
    if output_dir is not None:
      output_figure_path = os.path.join(output_dir,
          f'{self.session_id}_V1_AL_condLM_peak2_dots_group{num_groups}.pdf')
      plt.savefig(output_figure_path, bbox_inches='tight')  # pad_inches=0,
      print('save figure:', output_figure_path)
    plt.show()

    gs_kw = dict(width_ratios=[1]*num_groups, height_ratios=[1])
    fig, axs = plt.subplots(figsize=(4 *num_groups, 3), gridspec_kw=gs_kw,
        nrows=1, ncols=num_groups)
    plt.subplots_adjust(left=None, right=None, hspace=0.1, wspace=0)

    for g in range(num_groups):
      ax = fig.add_subplot(axs[g])
      if g == 0:
        ax.tick_params(left=True, labelbottom=True, bottom=True)
      else:
        ax.tick_params(labelleft=False, labelbottom=True, bottom=True)
      plt.plot([0, 500], [0, 500], ls='--', c='lightgrey')
      plt.plot(features_al, features_v1, '.', c='lightgrey')
      plt.plot(features_al[group_ids[g]], features_v1[group_ids[g]], 'k.')

      plt.axis('equal')
      plt.axis('square')
      ax.xaxis.set_tick_params(labelsize=10)
      ax.yaxis.set_tick_params(labelsize=10)
      plt.xticks(np.arange(0, 1000, 100))
      plt.yticks(np.arange(0, 1000, 100))
      plt.xlim(100, 400)
      plt.ylim(100, 400)
      plt.xlabel(f'AL-Peak-2 time [ms]', fontsize=12)
      if g == 0:
        plt.ylabel(f'V1-Peak-2 time [ms]', fontsize=12)
      plt.title(f'{quantiles[g]*100:.0f} - {quantiles[g+1]*100:.0f}%')

    if output_dir is not None:
      output_figure_path = os.path.join(output_dir,
          f'{self.session_id}_V1_AL_condLM_peak2_dots_group{num_groups}_seperate.pdf')
      plt.savefig(output_figure_path, bbox_inches='tight')  # pad_inches=0,
      print('save figure:', output_figure_path)
    plt.show()


  def plot_trials_peaks_dots_partial_corr(
      self,
      clist,
      spike_train_time_line,
      burn_in=0,
      end=None,
      step=1,
      fit_type='refine',
      verbose=False,
      output_dir=None):
    """Calculates the lead lag relations from samples.

    Args:
      fit_type: 'brutal', 'simple', 'refine'.
    """
    peak_name = 'Peak-1'
    i,j,z = 1,2,0

    import sklearn.linear_model
    import sklearn.metrics
    linear_model = sklearn.linear_model.LinearRegression()

    areas_names = ['V1', 'LM', 'AL']
    # Peak shifting.
    if peak_name == 'Peak-1':
      q_samples = np.stack(self.q_shift1[burn_in:end:step], axis=0)
    elif peak_name == 'Peak-2':
      q_samples = np.stack(self.q_shift2[burn_in:end:step], axis=0)
    num_samples, num_areas, num_trials, num_conditions = q_samples.shape
    print('q2 shape:', num_samples, num_areas, num_trials, num_conditions)
    # q_ = q_samples.mean(axis=0)  # Posterior mean across samples.
    q_ = np.quantile(q_samples, 0.5, axis=0)  # Posterior mean across samples.
    q_ = q_.transpose(0,2,1).reshape(num_areas, -1)
    print('q_median.shape', q_.shape)
    q_ *= 1000
    features_i, features_j, features_z = q_[i], q_[j], q_[z]

    y = features_i
    X = features_z.reshape(-1, 1)
    linear_model.fit(X, features_z)
    y_hat = linear_model.predict(X)
    r2_score = sklearn.metrics.r2_score(y, y_hat)
    residuals_i_z = y - y_hat
    print(r2_score)

    y = features_j
    X = features_z.reshape(-1, 1)
    linear_model.fit(X, features_z)
    y_hat = linear_model.predict(X)
    r2_score = sklearn.metrics.r2_score(y, y_hat)
    residuals_j_z = y - y_hat
    print(r2_score)

    corr = scipy.stats.pearsonr(residuals_i_z, residuals_j_z)
    print(f'corr: {corr[0]:.3f}\tp-val: {corr[1]:.3e}')

    gs_kw = dict(width_ratios=[1], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(3.5, 3), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    plt.subplots_adjust(left=None, right=None, hspace=0.1, wspace=0)
    ax = fig.add_subplot(axs)
    ax.tick_params(left=True, labelbottom=True, bottom=True)
    # plt.plot([-500, 500], [-500, 500], ls='--', c='lightgrey')
    plt.axhline(0, ls='--', c='lightgrey')
    plt.axvline(0, ls='--', c='lightgrey')
    plt.plot(residuals_i_z, residuals_j_z, '.k')
    plt.axis('equal')
    plt.axis('square')
    plt.xlabel(f'{areas_names[i]}-{peak_name} residual [ms]', fontsize=12)
    plt.ylabel(f'{areas_names[j]}-{peak_name} residual [ms]', fontsize=12)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    # ax.set_title('Regression residual', fontsize=16)
    plt.text(0.7, 0.92, fr'$\rho$={corr[0]:.2f}', transform=ax.transAxes)
    plt.xticks(np.arange(-1000, 1000, 25))
    plt.yticks(np.arange(-1000, 1000, 25))
    if peak_name == 'Peak-2':
      plt.xlim(-50, 50)
      plt.ylim(-50, 50)
    elif peak_name == 'Peak-1':
      plt.xlim(-25, 25)
      plt.ylim(-25, 25)
    # plt.legend()
    plt.tight_layout()
    if output_dir is not None:
      output_figure_path = os.path.join(output_dir,
          f'{self.session_id}_{areas_names[i]}_{areas_names[j]}_cond_'+
          f'{areas_names[z]}_{peak_name}_dots_residuals.pdf')
      plt.savefig(output_figure_path)
      print('save figure:', output_figure_path)
    plt.show()



  def plot_peak_distribution(
      self,
      clist,
      spike_train_time_line,
      burn_in=0,
      end=None,
      step=1,
      fit_type='refine',
      verbose=False,
      output_dir=None):
    """Calculates the lead lag relations from samples.

    Args:
      fit_type: 'brutal', 'simple', 'refine'.
    """
    areas_names = ['V1', 'LM', 'AL']
    dt = 0.002

    peak1_acrs, peak2_acrs, initiation_acrs = self.get_peaks_initiations_arcs(
        clist, spike_train_time_line, burn_in, end, step,
        fit_type=fit_type, verbose=False)
    peak1_ar = np.quantile(peak1_acrs[:,clist,:,:], 0.5, axis=3
        ).reshape(self.num_areas, -1) * 1000
    peak2_ar = np.quantile(peak2_acrs[:,clist,:,:], 0.5, axis=3
        ).reshape(self.num_areas, -1) * 1000
    print('peak1_acrs.shape:', peak1_acrs.shape)
    print('peak1_ar.shape:', peak1_ar.shape)
    print('peak2_ar.shape:', peak2_ar.shape)

    # Peak-1.
    num_stack = self.num_areas
    gs_kw = dict(width_ratios=[1], height_ratios=[1]*num_stack)
    fig, axs = plt.subplots(figsize=(4, 3), gridspec_kw=gs_kw,
        nrows=num_stack, ncols=1)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)

    for a in range(self.num_areas):
      ax = fig.add_subplot(axs[a])
      ax.tick_params(left=False, labelleft=False, bottom=True, labelbottom=False,
          direction='in')
      if a == num_stack-1:
        ax.tick_params(bottom=True, labelbottom=True, direction='in')
      x, y = seaborn.distplot(
          peak1_ar[a], color='grey').get_lines()[0].get_data()
      center = np.quantile(peak1_ar[a], 0.5)
      CI_left = np.quantile(peak1_ar[a], 0.025)
      CI_right = np.quantile(peak1_ar[a], 0.975)
      err_left = center - CI_left
      err_right = CI_right - center
      gkde=scipy.stats.gaussian_kde(peak1_ar[a])
      x = np.linspace(CI_left, CI_right, 201)
      y = gkde.evaluate(x)
      mode = x[np.argmax(y)]
      print(f'CI_left {CI_left:.2f}, CI_right {CI_right:.2f}, mode {mode:.2f}')

      plt.plot(1000, 0, 'k+', label='median')  # Fake point for median.
      plt.errorbar([center], [np.max(y)*1.5], xerr=[[err_left], [err_right]],
          fmt='+k', capsize=5, label='95% CI')
      # plt.axvline(x=CI_left, linestyle=':', color='k')
      # plt.axvline(x=CI_right, linestyle=':', color='k', label='%95 CI')
      # plt.axvline(x=center, color='k', ls='--', label='Median')
      plt.axvline(x=0, color='k', ls=':')
      plt.xlim(0, 150)
      plt.ylim(0, np.max(y)*2)
      plt.text(0.05, 0.7, f'{areas_names[a]}', transform=ax.transAxes)

      # if r == 0:
      #   if center > 0:
      #     plt.legend(loc='lower left', ncol=2, prop={'size': 8})
      #   else:
      #     plt.legend(loc='lower right', ncol=2, prop={'size': 8})
      if a == num_stack-1:
        plt.xlabel('Peak time [ms]')
      if a == 0:
        plt.title('Peak-1 time', fontsize=16)

    if output_dir is not None:
      file_path = os.path.join(output_dir, 'peak_1_distribution.pdf')
      plt.savefig(file_path, bbox_inches='tight')
      print('save figure:', file_path)
    plt.show()

    # Peak-2.
    num_stack = self.num_areas
    gs_kw = dict(width_ratios=[1], height_ratios=[1]*num_stack)
    fig, axs = plt.subplots(figsize=(4, 3), gridspec_kw=gs_kw,
        nrows=num_stack, ncols=1)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)

    for a in range(self.num_areas):
      ax = fig.add_subplot(axs[a])
      ax.tick_params(left=False, labelleft=False, bottom=True, labelbottom=False,
          direction='in')
      if a == num_stack-1:
        ax.tick_params(bottom=True, labelbottom=True, direction='in')
      x, y = seaborn.distplot(
          peak2_ar[a], bins=30, color='grey').get_lines()[0].get_data()
      center = np.quantile(peak2_ar[a], 0.5)
      CI_left = np.quantile(peak2_ar[a], 0.025)
      CI_right = np.quantile(peak2_ar[a], 0.975)
      err_left = center - CI_left
      err_right = CI_right - center
      gkde=scipy.stats.gaussian_kde(peak2_ar[a])
      x = np.linspace(CI_left, CI_right, 201)
      y = gkde.evaluate(x)
      mode = x[np.argmax(y)]
      print(f'CI_left {CI_left:.2f}, CI_right {CI_right:.2f}, mode {mode:.2f}')

      plt.plot(1000, 0, 'k+', label='median')  # Fake point for median.
      plt.errorbar([center], [np.max(y)*1.5], xerr=[[err_left], [err_right]],
          fmt='+k', capsize=5, label='95% CI')
      # plt.axvline(x=CI_left, linestyle=':', color='k')
      # plt.axvline(x=CI_right, linestyle=':', color='k', label='%95 CI')
      # plt.axvline(x=center, color='k', ls='--', label='Median')
      plt.axvline(x=0, color='k', ls=':')
      plt.xlim(50, 400)
      plt.ylim(0, np.max(y)*2)
      plt.text(0.05, 0.7, f'{areas_names[a]}', transform=ax.transAxes)

      # if r == 0:
      #   if center > 0:
      #     plt.legend(loc='lower left', ncol=2, prop={'size': 8})
      #   else:
      #     plt.legend(loc='lower right', ncol=2, prop={'size': 8})
      if a == num_stack-1:
        plt.xlabel('Peak time [ms]')
      if a == 0:
        plt.title('Peak-2 time', fontsize=16)

    if output_dir is not None:
      file_path = os.path.join(output_dir, 'peak_2_distribution.pdf')
      plt.savefig(file_path, bbox_inches='tight')
      print('save figure:', file_path)
    plt.show()


  def stack_q(
      self,
      burn_in=0,
      end=None,
      step=1,
      model_feature_type=None):
    """Stacks possible `q_arc`, `q_shift1_arc`, `q_shift2_arc`."""
    if model_feature_type == 'B':
      q = np.stack(self.q[burn_in:end:step], axis=0)
      num_samples, _, num_trials, _ = q.shape
      q = q.transpose(0,1,3,2).reshape(num_samples, self.num_areas, -1)
      print('q.shape', q.shape)
      qall_sq = q

    elif model_feature_type == 'BSS':
      q = np.stack(self.q[burn_in:end:step], axis=0)
      q_shift1 = np.stack(self.q_shift1[burn_in:end:step], axis=0)
      q_shift2 = np.stack(self.q_shift2[burn_in:end:step], axis=0)
      num_samples, _, num_trials, _ = q.shape
      q = q.transpose(0,1,3,2).reshape(num_samples, self.num_areas, -1)
      q_shift1 = q_shift1.transpose(0,1,3,2).reshape(num_samples, self.num_areas, -1)
      q_shift2 = q_shift2.transpose(0,1,3,2).reshape(num_samples, self.num_areas, -1)
      # print('q.shape', q.shape)

      qall_sq = np.zeros([num_samples, self.num_areas * self.num_qs,
                           num_trials * self.num_conditions])

      for a in range(self.num_areas):
        qall_sq[:,a*self.num_qs] = q[:,a]
        qall_sq[:,a*self.num_qs+1] = q_shift1[:,a]
        qall_sq[:,a*self.num_qs+2] = q_shift2[:,a]

    return qall_sq


  def get_qs_bayesian_r2(
      self,
      burn_in=0,
      end=None,
      step=1,
      model_feature_type=None):
    """Get the posterior of the R-squared."""
    import sklearn.linear_model
    import sklearn.metrics
    linear_model = sklearn.linear_model.LinearRegression()

    qs = self.stack_q(burn_in, end, step, model_feature_type)
    num_samples, num_qs, num_trials = qs.shape
    r2_scores = np.zeros([num_qs, num_samples])

    for y_ind in range(num_qs):
      x_ind = np.delete(np.arange(num_qs), y_ind)
      for s in range(num_samples):
        y = qs[s, y_ind, :]
        X = qs[s, x_ind, :].T
        linear_model.fit(X, y)
        y_hat = linear_model.predict(X)
        r2_scores[y_ind, s] = sklearn.metrics.r2_score(y, y_hat)

    return r2_scores


  def get_qs_subset_bayesian_r2(
      self,
      y_ind,
      x_ind,
      burn_in=0,
      end=None,
      step=1,
      model_feature_type=None,
      fig_title=None,
      file_path=None,
      verbose=True):
    """Get the posterior of the R-squared."""
    import sklearn.linear_model
    import sklearn.metrics
    linear_model = sklearn.linear_model.LinearRegression()

    qs = self.stack_q(burn_in, end, step, model_feature_type)
    num_samples, num_qs, num_trials = qs.shape
    r2_scores = np.zeros(num_samples)

    for s in range(num_samples):
      y = qs[s, y_ind, :]
      X = qs[s, x_ind, :].T
      linear_model.fit(X, y)
      y_hat = linear_model.predict(X)
      r2_scores[s] = sklearn.metrics.r2_score(y, y_hat)

    gs_kw = dict(width_ratios=[1], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(3.5, 2), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    # plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
    axs = [axs]
    ax = fig.add_subplot(axs[0])
    ax.tick_params(left=False, labelleft=False)
    # seaborn.distplot(r2_scores, bins=30, norm_hist=True, color='grey')
    center = np.quantile(r2_scores, 0.5)
    left_CI = np.quantile(r2_scores, 0.025)
    right_CI = np.quantile(r2_scores, 0.975)
    err_left = center - left_CI
    err_right = right_CI - center
    plt.errorbar([center], [0], xerr=[[err_left], [err_right]],
                 fmt='+k', capsize=5, label='95% CI')
    plt.xlim(0, 1)
    plt.xlabel(r'$R^2$')
    plt.title(fig_title)
    plt.legend()
    if verbose:
      plt.show()
    else:
      plt.close()

    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('Save file:', file_path)

    return r2_scores


  def get_qs_subset_bayesian_r2_multiple(
      self,
      yx_list,
      burn_in=0,
      end=None,
      step=1,
      model_feature_type=None,
      fig_title=None,
      sub_title_list=None,
      lay_out=None,
      file_path=None,
      verbose=True):
    """Plot multiple R2 together."""
    num_plots = len(yx_list)
    if lay_out is not None:
      num_rows, num_cols = lay_out
    else:
      num_rows, num_cols = 1, num_plots

    gs_kw = dict(width_ratios=[1]*num_cols, height_ratios=[1]*num_rows)
    fig, axs = plt.subplots(figsize=(2*num_cols, 1*num_rows),
        gridspec_kw=gs_kw, nrows=num_rows, ncols=num_cols)
    if num_rows > 1:
      plt.subplots_adjust(left=None, right=None, hspace=0.3, wspace=0.15)
    else:
      plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.15)
    axs = axs.reshape(-1)

    for c, (y_ind, x_ind) in enumerate(yx_list):
      r2_scores = self.get_qs_subset_bayesian_r2(y_ind, x_ind,
          burn_in, end, step, model_feature_type, verbose=False)

      ax = fig.add_subplot(axs[c])
      ax.tick_params(left=False, labelleft=False, labelbottom=True)
      x, y = seaborn.distplot(
          r2_scores, bins=30, color='tab:gray').get_lines()[0].get_data()
      center = np.quantile(r2_scores, 0.5)
      left_CI = np.quantile(r2_scores, 0.025)
      right_CI = np.quantile(r2_scores, 0.975)
      err_left = center - left_CI
      err_right = right_CI - center
      plt.errorbar([center], [np.max(y)*1.5], xerr=[[err_left], [err_right]],
                   fmt='+k', capsize=5, label='95% CI')
      plt.xlim(-0.02, 1.02)
      plt.ylim(0, np.max(y)*2.4)
      plt.text(0.5, 1.05, sub_title_list[c], ha='center', fontsize=12,
               transform=ax.transAxes)
      plt.text(0.05, 0.8, f'model {c}', fontsize=10, transform=ax.transAxes)

      print(sub_title_list[c])
      print(f'model {c}')
      print(f'{center:.2f} ({left_CI:.2f},{right_CI:.2f})')
      print()

      if num_rows == 1 and c == 0:
        plt.xlabel(r'$R^2$')
        ax.tick_params(labelbottom=True)
      elif num_rows == 1 and c == 1:
        plt.plot(1000, 0, 'k+', label='median')  # Fake point for median.
        plt.legend(loc=(0,-0.65), ncol=2)
      elif num_rows > 1 and c == num_plots - num_cols:
        plt.xlabel(r'$R^2$')
        ax.tick_params(labelbottom=True)

    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('Save file:', file_path)
    plt.show()


  def plot_r2_comparison(
      self,
      r2_scores0,
      r2_scores1,
      fig_title=None,
      file_path=None,):
    """Compare the R2 from two different models."""
    r2_scores = r2_scores0 - r2_scores1

    gs_kw = dict(width_ratios=[1], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(2, 1), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    # plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
    axs = [axs]
    ax = fig.add_subplot(axs[0])
    ax.tick_params(left=False, labelleft=False)
    x, y = seaborn.distplot(r2_scores, bins=30, color='tab:gray'
        ).get_lines()[0].get_data()
    center = np.quantile(r2_scores, 0.5)
    left_CI = np.quantile(r2_scores, 0.025)
    right_CI = np.quantile(r2_scores, 0.975)
    err_left = center - left_CI
    err_right = right_CI - center
    plt.errorbar([center], [np.max(y)*1.5], xerr=[[err_left], [err_right]],
                 fmt='+k', capsize=5, label='95% CI')
    # plt.axvline(0, c='lightgrey')
    plt.xlim(-0.1, 0.8)
    plt.ylim(0, np.max(y)*2)
    plt.xlabel(r'$R^2$ difference')
    plt.title(fig_title)
    # plt.legend()

    print(fig_title)
    print(f'{center:.2f} ({left_CI:.2f},{right_CI:.2f})')

    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('Save file:', file_path)


  def plot_qs_bayesian_r2(
      self,
      r2_scores,
      model_feature_type=None):
    """Plot the R-sqaured of a linear regression."""
    areas = ['V1', 'LM', 'AL']
    features = ['Gain', 'Peak-1', 'Peak-2']

    gs_kw = dict(width_ratios=[1] * 3,
                 height_ratios=[1] * 3)
    fig, axs = plt.subplots(figsize=(10, 5), gridspec_kw=gs_kw,
        nrows=3, ncols=3)
    # plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)

    for a in range(self.num_areas):
      for q in range(3):
        ax = fig.add_subplot(axs[q, a])
        ax.tick_params(left=True, labelleft=False, labelbottom=False,
                       bottom=True, top=False, labeltop=False)
        if a == 0 and q == 2:
          ax.tick_params(left=True, labelleft=False, labelbottom=True,
                         bottom=True, top=False, labeltop=False)
          plt.xticks([0, 1], [0, 1])

        seaborn.distplot(r2_scores[a*3+q], bins=30, norm_hist=True, color='grey')
        plt.xlim(0, 1)
        plt.ylim(0, 30)

        if q == 0:
          plt.title(areas[a])
        if a == 0:
          plt.ylabel(features[q])

    plt.show()


  def quantile_mcmc_error(
      self,
      samples,
      quantile,
      num_batches=20):
    """Get the quantile error.

    This is used for single variable MCMC quantile error. The method is from
    Doss 2014 - MCMC estimation of quantiles, section 3.2.
    """
    # estimation of sigma(y)
    num_samples = len(samples)
    batch_size = np.floor(num_samples / num_batches).astype(int)
    reminder = num_samples % batch_size
    if reminder != 0:
      samples = samples[0:-reminder] # Remove the short tail.
      num_samples = len(samples)

    xi = np.quantile(samples, quantile)
    Fn = (samples < xi).sum() / num_samples
    v = (samples <= xi) + 0
    v_batches = v.reshape(-1, batch_size) / batch_size
    U_batches = v_batches.sum(axis=1)
    sigma2_hat = np.sum(np.square(U_batches - Fn)) * batch_size / (num_batches - 1)

    # estimation of f(quantile)
    gkde=scipy.stats.gaussian_kde(samples)
    f_hat = gkde.evaluate(xi)

    gamma2_hat = sigma2_hat / f_hat / f_hat
    std_hat = np.sqrt(gamma2_hat) / np.sqrt(num_samples)

    return std_hat


  def rho_quantile_mcmc_error(
      self,
      burn_in=0,
      end=None,
      step=1,
      rho_type='marginal',
      model_feature_type=None,
      verbose=False):
    """rho quantile MCMC errors."""
    sigma_samples = np.stack(self.sigma_cross_pop[burn_in:end:step], axis=0)
    num_samples, rho_size, _ = sigma_samples.shape
    rho_samples = np.zeros(sigma_samples.shape)
    corrcoef_samples = np.zeros(sigma_samples.shape)
    qs = self.stack_q(burn_in, end, step, model_feature_type)

    if rho_type == 'marginal':
      for s in range(num_samples):
        sigma = sigma_samples[s,:,:]
        sigma_diag_sqrt = np.sqrt(np.diag(sigma))
        rho_samples[s] = sigma / np.outer(sigma_diag_sqrt, sigma_diag_sqrt)
        corrcoef_samples[s] = np.corrcoef(qs[s])
    else:
      return

    num_samples, rho_size, _ = rho_samples.shape
    if verbose:
      print('rho_samples.shape:', rho_samples.shape)

    # CI traps.
    CI_endpoints_std = np.zeros([rho_size, rho_size, 2])
    CI_endpoints_corrcoef_std = np.zeros([rho_size, rho_size, 2])

    for row in range(rho_size):
      for col in range(rho_size):
        if row >= col:
          continue
        # gkde=scipy.stats.gaussian_kde(sub_samples)
        # x = np.linspace(CI_left, CI_right, 201)
        # y = gkde.evaluate(x)
        # mode = x[np.argmax(y)]
        # center = x[np.argmax(y)]  # Mode.
        # center = np.quantile(sub_samples, 0.5)  # Median.

        sub_samples = rho_samples[:,row,col]
        corrcoef_sub_samples = corrcoef_samples[:,row,col]
        ci_std_left = self.quantile_mcmc_error(sub_samples, 0.025)
        ci_std_right = self.quantile_mcmc_error(sub_samples, 0.975)
        CI_endpoints_std[row,col,:] = [ci_std_left, ci_std_right]

        ci_std_left = self.quantile_mcmc_error(corrcoef_sub_samples, 0.025)
        ci_std_right = self.quantile_mcmc_error(corrcoef_sub_samples, 0.975)
        CI_endpoints_std[row,col,:] = [ci_std_left, ci_std_right]

        if verbose:
          CI_left = np.quantile(sub_samples, 0.025)
          CI_right = np.quantile(sub_samples, 0.975)
          gkde=scipy.stats.gaussian_kde(sub_samples)
          x = np.linspace(-1, 1, 201)
          y = gkde.evaluate(x)
          plt.figure()
          plt.plot(x, y)
          seaborn.distplot(sub_samples, bins=25, color='tab:gray')
          plt.axvline(x=CI_left, ls=':')
          plt.axvline(x=CI_right, ls=':')

    return CI_endpoints_std, CI_endpoints_corrcoef_std


  def get_ci_endpoints(
      self,
      burn_in=0,
      end=None,
      step=1,
      rho_type='marginal',
      model_feature_type=None,
      verbose=False):
    """Collect CI endpoints."""
    sigma_samples = np.stack(self.sigma_cross_pop[burn_in:end:step], axis=0)
    num_samples, rho_size, _ = sigma_samples.shape
    rho_samples = np.zeros(sigma_samples.shape)
    corrcoef_samples = np.zeros(sigma_samples.shape)
    qs = self.stack_q(burn_in, end, step, model_feature_type)

    if rho_type == 'marginal':
      for s in range(num_samples):
        sigma = sigma_samples[s,:,:]
        sigma_diag_sqrt = np.sqrt(np.diag(sigma))
        rho_samples[s] = sigma / np.outer(sigma_diag_sqrt, sigma_diag_sqrt)
        corrcoef_samples[s] = np.corrcoef(qs[s])
    else:
      return

    num_samples, rho_size, _ = rho_samples.shape
    if verbose:
      print('rho_samples.shape:', rho_samples.shape)

    # CI traps.
    CI_endpoints = np.zeros([rho_size, rho_size, 2])
    CI_endpoints_corrcoef = np.zeros([rho_size, rho_size, 2])

    for row in range(rho_size):
      for col in range(rho_size):
        if row >= col:
          continue
        # gkde=scipy.stats.gaussian_kde(sub_samples)
        # x = np.linspace(CI_left, CI_right, 201)
        # y = gkde.evaluate(x)
        # mode = x[np.argmax(y)]
        # center = x[np.argmax(y)]  # Mode.
        # center = np.quantile(sub_samples, 0.5)  # Median.

        sub_samples = rho_samples[:,row,col]
        CI_left = np.quantile(sub_samples, 0.025)
        CI_right = np.quantile(sub_samples, 0.975)
        CI_endpoints[row,col] = [CI_left, CI_right]

        corrcoef_sub_samples = corrcoef_samples[:,row,col]
        corrcoef_CI_left = np.quantile(corrcoef_sub_samples, 0.025)
        corrcoef_CI_right = np.quantile(corrcoef_sub_samples, 0.975)
        CI_endpoints_corrcoef[row,col] = [corrcoef_CI_left, corrcoef_CI_right]
    return CI_endpoints, CI_endpoints_corrcoef


  def rho_mse_ci_coverage(
      self,
      burn_in=0,
      end=None,
      step=1,
      rho_type=['marginal'],
      true_model=None,
      model_feature_type=None,
      verbose=False):
    """Calculate the CI coverage.

    There are two types of design, 
      random design: the ground truth is the true value of sigma;
      fixed design: the truth is the correlation derived from the true values
          of qs (features).
      hat design: sigma CI to trap q's correlations.
    """
    sigma_samples = np.stack(self.sigma_cross_pop[burn_in:end:step], axis=0)
    num_samples, rho_size, _ = sigma_samples.shape
    rho_samples = np.zeros(sigma_samples.shape)
    qs = self.stack_q(burn_in, end, step, model_feature_type)
    corrcoef_samples = np.zeros([qs.shape[0], qs.shape[1], qs.shape[1]])

    if 'marginal' in rho_type:
      for s in range(num_samples):
        rho_samples[s] = util.marginal_corr_from_cov(sigma_samples[s])
    if 'corrcoef' in rho_type:
      for s in range(qs.shape[0]):
        corrcoef_samples[s] = np.corrcoef(qs[s])

    num_samples, rho_size, _ = rho_samples.shape
    if verbose:
      print('rho_samples.shape:', rho_samples.shape)

    # True model.
    sigma_true = true_model.sigma_cross_pop
    sigma_diag_sqrt = np.sqrt(np.diag(sigma_true))
    rho_true_rnd = sigma_true / sigma_diag_sqrt / sigma_diag_sqrt.reshape(-1, 1)
    rhoz_true = util.fisher_transform(rho_true_rnd)
    q_true = true_model.stack_q()
    q_true = q_true.transpose(0,2,1).reshape(
        true_model.num_areas*true_model.num_qs, -1)
    rho_true_fix = np.corrcoef(q_true)
    rhoz_true_hat = util.fisher_transform(rho_true_fix)

    # CI traps.
    CI_trap_hat = np.zeros([rho_size, rho_size])
    CI_trap_rnd = np.zeros([rho_size, rho_size])
    CI_trap_fix = np.zeros([rho_size, rho_size])
    error_hat = np.zeros([rho_size, rho_size])
    error_rnd = np.zeros([rho_size, rho_size])
    error_fix = np.zeros([rho_size, rho_size])

    for row in range(rho_size):
      for col in range(rho_size):
        if row >= col:
          continue
        if 'marginal' in rho_type:
          # For the random design.
          sub_samples = rho_samples[:,row,col]
          CI_left = np.quantile(sub_samples, 0.025)
          CI_right = np.quantile(sub_samples, 0.975)
        if 'corrcoef' in rho_type:
          # For the fixed design.
          corrcoef_sub_samples = corrcoef_samples[:,row,col]
          corrcoef_CI_left = np.quantile(corrcoef_sub_samples, 0.025)
          corrcoef_CI_right = np.quantile(corrcoef_sub_samples, 0.975)

        if ('marginal' in rho_type and
            CI_left <= rho_true_fix[row,col] and
            CI_right >= rho_true_fix[row,col]):
          CI_trap_hat[row, col] = 1

        if ('marginal' in rho_type and
            CI_left <= rho_true_rnd[row,col] and
            CI_right >= rho_true_rnd[row,col]):
          CI_trap_rnd[row, col] = 1

        if ('corrcoef' in rho_type and
            corrcoef_CI_left <= rho_true_fix[row,col] and
            corrcoef_CI_right >= rho_true_fix[row,col]):
          CI_trap_fix[row, col] = 1

        if 'marginal' in rho_type:
          # gkde=scipy.stats.gaussian_kde(sub_samples)
          # x = np.linspace(CI_left, CI_right, 201)
          # y = gkde.evaluate(x)
          # mode = x[np.argmax(y)]
          # center = x[np.argmax(y)]  # Mode.
          center = np.quantile(sub_samples, 0.5)  # Median.
          error_hat[row, col] = center - rho_true_fix[row,col]
          error_rnd[row, col] = center - rho_true_rnd[row,col]
        if 'corrcoef' in rho_type:
          center_corrcoef = np.quantile(corrcoef_sub_samples, 0.5)  # Median.
          error_fix[row, col] = center_corrcoef - rho_true_fix[row,col]

    if verbose:
      print('CI traps hat total, ratio:',
          np.sum(CI_trap_hat), np.sum(CI_trap_hat)/rho_size/(rho_size-1)*2)
      print('CI traps rnd total, ratio:',
          np.sum(CI_trap_rnd), np.sum(CI_trap_rnd)/rho_size/(rho_size-1)*2)
      print('CI traps fix total, ratio:',
          np.sum(CI_trap_fix), np.sum(CI_trap_fix)/rho_size/(rho_size-1)*2)

    return error_rnd, error_hat, error_fix, CI_trap_rnd, CI_trap_hat, CI_trap_fix


  def samples_autocorr(
      self,
      burn_in=0,
      end=None,
      step=1,
      figure_path=None):
    """Plots mu, sigma, rho.

    Args:
        type: 'z', 'rho', 'corrcoef'
    """
    areas = ['V1', 'LM', 'AL']
    features = ['Gain', 'First peak shifting', 'Second peak shifting']
    sigma_samples = np.stack(self.sigma_cross_pop, axis=0)
    num_samples, rho_size, _ = sigma_samples.shape
    print('sigma_samples.shape:', sigma_samples.shape)
    rho_samples_all = np.zeros(sigma_samples.shape)

    for s in range(num_samples):
      sigma = sigma_samples[s,:,:]
      sigma_diag_sqrt = np.sqrt(np.diag(sigma))
      rho_samples_all[s] = sigma / np.outer(sigma_diag_sqrt, sigma_diag_sqrt)
    rho_samples = rho_samples_all[burn_in:end:step]
    print('rho_samples.shape:', rho_samples.shape)

    # -------- For dubug. Check a specific entry. --------
    # x = rho_samples[:,8,5]
    # lag, xcorr, CI_level = util.xcorr(x, x)

    # plt.figure(figsize=(6, 2))
    # plt.plot(x)

    # plt.figure(figsize=(6, 2))
    # plt.plot(lag, xcorr)
    # plt.axhline(y=0, color='grey', ls=':')
    # plt.axhline(y=CI_level, color='grey', ls=':', label='95% level')
    # plt.axhline(y=-CI_level, color='grey', ls=':')
    # # plt.ylim(-0.4, 0.4)
    # plt.xlabel('Index lag')
    # plt.ylabel('Auto-correlation')
    # plt.legend()
    # return
    # ---------------------------------------------------

    gs_kw = dict(width_ratios=[1] * rho_size,
                 height_ratios=[1] * rho_size)
    fig, axs = plt.subplots(figsize=(20, 10), gridspec_kw=gs_kw,
        nrows=rho_size, ncols=rho_size)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)

    # CI traps.
    for row in range(rho_size):
      for col in range(rho_size):
        ax = fig.add_subplot(axs[row, col])
        if row > col:
          ax.set_xticks([])
          ax.set_yticks([])
          ax.axis('off')
          continue
        if row == col:
          plt.text(0, 0.25, areas[row // 3], ha='center', va='center')
          plt.text(0, -0.25, features[row % 3], ha='center', va='center')
          plt.xlim(-1, 1); plt.ylim(-1, 1)
          ax.set_xticks([])
          ax.set_yticks([])
          ax.axis('off')
          continue
        x = rho_samples[:,row, col]
        lag, xcorr, CI_level = util.xcorr(x, x)
        plt.plot(lag, xcorr)
        # plt.axhline(y=0, color='grey', ls=':')
        plt.axhline(y=CI_level, color='grey', ls=':', label='95% level')
        plt.axhline(y=-CI_level, color='grey', ls=':')
        plt.ylim(-0.6, 1)
        # plt.xlabel('Index lag')
        # plt.ylabel('Auto-correlation')
        # plt.legend()
        ax.set_xticks([])
        ax.set_yticks([])

        if row == 0 and col == 1:
          plt.legend(loc=(-0.8, -1.2))

        # if row == 0 and col == 1:
        #   ax.tick_params(left=False, labelbottom=False, bottom=False,
        #                  top=True, labeltop=True)
        #   ax.set_xticks([-1, 0, 1])
        #   plt.plot([10], [0], '+k', label='median')
        #   plt.legend(loc=(-0.8, -1.2))
        # else:
        #   ax.set_xticks([])
        # ax.set_yticks([])
        # ax.tick_params(which='minor', direction='in',
        #                top=False, labeltop=False, bottom=True, labelbottom=True)
        # ax.xaxis.set_minor_locator(MultipleLocator(0.2))

    if figure_path is not None:
      plt.savefig(figure_path)
      print('Save figure:', figure_path)
      plt.show()
