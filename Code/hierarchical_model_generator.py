"""Data models."""
import os
import sys

from absl import logging
import collections
from collections import defaultdict
import io
import itertools
import numpy as np
import matplotlib
import matplotlib.pylab
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import pandas as pd
import seaborn
import scipy
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import time
import warnings

from data_model import AllenInstituteDataModel
from hierarchical_sampling_model import HierarchicalSamplingModel
import jitter
import smoothing_spline
import util


class HierarchicalModelGenerator(HierarchicalSamplingModel):

  def __init__(self, empty_samples=False):
    super().__init__(empty_samples=empty_samples)
    # Output setup.

  def initial_step(
      self,
      model_feature_type='B',
      num_trials=15,
      num_conditions=1,
      verbose=True):
    """Sets up initial conditions."""
    print('model_feature_type:', model_feature_type)
    # print('The model includes attributes:')
    # print(self.__dict__.keys())

    self.num_trials = num_trials
    self.num_conditions = num_conditions
    self.model_feature_type = model_feature_type

    if self.model_feature_type in ['B', 'S', 'S1', 'S2']:
      self.num_qs = 1
      self.f_warp_sources_arc = np.zeros((
          self.num_areas, self.num_trials, self.num_conditions, 3))
      self.f_warp_targets_arc = np.zeros((
          self.num_areas, self.num_trials, self.num_conditions, 3))
    elif self.model_feature_type in ['SS', 'BS']:
      self.num_qs = 2
      self.f_warp_sources_arc = np.zeros((
          self.num_areas, self.num_trials, self.num_conditions, 6))
      self.f_warp_targets_arc = np.zeros((
          self.num_areas, self.num_trials, self.num_conditions, 6))
    elif self.model_feature_type in ['BSS']:
      self.num_qs = 3
      self.f_warp_sources_arc = np.zeros((
          self.num_areas, self.num_trials, self.num_conditions, 6))
      self.f_warp_targets_arc = np.zeros((
          self.num_areas, self.num_trials, self.num_conditions, 6))

    print()
    print('num areas', self.num_areas, '  num trials', self.num_trials,
          '  num conditions', self.num_conditions, '  num qs', self.num_qs)

    # Setup trials and conditions.
    self.trials_indices = np.arange(self.num_trials * self.num_conditions)
    self.stimulus_condition_ids = np.arange(self.num_conditions)
    trials_data = {
        'stimulus_presentation_id' : self.trials_indices,
        'stimulus_condition_id' : np.kron(
            self.stimulus_condition_ids, np.ones(self.num_trials))}
    self.trials_df = pd.DataFrame(trials_data)
    self.trials_df.set_index('stimulus_presentation_id', inplace=True)
    self.trials_groups = self.trials_df.groupby('stimulus_condition_id')

    self.dt = self.spike_train_time_line[1] - self.spike_train_time_line[0]

    # Protect the original data.
    if not hasattr(self, 'q_arc_original'):
      self.q_arc_original = self.q_arc.copy()
    if not hasattr(self, 'q_shift1_arc_original'):
      self.q_shift1_arc_original = self.q_shift1_arc.copy()
    if not hasattr(self, 'q_shift2_arc_original'):
      self.q_shift2_arc_original = self.q_shift2_arc.copy()
    if not hasattr(self, 'f_pop_cag_original'):
      self.f_pop_cag_original = self.f_pop_cag.copy()
    if not hasattr(self, 'f_peak1_ac_original'):
      self.f_peak1_ac_original = self.f_peak1_ac.copy()
    if not hasattr(self, 'f_peak2_ac_original'):
      self.f_peak2_ac_original = self.f_peak2_ac.copy()
    if (not hasattr(self, 'sub_group_df_c_original') and
        hasattr(self, 'sub_group_df')):
      self.sub_group_df_c_original = [0]
      self.sub_group_df_c_original[0] = self.sub_group_df.copy()
    if (not hasattr(self, 'sub_group_df_c_original') and
        hasattr(self, 'sub_group_df_c')):
      self.sub_group_df_c_original = self.sub_group_df_c.copy()
    if not hasattr(self, 'p_gac_original'):
      self.p_gac_original = self.p_gac.copy()
    if not hasattr(self, 'sigma_cross_pop_original'):
      self.sigma_cross_pop_original = self.sigma_cross_pop.copy()

  def save_data(self, save_spikes=True, file_path=None, verbose=False):
    """Saves the memory to file, not including sample memory."""
    import pickle

    if save_spikes:
      excluded_attributes = ['session', 'samples']
    else:
      excluded_attributes = ['session', 'samples',
          'spike_trains', 'spike_times', 'spike_counts', 'spike_shifts',
          'log_lambda_nr']

    new_model = {}
    for key in self.__dict__.keys():
      if key in excluded_attributes:
        continue
      if self.__dict__[key] is None:
        print(key)
      new_model[key] = self.__dict__[key]

    with open(file_path, 'wb') as f:
      pickle.dump(new_model, f)
      if verbose:
        print('Save file to:', file_path)


  def load_data(self, file_path=None, verbose=False):
    """Loads the memory from file."""
    import pickle
    with open(file_path, 'rb') as f:
      new_model = pickle.load(f)
      # Copy everything from the loaded structure to myself.
      self.__dict__.update(new_model)
      if verbose:
        print('Load generator:', file_path)


  def corr_cross_pop(self):
    corr_size = self.num_qs * self.num_areas
    corr = util.marginal_corr_from_cov(self.sigma_cross_pop)
    fig, axs = plt.subplots(figsize=(6, 4), nrows=corr_size, ncols=corr_size)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
    for row in range(corr_size):
      for col in range(corr_size):
        if row > col:
          continue
        ax = fig.add_subplot(axs[row, col])
        ax.tick_params(left=False, labelleft=False, labelbottom=False,
                       bottom=False, top=False, labeltop=False)
        plt.axvline(x=corr[row,col])
        plt.xlim(-1, 1)
    plt.show()
    # return corr


  def pcorr_cross_pop(self):
    corr_size = self.num_qs * self.num_areas
    corr = util.partial_corr_from_cov(self.sigma_cross_pop)
    fig, axs = plt.subplots(figsize=(6, 4), nrows=corr_size, ncols=corr_size)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
    for row in range(corr_size):
      for col in range(corr_size):
        if row > col:
          continue
        ax = fig.add_subplot(axs[row, col])
        ax.tick_params(left=False, labelleft=False, labelbottom=False,
                       bottom=False, top=False, labeltop=False)
        plt.axvline(x=corr[row,col])
        plt.xlim(-1, 1)
    plt.show()
    # return corr


  def generate_mu_sigma(
      self,
      sample_type='fixed',
      df=40,
      verbose=True):
    """Generates the cross-pop parameters.

    Args:
      sample_type: 'fixed', 'random'.
    """
    self.mu_cross_pop = np.zeros(self.num_areas * self.num_qs)
    if self.model_feature_type in ['B'] and self.num_areas == 2:
      rho = 0.65
      cov = self.sigma_cross_pop_original.copy()
      cov[0,1] = rho * np.sqrt(cov[0,0] * cov[1,1])
      cov[1,0] = cov[0,1]
    if self.model_feature_type in ['B'] and self.num_areas == 3:
      cov = self.sigma_cross_pop_original.copy()
      cov[0,1] = 0.5 * np.sqrt(cov[0,0] * cov[1,1])
      cov[1,0] = cov[0,1]
      cov[0,2] = 0.7 * np.sqrt(cov[0,0] * cov[2,2])
      cov[2,0] = cov[0,2]
      cov[1,2] = 0.75 * np.sqrt(cov[1,1] * cov[2,2])
      cov[2,1] = cov[1,2]
    if self.model_feature_type in ['S1']:
      cov = np.array([[5e-4, 4e-4], [4e-4, 7e-4]])
    if self.model_feature_type in ['S2']:
      cov = np.array([[4e-4, 4e-4], [4e-4, 8e-4]])
    if self.model_feature_type in ['BSS']:
      # This is only for two areas.
      if self.num_areas == 2:
        cov = np.array([
            [9.42010828e-02,  1.52732009e-03, -3.31906674e-03,  5.28119729e-02,
             7.06256084e-04, -4.38873519e-03],
            [1.52732009e-03,  1.35553133e-04,  6.05657308e-05,  7.23482955e-04,
             1.40175025e-04, 7.39335840e-05],
            [-3.31906674e-03,  6.05657308e-05,  6.42238977e-04, -3.92200522e-03,
             8.92144767e-05, 5.63300566e-04],
            [5.28119729e-02,  7.23482955e-04, -3.92200522e-03,  6.89249031e-02,
            6.78451312e-04, -4.62127367e-03],
            [7.06256084e-04,  1.40175025e-04,  8.92144767e-05,  6.78451312e-04,
             2.04737951e-04, 1.62755039e-04],
            [-4.38873519e-03,  7.39335840e-05,  5.63300566e-04, -4.62127367e-03,
             1.62755039e-04, 7.47627938e-04]])
      else:
        cov = self.sigma_cross_pop

    if sample_type == 'random':
      scalor = df - 1 - self.num_areas * self.num_qs
      scale = cov * scalor
      self.sigma_cross_pop = scipy.stats.invwishart.rvs(scale=scale, df=df)
    elif sample_type == 'fixed':
      # Keep the current value of sigma_cross_pop.
      self.sigma_cross_pop = cov

    if verbose:
      self.corr_cross_pop()
      self.pcorr_cross_pop()
      corr_matrix = np.ones(self.sigma_cross_pop.shape)
      for row in range(len(self.sigma_cross_pop)):
        for col in range(len(self.sigma_cross_pop)):
          corr_matrix[row, col] = self.sigma_cross_pop[row,col] / np.sqrt(
        self.sigma_cross_pop[row,row] * self.sigma_cross_pop[col,col])

      print('sigma_cross_pop')
      print(self.sigma_cross_pop)
      print('sigma diag')
      print(np.diag(self.sigma_cross_pop))
      print('mu_cross_pop')
      print(self.mu_cross_pop)
      print('pairwise rho')
      print(corr_matrix)
      print('eiv')
      print(np.linalg.eigvals(self.sigma_cross_pop))


  def draw_limited_qs(
      self,
      r,
      c,
      max_iter=100,
      verbose=False):
    # Draw qs in a limited range just for one trial.
    for itr in range(max_iter):
      q_cnd = scipy.stats.multivariate_normal.rvs(
        mean=self.mu_cross_pop, cov=self.sigma_cross_pop)
      quanlified = True
      for a, probe in enumerate(self.probes):
        if self.model_feature_type in ['S', 'S1', 'S2']:
          qid1 = a * self.num_qs + 0
        if self.model_feature_type in ['BS']:
          qid1 = a * self.num_qs + 1
        if self.model_feature_type in ['SS']:
          qid1 = a * self.num_qs + 0
          qid2 = a * self.num_qs + 1
        if self.model_feature_type in ['BSS']:
          qid1 = a * self.num_qs + 1
          qid2 = a * self.num_qs + 2

        if self.model_feature_type in ['S', 'S1', 'S2', 'BS', 'SS', 'BSS']:
          # Peak-1
          f_peak1 = self.f_peak1_ac[a,c]
          f_peak1_cnd = q_cnd[qid1] + f_peak1
          if f_peak1_cnd < 0.01 or f_peak1_cnd > 0.14:
            quanlified = False
            if verbose:
              print(f'peak1 not quanlified: {f_peak1_cnd}')

        if self.model_feature_type in ['SS', 'BSS']:
          # Peak-2
          f_peak2 = self.f_peak2_ac[a,c]
          f_peak2_cnd = q_cnd[qid2] + f_peak2
          if f_peak2_cnd < 0.15 or f_peak2_cnd > 0.4:
            quanlified = False
            if verbose:
              print(f'peak2 not quanlified: {f_peak2_cnd}')

      if quanlified:
        break
    if itr == max_iter-1 and not quanlified:
      print('Warning: Drawing q_cnd samples runs out of iterations.')

    return q_cnd


  def generate_q(
      self,
      max_iter=100,
      verbose=False):
    """Generates features deviations."""
    # Old method without limiting the values in the peak ranges.
    # q_samples = scipy.stats.multivariate_normal.rvs(
    #     size=self.num_trials * self.num_conditions,
    #     mean=self.mu_cross_pop, cov=self.sigma_cross_pop)
    # # Fill trials axis first, then the conditions axis.
    # q_samples = q_samples.T.reshape(
    #     self.num_areas * self.num_qs, self.num_conditions, self.num_trials
    #     ).transpose(0,2,1)
    # q_samples_mean = q_samples.mean(axis=1).reshape(
    #     self.num_areas * self.num_qs, 1, self.num_conditions)
    # q_samples -= q_samples_mean
    # if verbose:
    #   print('q_samples.shape:', q_samples.shape)
    #   print('plotting original q dots.')

    q_samples = np.zeros([self.num_qs * self.num_areas,
        self.num_trials, self.num_conditions])
    for r in range(self.num_trials):
      for c in range(self.num_conditions):
        q_samples[:,r,c] = self.draw_limited_qs(r, c, verbose=False)

    if verbose and self.model_feature_type in ['BSS']:
      # Check peak-1 corr.
      qids = np.arange(self.num_areas) * self.num_qs + 1
      qs = q_samples[qids,:,:].transpose(0,2,1).reshape(self.num_areas,-1)
      corr = np.corrcoef(qs)
      print('peak-1 corr')
      print(corr)

    # Draw q's.
    if self.model_feature_type in ['B']:
      self.q_arc = q_samples
    if self.model_feature_type in ['S1', 'S2']:
      self.q_shift1_arc = q_samples
    if self.model_feature_type in ['BSS']:
      qids = np.arange(self.num_areas) * self.num_qs + 0
      mean_correction = np.mean(q_samples[qids,:,:], axis=1, keepdims=True)
      self.q_arc = q_samples[qids,:,:] - mean_correction
      qids = np.arange(self.num_areas) * self.num_qs + 1
      mean_correction = np.mean(q_samples[qids,:,:], axis=1, keepdims=True)
      self.q_shift1_arc = q_samples[qids,:,:] - mean_correction
      qids = np.arange(self.num_areas) * self.num_qs + 2
      mean_correction = np.mean(q_samples[qids,:,:], axis=1, keepdims=True)
      self.q_shift2_arc = q_samples[qids,:,:] - mean_correction

    # Update f_warp_sources_arc, f_warp_targets_arc
    if self.model_feature_type in ['BSS']:
      c = 0
      for r in range(self.num_trials):
        sources = [[] for i in range(self.num_areas)]
        targets = [[] for i in range(self.num_areas)]
        for a, probe in enumerate(self.probes):
          f_peak1 = self.f_peak1_ac[a,c]
          f_peak2 = self.f_peak2_ac[a,c]
          sources[a].extend([0, f_peak1, 0.15])
          targets[a].extend([0, f_peak1 + self.q_shift1_arc[a,r,c], 0.15])
          sources[a].extend([0.15, f_peak2, 0.4])
          targets[a].extend([0.15, f_peak2 + self.q_shift2_arc[a,r,c], 0.4])

        self.f_warp_sources_arc[:,r,c] = sources
        self.f_warp_targets_arc[:,r,c] = targets


    #--------------- verbose session ------------------
    if verbose and self.model_feature_type in ['B'] and self.num_areas == 2:
      corr_hat = np.corrcoef(self.q_arc.reshape(self.num_areas, -1))
      plt.figure(figsize=(8, 3))
      plt.subplot(121)
      plt.plot(self.q_arc_original[0,:,:].reshape(-1),
               self.q_arc_original[1,:,:].reshape(-1), '.')
      plt.title('Original qs')
      ax = plt.subplot(122)
      plt.plot(self.q_arc[0,:,:].reshape(-1),
               self.q_arc[1,:,:].reshape(-1), '.')
      plt.title('Generated qs')
      plt.text(0.1, 0.8, f'corr {corr_hat[0,1]}',
          transform=ax.transAxes)

    if verbose and self.model_feature_type in ['B'] and self.num_areas == 3:
      corr_hat = np.corrcoef(self.q_arc.reshape(self.num_areas, -1))
      plt.figure(figsize=(8, 2))
      plt.subplot(131)
      plt.plot(self.q_arc_original[0,:,:].reshape(-1),
               self.q_arc_original[1,:,:].reshape(-1), '.')
      plt.title('Original qs 0 -- 1')
      plt.subplot(132)
      plt.plot(self.q_arc_original[0,:,:].reshape(-1),
               self.q_arc_original[2,:,:].reshape(-1), '.')
      plt.title('Original qs 0 -- 2')
      plt.subplot(133)
      plt.plot(self.q_arc_original[1,:,:].reshape(-1),
               self.q_arc_original[2,:,:].reshape(-1), '.')
      plt.title('Original qs 1 -- 2')

      plt.figure(figsize=(8, 2))
      ax = plt.subplot(131)
      plt.plot(self.q_arc[0,:,:].reshape(-1),
               self.q_arc[1,:,:].reshape(-1), '.')
      plt.title('Generated qs 0 -- 1')
      plt.text(0.1, 0.8, f'corr {corr_hat[0,1]:.3f}',
          transform=ax.transAxes)
      ax = plt.subplot(132)
      plt.plot(self.q_arc[0,:,:].reshape(-1),
               self.q_arc[2,:,:].reshape(-1), '.')
      plt.title('Generated qs 0 -- 2')
      plt.text(0.1, 0.8, f'corr {corr_hat[0,2]:.3f}',
          transform=ax.transAxes)
      ax = plt.subplot(133)
      plt.plot(self.q_arc[1,:,:].reshape(-1),
               self.q_arc[2,:,:].reshape(-1), '.')
      plt.title('Generated qs 1 -- 2')
      plt.text(0.1, 0.8, f'corr {corr_hat[1,2]:.3f}',
          transform=ax.transAxes)

    if verbose and self.model_feature_type in ['S1']:
      corr_hat = np.corrcoef(self.q_shift1_arc.reshape(self.num_areas, -1))
      plt.figure(figsize=(8, 3))
      plt.subplot(121)
      plt.plot(self.q_shift1_arc_original[0,:,:].reshape(-1),
               self.q_shift1_arc_original[1,:,:].reshape(-1), '.')
      ax = plt.subplot(122)
      plt.plot(self.q_shift1_arc[0,:,:].reshape(-1),
               self.q_shift1_arc[1,:,:].reshape(-1), '.')
      plt.text(0.1, 0.8, f'corr {corr_hat[0,1]}',
          transform=ax.transAxes)

    if verbose and self.model_feature_type in ['S2']:
      corr_hat = np.corrcoef(self.q_shift1_arc.reshape(self.num_areas, -1))
      plt.figure(figsize=(8, 3))
      plt.subplot(121)
      plt.plot(self.q_shift2_arc_original[0,:,:].reshape(-1),
               self.q_shift2_arc_original[1,:,:].reshape(-1), '.')
      ax = plt.subplot(122)
      plt.plot(self.q_shift1_arc[0,:,:].reshape(-1),
               self.q_shift1_arc[1,:,:].reshape(-1), '.')
      plt.text(0.1, 0.8, f'corr {corr_hat[0,1]}',
          transform=ax.transAxes)

    if verbose and self.model_feature_type in ['BSS']:
      corr_hat = np.corrcoef(self.q_arc.reshape(self.num_areas, -1))
      plt.figure(figsize=(8, 4))
      plt.subplot(231)
      plt.plot(self.q_arc_original[0,:,:].reshape(-1),
               self.q_arc_original[1,:,:].reshape(-1), '.')
      ax = plt.subplot(234)
      plt.plot(self.q_arc[0,:,:].reshape(-1),
               self.q_arc[1,:,:].reshape(-1), '.')
      plt.text(0.1, 0.8, f'corr {corr_hat[0,1]:.3f}',
          transform=ax.transAxes)

      corr_hat = np.corrcoef(self.q_shift1_arc.reshape(self.num_areas, -1))
      plt.subplot(232)
      plt.plot(self.q_shift1_arc_original[0,:,:].reshape(-1),
               self.q_shift1_arc_original[1,:,:].reshape(-1), '.')
      ax = plt.subplot(235)
      plt.plot(self.q_shift1_arc[0,:,:].reshape(-1),
               self.q_shift1_arc[1,:,:].reshape(-1), '.')
      plt.text(0.1, 0.8, f'corr {corr_hat[0,1]:.3f}',
          transform=ax.transAxes)

      corr_hat = np.corrcoef(self.q_shift2_arc.reshape(self.num_areas, -1))
      plt.subplot(233)
      plt.plot(self.q_shift2_arc_original[0,:,:].reshape(-1),
               self.q_shift2_arc_original[1,:,:].reshape(-1), '.')
      ax = plt.subplot(236)
      plt.plot(self.q_shift2_arc[0,:,:].reshape(-1),
               self.q_shift2_arc[1,:,:].reshape(-1), '.')
      plt.text(0.1, 0.8, f'corr {corr_hat[0,1]:.3f}',
          transform=ax.transAxes)


  def generate_q_old(
      self,
      max_iter=100,
      verbose=False):
    """Generates features deviations.

    This file is recovered for Nov 21, 2020 for BSS_180 simulation.
    """
    q_samples = np.zeros([self.num_qs * self.num_areas,
        self.num_trials, self.num_conditions])
    for r in range(self.num_trials):
      for c in range(self.num_conditions):
        # NOTE: This part is the original code on Nov 21, 2020.
        q_samples[:,r,c] = self.draw_limited_qs(0, 1, verbose=False)

    if self.model_feature_type in ['B']:
      self.q_arc = q_samples
    if self.model_feature_type in ['S1', 'S2']:
      self.q_shift1_arc = q_samples
    if self.model_feature_type in ['BSS']:
      qids = np.arange(self.num_areas) * self.num_qs + 0
      mean_correction = np.mean(q_samples[qids,:,:], axis=1, keepdims=True)
      self.q_arc = q_samples[qids,:,:] - mean_correction
      qids = np.arange(self.num_areas) * self.num_qs + 1
      mean_correction = np.mean(q_samples[qids,:,:], axis=1, keepdims=True)
      self.q_shift1_arc = q_samples[qids,:,:] - mean_correction
      qids = np.arange(self.num_areas) * self.num_qs + 2
      mean_correction = np.mean(q_samples[qids,:,:], axis=1, keepdims=True)
      self.q_shift2_arc = q_samples[qids,:,:] - mean_correction

    if verbose and self.model_feature_type in ['BSS']:
      corr_hat = np.corrcoef(self.q_arc.reshape(self.num_areas, -1))
      plt.figure(figsize=(8, 4))
      plt.subplot(231)
      plt.plot(self.q_arc_original[0,:,:].reshape(-1),
               self.q_arc_original[1,:,:].reshape(-1), '.')
      ax = plt.subplot(234)
      plt.plot(self.q_arc[0,:,:].reshape(-1),
               self.q_arc[1,:,:].reshape(-1), '.')
      plt.text(0.1, 0.8, f'corr {corr_hat[0,1]:.3f}',
          transform=ax.transAxes)

      corr_hat = np.corrcoef(self.q_shift1_arc.reshape(self.num_areas, -1))
      plt.subplot(232)
      plt.plot(self.q_shift1_arc_original[0,:,:].reshape(-1),
               self.q_shift1_arc_original[1,:,:].reshape(-1), '.')
      ax = plt.subplot(235)
      plt.plot(self.q_shift1_arc[0,:,:].reshape(-1),
               self.q_shift1_arc[1,:,:].reshape(-1), '.')
      plt.text(0.1, 0.8, f'corr {corr_hat[0,1]:.3f}',
          transform=ax.transAxes)

      corr_hat = np.corrcoef(self.q_shift2_arc.reshape(self.num_areas, -1))
      plt.subplot(233)
      plt.plot(self.q_shift2_arc_original[0,:,:].reshape(-1),
               self.q_shift2_arc_original[1,:,:].reshape(-1), '.')
      ax = plt.subplot(236)
      plt.plot(self.q_shift2_arc[0,:,:].reshape(-1),
               self.q_shift2_arc[1,:,:].reshape(-1), '.')
      plt.text(0.1, 0.8, f'corr {corr_hat[0,1]:.3f}',
          transform=ax.transAxes)


  def evaluate_rho_variance(
      self,
      i=0,
      j=1,
      num_trials=15,
      num_samples=50000):
    """Plot generated rho distribution."""
    rho_hats = np.zeros(num_samples)

    for s in range(num_samples):
      q_ar = np.zeros([self.num_areas * self.num_qs, num_trials])
      q_ar = scipy.stats.multivariate_normal.rvs(
        size=num_trials,
        mean=self.mu_cross_pop, cov=self.sigma_cross_pop).T

      q_ = q_ar.T - self.mu_cross_pop
      q_cov = 1 / (num_trials - 1) * q_.T @ q_
      rho_hats[s] = q_cov[i,j] / np.sqrt(q_cov[i,i] * q_cov[j,j])

    rhoz_hats = util.fisher_transform(rho_hats)
    rho_true = self.sigma_cross_pop[i,j] / np.sqrt(
        self.sigma_cross_pop[i,i] * self.sigma_cross_pop[j,j])
    rhoz_true = util.fisher_transform(rho_true)

    plt.figure(figsize=(7, 2.5))
    plt.subplot(121)
    x, y = seaborn.distplot(rho_hats, bins=50).get_lines()[0].get_data()
    plt.axvline(x=np.quantile(rho_hats, 0.025),
                linestyle=':', color='k')
    plt.axvline(x=np.quantile(rho_hats, 0.975),
                linestyle=':', color='k', label='%95 CI')
    # plt.axvline(x=x[np.argmax(y)], color='b', label='mode')
    plt.axvline(x=rho_true, color='g', label='true')
    plt.xlim(-1, 1)
    plt.legend()

    # plt.subplot(122)
    # x, y = seaborn.distplot(rhoz_hats, bins=50).get_lines()[0].get_data()
    # plt.axvline(x=np.quantile(rhoz_hats, 0.025),
    #             linestyle=':', color='k')
    # plt.axvline(x=np.quantile(rhoz_hats, 0.975),
    #             linestyle=':', color='k', label='%95 CI')
    # plt.axvline(x=x[np.argmax(y)], color='b', label='mode')
    # plt.axvline(x=rhoz_true, color='g', label='true')
    # # plt.xlim(-1, 1)

    plt.legend()
    plt.show()


  def generate_f_pop_gac(
      self,
      select_clist=[0],
      same_as_cross=False,
      verbose=True):
    """Selects the f_pop among c."""
    self.select_clist = select_clist
    self.f_pop_cag = self.f_pop_cag_original[select_clist,:,:,:]
    self.f_peak1_ac = self.f_peak1_ac_original[:,select_clist]
    self.f_peak2_ac = self.f_peak2_ac_original[:,select_clist]

    if same_as_cross:
      for c in range(self.num_conditions):
        for a, probe in enumerate(self.probes):
          self.f_pop_cag[c,a,1] = self.f_pop_cag[c,a,0]

    if verbose:
      for c in range(self.num_conditions):
        self.plot_f_pop(c)


  def generate_z(
      self,
      num_neurons_per_probe=None,
      verbose=True):
    """Generates z.

    Args:
      num_neurons_per_probe: a list of numbers of neurons per probe.
    """
    # Strategy 1: use the previous fit as a template.
    if num_neurons_per_probe is None:
      sub_group_df_c = self.sub_group_df_c_original.copy()
      for c in range(self.num_conditions):
        group_id_col = sub_group_df_c[c].columns.get_loc('group_id')
        sub_group_df_c[c].iloc[0::3, group_id_col] = 0
        sub_group_df_c[c].iloc[1::3, group_id_col] = 1
        sub_group_df_c[c].iloc[2::3, group_id_col] = 2
      self.sub_group_df_c = sub_group_df_c

    # Strategy 2: use specific number of neurons.
    else:
      unit_ids = np.arange(np.sum(num_neurons_per_probe)).astype(int)
      group_id = np.zeros(len(unit_ids)).astype(int)
      import string
      probe_name_full_list = ['probe'] * len(string.ascii_uppercase)
      for i, char in enumerate(string.ascii_uppercase):
        probe_name_full_list[i] = probe_name_full_list[i] + char
      probes = ['probe?'] * len(unit_ids)
      index_range = [0] + num_neurons_per_probe
      index_range = np.cumsum(index_range)
      for i in range(len(num_neurons_per_probe)):
        probes[index_range[i]:index_range[i+1]] = (
            [probe_name_full_list[i]] * num_neurons_per_probe[i])

      sub_group_df = pd.DataFrame({
          'unit_ids': unit_ids,
          'probe': probes,
          'group_id': group_id})
      sub_group_df.set_index('unit_ids', inplace=True)
      sub_group_df_c = [sub_group_df.copy() for _ in range(self.num_conditions)]

      for c in range(self.num_conditions):
        group_id_col = sub_group_df_c[c].columns.get_loc('group_id')
        sub_group_df_c[c].iloc[0::3, group_id_col] = 0
        sub_group_df_c[c].iloc[1::3, group_id_col] = 1
        sub_group_df_c[c].iloc[2::3, group_id_col] = 2

      self.sub_group_df_c = sub_group_df_c

    if verbose:
      for c in range(self.num_conditions):
        print('Origial membership.')
        print(self.sub_group_df_c_original[c].groupby(
            ['probe', 'group_id']).size())
        print('New membership.')
        print(sub_group_df_c[c].groupby(['probe', 'group_id']).size())


  def generate_p_gac(
      self,
      verbose=True):
    self.p_gac = self.p_gac_original.copy()
    self.p_gac[:,:,:] = 1 / 3

    if verbose:
      print(self.p_gac)

  # TODO: make this for each neuron, not just for each group, so that it can 
  # handle the neuron variation simulation.
  def generate_lambda_garc(
      self,
      verbose=False):
    """Generates lambda for ks test purpose."""
    c = 0  # condition id in the simulation, NOT the c in the template.
    sub_group_df = self.sub_group_df_c[c]
    selected_units = self.selected_units
    units = selected_units[selected_units['probe_description'].isin(self.probes)]
    self.log_lambda_nr = pd.DataFrame(index=units.index.values,
                                      columns=np.arange(self.num_trials))
    num_bins = len(self.spike_train_time_line)
    lambda_garc = np.zeros([self.num_groups, self.num_areas, self.num_trials,
                            self.num_conditions, num_bins])

    for a, probe in enumerate(self.probes):

      # Group 0.
      g = 0
      units_cross_pop = sub_group_df[
          (sub_group_df['probe'] == probe) &
          (sub_group_df['group_id'] == g)].index.values
      for r in range(self.num_trials):
        log_lambda = self.f_pop_cag[c,a,g].copy()

        if self.model_feature_type in ['S1']:
          source = [0, self.f_peak1_ac[a,c], 0.13]
          target = [0, self.f_peak1_ac[a,c] + self.q_shift1_arc[a,r,c], 0.13]
          log_lambda = self.linear_time_warping(
              self.spike_train_time_line, log_lambda,
              source, target, verbose=False)
          lambda_garc[g,a,r,c] = np.exp(log_lambda)

        if self.model_feature_type in ['S2']:
          source = [0.13, self.f_peak2_ac[a,c], 0.4]
          target = [0.13, self.f_peak2_ac[a,c] + self.q_shift1_arc[a,r,c], 0.4]
          log_lambda = self.linear_time_warping(
              self.spike_train_time_line, log_lambda,
              source, target, verbose=False)
          lambda_garc[g,a,r,c] = np.exp(log_lambda)

        if self.model_feature_type in ['BSS']:
          source = [0, self.f_peak1_ac[a,c], 0.13,
                    0.13, self.f_peak2_ac[a,c], 0.4]
          target = [0, self.f_peak1_ac[a,c] + self.q_shift1_arc[a,r,c], 0.13,
                    0.13, self.f_peak2_ac[a,c] + self.q_shift2_arc[a,r,c], 0.4]
          log_lambda = self.linear_time_warping(
              self.spike_train_time_line, log_lambda,
              source, target, verbose=False)

        if self.model_feature_type in ['B', 'BS', 'BSS']:
          lambda_garc[g,a,r,c] = np.exp(log_lambda + self.q_arc[a,r,c])

        if any(np.isnan(log_lambda)):
          print('g 0 nan')

      # Group 1.
      g = 1
      log_lambda = self.f_pop_cag[c,a,g].copy()
      for r in range(self.num_trials):
        lambda_garc[g,a,r,c] = np.exp(log_lambda)

      # Group 2.
      g = 2
      log_lambda = self.f_pop_cag[c,a,g].copy()
      for r in range(self.num_trials):
        lambda_garc[g,a,r,c] = np.exp(log_lambda)
        if any(np.isnan(log_lambda)):
          print('g 2 nan')

    return lambda_garc

  # TODO: make this for each neuron, not just for each group, so that it can 
  # handle the neuron variation simulation.

  # def get_lambda_nrc(self):
  #   """Gets lambdas of each neuron for ks test purpose."""
  #   return np.exp(self.log_lambda_nr)

  # TODO: log_lambda_nr --> list of log_lambda_nr_c
  def generate_log_lambda_nargc(
      self,
      c=0,
      variation='none',
      laplacian_scalar=1,
      burn_in=0,
      end=None,
      step=1,
      verbose=True):
    """Generates the intensity function for each neuron each trial.

    Args:
      c: condition id in the simulation, NOT the c in the template.
      variation: 'samples', 'laplacian', 'none'
    """
    sub_group_df = self.sub_group_df_c[c]
    selected_units = self.selected_units
    units = selected_units[selected_units['probe_description'].isin(self.probes)]
    self.log_lambda_nr = pd.DataFrame(index=units.index.values,
                                      columns=np.arange(self.num_trials))
    select_c = self.select_clist[c]
    spike_train_time_line = self.spike_train_time_line
    num_bins = len(spike_train_time_line)

    if variation == 'samples':
      if not hasattr(self, 'samples') or len(self.samples.f_pop)==0:
        raise ValueError('No samples loaded. Needed for lambda variation.')
      f_samples = np.stack(self.samples.f_pop[burn_in:end:step], axis=0)
      print('f_samples.shape:', f_samples.shape)
      f_samples = f_samples[:,[select_c],:,:,:]
      num_samples, num_conditions, num_areas, num_groups, num_bins = f_samples.shape
      print(f_samples.shape)
    elif variation == 'laplacian':
      self.laplacian_scalar = laplacian_scalar

    for a, probe in enumerate(self.probes):
      # Group 0.
      g = 0
      units_cross_pop = sub_group_df[
          (sub_group_df['probe'] == probe) &
          (sub_group_df['group_id'] == g)].index.values

      for n in units_cross_pop:
        #---------- neuron-to-neuron variation ------------
        if variation == 'none':
          log_lambda_n = self.f_pop_cag[c,a,g].copy()
        elif variation == 'samples':
          index = np.random.randint(num_samples)
          log_lambda_n = f_samples[index,c,a,g]
        elif variation == 'laplacian':
          hessian = self.f_pop_par_cag[(select_c,a,g,'beta_hessian')]
          hessian_baseline = self.f_pop_par_cag[(select_c,a,g,'beta_baseline_hessian')]
          beta_cov = np.linalg.inv(hessian) * laplacian_scalar
          beta_baseline_cov = 1 / hessian_baseline * laplacian_scalar
          beta_mean = self.f_pop_beta_cag[(select_c,a,g,'beta')]
          beta_baseline_mean = self.f_pop_beta_cag[(select_c,a,g,'baseline')]
          beta_cnd = np.random.multivariate_normal(
              beta_mean.reshape(-1), beta_cov).reshape(-1, 1)
          beta_baseline_cnd = np.random.normal(beta_baseline_mean, beta_baseline_cov)
          log_lambda_n = (self.f_basis @ beta_cnd + beta_baseline_cnd).reshape(-1)

          # log_lambda_mean = (self.f_basis @ beta_mean + beta_baseline_mean).reshape(-1)
          # plt.figure(figsize=[4,2])
          # plt.plot(np.exp(log_lambda_mean))
          # plt.plot(np.exp(self.f_pop_cag[c,a,g]))
          # return

        for r in range(self.num_trials):
          log_lambda = log_lambda_n.copy()
          # ---------- neuron-to-neuron and trial-to-trial variation ------------
          # if variation == 'samples':
          #   index = np.random.randint(num_samples)
          #   log_lambda = f_samples[index,c,a,g]
          # elif variation == 'laplacian':
          #   hessian = self.f_pop_par_cag[(select_c,a,g,'beta_hessian')]
          #   hessian_baseline = self.f_pop_par_cag[(select_c,a,g,'beta_baseline_hessian')]
          #   beta_cov = np.linalg.inv(hessian) * laplacian_scalar
          #   beta_baseline_cov = 1 / hessian_baseline * laplacian_scalar
          #   beta_mean = self.f_pop_beta_cag[(select_c,a,g,'beta')]
          #   beta_baseline_mean = self.f_pop_beta_cag[(select_c,a,g,'baseline')]
          #   beta_cnd = np.random.multivariate_normal(
          #       beta_mean.reshape(-1), beta_cov).reshape(-1, 1)
          #   beta_baseline_cnd = np.random.normal(beta_baseline_mean, beta_baseline_cov)
          #   log_lambda = (self.f_basis @ beta_cnd + beta_baseline_cnd).reshape(-1)

          if self.model_feature_type in ['S1']:
            source = [0, self.f_peak1_ac[a,c], 0.13]
            target = [0, self.f_peak1_ac[a,c] + self.q_shift1_arc[a,r,c], 0.13]
            log_lambda = self.linear_time_warping(
                self.spike_train_time_line, log_lambda,
                source, target, verbose=False)
            self.log_lambda_nr.loc[n,r] = log_lambda

          if self.model_feature_type in ['S2']:
            source = [0.13, self.f_peak2_ac[a,c], 0.4]
            target = [0.13, self.f_peak2_ac[a,c] + self.q_shift1_arc[a,r,c], 0.4]
            log_lambda = self.linear_time_warping(
                self.spike_train_time_line, log_lambda,
                source, target, verbose=False)
            self.log_lambda_nr.loc[n,r] = log_lambda

          if self.model_feature_type in ['BSS']:
            source = [0, self.f_peak1_ac[a,c], 0.13,
                      0.13, self.f_peak2_ac[a,c], 0.4]
            target = [0, self.f_peak1_ac[a,c] + self.q_shift1_arc[a,r,c], 0.13,
                      0.13, self.f_peak2_ac[a,c] + self.q_shift2_arc[a,r,c], 0.4]
            log_lambda = self.linear_time_warping(
                self.spike_train_time_line, log_lambda,
                source, target, verbose=False)

          if self.model_feature_type in ['B', 'BS', 'BSS']:
            self.log_lambda_nr.loc[n, r] = log_lambda + self.q_arc[a,r,c]

          if any(np.isnan(log_lambda)):
            print('g 0 nan')

          if verbose:
            plt.figure(figsize=(4,2))
            plt.plot(self.spike_train_time_line, self.f_pop_cag[c,a,g], 'g')
            plt.plot(self.spike_train_time_line, self.log_lambda_nr.loc[n, r], 'r')
            plt.show()

      # Group 1.
      g = 1
      units_cross_pop = sub_group_df[
          (sub_group_df['probe'] == probe) &
          (sub_group_df['group_id'] == g)].index.values

      for n in units_cross_pop:
        #---------- neuron-to-neuron variation ------------
        if variation == 'none':
          log_lambda_n = self.f_pop_cag[c,a,g].copy()
        elif variation == 'samples':
          index = np.random.randint(num_samples)
          log_lambda_n = f_samples[index,c,a,g]
        elif variation == 'laplacian':
          hessian = self.f_pop_par_cag[(select_c,a,g,'beta_hessian')]
          hessian_baseline = self.f_pop_par_cag[(select_c,a,g,'beta_baseline_hessian')]
          beta_cov = np.linalg.inv(hessian) * laplacian_scalar
          beta_baseline_cov = 1 / hessian_baseline * laplacian_scalar
          beta_mean = self.f_pop_beta_cag[(select_c,a,g,'beta')]
          beta_baseline_mean = self.f_pop_beta_cag[(select_c,a,g,'baseline')]
          beta_cnd = np.random.multivariate_normal(
              beta_mean.reshape(-1), beta_cov).reshape(-1, 1)
          beta_baseline_cnd = np.random.normal(beta_baseline_mean, beta_baseline_cov)
          log_lambda_n = (self.f_basis @ beta_cnd + beta_baseline_cnd).reshape(-1)

        for r in range(self.num_trials):
          log_lambda = log_lambda_n.copy()
          #---------- neuron-to-neuron and trial-to-trial variation ------------
          # if variation == 'samples':
          #   index = np.random.randint(num_samples)
          #   log_lambda = f_samples[index,c,a,g]
          # elif variation == 'laplacian':
          #   hessian = self.f_pop_par_cag[(select_c,a,g,'beta_hessian')]
          #   hessian_baseline = self.f_pop_par_cag[(select_c,a,g,'beta_baseline_hessian')]
          #   beta_cov = np.linalg.inv(hessian) * laplacian_scalar
          #   beta_baseline_cov = 1 / hessian_baseline * laplacian_scalar
          #   beta_mean = self.f_pop_beta_cag[(select_c,a,g,'beta')]
          #   beta_baseline_mean = self.f_pop_beta_cag[(select_c,a,g,'baseline')]
          #   beta_cnd = np.random.multivariate_normal(
          #       beta_mean.reshape(-1), beta_cov).reshape(-1, 1)
          #   beta_baseline_cnd = np.random.normal(beta_baseline_mean, beta_baseline_cov)
          #   log_lambda = (self.f_basis @ beta_cnd + beta_baseline_cnd).reshape(-1)

          self.log_lambda_nr.loc[n, r] = log_lambda
          if any(np.isnan(log_lambda)):
            print('g 1 nan')

      # Group 2.
      g = 2
      units_cross_pop = sub_group_df[
          (sub_group_df['probe'] == probe) &
          (sub_group_df['group_id'] == g)].index.values

      for n in units_cross_pop:
        #---------- neuron-to-neuron variation ------------
        if variation == 'none':
          log_lambda_n = self.f_pop_cag[c,a,g].copy()
        elif variation == 'samples':
          index = np.random.randint(num_samples)
          log_lambda_n = f_samples[index,c,a,g]
        elif variation == 'laplacian':
          hessian_baseline = self.f_pop_par_cag[(select_c,a,g,'beta_baseline_hessian')]
          beta_baseline_cov = 1 / hessian_baseline * laplacian_scalar
          beta_baseline_mean = self.f_pop_beta_cag[(select_c,a,g,'baseline')]
          beta_baseline_cnd = np.random.normal(beta_baseline_mean, beta_baseline_cov)
          log_lambda_n = np.zeros(num_bins) + beta_baseline_cnd

        for r in range(self.num_trials):
          log_lambda = log_lambda_n.copy()
          #---------- neuron-to-neuron and trial-to-trial variation ------------
          # if variation == 'samples':
          #   index = np.random.randint(num_samples)
          #   log_lambda = f_samples[index,c,a,g]
          # elif variation == 'laplacian':
          #   hessian_baseline = self.f_pop_par_cag[(select_c,a,g,'beta_baseline_hessian')]
          #   beta_baseline_cov = 1 / hessian_baseline * laplacian_scalar
          #   beta_baseline_mean = self.f_pop_beta_cag[(select_c,a,g,'baseline')]
          #   beta_baseline_cnd = np.random.normal(beta_baseline_mean, beta_baseline_cov)
          #   log_lambda = np.zeros(num_bins) + beta_baseline_cnd

          self.log_lambda_nr.loc[n, r] = log_lambda
          if any(np.isnan(log_lambda)):
            print('g 2 nan')


  def log_lambda_laplacian_approximation(
      self,
      c=0,
      burn_in=0,
      end=None,
      step=1,):
    """Approximates the marginal posterior samles using Laplacian method."""
    select_c = self.select_clist[c]
    spike_train_time_line = self.spike_train_time_line
    num_bins = len(spike_train_time_line)
    if not hasattr(self, 'samples') or len(self.samples.f_pop)==0:
      raise ValueError('No samples loaded. Needed for lambda variation.')

    f_samples = np.stack(self.samples.f_pop[burn_in:end:step], axis=0)
    print('f_samples.shape:', f_samples.shape)
    f_samples = f_samples[:,[select_c],:,:,:]
    num_samples, num_conditions, num_areas, num_groups, num_bins = f_samples.shape
    print(f_samples.shape)

    log_sub_samples = f_samples[:500:20,c,1,0]
    sub_samples = np.exp(log_sub_samples)
    print(sub_samples.shape)
    plt.figure(figsize=[8,4])
    plt.plot(sub_samples.T)


    a = 1
    g = 0
    hessian = self.f_pop_par_cag[(select_c,a,g,'beta_hessian')]
    hessian_baseline = self.f_pop_par_cag[(select_c,a,g,'beta_baseline_hessian')]
    beta_cov = np.linalg.inv(hessian)
    beta_baseline_cov = 1 / hessian_baseline
    beta_mean = self.f_pop_beta_cag[(select_c,a,g,'beta')]
    beta_baseline_mean = self.f_pop_beta_cag[(select_c,a,g,'baseline')]

    log_lambda_mean = (self.f_basis @ beta_mean + beta_baseline_mean).reshape(-1)
    f_pop_mean = np.exp(log_lambda_mean) / self.dt

    log_lmbda_cov = np.linalg.inv(hessian)
    log_lmbda_cov = self.f_basis @ log_lmbda_cov @ self.f_basis.T
    CI_scale = scipy.stats.norm.ppf(0.975)  # 95% CI.
    log_lmbda_cov_std = np.sqrt(np.diag(log_lmbda_cov)) * CI_scale
    f_pop_err_up = np.exp(log_lambda_mean + log_lmbda_cov_std) / self.dt
    f_pop_err_dn = np.exp(log_lambda_mean - log_lmbda_cov_std) / self.dt

    plt.figure(figsize=[8,4])
    plt.plot(f_pop_mean, 'k')
    plt.plot(f_pop_err_up, 'k:')
    plt.plot(f_pop_err_dn, 'k:')


  def plot_log_lambda_nr(
      self,
      c=0,
      trials=[0,1,2,3,4,5]):
    """Plots the firing rate functions."""
    # log_lambdas = np.stack(self.log_lambda_nr.values.flatten('F'), axis=0)

    gs_kw = dict(width_ratios=[1] * self.num_groups,
                 height_ratios=[1] * self.num_areas)
    fig, axs = plt.subplots(figsize=(12, 5), gridspec_kw=gs_kw,
        nrows=self.num_areas, ncols=self.num_groups, constrained_layout=True)
    # plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)

    sub_group_df = self.sub_group_df_c[c]
    for a, probe in enumerate(self.probes):
      for g in range(self.num_groups):
        units = sub_group_df[(sub_group_df['probe']==probe) &
                             (sub_group_df['group_id']==g)].index.values
        log_lambdas = self.log_lambda_nr.loc[units, trials]
        log_lambdas = np.stack(log_lambdas.values.flatten('F'), axis=0)

        ax = fig.add_subplot(axs[a, g])
        plt.plot(np.exp(log_lambdas.T) / self.dt)
        plt.title(log_lambdas.shape)

    plt.show()


  @classmethod
  def generate_spike_train(
      cls,
      lmbd,
      dt=1,
      random_seed=None):
    """Generate one trial of spike train using firing rate lamdba.

    The spike train is assumed to be binary, i.e. the time interval is small
    enough.

    Args:
      lmbd: The firing rate.

    Returns:
      One spike train.
    """
    if random_seed:
      np.random.seed(random_seed)

    spike_train = np.zeros(len(lmbd))
    spike_times = []
    for t in range(len(lmbd)):
      num_spikes = np.random.poisson(lmbd[t])
      if num_spikes > 0:
        spike_train[t] = 1  # Binary event.
        # Poisson process theorem. Given the number of spikes in a certain
        # range, the events distributed evenly in the interval.
        spike_times.append(t * dt + np.random.rand() * dt)

    spike_times = np.array(spike_times)
    if len(spike_times) == 0:
      spike_shift = np.nan
    else:
      spike_shift = np.mean(spike_times)
    spike_count = np.sum(spike_train)
    return spike_train, spike_times, spike_count, spike_shift


  @classmethod
  def generate_spike_trains(
      cls,
      lmbd,
      dt=1,
      num_trials=1,
      random_seed=None):
    """Generate multiple spikes according to lmbd.

    Args:
      lmbd: The firing rate.

    Returns:
      Multiple spike trains.
    """
    spike_trains = np.zeros([num_trials, len(lmbd)])
    for r in range(num_trials):
      spike_trains[r],_,_,_ = cls.generate_spike_train(lmbd)

    return spike_trains


  def generate_spikes(
      self,
      verbose=False):
    """Generates spike trains and spike times according to model lambd."""
    c = 0
    log_lambda_nr = self.log_lambda_nr  # TODO: shoud be self.log_lambda_nr_c[c]
    sub_group_df = self.sub_group_df_c[c]
    units = sub_group_df[sub_group_df['probe'].isin(self.probes)].index.values
    trials = self.log_lambda_nr.columns.values
    dt = self.spike_train_time_line[1] - self.spike_train_time_line[0]

    self.spike_trains = pd.DataFrame(
        index=units, columns=np.arange(self.num_trials))
    self.spike_times = pd.DataFrame(
        index=units, columns=np.arange(self.num_trials))
    self.spike_counts = pd.DataFrame(
        index=units, columns=np.arange(self.num_trials))
    self.spike_shifts = pd.DataFrame(
        index=units, columns=np.arange(self.num_trials))
    self.spike_trains.index.name = 'units'
    self.spike_times.index.name = 'units'
    self.spike_counts.index.name = 'units'
    self.spike_shifts.index.name = 'units'

    for n in units:
      for r in trials:
        (self.spike_trains.loc[n,r], self.spike_times.loc[n,r],
         self.spike_counts.loc[n,r], self.spike_shifts.loc[n,r]) = (
            self.generate_spike_train(
                np.exp(log_lambda_nr.loc[n,r]), dt=dt))

    self.spike_counts = self.spike_counts.apply(pd.to_numeric, errors='coerce')
    self.spike_shifts = self.spike_shifts.apply(pd.to_numeric, errors='coerce')

    if verbose:
      plt.figure(figsize=(6, 3))
      for a, probe in enumerate(self.probes):
        units = sub_group_df[sub_group_df['probe'] == probe].index.values
        spikes = self.spike_trains.loc[units,:]
        spikes = np.stack(spikes.values.flatten('F'), axis=0)
        plt.plot(self.spike_train_time_line, np.mean(spikes, axis=0))


  @classmethod
  def generate_poisson_spike_times(
      cls,
      lmbd,
      length,
      num_trials=1,
      verbose=False):
    """Generate homogeneous Poisson spike times (as opposed to spike bins).

    Args:
      lmbd: 
      length: length of the trial. Unit in second.
    """
    spike_times = []
    mu = 1 / lmbd
    interval_sum = 0

    for r in range(num_trials):
      spike_times_ = cls._generate_poisson_spike_times_single_trial(lmbd, length)
      spike_times.append(np.array(spike_times_))

    if num_trials == 1:
      if verbose:
        print(spike_times[0])
      return np.array(spike_times[0])
    else:
      if verbose:
        plt.figure(figsize=[8, 2])
        for r in range(num_trials):
          plt.plot(spike_times[r], np.zeros(len(spike_times[r]))+r, 'ks', ms=1)
        plt.axvline(x=0, c='k', lw=0.3)
        plt.axvline(x=length, c='k', lw=0.3)
        plt.ylim(-1, num_trials)

      return spike_times


  @classmethod
  def _generate_poisson_spike_times_single_trial(
      cls,
      lmbd,
      length):
    """Generate homogeneous Poisson spike times (as opposed to spike bins).

    Args:
      lmbd: 
      length: length of the trial. Unit in second.
    """
    spike_times = []
    mu = 1 / lmbd
    curr_time = 0

    while curr_time < length:
      # if set size=1, then the output is a one element 1-d array.
      interval = np.random.exponential(scale=mu)
      curr_time += interval
      spike_times.append(curr_time)

    # The last one must stay outside the length.
    return spike_times[:-1]


  @classmethod
  def generate_inhomo_poisson_density_spike_times(
      cls,
      intensity_func=None,
      length=None,
      intensity_integral=None,
      intensity_max=None,
      verbose=False):
    """Generate contineous inhomogeneous Poisson process samples.

    Rejection sampling according to the density propto lambda(t).

    Args:
      lmbd_func: Closed form intensity function. It can output the intensity
          values at any time in the range.
      intensity_max: Used for constructing rejection sampling.
    """
    num_spikes = np.random.poisson(lam=intensity_integral)
    spike_times = np.zeros(num_spikes)
    spike_cnt = 0
    sample_cnt = 0
    while spike_cnt < num_spikes:
      sample_cnt += 1
      spike_cnd = np.random.rand() * length
      intensity_x = intensity_func(spike_cnd)
      reject_ratio = intensity_x / intensity_max
      u = np.random.rand()
      if u <= reject_ratio:  # Accept the sample.
        spike_times[spike_cnt] = spike_cnd
        spike_cnt += 1

    if verbose:
      print(f'Sample efficiency: {np.round(spike_cnt/sample_cnt*100, 1)}')
    return np.sort(spike_times)


  @classmethod
  def generate_amarasingham_example_spike_times(
      cls,
      num_peaks=40,
      sigma=0.04,
      trial_length=1,
      num_trials=1,
      baseline=10,
      sample_type='density',
      verbose=False,
      file_path=None):
    """Amarasingham et. al. 2011 simulation scinarios Fig. 1.

    Args:
      trial_length: unit second.
      sample_type:
          density: Draw number of samples first, then draw iid according to the
              density lamda(t) / Lambda. Efficiency is around 0.4 for sigma=0.02.
              Efficiency is around 0.2 for sigma=0.005. Sampling from the
              arbitrary density uses rejection sampling.
          transform: TODO. This has to compromise to approximated solution as
              arbitrary inverse integral of the intensity has no closed form.
              I will use binary search to find the closest point using binary
              search. But the efficiency is is not attractive.
          thinning: TODO. Draw homogeneous Poisson according to Lambda. Then
              remove the points according to the the ration = lambda(t) / Lambda.
              This method is actually exactly the same as the density method
              using rejection sampling.
    """
    intensity_integral = baseline * trial_length + num_peaks
    peaks = np.random.rand(num_peaks) * trial_length  # Uniform peak positions.
    peaks = np.sort(peaks)

    def intensity_func(t):
      """Intensity function with mixture of Laplacian's.

      The scalue is sigma/sqrt(2) by following the Amarasingham 2012 paper
      appendix.
      """
      t = np.array(t)
      if np.ndim(t) == 0:
        sample_points = scipy.stats.laplace.pdf(t - peaks,
            loc=0, scale=sigma/np.sqrt(2))
        intensity = sample_points.sum() + baseline
      else:
        num_t = len(t)
        intensity = np.zeros(num_t)
        for i in range(num_t):
          sample_points = scipy.stats.laplace.pdf(t[i] - peaks,
              loc=0, scale=sigma/np.sqrt(2))
          intensity[i] = sample_points.sum() + baseline
      return intensity

    intensity_peaks = intensity_func(peaks)
    intensity_max_x = peaks[np.argmax(intensity_peaks)]
    intensity_max = intensity_peaks.max()

    if sample_type == 'density':
      if num_trials == 1:
        spike_times = cls.generate_inhomo_poisson_density_spike_times(
            intensity_func=intensity_func, length=trial_length,
            intensity_integral=intensity_integral, intensity_max=intensity_max,
            verbose=False)
      else:
        if verbose==2:
          trange = tqdm(range(num_trials), ncols=100, file=sys.stdout)
        else:
          trange = range(num_trials)
        spike_times = []
        for r in trange:
          spike_times_single = cls.generate_inhomo_poisson_density_spike_times(
            intensity_func=intensity_func, length=trial_length,
            intensity_integral=intensity_integral, intensity_max=intensity_max,
            verbose=False)
          spike_times.append(spike_times_single)

    if verbose:
      jittertool = jitter.JitterTool()

      x = np.arange(0, trial_length, 0.001)
      x = np.concatenate((x, peaks))
      x = np.sort(x)
      y = intensity_func(x)
      gs_kw = dict(width_ratios=[1], height_ratios=[1,1,1])
      fig, axs = plt.subplots(figsize=(10, 8), gridspec_kw=gs_kw,
        nrows=3, ncols=1)
      plt.subplots_adjust(hspace=0, wspace=0)
      ax = fig.add_subplot(axs[0])
      ax.tick_params(labelbottom=False)
      for peak in peaks:
        plt.axvline(x=peak, c='grey', lw=0.3)
      plt.plot(intensity_max_x, intensity_max, 'r+', ms=12)
      plt.plot(x, y, 'k')
      plt.xlim(0, trial_length)
      plt.ylim(0, intensity_max+20)
      plt.ylabel('Firing rate [spk/sec]')

      ax = fig.add_subplot(axs[1])
      ax.tick_params(labelbottom=False, labelleft=True)
      if num_trials == 1:
        plt.plot(spike_times, np.zeros(len(spike_times)), 'ks', ms=2)
      else:
        plot_step = 4
        for r in range(0, num_trials, plot_step):
          plt.plot(spike_times[r], r+np.zeros(len(spike_times[r])), 'ks', ms=0.4)
      plt.xlim(0, trial_length)
      plt.ylim(-50, num_trials+50)
      plt.ylabel('Trials')

      ax = fig.add_subplot(axs[2])
      # ax.tick_params(labelleft=False)
      spike_hist, bins = jittertool.bin_spike_times(spike_times, 0.005, trial_length)
      plt.plot(bins, spike_hist.mean(axis=0) / 0.005, 'k')
      plt.xlim(0, trial_length)
      plt.ylim(0, intensity_max+20)
      plt.ylabel('Firing rate [spk/sec]')
      plt.xlabel('Time [sec]')

      if file_path is not None:
        plt.savefig(file_path, bbox_inches='tight')
        print('save figure:', file_path)
      plt.show()

    return spike_times


  @classmethod
  def inject_synchrony_to_spike_times(
      cls,
      spike_times_x,
      spike_times_y,
      synchrony_rate,
      trial_length):
    """Inject synchrony spikes into spike times x, y."""
    if synchrony_rate == 0:
      return spike_times_x, spike_times_y

    elif (isinstance(spike_times_x[0], list) or
        isinstance(spike_times_x[0], np.ndarray)):
      if len(spike_times_x) != len(spike_times_y):
        raise ValueError('spike_times_x, spike_times_y shape not match.')
      num_trials = len(spike_times_x)
      spike_times_sync_x = [np.empty(0)] * num_trials
      spike_times_sync_y = [np.empty(0)] * num_trials

      for r in range(num_trials):
        sync_spike_times = cls.generate_poisson_spike_times(
            synchrony_rate, trial_length)
        sync_spike_times = np.array(sync_spike_times)
        spike_times_tmp = np.concatenate([spike_times_x[r], sync_spike_times])
        spike_times_sync_x[r] = np.sort(spike_times_tmp)
        spike_times_tmp = np.concatenate([spike_times_y[r], sync_spike_times])
        spike_times_sync_y[r] = np.sort(spike_times_tmp)

    else:
      sync_spike_times = cls.generate_poisson_spike_times(
          synchrony_rate, trial_length)
      spike_times_sync_x = np.concatenate([spike_times_x, sync_spike_times])
      spike_times_sync_x = np.sort(spike_times_sync_x)
      spike_times_sync_y = np.concatenate([spike_times_y, sync_spike_times])
      spike_times_sync_y = np.sort(spike_times_sync_y)

    return spike_times_sync_x, spike_times_sync_y


  @classmethod
  def inject_prob_synchrony_to_spike_times(
      cls,
      spike_times_from,
      spike_times_to,
      synchrony_prob,
      trial_length):
    """Inject synchrony spikes into spike times x, y.

    It is based on lambda_Y = lambda + delta conv* x
    a spike in x trigers higher probability with additional delta.
    Based on Poisson distribution property that X ~ Poiss(lambda_x),
    Y ~ Poiss(lambda_y), then Z = X + Y ~ Poiss(lambda_x + lambda_y).
    """
    if synchrony_prob == 0:
      return spike_times_from, spike_times_to

    # Multiple trials.
    elif (isinstance(spike_times_from[0], list) or
        isinstance(spike_times_from[0], np.ndarray)):
      if len(spike_times_from) != len(spike_times_to):
        raise ValueError('spike_times_from, spike_times_to shape not match.')
      num_trials = len(spike_times_from)
      spike_times_to_sync = [np.empty(0)] * num_trials

      for r in range(num_trials):
        spike_times_from[r] = np.array(spike_times_from[r])
        sync_spike_times = []
        for t in spike_times_from[r]:
          if np.random.rand() < synchrony_prob:
            sync_spike_times.append(t)
        spike_times_tmp = np.concatenate([spike_times_to[r], sync_spike_times])
        spike_times_to_sync[r] = np.sort(spike_times_tmp)

    # Single trial.
    else:
      spike_times_from = np.array(spike_times_from)
      sync_spike_times = []
      for t in spike_times_from:
        if np.random.rand() < synchrony_prob:
          sync_spike_times.append(t)
      spike_times_tmp = np.concatenate([spike_times_to, sync_spike_times])
      spike_times_to_sync = np.sort(spike_times_tmp)

    return spike_times_from, spike_times_to_sync


  @classmethod
  def generate_poisson_spike_times_multi_sync(
      cls,
      lmbd,
      trial_length,
      num_nodes,
      num_trials,
      sync_prob_mat=None,
      sync_rate_mat=None,
      verbose=False):
    """Generate multivariate Poisson processes with synchrony."""
    spike_times = [[] for _ in range(num_nodes)]
    for n in range(num_nodes):
      if np.isscalar(lmbd):
        lmbd_node = lmbd
      elif isinstance(lmbd, list) or isinstance(lmbd, np.ndarray):
        lmbd_node = lmbd[n]
      spike_times[n] = cls.generate_poisson_spike_times(
          lmbd_node, trial_length, num_trials)

    # Inject synchrony.
    if sync_prob_mat is not None:
      spike_times_sync = [spike_times[n] for n in range(num_nodes)]
      for from_node in range(num_nodes):
        for to_node in range(num_nodes):
          if to_node == from_node:
            continue
          synchrony_prob = sync_prob_mat[from_node, to_node]
          _, spike_times_sync[to_node] = cls.inject_prob_synchrony_to_spike_times(
              spike_times[from_node], spike_times_sync[to_node], synchrony_prob,
              trial_length)

    # Inject pairwise synchrony using sync sequences.
    if sync_rate_mat is not None:
      spike_times_sync = spike_times.copy()
      for from_node in range(num_nodes):
        for to_node in range(num_nodes):
          if from_node == to_node:
            continue
          synchrony_rate = sync_rate_mat[from_node, to_node]
          (spike_times_sync[from_node], spike_times_sync[to_node]
              ) = cls.inject_synchrony_to_spike_times(
              spike_times_sync[from_node], spike_times_sync[to_node],
              synchrony_rate, trial_length)

    if verbose:
      n = 2
      jittertool = jitter.JitterTool()
      gs_kw = dict(width_ratios=[1], height_ratios=[1,1])
      fig, axs = plt.subplots(figsize=(8, 4), gridspec_kw=gs_kw,
        nrows=2, ncols=1)
      plt.subplots_adjust(hspace=0, wspace=0)
      ax = fig.add_subplot(axs[0])
      ax.tick_params(labelbottom=False, labelleft=False)
      if num_trials == 1:
        plt.plot(spike_times_sync[n], np.zeros(len(spike_times_sync[n])),
                 'ks', ms=2)
      else:
        for r in range(num_trials):
          plt.plot(spike_times_sync[n][r],
                   r+np.zeros(len(spike_times_sync[n][r])), 'ks', ms=0.4)
      plt.xlim(0, trial_length)

      ax = fig.add_subplot(axs[1])
      # ax.tick_params(labelleft=False)
      spike_hist, bins = jittertool.bin_spike_times(
          spike_times_sync[n], 0.005, trial_length)
      plt.plot(bins, spike_hist.mean(axis=0) / 0.005, 'k')
      plt.xlim(0, trial_length)

      plt.show()

    return spike_times_sync


  @classmethod
  def _laplacian_intensity_func(
      cls,
      t,
      peaks,
      sigma,
      baseline=10):
    """Intensity function with mixture of Laplacian's.

    The scalue is sigma/sqrt(2) by following the Amarasingham 2012 paper
    appendix.
    """
    t = np.array(t)
    if np.ndim(t) == 0:
      sample_points = scipy.stats.laplace.pdf(t - peaks,
          loc=0, scale=sigma/np.sqrt(2))
      intensity = sample_points.sum() + baseline
    else:
      num_t = len(t)
      intensity = np.zeros(num_t)
      for i in range(num_t):
        sample_points = scipy.stats.laplace.pdf(t[i] - peaks,
            loc=0, scale=sigma/np.sqrt(2))
        intensity[i] = sample_points.sum() + baseline
    return intensity


  @classmethod
  def norm_pdf(
      cls,
      x,
      loc,
      scale):
    """About twice faster than scipy.stats.norm.pdf."""
    pdfs = np.square(x-loc) / 2 / np.square(scale)
    pdfs = np.exp(-pdfs) / scale / np.sqrt(2*np.pi)
    return pdfs


  @classmethod
  def _gaussian_intensity_func(
      cls,
      t,
      peaks,
      sigma,
      baseline=10):
    """Intensity function with mixture of Gaussian."""
    t = np.array(t)
    if np.ndim(t) == 0:
      # sample_points = scipy.stats.norm.pdf(t - peaks, loc=0, scale=sigma)
      sample_points = cls.norm_pdf(t - peaks, loc=0, scale=sigma)
      intensity = sample_points.sum() + baseline
    else:
      num_t = len(t)
      intensity = np.zeros(num_t)
      for i in range(num_t):
        # sample_points = scipy.stats.norm.pdf(t[i] - peaks, loc=0, scale=sigma)
        sample_points = cls.norm_pdf(t[i] - peaks, loc=0, scale=sigma)
        intensity[i] = sample_points.sum() + baseline
    return intensity


  @classmethod
  def norm_pdf_sigmas(
      cls,
      x,
      loc,
      sigmas):
    """Evaluate pdfs with different sigmas."""
    if len(x) != len(sigmas):
      raise ValueError('len x != len sigmas')
    pdfs = np.square(x-loc) / 2 / np.square(sigmas)
    pdfs = np.exp(-pdfs) / sigmas / np.sqrt(2*np.pi)
    return pdfs


  @classmethod
  def _gaussian_varying_intensity_func(
      cls,
      t,
      peaks,
      sigmas,
      baseline=10):
    """Intensity function with mixture of scale varying Gaussian.

    Example usage:
    t = np.linspace(0, 5, 1000)
    peaks = [1,2,3,4]
    sigmas = [0.1, 0.05, 0.2, 0.08]
    y = generator._gaussian_varying_intensity_func(t, peaks, sigmas)
    plt.plot(t, y)
    """
    t = np.array(t)
    if np.ndim(t) == 0:
      sample_points = cls.norm_pdf_sigmas(t-peaks, loc=0, sigmas=sigmas)
      intensity = sample_points.sum() + baseline
    else:
      num_t = len(t)
      intensity = np.zeros(num_t)
      for i in range(num_t):
        sample_points = cls.norm_pdf_sigmas(t[i]-peaks, loc=0, sigmas=sigmas)
        intensity[i] = sample_points.sum() + baseline
    return intensity


  @classmethod
  def generate_amarasingham_spike_times_multi_sync(
      cls,
      num_peaks=40,
      sigma=0.04,
      trial_length=1,
      num_nodes=None,
      num_trials=1,
      num_drivers=3,
      sync_prob_mat=None,
      sync_rate_mat=None,
      sample_type='density',
      verbose=False):
    """Intensity with mixture of Laplacian windows.

    Args:
      num_trials: number of drivers that drive the share intensities.
    """
    baseline = 10
    intensity_integral = baseline + num_peaks
    peaks_all = np.zeros([num_drivers, num_peaks])

    # Create drivers.
    for d in range(num_drivers):
      peaks = np.random.rand(num_peaks) * trial_length
      peaks = np.sort(peaks)
      peaks_all[d] = peaks
      if verbose:
        x = np.arange(0, trial_length, 0.01)
        y = cls._laplacian_intensity_func(x, peaks_all[d], sigma, baseline)
        plt.figure(figsize=[8,2])
        plt.plot(x, y, 'k')
        plt.xlim(0, trial_length)
        plt.show()

    # Create clean spike times.
    spike_times = [[] for _ in range(num_nodes)]
    for n in range(num_nodes):
      # Make intensity functions.
      alpha = [1] * num_drivers
      drivers_weight = scipy.stats.dirichlet.rvs(alpha)[0]
      def node_intensity_func(t):
        intensity = 0
        for d in range(num_drivers):
          intensity += drivers_weight[d] * cls._laplacian_intensity_func(
              t, peaks_all[d], sigma, baseline)
        return intensity
      intensity_peaks = node_intensity_func(peaks_all.reshape(-1))
      intensity_max_x = peaks_all.reshape(-1)
      intensity_max_x = intensity_max_x[np.argmax(intensity_peaks)]
      intensity_max = intensity_peaks.max()

      # Generate spike trains.
      trial_spike_times = [0] * num_trials
      for r in range(num_trials):
        spike_times_single = cls.generate_inhomo_poisson_density_spike_times(
          intensity_func=node_intensity_func, length=trial_length,
          intensity_integral=intensity_integral, intensity_max=intensity_max,
          verbose=False)
        trial_spike_times[r] = spike_times_single
      if num_trials == 1:
        trial_spike_times = trial_spike_times[0]
      spike_times[n] = trial_spike_times
      if verbose:
        jittertool = jitter.JitterTool()
        x = np.arange(0, trial_length, 0.001)
        x = np.concatenate((x, peaks_all.reshape(-1)))
        x = np.sort(x)
        y = node_intensity_func(x)
        gs_kw = dict(width_ratios=[1], height_ratios=[1,1,1])
        fig, axs = plt.subplots(figsize=(8, 6), gridspec_kw=gs_kw,
          nrows=3, ncols=1)
        plt.subplots_adjust(hspace=0, wspace=0)
        ax = fig.add_subplot(axs[0])
        ax.tick_params(labelbottom=False)
        for peak in peaks_all.reshape(-1):
          plt.axvline(x=peak, c='grey', lw=0.3)
        plt.plot(intensity_max_x, intensity_max, 'r+', ms=12)
        plt.plot(x, y, 'k')
        plt.xlim(0, trial_length)
        plt.ylim(0, intensity_max+20)
        plt.title(f'{drivers_weight}')

        ax = fig.add_subplot(axs[1])
        ax.tick_params(labelbottom=False, labelleft=False)
        if num_trials == 1:
          plt.plot(spike_times[n], np.zeros(len(spike_times[n])), 'ks', ms=2)
        else:
          for r in range(num_trials):
            plt.plot(spike_times[n][r], r+np.zeros(len(spike_times[n][r])), 'ks', ms=0.4)
        plt.xlim(0, trial_length)

        ax = fig.add_subplot(axs[2])
        # ax.tick_params(labelleft=False)
        spike_hist, bins = jittertool.bin_spike_times(
            spike_times[n], 0.005, trial_length)
        plt.plot(bins, spike_hist.mean(axis=0) / 0.005, 'k')
        plt.xlim(0, trial_length)

        plt.show()

    # Inject synchrony using probability.
    if sync_prob_mat is not None:
      spike_times_sync = [spike_times[n] for n in range(num_nodes)]
      for from_node in range(num_nodes):
        for to_node in range(num_nodes):
          if to_node == from_node:
            continue
          synchrony_prob = sync_prob_mat[from_node, to_node]
          _, spike_times_sync[to_node] = cls.inject_prob_synchrony_to_spike_times(
              spike_times[from_node], spike_times_sync[to_node], synchrony_prob,
              trial_length)

    # Inject pairwise synchrony using sync sequences.
    if sync_rate_mat is not None:
      spike_times_sync = spike_times.copy()
      for from_node in range(num_nodes):
        for to_node in range(num_nodes):
          if from_node == to_node:
            continue
          synchrony_rate = sync_rate_mat[from_node, to_node]
          (spike_times_sync[from_node], spike_times_sync[to_node]
              ) = cls.inject_synchrony_to_spike_times(
              spike_times_sync[from_node], spike_times_sync[to_node],
              synchrony_rate, trial_length)

    return spike_times_sync


  @classmethod
  def spike_times_statistics(
      cls,
      spike_times,
      trial_length,
      hist_bin_width=0.005,
      verbose=0):
    """Check basic spike times statistics."""
    jittertool = jitter.JitterTool()

    def list_depth(x):
      for item in x:
        if not isinstance(item, list) and not isinstance(item, np.ndarray):
          return 1
        else:
          return list_depth(item)+1

    num_layers = list_depth(spike_times)
    if num_layers == 1:
      num_nodes = 1
      num_trials = 1
    if num_layers == 2:
      num_trials = len(spike_times)
      num_nodes = 1
    elif num_layers == 3:
      num_nodes = len(spike_times)
      num_trials = len(spike_times[0])
    print(f'layers {num_layers}, nodes {num_nodes}, trials {num_trials}')
    
    if num_layers == 2:
      # Spike counts.
      spike_cnts = np.zeros(num_trials)
      for r in range(num_trials):
        spikes = np.array(spike_times[r])
        spike_cnts[r] = len(spikes)
      mean = np.mean(spike_cnts)
      var = np.std(spike_cnts) ** 2
      mean_fr = spike_cnts.sum() / trial_length / num_trials

      # Inter-spike intervals.
      isi = []
      for r in range(num_trials):
        isi_trial = list(np.diff(np.array(spike_times[r])))
        isi.extend(isi_trial)
      scale_hat = np.mean(isi)
      spike_hist, bins = jittertool.bin_spike_times(
          spike_times, hist_bin_width, trial_length)

      print(f'meanFR {np.round(mean_fr,3)}\tmeanISI {np.round(1/ scale_hat,3)}')
      if verbose < 1:
        return

      gs_kw = dict(width_ratios=[1,1.5], height_ratios=[1])
      fig, axs = plt.subplots(figsize=(6, 2),
          gridspec_kw=gs_kw, nrows=1, ncols=2)
      x = np.arange(0, 6*np.max(scale_hat), np.min(scale_hat)/10)
      ax = fig.add_subplot(axs[0])
      y_exp = scipy.stats.expon.pdf(x, scale=scale_hat)
      plt.plot(x, y_exp, 'k', lw=0.7)
      seaborn.distplot(isi, bins=50, kde=False, norm_hist=True, color='grey')
      plt.xlim(-0.05*x[-1], x[-1])

      ax = fig.add_subplot(axs[1])
      plt.plot(bins, spike_hist.mean(axis=0) / hist_bin_width, c='k')
      plt.show()

    if num_layers == 3:
      # Spike counts.
      spike_cnts = np.zeros([num_nodes, num_trials])
      for n in range(num_nodes):
        for r in range(num_trials):
          spikes = np.array(spike_times[n][r])
          spike_cnts[n,r] = len(spikes)
      # mean = np.mean(spike_cnts, axis=1)
      # var = np.std(spike_cnts, axis=1) ** 2
      mean_fr = spike_cnts.sum(axis=1) / trial_length / num_trials

      # Inter-spike intervals.
      isi = [[] for _ in range(num_nodes)]
      scale_hat = np.zeros(num_nodes)
      for n in range(num_nodes):
        for r in range(num_trials):
          isi_trial = list(np.diff(np.array(spike_times[n][r])))
          isi[n].extend(isi_trial)
        scale_hat[n] = np.mean(isi[n])
      print(f'meanFR {np.round(mean_fr,3)}\tmeanISI {np.round(1/ scale_hat,3)}')

      if verbose < 1:
        return
      num_cols = min(8, num_nodes)
      num_rows = np.ceil(num_nodes / num_cols).astype(int)
      gs_kw = dict(width_ratios=[1]*num_cols, height_ratios=[1]*num_rows)
      fig, axs = plt.subplots(figsize=(3*num_cols, 2*num_rows),
          gridspec_kw=gs_kw, nrows=num_rows, ncols=num_cols)
      plt.subplots_adjust(hspace=0.1, wspace=0.2)
      axs = axs.reshape(-1) if num_cols > 1 else [axs]
      x = np.arange(0, 6*np.max(scale_hat), np.min(scale_hat)/10)
      for n in range(num_nodes):
        ax = fig.add_subplot(axs[n])
        y_exp = scipy.stats.expon.pdf(x, scale=scale_hat[n])
        plt.plot(x, y_exp, 'r')
        seaborn.distplot(isi[n], bins=100, kde=False, norm_hist=True, color='grey')
        plt.xlim(-0.05*x[-1], x[-1])
      plt.show()


  @classmethod
  def plot_psth(
      cls,
      spike_times,
      trial_length,
      hist_bin_width=0.1,
      ylim=None):
    """Plot PSTH of all neurons averaged over all trials."""
    def list_depth(x):
      for item in x:
        if not isinstance(item, list) and not isinstance(item, np.ndarray):
          return 1
        else:
          return list_depth(item)+1

    num_layers = list_depth(spike_times)
    if num_layers == 1:
      num_nodes = 1
      num_trials = 1
    if num_layers == 2:
      num_trials = len(spike_times)
      num_nodes = 1
    elif num_layers == 3:
      num_nodes = len(spike_times)
      num_trials = len(spike_times[0])
    print(f'layers {num_layers}, nodes {num_nodes}, trials {num_trials}')
    if num_layers < 3:
      print('Skip for simple case with 1,2 layers.')
      return

    jittertool = jitter.JitterTool()
    num_cols = min(3, num_nodes)
    num_rows = np.ceil(num_nodes / num_cols).astype(int)
    gs_kw = dict(width_ratios=[1]*num_cols, height_ratios=[1]*num_rows)
    fig, axs = plt.subplots(figsize=(6*num_cols, 2*num_rows),
        gridspec_kw=gs_kw, nrows=num_rows, ncols=num_cols)
    plt.subplots_adjust(hspace=0.1, wspace=0.15)
    axs = axs.reshape(-1) if num_cols > 1 else [axs]
    for n in range(num_nodes):
      ax = fig.add_subplot(axs[n])
      if n == num_cols * (num_rows-1):
        ax.tick_params(left=True, labelleft=True, labelbottom=True)
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Firing rate [spk/sec]')
      elif ylim:
        ax.tick_params(left=True, labelleft=False, labelbottom=False)
      spike_hist, bins = jittertool.bin_spike_times(
          spike_times[n], hist_bin_width, trial_length)
      plt.plot(bins, spike_hist.mean(axis=0) / hist_bin_width, c='lightgrey',
               label='PSTH')
      plt.ylim(ylim)
      # index_range = bins > (trial_length * 0.6)
      # mean_fr = np.mean(spike_hist[:,index_range]) / hist_bin_width
      # plt.axhline(y=mean_fr, ls='--', color='r', label='simulation mean')
      # print('Mean FR last half:', np.round(mean_fr, 3))
    plt.show()

    return axs


  @classmethod
  def delayed_square_couping_effects(
      cls,
      s,
      mu,
      alpha,
      beta,
      delay,
      spike_times,
      verbose=False):
    """Sum the coupling effects from source --> target."""
    mu = np.array(mu)
    alpha = np.array(alpha)
    beta = np.array(beta)
    num_nodes = len(mu)

    lambda_nt = mu.copy()
    for target in range(num_nodes):
      for source in range(num_nodes):
        spikes = np.array(spike_times[source])
        spikes = spikes[((s - delay - beta[target,source]) <= spikes) &
                        (spikes < s - delay)]
        if len(spikes) == 0:
          continue
        delays = s - spikes
        coupling_vals = alpha[target,source] * len(delays)
        lambda_nt[target] += coupling_vals.sum()

    return lambda_nt


  @classmethod
  def square_couping_effects(
      cls,
      s,
      mu,
      alpha,
      beta,
      spike_times,
      verbose=False):
    """Sum the coupling effects from source --> target."""
    mu = np.array(mu)
    alpha = np.array(alpha)
    beta = np.array(beta)
    num_nodes = len(mu)

    lambda_nt = mu.copy()
    for target in range(num_nodes):
      for source in range(num_nodes):
        spikes = np.array(spike_times[source])
        spikes = spikes[((s - beta[target,source]) <= spikes) & (spikes < s)]
        if len(spikes) == 0:
          continue
        delays = s - spikes
        coupling_vals = alpha[target,source] * len(delays)
        lambda_nt[target] += coupling_vals.sum()

    return lambda_nt


  @classmethod
  def triangle_couping_effects(
      cls,
      s,
      mu,
      alpha,
      beta,
      spike_times,
      verbose=False):
    """Sum the coupling effects from source --> target."""
    mu = np.array(mu)
    alpha = np.array(alpha)
    beta = np.array(beta)
    num_nodes = len(mu)

    lambda_nt = mu.copy()
    for target in range(num_nodes):
      for source in range(num_nodes):
        spikes = np.array(spike_times[source])
        spikes = spikes[((s - beta[target,source]) <= spikes) & (spikes < s)]
        if len(spikes) == 0:
          continue
        delays = s - spikes
        coupling_vals = (alpha[target,source] / beta[target,source] * 
                         (beta[target,source] - delays))
        lambda_nt[target] += coupling_vals.sum()

    return lambda_nt


  @classmethod
  def inv_triangle_couping_effects(
      cls,
      s,
      mu,
      alpha,
      beta,
      spike_times,
      verbose=False):
    """Sum the coupling effects from source --> target."""
    mu = np.array(mu)
    alpha = np.array(alpha)
    beta = np.array(beta)
    num_nodes = len(mu)

    lambda_nt = mu.copy()
    for target in range(num_nodes):
      for source in range(num_nodes):
        spikes = np.array(spike_times[source])
        spikes = spikes[((s - beta[target,source]) <= spikes) & (spikes < s)]
        if len(spikes) == 0:
          continue
        delays = s - spikes
        coupling_vals = alpha[target,source] / beta[target,source] * delays
        lambda_nt[target] += coupling_vals.sum()

    return lambda_nt


  @classmethod
  def exp_couping_effects(
      cls,
      s,
      mu,
      alpha,
      beta,
      spike_times,
      verbose=False):
    """Sum the coupling effects from source --> target."""
    mu = np.array(mu)
    alpha = np.array(alpha)
    beta = np.array(beta)
    num_nodes = len(mu)

    lambda_nt = mu.copy()
    for target in range(num_nodes):
      for source in range(num_nodes):
        spikes = np.array(spike_times[source])
        # spikes = spikes[(spikes < s)]
        spikes = spikes[((s - 50/beta[target,source]) < spikes) & (spikes < s)]
        if len(spikes) == 0:
          continue
        delays = s - spikes
        coupling_vals = alpha[target,source] * np.exp(-beta[target,source] * delays)
        lambda_nt[target] += coupling_vals.sum()

    return lambda_nt


  @classmethod
  def generate_hawkes_spike_times_single(
      cls,
      filter_par,
      trial_length,
      verbose=False):
    """Replication of algorithm in Chen 2016.

    Args:
      filter_par: 
    """
    mu = np.array(filter_par['mu']).astype('double')
    num_nodes = len(mu)
    gamma = np.array(filter_par['gamma'])
    lambda_stable = np.linalg.inv(np.eye(num_nodes) - gamma) @ mu.reshape(-1,1)

    # Check stability.
    eig_val = np.linalg.eigvals(gamma)
    eig_val_max = np.max(np.abs(eig_val))
    if eig_val_max > 1:
      warnings.warn(f'System unstable. Gamma eigval:{eig}')

    lambda_max = np.sum(lambda_stable) * 2
    if verbose:
      print('lambda stable:', lambda_stable.reshape(-1))
      print('lambda_max:', lambda_max)

    spike_times = [[] for _ in range(num_nodes)]
    s = 0
    err_cnt = 0
    while s < trial_length:
      # u = -np.log(np.random.rand()) /lambda_max
      u = np.random.exponential(scale=1/lambda_max)
      s = s + u
      if s > trial_length:
        break
      if filter_par['type'] == 'none':
        lambda_vals = mu.copy()
      elif filter_par['type'] == 'exp':
        lambda_vals = cls.exp_couping_effects(
            s, mu.copy(), filter_par['alpha'], filter_par['beta'], spike_times)
      elif filter_par['type'] == 'triangle':
        lambda_vals = cls.triangle_couping_effects(
            s, mu.copy(), filter_par['alpha'], filter_par['beta'], spike_times)
      elif filter_par['type'] == 'inv_triangle':
        lambda_vals = cls.inv_triangle_couping_effects(
            s, mu.copy(), filter_par['alpha'], filter_par['beta'], spike_times)
      elif filter_par['type'] == 'square':
        lambda_vals = cls.square_couping_effects(
            s, mu.copy(), filter_par['alpha'], filter_par['beta'], spike_times)
      elif filter_par['type'] == 'delayed_square':
        lambda_vals = cls.delayed_square_couping_effects(
            s, mu.copy(), filter_par['alpha'], filter_par['beta'],
            filter_par['delay'], spike_times)

      lambda_vals = np.insert(lambda_vals, 0, 0)
      lambda_cum_sum = np.cumsum(lambda_vals)
      lambda_cum_sum_ratio = lambda_cum_sum / lambda_max
      if lambda_cum_sum_ratio[-1] > 1:
        err_cnt += 1
        if err_cnt % 10 == 9:
          print(f'Warning: lambda > lambda_max. err cnt{err_cnt}')
        lambda_max = lambda_max * 2
        s = s - u
        continue
      d = np.random.rand()
      ind = np.where(lambda_cum_sum_ratio <= d)[0][-1]
      if ind != num_nodes:
        spike_times[ind].append(s)

    return spike_times


  @classmethod
  def generate_hawkes_spike_times_synchrony_single(
      cls,
      filter_par,
      trial_length,
      verbose=False):
    """Replication of algorithm in Chen 2016.

    Args:
      filter_par: 
    """
    mu = np.array(filter_par['mu']).astype('double')
    num_nodes = len(mu)
    alpha = np.array(filter_par['alpha']).astype('double')
    gamma = np.array(filter_par['gamma'])
    lambda_stable = np.linalg.inv(np.eye(num_nodes) - gamma) @ mu.reshape(-1,1)

    # Check stability.
    eig_val = np.linalg.eigvals(gamma)
    eig_val_max = np.max(np.abs(eig_val))
    if eig_val_max > 1:
      warnings.warn(f'System unstable. Gamma eigval:{eig_val}')
    lambda_max = np.sum(lambda_stable) * 2
    if lambda_max < 0:
      lambda_max = mu.sum() * 2
    if verbose:
      print('lambda stable:', lambda_stable.reshape(-1))
      print('lambda_max:', lambda_max)

    spike_times = [[] for _ in range(num_nodes)]
    s = 0
    err_cnt = 0
    while s < trial_length:

      u = np.random.exponential(scale=1/lambda_max)
      s = s + u
      if s > trial_length:
        break

      lambda_vals = mu.copy()
      lambda_vals = np.insert(lambda_vals, 0, 0)
      lambda_cum_sum = np.cumsum(lambda_vals)
      lambda_cum_sum_ratio = lambda_cum_sum / lambda_max
      if lambda_cum_sum_ratio[-1] > 1:
        err_cnt += 1
        if err_cnt % 10 == 9:
          print(f'Warning: lambda > lambda_max. err cnt{err_cnt}')
        lambda_max = lambda_max * 2
        s = s - u
        continue
      d = np.random.rand()
      ind = np.where(lambda_cum_sum_ratio <= d)[0][-1]
      if ind != num_nodes:
        spike_times[ind].append(s)
      else:
        continue

      # Synchrony echos. Continued with currently generated spike `s`.
      active_nodes_curr = [ind]
      active_nodes_next = []
      echo_cnt = 0

      while len(active_nodes_curr) > 0:
        for source in active_nodes_curr:
          targets = np.where(alpha[source] != 0)[0]
          if len(targets) == 0:  # No synchrony target.
            continue
          for target in targets:
            sync_cnt = np.random.poisson(alpha[target, source])
            if sync_cnt > 0:
              spike_times[target].extend([s]*sync_cnt)
              active_nodes_next.extend([target]*sync_cnt)

        active_nodes_curr = active_nodes_next.copy()
        active_nodes_next = []
        echo_cnt += 1

      # if echo_cnt > 2:
      #   print(echo_cnt)
    return spike_times


  @classmethod
  def generate_hawkes_spike_times(
      cls,
      model_par,
      verbose=False):
    """Replication of algorithm in Chen 2016.

    Args:
    """
    np.random.seed(model_par['random_seed'])
    trial_length = model_par['trial_length']
    trial_window = model_par['trial_window']
    num_trials = model_par['num_trials']
    mu = model_par['mu']
    mu = np.array(mu).astype('double')
    num_nodes = len(mu)

    if 'alpha' in model_par:
      alpha = np.array(model_par['alpha']).astype('double')
    if 'beta' in model_par:
      beta = np.array(model_par['beta']).astype('double')

    if model_par['type'] == 'exp':
      gamma = alpha / beta
    elif model_par['type'] == 'triangle':
      gamma = alpha * beta / 2
    elif model_par['type'] == 'inv_triangle':
      gamma = alpha * beta / 2
    elif model_par['type'] == 'square':
      gamma = alpha * beta
    elif model_par['type'] == 'delayed_square':
      gamma = alpha * beta
    elif model_par['type'] == 'synchrony':
      gamma = alpha
    elif model_par['type'] == 'none':
      gamma = np.array([[0]])

    model_par['gamma'] = gamma
    lambda_stable = np.linalg.inv(np.eye(num_nodes) - gamma) @ mu.reshape(-1,1)
    # Check stability.
    eig = np.linalg.eigvals(gamma)
    if any(np.abs(eig) > 1):
      warnings.warn(f'System unstable. Gamma eigval:{eig}')
    if verbose:
      print('lambda stable:', lambda_stable.reshape(-1))
      print('Gamma')
      print(gamma)

    spike_times = [[] for _ in range(num_nodes)]
    if verbose==2:
      trange = tqdm(range(num_trials), ncols=100, file=sys.stdout)
    else:
      trange = range(num_trials)
    for r in trange:
      if model_par['type'] == 'synchrony':
        spikes = cls.generate_hawkes_spike_times_synchrony_single(
            model_par, trial_length, False)
      else:
        spikes = cls.generate_hawkes_spike_times_single(
            model_par, trial_length, False)
      for n in range(num_nodes):
        spike_times[n].append(np.array(spikes[n]))

    return spike_times


  @classmethod
  def plot_hawkes_intensity(
      cls,
      filter_par,
      spike_times,
      trial_length,
      node_ids=[0],
      trial_ids=[0],
      plot_window=[None, None],
      ylim=None):
    """Plot trial intensity function."""
    mu = filter_par['mu']
    mu = np.array(mu).astype('double')
    num_nodes = len(mu)
    if filter_par['type'] == 'exp':
      alpha = np.array(filter_par['alpha'])
      beta = np.array(filter_par['beta'])
    elif filter_par['type'] == 'triangle':
      alpha = np.array(filter_par['alpha'])
      beta = np.array(filter_par['beta'])
    elif filter_par['type'] == 'inv_triangle':
      alpha = np.array(filter_par['alpha'])
      beta = np.array(filter_par['beta'])
    elif filter_par['type'] == 'square':
      alpha = np.array(filter_par['alpha'])
      beta = np.array(filter_par['beta'])
    elif filter_par['type'] == 'delayed_square':
      alpha = np.array(filter_par['alpha'])
      beta = np.array(filter_par['beta'])
      delay = np.array(filter_par['delay'])

    trial_id = trial_ids[0]
    trial_spike_times = [spike_times[n][trial_id] for n in range(num_nodes)]
    time_line = np.arange(0, trial_length, 0.005)
    intensity = np.zeros([num_nodes, len(time_line)])
    for ind, s in enumerate(time_line):
      if filter_par['type'] == 'none':
        intensity[:,ind] = mu.copy()
      elif filter_par['type'] == 'exp':
        intensity[:,ind] = cls.exp_couping_effects(
            s, mu.copy(), alpha, beta, trial_spike_times)
      elif filter_par['type'] == 'triangle':
        intensity[:,ind] = cls.triangle_couping_effects(
            s, mu.copy(), alpha, beta, trial_spike_times)
      elif filter_par['type'] == 'inv_triangle':
        intensity[:,ind] = cls.inv_triangle_couping_effects(
            s, mu.copy(), alpha, beta, trial_spike_times)
      elif filter_par['type'] == 'square':
        intensity[:,ind] = cls.square_couping_effects(
            s, mu.copy(), alpha, beta, trial_spike_times)
      elif filter_par['type'] == 'delayed_square':
        intensity[:,ind] = cls.delayed_square_couping_effects(
            s, mu.copy(), alpha, beta, delay, trial_spike_times)

    colors = ['k', 'r', 'g', 'b']
    for node_id in node_ids:
      plt.figure(figsize=[20,2.0])
      plt.plot(time_line, intensity[node_id])
      for n in range(num_nodes):
        plt.plot(trial_spike_times[n],
                 np.zeros(len(trial_spike_times[n])) - 0.2*n, '+', c=colors[n%4])
      plt.xlim(plot_window)
      plt.ylim(ylim)
      plt.show()


  @classmethod
  def plot_hawkes_psth(
      cls,
      spike_times,
      trial_length,
      hist_bin_width=0.1,
      filter_par=None):
    """Add theoretical value to the PSTH."""
    mu = filter_par['mu']
    mu = np.array(mu).astype('double')
    num_nodes = len(mu)

    if filter_par['type'] == 'exp':
      alpha = np.array(filter_par['alpha']).astype('double')
      beta = np.array(filter_par['beta']).astype('double')
      gamma = alpha / beta
    elif filter_par['type'] == 'triangle':
      alpha = np.array(filter_par['alpha']).astype('double')
      beta = np.array(filter_par['beta']).astype('double')
      gamma = alpha * beta / 2
    elif filter_par['type'] == 'inv_triangle':
      alpha = np.array(filter_par['alpha']).astype('double')
      beta = np.array(filter_par['beta']).astype('double')
      gamma = alpha * beta / 2
    elif filter_par['type'] == 'square':
      alpha = np.array(filter_par['alpha']).astype('double')
      beta = np.array(filter_par['beta']).astype('double')
      gamma = alpha * beta
    elif filter_par['type'] == 'delayed_square':
      alpha = np.array(filter_par['alpha']).astype('double')
      beta = np.array(filter_par['beta']).astype('double')
      gamma = alpha * beta
    elif filter_par['type'] == 'synchrony':
      alpha = np.array(filter_par['alpha']).astype('double')
      gamma = alpha
    elif filter_par['type'] == 'none':
      gamma = np.array([[0]])

    filter_par['gamma'] = gamma
    lambda_stable = np.linalg.inv(np.eye(num_nodes) - gamma) @ mu.reshape(-1,1)

    ylim = [0, lambda_stable.max()*1.4]
    axs = cls.plot_psth(spike_times, trial_length, hist_bin_width, ylim)
    for n in range(num_nodes):
      axs[n].axhline(y=lambda_stable[n], color='b', label='theoretical mean')
      if n == 0:
        axs[n].legend(loc=(0,1.1), ncol=3)
    plt.show()


  @classmethod
  def generate_inhomogeneous_hawkes_spike_times_single(
      cls,
      filter_par,
      intensity_func,
      trial_length,
      verbose=False):
    """Replication of algorithm in Chen 2016.

    Args:
      filter_par: 
    """
    num_nodes = filter_par['num_nodes']
    mu = filter_par['intensity_max']
    mu = np.zeros([num_nodes, 1]) + mu
    alpha = np.array(filter_par['alpha']).astype('double')
    beta = np.array(filter_par['beta']).astype('double')
    gamma = filter_par['gamma']
    eig_val = np.linalg.eigvals(gamma)
    eig_val_max = np.max(np.abs(eig_val))
    if eig_val_max > 1:
      warnings.warn(f'System unstable. Gamma eigval:{eig}')

    lambda_stable = np.linalg.inv(np.eye(num_nodes) - gamma) @ mu
    lambda_max = np.sum(lambda_stable) * 0.5
    if verbose:
      print('lambda stable:', lambda_stable.reshape(-1))
      print('lambda_max:', lambda_max)

    spike_times = [[] for _ in range(num_nodes)]
    s = 0
    accept_cnt = 0
    sample_cnt = 0
    err_cnt = 0
    while s < trial_length:
      # u = -np.log(np.random.rand()) /lambda_max
      u = np.random.exponential(scale=1/lambda_max)
      s = s + u
      sample_cnt += 1
      if s > trial_length:
        break
      # baseline = mu.copy()
      # baseline = 20 + np.zeros([num_nodes, 1])
      baseline = intensity_func(s) + np.zeros([num_nodes, 1])
      if filter_par['type'] == 'none':
        lambda_vals = baseline
      elif filter_par['type'] == 'exp':
        lambda_vals = cls.exp_couping_effects(
            s, baseline, alpha, beta, spike_times)
      elif filter_par['type'] == 'triangle':
        lambda_vals = cls.triangle_couping_effects(
            s, baseline, alpha, beta, spike_times)
      elif filter_par['type'] == 'inv_triangle':
        lambda_vals = cls.inv_triangle_couping_effects(
            s, baseline, alpha, beta, spike_times)
      elif filter_par['type'] == 'square':
        lambda_vals = cls.square_couping_effects(
            s, baseline, alpha, beta, spike_times)
      elif filter_par['type'] == 'delayed_square':
        lambda_vals = cls.delayed_square_couping_effects(
            s, baseline, alpha, beta, filter_par['delay'], spike_times)

      lambda_vals = np.insert(lambda_vals, 0, 0)
      lambda_cum_sum = np.cumsum(lambda_vals)
      lambda_cum_sum_ratio = lambda_cum_sum / lambda_max
      if lambda_cum_sum_ratio[-1] > 1:
        err_cnt += 1
        if err_cnt % 10 == 9:
          print(f'Warning: lambda > lambda_max. err cnt{err_cnt}')
        lambda_max = lambda_max * 2
        s = s - u
        continue
      d = np.random.rand()
      ind = np.where(lambda_cum_sum_ratio <= d)[0][-1]
      if ind != num_nodes:
        spike_times[ind].append(s)
        accept_cnt += 1

    if verbose:
      sample_efficiency = accept_cnt / sample_cnt
      print(f'sample efficiency: {np.round(sample_efficiency, 3)}')
    return spike_times


  @classmethod
  def generate_inhomogeneous_hawkes_spike_times_multiple_intensities_single(
      cls,
      filter_par,
      intensity_func,
      trial_length,
      verbose=False):
    """Replication of algorithm in Chen 2016.

    Args:
      filter_par: 
    """
    num_nodes = filter_par['num_nodes']
    mu = filter_par['intensity_max']
    mu = np.zeros([num_nodes, 1]) + mu
    alpha = np.array(filter_par['alpha']).astype('double')
    beta = np.array(filter_par['beta']).astype('double')
    gamma = filter_par['gamma']
    eig_val = np.linalg.eigvals(gamma)
    eig_val_max = np.max(np.abs(eig_val))
    if eig_val_max > 1:
      warnings.warn(f'System unstable. Gamma eigval:{eig_val_max}')

    lambda_stable = np.linalg.inv(np.eye(num_nodes) - gamma) @ mu
    lambda_max = np.sum(lambda_stable) * 0.5
    if verbose:
      print('lambda stable:', lambda_stable.reshape(-1))
      print('lambda_max:', lambda_max)

    spike_times = [[] for _ in range(num_nodes)]
    s = 0
    accept_cnt = 0
    sample_cnt = 0
    err_cnt = 0
    while s < trial_length:
      # u = -np.log(np.random.rand()) /lambda_max
      u = np.random.exponential(scale=1/lambda_max)
      s = s + u
      sample_cnt += 1
      if s > trial_length:
        break
      # baseline = mu.copy()
      # baseline = 20 + np.zeros([num_nodes, 1])
      baseline = intensity_func(s)
      baseline = baseline.reshape(num_nodes, 1) + np.zeros([num_nodes, 1])
      if filter_par['type'] == 'none':
        lambda_vals = baseline
      elif filter_par['type'] == 'exp':
        lambda_vals = cls.exp_couping_effects(
            s, baseline, alpha, beta, spike_times)
      elif filter_par['type'] == 'triangle':
        lambda_vals = cls.triangle_couping_effects(
            s, baseline, alpha, beta, spike_times)
      elif filter_par['type'] == 'inv_triangle':
        lambda_vals = cls.inv_triangle_couping_effects(
            s, baseline, alpha, beta, spike_times)
      elif filter_par['type'] == 'square':
        lambda_vals = cls.square_couping_effects(
            s, baseline, alpha, beta, spike_times)
      elif filter_par['type'] == 'delayed_square':
        lambda_vals = cls.delayed_square_couping_effects(
            s, baseline, alpha, beta, filter_par['delay'], spike_times)

      # Clip above zero incase of strong inhibition effect.
      lambda_vals = np.clip(lambda_vals, a_min=0, a_max=None)
      lambda_vals = np.insert(lambda_vals, 0, 0)
      lambda_cum_sum = np.cumsum(lambda_vals)
      lambda_cum_sum_ratio = lambda_cum_sum / lambda_max
      if lambda_cum_sum_ratio[-1] > 1:
        err_cnt += 1
        if err_cnt % 10 == 9:
          print(f'Warning: lambda > lambda_max. err cnt{err_cnt}')
        lambda_max = lambda_max * 2
        s = s - u
        continue
      d = np.random.rand()
      ind = np.where(lambda_cum_sum_ratio <= d)[0][-1]
      if ind != num_nodes:
        spike_times[ind].append(s)
        accept_cnt += 1

    if verbose:
      sample_efficiency = accept_cnt / sample_cnt
      print(f'sample efficiency: {np.round(sample_efficiency, 3)}')
    return spike_times


  @classmethod
  def generate_amarasingham_coupling_filter_spike_times(
      cls,
      model_par=None,
      verbose=False,
      output_figure_path=None):
    """Amarasingham et. al. 2011 simulation scinarios Fig. 1 with coupling.

    The baseline for all trials is the same.
    """
    np.random.seed(model_par['random_seed'])
    trial_length = model_par['trial_length']
    num_trials = model_par['num_trials']
    num_nodes = model_par['num_nodes']
    baseline = model_par['baseline']
    sigma = model_par['sigma']

    if 'num_peaks' in model_par:
      num_peaks = model_par['num_peaks']
      intensity_integral = baseline * trial_length + num_peaks
      peaks = np.random.rand(num_peaks) * trial_length  # Uniform peak positions.
      peaks = np.sort(peaks)
    elif 'rho' in model_par:
      rho = model_par['rho']
      mu = model_par['mu']
      num_peaks = np.random.poisson(lam=rho*trial_length)
      peaks = np.random.rand(num_peaks) * trial_length
      intensity_integral = baseline * trial_length + num_peaks
      peaks = np.sort(peaks)

    # Default window is Laplace.
    if 'window' not in model_par or model_par['window'] == 'laplacian':
      def intensity_func(t):
        return cls._laplacian_intensity_func(t, peaks, sigma, baseline)
    elif model_par['window'] == 'gaussian':
      def intensity_func(t):
        return cls._gaussian_intensity_func(t, peaks, sigma, baseline)

    intensity_peaks = intensity_func(peaks)
    intensity_max_x = peaks[np.argmax(intensity_peaks)]
    intensity_max = intensity_peaks.max()
    model_par['intensity_max'] = intensity_max

    # Check stability.
    if 'alpha' in model_par:
      alpha = np.array(model_par['alpha']).astype('double')
    if 'beta' in model_par:
      beta = np.array(model_par['beta']).astype('double')

    if model_par['type'] == 'exp':
      gamma = alpha / beta
    elif model_par['type'] == 'triangle':
      gamma = alpha * beta / 2
    elif model_par['type'] == 'inv_triangle':
      gamma = alpha * beta / 2
    elif model_par['type'] == 'square':
      gamma = alpha * beta
    elif model_par['type'] == 'delayed_square':
      gamma = alpha * beta
    elif model_par['type'] == 'synchrony':
      gamma = alpha
    elif model_par['type'] == 'none':
      gamma = np.array([[0]])
    model_par['gamma'] = gamma
    mu = model_par['intensity_max']
    mu = np.zeros([num_nodes, 1]) + mu
    lambda_stable = np.linalg.inv(np.eye(num_nodes) - gamma) @ mu
    lambda_max = np.sum(lambda_stable) * 2
    if verbose:
      print('lambda_max (proposal):', np.round(lambda_max, 2))
      print('lambda stable:', np.round(lambda_stable.reshape(-1), 2))

    spike_times = [[] for _ in range(num_nodes)]
    if verbose==2:
      trange = tqdm(range(num_trials), ncols=100, file=sys.stdout)
    else:
      trange = range(num_trials)
    for r in trange:
      spikes = cls.generate_inhomogeneous_hawkes_spike_times_single(
          model_par, intensity_func, trial_length, False)
      for n in range(num_nodes):
        spike_times[n].append(np.array(spikes[n]))

    return spike_times


  @classmethod
  def generate_amarasingham_coupling_filter_spike_times_nonrepeated(
      cls,
      model_par=None,
      verbose=False,
      file_path=None):
    """Amarasingham et. al. 2011 simulation scinarios Fig. 1 with coupling.

    The baselines for trials are different. But all neurons shared the same
    background intensity.
    """
    np.random.seed(model_par['random_seed'])
    trial_length = model_par['trial_length']
    num_trials = model_par['num_trials']
    num_nodes = model_par['num_nodes']
    baseline = model_par['baseline']
    sigma = model_par['sigma']

    spike_times = [[] for _ in range(num_nodes)]
    if verbose==2:
      trange = tqdm(range(num_trials), ncols=100, file=sys.stdout)
    else:
      trange = range(num_trials)

    for r in trange:
      if 'num_peaks' in model_par:
        num_peaks = model_par['num_peaks']
        intensity_integral = baseline * trial_length + num_peaks
        peaks = np.random.rand(num_peaks) * trial_length  # Uniform peak positions.
        peaks = np.sort(peaks)
      elif 'rho' in model_par:
        # The intensity for the background Poisson process.
        rho = model_par['rho']
        mu = model_par['mu']
        num_peaks = np.random.poisson(lam=rho*trial_length)
        peaks = np.random.rand(num_peaks) * trial_length
        intensity_integral = baseline * trial_length + num_peaks
        peaks = np.sort(peaks)

      if 'window' not in model_par or model_par['window'] == 'laplacian':
        def intensity_func(t):
          return cls._laplacian_intensity_func(t, peaks, sigma, baseline)
      elif model_par['window'] == 'gaussian':
        def intensity_func(t):
          return cls._gaussian_intensity_func(t, peaks, sigma, baseline)
      elif model_par['window'] == 'gaussian_varying':
        sigmas = np.random.uniform(low=sigma[0],high=sigma[1],size=len(peaks))
        def intensity_func(t):
          return cls._gaussian_varying_intensity_func(t, peaks, sigmas, baseline)

      intensity_peaks = intensity_func(peaks)
      if len(intensity_peaks) == 0:
        intensity_max_x = None
        intensity_max = mu + baseline
      else:
        intensity_max_x = peaks[np.argmax(intensity_peaks)]
        intensity_max = intensity_peaks.max()
      model_par['intensity_max'] = intensity_max

      # Check stability.
      if 'alpha' in model_par:
        alpha = np.array(model_par['alpha']).astype('double')
      if 'beta' in model_par:
        beta = np.array(model_par['beta']).astype('double')

      if model_par['type'] == 'exp':
        gamma = alpha / beta
      elif model_par['type'] == 'triangle':
        gamma = alpha * beta / 2
      elif model_par['type'] == 'inv_triangle':
        gamma = alpha * beta / 2
      elif model_par['type'] == 'square':
        gamma = alpha * beta
      elif model_par['type'] == 'delayed_square':
        gamma = alpha * beta
      elif model_par['type'] == 'synchrony':
        gamma = alpha
      elif model_par['type'] == 'none':
        gamma = np.array([[0]])
      model_par['gamma'] = gamma
      mu = model_par['intensity_max']
      mu = np.zeros([num_nodes, 1]) + mu
      lambda_stable = np.linalg.inv(np.eye(num_nodes) - gamma) @ mu
      lambda_max = np.sum(lambda_stable) * 2
      if verbose == 3:
        print('lambda_max (proposal):', np.round(lambda_max, 2))
        print('lambda stable:', np.round(lambda_stable.reshape(-1), 2))

      # All neurons shared the same background intensity.
      spikes = cls.generate_inhomogeneous_hawkes_spike_times_single(
          model_par, intensity_func, trial_length, False)
      for n in range(num_nodes):
        spike_times[n].append(np.array(spikes[n]))

      if verbose == 1:
        cls.spike_times_statistics(spike_times, model_par['trial_length'],
            verbose=1)
        jittertool = jitter.JitterTool()

        x = np.arange(0, trial_length, 0.001)
        x = np.concatenate((x, peaks))
        x = np.sort(x)
        y = intensity_func(x)
        gs_kw = dict(width_ratios=[1], height_ratios=[1,1,1])
        fig, axs = plt.subplots(figsize=(10, 8), gridspec_kw=gs_kw,
          nrows=3, ncols=1)
        plt.subplots_adjust(hspace=0, wspace=0)
        ax = fig.add_subplot(axs[0])
        ax.tick_params(labelbottom=False)
        for peak in peaks:
          plt.axvline(x=peak, c='grey', lw=0.3)
        plt.plot(intensity_max_x, intensity_max, 'r+', ms=12)
        plt.plot(x, y, 'k')
        plt.xlim(0, trial_length)
        plt.ylim(0, intensity_max+20)
        plt.ylabel('Firing rate [spk/sec]')

        ax = fig.add_subplot(axs[1])
        ax.tick_params(labelbottom=False, labelleft=True)
        plot_step = 1
        for n in range(0, num_nodes, plot_step):
          plt.plot(spikes[n], n+np.zeros(len(spikes[n])), 'ks', ms=0.4)
        plt.xlim(0, trial_length)
        plt.ylim(-50, num_nodes+50)
        plt.ylabel('Trials')

        ax = fig.add_subplot(axs[2])
        # ax.tick_params(labelleft=False)
        spike_hist, bins = jittertool.bin_spike_times(spikes, 0.005, trial_length)
        plt.plot(bins, spike_hist.mean(axis=0) / 0.005, 'k')
        plt.xlim(0, trial_length)
        plt.ylim(0, intensity_max+20)
        plt.ylabel('Firing rate [spk/sec]')
        plt.xlabel('Time [sec]')
        plt.show()

      if verbose == 3:
        jittertool = jitter.JitterTool()
        x = np.arange(0, trial_length, 0.001)
        x = np.concatenate((x, peaks))
        x = np.sort(x)
        y = intensity_func(x)
        gs_kw = dict(width_ratios=[1], height_ratios=[1])
        fig, axs = plt.subplots(figsize=(6, 2), gridspec_kw=gs_kw,
          nrows=1, ncols=1)
        plt.subplots_adjust(hspace=0, wspace=0)
        ax = fig.add_subplot(axs)
        ax.tick_params(labelbottom=False)
        for peak in peaks:
          plt.axvline(x=peak, c='grey', lw=0.3)
        plt.plot(x, y, 'k')
        plt.xlim(0, trial_length)
        plt.ylim(0, intensity_max+20)
        plt.ylabel('Firing rate [spk/sec]')
        plt.xlabel('Time [sec]')
        if file_path is not None:
          plt.savefig(file_path, bbox_inches='tight')
          print('save figure:', file_path)

    return spike_times


  @classmethod
  def generate_amarasingham_coupling_filter_spike_times_inject_nonshared(
      cls,
      model_par=None,
      verbose=False,
      output_figure_path=None):
    """Amarasingham et. al. 2011 simulation scinarios Fig. 1 with coupling.

    The baselines for trials are different. All neurons share part of the
    baseline, but each neuron also has its own component.
    """
    np.random.seed(model_par['random_seed'])
    trial_length = model_par['trial_length']
    num_trials = model_par['num_trials']
    num_nodes = model_par['num_nodes']
    baseline = model_par['baseline']
    sigma = model_par['sigma']

    if 'window' not in model_par or model_par['window'] == 'laplacian':
      def intensity_func_neuron(t, par):
        peaks_n, sigma, baseline = par
        return cls._laplacian_intensity_func(t, peaks_n, sigma, baseline)
    elif model_par['window'] == 'gaussian':
      def intensity_func_neuron(t, par):
        peaks_n, sigma, baseline = par
        return cls._gaussian_intensity_func(t, peaks_n, sigma, baseline)
    elif model_par['window'] == 'gaussian_varying':
      def intensity_func_neuron(t, par):
        peaks_n, sigmas, baseline = par
        return cls._gaussian_varying_intensity_func(t, peaks_n, sigmas, baseline)

    # Shared components.
    if 'num_peaks' in model_par:
      num_peaks = model_par['num_peaks']
      peaks = np.random.rand(num_peaks) * trial_length  # Uniform peak positions.
      peaks_shared = np.sort(peaks)
    elif 'rho' in model_par:
      # The intensity for the background Poisson process.
      rho = model_par['rho']
      mu = model_par['mu']
      num_peaks = np.random.poisson(lam=rho*trial_length)
      peaks = np.random.rand(num_peaks) * trial_length
      peaks_shared = np.sort(peaks)

    # Non-shared components.
    pars = [0] * num_nodes
    injected_peaks = [0] * num_nodes
    intensity_max_x_n = np.zeros(num_nodes)
    intensity_max_n = np.zeros(num_nodes)
    for n in range(num_nodes):
      rho_inj = model_par['rho_injection']
      num_peaks_inj = np.random.poisson(lam=rho_inj*trial_length)
      peaks_injection = np.random.rand(num_peaks_inj) * trial_length
      peaks_n = np.concatenate([peaks_shared, peaks_injection])
      peaks_n = np.sort(peaks_n)
      injected_peaks[n] = peaks_injection
      if model_par['window'] == 'laplacian':
        pars[n] = (peaks_n, sigma, baseline)
      elif model_par['window'] == 'gaussian':
        pars[n] = (peaks_n, sigma, baseline)
      elif model_par['window'] == 'gaussian_varying':
        sigmas = np.random.uniform(low=sigma[0],high=sigma[1],size=len(peaks_n))
        pars[n] = (peaks_n, sigmas, baseline)
      intensity_peaks = intensity_func_neuron(peaks_n, pars[n])
      intensity_max_x_n[n] = peaks_n[np.argmax(intensity_peaks)]
      intensity_max_n[n] = intensity_peaks.max()

    # Check stability.
    if 'alpha' in model_par:
      alpha = np.array(model_par['alpha']).astype('double')
    if 'beta' in model_par:
      beta = np.array(model_par['beta']).astype('double')
    if model_par['type'] == 'exp':
      gamma = alpha / beta
    elif model_par['type'] == 'triangle':
      gamma = alpha * beta / 2
    elif model_par['type'] == 'inv_triangle':
      gamma = alpha * beta / 2
    elif model_par['type'] == 'square':
      gamma = alpha * beta
    elif model_par['type'] == 'delayed_square':
      gamma = alpha * beta
    elif model_par['type'] == 'synchrony':
      gamma = alpha
    elif model_par['type'] == 'none':
      gamma = np.array([[0]])
    model_par['gamma'] = gamma
    max_id = np.argmax(intensity_max_n)
    intensity_max_x = intensity_max_x_n[max_id]
    intensity_max = intensity_max_n[max_id]
    model_par['intensity_max'] = intensity_max
    # Differnet neurons with differnt intensity funcs.
    def intensity_func(t):
      if np.isscalar(t):
        intensities = np.zeros(num_nodes)
      elif len(t) > 1:
        intensities = np.zeros([num_nodes, len(t)])
      for n in range(num_nodes):
        intensities[n] = intensity_func_neuron(t, pars[n])
      return intensities

    spike_times = [[] for _ in range(num_nodes)]
    if verbose==2:
      trange = tqdm(range(num_trials), ncols=100, file=sys.stdout)
    else:
      trange = range(num_trials)
    for r in trange:
      spikes = cls.generate_inhomogeneous_hawkes_spike_times_multiple_intensities_single(
          model_par, intensity_func, trial_length, False)
      for n in range(num_nodes):
        spike_times[n].append(np.array(spikes[n]))

    if verbose == 3: # Show each neuron's intensity.
      mu = model_par['intensity_max']
      mu = np.zeros([num_nodes, 1]) + mu
      lambda_stable = np.linalg.inv(np.eye(num_nodes) - gamma) @ mu
      lambda_max = np.sum(lambda_stable) * 2
      print('lambda_max (proposal):', np.round(lambda_max, 2))
      print('lambda stable:', np.round(lambda_stable.reshape(-1), 2))

      x = np.arange(0, trial_length, 0.001)
      x = np.concatenate((x, peaks_n))
      x = np.sort(x)
      y = intensity_func(x)
      gs_kw = dict(width_ratios=[1], height_ratios=[1])
      fig, axs = plt.subplots(figsize=(7, 3), gridspec_kw=gs_kw,
          nrows=1, ncols=1)
      plt.subplots_adjust(hspace=0, wspace=0)
      ax = fig.add_subplot(axs)
      ax.tick_params(labelbottom=False)
      for peak in peaks_shared:
        plt.axvline(x=peak, c='lightgrey', lw=0.3)
      for n, (peaks, sigma, baseline) in enumerate(pars):
        plt.plot(injected_peaks[n], 5*(n+1)*np.ones(len(injected_peaks[n])), '+')
      plt.plot(intensity_max_x, intensity_max, 'r+', ms=12)
      plt.plot(x, y.T)
      plt.xlim(0, trial_length)
      plt.ylim(0, intensity_max+20)
      plt.ylabel('Firing rate [spk/sec]')

    if verbose == 1:
      cls.spike_times_statistics(spike_times, model_par['trial_length'],
          verbose=1)
      jittertool = jitter.JitterTool()

      x = np.arange(0, trial_length, 0.001)
      x = np.concatenate((x, peaks_n))
      x = np.sort(x)
      y = intensity_func(x)
      gs_kw = dict(width_ratios=[1], height_ratios=[1])
      fig, axs = plt.subplots(figsize=(8,2), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
      plt.subplots_adjust(hspace=0, wspace=0)
      ax = fig.add_subplot(axs)
      ax.tick_params(labelbottom=False)
      for peak in peaks_n:
        plt.axvline(x=peak, c='grey', lw=0.3)
      plt.plot(intensity_max_x, intensity_max, 'r+', ms=12)
      plt.plot(x, y.T)
      plt.xlim(0, trial_length)
      plt.ylim(0, intensity_max+20)
      plt.ylabel('Firing rate [spk/sec]')

      for n in range(0, num_nodes):
        gs_kw = dict(width_ratios=[1], height_ratios=[1,1])
        fig, axs = plt.subplots(figsize=(8, 4), gridspec_kw=gs_kw,
          nrows=2, ncols=1)
        plt.subplots_adjust(hspace=0, wspace=0)
        ax = fig.add_subplot(axs[0])
        ax.tick_params(labelbottom=False, labelleft=True)
        plot_step = 1
        for r in range(0, num_trials, plot_step):
          plt.plot(spike_times[n][r], r+np.zeros(len(spike_times[n][r])),
              'ks', ms=0.4)
        plt.xlim(0, trial_length)
        # plt.ylim(-50, num_nodes+50)
        plt.ylabel('Trials')
        ax = fig.add_subplot(axs[1])
        # ax.tick_params(labelleft=False)
        spike_hist, bins = jittertool.bin_spike_times(spike_times[n], 0.005, trial_length)
        plt.plot(bins, spike_hist.mean(axis=0) / 0.005, 'k')
        plt.xlim(0, trial_length)
        plt.ylim(0, intensity_max+20)
        plt.ylabel('Firing rate [spk/sec]')
        plt.xlabel('Time [sec]')
        plt.show()

    return spike_times


  @classmethod
  def generate_amarasingham_coupling_filter_spike_times_inject_nonshared_nonrepeated(
      cls,
      model_par=None,
      verbose=False,
      output_figure_path=None):
    """Amarasingham et. al. 2011 simulation scinarios Fig. 1 with coupling.

    The baselines for trials are different. All neurons share part of the
    baseline, but each neuron also has its own component.
    """
    np.random.seed(model_par['random_seed'])
    trial_length = model_par['trial_length']
    num_trials = model_par['num_trials']
    num_nodes = model_par['num_nodes']
    baseline = model_par['baseline']
    sigma = model_par['sigma']
    if 'window' not in model_par or model_par['window'] == 'laplacian':
      def intensity_func_neuron(t, par):
        peaks_n, sigma, baseline = par
        return cls._laplacian_intensity_func(t, peaks_n, sigma, baseline)
    elif model_par['window'] == 'gaussian':
      def intensity_func_neuron(t, par):
        peaks_n, sigma, baseline = par
        return cls._gaussian_intensity_func(t, peaks_n, sigma, baseline)
    elif model_par['window'] == 'gaussian_varying':
      def intensity_func_neuron(t, par):
        peaks_n, sigmas, baseline = par
        return cls._gaussian_varying_intensity_func(t, peaks_n, sigmas, baseline)

    spike_times = [[] for _ in range(num_nodes)]
    if verbose==2:
      trange = tqdm(range(num_trials), ncols=100, file=sys.stdout)
    else:
      trange = range(num_trials)

    for r in trange:
      if 'num_peaks' in model_par:
        num_peaks = model_par['num_peaks']
        peaks = np.random.rand(num_peaks) * trial_length  # Uniform peak positions.
        peaks_shared = np.sort(peaks)
      elif 'rho' in model_par:
        # The intensity for the background Poisson process.
        rho = model_par['rho']
        mu = model_par['mu']
        num_peaks = np.random.poisson(lam=rho*trial_length)
        peaks = np.random.rand(num_peaks) * trial_length
        peaks_shared = np.sort(peaks)

      pars = [0] * num_nodes
      injected_peaks = [0] * num_nodes
      intensity_max_n = np.zeros(num_nodes)
      intensity_max_x_n = np.zeros(num_nodes)
      for n in range(num_nodes):
        rho_inj = model_par['rho_injection']
        num_peaks_inj = np.random.poisson(lam=rho_inj*trial_length)
        peaks_injection = np.random.rand(num_peaks_inj) * trial_length
        peaks_n = np.concatenate([peaks_shared, peaks_injection])
        peaks_n = np.sort(peaks_n)
        injected_peaks[n] = peaks_injection
        if 'window' not in model_par or model_par['window'] == 'laplacian':
          pars[n] = (peaks_n, sigma, baseline)
        elif model_par['window'] == 'gaussian':
          pars[n] = (peaks_n, sigma, baseline)
        elif model_par['window'] == 'gaussian_varying':
          sigmas = np.random.uniform(low=sigma[0],high=sigma[1],size=len(peaks_n))
          pars[n] = (peaks_n, sigmas, baseline)
        intensity_peaks = intensity_func_neuron(peaks_n, pars[n])
        intensity_max_x_n[n] = peaks_n[np.argmax(intensity_peaks)]
        intensity_max_n[n] = intensity_peaks.max()

      # Check stability.
      if 'alpha' in model_par:
        alpha = np.array(model_par['alpha']).astype('double')
      if 'beta' in model_par:
        beta = np.array(model_par['beta']).astype('double')
      if model_par['type'] == 'exp':
        gamma = alpha / beta
      elif model_par['type'] == 'triangle':
        gamma = alpha * beta / 2
      elif model_par['type'] == 'inv_triangle':
        gamma = alpha * beta / 2
      elif model_par['type'] == 'square':
        gamma = alpha * beta
      elif model_par['type'] == 'delayed_square':
        gamma = alpha * beta
      elif model_par['type'] == 'synchrony':
        gamma = alpha
      elif model_par['type'] == 'none':
        gamma = np.array([[0]])
      model_par['gamma'] = gamma
      max_id = np.argmax(intensity_max_n)
      intensity_max_x = intensity_max_x_n[max_id]
      intensity_max = intensity_max_n[max_id]
      model_par['intensity_max'] = intensity_max

      # Differnet neurons with differnt intensity funcs.
      def intensity_func(t):
        if np.isscalar(t):
          intensities = np.zeros(num_nodes)
        elif len(t) > 1:
          intensities = np.zeros([num_nodes, len(t)])
        for n in range(num_nodes):
          intensities[n] = intensity_func_neuron(t, pars[n])
        return intensities
      spikes = cls.generate_inhomogeneous_hawkes_spike_times_multiple_intensities_single(
          model_par, intensity_func, trial_length, False)
      for n in range(num_nodes):
        spike_times[n].append(np.array(spikes[n]))

      if verbose == 3: # Show each neuron's intensity.
        mu = model_par['intensity_max']
        mu = np.zeros([num_nodes, 1]) + mu
        lambda_stable = np.linalg.inv(np.eye(num_nodes) - gamma) @ mu
        lambda_max = np.sum(lambda_stable) * 2
        print('lambda_max (proposal):', np.round(lambda_max, 2))
        print('lambda stable:', np.round(lambda_stable.reshape(-1), 2))

        x = np.arange(0, trial_length, 0.001)
        x = np.concatenate((x, peaks_n))
        x = np.sort(x)
        y = intensity_func(x)
        gs_kw = dict(width_ratios=[1], height_ratios=[1])
        fig, axs = plt.subplots(figsize=(7, 3), gridspec_kw=gs_kw,
            nrows=1, ncols=1)
        plt.subplots_adjust(hspace=0, wspace=0)
        ax = fig.add_subplot(axs)
        ax.tick_params(labelbottom=False)
        for peak in peaks_shared:
          plt.axvline(x=peak, c='lightgrey', lw=0.3)
        for n, (peaks, sigma, baseline) in enumerate(pars):
          plt.plot(injected_peaks[n], 5*(n+1)*np.ones(len(injected_peaks[n])), '+')
        plt.plot(intensity_max_x, intensity_max, 'r+', ms=12)
        plt.plot(x, y.T)
        plt.xlim(0, trial_length)
        plt.ylim(0, intensity_max+20)
        plt.ylabel('Firing rate [spk/sec]')

      if verbose == 1:
        cls.spike_times_statistics(spike_times, model_par['trial_length'],
            verbose=1)
        jittertool = jitter.JitterTool()
        x = np.arange(0, trial_length, 0.001)
        x = np.concatenate((x, peaks_n))
        x = np.sort(x)
        y = intensity_func(x)
        gs_kw = dict(width_ratios=[1], height_ratios=[1])
        fig, axs = plt.subplots(figsize=(8, 2), gridspec_kw=gs_kw,
          nrows=1, ncols=1)
        plt.subplots_adjust(hspace=0, wspace=0)
        ax = fig.add_subplot(axs)
        ax.tick_params(labelbottom=False)
        for peak in peaks_n:
          plt.axvline(x=peak, c='grey', lw=0.3)
        plt.plot(intensity_max_x, intensity_max, 'r+', ms=12)
        plt.plot(x, y.T)
        plt.xlim(0, trial_length)
        plt.ylim(0, intensity_max+20)
        plt.ylabel('Firing rate [spk/sec]')

        gs_kw = dict(width_ratios=[1], height_ratios=[1,1])
        fig, axs = plt.subplots(figsize=(8, 4), gridspec_kw=gs_kw,
          nrows=2, ncols=1)
        plt.subplots_adjust(hspace=0, wspace=0)
        ax = fig.add_subplot(axs[0])
        ax.tick_params(labelbottom=False, labelleft=True)
        plot_step = 1
        for n in range(0, num_nodes, plot_step):
          plt.plot(spikes[n], n+np.zeros(len(spikes[n])), 'ks', ms=0.4)
        plt.xlim(0, trial_length)
        # plt.ylim(-50, num_nodes+50)
        plt.ylabel('Trials')
        ax = fig.add_subplot(axs[1])
        # ax.tick_params(labelleft=False)
        spike_hist, bins = jittertool.bin_spike_times(spikes, 0.005, trial_length)
        plt.plot(bins, spike_hist.mean(axis=0) / 0.005, 'k')
        plt.xlim(0, trial_length)
        plt.ylim(0, intensity_max+20)
        plt.ylabel('Firing rate [spk/sec]')
        plt.xlabel('Time [sec]')
        plt.show()

    return spike_times


  @classmethod
  def generate_amarasingham_coupling_filter_spike_times_delayed_nonrepeated(
      cls,
      model_par=None,
      verbose=False,
      output_figure_path=None):
    """Amarasingham et. al. 2011 simulation scinarios Fig. 1 with coupling.

    The baselines for trials are different. All neurons share part of the
    baseline, but each neuron also has its own component.
    """
    np.random.seed(model_par['random_seed'])
    trial_length = model_par['trial_length']
    num_trials = model_par['num_trials']
    num_nodes = model_par['num_nodes']
    baseline = model_par['baseline']
    sigma = model_par['sigma']
    delays = model_par['delays']

    if 'window' not in model_par or model_par['window'] == 'laplacian':
      def intensity_func_neuron(t, par):
        peaks_n, sigma, baseline = par
        return cls._laplacian_intensity_func(t, peaks_n, sigma, baseline)
    elif model_par['window'] == 'gaussian':
      def intensity_func_neuron(t, par):
        peaks_n, sigma, baseline = par
        return cls._gaussian_intensity_func(t, peaks_n, sigma, baseline)
    elif model_par['window'] == 'gaussian_varying':
      def intensity_func_neuron(t, par):
        peaks_n, sigmas, baseline = par
        return cls._gaussian_varying_intensity_func(t, peaks_n, sigmas, baseline)

    spike_times = [[] for _ in range(num_nodes)]
    if verbose==2:
      trange = tqdm(range(num_trials), ncols=100, file=sys.stdout)
    else:
      trange = range(num_trials)

    for r in trange:
      if 'num_peaks' in model_par:
        num_peaks = model_par['num_peaks']
        peaks = np.random.rand(num_peaks) * trial_length  # Uniform peak positions.
        peaks_shared = np.sort(peaks)
      elif 'rho' in model_par:
        # The intensity for the background Poisson process.
        rho = model_par['rho']
        mu = model_par['mu']
        num_peaks = np.random.poisson(lam=rho*trial_length)
        peaks = np.random.rand(num_peaks) * trial_length
        peaks_shared = np.sort(peaks)

      pars = [0] * num_nodes
      intensity_max_n = np.zeros(num_nodes)
      intensity_max_x_n = np.zeros(num_nodes)
      for n in range(num_nodes):
        peaks_n = peaks_shared + delays[n]
        peaks_n = peaks_n[(peaks_n >= 0) & (peaks_n <= trial_length)]

        if 'window' not in model_par or model_par['window'] == 'laplacian':
          pars[n] = (peaks_n, sigma, baseline)
        elif model_par['window'] == 'gaussian':
          pars[n] = (peaks_n, sigma, baseline)
        elif model_par['window'] == 'gaussian_varying':
          sigmas = np.random.uniform(low=sigma[0],high=sigma[1],size=len(peaks_n))
          pars[n] = (peaks_n, sigmas, baseline)
        intensity_peaks = intensity_func_neuron(peaks_n, pars[n])
        intensity_max_x_n[n] = peaks_n[np.argmax(intensity_peaks)]
        intensity_max_n[n] = intensity_peaks.max()

      # Check stability.
      if 'alpha' in model_par:
        alpha = np.array(model_par['alpha']).astype('double')
      if 'beta' in model_par:
        beta = np.array(model_par['beta']).astype('double')
      if model_par['type'] == 'exp':
        gamma = alpha / beta
      elif model_par['type'] == 'triangle':
        gamma = alpha * beta / 2
      elif model_par['type'] == 'inv_triangle':
        gamma = alpha * beta / 2
      elif model_par['type'] == 'square':
        gamma = alpha * beta
      elif model_par['type'] == 'delayed_square':
        gamma = alpha * beta
      elif model_par['type'] == 'synchrony':
        gamma = alpha
      elif model_par['type'] == 'none':
        gamma = np.array([[0]])
      model_par['gamma'] = gamma
      max_id = np.argmax(intensity_max_n)
      intensity_max_x = intensity_max_x_n[max_id]
      intensity_max = intensity_max_n[max_id]
      model_par['intensity_max'] = intensity_max

      # Differnet neurons with differnt intensity funcs.
      def intensity_func(t):
        if np.isscalar(t):
          intensities = np.zeros(num_nodes)
        elif len(t) > 1:
          intensities = np.zeros([num_nodes, len(t)])
        for n in range(num_nodes):
          intensities[n] = intensity_func_neuron(t, pars[n])
        return intensities
      spikes = cls.generate_inhomogeneous_hawkes_spike_times_multiple_intensities_single(
          model_par, intensity_func, trial_length, False)
      for n in range(num_nodes):
        spike_times[n].append(np.array(spikes[n]))

      if verbose == 3: # Show each neuron's intensity.
        mu = model_par['intensity_max']
        mu = np.zeros([num_nodes, 1]) + mu
        lambda_stable = np.linalg.inv(np.eye(num_nodes) - gamma) @ mu
        lambda_max = np.sum(lambda_stable) * 2
        print('lambda_max (proposal):', np.round(lambda_max, 2))
        print('lambda stable:', np.round(lambda_stable.reshape(-1), 2))

        x = np.arange(0, trial_length, 0.001)
        x = np.concatenate((x, peaks_n))
        x = np.sort(x)
        y = intensity_func(x)
        gs_kw = dict(width_ratios=[1], height_ratios=[1])
        fig, axs = plt.subplots(figsize=(7, 3), gridspec_kw=gs_kw,
            nrows=1, ncols=1)
        plt.subplots_adjust(hspace=0, wspace=0)
        ax = fig.add_subplot(axs)
        ax.tick_params(labelbottom=False)
        for peak in peaks_shared:
          plt.axvline(x=peak, c='lightgrey', lw=0.3)
        plt.plot(intensity_max_x, intensity_max, 'r+', ms=12)
        plt.plot(x, y.T)
        plt.xlim(0, trial_length)
        plt.ylim(0, intensity_max+20)
        plt.ylabel('Firing rate [spk/sec]')

      if verbose == 1:
        cls.spike_times_statistics(spike_times, model_par['trial_length'],
            verbose=1)
        jittertool = jitter.JitterTool()
        x = np.arange(0, trial_length, 0.001)
        x = np.concatenate((x, peaks_n))
        x = np.sort(x)
        y = intensity_func(x)
        gs_kw = dict(width_ratios=[1], height_ratios=[1])
        fig, axs = plt.subplots(figsize=(8, 2), gridspec_kw=gs_kw,
          nrows=1, ncols=1)
        plt.subplots_adjust(hspace=0, wspace=0)
        ax = fig.add_subplot(axs)
        ax.tick_params(labelbottom=False)
        for peak in peaks_n:
          plt.axvline(x=peak, c='grey', lw=0.3)
        plt.plot(intensity_max_x, intensity_max, 'r+', ms=12)
        plt.plot(x, y.T)
        plt.xlim(0, trial_length)
        plt.ylim(0, intensity_max+20)
        plt.ylabel('Firing rate [spk/sec]')

        gs_kw = dict(width_ratios=[1], height_ratios=[1,1])
        fig, axs = plt.subplots(figsize=(8, 4), gridspec_kw=gs_kw,
          nrows=2, ncols=1)
        plt.subplots_adjust(hspace=0, wspace=0)
        ax = fig.add_subplot(axs[0])
        ax.tick_params(labelbottom=False, labelleft=True)
        plot_step = 1
        for n in range(0, num_nodes, plot_step):
          plt.plot(spikes[n], n+np.zeros(len(spikes[n])), 'ks', ms=0.4)
        plt.xlim(0, trial_length)
        # plt.ylim(-50, num_nodes+50)
        plt.ylabel('Trials')
        ax = fig.add_subplot(axs[1])
        # ax.tick_params(labelleft=False)
        spike_hist, bins = jittertool.bin_spike_times(spikes, 0.005, trial_length)
        plt.plot(bins, spike_hist.mean(axis=0) / 0.005, 'k')
        plt.xlim(0, trial_length)
        plt.ylim(0, intensity_max+20)
        plt.ylabel('Firing rate [spk/sec]')
        plt.xlabel('Time [sec]')
        plt.show()

    return spike_times


  @classmethod
  def delayed_shared_input_demo(
      cls,
      output_dir=None):
    """Delayed shared input demo."""
    t = np.arange(-3, 3.5, 0.01); x = np.exp(-t**2); y = np.exp(-(t-0.5)**2)
    shared = np.vstack((x, y)); shared = np.min(shared, axis=0)
    plt.figure(figsize=[4,1.7])
    plt.plot(t, x, 'k', lw=2)
    plt.plot(t, y, 'tab:blue', lw=2)
    plt.plot(t, shared, 'g', lw=2)
    plt.axis('off')
    if output_dir is not None:
      file_path = output_dir + 'shared_input_demo_1.pdf'
      plt.savefig(file_path)
      print(file_path)

    plt.figure(figsize=[4,1.7])
    plt.plot(t, x, 'k', lw=2)
    plt.plot(t, shared, 'g', lw=2)
    plt.plot(t, x - shared, 'k--', lw=2)
    plt.axis('off')
    if output_dir is not None:
      file_path = output_dir + 'shared_input_demo_2.pdf'
      plt.savefig(file_path)
      print(file_path)

    plt.figure(figsize=[4,1.7])
    plt.plot(t, y, c='tab:blue', lw=2)
    plt.plot(t, shared, 'g', lw=2)
    plt.plot(t, y - shared, '--', c='tab:blue', lw=2)
    plt.axis('off')
    if output_dir is not None:
      file_path = output_dir + 'shared_input_demo_3.pdf'
      plt.savefig(file_path)
      print(file_path)


