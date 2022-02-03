"""Hirarchical sampling models."""
import os
import sys

import collections
from collections import defaultdict
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
import seaborn
import scipy
from scipy.ndimage import gaussian_filter1d
import sklearn
import time
from tqdm import tqdm

import data_model
import smoothing_spline
import util
import samples


class HierarchicalSamplingModel(data_model.AllenInstituteDataModel):

  def __init__(self, session=None, empty_samples=False):
    super().__init__(session)
    # Output setup.
    if hasattr(self.session, 'ecephys_session_id'):
      self.session_id = self.session.ecephys_session_id
    else:
      self.session_id = 0
    if empty_samples:
      self.samples = samples.Samples()


  def initial_step(
      self,
      spike_trains,
      spike_times,
      spike_train_time_line,
      selected_units,
      trials_groups,
      trial_time_window,
      probes,
      num_areas,
      num_groups=3,
      model_feature_type='B',
      prior_type='flat',
      eta_smooth_tuning=1e-8,
      verbose=True):
    """Initializes the parameters.

    group_id = 0: cross-pop.
    group_id = 1: local-pop non-constant activities.
    group_id = 2: local-pop constant activities.
    """
    # 'B', 'BS', 'BSS', 'S'. B, baseline, S, time shift.
    self.model_feature_type = model_feature_type
    print('Model feature type: ', model_feature_type)

    # Save the input settings.
    self.spike_trains = spike_trains
    self.spike_times = spike_times
    self.spike_train_time_line = spike_train_time_line
    self.dt = spike_train_time_line[1] - spike_train_time_line[0]
    self.selected_units = selected_units
    self.trials_groups = trials_groups
    self.trial_time_window = trial_time_window
    self.probes = probes

    # These variables are globally the same.
    self.num_areas = num_areas
    self.num_trials = trials_groups.size().max()  # Max trials per condition.
    self.total_num_trials = trials_groups.size().sum()
    self.num_conditions = len(trials_groups)
    self.num_groups = num_groups

    if self.model_feature_type in ['B', 'S', 'S1', 'S2']:
      self.num_qs = 1
    elif self.model_feature_type in ['BS', 'SS']:
      self.num_qs = 2
    elif self.model_feature_type == 'BSS':
      self.num_qs = 3

    # Sample collector.
    self.samples = samples.Samples(
        num_areas=self.num_areas, num_conditions=self.num_conditions,
        num_groups=self.num_groups, num_trials=self.num_trials,
        num_qs=self.num_qs, session_id=self.session_id,
        probes=self.probes)
    # Reset the samples memory.
    self.samples.clear_samples_memory()

    # Area specific variables.
    self.num_neurons_a = np.zeros(self.num_areas)
    # Set 3 for place holder even there are only 2 groups. Since the constant
    # group is always labed as 2. Same for `p_gac`.
    self.f_pop_cag = np.zeros([
        self.num_conditions, self.num_areas, 3,
        len(spike_train_time_line)])
    self.p_gac = np.ones((
        3, self.num_areas, self.num_conditions)) / self.num_groups
    self.q_arc = np.zeros((
        self.num_areas, self.num_trials, self.num_conditions))
    self.q_shift1_arc = np.zeros((
        self.num_areas, self.num_trials, self.num_conditions))
    self.q_shift2_arc = np.zeros((
        self.num_areas, self.num_trials, self.num_conditions))

    self.q_mean_ac = np.zeros((self.num_areas, self.num_conditions))
    self.q_shift1_mean_ac = np.zeros((self.num_areas, self.num_conditions))
    self.q_shift2_mean_ac = np.zeros((self.num_areas, self.num_conditions))

    # self.f_peak1_ac = np.ones([self.num_areas, self.num_conditions]) * 0.06
    # self.f_peak2_ac = np.ones([self.num_areas, self.num_conditions]) * 0.215
    # 791319847
    # self.f_peak1_ac = np.array([[0.055, 0.11, 0.07]] * self.num_conditions).T
    # self.f_peak2_ac = np.array([[0.22, 0.22, 0.22]] * self.num_conditions).T
    # 798911424
    self.f_peak1_ac = np.array([[0.055, 0.08, 0.07]] * self.num_conditions).T
    self.f_peak2_ac = np.array([[0.22, 0.23, 0.23]] * self.num_conditions).T

    self.mean_correct_ratio_q = 0.3
    self.mean_correct_ratio_s1 = 0.3
    self.mean_correct_ratio_s2 = 0.4

    # Shared between areas variables.
    if self.model_feature_type == 'B':
      self.mu_cross_pop = np.zeros(self.num_areas*self.num_qs)
      self.sigma_cross_pop = np.eye(self.num_areas) * 0.1
    elif self.model_feature_type in ['S', 'S1']:
      self.mu_cross_pop = np.zeros(self.num_areas*self.num_qs)
      self.sigma_cross_pop = np.eye(self.num_areas) * 0.0008
    elif self.model_feature_type in ['S2']:
      self.mu_cross_pop = np.zeros(self.num_areas*self.num_qs)
      self.sigma_cross_pop = np.eye(self.num_areas) * 0.0008
    elif self.model_feature_type == 'BS':
      self.mu_cross_pop = np.zeros(self.num_areas*self.num_qs)
      self.sigma_cross_pop = np.diag([0.1, 0.0008] * self.num_areas)
    elif self.model_feature_type == 'SS':
      self.mu_cross_pop = np.zeros(self.num_areas*self.num_qs)
      self.sigma_cross_pop = np.diag([0.0008, 0.001] * self.num_areas)
    elif self.model_feature_type == 'BSS':
      self.mu_cross_pop = np.zeros(self.num_areas*self.num_qs)
      self.sigma_cross_pop = np.diag([0.1, 0.0001, 0.0009] * self.num_areas)

    # Prior parameters.
    # For p_ac.
    self.alpha = 5
    self.xi0 = 3
    # Prior for `sigma_cross_pop`. The reason to multiply `self.nu0` infront of
    # `self.phi0` is that the expecatation of X ~ invWishart(nu0, Psi0)
    # E(X) = Psi0 / (nu0 - p - 1).
    # If nu = df + 1, then marginal distribution of correlation is uniform.
    # If nu = df + 1, scale=Phi, then mode=Phi/2, median= Phi * 1.4.
    self.nu0 = self.num_qs * self.num_areas + 1
    if self.model_feature_type == 'B':
      if prior_type == 'diag':
        # Diagonal prior. x2 is for nu = df + 1 case.
        self.phi0 = np.diag([0.1] * self.num_areas) * 2
      elif prior_type == 'non-diag':
        # Non-diagonal prior.
        self.phi0 = self.nu0 * np.array([[0.08, 0.06],
                                         [0.06, 0.08]])
      elif prior_type == 'flat':
        # Flat prior.
        self.nu0 = 0
        self.phi0 = np.zeros([self.num_areas, self.num_areas])
      self.mu0 = np.zeros(self.num_areas*self.num_qs)
    elif self.model_feature_type in ['S', 'S1']:
      self.mu0 = np.zeros(self.num_areas*self.num_qs)
      self.phi0 = np.eye(self.num_areas) * 0.0008
      self.f_warp_sources_arc = np.zeros((
          self.num_areas, self.num_trials, self.num_conditions, 3))
      self.f_warp_targets_arc = np.zeros((
          self.num_areas, self.num_trials, self.num_conditions, 3))
    elif self.model_feature_type in ['S2']:
      self.mu0 = np.zeros(self.num_areas*self.num_qs)
      self.phi0 = np.eye(self.num_areas) * 0.0008
      # self.phi0 = np.array([[0.0004, 0.0004],
      #                       [0.0004, 0.0008]])
      self.f_warp_sources_arc = np.zeros((
          self.num_areas, self.num_trials, self.num_conditions, 3))
      self.f_warp_targets_arc = np.zeros((
          self.num_areas, self.num_trials, self.num_conditions, 3))
    elif self.model_feature_type == 'BS':
      self.phi0 = np.diag([0.05, 0.001] * self.num_areas)
      self.mu0 = np.zeros(self.num_areas*self.num_qs)
      self.f_warp_sources_arc = np.zeros((
          self.num_areas, self.num_trials, self.num_conditions, 3))
      self.f_warp_targets_arc = np.zeros((
          self.num_areas, self.num_trials, self.num_conditions, 3))
    elif self.model_feature_type == 'SS':
      self.phi0 = np.diag([0.0008, 0.001] * self.num_areas)
      self.mu0 = np.zeros(self.num_areas*self.num_qs)
      self.f_warp_sources_arc = np.zeros((
          self.num_areas, self.num_trials, self.num_conditions, 6))
      self.f_warp_targets_arc = np.zeros((
          self.num_areas, self.num_trials, self.num_conditions, 6))
    elif self.model_feature_type == 'BSS':
      if prior_type == 'diag':
        # x2 is designed for nu = df + 1 case. See description above.
        self.phi0 = np.diag([0.1, 0.0001, 0.0009] * self.num_areas) * 2
      elif prior_type == 'flat':
        self.nu0 = 0  # Overwrite the previous one.
        self.phi0 = np.zeros([3 * self.num_areas, 3 * self.num_areas])

      self.mu0 = np.zeros(self.num_areas*self.num_qs)
      self.f_warp_sources_arc = np.zeros((
          self.num_areas, self.num_trials, self.num_conditions, 6))
      self.f_warp_targets_arc = np.zeros((
          self.num_areas, self.num_trials, self.num_conditions, 6))

    # Fit using smoothing splines.
    self.fit_model = smoothing_spline.SmoothingSpline()
    # Pre-build the basis and Omega matrix for performance need.
    #--------- smoothing spline --------
    self.eta_smooth_tuning = eta_smooth_tuning
    self.f_basis, self.f_Omega = self.fit_model.construct_basis_omega(
        spike_train_time_line, knots=100, verbose=verbose)
    #--------- simple spline fitting --------
    # self.eta_smooth_tuning = 0
    # knots = [0.03, 0.05, 0.07, 0.09, 0.11, 0.2, 0.22, 0.25, 0.27,
    #          0.3, 0.35, 0.4, 0.45]
    # self.f_basis, _ = self.fit_model.bspline_basis(
    #     spline_order=4, knots=knots,
    #     knots_range=[spike_train_time_line[0], spike_train_time_line[-1]],
    #     sample_points=spike_train_time_line, show_plot=False)
    # self.f_Omega = None

    # f_pop parameters.
    self.f_pop_par_cag = {}
    self.f_pop_beta_cag = {}

    # Initialize the group.
    self.spike_counts_c = [pd.DataFrame()] * self.num_conditions
    self.spike_shifts_c = [pd.DataFrame()] * self.num_conditions
    self.spike_shifts1_c = [pd.DataFrame()] * self.num_conditions
    self.spike_shifts2_c = [pd.DataFrame()] * self.num_conditions
    self.sub_group_df_c = [pd.DataFrame()] * self.num_conditions
    self.map_c_to_cid = {}
    self.units_probes = self.selected_units['probe_description']

    def truncate_window(x, left, right):
      """Truncate the spike times into a window."""
      x = x[(x > left) &(x > left)]
      y = np.mean(x) if len(x) != 0 else np.nan
      return y

    for c, (stimulus_condition_id, trials_table) in enumerate(trials_groups):
      print(f'Condition: {c}  stimulus_condition_id:{stimulus_condition_id}')
      self.map_c_to_cid[c] = stimulus_condition_id

      # Initialize the groups.
      trials_indices = trials_groups.get_group(stimulus_condition_id).index.values
      self.spike_counts_c[c] = spike_times.loc[:,trials_indices].applymap(
          lambda x: len(x))
      self.spike_shifts_c[c] = spike_times.loc[:,trials_indices].applymap(
          lambda x: truncate_window(x, left=0, right=0.4))
      self.spike_shifts1_c[c] = spike_times.loc[:,trials_indices].applymap(
          lambda x: truncate_window(x, left=0, right=0.13))
      self.spike_shifts2_c[c] = spike_times.loc[:,trials_indices].applymap(
          lambda x: truncate_window(x, left=0.13, right=0.4))
      # self.sub_group_df_c[c] = self.create_sub_group_df(c)
      # self.sub_group_df_c[c] = self.create_sub_group_df_old(c)
      self.sub_group_df_c[c] = self.create_sub_group_df_intensity(c)
    # Assign a template to the sample collector.
    self.samples.sub_group_df_c = self.sub_group_df_c

  # Deprecated.
  def create_sub_group_df(self, c):
    """Greates subgroup DataFrame."""
    unit_ids = self.selected_units.index.values
    probe_from = self.selected_units['probe_description'].values
    probes = self.probes
    probes = np.array(probes)

    # Only works for two areas. Select the composite probe. 1->0 or 0->1.
    probe_to = np.array(probe_from == probes[0]).astype(int)
    probe_to = probes[probe_to]
    group_id = np.zeros(len(unit_ids)).astype(int)
    sub_group_df = pd.DataFrame({
        'unit_ids': unit_ids.astype(int),
        'probe_from': probe_from,
        'probe_to': probe_to,
        'group_id': group_id})
    sub_group_df.set_index('unit_ids', inplace=True)
    # Idle group.
    unit_count = self.spike_counts_c[c].sum(axis=1)
    threshold = np.quantile(unit_count.values, 0.4)
    active_units = unit_count[unit_count > threshold]
    idle_units = unit_count[unit_count <= threshold]
    sub_group_df.loc[idle_units.index.values, 'group_id'] = 2

    # Active groups 0 & 1.
    scc = self.spike_counts_c[c].T.corr()
    # ssc = self.spike_shifts_c[c].T.corr()
    # ssc = self.spike_shifts1_c[c].T.corr()
    # ssc = self.spike_shifts2_c[c].T.corr()

    if self.num_groups == 3:
      sub_unitsc = self.group_neurons_by_correlation(
          sub_group_df, probes, scc, corr_threshold=0.2, quantile=0.5)
      sub_group_df.loc[sub_unitsc, 'group_id'] = 1

    neurons_counts = sub_group_df[
        (sub_group_df['probe_from'] == probes[0]) &
        (sub_group_df['probe_to'] == probes[1])
        ]['group_id'].value_counts()

    for probe in probes:
      print(f'{probes[0]}  g: {neurons_counts.index.values}  ' +
            f'counts: {neurons_counts.values}')
    return sub_group_df


  def create_sub_group_df_intensity(self, c):
    """Greates subgroup DataFrame."""
    unit_ids = self.selected_units.index.values
    probe_from = self.selected_units['probe_description'].values
    probes = self.probes
    probes = np.array(probes)

    # Only works for two areas. Select the composite probe. 1->0 or 0->1.
    # probe_to = np.array(probe_from == probes[0]).astype(int)
    # probe_to = probes[probe_to]

    group_id = np.zeros(len(unit_ids)).astype(int)
    sub_group_df = pd.DataFrame({
        'unit_ids': unit_ids.astype(int),
        'probe': probe_from,
        'group_id': group_id})
    sub_group_df.set_index('unit_ids', inplace=True)

    token = ''
    for probe in probes:
      units = sub_group_df[sub_group_df['probe'] == probe].index.values
      # Idle group. Low threshold
      unit_count = self.spike_counts_c[c].loc[units].sum(axis=1)
      threshold1 = np.quantile(unit_count.values, 0.4)
      idle_units = unit_count[unit_count <= threshold1]
      sub_group_df.loc[idle_units.index.values, 'group_id'] = 2

      if self.num_groups == 3:
        # Active group. Hight threshold.
        threshold2 = np.quantile(unit_count.values, 0.75)
        active_units = unit_count[(unit_count<=threshold2)&(unit_count>threshold1)]
        sub_group_df.loc[active_units.index.values, 'group_id'] = 1

      neurons_counts = sub_group_df[
          (sub_group_df['probe'] == probe)]['group_id'].value_counts()
      token += (f'{probe} g:{neurons_counts.index.values} ' +
                f'counts:{neurons_counts.values}  ')
    print(token)
    return sub_group_df

  # Deprecated.
  def create_sub_group_df_old(self, c):
    """Old method. i.e. gets group 0 first, then threshold out group 2."""
    probes = self.probes
    scc = self.spike_counts_c[c].T.corr()
    sub_group_df = self.seperate_probe_pairs_units_by_corr_matrix(
        scc, self.units_probes, corr_threshold=0.3, quantile=0.7)

    unit_count =  self.spike_counts_c[c].sum(axis=1)
    for a, probe_pair in enumerate(list(itertools.permutations(probes, 2))):
      units_local_pop = sub_group_df[
          (sub_group_df['probe_from'] == probe_pair[0]) &
          (sub_group_df['probe_to'] == probe_pair[1]) &
          (sub_group_df['group_id'] == 1)].index.values
      threshold = np.quantile(unit_count.loc[units_local_pop].values, 0.5)
      active_units = unit_count.loc[units_local_pop][unit_count > threshold]
      idle_units = unit_count.loc[units_local_pop][unit_count <= threshold]
      sub_group_df.loc[idle_units.index.values, 'group_id'] = 2

      neurons_counts = sub_group_df[
          (sub_group_df['probe_from'] == probe_pair[0]) &
          (sub_group_df['probe_to'] == probe_pair[1])
          ]['group_id'].value_counts()
      print(f'Area {probe_pair[0]}  ' + 
            f'g: {neurons_counts.index.values}  counts: {neurons_counts.values}')
    return sub_group_df


  def group_neurons_by_correlation(
      self,
      sub_group_df,
      probes,
      corr_matrix,
      corr_threshold=0.2,
      quantile=0.3):
    """Groups the neurons by the correlation matrix."""
    # Comment out for more than 2 areas interactions.
    # for probe0, probe1 in itertools.combinations(self.probes, 2):
    units0 = sub_group_df[(sub_group_df['probe_from']==probes[0]) &
                          (sub_group_df['group_id']==0)].index.values
    units1 = sub_group_df[(sub_group_df['probe_from']==probes[1]) &
                          (sub_group_df['group_id']==0)].index.values

    sub_corr_matrix = corr_matrix.loc[units0, units1]
    sub_corr_matrix = sub_corr_matrix[sub_corr_matrix > corr_threshold]

    # Sum the correlation from one area to the other.
    sub_units0 = sub_corr_matrix.sum(axis=1)
    sub_units1 = sub_corr_matrix.sum(axis=0)
    count_threshold0 = np.quantile(sub_units0, quantile)
    count_threshold1 = np.quantile(sub_units1, quantile)
    sub_units0 = sub_units0[sub_units0 > count_threshold0].index.values
    sub_units1 = sub_units1[sub_units1 > count_threshold1].index.values
    sub_units0c = np.setdiff1d(units0, sub_units0)
    sub_units1c = np.setdiff1d(units1, sub_units1)
    return np.hstack((sub_units0c, sub_units1c))


  def save_model(
      self,
      save_data=False,
      file_path=None,
      verbose=False):
    """Saves the memory to file, not including sample memory."""
    import pickle

    if save_data:
      excluded_attributes = ['session', 'samples']
    else:
      excluded_attributes = ['session', 'spike_trains', 'spike_times', 'samples']
    new_model = {}
    for key in self.__dict__.keys():
      if key in excluded_attributes:
        continue
      if self.__dict__[key] is None:
        print(f'warning: mode[{key}] is None.')
        continue
      new_model[key] = self.__dict__[key]

    with open(file_path, 'wb') as f:
      pickle.dump(new_model, f)
      if verbose:
        print('File saved to:', file_path)


  def load_model(
      self,
      file_path=None,
      verbose=False):
    """Loads the memory from file."""
    import pickle
    with open(file_path, 'rb') as f:
      new_model = pickle.load(f)
      # Copy everything from the loaded structure to myself.
      self.__dict__.update(new_model)
    if verbose:
      print('Load model:', file_path)


  def set_model(
      self,
      true_model,
      select_clist):
    """Set a model using another one.

    This function is used in simulation, where the initial values are set as
    truth.
    """
    attrs = ['mu_cross_pop', 'sigma_cross_pop', 'f_pop_cag',
             'q_arc', 'q_shift1_arc', 'q_shift2_arc', 'f_peak1_ac', 'f_peak2_ac',
             'f_warp_sources_arc', 'f_warp_targets_arc',
             'f_pop_par_cag', 'f_pop_beta_cag',
             'p_gac', 'sub_group_df_c']

    for key in attrs:
      self.__dict__[key] = true_model.__dict__[key].copy()
    # self.phi0 = true_model.sigma_cross_pop * 2

    for c, c_id in enumerate(select_clist):
      for a, probe in enumerate(self.probes):
        self.f_pop_beta_cag[(c,a,0,'beta')] = true_model.f_pop_beta_cag[
            (c_id,a,0,'beta')]
        self.f_pop_beta_cag[(c,a,0,'baseline')] = true_model.f_pop_beta_cag[
            (c_id,a,0,'baseline')]
        self.f_pop_beta_cag[(c,a,1,'beta')] = true_model.f_pop_beta_cag[
            (c_id,a,1,'beta')]
        self.f_pop_beta_cag[(c,a,1,'baseline')] = true_model.f_pop_beta_cag[
            (c_id,a,1,'baseline')]
        self.f_pop_beta_cag[(c,a,2,'baseline')] = true_model.f_pop_beta_cag[
            (c_id,a,2,'baseline')]

        self.f_pop_par_cag[(c_id,a,0,'beta_hessian')] = true_model.f_pop_par_cag[
            (c_id,a,0,'beta_hessian')]
        self.f_pop_par_cag[(c_id,a,0,'beta_baseline_hessian')] = true_model.f_pop_par_cag[
            (c_id,a,0,'beta_baseline_hessian')]
        self.f_pop_par_cag[(c_id,a,1,'beta_hessian')] = true_model.f_pop_par_cag[
            (c_id,a,1,'beta_hessian')]
        self.f_pop_par_cag[(c_id,a,1,'beta_baseline_hessian')] = true_model.f_pop_par_cag[
            (c_id,a,1,'beta_baseline_hessian')]
        self.f_pop_par_cag[(c_id,a,2,'beta_baseline_hessian')] = true_model.f_pop_par_cag[
            (c_id,a,2,'beta_baseline_hessian')]

    self.samples.f_pop.append(self.f_pop_cag.copy())
    self.samples.f_pop_beta.append(self.f_pop_beta_cag.copy())
    self.samples.mu_cross_pop.append(self.mu_cross_pop.copy())
    self.samples.sigma_cross_pop.append(self.sigma_cross_pop.copy())
    self.samples.p.append(self.p_gac.copy())

    # z = np.zeros((self.num_conditions, len(self.sub_group_df_c[0])))
    # for c in range(self.num_conditions):
    #   z[c] = self.sub_group_df_c[c]['group_id'].values
    # self.samples.z.append(z.copy())

    if self.model_feature_type in ['B', 'BS', 'BSS']:
      self.samples.q.append(self.q_arc.copy())
    if self.model_feature_type in ['S', 'S1', 'S2', 'BS', 'SS', 'BSS']:
      self.samples.q_shift1.append(self.q_shift1_arc.copy())
      self.samples.f_warp_sources.append(self.f_warp_sources_arc.copy())
      self.samples.f_warp_targets.append(self.f_warp_targets_arc.copy())
    if self.model_feature_type in ['SS', 'BSS']:
      self.samples.q_shift2.append(self.q_shift2_arc.copy())

  def recorder(
      self,
      itr,
      output_dir,
      experiment_folder,
      batch_size=200,
      force_save=False,
      reset_memory=True,
      verbose=False):
    """"Save the data in batches."""
    if itr < batch_size-1 and not force_save:
      return
    if not hasattr(self, 'batch_cnt'):
      self.batch_cnt = -1

    if itr % batch_size == batch_size-1 or force_save:
      # Batch count. This has to be added in the front. The count +1 won't be
      # saved if it is after the saveing.
      self.batch_cnt += 1

      if verbose:
        print(itr, self.batch_cnt, output_dir, experiment_folder)

      dump_dir = os.path.join(output_dir, experiment_folder)
      if not os.path.isdir(dump_dir):
        os.mkdir(dump_dir)
        if verbose:
          print('make folder:', dump_dir)
      timestr = time.strftime('%Y%m%d_%H%M%S')

      # Save model.
      file_name = (str(self.session.ecephys_session_id) + '_checkpoints_' +
                   f'batch{str(self.batch_cnt)}_' + timestr + '.pkl')
      # try:
      #   self.save_model(save_data=(self.batch_cnt==0),
      #                   file_path=os.path.join(dump_dir, file_name))
      # except:
      #   print('Error in saving model.', file_name)
      self.save_model(save_data=(self.batch_cnt==0),
                      file_path=os.path.join(dump_dir, file_name))

      # Save samples
      file_name = (str(self.session.ecephys_session_id) + '_samples_' +
                   f'batch{str(self.batch_cnt)}_' + timestr + '.pkl')
      # try:
      #   self.samples.save(file_path=os.path.join(dump_dir, file_name))
      #   if reset_memory:
      #     self.samples.clear_samples_memory()
      # except:
      #   print('Error in saving samples.', file_name)
      self.samples.save(file_path=os.path.join(dump_dir, file_name))
      if reset_memory:
        self.samples.clear_samples_memory()


  def swap_groups(self, clist, source, target):
    """Swap group order."""
    for c in clist:
      sub_group_df = self.sub_group_df_c[c].copy()
      sub_group_df['group_id'] = sub_group_df['group_id'].replace(source, target)
      self.sub_group_df_c[c] = sub_group_df


  @classmethod
  def compare_models(
      cls,
      model_0,
      model_1):
    """Compare model attributes."""
    attributes = ['session_id', 'model_feature_type', 'spike_train_time_line',
        'dt', 'selected_units', 'trials_groups', 'trial_time_window', 'probes',
        'num_areas', 'num_trials', 'total_num_trials', 'num_conditions',
        'num_groups', 'num_qs', 'num_neurons_a', 'f_basis', 'f_Omega',
        'alpha', 'xi0', 'nu0', 'phi0', 'mu0', 'eta_smooth_tuning',
        'sub_group_df_c', 'map_c_to_cid', 'units_probes', 'q_proposal_scalar_rc',
        'q_propose_cov', 'batch_cnt']

    for key in attributes:
      val_0 = model_0.__dict__[key]
      val_1 = model_1.__dict__[key]

      # print(key, type(val_0))
      if (isinstance(val_0, int) or isinstance(val_0, float) or
          isinstance(val_0, np.ndarray) or isinstance(val_0, np.int64)):
        val_0 = np.array(val_0)
        val_1 = np.array(val_1)
        if not np.array_equal(val_0, val_1):
          print(key, ' not equal.')


  @classmethod
  def print_conditions(
      cls,
      trials_groups):
    """Displays information of trials groups."""
    is_drifing_gratings = True
    trial_counter = 0

    for c, (condition_id, trials_df) in enumerate(trials_groups):
      if ('orientation' not in trials_df or
          'temporal_frequency' not in trials_df or
          'contrast' not in trials_df):
        is_drifing_gratings = False
        break
      orientation = trials_df['orientation'].unique()
      temporal_frequency = trials_df['temporal_frequency'].unique()
      contrast = trials_df['contrast'].unique()
      print(f'{c}  {condition_id} ' + 
            f'temp freq {temporal_frequency} ' +
            f'orient {orientation} ' +
            f'contrast {contrast} ' +
            f'{trials_df.index.values}')
      trial_counter += len(trials_df)

    if not is_drifing_gratings:
      for key in trials_groups.groups:
        trial_ids = trials_groups.groups[key].values
        print('stimulus_id', key, ' trial_id', trial_ids)
        trial_counter += len(trial_ids)

    print('total num trials:', trial_counter)


  def group_activity_statistic_c(
      self,
      trials_indices,
      c):
    """Calculates the group spike counts statistics."""
    sub_group_df = self.sub_group_df_c[c]
    group_ids = sub_group_df['group_id'].unique()
    probes = self.probes
    spike_train_time_line = self.spike_train_time_line

    group_sc = np.zeros([len(probes), self.num_groups, len(trials_indices)])
    group_sft = np.zeros([len(probes), self.num_groups, len(trials_indices)])
    group_sft1 = np.zeros([len(probes), self.num_groups, len(trials_indices)])
    group_sft2 = np.zeros([len(probes), self.num_groups, len(trials_indices)])

    for g in group_ids:
      for a, probe in enumerate(probes):
        units = sub_group_df[
            (sub_group_df['probe'] == probe) &
            (sub_group_df['group_id'] == g)].index.values

        group_sc[a,g] = (
            self.spike_counts_c[c].loc[units, trials_indices].mean(axis=0))
        group_sft[a,g] = (
            self.spike_shifts_c[c].loc[units, trials_indices].mean(axis=0))
        group_sft1[a,g] = (
            self.spike_shifts1_c[c].loc[units, trials_indices].mean(axis=0))
        group_sft2[a,g] = (
            self.spike_shifts2_c[c].loc[units, trials_indices].mean(axis=0))

      scc = np.corrcoef(group_sc[:,g,:])
      sftc = np.corrcoef(group_sft[:,g,:])
      sftc1 = np.corrcoef(group_sft1[:,g,:])
      sftc2 = np.corrcoef(group_sft2[:,g,:])
      print(f'Group id:{g}  ' +
            f'SCC:{scc[0, 1]:.3f}  SFTC:{sftc[0, 1]:.3f}  ' +
            f'SFTC1:{sftc1[0, 1]:.3f}  SFTC2:{sftc2[0, 1]:.3f}')

    mean_firing_rate = group_sc.mean(axis=2)
    for a, probe in enumerate(probes):
      neurons_counts = sub_group_df[
        (sub_group_df['probe'] == probe)]['group_id'].value_counts()
      print(f'{probe} g: {neurons_counts.index.values} ' +
            f'num_neurons: {neurons_counts.values} ' +
            f'MFR:{mean_firing_rate[a,neurons_counts.index.values]}')


  def group_activity_statistic(
      self,
      show_figure=True):
    """Loop every condition."""
    probes = self.probes
    trials_groups = self.trials_groups
    spike_train_time_line = self.spike_train_time_line
    for c, (stimulus_condition_id, trials_table) in enumerate(trials_groups):
      print(f'Condition: {c}  stimulus_condition_id:{stimulus_condition_id}')
      self.group_activity_statistic_c(trials_table.index.values, c)

      if not show_figure:
        continue

      spike_counts_c = self.spike_counts_c[c]
      plt.figure(figsize=(8,2))
      for a, probe in enumerate(probes):
        units = self.selected_units[self.selected_units['probe_description']==probe]
        spike_counts = spike_counts_c.loc[units.index.values].mean(axis=1)
        plt.subplot(1, len(probes), a+1)
        seaborn.distplot(spike_counts, bins=40, kde=False)
        plt.title(probe)
        # plt.xlim(0, 20)
      plt.show()


  def plot_z(
      self,
      clist,
      plot_type='all',
      output_figure_path=None):
    """Compares two subgroups.

    Arg:
      plot_type: 'all', 'separate'.
    """
    areas_names = ['V1', 'LM', 'AL']
    sub_group_df_c = self.sub_group_df_c.copy()

    # Colormap for the areas.
    from_list = matplotlib.colors.LinearSegmentedColormap.from_list
    color_list = [0] * 3
    color_list[0] = 'tab:red'
    color_list[1] = 'tab:blue'
    color_list[2] = 'tab:grey'
    areas_cmap = from_list(None, color_list, self.num_groups)

    fig = plt.figure(figsize=(5 * self.num_areas, 0.2 * len(clist)))
    gs = gridspec.GridSpec(1, self.num_areas,
        width_ratios=[1,1,1],
        height_ratios=[1], wspace=0.03, hspace=0.1)

    for a, probe in enumerate(self.probes):
      zlist = []
      # Setup template.
      sub_group_df = sub_group_df_c[clist[0]]
      sub_group_df = sub_group_df[sub_group_df['probe'] == probe]
      units_template = sub_group_df.sort_values('group_id').index.values

      for c in clist:
        z_df = sub_group_df_c[c].loc[units_template]
        z = z_df['group_id'].values
        zlist.append(z)

      # ax = plt.subplot(1, self.num_areas, a + 1)
      # ax1 = plt.subplot(gs[1, a])  # The empty space.
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

      ax.set_xticklabels([])
      ax.tick_params(bottom=False)

      # plt.yticks(np.arange(len(clist))+0.5, clist)
      # plt.ylim(len(clist), 0)
      # if a != 0:
      #   ax.set_yticklabels([])
      ax.tick_params(left=False)
      ax.set_yticklabels([])
      if a == 0:
        plt.xlabel('Neuron index', fontsize=16)
        plt.ylabel('Condition index', fontsize=16)
      ax.set_title(areas_names[a], fontsize=16)

    if output_figure_path is not None:
      plt.savefig(output_figure_path, bbox_inches='tight')  # pad_inches=0,
      print('save figure:', output_figure_path)
    plt.show()
    plt.close()

    if plot_type == 'all':
      zlist = []
      # Neuron list template.
      units_template = sub_group_df_c[clist[0]].sort_values('group_id').index

      for c in clist:
        sub_group_df_c[c] = sub_group_df_c[c].reindex(units_template)
        z = sub_group_df_c[c]['group_id'].values
        zlist.append(z)

      plt.figure(figsize=(15, 0.3*len(clist)))
      seaborn.heatmap(np.vstack(zlist))
      plt.show()


  def plot_z_model_locations(
      self,
      clist,
      output_figure_path=None):
    """Compares two subgroups."""
    areas_names = ['V1', 'LM', 'AL']
    sub_group_df_c = self.sub_group_df_c.copy()

    fig = plt.figure(figsize=(5*self.num_areas, 1.1 * len(clist)))
    gs = gridspec.GridSpec(len(clist), self.num_areas, wspace=0, hspace=0,
        width_ratios=[1]*self.num_areas, height_ratios=[1]*len(clist))

    for a, probe in enumerate(self.probes):
      # Setup template.
      sub_group_df = sub_group_df_c[clist[0]]
      sub_group_df = sub_group_df[sub_group_df['probe'] == probe]
      units_template = sub_group_df.sort_values('group_id').index.values

      lim_x_min = 100000
      lim_x_max = -1000
      for row, c in enumerate(clist):
        ax = plt.subplot(gs[row, a])
        z_df = sub_group_df_c[c].loc[units_template]

        units = z_df[z_df['group_id'] == 2].index.values
        units = self.selected_units.loc[units]
        v_pos2, h_pos2, _ = self.channel_position(units)
        plt.scatter(v_pos2, h_pos2, marker='o', s=80, c='lightgrey',
                    label='local-pop-2')

        units = z_df[z_df['group_id'] == 1].index.values
        units = self.selected_units.loc[units]
        v_pos1, h_pos1, _ = self.channel_position(units)
        plt.scatter(v_pos1, h_pos1, marker='o', facecolors='none', edgecolors='b',
            linewidth=0.4, s=80, label='local-pop-1')

        units = z_df[z_df['group_id'] == 0].index.values
        units = self.selected_units.loc[units]
        v_pos0, h_pos0, _ = self.channel_position(units)
        plt.scatter(v_pos0, h_pos0, marker='.', c='r', linewidth=0.08,
                    label='cross-pop')

        lim_x_min = np.min([lim_x_min, np.min(v_pos0), np.min(v_pos1), np.min(v_pos2)])
        lim_x_max = np.max([lim_x_max, np.max(v_pos0), np.max(v_pos1), np.max(v_pos2)])

        plt.yticks([11, 27, 43, 59])
        ax.xaxis.set_major_locator(MultipleLocator(200))
        ax.xaxis.set_minor_locator(MultipleLocator(20))
        ax.tick_params(left=True, labelleft=False,
            bottom=True, labelbottom=False, direction='in')
        ax.tick_params(which='minor', direction='in',
            left=False, labelleft=False, bottom=True, labelbottom=False)

        if row == 0:
          plt.title(f'{areas_names[a]}', fontsize=16)
        if a == 0:
          plt.text(0.02, 0.85, f'condition={self.map_c_to_cid[c]}',
                   size=8, transform=ax.transAxes)
        if row == len(clist) - 1 and a == 0:
          plt.yticks([11, 27, 43, 59])
          ax.tick_params(labelleft=True)
          plt.xlabel(r'Channel position [$\mu$m]', fontsize=12)
        if row == len(clist) - 1:
          ax.xaxis.set_major_locator(MultipleLocator(200))
          ax.tick_params(bottom=True, labelbottom=True)
        if row == len(clist) - 1 and a == 1:
          plt.legend(loc=(-0.03, -0.6), ncol=3)

        plt.xlim(lim_x_min-40, lim_x_max+39)
        plt.ylim(-10, 90)

      # return
    if output_figure_path is not None:
      plt.savefig(output_figure_path, bbox_inches='tight')
      print('save figure:', output_figure_path)
    plt.show()
    plt.close()


  def plot_z_samples_location(
      self,
      clist,
      burn_in=0,
      end=None,
      step=1,
      output_figure_path=None):
    """Plots z."""
    areas_names = ['V1', 'LM', 'AL']
    sub_group_df_c = self.sub_group_df_c.copy()
    # Get the membership count mode for each neuron.
    z_c = self.samples.get_z_mode(burn_in, end, step, sub_group_df_c)

    fig = plt.figure(figsize=(5*self.num_areas, 1.1 * len(clist)))
    gs = gridspec.GridSpec(len(clist), self.num_areas, wspace=0, hspace=0,
        width_ratios=[1]*self.num_areas, height_ratios=[1]*len(clist))

    for a, probe in enumerate(self.probes):
      # Setup units template across different conditions.
      sub_group_df = sub_group_df_c[clist[0]]
      sub_group_df = sub_group_df[sub_group_df['probe'] == probe]
      units_template = sub_group_df.sort_values('group_id').index

      lim_x_min = 100000
      lim_x_max = -100000
      for row, c in enumerate(clist):
        ax = plt.subplot(gs[row, a])

        z_df = z_c[c].loc[units_template.values]
        units = z_df[z_df == 2].index.values
        units = self.selected_units.loc[units]
        v_pos2, h_pos2, _ = self.channel_position(units)
        plt.scatter(v_pos2, h_pos2, marker='o', s=80, c='lightgrey',
                    label='local-2')

        units = z_df[z_df == 1].index.values
        units = self.selected_units.loc[units]
        v_pos1, h_pos1, _ = self.channel_position(units)
        plt.scatter(v_pos1, h_pos1, marker='o', facecolors='none', edgecolors='k',
            linewidth=0.4, s=80, label='local-1')

        units = z_df[z_df == 0].index.values
        units = self.selected_units.loc[units]
        v_pos0, h_pos0, _ = self.channel_position(units)
        plt.scatter(v_pos0, h_pos0, marker='.', c='r', s=40,
                    label='pop')

        lim_x_min = np.min([lim_x_min, np.min(v_pos0), np.min(v_pos1), np.min(v_pos2)])
        lim_x_max = np.max([lim_x_max, np.max(v_pos0), np.max(v_pos1), np.max(v_pos2)])

        plt.yticks([11, 27, 43, 59])
        ax.xaxis.set_major_locator(MultipleLocator(200))
        ax.xaxis.set_minor_locator(MultipleLocator(20))
        ax.tick_params(left=True, labelleft=False,
            bottom=True, labelbottom=False, direction='in')
        ax.tick_params(which='minor', direction='in',
            left=False, labelleft=False, bottom=True, labelbottom=False)

        if row == 0:
          plt.title(f'{areas_names[a]}', fontsize=16)
        if a == 0:
          plt.text(0.02, 0.85, f'condition={self.map_c_to_cid[c]}',
                   size=8, transform=ax.transAxes)
        if row == len(clist) - 1 and a == 0:
          plt.yticks([11, 27, 43, 59])
          ax.tick_params(labelleft=True)
          plt.xlabel(r'Channel position [$\mu$m]', fontsize=12)
        if row == len(clist) - 1:
          ax.xaxis.set_major_locator(MultipleLocator(200))
          ax.tick_params(bottom=True, labelbottom=True)
        if row == len(clist) - 1 and a == 1:
          plt.legend(loc=(-0.03, -0.6), ncol=3)

        plt.xlim(lim_x_min-40, lim_x_max+39)
        plt.ylim(-10, 90)

      # return
    if output_figure_path is not None:
      plt.savefig(output_figure_path, bbox_inches='tight')
      print('save figure:', output_figure_path)
    plt.show()
    plt.close()


  def channel_position(
      self,
      probe_units=None):
    """Find the channel position.

    Args:
      probe_units: these units need to be in the same probe.
    """
    all_units = self.session.units
    probe = probe_units['probe_description'].iloc[0]
    all_probe_units = all_units[all_units['probe_description']==probe]
    probe_vertical_pos_max = all_probe_units['probe_vertical_position'].max()
    v_pos = probe_units['probe_vertical_position'] - probe_vertical_pos_max
    h_pos = probe_units['probe_horizontal_position']
    v_pos = -v_pos
    unit_ids = probe_units.index.values
    return v_pos, h_pos, unit_ids


  def plot_z_spatial_frequency(
      self,
      clist,
      burn_in=0,
      end=None,
      step=1,
      output_figure_path=None):
    """Plots z."""
    areas_names = ['V1', 'LM', 'AL']
    sub_group_df_c = self.sub_group_df_c.copy()
    # Get the membership count mode for each neuron.
    z_c = self.samples.get_z_mode(burn_in, end, step, sub_group_df_c)
    num_conditions = len(clist)

    circle_scale = 300
    fig = plt.figure(figsize=[5*self.num_areas, 1])
    gs = gridspec.GridSpec(1, self.num_areas, wspace=0.05, hspace=0,
        width_ratios=[1]*self.num_areas, height_ratios=[1])

    for a, probe in enumerate(self.probes):
      # Setup units template across different conditions.
      sub_group_df = sub_group_df_c[clist[0]]
      sub_group_df = sub_group_df[sub_group_df['probe'] == probe]
      units_template = sub_group_df.sort_values('group_id').index
      lim_x_min = 100000
      lim_x_max = -100000
      pos_freq_cnt = {}
      pos_neuron_cnt = {}

      ax = plt.subplot(gs[a])
      for c_id, c in enumerate(clist):
        z_df = z_c[c].loc[units_template.values]

        units = z_df[z_df == 0].index.values
        units = self.selected_units.loc[units]
        v_pos, h_pos, unit_ids = self.channel_position(units)

        for (v, h, nid) in zip(v_pos, h_pos, unit_ids):
          if (v, h) not in pos_freq_cnt:
            pos_freq_cnt[(v, h)] = 1
          else:
            pos_freq_cnt[(v, h)] += 1

          if (v, h) not in pos_neuron_cnt:
            pos_neuron_cnt[(v, h)] = [nid]
          elif nid not in pos_neuron_cnt[(v, h)]:
            pos_neuron_cnt[(v, h)].append(nid)

      for (v, h), cnt in pos_freq_cnt.items():
        # print((v, h), cnt)
        plt.scatter(v, h, s=circle_scale*cnt/num_conditions,
            marker='.', c='k')
        neuron_cnt = len(pos_neuron_cnt[(v, h)])
        if neuron_cnt > 1:
          plt.text(v-5, h+cnt/4+5, neuron_cnt, fontsize=8)

      lim_x_min = np.min([lim_x_min, np.min(v_pos)])
      lim_x_max = np.max([lim_x_max, np.max(v_pos)])
      plt.yticks([11, 27, 43, 59])
      plt.xlim(lim_x_min-40, lim_x_max+39)
      plt.ylim(-0, 80)
      plt.title(f'{areas_names[a]}', fontsize=16)
      plt.yticks([11, 27, 43, 59])
      ax.xaxis.set_major_locator(MultipleLocator(100))
      ax.xaxis.set_minor_locator(MultipleLocator(20))
      ax.tick_params(left=True, labelleft=False,
          bottom=True, labelbottom=True, direction='out')
      ax.tick_params(which='minor', direction='out',
          left=False, labelleft=False, bottom=True, labelbottom=False)
      if a == 0:
        ax.tick_params(labelleft=True)
        plt.xlabel(r'Channel position [$\mu$m]', fontsize=12)
      if a == 1:
        sizes = [0.1, 0.5, 1, 1.5]
        labels = [0.1, 0.5, 1, 1.5]
        for i, s in enumerate(sizes):
          plt.scatter(10000, 10000, s=circle_scale*s, label=labels[i],
              marker='.', c='k')
        plt.legend(loc=(0.05, -0.7), ncol=5)

    if output_figure_path is not None:
      plt.savefig(output_figure_path, bbox_inches='tight')
      print('save figure:', output_figure_path)
    plt.show()
    plt.close()


  @classmethod
  def compare_z_samples_between_models(
      cls,
      model_0,
      model_1,
      burn_in=0,
      end=None,
      step=1):
    """Compares the membership between models."""
    z_c_0 = model_0.samples.get_z_mode(burn_in=burn_in, end=end, step=step,
        sub_group_df_c=model_0.sub_group_df_c)
    z_c_1 = model_1.samples.get_z_mode(burn_in=burn_in, end=end, step=step,
        sub_group_df_c=model_0.sub_group_df_c)

    total_cnt = 0
    diff_cnt = 0
    for c in model_0.map_c_to_cid:
      # total_cnt = 0
      # diff_cnt = 0
      diff_cnt += np.sum(z_c_0[c].values.T != z_c_1[c].values.T)
      total_cnt += len(z_c_0[c].values)
    print(c, diff_cnt, total_cnt, diff_cnt/total_cnt)


  @classmethod
  def compare_z_between_models(
      cls,
      model_0,
      model_1):
    """Compares the membership between models."""
    z_c_0 = model_0.sub_group_df_c
    z_c_1 = model_1.sub_group_df_c

    total_cnt = 0
    diff_cnt = 0
    for c in model_0.map_c_to_cid:
      diff_cnt += np.sum(z_c_0[c]['group_id'].values != z_c_1[c]['group_id'].values)
      total_cnt += len(z_c_0[c].values)
    print(c, diff_cnt, total_cnt, diff_cnt/total_cnt)


  def plot_f_pop(
      self,
      c):
    """Plots the results."""
    timeline = self.spike_train_time_line

    # Overlap plots.
    # plt.figure(figsize=(16,5))
    # ax1 = plt.subplot(231)
    # ax1.plot(timeline, np.exp(self.f_pop_cag[c,0,0]), label='A=0', color='b')
    # plt.legend(loc='lower right')
    # ax1.yaxis.set_tick_params(labelsize=7, labelcolor='b')
    # ax2 = ax1.twinx()
    # ax2.plot(timeline, np.exp(self.f_pop_cag[c,1,0]), 'r', label='A=1')
    # ax2.yaxis.set_tick_params(labelsize=7, labelcolor='r')
    # plt.legend(loc='upper right')

    # ax1 = plt.subplot(232)
    # plt.plot(timeline, np.exp(self.f_pop_cag[c,0,1]), color='b')
    # ax1.yaxis.set_tick_params(labelsize=7, labelcolor='b')
    # ax2 = ax1.twinx()
    # ax2.plot(timeline, np.exp(self.f_pop_cag[c,1,1]), 'r')
    # ax2.yaxis.set_tick_params(labelsize=7, labelcolor='r')

    # ax1 = plt.subplot(233)
    # plt.plot(timeline, np.exp(self.f_pop_cag[c,0,2]), color='b')
    # plt.plot(timeline, np.exp(self.f_pop_cag[c,1,2]), color='r')
    # plt.ylim(0, np.max(np.exp(self.f_pop_cag[c,0,0])))
    # ax1.yaxis.set_tick_params(labelsize=7)
    # plt.show()


    fig = plt.figure(figsize=(12, 4))
    for a, probe in enumerate(self.probes):
      ax = plt.subplot(self.num_areas, 3, a * 3 + 1)
      f_cross = np.exp(self.f_pop_cag[c,a,0]) / self.dt
      plt.plot(timeline, f_cross)
      plt.axvline(x=self.f_peak1_ac[a,c], linestyle='--', color='g')
      plt.axvline(x=self.f_peak2_ac[a,c], linestyle='--', color='g')
      ax.set_title(f'Area {a} g 0', fontsize=8)
      if a != self.num_areas-1:
        ax.set_xticklabels([])
      ax.xaxis.grid()

      ax = plt.subplot(self.num_areas, 3, a * 3 + 2)
      f_local1 = np.exp(self.f_pop_cag[c,a,1]) / self.dt
      plt.plot(timeline, f_local1)
      # plt.ylim(0, np.max(f_cross_1))
      ax.set_title(f'Area {a} g 1', fontsize=8)
      if a != self.num_areas-1:
        ax.set_xticklabels([])
      ax.xaxis.grid()

      ax = plt.subplot(self.num_areas, 3, a * 3 + 3)
      f_local2 = np.exp(self.f_pop_cag[c,a,2]) / self.dt
      plt.plot(timeline, f_local2)
      plt.ylim(bottom=0)
      ax.set_title(f'Area {a} g 2', fontsize=8)
      if a != self.num_areas-1:
        ax.set_xticklabels([])
      ax.xaxis.grid()

    fig.tight_layout()
    plt.show()


  def plot_f_pop_simple_avg(
      self,
      clist,
      spike_train_time_line,
      dt,
      output_dir=None,
      session_id=0):
    """Plots the results."""
    areas_names = ['V1', 'LM', 'AL']
    spike_trains = self.spike_trains
    spike_times = self.spike_times
    spike_train_time_line = self.spike_train_time_line
    probes = self.probes
    trials_groups = self.trials_groups
    all_trials = spike_trains.columns.values

    for c in clist:
      print('c: ', c)
      sub_group_df = self.sub_group_df_c[c]
      trials_indices = trials_groups.get_group(self.map_c_to_cid[c]).index.values
      fig = plt.figure(figsize=(10, 5))
      for a, probe in enumerate(self.probes):
        # for g in range(self.num_groups):
        g = 0
        units_cross_pop = sub_group_df[
            (sub_group_df['probe'] == probe) &
            (sub_group_df['group_id'] == g)].index.values
        spikes_nrc = spike_trains.loc[units_cross_pop, trials_indices]
        spikes_nrc = np.stack(spikes_nrc.values.flatten('F'), axis=0)

        log_lmbda_hat0, par = self.fit_model.poisson_regression_smoothing_spline(
            spikes_nrc, spike_train_time_line, constant_fit=False,
            basis=self.f_basis, Omega=self.f_Omega,
            lambda_tuning=self.eta_smooth_tuning, lambda_baseline_tuning=1e-5,
            max_num_iterations=100, verbose=0, verbose_warning=False)
        g = 1
        units_cross_pop = sub_group_df[
            (sub_group_df['probe'] == probe) &
            (sub_group_df['group_id'] == g)].index.values
        spikes_nrc = spike_trains.loc[units_cross_pop, trials_indices]
        spikes_nrc = np.stack(spikes_nrc.values.flatten('F'), axis=0)
        log_lmbda_hat1, par = self.fit_model.poisson_regression_smoothing_spline(
            spikes_nrc, spike_train_time_line, constant_fit=False,
            basis=self.f_basis, Omega=self.f_Omega,
            lambda_tuning=self.eta_smooth_tuning,
            max_num_iterations=100, verbose=0, verbose_warning=False)
        g = 2
        units_cross_pop = sub_group_df[
            (sub_group_df['probe'] == probe) &
            (sub_group_df['group_id'] == g)].index.values
        spikes_nrc = spike_trains.loc[units_cross_pop, trials_indices]
        spikes_nrc = np.stack(spikes_nrc.values.flatten('F'), axis=0)
        log_lmbda_hat2, par = self.fit_model.poisson_regression_smoothing_spline(
            spikes_nrc, spike_train_time_line, constant_fit=True,
            basis=self.f_basis, Omega=self.f_Omega,
            lambda_tuning=self.eta_smooth_tuning, lambda_baseline_tuning=1e-5,
            max_num_iterations=100, verbose=0, verbose_warning=False)

        ax = plt.subplot(self.num_areas, 3, a * 3 + 1)
        f_pop0 = np.exp(log_lmbda_hat0) / dt
        plt.plot(spike_train_time_line, f_pop0)
        plt.title(f'{areas_names[a]} cross-pop')
        ax.xaxis.grid()

        ax = plt.subplot(self.num_areas, 3, a * 3 + 2)
        f_pop1 = np.exp(log_lmbda_hat1) / dt
        plt.plot(spike_train_time_line, f_pop1)
        plt.title(f'{areas_names[a]} local-pop')
        ax.xaxis.grid()

        ax = plt.subplot(self.num_areas, 3, a * 3 + 3)
        f_pop2 = np.exp(log_lmbda_hat2) / dt
        plt.plot(spike_train_time_line, f_pop2)
        plt.ylim(0, np.max(f_pop0))
        plt.title(f'{areas_names[a]} local-pop-const')
        ax.xaxis.grid()

      fig.tight_layout()
      if output_dir is not None:
        file_path = os.path.join(output_dir,
                                 f'{session_id}_c{c}_f_pop_simple_avg.pdf')
        plt.savefig(file_path)

      plt.show()


  def get_f_cross_pop_arc(self):
    """Gets each trial's cross-pop timecourse."""
    spike_train_time_line = self.spike_train_time_line
    probes = self.probes
    f_cross_pop_arc = np.zeros([self.num_areas, self.num_trials,
        self.num_conditions, len(spike_train_time_line)])

    for c in range(self.num_conditions):
      for a, probe in enumerate(probes):
        g = 0
        log_lmbda_hat_cross = self.f_pop_cag[c,a,g]

        if self.model_feature_type in ['S', 'S1', 'S2', 'BS', 'SS', 'BSS']:
          sources = self.f_warp_sources_arc[a,:,c]
          targets = self.f_warp_targets_arc[a,:,c]
          log_lmbda_hat_cross = self.linear_time_warping(
              spike_train_time_line, log_lmbda_hat_cross, sources, targets,
              verbose=False)

        if self.model_feature_type in ['B', 'BS', 'BSS']:
          q_offset = self.q_arc[a,:,c].reshape(-1, 1)
          log_lmbda_hat_cross = log_lmbda_hat_cross + q_offset

        f_cross_pop_arc[a,:,c,:] = log_lmbda_hat_cross

    return f_cross_pop_arc


  def naive_q_arc(self, clist=None, verbose=False):
    """Simple way to calculate q in B mode."""
    areas_names = ['V1', 'LM', 'AL']
    sub_group_df_c = self.sub_group_df_c
    probes = self.probes
    trials_groups = self.trials_groups
    spike_train_time_line = self.spike_train_time_line
    msc_arc = np.zeros([len(probes), self.num_trials, self.num_conditions])

    for c in range(self.num_conditions):
      # `trials_groups` has the format (stimulus_condition_id, trials_table)
      trials_indices = list(trials_groups)[c][1].index.values
      sub_group_df = self.sub_group_df_c[c]
      for a, probe in enumerate(probes):
        units = sub_group_df[
            (sub_group_df['probe'] == probe) &
            (sub_group_df['group_id'] == 0)].index.values
        spike_counts = self.spike_counts_c[c].loc[units, trials_indices]
        msc_arc[a,:,c] = spike_counts.mean(axis=0)

    # Estimate q_arc.
    trial_duration = spike_train_time_line[-1] - spike_train_time_line[0]
    dt = self.dt
    msc_arc[msc_arc==0] = 0.2  # Avoid log zeros.
    group_mfr_ac = msc_arc.mean(axis=1)

    intensity_arc = np.log(msc_arc / trial_duration * dt)
    group_intensity_ac = np.log(group_mfr_ac / trial_duration * dt)
    group_intensity_ac = np.expand_dims(group_intensity_ac, axis=1)
    q_arc = intensity_arc - group_intensity_ac

    x = q_arc.transpose(0,2,1).reshape(self.num_areas, -1)
    for i, probei in enumerate(probes):
      for j, probej in enumerate(probes):
        if i >= j:
          continue
        corr, p_value = scipy.stats.pearsonr(x[i], x[j])
        print(f'{probei}-{probej} gain corr:{corr:.3f}  p:{p_value:.3e}')
        if verbose == 2:
          plt.figure(figsize=(3, 3))
          plt.plot(x[i], x[j], '.')
          plt.axis('equal')
          plt.show()

    if verbose == 1:
      plt.figure(figsize=(3, 3 * self.num_conditions))
      for c in range(self.num_conditions):
        for a, probe in enumerate(self.probes):
          ax = plt.subplot(self.num_areas * self.num_conditions, 1,
                           c * self.num_areas + a + 1)
          plt.plot(q_arc[a,:,c])
          # plt.text(0.4, 0.8, probe, transform=ax.transAxes)
          if a == 0:
            plt.title(f'c {c}  {areas_names[a]}')
          else:
            plt.title(areas_names[a])
          if a != self.num_areas-1:
            ax.set_xticklabels([])
        plt.tight_layout()
    return q_arc


  @classmethod
  def find_peak(
      cls,
      peak_range,
      spike_train_time_line,
      log_lambda):
    """Find peaks in a range.

    Args:
      log_lambda: Y x ... x Y x T matrix. The last axis has to be the timeline.
    """
    index_mask = ((spike_train_time_line > peak_range[0]) &
                  (spike_train_time_line < peak_range[1]))
    log_lambda_masked = log_lambda[..., index_mask]
    spike_train_time_line_masked = spike_train_time_line[index_mask]
    peak_id = np.argmax(log_lambda_masked, axis=-1)
    peak = spike_train_time_line_masked[peak_id]
    return peak


  def get_f_peak_arc(
      self,
      fit_type='quick'):
    """A simple way to get peaks using landmars directly.

    Note that the landmarks self.f_peak1_arc or self.f_peak2_arc may not be very
    accurate. This is a simple way to find the peaks. find_peak can find other
    noisy tips in the range.

    Arg:
      fit_type:
          'simple' landmarks + q_shift.
          'brutal' search in the whole time range.
          'quick' refined peaks + q_shift.
          'refine' search the peaks near 'quick'
    """
    if fit_type == 'simple':
      f_peak1 = self.q_shift1_arc + np.expand_dims(self.f_peak1_ac, axis=1)
      f_peak2 = self.q_shift2_arc + np.expand_dims(self.f_peak2_ac, axis=1)
    elif fit_type == 'brutal':
      log_lmbda_hat_arc = self.get_f_cross_pop_arc()
      f_peak1 = self.find_peak([0, 0.15], self.spike_train_time_line,
                                log_lmbda_hat_arc)
      f_peak2 = self.find_peak([0.16, 0.35], self.spike_train_time_line,
                                log_lmbda_hat_arc)
    elif fit_type == 'quick':
      # First correct the original landmarks.
      # Find estimated peaks using new landmarks.
      f_peak1_ = self.find_peak([0.03, 0.11], self.spike_train_time_line,
                                 self.f_pop_cag[:,:,0])
      f_peak2_ = self.find_peak([0.16, 0.35], self.spike_train_time_line,
                                 self.f_pop_cag[:,:,0])
      f_peak1_ = f_peak1_.T
      f_peak2_ = f_peak2_.T
      f_peak1 = self.q_shift1_arc + np.expand_dims(f_peak1_, axis=1)
      f_peak2 = self.q_shift2_arc + np.expand_dims(f_peak2_, axis=1)
    elif fit_type == 'refine':
      # First correct the original landmarks.
      # Find estimated peaks using new landmarks.
      f_peak1_ = self.find_peak([0.03, 0.11], self.spike_train_time_line,
                                 self.f_pop_cag[:,:,0])
      f_peak2_ = self.find_peak([0.16, 0.35], self.spike_train_time_line,
                                 self.f_pop_cag[:,:,0])
      f_peak1_ = f_peak1_.T
      f_peak2_ = f_peak2_.T
      f_peak1_arc_ = self.q_shift1_arc + np.expand_dims(f_peak1_, axis=1)
      f_peak2_arc_ = self.q_shift2_arc + np.expand_dims(f_peak2_, axis=1)
      # Next trap the peaks near the estimated peaks.
      log_lmbda_hat_arc = self.get_f_cross_pop_arc()
      f_peak1 = np.zeros([self.num_areas, self.num_trials, self.num_conditions])
      f_peak2 = np.zeros([self.num_areas, self.num_trials, self.num_conditions])
      for c in range(self.num_conditions):
        for a in range(self.num_areas):
          for r in range(self.num_trials):
            f_peak1[a,r,c] = self.find_peak(
                [f_peak1_arc_[a,r,c]-0.01, f_peak1_arc_[a,r,c]+0.01],
                self.spike_train_time_line, log_lmbda_hat_arc[a,r,c])
            f_peak2[a,r,c] = self.find_peak(
                [f_peak2_arc_[a,r,c]-0.01, f_peak2_arc_[a,r,c]+0.01],
                self.spike_train_time_line, log_lmbda_hat_arc[a,r,c])

    return f_peak1, f_peak2


  def naive_peak_shift_arc(
    self,
    clist,
    group_id=0,
    time_window=None,
    search_window=None,
    option='q_shift1',
    fit_type='kernel',
    kernel_par=2,
    output_dir=None,
    verbose=0):
    """Naive fit the first peak or the second peak.

    Args:
        fit_type: 'bspline', 'quad', 'kernel', 'model'
        option: 'q_shift1', 'q_shift2'
        verbose: 0, no plots.
                 1, plot peaks vs peaks.
                 2, plot fitted curves (output tons of figures)
    """
    spike_trains = self.spike_trains
    spike_train_time_line = self.spike_train_time_line
    trials_groups = self.trials_groups
    probes = self.probes
    areas = ['V1', 'LM', 'AL']
    features = ['Gain', 'First peak shifting', 'Second peak shifting']

    if time_window is None:
      time_window = [spike_train_time_line[0], spike_train_time_line[-1]+self.dt]
    if option == 'q_shift1':
      search_window=[0.0, 0.15]
    elif option == 'q_shift2':
      search_window=[0.15, 0.35]
    if option == 'q_shift1' and fit_type == 'model':
      search_window=[0.0, 0.15]
    if option == 'q_shift2' and fit_type == 'model':
      search_window=[0.16, 0.35]
    if option == 'q_shift1' and fit_type == 'quad':
      time_window = [0, 0.15]
    if option == 'q_shift2' and fit_type == 'quad':
      time_window = [0.12, 0.4]

    index_range = ((spike_train_time_line >= time_window[0]) &
        (spike_train_time_line < time_window[1]))
    spike_train_time_line = spike_train_time_line[index_range]

    log_lmbda_hat_arc = np.zeros([self.num_areas, self.num_trials,
        self.num_conditions, len(spike_train_time_line)])
    psth_arc = np.zeros([self.num_areas, self.num_trials,
        self.num_conditions, len(spike_train_time_line)])
    f_peak_arc = np.zeros([
          self.num_areas, self.num_trials, self.num_conditions])

    if fit_type == 'model':
      log_lmbda_hat_arc = self.get_f_cross_pop_arc()
      f_peak1_arc, f_peak2_arc = self.get_f_peak_arc(fit_type='refine')
      if option == 'q_shift1':
        f_peak_arc = f_peak1_arc
      elif option == 'q_shift2':
        f_peak_arc = f_peak2_arc

    for c in clist:
      sub_group_df = self.sub_group_df_c[c]
      trials_indices = trials_groups.get_group(self.map_c_to_cid[c]).index.values

      for a, probe in enumerate(probes):
        units = sub_group_df[(sub_group_df['probe'] == probe) &
                             (sub_group_df['group_id'] == group_id)].index.values

        for r, trial_id in enumerate(trials_indices):
          y = np.vstack(spike_trains.loc[units, trial_id].values.reshape(-1))
          if time_window is not None:
            y = y[:, index_range]
            psth_arc[a,r,c] = y.mean(axis=0)

          # The reason find peak is in each option separately is that they need
          # different process. 
          if fit_type == 'bspline':
            log_lmbda_hat_arc[a,r,c], _ = (
                self.fit_model.poisson_regression_smoothing_spline(
                    y, spike_train_time_line,
                    lambda_tuning=self.eta_smooth_tuning))
            f_peak_arc = self.find_peak(search_window, spike_train_time_line,
                                        log_lmbda_hat_arc)
          elif fit_type == 'quad':
            t = np.arange(len(spike_train_time_line)) * self.dt
            basis = np.vstack([t, t**2]).T
            num_basis = basis.shape[1]
            log_lmbda_hat_arc[a,r,c], _ = (
                self.fit_model.poisson_regression_smoothing_spline(
                    y, spike_train_time_line, basis=basis, Omega=np.eye(num_basis),
                    lambda_tuning=0, learning_rate=1, max_num_iterations=100,
                    verbose=False, verbose_warning=False))
            # TODO: Use quadratic property to find the peak.
            f_peak_arc = self.find_peak(search_window, spike_train_time_line,
                                        log_lmbda_hat_arc)
          elif fit_type == 'kernel':
            # Understanding gaussian_filter1d.
            # The following calculation is equivalent.
            # import scipy
            # from scipy.ndimage import gaussian_filter1d
            # t = np.arange(100)
            # x = np.zeros(100); x[50] = 1
            # y = gaussian_filter1d(x, 10)
            # z = scipy.stats.norm.pdf(t, loc=50, scale=10)

            if kernel_par == 0:
              x = psth_arc[a,r,c]
            else:
              x = scipy.ndimage.gaussian_filter1d(psth_arc[a,r,c], kernel_par)
            log_lmbda_hat_arc[a,r,c] = np.log(x + np.finfo(float).eps)
            f_peak_arc = self.find_peak(search_window, spike_train_time_line,
                                        log_lmbda_hat_arc)

          if option is not None:
            if y.sum() == 0:  # Not enough spikes.
              f_peak_arc[a,r,c] = np.nan


    ############## Begin plotting for analysis. ##############
    if verbose == 2:
      # Plot for fitted PSTH.
      for c in clist:
        plt.figure(figsize=(3 * self.num_areas, 1.8 * self.num_trials))
        for a, probe in enumerate(probes):
          for r, trial_id in enumerate(trials_indices):
            ax = plt.subplot(self.num_trials, self.num_areas, r*self.num_areas+a+1)
            plt.plot(spike_train_time_line,
                psth_arc[a,r,c] / self.dt, c='lightgrey')
            plt.plot(spike_train_time_line, 
                np.exp(log_lmbda_hat_arc[a,r,c]) / self.dt, 'b')
            # ax.xaxis.grid(True)
            plt.title(f'c{c} {probe} {trial_id}')
        plt.tight_layout()
        plt.show()

    if option is None:
      return log_lmbda_hat_arc

    # Calculate the peaks only for verbose=2.
    if verbose != 1:
      return f_peak_arc, log_lmbda_hat_arc

    # Correlation.
    f_peaks_all = f_peak_arc[:,:,clist].transpose(0,2,1).reshape(self.num_areas,-1)

    num_rows, num_cols = self.num_areas, self.num_areas
    plt.figure(figsize=(num_rows * 3, num_rows * 3))
    for row, probe1 in enumerate(probes):
      for col, probe2 in enumerate(probes):
        if row > col:
          continue
        if row == col:
          ax = plt.subplot(num_rows, num_cols, col+row*num_rows+1)
          plt.text(0, 0.2, f'{probe1}', ha='center', va='center')
          plt.xlim(-1, 1); plt.ylim(-1, 1)
          ax.set_yticklabels([])
          ax.set_xticklabels([])
          ax.axis('off')
          continue

        ax = plt.subplot(num_rows, num_cols, col+1+row*num_rows)
        # mask = np.logical_and(
        #     ~np.isnan(f_peaks_all)[row] & (f_peaks_all[row] > 0.05),
        #     ~np.isnan(f_peaks_all)[col] & (f_peaks_all[col] > 0.05))
        mask = np.logical_and(~np.isnan(f_peaks_all)[row],
                              ~np.isnan(f_peaks_all)[col])
        f_peaks = f_peaks_all[:, mask]
        corr, p_value = scipy.stats.pearsonr(f_peaks[col], f_peaks[row])
        f_peaks = f_peaks * 1000 # Adjust for ms unit.
        plt.plot([0, 10000], [0, 10000], '--', c='lightgrey')
        plt.plot(f_peaks[col], f_peaks[row], '.k')
        plt.axis('equal')
        plt.axis('square')
        # plt.text(0.01, 0.95, f'r={corr:.3f}  p={p_value:.2e}',
        #          transform=ax.transAxes, fontsize=13)
        plt.xlabel(f'{areas[col]}  post-stim time [ms]', fontsize=15)
        plt.ylabel(f'{areas[row]}  post-stim time [ms]', fontsize=15)
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)
        # plt.xlim(xmin=0.05)
        # plt.ylim(ymin=0.05)
        delta = 0.01 * 1000
        if option == 'q_shift2':
          plt.xlim(100, 400)
          plt.ylim(100, 400)
        elif option == 'q_shift1':
          # plt.xlim(0, 200)
          # plt.ylim(0, 200)
          plt.xlim(np.min(f_peaks[[row, col]])-delta,
                   np.max(f_peaks[[row, col]])+delta)
          plt.ylim(np.min(f_peaks[[row, col]])-delta,
                   np.max(f_peaks[[row, col]])+delta)
        # plt.xlim(np.min(f_peaks[[row, col]])-delta, np.max(f_peaks[[row, col]])+delta)
        # plt.ylim(np.min(f_peaks[[row, col]])-delta, np.max(f_peaks[[row, col]])+delta)
        # ax.set_title(f'{areas[col]}-{areas[row]}', fontsize=12)
    plt.tight_layout()
    plt.show()

    ################### Plot individual for paper figure ###################
    row, col = 0, 1
    mask = np.logical_and(~np.isnan(f_peaks_all)[row],
                          ~np.isnan(f_peaks_all)[col])
    f_peaks = f_peaks_all[:, mask]
    f_peaks = f_peaks * 1000 # Adjust for ms unit.
    lags = f_peaks[col] - f_peaks[row]
    lags = lags[np.abs(lags) < 300]
    lag_mean = np.mean(lags)
    lag_mean_se = np.std(lags) / np.sqrt(len(lags))
    # 95% CI. 0.025 reminder on each side, so q = 1 - 0.05/2.
    CI_scale = scipy.stats.norm.ppf(0.975)
    lag_mean_CI_05 = lag_mean_se * CI_scale
    print(f'# {len(lags)}  mean: {lag_mean:.2f}  SE_mean {lag_mean_se:.2f}  '+
          f'95% CI {lag_mean_CI_05:.2f}  '+
          f'[{lag_mean-lag_mean_CI_05:.2f}, {lag_mean+lag_mean_CI_05:.2f}]')

    plt.figure(figsize=(3.5, 3.5))
    ax = plt.gca()
    ax.tick_params(left=True, labelbottom=True, bottom=True,
                   top=False, labeltop=False)
    plt.plot([0, 10000], [0, 10000], '--', c='lightgrey')
    plt.plot(f_peaks[col], f_peaks[row], '.k')
    plt.axis('equal')
    plt.axis('square')
    # plt.text(0.01, 0.95, f'r={corr:.3f}  p={p_value:.2e}',
    #          transform=ax.transAxes, fontsize=13)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)

    if option == 'q_shift2':
      plt.xlabel(f'{areas[col]}-Peak-2 time [ms]', fontsize=12)
      plt.ylabel(f'{areas[row]}-Peak-2 time [ms]', fontsize=12)
      plt.xticks(np.arange(0, 1000, 100))
      plt.yticks(np.arange(0, 1000, 100))
      plt.xlim(100, 400)
      plt.ylim(100, 400)
    elif option == 'q_shift1':
      plt.xlabel(f'{areas[col]}-Peak-1 time [ms]', fontsize=12)
      plt.ylabel(f'{areas[row]}-Peak-1 time [ms]', fontsize=12)
      plt.xticks(np.arange(0, 1000, 50))
      plt.yticks(np.arange(0, 1000, 50))
      plt.xlim(0, 150)
      plt.ylim(0, 150)
    ax.set_title('Simple method', fontsize=16)
    plt.tight_layout()
    if output_dir is not None:
      output_figure_path = os.path.join(output_dir,
          f'{self.session_id}_{areas[col]}_{areas[row]}_{option}_{fit_type}.pdf')
      plt.savefig(output_figure_path, bbox_inches='tight')  # pad_inches=0,
      print('save figure:', output_figure_path)
    plt.show()

    return f_peak_arc, log_lmbda_hat_arc


  def plot_peaks(
    self,
    clist,
    group_id=0,
    time_window=None,
    search_window=None,
    option='q_shift1',
    fit_type='kernel',
    kernel_par=0.8,
    verbose=0,
    modes_ref=None,
    CIs_ref=None,
    output_dir=None,
    session_id=0):
    """Plots peaks distributions.

    Args:
        fit_type: 'bspline', 'quad', 'kernel', 'model'
        option: 'q_shift1', 'q_shift2'
    """
    areas_names = ['V1', 'LM', 'AL']
    f_peak_arc, log_lmbda_hat_arc = self.naive_peak_shift_arc(
        clist=clist, group_id=group_id, time_window=time_window,
        search_window=search_window, option=option,
        fit_type=fit_type, kernel_par=kernel_par)

    # data = f_peak_arc.copy()
    # data[data < 0.005] = np.nan
    # plt.figure(figsize=(3, 3 * self.num_areas))
    # for a in range(self.num_areas):
    #   ax = plt.subplot(self.num_areas, 1, a+1)
    #   x, y = seaborn.distplot(
    #       data[a], bins=30).get_lines()[0].get_data()
    #   plt.axvline(x=np.nanquantile(data[a], 0.025),
    #               linestyle=':', color='k')
    #   plt.axvline(x=np.nanquantile(data[a], 0.975),
    #               linestyle=':', color='k', label='%95 CI')
    #   plt.axvline(x=x[np.argmax(y)], color='g', label='mode')
    #   plt.xlim(0, 0.15)
    #   plt.xlabel('Time [sec]')
    #   plt.title(f'Peak 1 {areas_names[a]}')
    #   if a == 0:
    #     plt.legend()

    # plt.tight_layout()
    # if output_dir:
    #   plt.savefig(figure_path)
    #   print('save figure:', figure_path)
    # plt.show()

    # Modes and CIs.
    peaks_data = f_peak_arc.copy() * 1000
    peaks_data[peaks_data < 0.005] = np.nan
    fig = plt.figure(figsize=(3, 3))
    ax = plt.gca()
    for a in range(self.num_areas):
      row = -a
      data = peaks_data[a].reshape(-1)
      data = data[~np.isnan(data)]
      gkde=scipy.stats.gaussian_kde(data)
      CI_left = np.nanquantile(peaks_data[a], 0.025)
      CI_right = np.nanquantile(peaks_data[a], 0.975)
      x = np.linspace(CI_left, CI_right, 201)
      y = gkde.evaluate(x)
      center = np.nanquantile(peaks_data[a], 0.5)
      # center = x[np.argmax(y)]
      plt.plot([CI_left, CI_right], [row, row], 'k')
      plt.plot(center, row, 'kx')
      plt.text(center, row+0.2, f'{center:.0f}')
      # plt.xlim(0, 0.15)
      plt.xlabel('Time [ms]')
      plt.yticks(-np.arange(self.num_areas), areas_names)

    if modes_ref is not None:
      for a in range(self.num_areas):
        row = -a - 4
        CI_left = CIs_ref[a,0] * 1000
        CI_right = CIs_ref[a,1] * 1000
        center = modes_ref[a] * 1000
        plt.plot([CI_left, CI_right], [row, row], 'k')
        plt.plot(center, row, 'kx')
        plt.text(center, row+0.2, f'{center:.0f}')
        # plt.xlim(0, 0.15)
        plt.xlabel('Time [ms]')
        ys = [0, -1, -2, -4, -5, -6]
        yticks = areas_names + areas_names
        plt.yticks(ys, yticks)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.text(0.9, 0.9, 'naive', transform=ax.transAxes)
        plt.text(0.9, 0.3, 'model', transform=ax.transAxes)

    plt.tight_layout()
    if output_dir:
      figure_path = os.path.join(output_dir, f'{session_id}_peak1_median_CIs.pdf')
      plt.savefig(figure_path)
      print('save figure:', figure_path)
    plt.show()


  def plot_group_activity_per_trial(
      self,
      clist,
      group_id=0,
      time_window=None,
      search_window=None,
      option=None,
      fit_type='bspline',
      kernel_par=3,
      show_figure=True):
    """Exploration function for per trial group activity.

    Args:
        fit_type: 'bspline', 'quad', 'kernel', 'model'
    """
    spike_trains = self.spike_trains
    spike_train_time_line = self.spike_train_time_line
    trials_groups = self.trials_groups
    probes = self.probes
    if time_window is None:
      time_window = [spike_train_time_line[0], spike_train_time_line[-1]+self.dt]

    if option == 'q_shift1':
      search_window=[0.03, 0.15]
    elif option == 'q_shift2':
      search_window=[0.18, 0.3]
    if option == 'q_shift2' and fit_type == 'model':
      search_window=[0.15, 0.35]
    if option == 'q_shift1' and fit_type == 'quad':
      time_window = [0, 0.15]
    if option == 'q_shift2' and fit_type == 'quad':
      time_window = [0.12, 0.4]

    index_range = ((spike_train_time_line >= time_window[0]) &
        (spike_train_time_line < time_window[1]))
    spike_train_time_line = spike_train_time_line[index_range]

    if option is not None:
      f_peak_arc, log_lmbda_hat_arc = self.naive_peak_shift_arc(
          clist=clist, group_id=group_id, time_window=time_window,
          search_window=search_window, option=option,
          fit_type=fit_type, kernel_par=kernel_par)

      # kernel smoothing as background timecourse.
      _, log_intensity_arc = self.naive_peak_shift_arc(
          clist=clist, group_id=group_id, time_window=time_window,
          search_window=search_window, option=option,
          fit_type='kernel', kernel_par=kernel_par)
      intensity_arc = np.exp(log_intensity_arc)

    # Plot results.
    for c in clist:
      sub_group_df = self.sub_group_df_c[c]
      trials_indices = trials_groups.get_group(self.map_c_to_cid[c]).index.values
      num_row, num_col = len(trials_indices) // 2 + 1, 2 * self.num_areas
      plt.figure(figsize=(18, num_row * 1.6))
      for a, probe in enumerate(probes):
        units = sub_group_df[(sub_group_df['probe'] == probe) &
                             (sub_group_df['group_id'] == group_id)].index.values

        # f_pop timecourse.
        # intensity = np.exp(self.f_pop_cag[c,a,group_id])
        # intensity = intensity[index_range]
        for r, trial_id in enumerate(trials_indices):
          y = np.vstack(spike_trains.loc[units, trial_id].values.reshape(-1))
          if time_window is not None:
            y = y[:, index_range]
          ax = plt.subplot(num_row, num_col, r * self.num_areas + 1 + a)
          plt.plot(spike_train_time_line, y.mean(axis=0) / self.dt / 3, 'lightgray', lw=1)
          if option is not None:
            intensity = intensity_arc[a,r,c]
            plt.plot(spike_train_time_line, intensity / self.dt, 'g')
          else:
            # plt.plot(spike_train_time_line, y.mean(axis=0), 'g')
            pass

          if option is not None:
            plt.plot(spike_train_time_line,
                     np.exp(log_lmbda_hat_arc[a,r,c]) / self.dt, 'b')
            plt.axvline(x=f_peak_arc[a,r,c], linestyle='--', color='r')
            # plt.text(0.1, 0.8, f'a:{a} r:{r}', transform=ax.transAxes)
            plt.title(f'{probe} c:{c} r:{r}  {f_peak_arc[a,r,c]:.3f}')
          else:
            plt.title(f'{probe} c:{c} r:{r}')

          # plt.grid('on')
          if r != len(trials_indices) - 1:
            ax.set_xticklabels([])

      # TODO: add figure output path.
      # output_figure_path = os.path.join(
      #     self.output_dir, 'group_activity_per_trial.pdf')
      # plt.savefig(output_figure_path)
      # print('Save figure to: ', output_figure_path)
      plt.tight_layout()
      plt.show()


  def plot_group_activity_per_trial_demo(
      self,
      c,
      r,
      ylims=None,
      show_label=True,
      show_warp=False,
      show_peak=True,
      true_model=None,
      output_dir=True):
    """Exploration function for per trial group activity.

    Args:
        fit_type: 'bspline', 'quad', 'kernel', 'model'
    """
    areas_names = ['V1', 'LM', 'AL']
    ylim_up = [80, 80, 80]
    group_id = 0
    spike_trains = self.spike_trains
    spike_times = self.spike_times
    spike_train_time_line = self.spike_train_time_line
    trials_groups = self.trials_groups
    probes = self.probes
    time_window = [spike_train_time_line[0], spike_train_time_line[-1]+self.dt]

    index_range = ((spike_train_time_line >= time_window[0]) &
        (spike_train_time_line < time_window[1]))
    spike_train_time_line = spike_train_time_line[index_range]

    f_peak1_arc, log_lmbda_hat_arc = self.naive_peak_shift_arc(
        clist=[c], group_id=group_id, time_window=time_window,
        search_window=[0.03, 0.15], option='q_shift1',
        fit_type='model')

    f_peak2_arc, log_lmbda_hat_arc = self.naive_peak_shift_arc(
        clist=[c], group_id=group_id, time_window=time_window,
        search_window=[0.15, 0.35], option='q_shift2',
        fit_type='model')

    time_line = spike_train_time_line * 1000
    sub_group_df = self.sub_group_df_c[c]
    trials_indices = trials_groups.get_group(self.map_c_to_cid[c]).index.values

    # Firing rate intensity functions example.
    gs_kw = dict(width_ratios=[1,1,1])
    fig, axs = plt.subplots(figsize=(15, 2.2), gridspec_kw=gs_kw,
        nrows=1, ncols=3)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.1)

    for a, probe in enumerate(probes):
      ax = fig.add_subplot(axs[a])
      units = sub_group_df[(sub_group_df['probe'] == probe) &
                           (sub_group_df['group_id'] == group_id)].index.values
      trial_id = trials_indices[r]
      # psth = np.vstack(spike_trains.loc[units, trial_id].values.reshape(-1))
      # if time_window is not None:
      #   psth = psth[:, index_range]
      # psth = psth.mean(axis=0) / self.dt / 3
      # plt.plot(time_line, psth, 'gray', lw=0.8, label='spike train')

      spikes = spike_times.loc[units, trial_id].values.reshape(-1)
      bin_width = 0.01
      psth, bins = util.bin_spike_times(spikes, 0.01, time_window[1])
      bins = bins * 1000
      psth = psth.mean(axis=0) / bin_width
      plt.bar(bins, psth, width=bin_width*1000, color='lightgrey', label='spike train')

      plt.plot(time_line, np.exp(log_lmbda_hat_arc[a,r,c]) / self.dt,
               'k', lw=2, label='fitted activity')
      # plt.plot(time_line, np.exp(self.f_pop_cag[c,a,0]) / self.dt,
      #          '--k', lw=1, label='average activity')
      if show_peak:
        plt.axvline(x=f_peak2_arc[a,r,c] * 1000, linestyle='--', color='b', label='peak')
        plt.axvline(x=f_peak1_arc[a,r,c] * 1000, linestyle='--', color='b')
      # plt.text(0.1, 0.8, f'a:{a} r:{r}', transform=ax.transAxes)
      # plt.title(f'{probe} c:{c} r:{r}  {f_peak_arc[a,r,c]:.3f}')
      if (true_model is not None and
          hasattr(true_model, 'f_peak1_ac') and
          hasattr(true_model, 'f_peak2_ac')):
        true_peak1_arc = (np.expand_dims(true_model.f_peak1_ac, axis=1) +
            true_model.q_shift1_arc)
        true_peak2_arc = (np.expand_dims(true_model.f_peak2_ac, axis=1) +
            true_model.q_shift2_arc)
        plt.axvline(x=true_peak1_arc[a,r,c]*1000, linestyle='-', color='g', lw=1)
        plt.axvline(x=true_peak2_arc[a,r,c]*1000, linestyle='-', color='g', lw=1)

      if ylims is not None:
        plt.ylim(0, ylims[a])
      else:
        plt.ylim(0, ylim_up[a])
      if a == 0:
        # ax.set_title(f'Trial={trial_id}  {areas_names[a]}', fontsize=16)
        ax.set_title(f'{areas_names[a]}', fontsize=16)
        plt.text(0.75, 0.85, f'Trial={trial_id}', transform=ax.transAxes)
      else:
        ax.set_title(f'{areas_names[a]}', fontsize=16)
      ax.tick_params(left=True, labelbottom=True, bottom=True,
                     top=False, labeltop=False)

      if a == 0 and show_label:
        plt.xlabel('Time [ms]', fontsize=12)
        plt.ylabel('Firing rate [spikes/sec]', fontsize=12)
        # plt.legend()
      elif a == 0 and not show_label:
        plt.xlabel(' ')
        plt.ylabel(' ', fontsize=12)
      if not show_label:
        ax.set_xticklabels([])

    if output_dir is not None:
      output_figure_path = os.path.join(
          output_dir, f'{self.session_id}_t{trials_indices[r]}_trial_demo.pdf')
      plt.savefig(output_figure_path, bbox_inches='tight')
      print('Save figure to: ', output_figure_path)
    plt.show()

    if show_warp is False:
      return

    # Time warping function examples.
    gs_kw = dict(width_ratios=[1,1,1])
    fig, axs = plt.subplots(figsize=(6.5, 2), gridspec_kw=gs_kw,
        nrows=1, ncols=3)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.1)
    
    for a, probe in enumerate(probes):
      ax = fig.add_subplot(axs[a])
      ax.tick_params(left=True, labelbottom=True, bottom=True,
                     top=False, labeltop=False)
      sources = np.append(self.f_warp_sources_arc[a,r,c], 0.5) * 1000
      targets = np.append(self.f_warp_targets_arc[a,r,c], 0.5) * 1000
      # Time warping function.
      # log_lmbda_hat_cross = self.linear_time_warping(
      #     spike_train_time_line, self.f_pop_cag[c,a,0], sources, targets,
      #     verbose=True)

      plt.plot([0, 0.5], [0, 0.5], 'k:', lw=0.5)
      plt.plot(sources, targets, 'k-o', ms=3, lw=1)
      # for i in range(len(sources)):
      #   plt.plot([sources[i], sources[i]], [0, targets[i]], 'k:')
      #   plt.plot([0, sources[i]], [targets[i], targets[i]], 'k:')
      #   plt.plot(sources[i], targets[i], 'ko', mfc='none')
      plt.plot([0, 500], [0, 500], ':k')
      if a == 0:
        plt.xlabel('t [ms]', fontsize=12)
        plt.ylabel(r'$\varphi(t)$', fontsize=12)
      else:
        plt.xlabel(' ', fontsize=12)
        plt.ylabel(' ', fontsize=12)
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

      if a == 0:
        ax.set_title(f'Trial={trial_id}  {areas_names[a]}', fontsize=12)
      else:
        ax.set_title(f'{areas_names[a]}', fontsize=12)
      plt.xlim(0, 500)
      plt.ylim(0, 500)
      plt.axis('equal')
    if output_dir is not None:
      output_figure_path = os.path.join(
          output_dir, f'{self.session_id}_t{trials_indices[r]}_trial_phi_demo.pdf')
      plt.savefig(output_figure_path, bbox_inches='tight')
      print('Save figure to: ', output_figure_path)


  @classmethod
  def spike_trains_neg_log_likelihood(
      cls,
      log_lmbd,
      spike_trains):
    """Calculates the log-likelihood of a spike train given log firing rate.

    When it calculates the negative log_likelihood funciton, it assumes that it
    is a function of lambda instead of spikes. So it drops out the terms that
    are not related to the lambda, which is the y! (spikes factorial) term.

    Args:
      log_lmbd: The format can be in two ways.
          timebins 1D array.
          trials x timebins matrix. Different trials have differnet intensity.
              In this case, `spike_trains` and `log_lmbd` have matching rows.
      spike_trains: Trials x timebins matrix.
    """
    num_trials, num_bins = spike_trains.shape
    if num_trials == 0:
      return 0

    log_lmbd = np.array(log_lmbd)
    if len(log_lmbd.shape) == 2:  # Trialwise intensity.
      x, num_bins_log_lmbd = log_lmbd.shape
      if x != num_trials:
        raise ValueError('Number of trials does not match intensity size.')
      if num_bins != num_bins_log_lmbd:
        raise ValueError('The length of log_lmbd should be equal to spikes.')

      # Equivalent to row wise dot product then take the sum.
      nll = - np.sum(spike_trains * log_lmbd)
      nll += np.exp(log_lmbd).sum()
      return nll

    elif len(log_lmbd.shape) == 1:  # Single intensity for all trials.
      num_bins_log_lmbd = len(log_lmbd)
      if num_bins != num_bins_log_lmbd:
        raise ValueError('The length of log_lmbd should be equal to spikes.')
      nll = - np.dot(spike_trains.sum(axis=0), log_lmbd)
      nll += np.exp(log_lmbd).sum() * num_trials
      return nll


  def stack_q(self):
    """Stacks possible `q_arc`, `q_shift1_arc`, `q_shift2_arc`."""
    qall_qrc = np.zeros((self.num_areas * self.num_qs,
                         self.num_trials, self.num_conditions))

    if self.model_feature_type == 'B':
      qall_qrc = self.q_arc.copy()

    elif self.model_feature_type in ['S', 'S1', 'S2']:
      qall_qrc = self.q_shift1_arc.copy()

    elif self.model_feature_type == 'BS':
      for a in range(self.num_areas):
        qall_qrc[a*self.num_qs] = self.q_arc[a]
        qall_qrc[a*self.num_qs+1] = self.q_shift1_arc[a]

    elif self.model_feature_type == 'SS':
      for a in range(self.num_areas):
        qall_qrc[a*self.num_qs] = self.q_shift1_arc[a]
        qall_qrc[a*self.num_qs+1] = self.q_shift2_arc[a]

    elif self.model_feature_type == 'BSS':
      for a in range(self.num_areas):
        qall_qrc[a*self.num_qs] = self.q_arc[a]
        qall_qrc[a*self.num_qs+1] = self.q_shift1_arc[a]
        qall_qrc[a*self.num_qs+2] = self.q_shift2_arc[a]

    return qall_qrc


  def complete_log_likelihood(
      self,
      clist,
      verbose=True):
    """Complete log-likelihood."""
    spike_trains = self.spike_trains
    spike_train_time_line = self.spike_train_time_line
    trials_groups = self.trials_groups
    probes = self.probes

    log_likelihood = 0
    for c, (stimulus_condition_id, trials_table) in enumerate(trials_groups):
      # print(f'Condition: {c}  stimulus_condition_id:{stimulus_condition_id}')
      if c not in clist:
        continue
      log_likelihood += self.complete_log_likelihood_single_condition(c, verbose)

    # sigma_cross_pop for inverse Wishart.
    if self.nu0 > 0:
      log_likelihood += scipy.stats.invwishart.logpdf(
          self.sigma_cross_pop, df=self.nu0, scale=self.phi0)
    # If the model uses NIW for the cross-pop variables, it involves mu.
    # log_likelihood += scipy.stats.multivariate_normal.logpdf(
    #     self.mu_cross_pop, mean=self.mu0, cov=1/self.xi0*self.phi0)

    # Record the sample.
    self.samples.log_likelihood.append(log_likelihood)


  def complete_log_likelihood_single_condition(
      self,
      c,
      verbose=True):
    """Calculates the model's complete log likelihood.

    p(v,h,theta) = p(v | h) p(h). v, visible variables, h, hidden variable.
    theta, parameters.
    """
    spike_trains = self.spike_trains
    spike_train_time_line = self.spike_train_time_line
    trials_groups = self.trials_groups
    probes = self.probes
    sub_group_df = self.sub_group_df_c[c]
    log_likelihood = 0
    trials_indices = trials_groups.get_group(self.map_c_to_cid[c]).index.values

    # f_cross-pop + z
    for a, probe in enumerate(probes):
      units_cross_pop = sub_group_df[
          (sub_group_df['probe'] == probe) &
          (sub_group_df['group_id'] == 0)].index.values
      spikes_ac = spike_trains.loc[units_cross_pop, trials_indices]
      spikes_ac = np.stack(spikes_ac.values.flatten('F'), axis=0)

      # f_cross_pop time warping or baseline drifting. Note it must run
      # time-warping first then baseline drifint, since time-warping function
      # can only take one vector as the input.
      g = 0
      log_lmbda_hat_cross = self.f_pop_cag[c,a,g]

      if self.model_feature_type in ['S', 'S1', 'S2', 'BS', 'SS', 'BSS']:
        sources = np.kron(self.f_warp_sources_arc[a,:,c],
                            np.ones([len(units_cross_pop), 1]))
        targets = np.kron(self.f_warp_targets_arc[a,:,c],
                            np.ones([len(units_cross_pop), 1]))
        log_lmbda_hat_cross = self.linear_time_warping(
            spike_train_time_line, log_lmbda_hat_cross, sources, targets,
            verbose=False)

      if self.model_feature_type in ['B', 'BS', 'BSS']:
        q_offset = np.kron(self.q_arc[a,:,c].reshape(-1, 1),
                           np.ones([len(units_cross_pop), 1]))
        log_lmbda_hat_cross = log_lmbda_hat_cross + q_offset

      # Spike train likelihood Poisson.
      log_likelihood += -self.spike_trains_neg_log_likelihood(
          log_lmbda_hat_cross, spikes_ac)

      # z Catgorical.
      log_likelihood += np.log(self.p_gac[g,a,c]) * len(units_cross_pop)

    # q_arc cross-pop deviation.
    x_ = self.stack_q()
    x_ = x_[:,:,c]
    x_ = x_.T - self.mu_cross_pop
    S = x_.T @ x_
    log_likelihood += scipy.stats.multivariate_normal.logpdf(
        x_, mean=np.zeros(x_.shape[1]), cov=self.sigma_cross_pop).sum()

    # f_local-pop + z
    for a, probe in enumerate(probes):
      g = 1
      active_units = sub_group_df[
          (sub_group_df['probe'] == probe) &
          (sub_group_df['group_id'] == g)].index.values
      spikes_gac1 = spike_trains.loc[active_units, trials_indices]
      if len(spikes_gac1) != 0:
        spikes_gac1 = np.stack(spikes_gac1.values.flatten('F'), axis=0)
        log_likelihood += -self.spike_trains_neg_log_likelihood(
            self.f_pop_cag[c,a,g], spikes_gac1)
        # z Catgorical.
        log_likelihood += np.log(self.p_gac[g,a,c]) * len(active_units)

      g = 2
      idle_units = sub_group_df[
          (sub_group_df['probe'] == probe) &
          (sub_group_df['group_id'] == g)].index.values
      spikes_gac2 = spike_trains.loc[idle_units, trials_indices]
      if len(spikes_gac2) != 0:
        spikes_gac2 = np.stack(spikes_gac2.values.flatten('F'), axis=0)
        log_likelihood += -self.spike_trains_neg_log_likelihood(
            self.f_pop_cag[c,a,g], spikes_gac2)
         # z Catgorical.
        log_likelihood += np.log(self.p_gac[g,a,c]) * len(idle_units)

    # p_gac Dirichlet.
    log_likelihood += (self.alpha - 1) * np.sum(np.log(self.p_gac))

    # beta_cross_pop, beta_local_pop prior.
    for key in self.f_pop_par_cag.keys():
      c,a,g,par_type = key
      # We only impose prior on beta. Group 2 has constant baseline.
      if par_type != 'beta' or g == 2:
        continue
      beta = self.f_pop_par_cag[key]
      # If it is constant fit, then no need prior.
      if self.f_Omega is not None and len(beta) == self.f_Omega.shape[0]:
        log_prior = -self.eta_smooth_tuning * beta.T @ self.f_Omega @ beta
        log_likelihood += np.asscalar(log_prior)

    return log_likelihood


  def fit_q_arc(
      self,
      spikes,
      time_line,
      a, r, c,
      log_lambda_offset,
      learning_rate=0.5,
      max_num_iterations=100,
      verbose=True):
    """The beta is fitted using Newton's method gradient descent.

    Args:
      spikes: num_trials x num_spike_bins
      basis: num_samples x num_basis
      log_lambda_offset: The offset is an additional predictor variable, but
          with a coefficient value fixed at 1, which means we do not optimize
          over it. The format can be:
          num_spike_bins: 1-D array. non-constant baseline. For every trial.
          constant: scalar. For every trial.
          num_trials x 1: different constant offset for every trial.
          num_trials x num_spike_bins: different non-constant offsets for
              different trials.
    """
    num_trials, num_spike_bins = spikes.shape
    spikes_cum = spikes.sum(axis=0)
    log_lambda_offset = np.array(log_lambda_offset)

    basis = np.zeros((num_spike_bins, 1))
    Omega = np.zeros((1, 1))

    num_samples, num_basis = basis.shape

    if num_spike_bins != num_samples:
      raise ValueError(
          'The length of the basis should be the same as that of spikes.')

    # Initial with the current values.
    qall_qrc = self.stack_q()[:,r,c]
    q_cnd = self.q_arc[a,r,c].copy()
    log_lambda_hat = q_cnd + log_lambda_offset
    nll = self.spike_trains_neg_log_likelihood(log_lambda_hat, spikes)
    nll_old = float("inf")

    # aid, acid is used to index cross-pop parameters.
    aid = a * self.num_qs + 0
    # Determine the other area index.
    acid = np.setdiff1d(np.arange(self.num_areas * self.num_qs), a)
    theta_cross_pop = np.linalg.inv(self.sigma_cross_pop)

    for iter_index in range(max_num_iterations):
      eta = q_cnd + log_lambda_offset
      mu = np.exp(eta).reshape(-1, num_spike_bins)
      mu = mu * num_trials

      gradient_prior = (
          theta_cross_pop[aid, aid] * (q_cnd - self.mu_cross_pop[aid])
          + theta_cross_pop[aid, acid] @ (qall_qrc[acid] - self.mu_cross_pop[acid]))

      hessian_prior = theta_cross_pop[aid, aid]

      gradient_baseline = - spikes_cum.sum() + mu.sum() + gradient_prior
      hessian_baseline = mu.sum() + hessian_prior

      beta_baseline_delta = gradient_baseline / hessian_baseline

      # Backtracking line search.
      ALPHA = 0.4
      BETA = 0.2
      while True:
        beta_baseline_tmp = q_cnd - learning_rate * beta_baseline_delta
        log_lambda_tmp = beta_baseline_tmp + log_lambda_offset
        nll_left = self.spike_trains_neg_log_likelihood(log_lambda_tmp, spikes)
        nll_right = nll - ALPHA * learning_rate * (
              gradient_baseline * beta_baseline_delta)

        if (nll_left > nll_right or
            np.isnan(nll_left) or
            np.isnan(nll_right)):
          learning_rate *= BETA
          # if verbose:
          #   print('update learning_rate: ', learning_rate)
        else:
          break

      if iter_index == max_num_iterations - 1:
        print('Warning: Reaches maximum number of iterations.')
      q_cnd = beta_baseline_tmp

      nll = nll_left
      if iter_index % 100 == 0 and verbose:
        print(iter_index, nll)
      # Check convergence.
      if abs(nll - nll_old) < 1e-9:
        break
      nll_old = nll

    if verbose:
      print('Total iterations:', iter_index)
      print('update q_car:', q_cnd)

    log_lambda_hat = q_cnd + log_lambda_offset
    return log_lambda_hat, (q_cnd, log_lambda_offset)


  @classmethod
  def linear_time_warping(
      cls,
      t,
      f,
      sources,
      targets,
      verbose=True):
    """It handles multiple time warping."""
    sources = np.array(sources)
    targets = np.array(targets)
    if len(sources.shape) == 1:
      f_warp = cls.linear_time_warping_single(
          t, f, sources, targets, verbose=verbose)

    elif len(sources.shape) == 2:
      num_trials = sources.shape[0]
      f_warp = np.zeros((num_trials, len(t)))
      for ind in range(num_trials):
        f_warp[ind] = cls.linear_time_warping_single(
            t, f, sources[ind], targets[ind], verbose=verbose)

    return f_warp


  @classmethod
  def linear_time_warping_single(
      cls,
      t,
      f,
      sources,
      targets,
      output_file=None,
      verbose=True):
    """Time warping function for the intensity.

    Args:
      sources: Positions of input `f` needed to be shifted.
      targets: New positions of the sources. The rest of curve will be shifted
          linearly in between sources.
    """
    sources = np.array(sources)
    targets = np.array(targets)
    t_interp = t.copy()

    for i in range(1, len(sources)):
      source_left = sources[i-1]
      source_right = sources[i]
      target_left = targets[i-1]
      target_right = targets[i]

      # Linearly stretch the source intervals to the target interverals.
      t_target_index = (t >= target_left) & (t < target_right)
      t_target = t[t_target_index]
      if len(t_target) == 0:
        continue
      t_interp[t_target_index] = ((t_target - target_left) *
          (source_right - source_left) / (target_right - target_left)
          + source_left)

    # Run the linear interporation using the sample points.
    f_warp = np.interp(t_interp, t, f)

    if verbose:
      plt.figure(figsize=(7.5, 3.25))
      plt.subplot(121)
      plt.plot(sources, targets, 'k-o', ms=3, lw=1)
      plt.plot([sources[2], sources[2]], [0, targets[2]], 'k:')
      plt.plot([0, sources[2]], [targets[2], targets[2]], 'k:')
      plt.plot([0, 500], [0, 500], ':k')
      plt.xlabel('Time [ms]', fontsize=12)
      plt.ylabel(r'$\varphi(t)$', fontsize=12)
      # plt.xlim(0, 500)
      # plt.ylim(0, 500)
      # plt.axis('equal')

      plt.subplot(122)
      plt.plot(t, f, 'k:', label='origianl curve')
      plt.plot(t, f_warp, 'k', label='warped curve')
      # plt.grid('on')
      plt.legend(loc="upper left")
      plt.xlabel('Time [ms]', fontsize=12)
      plt.ylim(-2, 2)
      if output_file is not None:
        plt.savefig(output_file, bbox_inches='tight')
        print('save figure:', output_file)

      plt.show()

    return f_warp

  @classmethod
  def linear_time_warping_demo(
      cls,
      t=None,
      f=None,
      output_file=None):
    """This is an demo for the paper.

    This demo shows the time warping method is not sensitve to landmarks in a
    certain range.
    """
    if t is None and f is None:
      t = np.arange(0, 501, 0.01)
      f = np.sin(t*0.04 + 500)
      ylim = (-1.5, 2.5)
      sources1 = [0, 150, 240, 400, 500]
      targets1 = [0, 150, 210, 400, 500]
      sources2 = [0, 150, 280, 400, 500]
      targets2 = [0, 150, 250, 400, 500]
    else:
      shift = 30
      p1_s, p2_s = 233, 215
      p1_t, p2_t = p1_s + shift, p2_s + shift
      sources1 = [0, 150, p1_s, 400, 500]
      targets1 = [0, 150, p1_t, 400, 500]
      sources2 = [0, 150, p2_s, 400, 500]
      targets2 = [0, 150, p2_t, 400, 500]
      ylim = (0, 100)

    f_warp1 = cls.linear_time_warping_single(t, f, sources1, targets1,
        verbose=False)
    f_warp2 = cls.linear_time_warping_single(t, f, sources2, targets2,
        verbose=False)

    # gs_kw = dict(width_ratios=[1,2], height_ratios=[1,1])
    # fig, axs = plt.subplots(figsize=(12, 4.5), gridspec_kw=gs_kw,
    #     nrows=2, ncols=2)

    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(ncols=2, nrows=2, width_ratios=[1,3],
        height_ratios=[1,1])
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.15)
    ax = fig.add_subplot(gs[0,0])
    ax.tick_params(labelbottom=False)
    plt.plot(sources1, targets1, 'k-o', ms=3, lw=1)
    plt.plot([sources1[2], sources1[2]], [0, targets1[2]], 'k:')
    plt.plot([0, sources1[2]], [targets1[2], targets1[2]], 'k:')

    plt.plot([0, 500], [0, 500], ':k')
    # plt.xlabel('Time [ms]', fontsize=12)
    # plt.ylabel(r'$\varphi(t)$', fontsize=12)
    # plt.xlim(0, 500)
    # plt.ylim(0, 500)
    # plt.axis('equal')
    # plt.title(r'$f_1$')
    plt.text(0.1, 0.8, r'$f_1$', transform=ax.transAxes)

    ax = fig.add_subplot(gs[1,0])
    plt.plot(sources2, targets2, 'b-o', ms=3, lw=1)
    plt.plot([sources2[2], sources2[2]], [0, targets2[2]], 'k:')
    plt.plot([0, sources2[2]], [targets2[2], targets2[2]], 'k:')
    plt.plot([0, 500], [0, 500], ':k')
    plt.xlabel('Time [ms]', fontsize=12)
    # plt.ylabel(r'$\varphi(t)$', fontsize=12)
    # plt.xlim(0, 500)
    # plt.ylim(0, 500)
    # plt.axis('equal')
    # plt.title(r'$f_2$')
    plt.text(0.1, 0.8, r'$f_2$', transform=ax.transAxes)
    plt.ylabel(r'$\varphi(t)$', fontsize=12)

    ax = fig.add_subplot(gs[:,1])
    plt.plot(t, f, 'k:', label='origianl')
    plt.plot(t, f_warp1, 'k', label='$f_1$')
    plt.plot(t, f_warp2, 'b', label='$f_2$')
    # plt.grid('on')
    plt.legend(loc='upper left', ncol=3)
    plt.xlabel('Time [ms]', fontsize=12)
    plt.ylim(ylim)
    plt.close()

    fig = plt.figure(figsize=(6, 3))
    plt.plot(t, f, 'k:', label='origianl')
    plt.plot(t, f_warp1, 'k', lw=2, label='method 1')
    plt.plot(t, f_warp2, 'b', lw=2, label='method 2')
    # plt.grid('on')
    plt.legend(loc='upper left', ncol=3)
    plt.xlabel('Time [ms]', fontsize=12)
    plt.ylabel('Firing rate [spikes/s]', fontsize=12)
    plt.ylim(ylim)
    plt.arrow(p1_s, 70, shift, 0, head_width=4, color='k')
    plt.arrow(p2_s, 76, shift, 0, head_width=4, color='b')
    plt.plot([p1_s], [70], 'k.')
    plt.plot([p2_s], [76], '.', c='b')

    if output_file is not None:
      plt.savefig(output_file, bbox_inches='tight')
      print('save figure:', output_file)
    plt.show()


  @classmethod
  def spikes_linear_time_warping(
      cls,
      spike_times,
      spike_train_time_line,
      trials_indices,
      sources,
      targets,
      verbose=False):
    """Time warping function for spike_times DataFrame.

    Args:
      spike_times: A DataFrame for spike times. The rows are neurons, the
          columns are trials.
      sources: sources are in the same order of `trials_indices`.
      targets: targets are in the same order of `trials_indices`.

    Returns:
      spikes: The numpy matrix (spike trian trials x time bins).
    """
    # Loop for each experiment trial.
    dt = spike_train_time_line[1] - spike_train_time_line[0]
    num_neurons = spike_times.shape[0]
    num_trials = len(trials_indices)
    num_dts = len(spike_train_time_line)
    spike_trains = np.zeros([num_neurons * num_trials, num_dts])

    # spike_times_out = spike_times.copy()
    for t, trial in enumerate(trials_indices):
      source = sources[t]
      target = targets[t]

      # Loop for each spike train.
      for n, neuron in enumerate(spike_times.index):
        # The .copy() is critical, other with it will change the original data
        # IN PLACE, since .loc gets the reference.
        spikes = spike_times.loc[neuron, trial].copy()
        spikes_ = spikes.copy()
        if len(spikes) == 0:
          continue

        # Loop for each warping sessions.
        for s in range(1, len(source)):
          source_left = source[s-1]
          source_right = source[s]
          target_left = target[s-1]
          target_right = target[s]

          ind_range = ((spikes >= source_left) &
                       (spikes < source_right))
          if not any(ind_range):
            continue
          spikes_[ind_range] = (
              (spikes[ind_range] - source_left) *
              (target_right - target_left) /
              (source_right - source_left) + target_left)

        spikes_ind = np.floor(spikes_ // dt).astype(int)
        # If the spikes are shifted out, then remove them.
        spikes_ind = spikes_ind[(spikes_ind >= 0) & (spikes_ind < num_dts)]
        spike_trains[n + t*num_neurons, spikes_ind] = 1

    if verbose:
      plt.figure()
      plt.plot(spike_train_time_line, spike_trains.mean(axis=0), 'gray')
      plt.title('PSTH')
      plt.show()

    return spike_trains


  def fit_time_warping(
      self,
      spikes,
      log_lambda,
      source,
      spike_train_time_line,
      search_lim=[-0.1, 0.1],
      step_size=0.002,
      verbose=False):
    """Fits the single point time warping by grid search."""
    search_left = max(source[0] - source[1], search_lim[0])
    search_right = min(source[2] - source[1], search_lim[1])

    shifts = np.linspace(search_left, search_right,
        int((search_right - search_left + step_size) / step_size))

    nlls = np.zeros(len(shifts))
    for index, shift in enumerate(shifts):
      target = source.copy()
      target[1] = shift + source[1]
      log_lambda_cnd = self.linear_time_warping(
          spike_train_time_line, log_lambda,
          source, target, verbose=False)
      nlls[index] = self.spike_trains_neg_log_likelihood(
            log_lambda_cnd, spikes)

    best_match_shift = shifts[np.argmin(nlls)]
    return best_match_shift

  def fit_q_shift_arc(
      self,
      spikes,
      spike_train_time_line,
      a, r, c,
      log_lambda,
      source,
      search_lim1=[-0.05, 0.05],
      search_lim2=[-0.1, 0.1],
      step_size=0.002,
      verbose=False):
    """Fit the q_shift_arc by grid search.

    The grid search for each `a` can be understood as the coordinate descent.
    Pay attention to the step size, if it is too large, it can overshoot.
    """
    q_cnd = self.stack_q()[:,r,c]
    if self.model_feature_type in ['S', 'S1', 'S2']:
      shift1_prev = self.q_shift1_arc[a,r,c]
      aid1 = a * self.num_qs + 0
    if self.model_feature_type in ['BS']:
      shift1_prev = self.q_shift1_arc[a,r,c]
      aid1 = a * self.num_qs + 1
    if self.model_feature_type in ['SS']:
      shift1_prev = self.q_shift1_arc[a,r,c]
      shift2_prev = self.q_shift2_arc[a,r,c]
      aid1 = a * self.num_qs + 0
      aid2 = a * self.num_qs + 1
    if self.model_feature_type in ['BSS']:
      shift1_prev = self.q_shift1_arc[a,r,c]
      shift2_prev = self.q_shift2_arc[a,r,c]
      aid1 = a * self.num_qs + 1
      aid2 = a * self.num_qs + 2

    # First time warping match.
    search_left = max(source[0] - source[1], search_lim1[0])
    search_right = min(source[2] - source[1], search_lim1[1])
    shifts = np.linspace(search_left, search_right,
        int((search_right - search_left + step_size) / step_size))
    log_likelihoods = np.zeros(len(shifts))
    log_priors = np.zeros(len(shifts))
    for index, shift in enumerate(shifts):
      target = source.copy()
      target[1] = shift + source[1]
      log_lambda_cnd = self.linear_time_warping(
          spike_train_time_line, log_lambda,
          source, target, verbose=False)
      log_likelihoods[index] = - self.spike_trains_neg_log_likelihood(
          log_lambda_cnd, spikes)

      # Prior.
      q_cnd[aid1] = shift
      log_prior = scipy.stats.multivariate_normal.logpdf(
          q_cnd, mean=self.mu_cross_pop, cov=self.sigma_cross_pop)
      # log_prior = scipy.stats.multivariate_normal.logpdf(
      #     q_cnd, mean=self.mu_cross_pop, cov=self.phi0*3)
      log_priors[index] = log_prior

    best_match_shift1 = shifts[np.argmax(log_likelihoods + log_priors)]
    best_match_shift1 = 0.2 * shift1_prev + 0.8 * best_match_shift1
    target = source.copy()
    target[1] = best_match_shift1 + source[1]
    log_lambda_cnd = self.linear_time_warping(
        spike_train_time_line, log_lambda, source, target, verbose=False)

    if verbose:
      plt.figure(figsize=(12,3))
      plt.subplot(121)
      plt.plot(shifts, log_likelihoods + log_priors, 'g', label='log-posterior')
      plt.plot(best_match_shift1, max(log_likelihoods + log_priors), 'x')
      plt.plot(shifts, log_likelihoods, 'r', label='log-likelihood')
      plt.title(f'a:{a} r:{r} c:{c} p:1')
      plt.legend(loc="upper left")
      plt.grid('on')
      plt.subplot(122)
      plt.plot(spike_train_time_line, np.mean(spikes, axis=0), 'gray')
      plt.plot(spike_train_time_line, np.exp(log_lambda), 'k')
      plt.plot(spike_train_time_line, np.exp(log_lambda_cnd), 'g')
      plt.grid('on')

    # Second time warping match.
    if self.model_feature_type not in ['BSS', 'SS']:
      return log_lambda_cnd, (best_match_shift1, target)

    search_left = max(source[3] - source[4], search_lim2[0])
    search_right = min(source[5] - source[4], search_lim2[1])
    shifts = np.linspace(search_left, search_right,
        int((search_right - search_left + step_size) / step_size))
    log_likelihoods = np.zeros(len(shifts))
    log_priors = np.zeros(len(shifts))
    for index, shift in enumerate(shifts):
      target = source.copy()
      target[4] = shift + source[4]
      log_lambda_cnd = self.linear_time_warping(
          spike_train_time_line, log_lambda, source, target, verbose=False)
      log_likelihoods[index] = - self.spike_trains_neg_log_likelihood(
          log_lambda_cnd, spikes)

      # Prior.
      q_cnd[aid2] = shift
      log_prior = scipy.stats.multivariate_normal.logpdf(
          q_cnd, mean=self.mu_cross_pop, cov=self.sigma_cross_pop)
      # log_prior = scipy.stats.multivariate_normal.logpdf(
      #     q_cnd, mean=self.mu_cross_pop, cov=self.phi0*3)
      log_priors[index] = log_prior

    best_match_shift2 = shifts[np.argmax(log_likelihoods + log_priors)]
    best_match_shift2 = 0.2 * shift2_prev + 0.8 * best_match_shift2

    # Put two shifts together.
    target = source.copy()
    target[1] = best_match_shift1 + source[1]
    target[4] = best_match_shift2 + source[4]
    log_lambda_cnd = self.linear_time_warping(
        spike_train_time_line, log_lambda, source, target, verbose=False)

    if verbose:
      plt.figure(figsize=(12,3))
      plt.subplot(121)
      plt.plot(shifts, log_likelihoods + log_priors, 'g', label='log-posterior')
      plt.plot(best_match_shift2, max(log_likelihoods + log_priors), 'x')
      plt.plot(shifts, log_likelihoods, 'r', label='log-likelihood')
      plt.title(f'a:{a} r:{r} c:{c} p:2')
      plt.grid('on')
      plt.legend(loc="upper left")
      plt.subplot(122)
      plt.plot(spike_train_time_line, np.mean(spikes, axis=0), 'gray')
      plt.plot(spike_train_time_line, np.exp(log_lambda), 'k')
      plt.plot(spike_train_time_line, np.exp(log_lambda_cnd), 'g')
      plt.grid('on')

    return log_lambda_cnd, ([best_match_shift1, best_match_shift2], target)


  def draw_q_cnd(
      self,
      c,
      r,
      q_prev,
      proposal_scalar=0.05,
      sample_type ='truncated',
      max_iter=100,
      verbose=False):
    """Draw samples in truncated distributions.

    Args:
      sample_type: 'truncated', truncate the distribution within a range.
                   'clipped', clip the samples values outside range.
                   'none', no trimming.
    """
    # Adjust `proposal_scalar` if a trial has low accept rate. This adjustment
    # is done for every trial every condition.
    if not hasattr(self, 'q_proposal_scalar_rc'):
      self.q_proposal_scalar_rc = proposal_scalar * np.ones([
          self.num_trials, self.num_conditions])

    if (self.samples.q_sample_cnt_rc[r,c] > 200 and
        self.samples.q_sample_cnt_rc[r,c] % 200 == 0):
      accept_ratio = (self.samples.q_sample_accept_cnt_rc[r,c] /
                      self.samples.q_sample_cnt_rc[r,c])
      if accept_ratio < 0.1:
        self.q_proposal_scalar_rc[r,c] = max(
            self.q_proposal_scalar_rc[r,c] / 2, proposal_scalar/15)
        if verbose:
          print(f'q accrt r{r}c{c} {accept_ratio} ' + 
              f'cnt{self.samples.q_sample_cnt_rc[r,c]} ' +
              f'scale{self.q_proposal_scalar_rc[r,c]}')
      if accept_ratio > 0.8:
        self.q_proposal_scalar_rc[r,c] = min(
            self.q_proposal_scalar_rc[r,c] * 2, proposal_scalar*10)
        if verbose:
          print(f'q accrt r{r}c{c} {accept_ratio} ' + 
              f'cnt{self.samples.q_sample_cnt_rc[r,c]} ' +
              f'scale{self.q_proposal_scalar_rc[r,c]}')

    self.q_proposal_scalar_rc = proposal_scalar * np.ones([
        self.num_trials, self.num_conditions])

    # Setup the proposal cov.
    if self.samples.q_sample_cnt_rc[0,0] < 100:
      # proposal_cov = (self.q_proposal_scalar_rc[r,c] *
      #     np.diag(np.diag(self.sigma_cross_pop)))
      self.q_propose_cov = self.sigma_cross_pop
      proposal_cov = self.q_proposal_scalar_rc[r,c] * self.sigma_cross_pop
    elif self.samples.q_sample_cnt_rc[0,0] % 200 == 100:
      self.q_propose_cov = self.sigma_cross_pop
      proposal_cov = self.q_proposal_scalar_rc[r,c] * self.q_propose_cov
    else:
      proposal_cov = self.q_proposal_scalar_rc[r,c] * self.q_propose_cov

    # Draw a sample here.
    if self.model_feature_type in ['B']:
      q_cnd = np.random.multivariate_normal(q_prev, proposal_cov)
      return q_cnd, proposal_cov
    if sample_type == 'none':
      q_cnd = np.random.multivariate_normal(q_prev, proposal_cov)
    elif sample_type == 'clipped':
      q_cnd = np.random.multivariate_normal(q_prev, proposal_cov)
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

        f_peak1 = self.f_peak1_ac[a,c]
        q_cnd[qid1] = np.clip(f_peak1 + q_cnd[qid1], a_min=0.01, a_max=0.14) - f_peak1
        # q_cnd[qid1] = np.clip(q_cnd[qid1], a_min=-0.05, a_max=0.05)
        f_peak2 = self.f_peak2_ac[a,c]
        q_cnd[qid2] = np.clip(f_peak2 + q_cnd[qid2], a_min=0.15, a_max=0.4) - f_peak2

    elif sample_type == 'truncated':
      for itr in range(max_iter):
        q_cnd = np.random.multivariate_normal(q_prev, proposal_cov)
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
          # Peak-1
          f_peak1 = self.f_peak1_ac[a,c]
          f_peak1_cnd = q_cnd[qid1] + f_peak1
          if f_peak1_cnd < 0.01 or f_peak1_cnd > 0.14:
            quanlified = False
          # Peak-2
          f_peak2 = self.f_peak2_ac[a,c]
          f_peak2_cnd = q_cnd[qid2] + f_peak2
          if f_peak2_cnd < 0.15 or f_peak2_cnd > 0.4:
            quanlified = False
        if quanlified:
          break
      if itr == max_iter-1 and not quanlified:
        print(f'Warning: Drawing q_cnd samples runs out of iterations c={c} r={r}')
        # print(q_prev)

    return q_cnd, proposal_cov


  def update_q_arc(
      self,
      c,
      sample_type='fit',
      fit_peak_ratio=0,
      proposal_scalar=0.05,
      record=True,
      verbose=True):
    """Updates q_arc.

    In the following scenarios, we subtract the q means (calculated from the
    current values) to constrain the identifiability problem. The mean is
    averaged over all trials.

    Args:
        sample_type: 'fit' or 'sample'
    """
    spike_trains = self.spike_trains
    spike_train_time_line = self.spike_train_time_line
    probes = self.probes
    trials_groups = self.trials_groups
    sub_group_df = self.sub_group_df_c[c]
    trials_indices = trials_groups.get_group(self.map_c_to_cid[c]).index.values
    dt = self.dt

    for r, trial_id in enumerate(trials_indices):

      # Prepare for MH sampling.
      if sample_type == 'sample':
        # Gaussian proposal distribution.
        q_prev = self.stack_q()[:,r,c]
        q_cnd, q_proposal_cov = self.draw_q_cnd(c, r, q_prev,
            proposal_scalar=proposal_scalar, sample_type='truncated')
        shrinker = np.zeros_like(q_cnd)

        # Note that do NOT use [[]]*n. This is not the correct initialization.
        # It doesn't create multiple empty liest, but different references to
        # the same instance.
        sources = [[] for i in range(self.num_areas)]
        targets = [[] for i in range(self.num_areas)]
        log_likelihood_cnd = 0
        log_likelihood_prev = 0

      for a, probe in enumerate(probes):

        log_lambda = self.f_pop_cag[c,a,0]
        units_cross_pop = sub_group_df[
            (sub_group_df['probe'] == probe) &
            (sub_group_df['group_id'] == 0)].index.values

        spikes_nrc = spike_trains.loc[units_cross_pop, trial_id]
        spikes_nrc = np.stack(spikes_nrc.values.flatten('F'), axis=0)

        ############################# Fit methods #############################
        log_lambda_cnd = log_lambda.copy()
        if (sample_type == 'fit' and
            self.model_feature_type in ['S', 'S1', 'S2', 'BS1']):

          if self.model_feature_type in ['S', 'S1']:
            if fit_peak_ratio > 0:
              self.f_peak1_ac[a,c] = (
                  (1 - fit_peak_ratio) * self.f_peak1_ac[a,c] +
                  fit_peak_ratio * self.find_peak(
                      [0.04, 0.1], spike_train_time_line, log_lambda))
            source = [0, self.f_peak1_ac[a,c], 0.13]
            search_lim1 = [-0.05, 0.05]
          if self.model_feature_type in ['S2', 'BS2']:
            if fit_peak_ratio > 0:
              self.f_peak2_ac[a,c] = (
                  (1 - fit_peak_ratio) * self.f_peak2_ac[a,c] +
                  fit_peak_ratio * self.find_peak(
                      [0.18, 0.25], spike_train_time_line, log_lambda))
            source = [0.13, self.f_peak2_ac[a,c], 0.4]
            search_lim1 = [-0.05, 0.05]

          log_lambda_cnd, (shift_hat, target) = self.fit_q_shift_arc(
              spikes_nrc, spike_train_time_line, a, r, c, log_lambda, source,
              search_lim1=search_lim1, verbose=verbose)
          # Subtract the mean to avoid identifiability. The coefficient before
          # the mean subtraction is to avoid too strong change.
          self.q_shift1_arc[a,r,c] = (
              shift_hat - 0.3 * self.q_shift1_mean_ac[a,c])
          target[1] = self.q_shift1_arc[a,r,c] + source[1]
          self.f_warp_sources_arc[a,r,c] = source.copy()
          self.f_warp_targets_arc[a,r,c] = target.copy()

        if sample_type == 'fit' and self.model_feature_type in ['SS', 'BSS']:
          if fit_peak_ratio > 0:
            self.f_peak1_ac[a,c] = (
                (1 - fit_peak_ratio) * self.f_peak1_ac[a,c] +
                fit_peak_ratio * self.find_peak(
                    [0.04, 0.1], spike_train_time_line, log_lambda))
            self.f_peak2_ac[a,c] = (
                (1 - fit_peak_ratio) * self.f_peak2_ac[a,c] +
                fit_peak_ratio * self.find_peak(
                    [0.18, 0.25], spike_train_time_line, log_lambda))
          source = [0, self.f_peak1_ac[a,c], 0.13, 0.13, self.f_peak2_ac[a,c], 0.4]
          search_lim1, search_lim2 = [-0.05, 0.05], [-0.05, 0.05]
          log_lambda_cnd, (shift_hat, target) = self.fit_q_shift_arc(
              spikes_nrc, spike_train_time_line, a, r, c, log_lambda, source,
              search_lim1=search_lim1, search_lim2=search_lim2, verbose=verbose)
          # Subtract the mean to avoid identifiability. The coefficient before
          # the mean subtraction is to avoid too strong change.
          self.q_shift1_arc[a,r,c] = (shift_hat[0] -
              0.3 * self.q_shift1_mean_ac[a,c])
          self.q_shift2_arc[a,r,c] = (shift_hat[1] -
              0.95 * self.q_shift2_mean_ac[a,c])
          target[1] = self.q_shift1_arc[a,r,c] + source[1]
          target[4] = self.q_shift2_arc[a,r,c] + source[4]
          self.f_warp_sources_arc[a,r,c] = source.copy()
          self.f_warp_targets_arc[a,r,c] = target.copy()

        if (sample_type == 'fit' and
            self.model_feature_type in ['B', 'BS', 'BS1', 'BS2', 'BSS']):
          _, (q_arc_hat, _) = self.fit_q_arc(
              spikes_nrc, spike_train_time_line, a, r, c,
              log_lambda_offset=log_lambda_cnd, verbose=verbose)
          # Subtract the mean to avoid identifiability. The coefficient before
          # the mean subtraction is to avoid too strong change.
          self.q_arc[a,r,c] = (
              q_arc_hat - 0.3 * self.q_mean_ac[a,c])

        ########################### Sampling methods ###########################
        if (sample_type == 'sample' and
            self.model_feature_type in ['S', 'S1', 'S2', 'BS1', 'BS2']):

          if self.model_feature_type in ['S', 'S1', 'S2']:
            qid = a * self.num_qs + 0
          elif self.model_feature_type in ['BS1', 'BS2']:
            qid = a * self.num_qs + 1

          if self.model_feature_type in ['S', 'S1', 'BS1']:
            if fit_peak_ratio > 0:
              self.f_peak1_ac[a,c] = (
                  (1 - fit_peak_ratio) * self.f_peak1_ac[a,c] +
                  fit_peak_ratio * self.find_peak(
                      [0.04, 0.09], spike_train_time_line, log_lambda))

            # First peak.
            f_peak = self.f_peak1_ac[a,c]
            # q_cnd[qid] = np.clip(q_cnd[qid], a_min=-0.05, a_max=0.05)
            q_cnd[qid] = np.clip(f_peak + q_cnd[qid], a_min=0.01, a_max=0.14) - f_peak
            q_cnd[qid] -= self.mean_correct_ratio_s1 * self.q_shift1_mean_ac[a,c]
            shrinker[qid] = self.mean_correct_ratio_s1 * self.q_shift1_mean_ac[a,c]
            sources[a].extend([0, f_peak, 0.15])
            targets[a].extend([0, f_peak + q_cnd[qid], 0.15])

          if self.model_feature_type in ['S2', 'BS2']:
            if fit_peak_ratio > 0:
              self.f_peak2_ac[a,c] = (
                  (1 - fit_peak_ratio) * self.f_peak2_ac[a,c] +
                  fit_peak_ratio * self.find_peak(
                      [0.18, 0.25], spike_train_time_line, log_lambda))

            # Second peak.
            f_peak = self.f_peak2_ac[a,c]
            q_cnd[qid] = np.clip(f_peak + q_cnd[qid], a_min=0.15, a_max=0.4) - f_peak
            q_cnd[qid] -= self.mean_correct_ratio_s1 * self.q_shift1_mean_ac[a,c]
            shrinker[qid] = self.mean_correct_ratio_s1 * self.q_shift1_mean_ac[a,c]
            sources[a].extend([0.15, f_peak, 0.4])
            targets[a].extend([0.15, f_peak + q_cnd[qid], 0.4])

          if self.model_feature_type in ['Sw']:
            # Whole trial shift.
            q_cnd[qid] -= self.mean_correct_ratio_s1 * self.q_shift1_mean_ac[a,c]
            shrinker[qid] = self.mean_correct_ratio_s1 * self.q_shift1_mean_ac[a,c]
            sources[a] = [spike_train_time_line[0],
                          spike_train_time_line[0],
                          spike_train_time_line[-1] + dt]
            targets[a] = [spike_train_time_line[0] + q_cnd[qid],
                          spike_train_time_line[0] + q_cnd[qid],
                          spike_train_time_line[-1] + dt + q_cnd[qid]]

        if sample_type == 'sample' and self.model_feature_type in ['SS', 'BSS']:
          if self.model_feature_type in ['SS']:
            qid1 = a * self.num_qs + 0
            qid2 = a * self.num_qs + 1
          if self.model_feature_type in ['BSS']:
            qid1 = a * self.num_qs + 1
            qid2 = a * self.num_qs + 2

          if fit_peak_ratio > 0:
            self.f_peak1_ac[a,c] = (
                (1 - fit_peak_ratio) * self.f_peak1_ac[a,c] +
                fit_peak_ratio * self.find_peak(
                    [0.04, 0.09], spike_train_time_line, log_lambda))
            self.f_peak2_ac[a,c] = (
                (1 - fit_peak_ratio) * self.f_peak2_ac[a,c] +
                fit_peak_ratio * self.find_peak(
                    [0.18, 0.25], spike_train_time_line, log_lambda))

          # First peak.
          f_peak = self.f_peak1_ac[a,c]
          q_cnd[qid1] = np.clip(f_peak + q_cnd[qid1], a_min=0.01, a_max=0.14) - f_peak
          q_cnd[qid1] -= self.mean_correct_ratio_s1 * self.q_shift1_mean_ac[a,c]
          shrinker[qid1] = self.mean_correct_ratio_s1 * self.q_shift1_mean_ac[a,c]
          sources[a].extend([0, f_peak, 0.15])
          targets[a].extend([0, f_peak + q_cnd[qid1], 0.15])

          # Second peak.
          f_peak = self.f_peak2_ac[a,c]
          q_cnd[qid2] = np.clip(f_peak + q_cnd[qid2], a_min=0.15, a_max=0.4) - f_peak
          q_cnd[qid2] -= self.mean_correct_ratio_s2 * self.q_shift2_mean_ac[a,c]
          shrinker[qid2] = self.mean_correct_ratio_s2 * self.q_shift2_mean_ac[a,c]
          sources[a].extend([0.15, f_peak, 0.4])
          targets[a].extend([0.15, f_peak + q_cnd[qid2], 0.4])

        # Warp the timeline.
        log_lambda_cnd, log_lambda_prev = log_lambda.copy(), log_lambda.copy()
        if (sample_type == 'sample' and
            self.model_feature_type in ['S', 'S1', 'S2', 'BS', 'SS', 'BSS']):
          log_lambda_cnd = self.linear_time_warping(spike_train_time_line,
              log_lambda_cnd, sources[a], targets[a], verbose=False)
          log_lambda_prev = self.linear_time_warping(spike_train_time_line,
              log_lambda_prev, self.f_warp_sources_arc[a,r,c],
              self.f_warp_targets_arc[a,r,c], verbose=False)

        # Drift the baseline.
        if (sample_type == 'sample' and
            self.model_feature_type in ['B', 'BS', 'BSS']):
          qid = a * self.num_qs + 0
          q_cnd[qid] -= self.mean_correct_ratio_q * self.q_mean_ac[a,c]
          shrinker[qid] = self.mean_correct_ratio_q * self.q_mean_ac[a,c]
          log_lambda_cnd = log_lambda_cnd + q_cnd[qid]
          log_lambda_prev = log_lambda_prev + self.q_arc[a,r,c]

        # Spike train likelihood.
        if sample_type == 'sample':
          log_likelihood_cnd += -self.spike_trains_neg_log_likelihood(
              log_lambda_cnd, spikes_nrc)
          log_likelihood_prev += -self.spike_trains_neg_log_likelihood(
              log_lambda_prev, spikes_nrc)

      # Metropolis-Hastings sampling.
      if sample_type == 'sample':
        self.samples.q_sample_cnt_rc[r,c] += 1
        log_likelihood_cnd += scipy.stats.multivariate_normal.logpdf(
            q_cnd, self.mu_cross_pop, self.sigma_cross_pop)
        log_likelihood_prev += scipy.stats.multivariate_normal.logpdf(
            q_prev, self.mu_cross_pop, self.sigma_cross_pop)

        # Warning, be careful with self.q_log_likelihood_rc.
        # if other parameters except for q have changes, the likelihood from
        # previous step is not the correct value.
        # WRONG: mh_ratio = np.exp(
        #    log_likelihood_cnd - self.q_log_likelihood_rc[r,c])
        mh_ratio = np.exp(log_likelihood_cnd - log_likelihood_prev)
        # print('MH ratio', mh_ratio)
        u = np.random.rand()
        # Accept the new candicate, otherwise doesn't change.
        if u < mh_ratio:
          if self.model_feature_type in ['B', 'BS', 'BSS']:
            qids = np.arange(self.num_areas) * self.num_qs + 0
            self.q_arc[:,r,c] = q_cnd[qids]
          if self.model_feature_type in ['S', 'S1', 'S2']:
            self.q_shift1_arc[:,r,c] = q_cnd
          if self.model_feature_type in ['BS']:
            qids = np.arange(self.num_areas) * self.num_qs + 1
            self.q_shift1_arc[:,r,c] = q_cnd[qids]
          if self.model_feature_type in ['SS']:
            qids = np.arange(self.num_areas) * self.num_qs + 0
            self.q_shift1_arc[:,r,c] = q_cnd[qids]
            qids = np.arange(self.num_areas) * self.num_qs + 1
            self.q_shift2_arc[:,r,c] = q_cnd[qids]
          if self.model_feature_type in ['BSS']:
            qids = np.arange(self.num_areas) * self.num_qs + 1
            self.q_shift1_arc[:,r,c] = q_cnd[qids]
            qids = np.arange(self.num_areas) * self.num_qs + 2
            self.q_shift2_arc[:,r,c] = q_cnd[qids]
          # Sources, targets.
          if self.model_feature_type in ['S', 'S1', 'S2', 'BS', 'BSS']:
            self.f_warp_sources_arc[:,r,c] = sources
            self.f_warp_targets_arc[:,r,c] = targets

          self.samples.q_sample_accept_cnt_rc[r,c] += 1

    # Get the means of the iteration.
    if self.model_feature_type in ['B', 'BS', 'BSS']:
      self.q_mean_ac = np.mean(self.q_arc, axis=1)
    if self.model_feature_type in ['S', 'S1', 'S2', 'BS', 'SS', 'BSS']:
      self.q_shift1_mean_ac = np.mean(self.q_shift1_arc, axis=1)
    if self.model_feature_type in ['SS', 'BSS']:
      self.q_shift2_mean_ac = np.mean(self.q_shift2_arc, axis=1)

    # Add to the samples collector.
    if record:
      if self.model_feature_type in ['B', 'BS', 'BSS']:
        self.samples.q.append(self.q_arc.copy())
      if self.model_feature_type in ['S', 'S1', 'S2', 'BS', 'SS', 'BSS']:
        self.samples.q_shift1.append(self.q_shift1_arc.copy())
        self.samples.f_warp_sources.append(self.f_warp_sources_arc.copy())
        self.samples.f_warp_targets.append(self.f_warp_targets_arc.copy())
      if self.model_feature_type in ['SS', 'BSS']:
        self.samples.q_shift2.append(self.q_shift2_arc.copy())


  def update_mu_simga(
      self,
      clist,
      sample_type='fit',
      update_prior_ratio=0,
      record=True,
      verbose=True):
    """Updates sigma using IW.

    Inverse Wishart means we treat the mean as known and fixed, but the
    covariance matrix is unknow and is the variable.
    """
    num_q = self.num_trials * len(clist)
    q_samples = np.zeros([self.num_qs * self.num_areas, num_q])

    # q assignment.
    if self.model_feature_type in ['B', 'BS', 'BSS']:
      qids = np.arange(self.num_areas) * self.num_qs + 0
      q_samples[qids] = (
          self.q_arc[:,:,clist].transpose(0,2,1).reshape(self.num_areas,-1))

    # q_shift_1 assignment.
    if self.model_feature_type in ['S', 'S1', 'S2', 'SS']:
      qids = np.arange(self.num_areas) * self.num_qs + 0
      q_samples[qids] = (
          self.q_shift1_arc[:,:,clist].transpose(0,2,1).reshape(self.num_areas,-1))
    if self.model_feature_type in ['BS', 'BSS']:
      qids = np.arange(self.num_areas) * self.num_qs + 1
      q_samples[qids] = (
          self.q_shift1_arc[:,:,clist].transpose(0,2,1).reshape(self.num_areas,-1))

    # q_shift_2 assignment.
    if self.model_feature_type in ['SS']:
      qids = np.arange(self.num_areas) * self.num_qs + 1
      q_samples[qids] = (
          self.q_shift2_arc[:,:,clist].transpose(0,2,1).reshape(self.num_areas,-1))
    if self.model_feature_type in ['BSS']:
      qids = np.arange(self.num_areas) * self.num_qs + 2
      q_samples[qids] = (
          self.q_shift2_arc[:,:,clist].transpose(0,2,1).reshape(self.num_areas,-1))

    if sample_type == 'fit':
      q_ = q_samples.T - self.mu_cross_pop
      q_cov = 1 / (num_q - 1) * q_.T @ q_
      self.sigma_cross_pop = q_cov

    elif sample_type == 'iw_fit':
      q_ = q_samples.T - self.mu_cross_pop
      nu_ = self.nu0 + num_q
      phi_ = self.phi0 + q_.T @ q_
      q_cov = phi_ / nu_
      self.sigma_cross_pop = q_cov

    elif sample_type == 'iw_sample':
      q_ = q_samples.T - self.mu_cross_pop
      nu_ = self.nu0 + num_q
      phi_ = self.phi0 + q_.T @ q_
      q_cov = scipy.stats.invwishart.rvs(scale=phi_, df=nu_)
      self.sigma_cross_pop = q_cov

    elif sample_type == 'niw_fit':
      q_mean = np.mean(q_samples, axis=1)
      q_ = q_samples.T - q_mean
      q_cov = 1 / (num_q - 1) * q_.T @ q_
      self.mu_cross_pop = q_mean
      self.sigma_cross_pop = q_cov

    elif sample_type == 'niw_sample':
      q_mean = np.mean(q_samples, axis=1)
      q_ = q_samples.T - q_mean
      xi_ = self.xi0 + num_q
      nu_ = self.nu0 + num_q
      mu_ = (self.xi0 * self.mu0 + num_q * q_mean) / xi_
      q_mean_mu0 = (q_mean - self.mu0).reshape(-1, 1)
      phi_ = (self.phi0 + q_.T @ q_ +
          (self.xi0*num_q/xi_) * q_mean_mu0 @ q_mean_mu0.T)
      q_cov = scipy.stats.invwishart.rvs(scale=phi_, df=nu_)
      self.sigma_cross_pop = q_cov
      self.mu_cross_pop = scipy.stats.multivariate_normal.rvs(
          mean=mu_, cov=q_cov/xi_)

    # Update the pior using the current fit.
    self.phi0 = update_prior_ratio * q_cov + (1-update_prior_ratio) * self.phi0

    # Direct calculation for verification.
    q_corrcoef = np.corrcoef(q_samples)

    # Add to the samples collector.
    if record:
      self.samples.mu_cross_pop.append(self.mu_cross_pop.copy())
      self.samples.sigma_cross_pop.append(self.sigma_cross_pop.copy())
      self.samples.rho_cross_pop_simple.append(q_corrcoef)

    if verbose:
      print('mu_cross_pop')
      print(mu)
      print('sigma_cross_pop')
      print(sigma)


  def update_z_ngac(
      self,
      c,
      sample_type='fit',
      record=True,
      verbose=True):
    """Updates group indicator.

    When we take the likelihood about the spike trains, the difference is
    usually very large. So the probability about groups can be ignored. And the
    soft EM is not really necessary, as the likelihood ratio is super large in
    evaluating the point process.

    Args:
      sample_type: 'fit' or 'sample'.
    """
    spike_trains = self.spike_trains
    spike_train_time_line = self.spike_train_time_line
    probes = self.probes
    trials_groups = self.trials_groups
    sub_group_df = self.sub_group_df_c[c]
    trials_indices = trials_groups.get_group(self.map_c_to_cid[c]).index.values
    all_trials = spike_trains.columns.values

    for a, probe in enumerate(probes):
      area_units = sub_group_df[sub_group_df['probe'] == probe]
      area_units_ids = area_units.index.values
      z_nll = np.zeros((len(area_units_ids), 3))

      for n in range(len(area_units_ids)):
        # This is only for speeding up. ':' is faster.
        if len(trials_indices) == len(all_trials):
          spikes_narc = spike_trains.loc[area_units_ids[n], :]
        else:
          spikes_narc = spike_trains.loc[area_units_ids[n], trials_indices]
        spikes_narc = np.stack(spikes_narc.values.flatten('F'), axis=0)

        # cross-pop group NLL.
        g = 0
        log_lmbd_ar = self.f_pop_cag[c,a,g].copy()
        # Time warping has to be executed first since it only handles 1 vector.
        if self.model_feature_type in ['S', 'S1', 'S2', 'BS', 'SS', 'BSS']:
          sources = self.f_warp_sources_arc[a,:,c]
          targets = self.f_warp_targets_arc[a,:,c]
          log_lmbd_ar = self.linear_time_warping(
              spike_train_time_line, log_lmbd_ar, sources, targets,
          verbose=False)
        if self.model_feature_type in ['B', 'BS', 'BSS']:
          log_lmbd_ar = log_lmbd_ar + self.q_arc[a,:,c].reshape(
              len(trials_indices), 1)
        nll_cross_pop = self.spike_trains_neg_log_likelihood(
            log_lmbd_ar, spikes_narc)
        nll_cross_pop -= np.log(self.p_gac[g,a,c])

        # local-pop group 1 NLL.
        g = 1
        if self.num_groups == 2:
          nll_local_pop_g1 = np.inf
        else:
          nll_local_pop_g1 = self.spike_trains_neg_log_likelihood(
              self.f_pop_cag[c,a,g], spikes_narc)
          nll_local_pop_g1 -= np.log(self.p_gac[g,a,c])

        # local-pop group 2 NLL.
        g = 2
        nll_local_pop_g2 = self.spike_trains_neg_log_likelihood(
            self.f_pop_cag[c,a,g], spikes_narc)
        nll_local_pop_g2 -= np.log(self.p_gac[g,a,c])

        # Put all groups together.
        z_nll[n] = [nll_cross_pop, nll_local_pop_g1, nll_local_pop_g2]

      if sample_type == 'fit':
        z_group_indicator = np.argmin(z_nll, axis=1)
      elif sample_type == 'sample':
        z_prob = np.exp(- z_nll - np.max(- z_nll, axis=1).reshape(-1,1))
        z_prob = z_prob / z_prob.sum(axis=1).reshape(-1,1)
        z_group_indicator = [np.random.choice(3, size=1, p=p_)[0]
                             for p_ in z_prob]

      sub_group_df.loc[area_units_ids, 'group_id'] = z_group_indicator
      if verbose:
        print(area_units['group_id'].values)

    # Add to the samples collector.
    if record:
      z = np.zeros((self.num_conditions, len(sub_group_df)))
      for c in range(self.num_conditions):
        z[c] = self.sub_group_df_c[c]['group_id'].values
      self.samples.z.append(z.copy())


  def update_p_gac(
      self,
      c,
      sample_type='fit',
      record=True,
      verbose=True):
    """Updates p_ac.

    If some group has zero elements, then the update automatically rules that
    out here, since `group_count` has not entry of that cluster, then the
    dimension becomes smaller too. So no need to worry about the whether the
    number of groups is 2 or 3.
    """
    probes = self.probes
    sub_group_df = self.sub_group_df_c[c]
    for a, probe in enumerate(probes):
      area_units = sub_group_df[sub_group_df['probe'] == probe]
      area_units_ids = area_units.index.values
      group_count = sub_group_df.loc[area_units_ids, 'group_id'].value_counts()

      if sample_type == 'fit':
        self.p_gac[group_count.index.values,a,c] = (
            group_count.values / group_count.values.sum())
      elif sample_type == 'sample':
        dir_p = group_count.values + self.alpha
        group_ind = group_count.index.values.astype(int)
        self.p_gac[group_ind,a,c] = (
            np.random.dirichlet(dir_p))

    if record:
      # Add to the samples collector.
      self.samples.p.append(self.p_gac.copy())

    if verbose:
      print('p_gac', self.p_gac)


  def f_pop_spline_fit_demo(
      self,
      g=0,
      a=0,
      c=0,
      spike_train_time_line=None,
      spikes_rt=None,
      show_samples=False):
    """Evaluates the f_pop fit.

    This is maninly used to evaluate the uncertianty of the estimation.
    """
    if spikes_rt is None:
      spike_trains = self.spike_trains
      spike_times = self.spike_times
      spike_train_time_line = self.spike_train_time_line
      dt = self.dt
      probes = self.probes
      trials_groups = self.trials_groups
      probe = self.probes[a]
      sub_group_df = self.sub_group_df_c[c]
      trials_indices = trials_groups.get_group(self.map_c_to_cid[c]).index.values

      units_cross_pop = sub_group_df[
          (sub_group_df['probe'] == probe) &
          (sub_group_df['group_id'] == g)].index.values
      spikes_nrc = spike_trains.loc[units_cross_pop, trials_indices]
      spikes_nrc = np.stack(spikes_nrc.values.flatten('F'), axis=0)
      print(spikes_nrc.shape)

      q_offset = np.kron(self.q_arc[a,:,c].reshape(-1, 1),
                         np.ones([len(units_cross_pop), 1]))
      sources = self.f_warp_targets_arc[a,:,c]
      targets = self.f_warp_sources_arc[a,:,c]

      spikes_nrc = spike_times.loc[units_cross_pop, trials_indices]
      spikes_rt = self.spikes_linear_time_warping(
          spikes_nrc, spike_train_time_line, trials_indices, sources, targets,
          verbose=False)

    print('spikes_rt.shape', spikes_rt.shape)
    knots = [0.045, 0.055, 0.065, 0.09, 0.15, 0.20, 0.23, 0.24, 0.25, 0.28,
             0.31, 0.36, 0.4, 0.45]
    basis, _ = self.fit_model.bspline_basis(
        spline_order=4, knots=knots,
        knots_range=[spike_train_time_line[0], spike_train_time_line[-1]],
        sample_points=spike_train_time_line, show_plot=False)
    log_lmbda_hat, par = self.fit_model.poisson_regression_smoothing_spline(
        spikes_rt, spike_train_time_line, basis=basis,
        lambda_tuning=0, verbose=0)

    hessian = par[3]
    log_lmbda_cov = np.linalg.inv(hessian)
    log_lmbda_cov = basis @ log_lmbda_cov @ basis.T

    CI_scale = scipy.stats.norm.ppf(0.975)  # 95% CI.
    log_lmbda_cov_std = np.sqrt(np.diag(log_lmbda_cov)) * CI_scale
    f_pop_err_up = np.exp(log_lmbda_hat + log_lmbda_cov_std) / self.dt
    f_pop_err_dn = np.exp(log_lmbda_hat - log_lmbda_cov_std) / self.dt
    f_pop_mean = np.exp(log_lmbda_hat) / self.dt

    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(111)
    plt.plot(spike_train_time_line, spikes_rt.mean(axis=0) / self.dt, 'k', lw=0.3)
    plt.fill_between(spike_train_time_line, f_pop_err_dn, f_pop_err_up,
                     facecolor='tab:blue', alpha=0.3)
    plt.plot(spike_train_time_line, f_pop_mean, 'b')
    plt.plot(knots, np.zeros(len(knots)), 'k+')

    if show_samples:
      # Draw samples.
      fig = plt.figure(figsize=(6, 3))
      for _ in range(10):
        log_lmbda_ = np.random.multivariate_normal(log_lmbda_hat, log_lmbda_cov)
        f_pop_ = np.exp(log_lmbda_) / self.dt
        plt.plot(f_pop_)
      plt.title('Samples from fitted model')

    return f_pop_mean, f_pop_err_dn, f_pop_err_up, knots


  def f_pop_map_fit_demo(
      self,
      g=0,
      a=0,
      c=0,
      spike_train_time_line=None,
      spikes_rt=None,
      basis='smoothing',
      eta_smooth_tuning=3e-8,
      file_path=None,
      verbose=False):
    """Evaluates the f_pop fit.

    This is maninly used to evaluate the uncertianty of the estimation. This is
    a simple demonstration of the Bayesian smoothing spline fitting WITHOUT
    considering trial offsets.

    Args:
      basis: 'smoothing', 'spline'.
    """

    if spikes_rt is None:
      spike_trains = self.spike_trains
      spike_times = self.spike_times
      spike_train_time_line = self.spike_train_time_line
      probes = self.probes
      trials_groups = self.trials_groups
      probe = self.probes[a]

      sub_group_df = self.sub_group_df_c[c]
      trials_indices = trials_groups.get_group(self.map_c_to_cid[c]).index.values

      units_cross_pop = sub_group_df[
          (sub_group_df['probe'] == probe) &
          (sub_group_df['group_id'] == g)].index.values
      spikes_rt = spike_trains.loc[units_cross_pop, trials_indices]
      spikes_rt = np.stack(spikes_rt.values.flatten('F'), axis=0)

    num_trains, num_bins = spikes_rt.shape
    print('spikes_rt.shape', spikes_rt.shape)

    if basis == 'smoothing':
      f_basis, f_Omega = self.fit_model.construct_basis_omega(
          spike_train_time_line, knots=100)
    elif basis == 'spline':
      knots = [0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.2, 0.22, 0.25, 0.27,
               0.3, 0.35, 0.4, 0.45]
      f_basis, _ = self.fit_model.bspline_basis(
          spline_order=4, knots=knots,
          knots_range=[spike_train_time_line[0], spike_train_time_line[-1]],
          sample_points=spike_train_time_line, show_plot=False)
      f_Omega = None

    num_times, num_basis = f_basis.shape
    print('basis shape:', f_basis.shape)

    log_lmbda_map, par = self.fit_model.poisson_regression_smoothing_spline(
        spikes_rt, spike_train_time_line,
        basis=f_basis, Omega=f_Omega,
        lambda_tuning=eta_smooth_tuning,
        max_num_iterations=100, verbose=0, verbose_warning=False)

    # Degrees of freedom.
    if basis == 'smoothing':
      lmbd = np.exp(log_lmbda_map).reshape(-1,1)
      X = f_basis
      XW = lmbd * f_basis
      XWX = X.T @ XW
      XX = X.T @ X
      proj_mat = X @ np.linalg.inv(XWX + 2*num_trains*eta_smooth_tuning*f_Omega) @ XW.T
      df = np.trace(proj_mat)
      print('IRLS df:', df)

      proj_mat1 = X @ np.linalg.inv(XX + 2*num_trains*eta_smooth_tuning*f_Omega) @ X.T
      df = np.trace(proj_mat1)
      print('Linear df:', df)

    elif basis == 'spline':
      X = f_basis
      XTX = f_basis.T @ f_basis
      proj_mat = X @ np.linalg.inv(XTX) @ X.T
      df = np.trace(proj_mat)
      print('df:', df)

    plt.figure(figsize=[6, 5])
    ax = plt.subplot(211)
    plt.plot(spike_train_time_line, spikes_rt.mean(axis=0) / self.dt, 'k', lw=0.3)
    plt.plot(spike_train_time_line, np.exp(log_lmbda_map) / self.dt, 'b')
    ax = plt.subplot(212)
    # ax.set_color_cycle(seaborn.color_palette("coolwarm_r",num_lines))
    # plt.plot(proj_mat[10])
    plt.plot(proj_mat[30]); plt.axvline(30, c='grey', lw=0.2)
    plt.plot(proj_mat[50]); plt.axvline(50, c='grey', lw=0.2)
    plt.plot(proj_mat[70]); plt.axvline(70, c='grey', lw=0.2)
    plt.plot(proj_mat[80]); plt.axvline(80, c='grey', lw=0.2)
    plt.plot(proj_mat[118]) ; plt.axvline(118, c='grey', lw=0.2)
    plt.plot(proj_mat[150]); plt.axvline(150, c='grey', lw=0.2)
    plt.plot(proj_mat[185]) ; plt.axvline(185, c='grey', lw=0.2)
    plt.plot(proj_mat[220]) ; plt.axvline(220, c='grey', lw=0.2)
    plt.axhline(0, c='grey', lw=0.3)
    plt.show()

    dt = 2 # ms.
    timeline = np.arange(250) * 2
    gs_kw = dict(width_ratios=[1], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(6, 2.5), gridspec_kw=gs_kw, nrows=1, ncols=1)
    ax = fig.add_subplot(axs)
    ax.tick_params(left=False, labelleft=True)
    plt.plot(timeline, proj_mat[30], 'tab:red')
    plt.axvline(30*dt, c='grey', lw=0.2)
    plt.text(0.13, 1.05, r'$t_1$', ha='center', transform=ax.transAxes, fontsize=14)
    plt.plot(timeline, proj_mat[118], 'tab:green')
    plt.axvline(118*dt, c='grey', lw=0.2)
    plt.text(0.48, 1.05, r'$t_2$', ha='center', transform=ax.transAxes, fontsize=14)
    plt.plot(timeline, proj_mat[155], 'tab:blue')
    plt.axvline(155*dt, c='grey', lw=0.2)
    plt.text(0.62, 1.05, r'$t_3$', ha='center', transform=ax.transAxes, fontsize=14)
    plt.plot(timeline, proj_mat[185], 'tab:orange')
    plt.axvline(185*dt, c='grey', lw=0.2)
    plt.text(0.75, 1.05, r'$t_4$', ha='center', transform=ax.transAxes, fontsize=14)
    plt.axhline(0, c='grey', lw=0.3)
    plt.xlabel('Time [ms]')
    plt.yticks([0], [0])
    plt.xlim(0, 500)
    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('Save figure to: ', file_path)
    plt.show()


  def f_pop_map_logistic_fit_demo(
      self,
      g=0,
      a=0,
      c=0,
      spike_train_time_line=None,
      spikes_rt=None,
      basis='smoothing',
      eta_smooth_tuning=3e-8,
      verbose=False):
    """Evaluates the f_pop fit.

    This is maninly used to evaluate the uncertianty of the estimation. This is
    a simple demonstration of the Bayesian smoothing spline fitting WITHOUT
    considering trial offsets.

    Args:
      basis: 'smoothing', 'spline'.
    """

    if spikes_rt is None:
      spike_trains = self.spike_trains
      spike_times = self.spike_times
      spike_train_time_line = self.spike_train_time_line
      probes = self.probes
      trials_groups = self.trials_groups
      probe = self.probes[a]

      sub_group_df = self.sub_group_df_c[c]
      trials_indices = trials_groups.get_group(self.map_c_to_cid[c]).index.values

      units_cross_pop = sub_group_df[
          (sub_group_df['probe'] == probe) &
          (sub_group_df['group_id'] == g)].index.values
      spikes_rt = spike_trains.loc[units_cross_pop, trials_indices]
      spikes_rt = np.stack(spikes_rt.values.flatten('F'), axis=0)

    num_trains, num_bins = spikes_rt.shape
    print('spikes_rt.shape', spikes_rt.shape)

    if basis == 'smoothing':
      f_basis, f_Omega = self.fit_model.construct_basis_omega(
          spike_train_time_line, knots=100)
    elif basis == 'spline':
      knots = [0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.2, 0.22, 0.25, 0.27,
               0.3, 0.35, 0.4, 0.45]
      f_basis, _ = self.fit_model.bspline_basis(
          spline_order=4, knots=knots,
          knots_range=[spike_train_time_line[0], spike_train_time_line[-1]],
          sample_points=spike_train_time_line, show_plot=False)
      f_Omega = None

    num_times, num_basis = f_basis.shape
    print('basis shape:', f_basis.shape)

    logit_lmbda_map, par = self.fit_model.logistic_regression_smoothing_spline(
        spikes_rt, spike_train_time_line,
        basis=f_basis, Omega=f_Omega,
        lambda_tuning=eta_smooth_tuning,
        learning_rate=1e-3,
        max_num_iterations=200, verbose=1, verbose_warning=False)

    # Degrees of freedom.
    if basis == 'smoothing':
      lmbd = self.fit_model.sigmoid(logit_lmbda_map).reshape(-1,1)
      X = f_basis
      XW = lmbd * f_basis
      XWX = X.T @ XW
      XX = X.T @ X
      proj_mat = X @ np.linalg.inv(XWX + 2*num_trains*eta_smooth_tuning*f_Omega) @ XW.T
      df = np.trace(proj_mat)
      print('IRLS df:', df)

      proj_mat1 = X @ np.linalg.inv(XX + 2*num_trains*eta_smooth_tuning*f_Omega) @ X.T
      df = np.trace(proj_mat1)
      print('Linear df:', df)

    elif basis == 'spline':
      X = f_basis
      XTX = f_basis.T @ f_basis
      proj_mat = X @ np.linalg.inv(XTX) @ X.T
      df = np.trace(proj_mat)
      print('df:', df)

    lmbd = self.fit_model.sigmoid(logit_lmbda_map)
    fig = plt.figure(figsize=(6, 3))
    plt.plot(spike_train_time_line, spikes_rt.mean(axis=0) / self.dt, 'k', lw=0.3)
    plt.plot(spike_train_time_line, lmbd / self.dt, 'b')
    plt.title('MAP')
    plt.show()

    plt.figure(figsize=[6, 5])
    ax = plt.subplot(211)
    plt.plot(spike_train_time_line, lmbd / self.dt, 'b')
    ax = plt.subplot(212)
    # ax.set_color_cycle(seaborn.color_palette("coolwarm_r",num_lines))
    # plt.plot(proj_mat[10])
    plt.plot(proj_mat[40]); plt.axhline(40, c='grey', lw=0.2)
    # plt.plot(proj_mat[70])
    # plt.plot(proj_mat[80])
    plt.plot(proj_mat[105]); plt.axhline(105, c='grey', lw=0.2)
    # plt.plot(proj_mat[170])
    plt.plot(proj_mat[180]); plt.axhline(180, c='grey', lw=0.2)
    # plt.plot(proj_mat[220])
    plt.axhline(0, c='grey', lw=0.3)
    plt.grid()
    plt.show()


  def f_pop_sampling_fit_demo(
      self,
      g=0,
      a=0,
      c=0,
      spike_train_time_line=None,
      spikes_rt=None,
      basis='smoothing',
      eta_smooth_tuning=3e-8,
      num_samples=500,
      random_seed=0,
      verbose=False):
    """Evaluates the f_pop fit.

    This is maninly used to evaluate the uncertianty of the estimation. This is
    a simple demonstration of the Bayesian smoothing spline fitting WITHOUT
    considering trial offsets.

    Args:
      basis: 'smoothing', 'spline'.
    """
    np.random.seed(random_seed)

    if spikes_rt is None:
      spike_trains = self.spike_trains
      spike_times = self.spike_times
      spike_train_time_line = self.spike_train_time_line
      probes = self.probes
      trials_groups = self.trials_groups
      probe = self.probes[a]

      sub_group_df = self.sub_group_df_c[c]
      trials_indices = trials_groups.get_group(self.map_c_to_cid[c]).index.values

      units_cross_pop = sub_group_df[
          (sub_group_df['probe'] == probe) &
          (sub_group_df['group_id'] == g)].index.values
      spikes_rt = spike_trains.loc[units_cross_pop, trials_indices]
      spikes_rt = np.stack(spikes_rt.values.flatten('F'), axis=0)

    num_trains, num_bins = spikes_rt.shape
    print('spikes_rt.shape', spikes_rt.shape)

    if basis == 'smoothing':
      f_basis, f_Omega = self.fit_model.construct_basis_omega(
          spike_train_time_line, knots=100)
    elif basis == 'spline':
      knots = [0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.2, 0.22, 0.25, 0.27,
               0.3, 0.35, 0.4, 0.45]
      f_basis, _ = self.fit_model.bspline_basis(
          spline_order=4, knots=knots,
          knots_range=[spike_train_time_line[0], spike_train_time_line[-1]],
          sample_points=spike_train_time_line, show_plot=False)
      f_Omega = None

    num_times, num_basis = f_basis.shape
    print('basis shape:', f_basis.shape)

    log_lmbda_map, par = self.fit_model.poisson_regression_smoothing_spline(
        spikes_rt, spike_train_time_line,
        basis=f_basis, Omega=f_Omega,
        lambda_tuning=eta_smooth_tuning,
        max_num_iterations=100, verbose=0, verbose_warning=False)

    fig = plt.figure(figsize=(6, 3))
    plt.plot(spike_train_time_line, spikes_rt.mean(axis=0) / self.dt, 'k', lw=0.3)
    plt.plot(spike_train_time_line, np.exp(log_lmbda_map) / self.dt, 'b')
    plt.title('MAP')
    if verbose:
      plt.show()
    else:
      plt.close()

    # Sampling methods.
    hessian = par[3]
    hessian_baseline = par[4]
    beta_cov = np.linalg.inv(hessian) * 0.04
    # beta_cov = np.eye(num_basis) * 0.001
    beta_baseline_cov = 1 / hessian_baseline * 0.04

    # Initialization with best fit.
    beta_cnd = par[0]
    beta_baseline_cnd = par[1]
    # Simple initialization.
    # beta_cnd = np.zeros(beta_cnd.shape)
    # beta_baseline_cnd = np.zeros(beta_baseline_cnd.shape)

    beta_samples = []
    beta_baseline_samples = []
    log_posterior = []
    accept_cnt = 0
    beta_samples.append(beta_cnd.copy())
    beta_baseline_samples.append(beta_baseline_cnd)

    for _ in range(num_samples-1):
      beta_ = np.random.multivariate_normal(beta_cnd.reshape(-1), beta_cov).reshape(-1, 1)
      beta_baseline_ = np.random.normal(beta_baseline_cnd, beta_baseline_cov)
      log_lambda_ = (f_basis @ beta_ + beta_baseline_).reshape(-1)

      log_posterior_ = - self.spike_trains_neg_log_likelihood(log_lambda_, spikes_rt)
      log_prior_ = - num_trains * eta_smooth_tuning * beta_.T @ f_Omega @ beta_
      log_posterior_ += log_prior_
      log_posterior_ = np.asscalar(log_posterior_)

      log_lambda_prev = (f_basis @ beta_cnd + beta_baseline_cnd).reshape(-1)
      log_posterior_prev = - self.spike_trains_neg_log_likelihood(log_lambda_prev, spikes_rt)
      log_prior_prev = - num_trains * eta_smooth_tuning * beta_cnd.T @ f_Omega @ beta_cnd
      log_posterior_prev += log_prior_prev
      log_posterior_prev = np.asscalar(log_posterior_prev)

      log_posterior.append(log_posterior_prev)

      # Proposal ratio.
      mh_ratio = np.exp(log_posterior_ - log_posterior_prev)
      # mh_ratio = 1
      u = np.random.rand()
      # Accept the new candicate, otherwise doesn't change.
      if u < mh_ratio:
        beta_cnd = beta_
        beta_baseline_cnd = beta_baseline_
        accept_cnt += 1

      beta_samples.append((beta_cnd.copy(), beta_baseline_cnd))

    print('Accept ratio:',accept_cnt / num_samples)

    fig = plt.figure(figsize=(6, 3))
    plt.plot(log_posterior)
    plt.title('log-posterior trace')
    if verbose:
      plt.show()
    else:
      plt.close()

    # Burn-in 1/3, sclice 5th.
    samples = beta_samples[num_samples//3::1]
    print('samples len:', len(samples))
    f_pop = np.zeros([len(samples), num_times])
    for i, (beta, beta_baseline) in enumerate(samples):
      lmbd = np.exp(f_basis @ beta + beta_baseline).reshape(-1) / self.dt
      f_pop[i] = lmbd
    f_pop_up = np.quantile(f_pop, 0.975, axis=0)
    f_pop_dn = np.quantile(f_pop, 0.025, axis=0)
    # f_pop_center = np.mean(f_pop, axis=0)
    f_pop_center = np.quantile(f_pop, 0.5, axis=0)

    fig = plt.figure(figsize=(5, 3))
    plt.plot(spike_train_time_line, spikes_rt.mean(axis=0) / self.dt, 'k', lw=0.3)
    plt.plot(spike_train_time_line, f_pop.T, 'b', lw=0.2)
    plt.title('f_pop traces')
    if verbose:
      plt.show()
    else:
      plt.close()

    fig = plt.figure(figsize=(6, 3))
    plt.plot(spike_train_time_line, spikes_rt.mean(axis=0) / self.dt, 'k', lw=0.3)
    plt.fill_between(spike_train_time_line, f_pop_dn, f_pop_up,
                     facecolor='tab:blue', alpha=0.3)
    plt.plot(spike_train_time_line, f_pop_center, 'b')
    # plt.ylim(0, 90)
    plt.title('f_pop')
    if verbose:
      plt.show()
    else:
      plt.close()

    return f_pop_center, f_pop_dn, f_pop_up


  def plot_f_pop_demo_comparison(
      self,
      g=0,
      a=0,
      c=0,
      burn_in=0,
      end=None,
      step=1,
      add_labels=True,
      output_dir=None):
    """Compares the simple PSTH with model."""
    np.random.seed(0)
    areas_names = ['V1', 'LM', 'AL']
    f0, f_dn0, f_up0 = self.f_pop_sampling_fit_demo(
        g, a, c, num_samples=3000)
    _, f1, f_dn1, f_up1 = self.samples.plot_f_pop_CI_demo(
        c, a, self.spike_train_time_line,
        burn_in=burn_in, end=end, step=step, output_dir=None, show_plot=False)
    timeline = self.spike_train_time_line * 1000

    fig = plt.figure(figsize=(3.75, 2))
    ax = plt.gca()
    ax.tick_params(left=True, labelbottom=True, bottom=True,
                   top=False, labeltop=False)

    plt.fill_between(timeline, f_dn0, f_up0,
                     facecolor='b', alpha=0.3, label='95% CI')
    # plt.plot(timeline, f_dn0, '--b', lw=1.2)
    # plt.plot(timeline, f_up0, '--b', lw=1.2)
    plt.fill_between(timeline, f_dn1, f_up1,
                     facecolor='tab:grey', alpha=0.3, label='95% CI')
    plt.plot(timeline, f0, 'b', label='PSTH')
    plt.plot(timeline, f1, 'k', label='model')
    if add_labels:
      plt.ylabel('Firing rate [Spikes/sec]')
      plt.xlabel('Time [ms]')
      plt.legend(fontsize=8)
    else:
      plt.ylabel(' ')
      plt.xlabel(' ')
    plt.title(f'condition={self.map_c_to_cid[c]}  {areas_names[a]}')

    if output_dir is not None:
      file_path = os.path.join(output_dir,
          f'{self.session_id}_f_pop_comp_c{c}_a{a}.pdf')
      plt.savefig(file_path, bbox_inches='tight')
      print('save figure:', file_path)
    plt.show()


  def update_f_local_pop_cag(
      self,
      c,
      constant_fit=False,
      sample_type='fit',
      record=True,
      verbose=True):
    """Updates local subgroup activities."""
    spike_trains = self.spike_trains
    spike_train_time_line = self.spike_train_time_line
    probes = self.probes
    trials_groups = self.trials_groups
    sub_group_df = self.sub_group_df_c[c]
    trials_indices = trials_groups.get_group(self.map_c_to_cid[c]).index.values

    for a, probe in enumerate(probes):
      # If there are only two groups, then ignore the inhomo group.
      if self.num_groups != 2:
        # Group 1
        g = 1
        active_units = sub_group_df[
            (sub_group_df['probe'] == probe) &
            (sub_group_df['group_id'] == g)].index.values
        spikes_gac1 = spike_trains.loc[active_units, trials_indices]
        spikes_gac1 = np.stack(spikes_gac1.values.flatten('F'), axis=0)
        # Load cache.
        if (c,a,g,'beta') in self.f_pop_beta_cag:
          beta = self.f_pop_beta_cag[(c,a,g,'beta')]
        else:
          beta = None
        if (c,a,g,'baseline') in self.f_pop_beta_cag:
          beta_baseline = self.f_pop_beta_cag[(c,a,g,'baseline')]
        else:
          beta_baseline = None

        if sample_type == 'fit':
          log_lmbda_hat, par = self.fit_model.poisson_regression_smoothing_spline(
              spikes_gac1, spike_train_time_line, constant_fit=constant_fit,
              lambda_tuning=self.eta_smooth_tuning,
              beta_initial=beta, beta_baseline_initial=beta_baseline,
              basis=self.f_basis, Omega=self.f_Omega,
              verbose=0, verbose_warning=False)
          self.f_pop_cag[c,a,g] = log_lmbda_hat
          self.f_pop_beta_cag[(c,a,g,'beta')] = par[0]
          self.f_pop_beta_cag[(c,a,g,'baseline')] = par[1]
          self.f_pop_par_cag[(c,a,g,'beta_hessian')] = par[3]
          self.f_pop_par_cag[(c,a,g,'beta_baseline_hessian')] = par[4]
        elif sample_type == 'sample':
          # self.samples.f_sample_cnt += 1

          num_trains, _ = spikes_gac1.shape
          hessian = self.f_pop_par_cag[(c,a,g,'beta_hessian')]
          hessian_baseline = self.f_pop_par_cag[(c,a,g,'beta_baseline_hessian')]
          beta_cov = np.linalg.inv(hessian) * 0.1
          beta_baseline_cov = 1 / hessian_baseline * 0.1
          beta_prev = self.f_pop_beta_cag[(c,a,g,'beta')]
          beta_baseline_prev = self.f_pop_beta_cag[(c,a,g,'baseline')]
          # Draw candicate sample.
          beta_cnd = np.random.multivariate_normal(
              beta_prev.reshape(-1), beta_cov).reshape(-1, 1)
          beta_baseline_cnd = np.random.normal(beta_baseline_prev, beta_baseline_cov)
          # Candidate.
          log_lambda_cnd = (self.f_basis @ beta_cnd + beta_baseline_cnd).reshape(-1)
          log_posterior_cnd = - self.spike_trains_neg_log_likelihood(
              log_lambda_cnd, spikes_gac1)
          log_prior_cnd = (- num_trains * self.eta_smooth_tuning *
                        beta_cnd.T @ self.f_Omega @ beta_cnd)
          log_posterior_cnd += log_prior_cnd
          log_posterior_cnd = np.asscalar(log_posterior_cnd)
          # Previous.
          log_lambda_prev = (self.f_basis @ beta_prev + beta_baseline_prev).reshape(-1)
          log_posterior_prev = - self.spike_trains_neg_log_likelihood(
              log_lambda_prev, spikes_gac1)
          log_prior_prev = (- num_trains * self.eta_smooth_tuning *
                            beta_prev.T @ self.f_Omega @ beta_prev)
          log_posterior_prev += log_prior_prev
          log_posterior_prev = np.asscalar(log_posterior_prev)

          mh_ratio = np.exp(log_posterior_cnd - log_posterior_prev)
          u = np.random.rand()
          if u < mh_ratio:
            self.f_pop_cag[c,a,g] = log_lambda_cnd
            self.f_pop_beta_cag[(c,a,g,'beta')] = beta_cnd
            self.f_pop_beta_cag[(c,a,g,'baseline')] = beta_baseline_cnd
            # self.samples.f_sample_accept_cnt += 1

      # Group 2
      g = 2
      idle_units = sub_group_df[
          (sub_group_df['probe'] == probe) &
          (sub_group_df['group_id'] == g)].index.values
      spikes_gac2 = spike_trains.loc[idle_units, trials_indices]
      spikes_gac2 = np.stack(spikes_gac2.values.flatten('F'), axis=0)
      # Load cache. No need for beta in constant fit.
      if (c,a,g,'baseline') in self.f_pop_par_cag:
        beta_baseline = self.f_pop_par_cag[(c,a,g,'baseline')]
      else:
        beta_baseline = None

      if sample_type == 'fit':
        log_lmbda_hat, par = self.fit_model.poisson_regression_smoothing_spline(
            spikes_gac2, spike_train_time_line, constant_fit=True,
            beta_baseline_initial=beta_baseline,
            verbose=0, verbose_warning=False)
        self.f_pop_cag[c,a,g] = log_lmbda_hat
        self.f_pop_beta_cag[(c,a,g,'baseline')] = par[1]
        self.f_pop_par_cag[(c,a,g,'beta_baseline_hessian')] = par[4]

      elif sample_type == 'sample':
        # self.samples.f_sample_cnt += 1

        num_trains, num_bins = spikes_gac2.shape
        hessian_baseline = self.f_pop_par_cag[(c,a,g,'beta_baseline_hessian')]
        beta_baseline_cov = 1 / hessian_baseline * 0.1

        beta_baseline_prev = self.f_pop_beta_cag[(c,a,g,'baseline')]
        beta_baseline_cnd = np.random.normal(beta_baseline_prev, beta_baseline_cov)
        # Candidate.
        log_lambda_cnd = np.zeros(num_bins) + beta_baseline_cnd
        log_posterior_cnd = - self.spike_trains_neg_log_likelihood(
            log_lambda_cnd, spikes_gac2)
        log_posterior_cnd = np.asscalar(log_posterior_cnd)
        # Previos.
        log_lambda_prev = np.zeros(num_bins) + beta_baseline_prev
        log_posterior_prev = - self.spike_trains_neg_log_likelihood(
            log_lambda_prev, spikes_gac2)
        log_posterior_prev = np.asscalar(log_posterior_prev)

        mh_ratio = np.exp(log_posterior_cnd - log_posterior_prev)
        u = np.random.rand()
        if u < mh_ratio:
          # print('accept', a, g)
          self.f_pop_cag[c,a,g] = log_lambda_cnd
          self.f_pop_beta_cag[(c,a,g,'baseline')] = beta_baseline_cnd
          # self.samples.f_sample_accept_cnt += 1

      if verbose:
        plt.figure(figsize=(12, 3))
        if self.num_groups != 2:
          plt.subplot(121)
          g = 1
          plt.plot(spike_train_time_line, spikes_gac1.mean(axis=0), linewidth=0.5)
          plt.plot(spike_train_time_line, np.exp(self.f_pop_cag[c,a,g]))
          # plt.ylim(0, 0.08)

        plt.subplot(122)
        g = 2
        plt.plot(spike_train_time_line, spikes_gac2.mean(axis=0), linewidth=0.5)
        plt.plot(spike_train_time_line, np.exp(self.f_pop_cag[c,a,g]))
        # plt.ylim(0, 0.08)
        plt.show()

    # Add to the samples collector.
    # Note that the samples are recorded by `update_f_cross_pop_ac` through the
    # variable `f_pop_cag` all together.


  def update_f_cross_pop_ca(
      self,
      c,
      sample_type='fit',
      constant_fit=False,
      record=True,
      verbose=True):
    """Updates f_cross_pop."""
    spike_trains = self.spike_trains
    spike_times = self.spike_times
    spike_train_time_line = self.spike_train_time_line
    probes = self.probes
    trials_groups = self.trials_groups
    g = 0
    sub_group_df = self.sub_group_df_c[c]
    trials_indices = trials_groups.get_group(self.map_c_to_cid[c]).index.values
    all_trials = spike_trains.columns.values
    spikes_rc_list = []
    q_offset_rc = []

    for a, probe in enumerate(probes):
      units_cross_pop = sub_group_df[
          (sub_group_df['probe'] == probe) &
          (sub_group_df['group_id'] == g)].index.values

      # Offsets.
      if self.model_feature_type in ['B', 'BS', 'BSS']:
        q_offset = np.kron(self.q_arc[a,:,c].reshape(-1, 1),
                           np.ones([len(units_cross_pop), 1]))
      if self.model_feature_type in ['S', 'S1', 'S2', 'SS']:
        q_offset = 0

      # Spikes.
      if self.model_feature_type in ['B']:
        spikes_nrc = spike_trains.loc[units_cross_pop, trials_indices]
        # 'C' (row-major), 'F' Fortran (column-major) order.
        spikes_nrc = np.stack(spikes_nrc.values.flatten('F'), axis=0)
      if self.model_feature_type in ['S', 'S1', 'S2', 'BS', 'SS', 'BSS']:
        # Pay attention that the sources-targets pair is the opposite of the
        # intensity function f_wap. In another word, the time warpping function
        # for spike trains is the inverse function of the intensity one.
        sources = self.f_warp_targets_arc[a,:,c]
        targets = self.f_warp_sources_arc[a,:,c]
        spikes_nrc = spike_times.loc[units_cross_pop, trials_indices]
        spikes_nrc = self.spikes_linear_time_warping(
            spikes_nrc, spike_train_time_line, trials_indices, sources, targets,
            verbose=False)

        if verbose:
          plt.figure()
          plt.plot(spike_train_time_line,
                   spikes_nrc.mean(axis=0), label='before warpping')
          spikes_nrc_ = spike_trains.loc[units_cross_pop, trials_indices]
          spikes_nrc_ = np.stack(spikes_nrc_.values.flatten('F'), axis=0)
          plt.plot(spike_train_time_line,
                   spikes_nrc_.mean(axis=0), label='after warpping')
          plt.legend()
          plt.show()

      # Load cache.
      if (c,a,g,'beta') in self.f_pop_beta_cag:
        beta = self.f_pop_beta_cag[(c,a,g,'beta')]
      else:
        beta = None
      if (c,a,g,'baseline') in self.f_pop_beta_cag:
        beta_baseline = self.f_pop_beta_cag[(c,a,g,'baseline')]
      else:
        beta_baseline = None

      if sample_type == 'fit':
        log_lmbda_hat, par = self.fit_model.poisson_regression_smoothing_spline(
            spikes_nrc, spike_train_time_line, constant_fit=constant_fit,
            log_lambda_offset=q_offset,
            beta_initial=beta, beta_baseline_initial=beta_baseline,
            basis=self.f_basis, Omega=self.f_Omega,
            lambda_tuning=self.eta_smooth_tuning, lambda_baseline_tuning=0,
            max_num_iterations=100, verbose=0, verbose_warning=False)

        self.f_pop_cag[c,a,g] = log_lmbda_hat
        self.f_pop_beta_cag[(c,a,g,'beta')] = par[0]
        self.f_pop_beta_cag[(c,a,g,'baseline')] = par[1]
        self.f_pop_par_cag[(c,a,g,'beta_hessian')] = par[3]
        self.f_pop_par_cag[(c,a,g,'beta_baseline_hessian')] = par[4]

      elif sample_type == 'sample':
        self.samples.f_sample_cnt += 1

        num_trains, _ = spikes_nrc.shape
        hessian = self.f_pop_par_cag[(c,a,g,'beta_hessian')]
        hessian_baseline = self.f_pop_par_cag[(c,a,g,'beta_baseline_hessian')]
        beta_cov = np.linalg.inv(hessian) * 0.1
        beta_baseline_cov = 1 / hessian_baseline * 0.1
        beta_prev = self.f_pop_beta_cag[(c,a,g,'beta')]
        beta_baseline_prev = self.f_pop_beta_cag[(c,a,g,'baseline')]
        # Draw a sample.
        beta_cnd = np.random.multivariate_normal(
            beta_prev.reshape(-1), beta_cov).reshape(-1, 1)
        beta_baseline_cnd = np.random.normal(beta_baseline_prev, beta_baseline_cov)

        # Candidate.
        log_lambda_cnd = (self.f_basis @ beta_cnd + beta_baseline_cnd).reshape(-1)
        log_lambda_cnd = log_lambda_cnd.reshape(-1) + q_offset
        log_posterior_cnd = - self.spike_trains_neg_log_likelihood(
            log_lambda_cnd, spikes_nrc)
        log_prior_cnd = (- num_trains * self.eta_smooth_tuning *
                      beta_cnd.T @ self.f_Omega @ beta_cnd)
        log_posterior_cnd += log_prior_cnd
        log_posterior_cnd = np.asscalar(log_posterior_cnd)

        # Previous.
        log_lambda_prev = (self.f_basis @ beta_prev + beta_baseline_prev).reshape(-1)
        log_lambda_prev = log_lambda_prev.reshape(-1) + q_offset
        log_posterior_prev = - self.spike_trains_neg_log_likelihood(
            log_lambda_prev, spikes_nrc)
        log_prior_prev = (- num_trains * self.eta_smooth_tuning *
                          beta_prev.T @ self.f_Omega @ beta_prev)
        log_posterior_prev += log_prior_prev
        log_posterior_prev = np.asscalar(log_posterior_prev)

        mh_ratio = np.exp(log_posterior_cnd - log_posterior_prev)
        u = np.random.rand()
        if u < mh_ratio:
          self.f_pop_cag[c,a,g] = (self.f_basis @ beta_cnd + beta_baseline_cnd).reshape(-1)
          self.f_pop_beta_cag[(c,a,g,'beta')] = beta_cnd
          self.f_pop_beta_cag[(c,a,g,'baseline')] = beta_baseline_cnd
          self.samples.f_sample_accept_cnt += 1

      if verbose:
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        plt.plot(spike_train_time_line, spikes_nrc.mean(axis=0), 'k', lw=0.3)
        plt.plot(spike_train_time_line,
                 np.exp(log_lmbda_hat + np.mean(q_offset)), 'b')
        plt.text(0.6, 0.8, 'With mean offset approx', transform=ax.transAxes)
        # plt.ylim(0, 0.08)

    # Add to the samples collector.
    if record:
      self.samples.f_pop.append(self.f_pop_cag.copy())
      self.samples.f_pop_beta.append(self.f_pop_beta_cag.copy())


  def update_f_cross_pop_ca_2(
      self,
      c,
      sample_type='fit',
      constant_fit=False,
      record=True,
      verbose=True):
    """Updates f_cross_pop."""
    spike_trains = self.spike_trains
    spike_times = self.spike_times
    spike_train_time_line = self.spike_train_time_line
    probes = self.probes
    trials_groups = self.trials_groups
    g = 0
    sub_group_df = self.sub_group_df_c[c]
    trials_indices = trials_groups.get_group(self.map_c_to_cid[c]).index.values

    for a, probe in enumerate(probes):
      units_cross_pop = sub_group_df[
          (sub_group_df['probe'] == probe) &
          (sub_group_df['group_id'] == g)].index.values

      # Spikes.
      spikes_nrc = spike_trains.loc[units_cross_pop, trials_indices]
      # 'C' (row-major), 'F' Fortran (column-major) order.
      spikes_nrc = np.stack(spikes_nrc.values.flatten('F'), axis=0)

      # Load cache.
      if (c,a,g,'beta') in self.f_pop_beta_cag:
        beta = self.f_pop_beta_cag[(c,a,g,'beta')]
      else:
        beta = None
      if (c,a,g,'baseline') in self.f_pop_beta_cag:
        beta_baseline = self.f_pop_beta_cag[(c,a,g,'baseline')]
      else:
        beta_baseline = None

      if sample_type == 'fit':
        # Offsets.
        if self.model_feature_type in ['B', 'BS', 'BSS']:
          q_offset = np.kron(self.q_arc[a,:,c].reshape(-1, 1),
                             np.ones([len(units_cross_pop), 1]))
        if self.model_feature_type in ['S', 'S1', 'S2', 'SS']:
          q_offset = 0
        if self.model_feature_type in ['S', 'S1', 'S2', 'BS', 'SS', 'BSS']:
          # Pay attention that the sources-targets pair is the opposite of the
          # intensity function f_wap. In another word, the time warpping function
          # for spike trains is the inverse function of the intensity one.
          sources = self.f_warp_targets_arc[a,:,c]
          targets = self.f_warp_sources_arc[a,:,c]
          spikes_nrc = spike_times.loc[units_cross_pop, trials_indices]
          spikes_nrc = self.spikes_linear_time_warping(
              spikes_nrc, spike_train_time_line, trials_indices, sources, targets,
              verbose=False)

        log_lmbda_hat, par = self.fit_model.poisson_regression_smoothing_spline(
            spikes_nrc, spike_train_time_line, constant_fit=constant_fit,
            log_lambda_offset=q_offset,
            beta_initial=beta, beta_baseline_initial=beta_baseline,
            basis=self.f_basis, Omega=self.f_Omega,
            lambda_tuning=self.eta_smooth_tuning, lambda_baseline_tuning=0, # 1e-3
            max_num_iterations=100, verbose=0, verbose_warning=False)

        self.f_pop_cag[c,a,g] = log_lmbda_hat
        self.f_pop_beta_cag[(c,a,g,'beta')] = par[0]
        self.f_pop_beta_cag[(c,a,g,'baseline')] = par[1]
        self.f_pop_par_cag[(c,a,g,'beta_hessian')] = par[3]
        self.f_pop_par_cag[(c,a,g,'beta_baseline_hessian')] = par[4]

      elif sample_type == 'sample':
        self.samples.f_sample_cnt += 1

        num_trains, _ = spikes_nrc.shape
        hessian = self.f_pop_par_cag[(c,a,g,'beta_hessian')]
        hessian_baseline = self.f_pop_par_cag[(c,a,g,'beta_baseline_hessian')]
        beta_cov = np.linalg.inv(hessian) * 0.05  # proposal Cov.
        beta_baseline_cov = 1 / hessian_baseline * 0.05
        beta_prev = self.f_pop_beta_cag[(c,a,g,'beta')]
        beta_baseline_prev = self.f_pop_beta_cag[(c,a,g,'baseline')]
        # Draw a sample.
        beta_cnd = np.random.multivariate_normal(
            beta_prev.reshape(-1), beta_cov).reshape(-1, 1)
        beta_baseline_cnd = np.random.normal(beta_baseline_prev, beta_baseline_cov)

        # Candidate.
        log_lambda_cnd = (self.f_basis @ beta_cnd + beta_baseline_cnd).reshape(-1)
        # Replicate for many trials.
        log_lambda_cnd = log_lambda_cnd + np.zeros((len(trials_indices),1))
        if self.model_feature_type in ['S', 'S1', 'S2', 'BS', 'SS', 'BSS']:
          for r, trial_id in enumerate(trials_indices):
            log_lambda_cnd[r] = log_lambda_cnd[r] + self.q_arc[a,r,c]
            sources = self.f_warp_sources_arc[a,r,c]
            targets = self.f_warp_targets_arc[a,r,c]
            log_lambda_cnd[r] = self.linear_time_warping(spike_train_time_line,
                log_lambda_cnd[r], sources, targets, verbose=False)
        log_lambda_cnd = np.kron(log_lambda_cnd,
                                 np.ones([len(units_cross_pop), 1]))
        log_posterior_cnd = - self.spike_trains_neg_log_likelihood(
            log_lambda_cnd, spikes_nrc)
        log_prior_cnd = (- num_trains * self.eta_smooth_tuning *
                      beta_cnd.T @ self.f_Omega @ beta_cnd)
        log_posterior_cnd += log_prior_cnd
        log_posterior_cnd = np.asscalar(log_posterior_cnd)

        # Previous.
        log_lambda_prev = (self.f_basis @ beta_prev + beta_baseline_prev).reshape(-1)
        log_lambda_prev = log_lambda_prev + np.zeros((len(trials_indices),1))
        if self.model_feature_type in ['S', 'S1', 'S2', 'BS', 'SS', 'BSS']:
          for r, trial_id in enumerate(trials_indices):
            log_lambda_prev[r] = log_lambda_prev[r] + self.q_arc[a,r,c]
            sources = self.f_warp_sources_arc[a,r,c]
            targets = self.f_warp_targets_arc[a,r,c]
            log_lambda_prev[r] = self.linear_time_warping(spike_train_time_line,
                log_lambda_prev[r], sources, targets, verbose=False)
        log_lambda_prev = np.kron(log_lambda_prev,
                                  np.ones([len(units_cross_pop), 1]))
        log_posterior_prev = - self.spike_trains_neg_log_likelihood(
            log_lambda_prev, spikes_nrc)
        log_prior_prev = (- num_trains * self.eta_smooth_tuning *
                          beta_prev.T @ self.f_Omega @ beta_prev)
        log_posterior_prev += log_prior_prev
        log_posterior_prev = np.asscalar(log_posterior_prev)

        mh_ratio = np.exp(log_posterior_cnd - log_posterior_prev)
        u = np.random.rand()
        if u < mh_ratio:
          self.f_pop_cag[c,a,g] = (self.f_basis @ beta_cnd + beta_baseline_cnd).reshape(-1)
          self.f_pop_beta_cag[(c,a,g,'beta')] = beta_cnd
          self.f_pop_beta_cag[(c,a,g,'baseline')] = beta_baseline_cnd
          self.samples.f_sample_accept_cnt += 1

      if verbose:
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        plt.plot(spike_train_time_line, spikes_nrc.mean(axis=0), 'k', lw=0.3)
        plt.plot(spike_train_time_line,
                 np.exp(log_lmbda_hat + np.mean(q_offset)), 'b')
        plt.text(0.6, 0.8, 'With mean offset approx', transform=ax.transAxes)
        # plt.ylim(0, 0.08)

    # Add to the samples collector.
    if record:
      self.samples.f_pop.append(self.f_pop_cag.copy())
      self.samples.f_pop_beta.append(self.f_pop_beta_cag.copy())


  def shuffle_trials_groups(
      self,
      num_fold=5,
      shuffle_type='kfold',
      verbose=False):
    """Split the trials for cross-validation.

    Args:
      shuffle_type: Currently only supports this one.
    """
    def apply_index(value, indices):
      value = value.iloc[indices]
      return value
    if verbose:
      print('---------- Whole trials ----------')
      self.print_conditions(self.trials_groups)
      print('---------- kfold trials ----------')

    trials_groups = self.trials_groups
    trials_groups_kfold = []
    kfold = sklearn.model_selection.KFold(
        n_splits=num_fold, random_state=None, shuffle=False)
    for train_index, test_index in kfold.split(np.arange(self.num_trials)):
      trials_groups_train = trials_groups.apply(apply_index, indices=train_index)
      # trials_groups_train = trials_groups_train.reset_index(level='stimulus_condition_id')
      trials_groups_train = trials_groups_train.droplevel('stimulus_condition_id')
      trials_groups_train = trials_groups_train.groupby('stimulus_condition_id')

      trials_groups_test = trials_groups.apply(apply_index, indices=test_index)
      # trials_groups_test = trials_groups_test.reset_index(level='stimulus_condition_id')
      trials_groups_test = trials_groups_test.droplevel('stimulus_condition_id')
      trials_groups_test = trials_groups_test.groupby('stimulus_condition_id')

      trials_groups_kfold.append((trials_groups_train, trials_groups_test))

      if verbose:
        self.print_conditions(trials_groups_train)
        self.print_conditions(trials_groups_test)
        print()

    return trials_groups_kfold


  def f_pop_cross_validation(
      self,
      c,
      trials_groups_train,
      trials_groups_test,
      eta_smooth_tuning,
      include_non_cross_groups=False,
      verbose=False):
    """Updates f_cross_pop.

    The corss validation can be done after initial fitting, since it needs the
    q_arc to shift the peaks.

    Args:
      separate_fitting: Fits brain areas with differnt timecourses.
    """
    spike_trains = self.spike_trains
    spike_times = self.spike_times
    spike_train_time_line = self.spike_train_time_line
    probes = self.probes
    sub_group_df = self.sub_group_df_c[c]
    # all, train, test.
    trials_indices = self.trials_groups.get_group(
        self.map_c_to_cid[c]).index.values
    trials_indices_train = trials_groups_train.get_group(
        self.map_c_to_cid[c]).index.values
    trials_indices_test = trials_groups_test.get_group(
        self.map_c_to_cid[c]).index.values
    r_train = np.where(np.in1d(trials_indices, trials_indices_train))[0]
    r_test = np.where(np.in1d(trials_indices, trials_indices_test))[0]

    all_trials = spike_trains.columns.values
    spikes_rc_list = []
    q_offset_rc = []
    nll_train = 0
    nll_test = 0

    for a, probe in enumerate(probes):
      #--------- group 0 ----------
      g = 0
      units_cross_pop = sub_group_df[
          (sub_group_df['probe'] == probe) &
          (sub_group_df['group_id'] == g)].index.values

      # Offsets.
      if self.model_feature_type in ['B', 'BS', 'BSS']:
        q_offset_train = np.kron(self.q_arc[a,r_train,c].reshape(-1, 1),
                                 np.ones([len(units_cross_pop), 1]))
        q_offset_test = np.kron(self.q_arc[a,r_test,c].reshape(-1, 1),
                                 np.ones([len(units_cross_pop), 1]))
      if self.model_feature_type in ['S', 'S1', 'S2', 'SS']:
        q_offset_train = 0
        q_offset_test = 0

      # Spikes.
      if self.model_feature_type in ['B']:
        spikes_nrc_train = spike_trains.loc[units_cross_pop, trials_indices_train]
        spikes_nrc_train = np.stack(spikes_nrc_train.values.flatten('F'), axis=0)
        spikes_nrc_test = spike_trains.loc[units_cross_pop, trials_indices_test]
        spikes_nrc_test = np.stack(spikes_nrc_test.values.flatten('F'), axis=0)
      if self.model_feature_type in ['S', 'S1', 'S2', 'BS', 'SS', 'BSS']:
        # Pay attention that the sources-targets pair is the opposite of the
        # intensity function f_wap. In another word, the time warpping function
        # for spike trains is the inverse function of the intensity one.
        # Train spikes.
        sources = self.f_warp_targets_arc[a,r_train,c]
        targets = self.f_warp_sources_arc[a,r_train,c]
        spikes_nrc_train = spike_times.loc[units_cross_pop, trials_indices_train]
        spikes_nrc_train = self.spikes_linear_time_warping(
            spikes_nrc_train, spike_train_time_line, trials_indices_train,
            sources, targets, verbose=False)
        # Test spikes.
        sources = self.f_warp_targets_arc[a,r_test,c]
        targets = self.f_warp_sources_arc[a,r_test,c]
        spikes_nrc_test = spike_times.loc[units_cross_pop, trials_indices_test]
        spikes_nrc_test = self.spikes_linear_time_warping(
            spikes_nrc_test, spike_train_time_line, trials_indices_test,
            sources, targets, verbose=False)

      # Load cache.
      if (c,a,g,'beta') in self.f_pop_beta_cag:
        beta = self.f_pop_beta_cag[(c,a,g,'beta')]
      else:
        beta = None
      if (c,a,g,'baseline') in self.f_pop_beta_cag:
        beta_baseline = self.f_pop_beta_cag[(c,a,g,'baseline')]
      else:
        beta_baseline = None

      log_lmbda_train, par_train = self.fit_model.poisson_regression_smoothing_spline(
          spikes_nrc_train, spike_train_time_line, constant_fit=False,
          log_lambda_offset=q_offset_train,
          # beta_initial=beta, beta_baseline_initial=beta_baseline,
          basis=self.f_basis, Omega=self.f_Omega,
          lambda_tuning=eta_smooth_tuning, lambda_baseline_tuning=1e-5,
          max_num_iterations=500, verbose=0, verbose_warning=False)

      if verbose:
        print('eta_smooth_tuning', eta_smooth_tuning)
        print('spikes_nrc_train.shape', spikes_nrc_train.shape)
        plt.plot(figsize=[4, 2])
        plt.plot(np.exp(log_lmbda_train) / 0.002)
        plt.plot(spikes_nrc_train.mean(axis=0) / 0.002)
        plt.show()
        print('STOP! Plot only the first one. For others, comment out <return>')
        return nll_train, nll_test

      nll_train += par_train[5]
      log_lambda_test = log_lmbda_train.reshape(-1) + q_offset_test
      nll_test += self.spike_trains_neg_log_likelihood(log_lambda_test, spikes_nrc_test)

      #--------- group 1 ----------
      if not include_non_cross_groups:
        continue

      # Group 1
      g = 1
      active_units = sub_group_df[
          (sub_group_df['probe'] == probe) &
          (sub_group_df['group_id'] == g)].index.values
      spikes_nrc_train = spike_trains.loc[active_units, trials_indices_train]
      spikes_nrc_train = np.stack(spikes_nrc_train.values.flatten('F'), axis=0)
      spikes_nrc_test = spike_trains.loc[active_units, trials_indices_test]
      spikes_nrc_test = np.stack(spikes_nrc_test.values.flatten('F'), axis=0)
      # Load cache.
      if (c,a,g,'beta') in self.f_pop_beta_cag:
        beta = self.f_pop_beta_cag[(c,a,g,'beta')]
      else:
        beta = None
      if (c,a,g,'baseline') in self.f_pop_beta_cag:
        beta_baseline = self.f_pop_beta_cag[(c,a,g,'baseline')]
      else:
        beta_baseline = None

      log_lmbda_train, par_train = self.fit_model.poisson_regression_smoothing_spline(
          spikes_nrc_train, spike_train_time_line, constant_fit=False,
          lambda_tuning=eta_smooth_tuning,
          beta_initial=beta, beta_baseline_initial=beta_baseline,
          basis=self.f_basis, Omega=self.f_Omega,
          verbose=0, verbose_warning=False)

      nll_train += par_train[5]
      log_lambda_test = log_lmbda_train.reshape(-1)
      nll_test += self.spike_trains_neg_log_likelihood(log_lambda_test, spikes_nrc_test)

    return nll_train, nll_test


  def eta_smoothing_cross_validation(
      self,
      tuning_par_list,
      num_fold=5,
      include_non_cross_groups=False,
      verbose=False):
    """Select the smoothing tuning parameter using CV."""
    nll_cv_train = np.zeros([len(tuning_par_list), num_fold])
    nll_cv_test = np.zeros([len(tuning_par_list), num_fold])
    kfold = self.shuffle_trials_groups(num_fold=num_fold, verbose=False)
    for t, eta_smooth_tuning in enumerate(tuning_par_list):
      print('tuning_par', t, eta_smooth_tuning)
      for k, (trials_groups_train, trials_groups_test) in enumerate(kfold):
        nll_train_sumc, nll_test_sumc = 0, 0

        for c in range(self.num_conditions):
          nll_train, nll_test = self.f_pop_cross_validation(
              c, trials_groups_train, trials_groups_test,
              eta_smooth_tuning=eta_smooth_tuning,
              include_non_cross_groups=include_non_cross_groups, verbose=verbose)
          nll_train_sumc += nll_train
          nll_test_sumc += nll_test
          # return nll_cv_train, nll_cv_test

        nll_cv_train[t, k] = nll_train_sumc
        nll_cv_test[t, k] = nll_test_sumc

    return nll_cv_train, nll_cv_test


  def unit_correction(
      self,
      lambda_r,
      spike_counts_r,
      spike_trains_r,
      f_peak1,
      f_peak2,
      correction='B',
      log_fr_offset=0,
      verbose=False):
    """Correct the polulation activity for each neuron.

    Correct for all trials together. This does not affect the trial-to-trial
    correlation, and it does not influence the trial-to-trial features, but it
    used to improve the goodness-of-fit.

    Args:
      lambda_r: num_trials x num_bins lambdas for all trials for a unit.
      spike_trains_r: DataFrame num_trials x num_bins spike trains for all
          trials for a unit.
    """
    num_trials, num_bins = lambda_r.shape
    spike_trains_r = np.stack(spike_trains_r, axis=0)
    log_lambda_r = np.log(lambda_r + np.finfo(float).eps)

    # Correct the baseline (or gain).
    if correction in ['B', 'BS', 'BSS']:
      log_lambda_mean = np.mean(log_lambda_r)
      log_spike_count_mean= np.log(np.mean(spike_trains_r))
      log_lambda_correction = log_spike_count_mean - log_lambda_mean + log_fr_offset
      log_lambda_r = log_lambda_r + log_lambda_correction
      # return np.exp(log_lambda_r)

    # Baseline correction in different segments.
    if correction in ['BB', 'BBS', 'BBSS']:
      peak1_range = self.spike_train_time_line < 0.15
      peak2_range = self.spike_train_time_line >= 0.15
      log_lambda_mean1 = np.mean(log_lambda_r[:, peak1_range])
      log_lambda_mean2 = np.mean(log_lambda_r[:, peak2_range])
      log_spike_count_mean1 = np.log(np.mean(spike_trains_r[:, peak1_range]))
      log_spike_count_mean2 = np.log(np.mean(spike_trains_r[:, peak2_range]))
      log_lambda_correction1 = log_spike_count_mean1 - log_lambda_mean1
      log_lambda_correction2 = log_spike_count_mean2 - log_lambda_mean2
      log_lambda_r[:, peak1_range] = (log_lambda_r[:, peak1_range] +
          log_lambda_correction1)
      log_lambda_r[:, peak2_range] = (log_lambda_r[:, peak2_range] +
          log_lambda_correction2)
      # log_lambda_r[:, peak1_range] = (log_lambda_r[:, peak1_range] +
      #     log_lambda_correction1 - 0.1)
      # log_lambda_r[:, peak2_range] = (log_lambda_r[:, peak2_range] +
      #     log_lambda_correction2 - 0.1)

    # Peak-1.
    if correction in ['S', 'SS', 'BS', 'BSS', 'BBS', 'BBSS']:
      source = [0, f_peak1, 0.18, f_peak2, 0.4]
      search_left1 = max(f_peak1-0.05, 0.02)
      search_right1 = min(f_peak1+0.08, 0.16)
      step_size = 0.002
      peaks1 = np.linspace(search_left1, search_right1,
          int((search_right1 - search_left1 + step_size) / step_size))
      log_likelihoods1 = np.zeros(len(peaks1))
      log_lambda_cnd = np.ones(lambda_r.shape)
      for index, peak in enumerate(peaks1):
        target1 = source.copy()
        target1[1] = peak
        for r in range(num_trials):
          log_lambda_cnd[r] = self.linear_time_warping(self.spike_train_time_line,
              log_lambda_r[r], source, target1, verbose=False)
        log_prior = scipy.stats.multivariate_normal.logpdf(
            peak, mean=f_peak1, cov=2e-4)
        # log_prior = 0
        log_likelihoods1[index] = - self.spike_trains_neg_log_likelihood(
            log_lambda_cnd, spike_trains_r) + log_prior
      max_ind = np.argmax(log_likelihoods1)
      target1 = source.copy()
      target1[1] = peaks1[max_ind]
      # log_lambda_cnd = np.ones(lambda_r.shape)
      for r in range(num_trials):
        log_lambda_r[r] = self.linear_time_warping(self.spike_train_time_line,
            log_lambda_r[r], source, target1, verbose=False)

    # Peak-2
    if correction in ['SS', 'BSS', 'BBSS']:
      source = [0, f_peak1, 0.13, f_peak2, 0.4]
      search_left2 = max(f_peak2-0.08, 0.14)
      search_right2 = min(f_peak2+0.08, 0.35)
      step_size = 0.002
      peaks2 = np.linspace(search_left2, search_right2,
          int((search_right2 - search_left2 + step_size) / step_size))
      log_likelihoods2 = np.zeros(len(peaks2))
      log_lambda_cnd = np.ones(lambda_r.shape)
      for index, peak in enumerate(peaks2):
        target2 = source.copy()
        target2[3] = peak
        for r in range(num_trials):
          log_lambda_cnd[r] = self.linear_time_warping(self.spike_train_time_line,
              log_lambda_r[r], source, target2, verbose=False)
        log_prior = scipy.stats.multivariate_normal.logpdf(
            peak, mean=f_peak2, cov=1e-4)
        # log_prior = 0
        log_likelihoods2[index] = - self.spike_trains_neg_log_likelihood(
            log_lambda_cnd, spike_trains_r) + log_prior
      max_ind = np.argmax(log_likelihoods2)
      target2 = source.copy()
      target2[3] = peaks2[max_ind]
      log_lambda_cnd = np.ones(lambda_r.shape)
      for r in range(num_trials):
        log_lambda_r[r] = self.linear_time_warping(self.spike_train_time_line,
            log_lambda_r[r], source, target2, verbose=False)

    if verbose:
      plt.figure(figsize=[4.5,1.5])
      if correction in ['S', 'SS', 'BS', 'BSS']:
        plt.subplot(121)
        plt.plot(peaks1, log_likelihoods1)
        plt.axvline(x=f_peak1, ls='--', color='k')
      if correction in ['SS', 'BSS']:
        plt.subplot(122)
        plt.plot(peaks2, log_likelihoods2)
        plt.axvline(x=f_peak2, ls='--', color='k')
      plt.tight_layout()
      plt.show()

    lambda_r = np.exp(log_lambda_r)
    return lambda_r


  def ks_test(
      self,
      clist,
      lambda_garc,
      z_c,
      test_size=0.05,
      correction='B',
      null_type='theoretical',
      num_null_samples=1000):
    """KS test to verify the model.

    Args:
      lambda_garc:
      spike_times: neurons x trials.
      z_c: neurons x c.
      correction: 'B', 'BS', 'BSS'
      null_type: 'theoretical', 'sampling'
    """
    # For loop over c -> a -> g -> n -> r.
    spike_times = self.spike_times
    spike_trains = self.spike_trains
    CI_trap_cnt = 0
    CI_cnt = 0
    zero_spk_unit_cnt = 0

    trange = tqdm(clist, ncols=100, file=sys.stdout)
    for c in trange:
      z = z_c[c]
      trials_df = self.trials_groups.get_group(self.map_c_to_cid[c])
      trials_indices = trials_df.index.values
      for a, probe in enumerate(self.probes):
        sub_group_df = self.sub_group_df_c[0]
        area_units = sub_group_df[sub_group_df['probe'] == probe].index.values
        for g in range(self.num_groups):
          group_units = z.loc[area_units]
          group_units = group_units[group_units.values == g].index.values
          # ------- Uncomment to debug -------
          # if g != 0:
          #   continue
          # if c != 3 or a != 0 or g != 0:
          #   continue
          # file_path = f'D:/Brain_Network/Output/fig/{c}_{a}_{g}_spikes.pdf'
          # self.plot_spikes(spike_times.loc[group_units, trials_indices], file_path)
          # file_path = f'D:/Brain_Network/Output/fig/{c}_{a}_{g}_lambdas.pdf'
          # self.plot_lambdas(self.spike_train_time_line * 1000,
          #                   lambda_garc[g,a,:,c], file_path)
          # --------------------------
          for n, neuron_id in enumerate(group_units):
            # ------- Uncomment to debug -------
            # if n != 12 and n != 15:
            #   continue
            # --------------------------
            u_list = []
            if correction is not None and g in [0]:
              spike_counts_r = self.spike_counts_c[c].loc[neuron_id, trials_indices]
              spike_trains_r = spike_trains.loc[neuron_id, trials_indices]
              lambdas_r = self.unit_correction(
                  lambda_garc[g,a,:,c], spike_counts_r, spike_trains_r,
                  self.f_peak1_ac[a,c], self.f_peak2_ac[a,c],
                  correction=correction, log_fr_offset=-0.1, verbose=False)
            else:
              lambdas_r = lambda_garc[g,a,:,c]
            # ------- Uncomment to debug -------
            # spike_counts_r = self.spike_counts_c[c].loc[neuron_id, trials_indices]
            # spike_trains_r = spike_trains.loc[neuron_id, trials_indices]
            # # self.plot_spike_trains(self.spike_train_time_line * 1000, spike_trains_r)
            # # self.plot_spikes(spike_times.loc[[neuron_id], trials_indices])
            # # self.plot_lambdas(self.spike_train_time_line * 1000,
            # #                   lambda_garc[g,a,:,c])
            # self.plot_spikes_psth_lambda(
            #     self.spike_train_time_line * 1000,
            #     spike_times.loc[[neuron_id], trials_indices],
            #     spike_trains_r,
            #     lambda_garc[g,a,:,c])
            # return
            # --------------------------

            for r, trial_id in enumerate(trials_indices):
              spikes = spike_times.loc[neuron_id, trial_id]
              lmbd = lambdas_r[r]
              u_vals = self.ks_test_interval_transform(spikes, lmbd, dt=self.dt)
              u_list.extend(u_vals)

            # For no spike neurons, if it is grouped into `2`, then it is
            # correct, otherwise, wrong.
            if len(u_list) == 0 and g == 2:
              CI_trap_cnt += 1
              CI_cnt += 1
              zero_spk_unit_cnt += 1
              continue
            elif len(u_list) == 0 and g != 2:
              CI_trap_cnt += 0
              CI_cnt += 1
              zero_spk_unit_cnt += 1
              continue
            # The theoretical null CDF is biased due to limited trial lenght.
            # This function will correct the bias.
            if null_type == 'sampling':
              _, null_cdf = self.get_ks_test_null_cdf(
                  lmbd, t_end=self.spike_train_time_line[-1]+self.dt,
                  num_trials=num_null_samples, bin_width=0.02)
            elif null_type == 'theoretical':
              null_cdf = None
            CI_trap = self.check_ks(u_list, null_cdf=null_cdf, bin_width=0.02,
                test_size=test_size, verbose=False)
            CI_trap_cnt += CI_trap
            CI_cnt += 1
            #r
          #n
        #g
      #a
    #c
    CI_trap_ratio = CI_trap_cnt / CI_cnt
    print(f'CI cnt {CI_cnt}  CI trap ratio {CI_trap_ratio}')
    print(f'zero neuron:{zero_spk_unit_cnt}')

    # Save results.
    if not hasattr(self, 'ks_test_CI_trap_cnt'):
      self.ks_test_CI_trap_cnt = {}
    if not hasattr(self, 'ks_test_CI_cnt'):
      self.ks_test_CI_cnt = {}

    self.ks_test_CI_trap_cnt[test_size] = CI_trap_cnt
    self.ks_test_CI_cnt[test_size] = CI_cnt

    return


  def ks_test_group(
      self,
      clist,
      lambda_garc,
      z_c,
      targets_arc=None,
      test_size=0.05,
      null_type='theoretical',
      fixed_censor_interval=None,
      num_null_samples=1000):
    """KS test to verify the model.

    Args:
      lambda_garc:
      spike_times: neurons x trials.
      z_c: neurons x c.
      null_type: 'theoretical', 'sampling'.
      fixed_censor_interval: [t1, t2]. Only use spikes between t1 and t2.
      targets_arc: used to select censor range. lower priority than
          `fixed_censor_interval`.
    """
    # For loop over c -> a -> g -> n -> r.
    g = 0  # Only focus on cross-pop, the main subgroup.
    spike_times = self.spike_times
    spike_trains = self.spike_trains
    CI_trap_cnt = 0
    CI_cnt = 0
    zero_spk_unit_cnt = 0
    ks_curves_arc = {}

    # trange = tqdm(clist, ncols=100, file=sys.stdout)
    trange = clist
    for c in trange:
      z = z_c[c]
      trials_df = self.trials_groups.get_group(self.map_c_to_cid[c])
      trials_indices = trials_df.index.values
      for a, probe in enumerate(self.probes):
        sub_group_df = self.sub_group_df_c[0]
        area_units = sub_group_df[sub_group_df['probe'] == probe].index.values
        group_units = z.loc[area_units]
        group_units = group_units[group_units.values == g].index.values

        for r, trial_id in enumerate(trials_indices):
          u_list = []
          lambdas_r = lambda_garc[g,a,r,c]

          for n, neuron_id in enumerate(group_units):
            spikes = spike_times.loc[neuron_id, trial_id]
            # Limit spikes into a range.
            if fixed_censor_interval is not None:
              spikes = spikes[(spikes>=fixed_censor_interval[0]) &
                              (spikes<=fixed_censor_interval[1])]
              u_vals = self.ks_test_interval_transform(spikes, lambdas_r,
                  t_end=None, dt=self.dt)
              u_list.extend(u_vals)
            elif targets_arc is not None:
              w = 0.05
              p1_l = max(0, targets_arc[a,r,c,1]-w)
              p1_r = targets_arc[a,r,c,1]+w
              p2_l = targets_arc[a,r,c,4]-w
              p2_r = min(0.4, targets_arc[a,r,c,4]+w)

              if p1_r >= p2_l:
                # p1 p2 overlap.
                # print(p1_l, p2_r)
                spikes = spikes[(spikes>=p1_l) & (spikes<=p2_r)]
                u_vals = self.ks_test_interval_transform(spikes, lambdas_r,
                    t_end=None, dt=self.dt)
                u_list.extend(u_vals)
              else:
                # P1 P2 separate. Run KS twice.
                # print(p1_l, p1_r, p2_l, p2_r)
                spikes_p1 = spikes[(spikes>=p1_l) & (spikes<=p1_r)]
                spikes_p2 = spikes[(spikes>=p2_l) & (spikes<=p2_r)]
                if len(spikes_p1) == 0  and len(spikes_p2) == 0:
                  u_vals = self.ks_test_interval_transform(spikes, lambdas_r,
                      t_end=None, dt=self.dt)
                  u_list.extend(u_vals)
                else:
                  u_vals = self.ks_test_interval_transform(spikes_p1, lambdas_r,
                      t_end=None, dt=self.dt)
                  u_list.extend(u_vals)
                  u_vals = self.ks_test_interval_transform(spikes_p2, lambdas_r,
                      t_end=None, dt=self.dt)
                  u_list.extend(u_vals)

            else:
              u_vals = self.ks_test_interval_transform(spikes, lambdas_r,
                  t_end=None, dt=self.dt)
              u_list.extend(u_vals)

          # The theoretical null CDF is biased due to limited trial lenght.
          # This function will correct the bias.
          if null_type == 'sampling':
            _, null_cdf = self.get_ks_test_null_cdf(
                lambdas_r, t_end=None,
                num_trials=num_null_samples, bin_width=0.02,
                censor_interval=censor_interval)
          elif null_type == 'theoretical':
            null_cdf = None
          CI_trap, mcdf, ecdf, CI_up, CI_dn = self.check_ks(u_list,
              test_size, bin_width=0.02, null_cdf=null_cdf, verbose=False)
          ks_curves_arc[(a,r,c)] = (mcdf, ecdf, CI_up, CI_dn)
          if CI_trap is None:
            continue

          CI_trap_cnt += CI_trap
          CI_cnt += 1
          #n
        #r
      #a
    #c
    CI_trap_ratio = CI_trap_cnt / CI_cnt
    print(f'CI cnt {CI_cnt}  CI trap ratio {CI_trap_ratio}')
    print(f'zero neuron:{zero_spk_unit_cnt}')

    # Save results.
    # if not hasattr(self, 'ks_test_CI_trap_cnt'):
    #   self.ks_test_CI_trap_cnt = {}
    # if not hasattr(self, 'ks_test_CI_cnt'):
    #   self.ks_test_CI_cnt = {}

    # self.ks_test_CI_trap_cnt[test_size] = CI_trap_cnt
    # self.ks_test_CI_cnt[test_size] = CI_cnt

    return ks_curves_arc


  def plot_cross_pop_ks_curves_grid(
      self,
      ks_curves_arc,
      green_highlight={},
      red_highlight={},
      output_dir=None):
    """Plot KS curves in a grid."""
    areas_names = ['V1', 'LM', 'AL']
    num_conditions, num_trials = 13, 15
    num_rows, num_cols = num_conditions, num_trials

    for aid in range(self.num_areas):
      gs_kw = dict(height_ratios=[1]*num_rows, width_ratios=[1]*num_cols)
      fig, axs = plt.subplots(figsize=(12, 10), gridspec_kw=gs_kw,
          nrows=num_rows, ncols=num_cols)
      plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)

      for cid in range(num_conditions):  # row index.
        for rid in range(num_trials):  # column index.
          mcdf, ecdf, CI_up, CI_dn = ks_curves_arc[aid,rid,cid]

          ax = fig.add_subplot(axs[cid,rid])
          ax.axis('equal')
          ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
          plt.plot(mcdf, ecdf, 'k')
          plt.plot(mcdf, CI_up, 'k:', lw=0.7)
          plt.plot(mcdf, CI_dn, 'k:', lw=0.7)
          plt.xlim(0, 1)
          plt.ylim(0, 1)

          # Highligh background.
          if green_highlight is not None and (aid,rid,cid) in green_highlight:
            ax.set_facecolor('lightgreen')
          if red_highlight is not None and (aid,rid,cid) in red_highlight:
            ax.set_facecolor('lightpink')

          if cid == num_rows-1 and rid == 0:
            ax.tick_params(left=True, labelleft=True, bottom=True, labelbottom=True)
            plt.xticks([0, 1], [0, 1])
            plt.yticks([0, 1], [0, 1])
          if rid == 0:
            plt.text(-0.7, 0.5, f'{self.map_c_to_cid[cid]}', transform=ax.transAxes)
          if cid == 0:
            plt.text(0.47, 1.1, f'{rid}', transform=ax.transAxes)
          if rid == 0 and cid == 0:
            plt.text(0.5, 1.5, 'Trials', fontsize=15, transform=ax.transAxes)
          if cid == 2 and rid == 0:
            plt.text(-1.2, 0.5, f'Conditions', fontsize=15, rotation=90, transform=ax.transAxes)
          if rid == 7 and cid == 0:
            plt.text(0.5, 1.5, f'{areas_names[aid]}', fontsize=18, transform=ax.transAxes)

      if output_dir:
        figure_path = os.path.join(output_dir,
            f'{self.session_id}_{areas_names[aid]}_cross_pop_KS_alpha_01.pdf')
        plt.savefig(figure_path) # bbox_inches='tight'
        print('save figure:', figure_path)
      plt.show()


  def plot_group_templates_individual_fit(
      self,
      lambda_garc,
      z_c,
      a,
      r,
      c,
      null_type='theoretical',
      test_size=0.05,
      ylim=None,
      output_dir=True):
    """Exploration function for per trial group activity.

    Args:
        fit_type: 'bspline', 'quad', 'kernel', 'model'
    """
    areas_names = ['V1', 'LM', 'AL']
    ylim_up = [80, 60, 50]
    g = 0
    spike_trains = self.spike_trains
    spike_times = self.spike_times
    trials_groups = self.trials_groups
    probes = self.probes
    spike_train_time_line = self.spike_train_time_line
    time_line = spike_train_time_line * 1000
    time_window = [spike_train_time_line[0], spike_train_time_line[-1]+self.dt]
    sub_group_df = self.sub_group_df_c[c]
    trials_indices = trials_groups.get_group(self.map_c_to_cid[c]).index.values
    trial_id = trials_indices[r]
    z = z_c[c]
    probe = probes[a]
    sub_group_df = self.sub_group_df_c[0]
    area_units = sub_group_df[sub_group_df['probe'] == probe].index.values
    group_units = z.loc[area_units]
    group_units = group_units[group_units.values == g].index.values

    # Individual KS test.
    censor_interval = [0.0, 0.35]
    u_list = []
    lambdas_r = lambda_garc[g,a,r,c]

    # Manual correction.
    # log_lambda = np.log(lambdas_r)
    # log_lambda = log_lambda + 0
    # f_samples[s] = hsm.HierarchicalSamplingModel.linear_time_warping(
    #     spike_train_time_line, log_lambda,
    #     sources[s,a,r,c], targets[s,a,r,c], verbose=False)
    # lambdas_r = np.exp(log_lambda)

    for n, neuron_id in enumerate(group_units):
      spikes = spike_times.loc[neuron_id, trial_id]
      # Limit spikes into a range.
      if censor_interval is not None:
        spikes = spikes[(spikes>=censor_interval[0]) &
                        (spikes<=censor_interval[1])]
      u_vals = self.ks_test_interval_transform(spikes, lambdas_r, t_end=None,
        dt=self.dt)
      u_list.extend(u_vals)

    if null_type == 'sampling':
      _, null_cdf = self.get_ks_test_null_cdf(
          lambdas_r, t_end=self.spike_train_time_line[-1]+self.dt,
          num_trials=num_null_samples, bin_width=0.02,
          censor_interval=censor_interval)
    elif null_type == 'theoretical':
      null_cdf = None
    CI_trap, mcdf, ecdf, CI_up, CI_dn = self.check_ks(u_list,
        test_size, bin_width=0.02, null_cdf=null_cdf, verbose=True)

    spikes = spike_times.loc[group_units, trial_id].values.reshape(-1)
    bin_width = 0.01
    psth, bins = util.bin_spike_times(spikes, bin_width, time_window[1])
    bins = bins * 1000
    print(f'lambda mean {np.mean(lambdas_r)/self.dt}   spk mean {np.mean(psth)/bin_width}')

    psth_raw, bins_kernel = util.bin_spike_times(spikes, 0.002, time_window[1])
    psth_raw = psth_raw.mean(axis=0)
    bins_kernel = bins_kernel * 1000
    psth_kernel = scipy.ndimage.gaussian_filter1d(psth_raw, 3)/0.002

    # Firing rate intensity functions example.
    gs_kw = dict(width_ratios=[1])
    fig, axs = plt.subplots(figsize=(4, 2), gridspec_kw=gs_kw, nrows=1, ncols=1)
    ax = fig.add_subplot(axs)
    trial_id = trials_indices[r]
    psth = psth.mean(axis=0) / bin_width
    plt.bar(bins, psth, width=bin_width*1000, color='lightgrey', label='spike train')
    plt.plot(time_line, lambdas_r / self.dt, 'k', lw=2, label='fitted activity')
    # plt.plot(bins_kernel, psth_kernel, 'g')
    if ylim is not None:
      plt.ylim(ylim)
    else:
      plt.ylim(0, ylim_up[a])
    ax.set_title(f'{areas_names[a]}  condition: {self.map_c_to_cid[c]}  trial: {r}',
        fontsize=12)
    plt.xlabel('Time [ms]')
    plt.ylabel('Firing rate [spikes/sec]')
    if output_dir is not None:
      output_figure_path = os.path.join(output_dir,
          f'{self.session_id}_c{c}_r{r}_{areas_names[a]}_trial_demo.pdf')
      plt.savefig(output_figure_path, bbox_inches='tight')
      print('Save figure to: ', output_figure_path)
    plt.show()


  def plot_group_templates_fits(
      self,
      lambda_garc,
      z_c,
      rs,
      c,
      ylims=[90, 80, 130],
      output_dir=True):
    """Exploration function for per trial group activity.

    Args:
        fit_type: 'bspline', 'quad', 'kernel', 'model'
    """
    areas_names = ['V1', 'LM', 'AL']
    g = 0
    spike_trains = self.spike_trains
    spike_times = self.spike_times
    trials_groups = self.trials_groups
    probes = self.probes
    spike_train_time_line = self.spike_train_time_line
    time_line = spike_train_time_line * 1000
    time_window = [spike_train_time_line[0], spike_train_time_line[-1]+self.dt]
    sub_group_df = self.sub_group_df_c[c]
    trials_indices = trials_groups.get_group(self.map_c_to_cid[c]).index.values
    z = z_c[c]
    num_rows = len(rs)

    gs_kw = dict(width_ratios=[1]*self.num_areas, height_ratios=[1]*num_rows)
    fig, axs = plt.subplots(figsize=(10, 1.2*num_rows), gridspec_kw=gs_kw,
        nrows=num_rows, ncols=self.num_areas)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.2)

    for row, r in enumerate(rs):
      trial_id = trials_indices[r]
      for a, probe in enumerate(self.probes):
        probe = probes[a]
        sub_group_df = self.sub_group_df_c[0]
        area_units = sub_group_df[sub_group_df['probe'] == probe].index.values
        group_units = z.loc[area_units]
        group_units = group_units[group_units.values == g].index.values

        lambdas_r = lambda_garc[g,a,r,c]
        spikes = spike_times.loc[group_units, trial_id].values.reshape(-1)
        bin_width = 0.01
        psth, bins = util.bin_spike_times(spikes, bin_width, time_window[1])
        bins = bins * 1000

        ax = fig.add_subplot(axs[row, a])
        ax.tick_params(labelbottom=False, bottom=True, direction='in')
        trial_id = trials_indices[r]
        psth = psth.mean(axis=0) / bin_width
        plt.bar(bins, psth, width=bin_width*1000, color='lightgrey', label='spike train')
        plt.plot(time_line, lambdas_r / self.dt, 'k', lw=2, label='fitted activity')
        if ylims is not None:
          plt.ylim(0, ylims[a])
        plt.xticks([0,100, 200, 300,400,500], [0,100, 200, 300,400,500])
        if a == 0:
          plt.yticks([0,30,60],[0,30,60])
        elif a == 1:
          # plt.yticks([0,30,60],[0,30,60])
          plt.yticks([0,20,40],[0,20,40])
        elif a == 2:
          # plt.yticks([0,50,100],[0,50,100])
          plt.yticks([0,30,60],[0,30,60])

        plt.text(0.6, 0.85,
            f'{areas_names[a]}  trial: {trial_id}',
            fontsize=8, transform=ax.transAxes)

        if row == num_rows-1 and a == 0:
          plt.xlabel('Time [ms]')
          ax.tick_params(labelbottom=True, bottom=True, direction='in')
        if row == num_rows-2 and a == 0:
          plt.ylabel('Firing rate [spikes/sec]')

    if output_dir is not None:
      output_figure_path = os.path.join(output_dir,
          f'{self.session_id}_c{c}_all_ares_trials_demo.pdf')
      plt.savefig(output_figure_path, bbox_inches='tight')
      print('Save figure to: ', output_figure_path)
    plt.show()


  @classmethod
  def plot_spikes(
      cls,
      spike_times,
      output_file=None):
    """Plots spike trains.

    Args:
      spike_times: num_units x num_trials DataFrame of spike times.
    """
    num_units, num_trials = spike_times.shape
    units_ids = spike_times.index.values
    trials_indices = spike_times.columns.values

    plt.figure(figsize=[5, num_units*0.6])
    ax = plt.gca()
    ax.tick_params(left=False, labelleft=False, labelbottom=True, bottom=True,
                   top = False, labeltop=False)

    for n, unit_id in enumerate(units_ids):
      for r, trial_id in enumerate(trials_indices):
        spikes = spike_times.loc[unit_id, trial_id] * 1000
        y_values = np.zeros(spikes.shape) - (r / num_trials + n)
        plt.plot(spikes, y_values, 's', markersize=0.5,
                 color='r' if n % 2 == 0 else 'blue')

      plt.text(510, -n - 0.6, n, fontsize=12)
    plt.xlim(0, 500)
    if output_file is not None:
      plt.savefig(output_file)
      print('Save file to: ', output_file)
      plt.close()
    else:
      plt.show()


  @classmethod
  def plot_spike_trains(
      cls,
      spike_train_time_line,
      spike_trains):
    """Plots spike trains.

    Args:
      spike_trains: DataFrame.
    """
    spike_trains = np.stack(spike_trains, axis=0)
    plt.figure(figsize=(5,1.5))
    ax = plt.gca()
    ax.tick_params(left=False, labelleft=False, labelbottom=True, bottom=True,
                   right=True, labelright=True, top = False, labeltop=False)
    plt.plot(spike_train_time_line, spike_trains.mean(axis=0))
    plt.xlim(0, 500)


  @classmethod
  def plot_lambdas(
      cls,
      spike_train_time_line,
      lambdas,
      output_file=None):
    """Plots spike trains.

    Args:
      lambdas: num_trials x time_bins.
    """
    num_trials, num_bins = lambdas.shape
    colors = matplotlib.pylab.cm.jet(np.linspace(0, 1, num_trials))
    plt.figure(figsize=[5, 1.5])
    ax = plt.gca()
    ax.tick_params(left=False, labelleft=False, labelbottom=True, bottom=True,
                   top = False, labeltop=False)
    for i in range(num_trials):
      plt.plot(spike_train_time_line, lambdas[i], color=colors[i])
    plt.xlim(0, 500)
    if output_file is not None:
      plt.savefig(output_file)
      print('Save file to: ', output_file)
      plt.close()
    else:
      plt.show()


  @classmethod
  def plot_spikes_psth_lambda(
      cls,
      spike_train_time_line=None,
      spike_times=None,
      spike_trains=None,
      lambdas=None):
    """Plots spike trains, psth, and lambda for one intensity.

    Args:
      spike_times: num_units x num_trials DataFrame of spike times.
    """
    # spike times.
    num_units, num_trials = spike_times.shape
    units_ids = spike_times.index.values
    trials_indices = spike_times.columns.values
    # PSTH.
    spike_trains = np.stack(spike_trains, axis=0)
    # lambd - firing rate.
    num_trials, num_bins = lambdas.shape
    colors = matplotlib.pylab.cm.jet(np.linspace(0, 1, num_trials))

    gs_kw = dict(height_ratios=[0.4, 1])
    fig, axs = plt.subplots(figsize=(6, 4), gridspec_kw=gs_kw,
        nrows=2, ncols=1)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)

    ax = fig.add_subplot(axs[0])
    ax.tick_params(left=False, labelleft=False, labelbottom=False, bottom=True,
                   top=False, labeltop=False, direction='in')
    for n, unit_id in enumerate(units_ids):
      for r, trial_id in enumerate(trials_indices):
        spikes = spike_times.loc[unit_id, trial_id] * 1000
        y_values = np.zeros(spikes.shape) - (r / num_trials + n)
        plt.plot(spikes, y_values, 's', markersize=1,
                 color='k' if n % 2 == 0 else 'blue')
      plt.text(510, -n - 0.6, f'n:{n}', fontsize=12)
    plt.xlim(0, 500)

    ax = fig.add_subplot(axs[1])
    ax.tick_params(left=False, labelleft=False, labelbottom=False, bottom=True,
                   top=False, labeltop=False, direction='in')
    plt.plot(spike_train_time_line, spike_trains.mean(axis=0), 'grey')
    for i in range(num_trials):
      plt.plot(spike_train_time_line, lambdas[i], color=colors[i])
    plt.xlim(0, 500)

    # ax = fig.add_subplot(axs[2])
    # for i in range(num_trials):
    #   plt.plot(spike_train_time_line, lambdas[i], color=colors[i])
    # plt.xlim(0, 500)


  @classmethod
  def check_ks(
      cls,
      u_list,
      test_size=0.05,
      bin_width=0.01,
      null_cdf=None,
      lmbd=None,
      verbose=False,
      figure_path=None):
    """Plot the Q-Q curve.

    Calculation of CI bandwidth:
    https://en.wikipedia.org/wiki/Kolmogorov-Smirnov_test

    Args:
      null_cdf: Used for null distribution correction.
    """
    if len(u_list) == 0:
      return True, None, None, None, None

    bins = np.linspace(0, 1, int(1 / bin_width) + 1)
    c_alpha = np.sqrt(-np.log(test_size / 2) / 2)
    epdf, bin_edges = np.histogram(u_list, bins=bins, density=True)
    ecdf = np.cumsum(epdf) * bin_width

    if null_cdf is None:
      mcdf = bin_edges[1:]
    else:
      mcdf = null_cdf

    CI_up = mcdf + c_alpha/np.sqrt(len(u_list))
    CI_dn = mcdf - c_alpha/np.sqrt(len(u_list))
    CI_trap = ((ecdf > CI_up) | (ecdf < CI_dn)).sum()

    # if verbose and CI_trap > 0:
    if verbose:
      plt.figure(figsize=[4, 1.8])
      plt.subplot(121)
      plt.plot(mcdf, ecdf)
      plt.plot(mcdf, CI_up, 'k--')
      plt.plot(mcdf, CI_dn, 'k--')
      # plt.plot(mcdf, mcdf, 'k--')
      plt.title(f'Trap {CI_trap}')
      plt.axis([0,1,0,1])
      plt.xticks([0, 1], [0, 1])
      plt.yticks([0, 1], [0, 1])
      # plt.grid('on')
      plt.subplot(122)
      seaborn.distplot(u_list, bins=30,
          norm_hist=True, kde=False, color='grey')
      plt.plot([0, 1], [1, 1], 'k')
      # plt.ylim(0, 1.5)
      plt.xlim(0, 1)
      plt.xticks([0, 1], [0, 1])
      plt.yticks([], [])
      plt.show()

      # This part is for paper KS bias figures.
      # plt.figure(figsize=[3, 2.5])
      # seaborn.distplot(u_list, bins=30,
      #     norm_hist=True, kde=False, color='grey')
      # plt.plot([0, 1], [1, 1], 'k')
      # plt.xlim(0, 1)
      # plt.ylim(0, 1.5)
      # plt.xticks([0, 1], [0, 1])
      # plt.yticks([0, 1], [0, 1])
      # plt.title('FR=30 Hz   Len=2 s')
      # # plt.tight_layout()
      # if figure_path is not None:
      #   plt.savefig(figure_path)
      #   print('Save figure to:', figure_path)
      # plt.show()

    return CI_trap == 0, mcdf, ecdf, CI_up, CI_dn


  @classmethod
  def ks_test_interval_transform(
      cls,
      spikes,
      lmbd,
      t_end=None,
      dt=0.002,
      verbose=False):
    """Unit KS test. This is based on time rescale theorem.

    Args:
      lmbd: Intensity function. Each bin represent the integral over dt.
      spikes: Array of spike times.
    """
    # if len(spikes) < 2:
    #   return []

    # if (len(spikes) > 0 and spikes[0] != 0) or len(spikes) == 0:
    #   spikes = np.insert(spikes, 0, 0)
    if len(spikes) == 0:
      return []

    if (t_end is not None and append_interval_end and len(spikes) > 0 and
        spikes[-1] < t_end):
      spikes = np.append(spikes, t_end)

    u_list = []
    for spike_id in range(1, len(spikes)):
      interval_left = int(spikes[spike_id-1] // dt + 1) # Not include the start bin.
      # Include the end bin. Python array index does not include the last index.
      interval_right = int(spikes[spike_id] // dt)

      # Haslinger correction.
      r = np.random.rand()
      lmbd_last = lmbd[interval_right]
      p_last = 1 - np.exp(-lmbd_last)
      delta = -np.log(1 - r * p_last)
      # delta = 0  # No correction.

      tau = lmbd[interval_left:interval_right].sum() + delta
      u_val = 1 - np.exp(-tau)
      u_list.append(u_val)

    if verbose:
      plt.figure(figsize=(3,3))
      seaborn.distplot(u_list, bins=30, norm_hist=True, kde=False, color='grey')
      plt.plot([0, 1], [1, 1], 'k')

    return u_list


  @classmethod
  def get_ks_test_null_cdf(
      cls,
      lmbd,
      t_end,
      bin_width=0.01,
      num_trials=1000,
      censor_interval=None,
      verbose=False):
    """Gets the null CDF using simulation.

    Args:
      lmbd: Single trial intensity function.

    Returns:
      bins:
      cdf:
    """
    import hierarchical_model_generator
    num_bins = len(lmbd)
    u_list = []
    for i in range(num_trials):
      generator = hierarchical_model_generator.HierarchicalModelGenerator
      _, spk_times, _, _ = generator.generate_spike_train(
          lmbd, dt=0.002, random_seed=None)
      if censor_interval is not None:
        spk_times = spk_times[(spk_times>=censor_interval[0]) &
                              (spk_times<=censor_interval[1])]
      u_val = cls.ks_test_interval_transform(spk_times, lmbd, t_end=t_end,
          dt=0.002, verbose=False)
      u_list.extend(u_val)

    bins = np.linspace(0, 1, int(1 / bin_width) + 1)
    pdf, bin_edges = np.histogram(u_list, bins=bins, density=True)
    cdf = np.cumsum(pdf) * bin_width

    if verbose:
      plt.figure(figsize=[3, 3])
      plt.plot(bins[1:], cdf)

    return bins[1:], cdf
