import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

bounds_map = {'exp_c1_obs': 'Exp(c1) + Obs', 'exp_c2_obs': 'Exp(c2) + Obs', 'exp_c1_c2_obs': 'Exp(c1) + Exp(c2) + Obs',
              'manski': 'Obs'}

data_name_map = {'synthetic_continuous': 'Synthetic', 'synthetic_dis': 'Synthetic', 'IST': 'IST'}

fontsize = 42
labelsize = 42
sns.set(style="ticks")
mpl.rc('xtick', labelsize=labelsize)
mpl.rc('ytick', labelsize=labelsize)
mpl.rcParams["axes.labelweight"] = "bold"
mpl.rcParams["axes.labelsize"] = labelsize
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern']
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams["font.size"] = fontsize
mpl.rcParams['text.usetex'] = True
sns.set_style("whitegrid")

plt.grid()

method_map_ist = {'learned_obs_expc1_expc2': r'Alg 1-$l_1, l_2, l_3, l_4$',
                  'learned_obs_expc1': r'Alg 1-$l_1, l_2$',
                  'learned_obs_expc2': r'Alg 1-$l_1, l_3$',
                  'learned_obs': r'Alg 1-$l_1$',
                  'behavior_c1c2': r'Behavior'}

ipw_name_key_ist = {'learned_obs_expc1_expc2': 'mean_y_model_obs_expc1_expc2_ipw',
                    'learned_obs_expc1': 'mean_y_model_obs_expc1_ipw',
                    'learned_obs_expc2': 'mean_y_model_obs_expc2_ipw',
                    'learned_obs': 'mean_y_model_obs_ipw',
                    'behavior_c1c2': 'mean_y_behavior_ipw',
                    'random': 'mean_y_random_ipw', 'oracle': 'mean_y_oracle_ipw'}

method_map_synth = {'learned_obs_expc1_expc2': r'Alg 1-$l_1, l_2, l_3, l_4$',
                    'learned_obs_expc1': r'Alg 1-$l_1, l_2$',
                    'learned_obs_expc2': r'Alg 1-$l_1, l_3$',
                    'learned_obs': r'Alg 1-$l_1$',
                    'behavior_c1c2': r'Behavior',
                    'random': r'Random', 'oracle': r'Oracle Policy'}

method_map_synth_text = {'learned_obs_expc1_expc2': 'l1, l2, l3, l4',
                         'learned_obs_expc1': 'l1, l2',
                         'learned_obs_expc2': 'l1, l3',
                         'learned_obs': 'l1',
                         'behavior_c1c2': 'Behavior Policy',
                         'random': 'Random Policy', 'oracle': 'Oracle Policy'}

ipw_name_key_synth = {'learned_obs_expc1_expc2': 'mean_y_model_obs_expc1_expc2_ipw',
                      'learned_obs_expc1': 'mean_y_model_obs_expc1_ipw',
                      'learned_obs_expc2': 'mean_y_model_obs_expc2_ipw',
                      'learned_obs': 'mean_y_model_obs_ipw',
                      'behavior_c1c2': 'mean_y_behavior_ipw',
                      'random': 'mean_y_random_ipw', 'oracle': 'mean_y_oracle_ipw'}

palette = {'learned_obs_expc1_expc2': '#CC3311', 'learned_obs_expc1': '#EE3377', 'learned_obs_expc2': '#EE7733',
           'learned_obs': "#33157d", 'behavior_c1c2': '#009988', 'random': '#117733', 'oracle': '#000000'}

palette_mapped_synth = {method_map_synth['learned_obs_expc1_expc2']: '#CC3311',
                        method_map_synth['learned_obs_expc1']: '#EE3377',
                        method_map_synth['learned_obs_expc2']: '#EE7733',
                        method_map_synth['learned_obs']: "#33157d", method_map_synth['behavior_c1c2']: '#009988',
                        method_map_synth['random']: '#117733', method_map_synth['oracle']: '#000000'}

palette_mapped_ist = {method_map_ist['learned_obs_expc1_expc2']: '#CC3311',
                      method_map_ist['learned_obs_expc1']: '#EE3377', method_map_ist['learned_obs_expc2']: '#EE7733',
                      method_map_ist['learned_obs']: "#33157d", method_map_ist['behavior_c1c2']: '#009988'}

palette_bounds = {'manski_1': '#CC3311',
                  'manski_0': '#EE7733',
                  'exp_c1_obs_0': '#a48fd9',
                  'exp_c1_obs_1': '#33157d',
                  'exp_c2_obs_1': '#0077BB',
                  'exp_c2_obs_0': '#33BBEE',
                  'exp_c1_c2_obs_0': '#999933',
                  'exp_c1_c2_obs_1': '#117733'}

markers_dict_bounds = {'manski': "o", 'expc1': "X",
                       'expc2': "s", 'expc1c2': "^",
                       'random': "P", "oracle": "."}

markers_dict = {'learned_obs_expc1_expc2': "o",
                'learned_obs_expc1': ">",
                'learned_obs_expc2': "D",
                'learned_obs': "<",
                'behavior_c1c2': "X",
                'random': "P", "oracle": "."}
linestyle_map = {'learned_obs_expc1_expc2': 'solid',
                 'learned_obs_expc1': 'solid', 'learned_obs_expc2': 'solid',
                 'learned_obs': 'solid',
                 'behavior_c1c2': 'dashed',
                 'random': ':', 'oracle': "dashed"}
lines_dict = {0: 'dashed', 1: 'solid', 2: 'densely dotted', 3: 'densely dashed', 4: 'dotted', 5: 'dashdotdotted',
              6: 'dashed', 7: 'loosely dotted'}

markers_dict_treatment = {0: "o", 1: "X", 2: "s", 3: "^", 4: "*", 5: "P", 6: "D", 7: "<", 8: "."}

width = 72
height = 84
markersize = 14


def plot_weights(weight_mat, c_mat, data_name='synthetic_continuous', scaler=None):
    n_samples, n_bounds, x_dim, maxcv = weight_mat.shape
    c_dim = c_mat.shape[1]

    b_cols = ['exp_c1_obs', 'exp_c2_obs', 'exp_c1_c2_obs', 'manski', 'cv']
    c_cols = [f'C{kk + 1}' for kk in np.arange(c_dim)]
    all_cols = [c + '_' + str(x) for c in b_cols for x in [0, 1]] + c_cols
    all_df_dict = {}
    for cv in range(maxcv):
        for i in range(x_dim):
            all_df_dict['manski_' + str(i)] = weight_mat[:, 0, i, cv]
            all_df_dict['exp_c1_obs_' + str(i)] = weight_mat[:, 1, i, cv]
            all_df_dict['exp_c2_obs_' + str(i)] = weight_mat[:, 2, i, cv]
            all_df_dict['exp_c1_c2_obs_' + str(i)] = weight_mat[:, 3, i, cv]
            all_df_dict['cv_' + str(i)] = cv * np.ones(n_samples)

    for i in range(c_dim):
        if scaler is not None:
            all_df_dict[c_cols[i]] = scaler.inverse_transform(c_mat[:, i].reshape(-1, 1))[:, 0]
        else:
            all_df_dict[c_cols[i]] = c_mat[:, i]

    all_df = pd.DataFrame(all_df_dict)

    if c_dim == 1:  # true only for IST dataset
        _, ax = plt.subplots(4, x_dim, figsize=(width, height))
        for i in range(c_dim):
            for j in range(x_dim):
                sns.lineplot(x=c_cols[i], y='manski_' + str(j),
                             data=all_df[['manski_' + str(j), c_cols[i], 'cv_' + str(j)]],
                             ax=ax[0, j], lw=2,
                             linestyle=lines_dict[j], color=palette_bounds['manski_%d' % j], estimator='mean',
                             errorbar='se',
                             markers=True, marker=markers_dict_treatment[j], markersize=markersize)
                sns.lineplot(x=c_cols[i], y='exp_c1_obs_' + str(j),
                             data=all_df[['exp_c1_obs_' + str(j), c_cols[i], 'cv_' + str(j)]],
                             ax=ax[1, j], lw=2,
                             linestyle=lines_dict[j], color=palette_bounds['exp_c1_obs_%d' % j], estimator='mean',
                             errorbar='se',
                             markers=True, marker=markers_dict_treatment[j], markersize=markersize)
                sns.lineplot(x=c_cols[i], y='exp_c2_obs_' + str(j),
                             data=all_df[['exp_c2_obs_' + str(j), c_cols[i], 'cv_' + str(j)]],
                             ax=ax[2, j], lw=2,
                             linestyle=lines_dict[j], color=palette_bounds['exp_c2_obs_%d' % j], estimator='mean',
                             errorbar='se',
                             markers=True, marker=markers_dict_treatment[j], markersize=markersize)
                sns.lineplot(x=c_cols[i], y='exp_c1_c2_obs_' + str(j),
                             data=all_df[['exp_c1_c2_obs_' + str(j), c_cols[i], 'cv_' + str(j)]],
                             ax=ax[3, j], lw=2,
                             linestyle=lines_dict[j], color=palette_bounds['exp_c1_c2_obs_%d' % j], estimator='mean',
                             errorbar='se',
                             markers=True, marker=markers_dict_treatment[j], markersize=markersize)

                for k in range(4):
                    ax[k, j].set_xlabel('Systolic Blood Pressure')
                    if k == 0:
                        ax[k, j].set_ylabel('Obs LB')
                    elif k == 1:
                        ax[k, j].set_ylabel('Obs + Exp(Age, Sex) LB')
                    elif k == 2:
                        ax[k, j].set_ylabel('Obs + Exp(Sys BP) LB')
                    elif k == 3:
                        ax[k, j].set_ylabel('Obs + Exp(Age, Sex) + Exp(Sys BP) LB')
                    if j == 0:
                        ax[k, j].set_title('No treatment (x=0)', fontsize=fontsize, fontweight="bold")
                    else:
                        ax[k, j].set_title('Aspirin treatment (x=1)', fontsize=fontsize, fontweight="bold")
            if not os.path.exists(os.path.join('plots', data_name)):
                os.makedirs(os.path.join('plots', data_name))
            plt.savefig(os.path.join('plots', data_name, 'bounds_plot.pdf'), dpi=300,
                        bbox_inches='tight', bbox_extra_artists=[])
    else:  # synthetic data
        _, ax = plt.subplots(4, c_dim * x_dim, figsize=(width, height))
        for i in range(c_dim):
            for j in range(x_dim):
                sns.lineplot(x=c_cols[i], y='manski_' + str(j),
                             data=all_df[['manski_' + str(j), c_cols[i], 'cv_' + str(j)]],
                             ax=ax[0, i * x_dim + j], lw=2,
                             linestyle=lines_dict[j], color=palette_bounds['manski_%d' % j], estimator='mean',
                             errorbar='se',
                             markers=True, marker=markers_dict_treatment[j], markersize=markersize)
                sns.lineplot(x=c_cols[i], y='exp_c1_obs_' + str(j),
                             data=all_df[['exp_c1_obs_' + str(j), c_cols[i], 'cv_' + str(j)]],
                             ax=ax[1, i * x_dim + j], lw=2,
                             linestyle=lines_dict[j], color=palette_bounds['exp_c1_obs_%d' % j], estimator='mean',
                             errorbar='se',
                             markers=True, marker=markers_dict_treatment[j], markersize=markersize)
                sns.lineplot(x=c_cols[i], y='exp_c2_obs_' + str(j),
                             data=all_df[['exp_c2_obs_' + str(j), c_cols[i], 'cv_' + str(j)]],
                             ax=ax[2, i * x_dim + j], lw=2,
                             linestyle=lines_dict[j], color=palette_bounds['exp_c2_obs_%d' % j], estimator='mean',
                             errorbar='se',
                             markers=True, marker=markers_dict_treatment[j], markersize=markersize)
                sns.lineplot(x=c_cols[i], y='exp_c1_c2_obs_' + str(j),
                             data=all_df[['exp_c1_c2_obs_' + str(j), c_cols[i], 'cv_' + str(j)]],
                             ax=ax[3, i * x_dim + j], lw=2,
                             linestyle=lines_dict[j], color=palette_bounds['exp_c1_c2_obs_%d' % j], estimator='mean',
                             errorbar='se',
                             markers=True, marker=markers_dict_treatment[j], markersize=markersize)

                for k in range(4):
                    ax[k, i * x_dim + j].set_xlabel(c_cols[i])
                    ax[k, i * x_dim + j].set_title(r'treatment : %d' % j, fontsize=48, fontweight="bold")
                    if k == 0:
                        ax[k, i * x_dim + j].set_ylabel(r'Obs LB')
                    elif k == 1:
                        ax[k, i * x_dim + j].set_ylabel(r'Obs + Exp($C_1$) LB')
                    elif k == 2:
                        ax[k, i * x_dim + j].set_ylabel(r'Obs + Exp($C_2$) LB')
                    elif k == 3:
                        ax[k, i * x_dim + j].set_ylabel(r'Obs + Exp($C_1$) + Exp($C_2$) LB')
            plt.legend()
            if not os.path.exists(os.path.join('plots', data_name)):
                os.makedirs(os.path.join('plots', data_name))
            plt.savefig(os.path.join('plots', data_name, 'bounds_plot.pdf'), dpi=300,
                        bbox_inches='tight', bbox_extra_artists=[])


def plot_rollout_eval(result_df, data_name, maxcv=0, eval_prop=0, eval_sort=True):
    # plot
    if data_name == 'IST':
        method_map = method_map_ist
        _, ax = plt.subplots(1, 1, figsize=(10, 8))

        sns.lineplot(x='th', y='mean_y_behavior', data=result_df, label=method_map_ist['behavior_c1c2'],
                     estimator='mean', errorbar='se', color=palette['behavior_c1c2'],
                     marker=markers_dict['behavior_c1c2'],
                     markersize=markersize, err_style='bars')
        sns.lineplot(x='th', y='mean_y_model_obs', data=result_df, label=method_map_ist['learned_obs'],
                     estimator='mean',
                     errorbar='se', color=palette['learned_obs'], marker=markers_dict['learned_obs'],
                     markersize=markersize, err_style='bars', lw=2, linestyle=linestyle_map['learned_obs'])
        sns.lineplot(x='th', y='mean_y_model_obs_expc1', data=result_df,
                     label=method_map_ist['learned_obs_expc1'],
                     estimator='mean',
                     errorbar='se', color=palette['learned_obs_expc1'], marker=markers_dict['learned_obs_expc1'],
                     markersize=markersize, err_style='bars', lw=2, linestyle=linestyle_map['learned_obs_expc1'])
        sns.lineplot(x='th', y='mean_y_model_obs_expc2', data=result_df,
                     label=method_map_ist['learned_obs_expc2'],
                     estimator='mean',
                     errorbar='se', color=palette['learned_obs_expc2'], marker=markers_dict['learned_obs_expc2'],
                     markersize=markersize, err_style='bars', lw=2, linestyle=linestyle_map['learned_obs_expc2'])
        sns.lineplot(x='th', y='mean_y_model_obs_expc1_expc2', data=result_df,
                     label=method_map_ist['learned_obs_expc1_expc2'], estimator='mean',
                     errorbar='se', color=palette['learned_obs_expc1_expc2'],
                     marker=markers_dict['learned_obs_expc1_expc2'],
                     markersize=markersize, err_style='bars', lw=2, linestyle=linestyle_map['learned_obs_expc1_expc2'])
        ax.invert_xaxis()  # comment out if not plotting w.r.t. threshold
    else:
        _, ax = plt.subplots(1, 1, figsize=(10, 8))
        method_map = method_map_synth
        #sns.lineplot(x='th', y='mean_y_random', data=result_df, label=method_map_synth['random'], estimator='mean',
        #             errorbar='se', color=palette['random'], marker=markers_dict['random'], markersize=markersize,
        #             err_style='bars', lw=2,
        #             linestyle=linestyle_map['random'])
        #sns.lineplot(x='th', y='mean_y_oracle', data=result_df, label=method_map_synth['oracle'], estimator='mean',
        #             errorbar='se', color=palette['oracle'], marker=markers_dict['oracle'], markersize=markersize,
        #             err_style='bars', lw=2,
        #             linestyle=linestyle_map['oracle'])
        sns.lineplot(x='th', y='mean_y_behavior', data=result_df, label=method_map_synth['behavior_c1c2'],
                     estimator='mean', errorbar='se', color=palette['behavior_c1c2'],
                     marker=markers_dict['behavior_c1c2'],
                     markersize=markersize, err_style='bars', lw=2, linestyle=linestyle_map['behavior_c1c2'])
        sns.lineplot(x='th', y='mean_y_model_obs', data=result_df, label=method_map_synth['learned_obs'],
                     estimator='mean',
                     errorbar='se', color=palette['learned_obs'], marker=markers_dict['learned_obs'],
                     markersize=markersize, err_style='bars', lw=2, linestyle=linestyle_map['learned_obs'])
        sns.lineplot(x='th', y='mean_y_model_obs_expc1', data=result_df, label=method_map_synth['learned_obs_expc1'],
                     estimator='mean',
                     errorbar='se', color=palette['learned_obs_expc1'], marker=markers_dict['learned_obs_expc1'],
                     markersize=markersize, err_style='bars', lw=2, linestyle=linestyle_map['learned_obs_expc1'])
        sns.lineplot(x='th', y='mean_y_model_obs_expc2', data=result_df, label=method_map_synth['learned_obs_expc2'],
                     estimator='mean',
                     errorbar='se', color=palette['learned_obs_expc2'], marker=markers_dict['learned_obs_expc2'],
                     markersize=markersize, err_style='bars', lw=2, linestyle=linestyle_map['learned_obs_expc2'])
        sns.lineplot(x='th', y='mean_y_model_obs_expc1_expc2', data=result_df,
                     label=method_map_synth['learned_obs_expc1_expc2'], estimator='mean',
                     errorbar='se', color=palette['learned_obs_expc1_expc2'],
                     marker=markers_dict['learned_obs_expc1_expc2'],
                     markersize=markersize, err_style='bars', lw=2, linestyle=linestyle_map['learned_obs_expc1_expc2'])

        ax.invert_xaxis()
    legend = ax.legend(title='Method', loc=2, ncol=1, bbox_to_anchor=(1.02, 1), fontsize=24,
                       title_fontsize=24,
                       markerscale=1)
    if data_name == 'IST':
        legend.remove()  # save separately
        plt.xlabel('Threshold', fontsize=32)
        plt.title('Policy Evaluation I', fontdict={'fontsize': 32, 'fontweight': 'bold'})
    else:
        legend.remove()  # save separately
        plt.xlabel('Threshold', fontsize=32)
        plt.title('Policy Evaluation', fontdict={'fontsize': 32, 'fontweight': 'bold'})
    plt.ylabel('Mean outcome', fontsize=32)

    if not os.path.exists("./plots"):
        os.mkdir("./plots")
    if not os.path.exists(os.path.join("./plots/", data_name)):
        os.mkdir(os.path.join("./plots/", data_name))
    plt.savefig(os.path.join('./plots', data_name, 'rollout_threshold_plot_cv_%d.pdf' % maxcv), dpi=300,
                bbox_inches='tight')
    fig_legend = plt.figure(figsize=(24, 1.5))
    handles, labels = ax.get_legend_handles_labels()
    plt.figlegend(handles, labels, title='Method', loc='upper center', ncol=1, bbox_to_anchor=(0.5, 1.2),
                  markerscale=3, fontsize=32, title_fontsize=32)
    fig_legend.savefig(os.path.join('./plots', data_name, 'rollout_threshold_plot_legend_cv_%d.pdf' % maxcv),
                       dpi=300,
                       bbox_inches='tight')

    if eval_sort:  # only for IST
        _, ax = plt.subplots(1, 1, figsize=(10, 8))
        # sns.lineplot(x='th', y='mean_y_random_select_top', data=result_df, label=method_map['random'],
        #             estimator='mean', errorbar='se', color=palette['random'],
        #             marker=markers_dict['random'],
        #             markersize=markersize, err_style='bars', lw=2,
        #             linestyle=linestyle_map['random'])
        sns.lineplot(x='th', y='mean_y_behavior_select_top', data=result_df, label=method_map['behavior_c1c2'],
                     estimator='mean', errorbar='se', color=palette['behavior_c1c2'],
                     marker=markers_dict['behavior_c1c2'],
                     markersize=markersize, err_style='bars', lw=2,
                     linestyle=linestyle_map['behavior_c1c2'])
        sns.lineplot(x='th', y='mean_y_model_obs_select_top', data=result_df, label=method_map['learned_obs'],
                     estimator='mean',
                     errorbar='se', color=palette['learned_obs'], marker=markers_dict['learned_obs'],
                     markersize=markersize, err_style='bars', lw=2,
                     linestyle=linestyle_map['learned_obs'])
        sns.lineplot(x='th', y='mean_y_model_obs_expc1_select_top', data=result_df,
                     label=method_map['learned_obs_expc1'],
                     estimator='mean',
                     errorbar='se', color=palette['learned_obs_expc1'], marker=markers_dict['learned_obs_expc1'],
                     markersize=markersize, err_style='bars', lw=2,
                     linestyle=linestyle_map['learned_obs_expc1'])
        sns.lineplot(x='th', y='mean_y_model_obs_expc2_select_top', data=result_df,
                     label=method_map['learned_obs_expc2'],
                     estimator='mean',
                     errorbar='se', color=palette['learned_obs_expc2'], marker=markers_dict['learned_obs_expc2'],
                     markersize=markersize, err_style='bars', lw=2,
                     linestyle=linestyle_map['learned_obs_expc2'])
        sns.lineplot(x='th', y='mean_y_model_obs_expc1_expc2_select_top', data=result_df,
                     label=method_map['learned_obs_expc1_expc2'], estimator='mean',
                     errorbar='se', color=palette['learned_obs_expc1_expc2'],
                     marker=markers_dict['learned_obs_expc1_expc2'],
                     markersize=markersize, err_style='bars', lw=2, linestyle=linestyle_map['learned_obs_expc1_expc2'])

        legend = ax.legend(title='Method', loc=2, ncol=1, bbox_to_anchor=(1.02, 1), fontsize=24,
                           title_fontsize=24,
                           markerscale=1)
        if data_name == 'IST':
            legend.remove()

        plt.xlabel('Fraction treated', fontsize=32)
        plt.ylabel('Mean outcome', fontsize=32)

        plt.title('Policy Evaluation II', fontdict={'fontsize': 32, 'fontweight': 'bold'})
        if not os.path.exists("./plots"):
            os.mkdir("./plots")
        if not os.path.exists(os.path.join("./plots/", data_name)):
            os.mkdir(os.path.join("./plots/", data_name))
        plt.savefig(os.path.join('./plots', data_name, 'rollout_threshold_plot_select_top_cv_%d.pdf' % maxcv), dpi=300,
                    bbox_inches='tight', bbox_extra_artists=[])
        fig_legend = plt.figure(figsize=(24, 1.5))
        handles, labels = ax.get_legend_handles_labels()
        plt.figlegend(handles, labels, title='Method', loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.2),
                      markerscale=3, fontsize=32, title_fontsize=32)
        fig_legend.savefig(
            os.path.join('./plots', data_name, 'rollout_threshold_plot_legend_select_top_cv_%d.pdf' % maxcv),
            dpi=300,
            bbox_inches='tight')

    # ipw
    if data_name == 'IST':
        plot_df = pd.DataFrame(columns=['method', 'ipw', 'cv'])
        result_df_sub = result_df.loc[result_df['th'] == 0.0]
        result_df_sub = result_df_sub[['mean_y_model_obs_expc1_expc2_ipw',
                                       'mean_y_model_obs_expc1_ipw',
                                       'mean_y_model_obs_expc2_ipw',
                                       'mean_y_model_obs_ipw',
                                       'mean_y_behavior_ipw',
                                       'cv']]
        for m in method_map_ist.keys():
            if m == 'oracle' or m == 'random':
                continue
            for cv in np.unique(result_df_sub['cv']):
                cv_df = result_df_sub.loc[result_df_sub['cv'] == cv]
                plot_df = plot_df.append(
                    pd.DataFrame(
                        {'method_label': [method_map_ist[m]], 'ipw': [cv_df[ipw_name_key_ist[m]].values[0]],
                         'cv': [cv], 'method': [m]}))
        _, ax = plt.subplots(1, 1, figsize=(64, 32))
        sns.boxplot(data=plot_df, x='method_label', y='ipw', ax=ax, orient='v', palette=palette_mapped_ist,
                    showmeans=True, meanprops={"marker": "D", "markersize": 64, "markerfacecolor": 'black'})
        plt.xlabel('Method', fontsize=128)
        plt.ylabel('IPW Estimate', fontsize=128)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=128)

        plt.title('IST Data', fontdict={'fontsize': 128, 'fontweight': 'bold'})
        if not os.path.exists("./plots"):
            os.mkdir("./plots")
        if not os.path.exists(os.path.join("./plots/", data_name)):
            os.mkdir(os.path.join("./plots/", data_name))
        plt.savefig(os.path.join('./plots', data_name, 'IPW_Evaluation.pdf'),
                    dpi=300,
                    bbox_inches='tight', bbox_extra_artists=[])
    else:
        method_map = method_map_synth
        plot_df = pd.DataFrame(columns=['method', 'ipw', 'cv'])
        # print('result_df',result_df['th'])
        result_df_sub = result_df.loc[result_df['th'] == 0.1]
        # print('result_df_sub', result_df_sub)
        # exit(1)
        result_df_sub = result_df_sub[['mean_y_model_obs_expc1_expc2_ipw',
                                       'mean_y_model_obs_expc1_ipw',
                                       'mean_y_model_obs_expc2_ipw',
                                       'mean_y_model_obs_ipw',
                                       'mean_y_behavior_ipw',
                                       'mean_y_random_ipw',
                                       'mean_y_oracle_ipw',
                                       'cv']]
        for m in method_map_synth.keys():
            if m == 'oracle' or m == 'random':
                continue
            for cv in np.unique(result_df_sub['cv']):
                cv_df = result_df_sub.loc[result_df_sub['cv'] == cv]
                plot_df = plot_df.append(pd.DataFrame(
                    {'method_label': [method_map_synth[m]], 'ipw': [cv_df[ipw_name_key_synth[m]].values[0]],
                     'cv': [cv], 'method': [m]}))
        _, ax = plt.subplots(1, 1, figsize=(64, 32))
        print('plot diff', plot_df.head(20))
        sns.boxplot(data=plot_df, x='method_label', y='ipw', ax=ax, orient='v', palette=palette_mapped_synth,
                    width=0.5, showmeans=True, meanprops={"marker": "D", "markersize": 64, "markerfacecolor": 'black'})
        plt.xlabel('Method', fontsize=128)
        plt.ylabel('IPW Estimate', fontsize=128)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=128)

        plt.title('Synthetic Data', fontdict={'fontsize': 128, 'fontweight': 'bold'})
        if not os.path.exists("./plots"):
            os.mkdir("./plots")
        if not os.path.exists(os.path.join("./plots/", data_name)):
            os.mkdir(os.path.join("./plots/", data_name))
        plt.savefig(os.path.join('./plots', data_name, 'IPW_Evaluation.pdf'),
                    dpi=300,
                    bbox_inches='tight', bbox_extra_artists=[])
