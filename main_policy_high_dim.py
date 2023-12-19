import argparse
from policy_learning_utils import *
from test_synthetic import *
from plotting_utils import *
from sklearn.model_selection import ShuffleSplit

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", default=False)
    parser.add_argument("--seed", default=0)
    parser.add_argument("--data", default='synthetic')
    parser.add_argument("--data_file", default='./data/simulated_data_cont_raw_n_%d_d_%d.pkl')
    parser.add_argument("--generate_data", default=False)
    parser.add_argument("--n", default=10000, type=int)
    parser.add_argument("--d", default=2, type=int)
    parser.add_argument("--nfull", default=10000, type=int)
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--method", default="variational_policy_learn")
    parser.add_argument("--maxcv", default=5, type=int)
    parser.add_argument("--loss_type", default="weighted_cross_entropy")

    args = parser.parse_args()
    plot = args.plot
    seed = args.seed
    data = args.data
    method = args.method
    eval_mode = args.eval
    maxcv = args.maxcv
    loss_type = args.loss_type  # other option is smooth_max

    nn = args.n
    dim = args.d
    n_full = args.nfull
    data_file_name = args.data_file

    np.random.seed(seed)
    x_dim = 1
    y_dim = 1
    X_card = np.asarray([0, 1], dtype=np.float32)
    Y_card = np.asarray([0, 1], dtype=np.float32)
    c_dim = dim
    u_dim = 1

    if data == "synthetic_continuous":
        data_gen_method = 'integrate'
        if args.generate_data:
            data_gen_obj = SyntheticDataContinuousInt(c_dim, u_dim, x_dim, y_dim, nn)
            data_dict, dml_data_obj, df_full_obs_int = data_gen_obj.load_simulated_toy_highdim_cont()
            with open('./data/synthetic_fasrc/synthetic_continuous_data_obj.pkl', 'wb') as f:
                pkl.dump(data_gen_obj, f, pkl.HIGHEST_PROTOCOL)
            exit(0)
        else:
            if not eval_mode:
                with open(data_file_name % (nn, dim), 'rb') as f:
                    data_all = pkl.load(f)
                data_dict = data_all['prob_data']
                data_mat_df = data_all['data_frame']

                target_df = data_dict['d_mat']

                weight_matrix = np.zeros((data_mat_df.shape[0], 4, len(X_card)))
                weight_matrix[:, 0, 0] = data_dict['manski_bounds_x0'][:, 0]
                weight_matrix[:, 0, 1] = data_dict['manski_bounds_x1'][:, 0]
                weight_matrix[:, 1, 0] = data_dict['int_bounds_x0_expc1'][:, 0]
                weight_matrix[:, 1, 1] = data_dict['int_bounds_x1_expc1'][:, 0]
                weight_matrix[:, 2, 0] = data_dict['int_bounds_x0_expc2'][:, 0]
                weight_matrix[:, 2, 1] = data_dict['int_bounds_x1_expc2'][:, 0]
                weight_matrix[:, 3, 0] = data_dict['int_bounds_x0_expc1_expc2'][:, 0]
                weight_matrix[:, 3, 1] = data_dict['int_bounds_x1_expc1_expc2'][:, 0]

                weight_matrix = np.clip(weight_matrix, 1e-6, 1)

                weight_matrix_max = np.max(weight_matrix, axis=1).reshape(
                    (weight_matrix.shape[0], weight_matrix.shape[2]))

                target_vec = np.argmax(weight_matrix_max, axis=1).reshape((-1, 1))

                c_cols = [f'C{kk + 1}' for kk in np.arange(c_dim)]

                sk5fold = ShuffleSplit(n_splits=5, test_size=0.1, random_state=42)
                for cv in range(0, maxcv):
                    train_idx, valid_idx = \
                        list(sk5fold.split(data_mat_df[c_cols], target_vec))[cv]

                    for bt in range(4):
                        if bt < 2:
                            weight_matrix_bt = np.max(weight_matrix[:, :bt + 1, :], axis=1).reshape(
                                (weight_matrix.shape[0], weight_matrix.shape[2]))
                            if bt == 0:
                                model_name = 'obs'
                            else:
                                model_name = 'obs_expc1'
                        elif 2 <= bt < 3:
                            weight_matrix_bt = np.max(weight_matrix[:, [0, 2], :], axis=1).reshape(
                                (weight_matrix.shape[0], weight_matrix.shape[2]))
                            model_name = 'obs_expc2'
                        else:
                            weight_matrix_bt = weight_matrix_max
                            model_name = 'obs_expc1_expc2'

                        train_wrapper(X=np.asarray(data_mat_df[c_cols]), targets=target_vec,
                                      lower_bounds=weight_matrix_bt,
                                      data_name=data, n_targets=len(X_card), loss_type=loss_type, cv=cv,
                                      train_idx=train_idx, n_epochs=15, model_name=model_name)

            # evaluate via rollouts
            with open('./data/synthetic_fasrc/synthetic_continuous_data_obj.pkl', 'rb') as f:
                data_gen_obj = pkl.load(f)
            with open(data_file_name % (nn, dim), 'rb') as f:
                data_all = pkl.load(f)
            data_mat_df = data_all['data_frame']
            data_dict = data_all['prob_data']
            c_cols = [f'C{kk + 1}' for kk in np.arange(c_dim)]
            weight_matrix = np.zeros((data_mat_df.shape[0], 4, len(X_card)))
            weight_matrix[:, 0, 0] = data_dict['manski_bounds_x0'][:, 0]
            weight_matrix[:, 0, 1] = data_dict['manski_bounds_x1'][:, 0]

            weight_matrix[:, 1, 0] = data_dict['int_bounds_x0_expc1'][:, 0]
            weight_matrix[:, 1, 1] = data_dict['int_bounds_x1_expc1'][:, 0]

            weight_matrix[:, 2, 0] = data_dict['int_bounds_x0_expc2'][:, 0]
            weight_matrix[:, 2, 1] = data_dict['int_bounds_x1_expc2'][:, 0]

            weight_matrix[:, 3, 0] = data_dict['int_bounds_x0_expc1_expc2'][:, 0]
            weight_matrix[:, 3, 1] = data_dict['int_bounds_x1_expc1_expc2'][:, 0]

            weight_matrix = np.clip(weight_matrix, 1e-6, 1)

            plot_weights(weight_mat=weight_matrix[:, :, :, None], c_mat=np.asarray(data_mat_df[c_cols]), data_name=data)
            eval_via_rollout(data_obj=data_gen_obj, n=nn, d=c_dim, maxcv=maxcv, data_name=data, loss_type=loss_type,
                             train_policies=False)
    elif data == 'synthetic_no_bounds':
        if args.generate_data:
            data_gen_obj = SyntheticDataNoBounds(c_dim, u_dim, x_dim, y_dim, nn)
            data_gen_obj.load_simulated_toy_highdim_cont()
            with open('./data/synthetic_fasrc/synthetic_continuous_no_bounds_data_obj.pkl', 'wb') as f:
                pkl.dump(data_gen_obj, f, pkl.HIGHEST_PROTOCOL)
            exit(0)
        else:
            data_dict = None
            data_gen_obj = None
            if not eval_mode:
                with open('./data/synthetic_data_dict_no_bounds_n_%d_d_%d.pkl' % (nn, c_dim), 'rb') as f:
                    data_dict = pkl.load(f)

                sk5fold = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=42)
                for cv in range(0, maxcv):
                    train_idx, valid_idx = \
                        list(sk5fold.split(data_dict['train_df_obs'], data_dict['train_df_obs']['X']))[cv]

                    weight_matrix_train, weight_matrix_test = learn_data_bounds(data_dict=data_dict, train_all=True,
                                                                                cv=cv, train_idx=train_idx,
                                                                                data_name=data, n_total=nn)

                    weight_matrix_train_max = np.max(weight_matrix_train, axis=1).reshape(
                        (weight_matrix_train.shape[0], weight_matrix_train.shape[2]))
                    weight_matrix_test_max = np.max(weight_matrix_test, axis=1).reshape(
                        (weight_matrix_test.shape[0], weight_matrix_test.shape[2]))

                    train_df_obs = data_dict['train_df_obs'].iloc[train_idx]
                    test_df_obs = data_dict['test_df_obs']

                    target_vec_train = np.argmax(weight_matrix_train_max, axis=1).reshape((-1, 1))
                    target_vec_test = np.argmax(weight_matrix_test_max, axis=1).reshape((-1, 1))

                    c1_dims = data_dict['c1_dims']
                    c2_dims = data_dict['c2_dims']

                    X_train = np.asarray(train_df_obs[c1_dims + c2_dims])
                    X_test = np.asarray(test_df_obs[c1_dims + c2_dims])
                    X = np.concatenate((X_train, X_test), axis=0)

                    for bt in range(4):
                        if bt < 2:
                            weight_matrix_bt_train = np.max(weight_matrix_train[:, :bt + 1, :], axis=1).reshape(
                                (weight_matrix_train.shape[0], weight_matrix_train.shape[2]))

                            weight_matrix_bt_test = np.max(weight_matrix_test[:, :bt + 1, :], axis=1).reshape(
                                (weight_matrix_test.shape[0], weight_matrix_test.shape[2]))
                            if bt == 0:
                                model_name = 'obs'
                            else:
                                model_name = 'obs_expc1'
                        elif 2 <= bt < 3:
                            weight_matrix_bt_train = np.max(weight_matrix_train[:, [0, 2], :], axis=1).reshape(
                                (weight_matrix_train.shape[0], weight_matrix_train.shape[2]))

                            weight_matrix_bt_test = np.max(weight_matrix_test[:, [0, 2], :], axis=1).reshape(
                                (weight_matrix_test.shape[0], weight_matrix_test.shape[2]))
                            model_name = 'obs_expc2'
                        else:
                            weight_matrix_bt_train = weight_matrix_train_max
                            weight_matrix_bt_test = weight_matrix_test_max
                            model_name = 'obs_expc1_expc2'

                        target_vec = np.concatenate((target_vec_train, target_vec_test), axis=0)
                        weight_matrix_max = np.concatenate((weight_matrix_train_max, weight_matrix_test_max), axis=0)
                        weight_matrix_bt = np.concatenate((weight_matrix_bt_train, weight_matrix_bt_test), axis=0)
                        n_epochs = 15

                        train_wrapper(X=X, targets=target_vec, lower_bounds=weight_matrix_bt,
                                      data_name=data, loss_type=loss_type,
                                      train_idx=np.arange(weight_matrix_train.shape[0]),
                                      valid_idx=np.arange(weight_matrix_train.shape[0],
                                                          weight_matrix_train.shape[0] + weight_matrix_test.shape[0]),
                                      learning_rate=0.001, n_targets=len(np.unique(target_vec)), cv=cv,
                                      model_name=model_name, n_epochs=n_epochs)

            if data_gen_obj is None:
                with open('./data/synthetic_fasrc/synthetic_continuous_no_bounds_data_obj.pkl', 'rb') as f:
                    data_gen_obj = pkl.load(f)

            if data_dict is None:
                with open('./data/synthetic_no_bounds_data_dict_n_%d_d_%d.pkl' % (nn, c_dim), 'rb') as f:
                    data_dict = pkl.load(f)

            target_vec_train = np.asarray(data_dict['train_df_obs']['X'])
            target_vec_test = np.asarray(data_dict['test_df_obs']['X'])
            test_df_obs = data_dict['test_df_obs']
            c2_dims = data_dict['c2_dims']
            n_targets = len(np.unique(target_vec_train))
            weight_matrix_test_all = np.zeros((target_vec_test.shape[0], 4, n_targets, maxcv))
            for cv in range(maxcv):
                weight_matrix_test_all[:, :, :, cv] = data_dict['weight_matrix_test_%d' % cv]
            plot_weights(weight_mat=weight_matrix_test_all, c_mat=np.asarray(test_df_obs[c2_dims]).reshape(-1, 1),
                         data_name=data)
            eval_via_rollout(data_obj=data_gen_obj, n=465, d=c_dim, maxcv=maxcv, data_name=data, loss_type=loss_type,
                             train_policies=False)

    elif data == 'IST':
        if args.generate_data:
            generate_stroke_trial_data()
            exit(0)
        else:
            data_dict = None
            if not eval_mode:
                with open('./data/IST_data_dict.pkl', 'rb') as f:
                    data_dict = pkl.load(f)

                sk5fold = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=42)
                for cv in range(0, maxcv):
                    train_idx, valid_idx = \
                        list(sk5fold.split(data_dict['train_df_obs'], data_dict['train_df_obs']['X']))[cv]

                    weight_matrix_train, weight_matrix_test = learn_data_bounds(data_dict=data_dict, train_all=True,
                                                                                cv=cv, train_idx=train_idx, n_total=data_dict['train_df_obs'].shape[0])

                    weight_matrix_train_max = np.max(weight_matrix_train, axis=1).reshape(
                        (weight_matrix_train.shape[0], weight_matrix_train.shape[2]))
                    weight_matrix_test_max = np.max(weight_matrix_test, axis=1).reshape(
                        (weight_matrix_test.shape[0], weight_matrix_test.shape[2]))

                    train_df_obs = data_dict['train_df_obs'].iloc[train_idx]
                    test_df_obs = data_dict['test_df_obs']

                    target_vec_train = np.argmax(weight_matrix_train_max, axis=1).reshape((-1, 1))
                    target_vec_test = np.argmax(weight_matrix_test_max, axis=1).reshape((-1, 1))

                    c1_dims = data_dict['c1_dims']
                    c2_dims = data_dict['c2_dims']

                    X_train = np.asarray(train_df_obs[c1_dims + c2_dims])
                    X_test = np.asarray(test_df_obs[c1_dims + c2_dims])
                    X = np.concatenate((X_train, X_test), axis=0)

                    for bt in range(4):
                        if bt < 2:
                            weight_matrix_bt_train = np.max(weight_matrix_train[:, :bt + 1, :], axis=1).reshape(
                                (weight_matrix_train.shape[0], weight_matrix_train.shape[2]))

                            weight_matrix_bt_test = np.max(weight_matrix_test[:, :bt + 1, :], axis=1).reshape(
                                (weight_matrix_test.shape[0], weight_matrix_test.shape[2]))
                            if bt == 0:
                                model_name = 'obs'
                            else:
                                model_name = 'obs_expc1'
                        elif 2 <= bt < 3:
                            weight_matrix_bt_train = np.max(weight_matrix_train[:, [0, 2], :], axis=1).reshape(
                                (weight_matrix_train.shape[0], weight_matrix_train.shape[2]))

                            weight_matrix_bt_test = np.max(weight_matrix_test[:, [0, 2], :], axis=1).reshape(
                                (weight_matrix_test.shape[0], weight_matrix_test.shape[2]))
                            model_name = 'obs_expc2'
                        else:
                            weight_matrix_bt_train = weight_matrix_train_max
                            weight_matrix_bt_test = weight_matrix_test_max
                            model_name = 'obs_expc1_expc2'

                        target_vec = np.concatenate((target_vec_train, target_vec_test), axis=0)
                        weight_matrix_max = np.concatenate((weight_matrix_train_max, weight_matrix_test_max), axis=0)
                        weight_matrix_bt = np.concatenate((weight_matrix_bt_train, weight_matrix_bt_test), axis=0)
                        n_epochs = 15

                        train_wrapper(X=X, targets=target_vec, lower_bounds=weight_matrix_bt,
                                      data_name=data, loss_type=loss_type,
                                      train_idx=np.arange(weight_matrix_train.shape[0]),
                                      valid_idx=np.arange(weight_matrix_train.shape[0],
                                                          weight_matrix_train.shape[0] + weight_matrix_test.shape[0]),
                                      learning_rate=0.001, n_targets=len(np.unique(target_vec)), cv=cv,
                                      model_name=model_name, n_epochs=n_epochs)

            if data_dict is None:
                with open('./data/IST_data_dict_n_%d_d_%d.pkl' %(weight_matrix_train.shape[0]+weight_matrix_test.shape[0], len(c1_dims)+len(c2_dims)), 'rb') as f:
                    data_dict = pkl.load(f)

            target_vec_train = np.asarray(data_dict['train_df_obs']['X'])
            target_vec_test = np.asarray(data_dict['test_df_obs']['X'])
            test_df_obs = data_dict['test_df_obs']
            c1_dims = data_dict['c1_dims']
            c2_dims = data_dict['c2_dims']
            n_targets = len(np.unique(target_vec_train))
            weight_matrix_test_all = np.zeros((target_vec_test.shape[0], 4, n_targets, maxcv))
            for cv in range(maxcv):
                weight_matrix_test_all[:, :, :, cv] = data_dict['weight_matrix_test_%d' % cv]
            plot_weights(weight_mat=weight_matrix_test_all, c_mat=np.asarray(test_df_obs[c2_dims]).reshape(-1, 1),
                         data_name=data, scaler=data_dict['sysbp_scaler'])
            eval_via_pseudo_rollout(data_dict=data_dict, data_name=data, n_targets=n_targets, maxcv=maxcv)
