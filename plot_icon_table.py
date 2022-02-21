from Tests.plot_templates import mean_std_template
import numpy as np

from dnl.Utils import read_file

BL = 0
DNL = 0
SPO = 1
QPTL = 2
SPOTREE = 3
INTOPT = 4

file_folder_predict = 'Tests/icon/Easy/load12/spartan/greedy'
file_folder_spo = 'Tests/icon/Easy/load12/spo_long'
number_of_rows_qptl = 72
n_iter = 2


def plot_minibatch_icon_mean_kfold(file_prefix, file_suffix, kfolds, frameworks, dest_file_name,
                                   file_folder_predict='Tests\experimental', file_folder_spo='Tests\experimental',
                                   file_folder_qptl='Tests\experimental', file_folder_spotree='Tests/spotree/', file_folder_intopt="Tests/intopt/",
                                   title='experimental ICON mean', is_show=True, is_save=False, is_run_time=False,
                                   is_normalize=True):
    if frameworks[DNL]:
        file_names_dnl = [str(file_prefix[DNL] + str(kfold) + file_suffix[DNL]) for kfold in kfolds]
    else:
        file_names_dnl = []
    if frameworks[SPO]:
        file_names_spo = [str(file_prefix[SPO] + str(kfold) + file_suffix[SPO]) for kfold in kfolds]
    else:
        file_names_spo = []
    if frameworks[QPTL]:
        file_names_qptl = [str(file_prefix[QPTL] + str(kfold) + file_suffix[QPTL]) for kfold in kfolds]
    else:
        file_names_qptl = []
    if frameworks[SPOTREE]:
        file_names_spotree = [str(file_prefix[SPOTREE] + str(kfold) + file_suffix[SPOTREE]) for kfold in kfolds]
    else:
        file_names_spotree = []

    if frameworks[INTOPT]:
        file_names_intopt = [str(file_prefix[INTOPT] + str(kfold) + file_suffix[INTOPT]) for kfold in kfolds]
    else:
        file_names_intopt = []

    regrets_allfolds = [[] for i, kfold in enumerate(kfolds)]
    for i, regret_onefold in enumerate(regrets_allfolds):
        regrets_allfolds[i] = mean_std_template(file_folder_predict=file_folder_predict,
                                                file_folder_spo=file_folder_spo,
                                                file_folder_qptl=file_folder_qptl,
                                                files_predict=[file_names_dnl[i]], files_spo=[file_names_spo[i]],
                                                files_qptl=[file_names_qptl[i]],
                                                title=title,
                                                dest_file_name=dest_file_name,
                                                predict_model_name='us', is_show=is_show, is_save=is_save,
                                                iter_no_pdo=n_iter, number_of_rows_qptl=number_of_rows_qptl,
                                                is_run_time=is_run_time)
        # Normalize
        if frameworks[SPOTREE]:
            regret_allfolds_spotree = get_regrets_spotree(file_folder=file_folder_spotree,
                                                          files_spotree=file_names_spotree[i], is_run_time=is_run_time)
        else:
            regret_allfolds_spotree = regrets_allfolds[i][BL]

        if frameworks[INTOPT]:
            regret_allfolds_intopt = get_regrets_intopt(file_folder=file_folder_intopt,
                                                          files_intopt=file_names_intopt[i], is_run_time=is_run_time)
        else:
            regret_allfolds_intopt = regrets_allfolds[i][BL]
        regrets_allfolds[i].append(regret_allfolds_spotree)
        regrets_allfolds[i].append(regret_allfolds_intopt)
        if is_normalize:
            regrets_allfolds[i] = regrets_allfolds[i] / regrets_allfolds[i][0]

    regrets_all = {'baseline': [regrets_allfolds[i][BL] for i, kfold in enumerate(kfolds)],
                   'dnl': [regrets_allfolds[i][DNL + 1] for i, kfold in enumerate(kfolds)],
                   'spo': [regrets_allfolds[i][SPO + 1] for i, kfold in enumerate(kfolds)],
                   'qptl': [regrets_allfolds[i][QPTL + 1] for i, kfold in enumerate(kfolds)],
                   'spotree': [regrets_allfolds[i][SPOTREE + 1] for i, kfold in enumerate(kfolds)],
                   'intopt': [regrets_allfolds[i][INTOPT + 1] for i, kfold in enumerate(kfolds)]}
    print(regrets_allfolds)
    return regrets_all

def get_regrets_intopt(file_folder, files_intopt, is_run_time=False):
    # if models[6]:
    #     for c in capacities:
    #         file_names = []
    #         for kfold in range(total_folds):
    #             if isUnit:
    #                 file_name = "Tests/spotree/spotree_uknap_c" + str(c) + 'k' + str(
    #                     kfold) + ".csv"
    #             else:
    #                 file_name = "Tests/spotree/spotree_knap_c" + str(c) + 'k' + str(
    #                     kfold) + ".csv"
    #             file_names.append(file_name)
    #         files_spotree_folders.append(file_names)
    regrets = np.zeros(len(files_intopt))

    fname = file_folder + files_intopt
    df_list = read_file(filename=fname, folder_path="", delimiter=',')

    regret, runtime = get_minval_regret_intopt(df_list)
    if is_run_time:
        return runtime
    else:
        return regret

def get_minval_regret_intopt(df_list):
    df_list = np.array(df_list)
    val_regret = df_list[1:,0].astype('float')
    regret = df_list[1:,2].astype('float')
    run_time = df_list[1:,7].astype('float')
    min_val_regret = regret[np.argmin(val_regret)]
    min_val_runtime = run_time[np.argmin(val_regret)]
    print('het')
    return min_val_regret, min_val_runtime
def get_regrets_spotree(file_folder, files_spotree, is_run_time=False):
    # if models[6]:
    #     for c in capacities:
    #         file_names = []
    #         for kfold in range(total_folds):
    #             if isUnit:
    #                 file_name = "Tests/spotree/spotree_uknap_c" + str(c) + 'k' + str(
    #                     kfold) + ".csv"
    #             else:
    #                 file_name = "Tests/spotree/spotree_knap_c" + str(c) + 'k' + str(
    #                     kfold) + ".csv"
    #             file_names.append(file_name)
    #         files_spotree_folders.append(file_names)
    regrets = np.zeros(len(files_spotree))

    fname = file_folder + files_spotree
    df_list = read_file(filename=fname, folder_path="", delimiter=' ')
    if is_run_time:
        regret = float(df_list[1][4])
    else:
        regret = float(df_list[1][5])
    return regret


def get_regret_load(loads, kfolds, frameworks, file_prefix, file_suffix, file_fold_suffix,
                    file_folder_predict='Tests/dnl', dest_file_name='experiment', is_run_time=False, is_normalize=True):
    file_prefix_loads = [
        [str(file_prefix[i] + str(load) + file_suffix[i]) for i, framework in enumerate(frameworks) if framework] for
        load in loads]
    regrets = [[] for load in loads]
    for i, load in enumerate(loads):
        regrets[i] = plot_minibatch_icon_mean_kfold(file_prefix_loads[i], file_fold_suffix, kfolds, frameworks,
                                                    dest_file_name=dest_file_name,
                                                    file_folder_predict=file_folder_predict,
                                                    file_folder_spo='Tests/spo',
                                                    file_folder_qptl='Tests/qptl', file_folder_spotree='Tests/spotree/',
                                                    title='experimental ICON mean', is_show=False, is_save=False,
                                                    is_run_time=is_run_time, is_normalize=is_normalize)

    return regrets


def generate_runtime_table():
    kfolds = [0, 1, 2, 3, 4]
    file_prefix = ['iconmax-l', 'Load', 'Load', 'spotree_icon_l', '0lintoptl']
    file_suffix = ['k', 'SPOmax_spartan_kfold', 'qptlmax_spartan_kfold', 'k', 'k']
    file_fold_suffix = ['-DIVIDE_AND_CONQUER_GREEDY-1-1.csv', '.csv', '.csv', '.csv', '.csv']

    loads = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57,
             400, 500, 501]
    is_run_time = True
    is_normalize = False
    regrets = get_regret_load(loads=loads, is_run_time=is_run_time, kfolds=kfolds, frameworks=[1, 1, 1, 1, 1],
                              file_folder_predict="Tests/dnl_large", file_prefix=file_prefix, file_suffix=file_suffix,
                              is_normalize=is_normalize, file_fold_suffix=file_fold_suffix)
    # file_fold_suffix = ['-DIVIDE_AND_CONQUER_GREEDY-0-1.csv', '.csv', '.csv', '.csv']
    # regrets2 = get_regret_load(loads=loads, kfolds=kfolds, frameworks=[1, 1, 1, 1], file_prefix=file_prefix,
    #                           file_suffix=file_suffix, file_fold_suffix=file_fold_suffix, file_folder_predict='Tests/dnl_large')
    print('Load, Baseline, Dnl, SPO, QPTL, SPOTree, IntOpt')
    for i, load in enumerate(loads):
        # print('LOAD {}'.format(load))
        baseline_mean = np.mean(regrets[i].get('baseline'))
        dnl_mean = np.mean(regrets[i].get('dnl'))
        spo_mean = np.mean(regrets[i].get('spo'))
        qptl_mean = np.mean(regrets[i].get('qptl'))
        spotree_mean = np.mean(regrets[i].get('spotree'))
        intopt_mean = np.mean(regrets[i].get('intopt'))


        baseline_std = np.std(regrets[i].get('baseline'))
        dnl_std = np.std(regrets[i].get('dnl'))
        spo_std = np.std(regrets[i].get('spo'))
        qptl_std = np.std(regrets[i].get('qptl'))
        spotree_std = np.std(regrets[i].get('spotree'))
        intopt_std = np.std(regrets[i].get('intopt'))
        file_fold_suffix = ['-DIVIDE_AND_CONQUER_GREEDY-1-1.csv', '.csv', '.csv', '.csv']

        dnl_sum = np.sum(regrets[i].get('dnl')[fold] <= regrets[i].get('baseline')[fold] for fold in kfolds)
        spo_sum = np.sum(regrets[i].get('spo')[fold] <= regrets[i].get('baseline')[fold] for fold in kfolds)
        qptl_sum = np.sum(regrets[i].get('qptl')[fold] <= regrets[i].get('baseline')[fold] for fold in kfolds)
        spotree_sum = np.sum(regrets[i].get('spotree')[fold] <= regrets[i].get('baseline')[fold] for fold in kfolds)
        intopt_sum = np.sum(regrets[i].get('intopt')[fold] <= regrets[i].get('baseline')[fold] for fold in kfolds)

        if not is_run_time:
            print('{}: {}+{}, {}+{} ({}), {}+{} ({}), {}+{} ({}), {}+{} ({}),{}+{} ({})'.format(load, round(baseline_mean, 2),
                                                                                     round(baseline_std, 2),
                                                                                     round(dnl_mean, 2),
                                                                                     round(dnl_std, 2), dnl_sum,
                                                                                     round(spo_mean, 2),
                                                                                     round(spo_std, 2), spo_sum,
                                                                                     round(qptl_mean, 2),
                                                                                     round(qptl_std, 2), qptl_sum,
                                                                                     round(spotree_mean, 2),
                                                                                     round(spotree_std, 2),
                                                                                     spotree_sum, round(intopt_mean, 2),
                                                                                     round(intopt_std, 2),
                                                                                     intopt_sum))
        else:
            print('{}: {}+{}, {}+{}, {}+{}, {}+{}, {}+{}, {}+{}'.format(load, round(baseline_mean, 2), round(baseline_std, 2),
                                                                 round(dnl_mean, 2), round(dnl_std, 2),
                                                                 round(spo_mean, 2), round(spo_std, 2),
                                                                 round(qptl_mean, 2), round(qptl_std, 2),
                                                                 round(spotree_mean, 2), round(spotree_std, 2),
                                                                 round(intopt_mean, 2), round(intopt_std, 2)
                                                                 ))


def generate_regret_table():
    kfolds = [0, 1, 2, 3, 4]
    file_prefix = ['iconmax-l', 'Load', 'Load', 'spotree_icon_l', '1lintoptl']
    file_suffix = ['k', 'SPOmax_spartan_kfold', 'qptlmax_spartan_kfold', 'k', 'k']
    file_fold_suffix = ['-DIVIDE_AND_CONQUER_GREEDY-1-1.csv', '.csv', '.csv', '.csv', '.csv']

    loads = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57,
             400, 500, 501]

    regrets = get_regret_load(loads=loads, kfolds=kfolds, frameworks=[1, 1, 1, 1,1],
                              file_folder_predict="Tests/dnl_large", file_prefix=file_prefix, file_suffix=file_suffix,
                              file_fold_suffix=file_fold_suffix)
    # file_fold_suffix = ['-DIVIDE_AND_CONQUER_GREEDY-0-1.csv', '.csv', '.csv', '.csv']
    # regrets2 = get_regret_load(loads=loads, kfolds=kfolds, frameworks=[1, 1, 1, 1], file_prefix=file_prefix,
    #                           file_suffix=file_suffix, file_fold_suffix=file_fold_suffix, file_folder_predict='Tests/dnl_large')
    print('Load, Baseline, Dnl, SPO, QPTL, SPOTree, IntOpt')
    for i, load in enumerate(loads):
        # print('LOAD {}'.format(load))
        baseline_mean = np.mean(regrets[i].get('baseline'))
        dnl_mean = np.mean(regrets[i].get('dnl'))
        spo_mean = np.mean(regrets[i].get('spo'))
        qptl_mean = np.mean(regrets[i].get('qptl'))
        spotree_mean = np.mean(regrets[i].get('spotree'))
        intopt_mean = np.mean(regrets[i].get('intopt'))



        baseline_std = np.std(regrets[i].get('baseline'))
        dnl_std = np.std(regrets[i].get('dnl'))
        spo_std = np.std(regrets[i].get('spo'))
        qptl_std = np.std(regrets[i].get('qptl'))
        spotree_std = np.std(regrets[i].get('spotree'))
        intopt_std = np.std(regrets[i].get('intopt'))
        file_fold_suffix = ['-DIVIDE_AND_CONQUER_GREEDY-1-1.csv', '.csv', '.csv', '.csv']

        dnl_sum = np.sum(regrets[i].get('dnl')[fold] <= regrets[i].get('baseline')[fold] for fold in kfolds)
        spo_sum = np.sum(regrets[i].get('spo')[fold] <= regrets[i].get('baseline')[fold] for fold in kfolds)
        qptl_sum = np.sum(regrets[i].get('qptl')[fold] <= regrets[i].get('baseline')[fold] for fold in kfolds)
        spotree_sum = np.sum(regrets[i].get('spotree')[fold] <= regrets[i].get('baseline')[fold] for fold in kfolds)
        intopt_sum = np.sum(regrets[i].get('intopt')[fold] <= regrets[i].get('baseline')[fold] for fold in kfolds)

        print('{}: {}+{}, {}+{} ({}), {}+{} ({}), {}+{} ({}), {}+{} ({}),{}+{} ({})'.format(load, round(baseline_mean, 2),
                                                                                 round(baseline_std, 2),
                                                                                 round(dnl_mean, 2), round(dnl_std, 2),
                                                                                 dnl_sum,
                                                                                 round(spo_mean, 2), round(spo_std, 2),
                                                                                 spo_sum,
                                                                                 round(qptl_mean, 2),
                                                                                 round(qptl_std, 2), qptl_sum,
                                                                                 round(spotree_mean, 2),
                                                                                 round(spotree_std, 2), spotree_sum,
                                                                                  round(intopt_mean, 2),
                                                                                  round(intopt_std, 2), intopt_sum))

if __name__ == '__main__':
    generate_regret_table()

    generate_runtime_table()
