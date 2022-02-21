import numpy as np

from dnl.Utils import get_file_path, read_file
import matplotlib.pyplot as plt

"""
    model_method_names = ['Exhaustive',
                          'Exhaustive Select Max',
                          'Divide and Conquer',
                          'Divide and Conquer Select Max',
                          'Divide and Conquer Select Greedy'
                          ]
                          
     results = {'regrets' : np.mean(regrets, axis=1),
    'test_MSEs' : np.mean(test_MSES,axis=1),
    'run_times' : np.mean(run_times,axis=1),
    'training_obj_values' : np.mean(training_obj_values,axis=1),
    'epochs' : np.mean(epochs,axis=1)}
"""


def plot_run_time(file_folder='Tests/Knapsack/weighted'):
    folder_path = get_file_path(filename="", folder_path=file_folder)
    files = ['knapsack-c12-0105.csv', 'knapsack-c12-0205.csv', 'knapsack-c12-1205.csv']
    models = ['Exhaustive', 'Exhaustive Max Selection', 'Divide and Conquer', 'Divide and Conquer Max Selection',
              'Greedy', 'Linear Regression']
    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00"]

    sample_points = [30, 300, 3000]
    run_times = np.zeros((5, 3))
    regrets = np.zeros((6, 3))
    dfs = []
    for index, file in enumerate(files):
        df = read_file(filename=file, folder_path=file_folder)
        run_times[:, index] = df[2][:]
        regrets[:, index] = df[0][:]
        dfs.append(df)
    # print(regrets)

    #
    # plt.subplot(2, 1, 2)
    # for index, file in enumerate(models):
    #     if index<5:
    #       plt.plot(sample_points,run_times[index,:],colors[index])
    #
    # plt.subplot(2, 1, 2)
    # for index, file in enumerate(models):
    #
    #     plt.plot(sample_points, regrets[index, :],colors[index])
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ind = np.arange(0, 3)
    ax1.set_xticks(ind)
    ax1.set_xticklabels((sample_points))
    ax1.set_title('Run Time vs Sample Points')
    ax1.set_xlabel('Sample Points')
    ax1.set_ylabel('Run Time(s)')
    for index, file in enumerate(models):
        if index < 5:
            ax1.bar(ind + (0.15 * index - 37.5), run_times[index, :], color=colors[index], width=0.15)

    ax2 = fig.add_subplot(1, 2, 2)
    ind = np.arange(0, 3)
    ax2.set_xticks(ind)
    ax2.set_xticklabels((sample_points))
    ax2.set_title('Regret vs Sample Points')
    ax2.set_xlabel('Sample Points')
    ax2.set_ylabel('Regret')

    for index, file in enumerate(models):
        ax2.bar(ind + (0.15 * index - 37.5), regrets[index, :], color=colors[index], width=0.15)
    ax2.legend(models)
    plt.show()


def plot_knapsack_weighted():
    file_folder = 'Knapsack/weighted'
    folder_path = get_file_path(filename="", folder_path=file_folder)
    files = ['knapsack-c12-0105.csv', 'knapsack-c24-0105.csv', 'knapsack-c48-0105.csv', 'knapsack-c72-0105.csv',
             'knapsack-c96-0105.csv', 'knapsack-c120-0105.csv']
    colors = ['y', 'g', 'c', 'm']
    capacities = ['%0', '%5', '%10', '%20', '%30', '%40', '%50']
    capacities2 = [5, 10, 20, 30, 40, 50]
    models = ['Divide and Conquer', 'Divide and Conquer Max Selection',
              'Greedy', 'Linear Regression']

    regrets = np.zeros((4, 6))
    dfs = []
    for index, file in enumerate(files):
        df = read_file(filename=file, folder_path=file_folder)
        print(df[0])
        print(index)
        regrets[:, index] = df[0][2:]
        dfs.append(df)

    # files = ['knapsack-c12-0105.csv', 'knapsack-c12-0205.csv', 'knapsack-c12-1205.csv']
    # models = ['Exhaustive', 'Exhaustive Max Selection', 'Divide and Conquer', 'Divide and Conquer Max Selection',
    #           'Greedy', 'Linear Regression']
    # colors = ['r', 'b', 'y', 'g', 'c', 'm']
    # sample_points = [30, 300, 3000]
    # run_times = np.zeros((5, 3))
    # regrets = np.zeros((6, 3))
    # dfs = []
    # for index, file in enumerate(files):
    #     df = read_file(filename=file, folder_path=file_folder)
    #     run_times[:, index] = df[2][:]
    #     regrets[:, index] = df[0][:]
    #     dfs.append(df)

    fig = plt.figure()
    # ax1 = fig.add_subplot(1, 2, 1)
    # ind = np.arange(0, 6)
    # ax1.set_xticks(ind)
    # ax1.set_xticklabels(capacities)
    # ax1.set_title('Run Time vs Sample Points')
    # ax1.set_xlabel('Sample Points')
    # ax1.set_ylabel('Run Time(s)')
    # for index, file in enumerate(models):
    #     if index < 5:
    #         ax1.bar(ind + (0.15 * index - 37.5), run_times[index, :], color=colors[index], width=0.15)

    ax2 = fig.add_subplot(1, 1, 1)
    ind = np.arange(0, 6) + 1
    xtickslocs = ind
    ax2.set_xticks(ind + 1, capacities)
    ax2.set_xticklabels(capacities)
    ax2.set_title('Regret vs Capacities')
    # ax2.set_xlabel('Capacities')
    ax2.set_ylabel('Regret')

    for index, model in enumerate(models):
        ax2.bar(ind + ((+0.15 * index) - 0.22), regrets[index, :], color=colors[index], width=0.15)
    # ax2.xaxis.set_major_locator(mticker.FixedLocator(xtickslocs))
    ax2.legend(models)
    plt.show()


def correlated_knapsack_template(dest_file_name='',
                                 is_show=True, is_plot=False, is_save=True,
                                 xtick_labels=None,
                                 file_folder='Tests/Knapsack/weighted/',
                                 c=12, models=None,
                                 w_tag_str='w', plot_title='regret vs capacities', correlations=None,
                                 number_of_rows_spo=85):
    if models is None:
        models = [1, 0, 0, 1]
    if correlations is None:
        correlations = [25, 50, 75]
    total_folds = 5
    files_predict_dnc_folders = []
    files_spo_folders = []
    facecolors = ["#009E73", "#56B4E9", "#000000", "#CC79A7", "#0072B2"]
    patterns = ('-', '+', 'x', '\\', '*', 'o')
    edgecolors = ['#089FFF', '#2ba14d', '#eb6c65', '#bd78de']

    for corr in correlations:
        file_names = []
        for kfold in range(total_folds):
            file_name = file_folder + "corr" + str(corr) + "_weighted" + "/laptop/knapsack-" + w_tag_str + "c" + str(
                c) + '-k' + str(
                kfold) + "-DIVIDE_AND_CONQUER_GREEDY-0-1.csv"
            file_names.append(file_name)
        files_predict_dnc_folders.append(file_names)

    for corr in correlations:
        file_names = []
        for kfold in range(total_folds):
            file_name = file_folder + "corr" + str(corr) + "_weighted" + "/spo/" + "knapsack_SPOk" + str(
                kfold) + "_c" + str(c) + ".csv"
            file_names.append(file_name)
        files_spo_folders.append(file_names)

    dnc_regrets = np.zeros((len(correlations), total_folds))
    spo_regrets = np.zeros((len(correlations), total_folds))
    baseline_regrets = np.zeros((len(correlations), total_folds))

    for index, corr in enumerate(correlations):
        for kfold in range(total_folds):
            file_dnc = files_predict_dnc_folders[index]
            file_spo = files_spo_folders[index]

            dnc_regret, greedy_regret, max_regret, spo_regret, baseline_regret = find_regrets_knapsack(file_dnc[kfold],
                                                                                                       [],
                                                                                                       [],
                                                                                                       file_spo[kfold],
                                                                                                       facecolors=facecolors,
                                                                                                       edgecolors=edgecolors,
                                                                                                       models=models,
                                                                                                       is_plot=is_plot,
                                                                                                       number_of_rows_spo=number_of_rows_spo)
            dnc_regrets[index, kfold] = dnc_regret
            spo_regrets[index, kfold] = spo_regret
            baseline_regrets[index, kfold] = baseline_regret

    fig = plt.figure()
    # ax1 = fig.add_subplot(1, 2, 1)
    # ind = np.arange(0, 6)
    # ax1.set_xticks(ind)
    # ax1.set_xticklabels(capacities)
    # ax1.set_title('Run Time vs Sample Points')
    # ax1.set_xlabel('Sample Points')
    # ax1.set_ylabel('Run Time(s)')
    # for index, file in enumerate(models):
    #     if index < 5:
    #         ax1.bar(ind + (0.15 * index - 37.5), run_times[index, :], color=colors[index], width=0.15)

    ax2 = fig.add_subplot(1, 1, 1)
    if xtick_labels is None:
        xtick_labels = correlations
    ind = np.arange(0, len(xtick_labels)) + 1
    xtickslocs = ind
    ax2.set_xticks(ind)
    ax2.set_xticklabels(xtick_labels)
    ax2.set_title(plot_title)
    ax2.set_xlabel('Noise coefficient')
    ax2.set_ylabel('Test Regret')

    regrets = np.array([np.mean(dnc_regrets, axis=1), np.mean(dnc_regrets, axis=1), np.mean(dnc_regrets, axis=1),
                        np.mean(spo_regrets, axis=1), np.mean(baseline_regrets, axis=1)])
    errors = np.array([np.std(dnc_regrets, axis=1), np.std(dnc_regrets, axis=1), np.std(dnc_regrets, axis=1),
                       np.std(spo_regrets, axis=1), np.std(baseline_regrets, axis=1)])

    models.append([1])
    for index, model in enumerate(models):
        if model:
            if index == 0:
                ind_plot = 2
            else:
                ind_plot = index
            bar = ax2.bar(ind + ((+0.15 * ind_plot) - 0.22), regrets[index, :], yerr=errors[index, :],
                          color=facecolors[index], width=0.15, hatch=patterns[index])

    # ax2.xaxis.set_major_locator(mticker.FixedLocator(xtickslocs))
    labels = ['DnL-Greedy', 'SPO', "Ridge Regression"]
    ax2.legend(labels)
    if is_save:
        plt.savefig('figs/' + str(dest_file_name))
    if is_show:
        plt.show()


def constrained_ICON_template(dest_file_name='',
                              is_show=True, is_plot=False, is_save=True,
                              xtick_labels=None,
                              file_folder='Tests/Knapsack/weighted/',
                              loads=[32, 33], models=[1, 1],
                              w_tag_str='w', plot_title='regret vs capacities', number_of_rows_spo=65):
    total_folds = 5

    files_predict_dnc_g_folders = []
    files_spo_folders = []
    facecolors = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2"]
    edgecolors = ['#089FFF', '#2ba14d', '#eb6c65', '#bd78de']
    if models[0]:
        for l in loads:
            file_names = []
            for kfold in range(total_folds):
                file_name = file_folder + 'load' + str(l) + "/spartan/icon-l" + str(l) + 'k' + str(
                    kfold) + "-DIVIDE_AND_CONQUER_GREEDY-0-1.csv"
                file_names.append(file_name)
            files_predict_dnc_g_folders.append(file_names)

    if models[1]:
        for l in loads:
            file_names = []
            for kfold in range(total_folds):
                file_name = file_folder + 'load' + str(l) + "/spo/" + str(l) + "Load" + str(
                    l) + "SPO_warmstart_corrected_kfold" + str(kfold) + ".csv"
                file_names.append(file_name)
            files_spo_folders.append(file_names)

    dnc_regrets = np.zeros((len(loads), total_folds))
    greedy_regrets = np.zeros((len(loads), total_folds))
    spo_regrets = np.zeros((len(loads), total_folds))
    baseline_regrets = np.zeros((len(loads), total_folds))

    for index, l in enumerate(loads):
        for kfold in range(total_folds):
            file_greedy = files_predict_dnc_g_folders[index]
            file_spo = files_spo_folders[index]

            greedy_regret, spo_regret, baseline_regret = find_regrets_icon_constraiend(file_greedy[
                                                                                           kfold],
                                                                                       file_spo[kfold],
                                                                                       facecolors=facecolors,
                                                                                       edgecolors=edgecolors,
                                                                                       models=models,
                                                                                       is_plot=is_plot,
                                                                                       number_of_rows_spo=number_of_rows_spo)
            greedy_regrets[index, kfold] = greedy_regret
            spo_regrets[index, kfold] = spo_regret
            baseline_regrets[index, kfold] = baseline_regret

    fig = plt.figure()
    # ax1 = fig.add_subplot(1, 2, 1)
    # ind = np.arange(0, 6)
    # ax1.set_xticks(ind)
    # ax1.set_xticklabels(capacities)
    # ax1.set_title('Run Time vs Sample Points')
    # ax1.set_xlabel('Sample Points')
    # ax1.set_ylabel('Run Time(s)')
    # for index, file in enumerate(models):
    #     if index < 5:
    #         ax1.bar(ind + (0.15 * index - 37.5), run_times[index, :], color=colors[index], width=0.15)

    ax2 = fig.add_subplot(1, 1, 1)
    if xtick_labels is None:
        xtick_labels = ['%0', '%50', '%75']
    ind = np.arange(0, len(xtick_labels)) + 1
    xtickslocs = ind
    ax2.set_xticks(ind)
    ax2.set_xticklabels(xtick_labels)
    ax2.set_title(plot_title)
    ax2.set_xlabel('Compression Percentage')
    ax2.set_ylabel('Test Regret')

    regrets = np.array([np.mean(greedy_regrets, axis=1),
                        np.mean(spo_regrets, axis=1), np.mean(baseline_regrets, axis=1)])
    errors = np.array([np.std(greedy_regrets, axis=1),
                       np.std(spo_regrets, axis=1), np.std(baseline_regrets, axis=1)])

    models.extend([1])
    for index, model in enumerate(models):
        ind_plot = index
        ax2.bar(ind + ((+0.15 * ind_plot) - 0.22), regrets[index, :], yerr=errors[index, :], color=facecolors[index],
                width=0.15)

    # ax2.xaxis.set_major_locator(mticker.FixedLocator(xtickslocs))
    labels = ['GREEDY', 'SPO', 'regression']
    ax2.legend(labels)
    if is_save:
        plt.savefig('figs/' + str(dest_file_name))
    if is_show:
        plt.show()


def weighted_knapsack_template(dest_file_name='',
                               is_show=True, is_plot=False, is_save=True,
                               xtick_labels=None,
                               file_folder='Tests/Knapsack/weighted/',
                               capacities=None, models=None,
                               w_tag_str='w',  plot_title='regret vs capacities', number_of_rows_spo=168,
                               number_of_rows_qptl=174, ylim=None, isUnit=False, is_run_time = False, is_normalize = True, unit_tag_intopt = ""):
    if capacities == None:
        capacities = [12, 24, 48, 72, 96, 120, 144, 172, 196, 220]
    if models == None:
        models = [1, 1, 1, 1, 1, 1]
    total_folds = 5
    files_predict_dnc_folders = []
    files_predict_dnc_m_folders = []
    files_predict_dnc_g_folders = []
    files_spo_folders = []
    files_dp_folders = []
    files_qptl_folders = []
    files_spotree_folders = []
    files_intopt_folders = []

    # facecolors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
    facecolors = ['#a6cee3',
                  '#1f78b4',
                  '#b2df8a',
                  '#33a02c',
                  '#a6cee3',
                  '#1f78b4',
                  '#b2df8a',
                  '#33a02c',
                  '#a6cee3'
                  ]
    # facecolors = []
    # ecolors = ["#a87400", "#428ab3", "#007354", "#915777", "#696969", "#a87400", "#00456b", "#428ab3"]
    ecolors = ['#a6cee3',
               '#1f78b4',
               '#b2df8a',
               '#33a02c']
    patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '+')
    patterns = ('', '', '', '', '/', '--', '\\', 'o', '+')
    edgecolors = ['#089FFF', '#2ba14d', '#eb6c65', '#bd78de']
    if models[0]:
        for c in capacities:
            file_names = []
            for kfold in range(total_folds):
                if isUnit:
                    file_name = file_folder + "c" + str(c) + "/laptop/knapsack-c" + w_tag_str + str(
                        c) + '-k' + str(
                        kfold) + "-DIVIDE_AND_CONQUER-0-1.csv"
                else:
                    file_name = file_folder + "c" + str(c) + "/laptop/gurobi_knapsack-" + w_tag_str + "sparc" + str(
                        c) + '-k' + str(
                        kfold) + "-DIVIDE_AND_CONQUER-0-1.csv"
                file_names.append(file_name)
            files_predict_dnc_folders.append(file_names)
    if models[1]:
        for c in capacities:
            file_names = []
            for kfold in range(total_folds):
                if isUnit:
                    file_name = file_folder + "c" + str(c) + "/laptop/knapsack-c" + w_tag_str + str(
                        c) + '-k' + str(
                        kfold) + "-DIVIDE_AND_CONQUER_MAX-0-1.csv"
                else:
                    file_name = file_folder + "c" + str(c) + "/laptop/gurobi_knapsack-" + w_tag_str + "sparc" + str(
                        c) + '-k' + str(
                        kfold) + "-DIVIDE_AND_CONQUER_MAX-0-1.csv"
                file_names.append(file_name)
            files_predict_dnc_m_folders.append(file_names)
    if models[2]:
        for c in capacities:
            file_names = []
            for kfold in range(total_folds):
                if isUnit:
                    file_name = file_folder + "c" + str(c) + "/laptop/knapsack-c" + w_tag_str + str(
                        c) + '-k' + str(
                        kfold) + "-DIVIDE_AND_CONQUER_GREEDY-0-1.csv"
                else:
                    file_name = file_folder + "c" + str(c) + "/laptop/gurobi_knapsack-" + w_tag_str + "sparc" + str(
                        c) + '-k' + str(
                        kfold) + "-DIVIDE_AND_CONQUER_GREEDY-0-1.csv"
                file_names.append(file_name)
            files_predict_dnc_g_folders.append(file_names)
    if models[3]:
        for c in capacities:
            file_names = []
            for kfold in range(total_folds):
                if isUnit:
                    file_name = file_folder + "spo/" + "knapsack_SPOk" + str(kfold) + "_c" + str(c) + ".csv"
                else:
                    file_name = file_folder + "spo/" + "gurobi_knapsack_SPOk" + str(kfold) + "_c" + str(c) + ".csv"
                file_names.append(file_name)
            files_spo_folders.append(file_names)

    if models[4]:
        for c in capacities:
            file_names = []
            for kfold in range(total_folds):
                file_name = file_folder + "qptl/" + "knapsack_qptlk" + str(kfold) + "_c" + str(c) + ".csv"
                file_names.append(file_name)
            files_qptl_folders.append(file_names)
    if models[5]:
        for c in capacities:
            file_names = []
            for kfold in range(total_folds):
                file_name = file_folder + "c" + str(c) + "/dp/knapsack-" + w_tag_str + "c" + str(c) + '-k' + str(
                    kfold) + "-DP.csv"
                file_names.append(file_name)
            files_dp_folders.append(file_names)

    if models[6]:
        for c in capacities:
            file_names = []
            for kfold in range(total_folds):
                if isUnit:
                    file_name = "Tests/spotree/spotree_uknap_c" + str(c) + 'k' + str(
                        kfold) + ".csv"
                else:
                    file_name = "Tests/spotree/spotree_knap_c" + str(c) + 'k' + str(
                        kfold) + ".csv"
                file_names.append(file_name)
            files_spotree_folders.append(file_names)

    if models[7]:
        for c in capacities:
            file_names = []
            for kfold in range(total_folds):
                file_name = "Tests/intopt/0lintopt" + unit_tag_intopt + "c" + str(c) + 'k' + str(
                    kfold) + ".csv"
                file_names.append(file_name)
            files_intopt_folders.append(file_names)

    dnc_regrets = np.zeros((len(capacities), total_folds))
    greedy_regrets = np.zeros((len(capacities), total_folds))
    max_regrets = np.zeros((len(capacities), total_folds))
    spo_regrets = np.zeros((len(capacities), total_folds))
    baseline_regrets = np.zeros((len(capacities), total_folds))
    dp_regrets = np.zeros((len(capacities), total_folds))
    qptl_regrets = np.zeros((len(capacities), total_folds))
    spotree_regrets = np.zeros((len(capacities), total_folds))
    intopt_regrets = np.zeros((len(capacities), total_folds))

    for index, c in enumerate(capacities):
        for kfold in range(total_folds):
            file_dnc = files_predict_dnc_folders[index]
            file_m = files_predict_dnc_folders[index]
            file_greedy = files_predict_dnc_g_folders[index]
            file_spo = files_spo_folders[index]
            file_dp = files_dp_folders[index]
            file_qptl = files_qptl_folders[index]
            file_spotree = files_spotree_folders[index]
            file_intopt = files_intopt_folders[index]

            dnc_regret, max_regret, greedy_regret, spo_regret, qptl_regret, baseline_regret = find_regrets_knapsack(
                file_dnc[kfold],
                file_m[kfold],
                file_greedy[
                    kfold],
                file_spo[kfold], file_qptl[kfold],
                facecolors=facecolors,
                edgecolors=ecolors,
                models=models,
                is_plot=is_plot,
                number_of_rows_spo=number_of_rows_spo, number_of_rows_qptl=number_of_rows_qptl, is_run_time=is_run_time)
            spotree_runtime, spotree_regret = find_regrets_knapsack_spotree(file_spotree[kfold])
            intopt_runtime, intopt_regret = find_regrets_knap_intopt(file_intopt[kfold])
            dp_regret = find_regrets_knapsack_dp(file_dp[kfold], is_run_time = is_run_time)
            if is_normalize:
                spotree_regrets[index, kfold] = spotree_regret / baseline_regret
                dp_regrets[index, kfold] = dp_regret / baseline_regret
                dnc_regrets[index, kfold] = dnc_regret / baseline_regret
                greedy_regrets[index, kfold] = greedy_regret / baseline_regret
                max_regrets[index, kfold] = max_regret / baseline_regret
                spo_regrets[index, kfold] = spo_regret / baseline_regret
                baseline_regrets[index, kfold] = baseline_regret / baseline_regret
                qptl_regrets[index, kfold] = qptl_regret / baseline_regret
                intopt_regrets[index,kfold] = intopt_regret / baseline_regret

            else:
                spotree_regrets[index, kfold] = spotree_regret
                dp_regrets[index, kfold] = dp_regret
                dnc_regrets[index, kfold] = dnc_regret
                greedy_regrets[index, kfold] = greedy_regret
                max_regrets[index, kfold] = max_regret
                spo_regrets[index, kfold] = spo_regret
                baseline_regrets[index, kfold] = baseline_regret
                qptl_regrets[index, kfold] = qptl_regret
                intopt_regrets[index, kfold] = intopt_regret

            if is_run_time:
                intopt_regrets[index, kfold] = intopt_runtime
                spotree_regrets[index, kfold] = spotree_runtime
                baseline_regrets[index, kfold] = baseline_regret * 0

    fig = plt.figure()
    # ax1 = fig.add_subplot(1, 2, 1)
    # ind = np.arange(0, 6)
    # ax1.set_xticks(ind)
    # ax1.set_xticklabels(capacities)
    # ax1.set_title('Run Time vs Sample Points')
    # ax1.set_xlabel('Sample Points')
    # ax1.set_ylabel('Run Time(s)')
    # for index, file in enumerate(models):
    #     if index < 5:
    #         ax1.bar(ind + (0.15 * index - 37.5), run_times[index, :], color=colors[index], width=0.15)

    ax2 = fig.add_subplot(1, 1, 1)
    if xtick_labels is None:
        xtick_labels = capacities
    ind = np.arange(0, len(xtick_labels)) + 1
    xtickslocs = ind
    ax2.set_xticks(ind)
    ax2.set_xticklabels(xtick_labels)
    ax2.set_title(plot_title)
    ax2.set_xlabel('Capacities')
    ax2.set_ylabel('Test Regret')

    if is_run_time:
        ax2.set_ylabel('Run Time Until Early Stop(s)')

    regrets = np.array([np.mean(dnc_regrets, axis=1), np.mean(greedy_regrets, axis=1), np.mean(dp_regrets, axis=1),
                        np.mean(baseline_regrets, axis=1), np.mean(spotree_regrets, axis=1),
                        np.mean(spo_regrets, axis=1), np.mean(qptl_regrets, axis=1), np.mean(intopt_regrets,axis=1),
                        np.mean(max_regrets, axis=1)])
    errors = np.array([np.std(dnc_regrets, axis=1), np.std(greedy_regrets, axis=1), np.std(dp_regrets, axis=1),
                       np.std(baseline_regrets, axis=1), np.std(spotree_regrets, axis=1),
                       np.std(spo_regrets, axis=1), np.std(qptl_regrets, axis=1), np.std(intopt_regrets,axis=1),
                       np.std(max_regrets, axis=1)])
    models.extend([1])
    max_regret = np.max(np.mean(dnc_regrets, axis=1))
    for index, model in enumerate(models):
        ind_plot = index
        ax2.bar(ind + ((+0.10 * ind_plot) - 0.22), regrets[index, :], yerr=errors[index, :], color=facecolors[index],
                width=0.1, hatch=patterns[index],
                error_kw=dict(lw=0.5, capsize=1.5, capthick=1))  # ecolor=ecolors[index]

        if ylim is None:
            if is_normalize:
                ylim = 2
            else:
                ylim = max_regret* 1.5
        ax2.set_ylim(0, ylim)

    # ax2.xaxis.set_major_locator(mticker.FixedLocator(xtickslocs))
    labels = ['DnL', 'DnL-Greedy', 'DP', 'Ridge Regression', 'SPO-Forest', 'SPO-Relax', 'QPTL', 'IntOpt']
    ax2.legend(labels)
    if is_save:
        plt.savefig('../figs/' + str(dest_file_name))
    if is_show:
        plt.show()


def find_regrets_knap_intopt(file_intopt, is_run_time=False):

    df_list = read_file(filename=file_intopt, folder_path="", delimiter=',')

    regret, runtime = get_minval_regret_intopt(df_list)
    return runtime, regret

def get_minval_regret_intopt(df_list):
    df_list = np.array(df_list)
    val_regret = df_list[1:,0].astype('float')
    regret = df_list[1:,2].astype('float')
    run_time = df_list[1:,7].astype('float')
    min_val_regret = regret[np.argmin(val_regret)]
    min_val_runtime = run_time[np.argmin(val_regret)]
    print('het')
    return min_val_regret, min_val_runtime


def find_regrets_knapsack_dp(file_dp, is_run_time=False):
    # number_of_rows_spo

    run_times_dp, sub_epochs_dp, test_regrets_dp, val_regrets_dp = read_knapsack_files_dp(file_dp)

    min_len = min([len(arr) for arr in test_regrets_dp])

    min_val_test_regret_ddp, min_val_col_ddp, min_val_row_ddp = get_min_val_test_regret(val_regrets_dp, test_regrets_dp, min_len)
    if is_run_time:
        min_val_test_regret_ddp = run_times_dp[min_val_row_ddp][min_val_col_ddp]
    return min_val_test_regret_ddp


def find_regrets_knapsack_spotree(file_spotree,is_run_time=False):
    # regrets are all same for spo tree
    run_time, regret = read_knapsack_files_spotree(file_spotree)
    return run_time, regret


def read_knapsack_files_spotree(file_spotree,is_run_time=False):
    df_list = read_file(filename=file_spotree, folder_path="", delimiter=' ')
    run_time = float(df_list[1][4])
    regret = float(df_list[1][5])
    return run_time, regret


def find_regrets_icon_constraiend(file_greedy, file_spo, facecolors, edgecolors, models=[1, 1],
                                  is_plot=False, number_of_rows_spo=168):
    # number_of_rows_spo

    min_val_test_regret_dnc_greedy = 0
    min_val_test_regret_spo = 0
    if models[1]:
        run_times_greedy, sub_epochs_greedy, test_regrets_greedy, val_regrets_greedy = read_knapsack_files_predict(
            file_greedy)

    sub_epochs_spo, test_regrets_spo, val_regrets_spo = read_icon_files_spo(file_spo, number_of_rows_spo)
    min_len = min([len(arr) for arr in test_regrets_greedy])
    baseline_regret = test_regrets_greedy[0][0]

    if is_plot:

        fig = plt.figure()
        ax2 = fig.add_subplot(1, 1, 1)

        ax2.hlines(baseline_regret, 0, 6)
        # print predict opt
        ax2.set_title(file_greedy)
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('Test Regret')

        plot_regret_models = [test_regrets_greedy, test_regrets_spo]
        plot_epochs_models = [sub_epochs_greedy, sub_epochs_spo]
        for i, m in enumerate(models):
            if m:
                y_axis, x_axis = get_knapsack_axis(plot_regret_models[i], plot_epochs_models[i])
                mean_regret = np.median(y_axis, axis=0)
                plt.plot(x_axis, mean_regret)
                std = np.std(y_axis, axis=0)
                plt.fill_between(x_axis, mean_regret - std, mean_regret + std,
                                 alpha=0.5, edgecolor=edgecolors[i], facecolor=facecolors[i])

                if i == 3:
                    ax2.set_ylim(0, max(np.min(y_axis) * 2, baseline_regret * 2))
        plt.legend(['g', 'spo', 'regression'])

    if models[0]:
        min_len = min([len(arr) for arr in test_regrets_greedy])
        min_val_test_regret_dnc_greedy, min_val_col_greedy = get_min_val_test_regret(val_regrets_greedy,
                                                                                     test_regrets_greedy, min_len)
    if models[1]:
        min_val_test_regret_spo, min_val_col_spo = get_min_val_test_regret(val_regrets_spo, test_regrets_spo, min_len)

    if is_plot:
        plot_models_min_val_test_regret = [
            min_val_test_regret_dnc_greedy, min_val_test_regret_spo]
        plot_models_min_val_col = [min_val_col_greedy, min_val_col_spo]
        for i, m in enumerate(models):
            if m:
                plt.scatter(x_axis[plot_models_min_val_col[i]], plot_models_min_val_test_regret[i], color=facecolors[i],
                            ls='--')
        plt.show()

    return min_val_test_regret_dnc_greedy, min_val_test_regret_spo, baseline_regret


def find_regrets_knapsack(file_dnc, file_m, file_greedy, file_spo, file_qptl, facecolors, edgecolors,
                          models=[1, 1, 1, 1],
                          is_plot=False, number_of_rows_spo=168, number_of_rows_qptl=73, is_run_time=False):
    # number_of_rows_spo
    min_val_test_regret_dnc = 0
    min_val_test_regret_dnc_m = 0
    min_val_test_regret_dnc_greedy = 0
    min_val_test_regret_spo = 0
    baseline_regret = 0
    if models[0]:
        run_times_dnc, sub_epochs_dnc, test_regrets_dnc, val_regrets_dnc = read_knapsack_files_predict(file_dnc)
    if models[1]:
        run_times_m, sub_epochs_m, test_regrets_m, val_regrets_m = read_knapsack_files_predict(file_m)
    if models[2]:
        run_times_greedy, sub_epochs_greedy, test_regrets_greedy, val_regrets_greedy = read_knapsack_files_predict(
            file_greedy)

    run_times_spo,sub_epochs_spo, test_regrets_spo, val_regrets_spo = read_knapsack_files_spo(file_spo, number_of_rows_spo)

    run_times_qptl,sub_epochs_qptl, test_regrets_qptl, val_regrets_qptl = read_knapsack_files_qptl(file_qptl, number_of_rows_qptl)



    min_len = min([len(arr) for arr in test_regrets_dnc])
    baseline_regret = test_regrets_dnc[0][0]

    if is_plot:

        fig = plt.figure()
        ax2 = fig.add_subplot(1, 1, 1)

        ax2.hlines(baseline_regret, 0, 6)
        # print predict opt
        ax2.set_title(file_dnc)
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('Test Regret')

        plot_regret_models = [test_regrets_dnc, test_regrets_m, test_regrets_greedy, test_regrets_spo,
                              test_regrets_qptl]
        plot_epochs_models = [sub_epochs_dnc, sub_epochs_m, sub_epochs_greedy, sub_epochs_spo, sub_epochs_qptl]
        # plot_regret_models = [test_regrets_dnc, test_regrets_greedy, test_regrets_spo, test_regrets_qptl]
        # plot_epochs_models = [sub_epochs_dnc, sub_epochs_greedy, sub_epochs_spo, sub_epochs_qptl]
        patterns = ["-", "-", "-", "--", "-."]
        for i, m in enumerate(models[:5]):
            if m:
                print(i)
                y_axis, x_axis = get_knapsack_axis(plot_regret_models[i], plot_epochs_models[i])
                mean_regret = np.median(y_axis, axis=0)
                plt.plot(x_axis, mean_regret, linestyle=patterns[i])
                std = np.std(y_axis, axis=0)
                plt.fill_between(x_axis, mean_regret - std, mean_regret + std,
                                 alpha=0.5, edgecolor=edgecolors[i], facecolor=facecolors[i])

                if i == 3:
                    ax2.set_ylim(0, max(np.min(y_axis) * 2, baseline_regret * 2))
        plt.legend(['DnL', 'DnL-Max', 'DnL-Greedy', 'SPO-Relax', 'QPTL', 'Ridge Regression'])
    if models[0]:
        min_val_test_regret_dnc, min_val_col_dnc, min_val_row_dnc = get_min_val_test_regret(val_regrets_dnc, test_regrets_dnc, min_len)
    if models[2]:
        min_val_test_regret_dnc_greedy, min_val_col_greedy, min_val_row_greedy = get_min_val_test_regret(val_regrets_greedy,
                                                                                     test_regrets_greedy, min_len)
    if models[3]:
        min_val_test_regret_spo, min_val_col_spo,min_val_row_spo = get_min_val_test_regret(val_regrets_spo, test_regrets_spo, min_len)
    if models[4]:
        min_val_test_regret_qptl, min_val_col_qptl, min_val_row_qptl = get_min_val_test_regret(val_regrets_qptl, test_regrets_qptl,
                                                                             min_len)

    if is_run_time:
        if models[0]:
            min_val_test_regret_dnc = run_times_dnc[min_val_row_dnc][min_val_col_dnc]
        if models[1]:
            test_regrets_m = run_times_m[min_val_row_dnc][min_val_col_dnc]
        if models[2]:
            min_val_test_regret_dnc_greedy = run_times_greedy[min_val_row_greedy][min_val_col_greedy]
        min_val_test_regret_spo = run_times_spo[min_val_row_spo][min_val_col_spo]
        min_val_test_regret_qptl = run_times_qptl[min_val_row_qptl][min_val_col_qptl]

    if is_plot:
        plot_models_min_val_test_regret = [min_val_test_regret_dnc, min_val_test_regret_dnc_m,
                                           min_val_test_regret_dnc_greedy, min_val_test_regret_spo,
                                           min_val_test_regret_qptl]
        plot_models_min_val_col = [min_val_col_dnc, min_val_col_m, min_val_col_greedy, min_val_col_spo,
                                   min_val_col_qptl]
        for i, m in enumerate(models[:5]):
            if m:
                plt.scatter(x_axis[plot_models_min_val_col[i]], plot_models_min_val_test_regret[i], color=facecolors[i],
                            ls='--')
        plt.show()

    return min_val_test_regret_dnc, min_val_test_regret_dnc_m, min_val_test_regret_dnc_greedy, min_val_test_regret_spo, min_val_test_regret_qptl, baseline_regret


def get_min_val_test_regret(val_regrets, test_regrets, min_len):
    test_regrets = [n_iter_regrets[:min_len] for n_iter_regrets in test_regrets]
    test_regrets = np.array(test_regrets)
    val_regrets = [n_iter_regrets[:min_len] for n_iter_regrets in val_regrets]
    val_regrets = np.array(val_regrets)
    tmp_val_regrets = val_regrets
    print(tmp_val_regrets.argmin().any())
    min_val_row, min_val_col = np.unravel_index(tmp_val_regrets.argmin(), tmp_val_regrets.shape)
    min_val_test_regret = test_regrets[min_val_row, min_val_col]
    return min_val_test_regret, min_val_col, min_val_row


def get_knapsack_axis(test_regrets, sub_epochs):
    min_len = min([len(arr) for arr in test_regrets])
    test_regrets = [n_iter_regrets[:min_len] for n_iter_regrets in test_regrets]
    test_regrets = np.array(test_regrets)
    sub_epochs = sub_epochs[0]
    sub_epochs = sub_epochs[:min_len]
    x_axis = sub_epochs

    y_axis = test_regrets
    return y_axis, x_axis


def read_knapsack_files_spo(file, number_of_rows_spo=168):
    sub_epochs = []
    test_regrets = []
    val_regrets = []
    run_times = []
    df = np.array(read_file(filename=file, folder_path="", delimiter=','))
    iter_range = range(int((len(df) - 1) / number_of_rows_spo))
    for i in iter_range:
        test_regrets.append(df[1 + (i * number_of_rows_spo):(i + 1) * number_of_rows_spo, 7].astype(float))
        run_times.append(df[1 + (i * number_of_rows_spo):(i + 1) * number_of_rows_spo, 10].astype(float))
        sub_epoch = df[1 + (i * number_of_rows_spo):(i + 1) * number_of_rows_spo, 9].astype(float)
        sub_epochs.append(sub_epoch / 550)

        val_regrets.append(df[1 + (i * number_of_rows_spo):(i + 1) * number_of_rows_spo, 4].astype(float))
    return run_times, sub_epochs, test_regrets, val_regrets


def read_knapsack_files_qptl(file, number_of_rows_qptl=73):
    sub_epochs = []
    test_regrets = []
    val_regrets = []
    run_times = []
    df = np.array(read_file(filename=file, folder_path="", delimiter=','))
    iter_range = range(int((len(df) - 1) / number_of_rows_qptl))
    for i in iter_range:
        test_regrets.append(df[1 + (i * number_of_rows_qptl):(i + 1) * number_of_rows_qptl, 4].astype(float))
        run_times.append(df[1 + (i * number_of_rows_qptl):(i + 1) * number_of_rows_qptl, 10].astype(float))
        sub_epoch = df[1 + (i * number_of_rows_qptl):(i + 1) * number_of_rows_qptl, 8].astype(float)
        sub_epochs.append(sub_epoch / 550)

        val_regrets.append(df[1 + (i * number_of_rows_qptl):(i + 1) * number_of_rows_qptl, 0].astype(float))
    return run_times, sub_epochs, test_regrets, val_regrets


def read_icon_files_spo(file, number_of_rows_spo=65):
    sub_epochs = []
    test_regrets = []
    val_regrets = []
    df = np.array(read_file(filename=file, folder_path="", delimiter=','))
    iter_range = range(int((len(df) - 1) / number_of_rows_spo))
    for i in iter_range:
        test_regrets.append(df[1 + (i * number_of_rows_spo):(i + 1) * number_of_rows_spo, 3].astype(float))

        sub_epoch = df[1 + (i * number_of_rows_spo):(i + 1) * number_of_rows_spo, 4].astype(float)
        sub_epochs.append(sub_epoch / 550)
        val_regrets.append(df[1 + (i * number_of_rows_spo):(i + 1) * number_of_rows_spo, 1].astype(float))
    return sub_epochs, test_regrets, val_regrets


def read_knapsack_files_predict(file, header_length_pdo=3):
    run_times = []
    sub_epochs = []
    test_regrets = []
    val_regrets = []

    df_list = read_file(filename=file, folder_path="", delimiter=' ')
    start_indexes, end_indexes = find_indexes(df_list)
    number_of_rows_pdo = int((len(df_list) - 2 * header_length_pdo) / 2) - 1
    iter_no_pdo = int(len(start_indexes))
    for i in range(iter_no_pdo):
        start_index = start_indexes[i]
        end_index = end_indexes[i]
        df = np.array(df_list[start_index:end_index]).astype(float)
        run_times.append(df[:, 2])
        sub_epochs.append((np.max(df[:, 0]) + 1) * df[:, 1] / np.max(df[:, 1]))
        test_regrets.append(df[:, 4])
        val_regrets.append(df[:, 5])

    return run_times, sub_epochs, test_regrets, val_regrets


def read_knapsack_files_dp(file, header_length_pdo=3):
    run_times = []
    sub_epochs = []
    test_regrets = []
    val_regrets = []

    df_list = read_file(filename=file, folder_path="", delimiter=' ')
    start_indexes, end_indexes = find_indexes(df_list)
    # number_of_rows_pdo = int((len(df_list) - 2 * header_length_pdo) / 2) - 1
    iter_no_pdo = int(len(start_indexes))
    for i in range(iter_no_pdo):
        start_index = start_indexes[i]
        end_index = end_indexes[i]
        df = np.array(df_list[start_index:end_index]).astype(float)
        run_times.append(df[:, 2])
        sub_epochs.append((np.max(df[:, 0]) + 1) * df[:, 1] / np.max(df[:, 1]))
        test_regrets.append(df[:, 4])
        val_regrets.append(df[:, 5])

    return run_times, sub_epochs, test_regrets, val_regrets


def plot_ICON():
    file_folder_predict = 'icon/Easy'
    file_folder_spo = '../SPO'

    files_predict = ['icon-l1-0115.csv', 'icon-l8-0115.csv', 'icon-l12-0115.csv']
    # files_spo = ['Load1_SPO_warmstart_corrected.csv', 'Load8_SPO_warmstart_corrected.csv',
    #              'Load12_SPO_warmstart_corrected.csv', 'Load20_SPO_warmstart_corrected.csv']

    files_spo = ['Load1_SPO_warmstart_corrected.csv', 'Load8_SPO_warmstart_corrected.csv',
                 'Load12_SPO_warmstart_corrected.csv']

    titles = ['ICON Load1 - 3 Machines 10 Jobs', 'ICON Load8 - 10 Machines 50 Jobs', 'ICON load12 - 3 Machines 10 Jobs']

    number_of_files = 3
    colors = ['y', 'g', 'c', 'm']
    capacities = ['1', '8', '12']
    models = ['Divide and Conquer', 'Divide and Conquer Max Selection',
              'Greedy', 'Linear Regression']

    regrets_predict = np.zeros((4, number_of_files))
    dfs = []
    for index, file in enumerate(files_predict):
        df = np.array(read_file(filename=file, folder_path=file_folder_predict))
        # print(df[0])
        # print(index)
        regrets_predict[:, index] = df[0][2:]
        dfs.append(df)

    number_of_rows = 55
    regrets_spo = np.zeros((number_of_rows - 1, number_of_files))
    sub_epochs = np.zeros((number_of_rows - 1, number_of_files))
    val_spo = np.zeros((number_of_rows - 1, number_of_files))
    for index, file in enumerate(files_spo):
        df = np.array(read_file(filename=file, folder_path=file_folder_spo, delimiter=','))
        # print(df[1])
        # print(index)
        # print(df)
        print(df[1:number_of_rows, 4])
        regrets_spo[:, index] = df[1:number_of_rows, 3]
        sub_epochs[:, index] = df[1:number_of_rows, 4]
        val_spo[:, index] = df[1:number_of_rows, 1]
        dfs.append(df)

    # fig = plt.figure()
    # ax2 = fig.add_subplot(1, 1,1)
    # ax2.set_title('Original Load1 3 Machines 10 Jobs')
    # ax2.set_xlabel('Subepochs')
    # ax2.set_ylabel('Regret')
    # ax2.hlines(17000, 0, sub_epochs[-1, 3], colors[index])
    # ax2.plot(sub_epochs[:, 3], regrets_spo[:, 3])
    # ax2.plot(sub_epochs[:, 3], val_spo[:, 3])
    # labels_spo = ['SPO_regret', 'SPO_val_regret']
    # legend_labels = labels_spo + models
    # plt.legend(labels=legend_labels)
    # plt.show()

    for file_no in range(number_of_files):

        # file_no = 2
        fig = plt.figure()
        ax2 = fig.add_subplot(1, 1, 1)
        ind = np.arange(0, number_of_files) + 1
        ax2.set_title(titles[file_no])
        ax2.set_xlabel('Subepochs')
        ax2.set_ylabel('Regret')
        ax2.plot(sub_epochs[:, file_no], regrets_spo[:, file_no])
        # ax2.plot(sub_epochs[:, file_no], val_spo[:, file_no])

        for index, model in enumerate(models):
            ax2.hlines(regrets_predict[index, file_no], 0, sub_epochs[-1, file_no], colors[index])
            # labels_spo = ['SPO_regret','SPO_val_regret']

            labels_spo = ['SPO_regret']
            legend_labels = labels_spo + models
            # handles, _ = ax2.get_legend_handles_labels()
            # print(handles)
            plt.legend(labels=legend_labels)
        # ax2.legend(models)
        xtickslocs = ind
        # ax2.set_xticks(ind + 1, capacities)
        # ax2.set_xticklabels(capacities)
        # ax2.set_title('Regret vs Capacities')
        # # ax2.set_xlabel('Capacities')
        # ax2.set_ylabel('Regret')
        #
        # for index, file in enumerate(models):
        #     ax2.bar(ind + ((+0.15 * index) - 0.22), regrets_predict[index, :], color=colors[index], width=0.15)
        # ax2.xaxis.set_major_locator(mticker.FixedLocator(xtickslocs))

    plt.show()


def plot_spo_relax_og():
    file_folder_spo = 'icon/Easy/spo_og'

    files_spo = ['Load1_SPO_warmstart_corrected.csv']

    titles = ['ICON Load12 - 3 Machines 10 Jobs']

    number_of_files = 1
    colors = ['y', 'm']
    capacities = ['12']

    number_of_rows = 66
    n_iter = 10
    regrets_spo = np.zeros((number_of_rows - 1, n_iter))
    sub_epochs = np.zeros((number_of_rows - 1, n_iter))
    val_spo = np.zeros((number_of_rows - 1, n_iter))
    for index, file in enumerate(files_spo):
        df = np.array(read_file(filename=file, folder_path=file_folder_spo, delimiter=','))
        # print(df[1])
        # print(index)
        # print(df)
        for iter in range(n_iter):
            # print(df[1+iter*number_of_rows:(iter+1)*number_of_rows, 4])
            regrets_spo[:, iter] = df[1 + (iter * number_of_rows):(iter + 1) * number_of_rows, 3]
            sub_epochs[:, iter] = df[1 + (iter * number_of_rows):(iter + 1) * number_of_rows, 4]
            val_spo[:, iter] = df[1 + (iter * number_of_rows):(iter + 1) * number_of_rows, 1]

    fig = plt.figure()
    ax2 = fig.add_subplot(1, 1, 1)
    ax2.set_title('Original Load12 3 Machines 10 Jobs')
    ax2.set_xlabel('Subepochs')
    ax2.set_ylabel('Regret')
    ax2.hlines(17000, 0, sub_epochs[-1, 0], colors[index])
    for iter in range(10):
        print(sub_epochs[:, iter])
        ax2.plot(sub_epochs[:, iter], regrets_spo[:, iter])
        # ax2.plot(sub_epochs[:, iter], val_spo[:, iter])
    # labels_spo = ['SPO_regret', 'SPO_val_regret']
    # legend_labels = labels_spo + models
    # plt.legend(labels=legend_labels)
    plt.show()


def plot_minibatch():
    file_folder_predict = 'icon/Easy'
    file_folder_spo = '../SPO'

    files_predict = ['icon_minibatch_greedy_load12.txt']
    # files_spo = ['Load1_SPO_warmstart_corrected.csv', 'Load8_SPO_warmstart_corrected.csv',
    #              'Load12_SPO_warmstart_corrected.csv', 'Load20_SPO_warmstart_corrected.csv']

    titles = ['ICON Load1 - 3 Machines 10 Jobs', 'ICON Load8 - 10 Machines 50 Jobs', 'ICON load12 - 3 Machines 10 Jobs']

    number_of_files = 1
    colors = ['y']
    dfs = []
    for index, file in enumerate(files_predict):
        df = np.array(read_file(filename=file, folder_path=file_folder_predict, delimiter=' '))
        df = list(filter(None, df))
        df = [float(regret) for sub in df for regret in sub]
    fig = plt.figure()
    ax2 = fig.add_subplot(1, 1, 1)
    ax2.set_title('Original Load12 3 Machines 10 Jobs')
    ax2.set_xlabel('Subepochs')
    ax2.set_ylabel('Regret')
    ax2.hlines(17000, 0, len(df))
    plt.plot(df)
    plt.show()


def plot_minibatch_coordinate():
    file_folder_predict = 'icon/Easy'
    file_folder_spo = '../SPO'

    files_predict = ['icon-l12k0N0-DIVIDE_AND_CONQUER-0-1.csv', 'icon-l12DIVIDE_AND_CONQUER_MAX-0-1.csv',
                     'icon-l12DIVIDE_AND_CONQUER_GREEDY-0-1.csv']
    # files_spo = ['Load1_SPO_warmstart_corrected.csv', 'Load8_SPO_warmstart_corrected.csv',
    #              'Load12_SPO_warmstart_corrected.csv', 'Load20_SPO_warmstart_corrected.csv']

    files_spo = ['Load12_SPO_warmstart_corrected.csv']

    titles = ['ICON Load1 - 3 Machines 10 Jobs', 'ICON Load8 - 10 Machines 50 Jobs', 'ICON load12 - 3 Machines 10 Jobs']

    number_of_files = 3
    colors = ['y', 'g', 'c', 'm']
    capacities = ['1', '8', '12']
    models = ['Divide and Conquer', 'Divide and Conquer Max Selection',
              'Greedy', 'Linear Regression']

    number_of_files = len(files_predict)
    colors = ['y']
    dfs = []
    run_times = []
    sub_epochs = []
    test_regrets = []
    for index, file in enumerate(files_predict):
        df = read_file(filename=file, folder_path=file_folder_predict, delimiter=' ')
        df = np.array(df[3:]).astype(float)
        run_times.append(df[:, 2])
        sub_epochs.append(3 * df[:, 1] / np.max(df[:, 1]))
        test_regrets.append(df[:, 4])
        # print(df[0])

    iter_no = 10
    iter_range = range(iter_no)
    number_of_rows = 66
    regrets_spo = []
    # regrets_spo = np.zeros((number_of_rows - 1, number_of_files))
    sub_epochs_spo = np.zeros((number_of_rows - 1, number_of_files))
    val_spo = np.zeros((number_of_rows - 1, number_of_files))
    styles = ['-', '-x', '-o']

    for index, file in enumerate(files_spo):
        regrets_spo_np = np.zeros((number_of_rows - 1, iter_no))
        for i in iter_range:
            df = np.array(read_file(filename=file, folder_path=file_folder_spo, delimiter=','))
            # print(df[1])
            # print(index)
            # print(df)

            regrets_spo_np[:, i] = df[1 + (i * number_of_rows):(i + 1) * number_of_rows, 3].astype(float)

            sub_epochs_spo[:, index] = df[1 + (i * number_of_rows):(i + 1) * number_of_rows, 4]
            dfs.append(df)
        regrets_spo.append(regrets_spo_np)
    x_axis = sub_epochs
    y_axis = test_regrets

    fig = plt.figure()
    ax2 = fig.add_subplot(1, 1, 1)
    ax2.set_title('Load12 3 Machines 10 Jobs')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('Regret')

    # ax2.hlines(17663, 0, max([np.max(array) for array in x_axis]))
    ax2.hlines(17663, 0, 6)
    N = 2
    plot_no = 8
    for i in range(number_of_files):
        plt.plot(x_axis[i][:-N + 1], np.convolve(y_axis[i], np.ones((N,)) / N, mode='valid'), styles[i])
    for index, file in enumerate(files_spo):
        y_axis = regrets_spo[index]
        x_axis = 6 * sub_epochs_spo[:, index] / np.max(sub_epochs_spo[:, index])
        # mean spo relax
        plt.plot(x_axis, np.median(y_axis, axis=1), color='r', ls='--')
        # individual spo relax
        # plt.plot(x_axis, y_axis[:,plot_no], color='r', ls='--')
        mean_regret = np.median(y_axis, axis=1)
        error = [mean_regret - np.min(y_axis, axis=1), np.max(y_axis, axis=1) - mean_regret]
        std = np.std(y_axis, axis=1)
        error = std
        print(error)
        ax2.errorbar(x_axis, np.median(y_axis, axis=1), yerr=error, fmt='-o')
        # for j in range(len(x_axis)-1):
        #
        #     print(y_axis.shape)
        #     ax2.add_patch(
        #         patches.Rectangle(
        #             (0.1, 0.1),
        #             0.5,
        #             0.5,
        #             fill=False  # remove background
        #         ))
        # ax2.axhspan(xmin=x_axis[j], xmax=x_axis[j + 1], ymin=np.min(y_axis[j, :]),
        #             ymax=np.max(y_axis[j, :]), color='r', alpha=0.5)

    labels = ['dnc', 'dnc_max', 'greedy', 'SPO', 'baseline']
    plt.legend(labels=labels)
    plt.plot()
    plt.show()


def plot_minibatch_weighted_knapsack():
    file_folder_predict = 'Knapsack/weighted'
    # file_folder_predict = 'Tests/icon/Easy'
    file_folder_spo = '../SPO'

    # files_predict = ['knapsack-w-c-24-DIVIDE_AND_CONQUER-0-1.csv', 'knapsack-w-c-24-DIVIDE_AND_CONQUER_GREEDY-0-1.csv', 'knapsack-w-c-24-DIVIDE_AND_CONQUER_MAX-0-1.csv']

    # files_predict = ['knapsack-w-c-12-DIVIDE_AND_CONQUER-0-1.csv', 'knapsack-w-c-12-DIVIDE_AND_CONQUER_GREEDY-0-1.csv', 'knapsack-w-c-12-DIVIDE_AND_CONQUER_MAX-0-1.csv']
    files_predict = ['knapsack-w-c-48-DIVIDE_AND_CONQUER-0-1.csv', 'knapsack-w-c-48-DIVIDE_AND_CONQUER_GREEDY-0-1.csv',
                     'knapsack-w-c-48-DIVIDE_AND_CONQUER_MAX-0-1.csv']
    # files_predict = ['icon-l12DIVIDE_AND_CONQUER-0-1.csv', 'icon-l12DIVIDE_AND_CONQUER_MAX-0-1.csv', 'icon-l12DIVIDE_AND_CONQUER_GREEDY-0-1.csv']

    files_spo = ['Load1_SPO_warmstart_corrected.csv', 'Load8_SPO_warmstart_corrected.csv',
                 'Load12_SPO_warmstart_corrected.csv']

    titles = ['ICON Load1 - 3 Machines 10 Jobs', 'ICON Load8 - 10 Machines 50 Jobs', 'ICON load12 - 3 Machines 10 Jobs']

    number_of_files = 3
    colors = ['y', 'g', 'c', 'm']
    capacities = ['1', '8', '12']
    models = ['Divide and Conquer', 'Divide and Conquer Max Selection',
              'Greedy', 'Linear Regression']

    number_of_files = len(files_predict)
    colors = ['y']
    dfs = []
    run_times = []
    sub_epochs = []
    test_regrets = []
    for index, file in enumerate(files_predict):
        df = read_file(filename=file, folder_path=file_folder_predict, delimiter=' ')
        df = np.array(df[3:]).astype(float)
        run_times.append(df[:, 2])
        sub_epochs.append(df[:, 1])
        test_regrets.append(df[:, 4])
        # print(df[0])
    #
    # number_of_rows = 55
    # regrets_spo = np.zeros((number_of_rows - 1, number_of_files))
    # sub_epochs = np.zeros((number_of_rows - 1, number_of_files))
    # val_spo = np.zeros((number_of_rows - 1, number_of_files))
    # for index, file in enumerate(files_spo):
    #     df = np.array(read_file(filename=file, folder_path=file_folder_spo, delimiter=','))
    #     # print(df[1])
    #     # print(index)
    #     # print(df)
    #     print(df[1:number_of_rows, 4])
    #     regrets_spo[:, index] = df[1:number_of_rows, 3]
    #     sub_epochs[:, index] = df[1:number_of_rows, 4]
    #     val_spo[:, index] = df[1:number_of_rows, 1]
    #     dfs.append(df)

    x_axis = run_times
    y_axis = test_regrets
    fig = plt.figure()
    ax2 = fig.add_subplot(1, 1, 1)
    ax2.set_title('%20 Capacity ')
    ax2.set_xlabel('Run Time')
    ax2.set_ylabel('Regret')

    ax2.hlines(test_regrets[0][0], 0, max([np.max(array) for array in x_axis]))
    N = 16
    for i in range(number_of_files):
        plt.plot(x_axis[i][:-N + 1], np.convolve(y_axis[i], np.ones((N,)) / N, mode='valid'))
    # for index, file in enumerate(files_spo):
    #     plt.plot(sub_epochs[:,i], regrets_spo[:,i])
    labels = ['dnc', 'greedy', 'dnc_max', 'spo', 'baseline']
    plt.legend(labels=labels)
    plt.plot()
    plt.show()


def plot_minibatch_icon_mean():
    file_folder_predict = 'icon/Easy'
    file_folder_spo = '../SPO'

    files_predict = ['icon-l12DIVIDE_AND_CONQUER-0-1.csv', 'icon-l12-N0-DIVIDE_AND_CONQUER-0-1.csv',
                     'icon-l12-N1-DIVIDE_AND_CONQUER-0-1.csv']
    # files_spo = ['Load1_SPO_warmstart_corrected.csv', 'Load8_SPO_warmstart_corrected.csv',
    #              'Load12_SPO_warmstart_corrected.csv', 'Load20_SPO_warmstart_corrected.csv']

    files_spo = ['Load12_SPO_warmstart_corrected.csv']

    titles = ['ICON Load1 - 3 Machines 10 Jobs', 'ICON Load8 - 10 Machines 50 Jobs', 'ICON load12 - 3 Machines 10 Jobs']

    number_of_files = 3
    colors = ['y', 'g', 'c', 'm']
    capacities = ['1', '8', '12']
    models = ['Divide and Conquer', 'Divide and Conquer Max Selection',
              'Greedy', 'Linear Regression']

    number_of_files = len(files_predict)
    colors = ['y']
    dfs = []
    run_times = []
    sub_epochs = []
    test_regrets = []
    for index, file in enumerate(files_predict):
        df = read_file(filename=file, folder_path=file_folder_predict, delimiter=' ')
        df = np.array(df[3:]).astype(float)
        run_times.append(df[:, 2])
        sub_epochs.append(3 * df[:, 1] / np.max(df[:, 1]))
        test_regrets.append(df[:, 4].astype(float))
        # print(df[0])

    iter_no = 10
    iter_range = range(iter_no)
    number_of_rows = 66
    regrets_spo = []
    # regrets_spo = np.zeros((number_of_rows - 1, number_of_files))
    sub_epochs_spo = np.zeros((number_of_rows - 1, number_of_files))
    val_spo = np.zeros((number_of_rows - 1, number_of_files))
    styles = ['-', '-x', '-o']

    for index, file in enumerate(files_spo):
        regrets_spo_np = np.zeros((number_of_rows - 1, iter_no))
        for i in iter_range:
            df = np.array(read_file(filename=file, folder_path=file_folder_spo, delimiter=','))
            # print(df[1])
            # print(index)
            # print(df)

            regrets_spo_np[:, i] = df[1 + (i * number_of_rows):(i + 1) * number_of_rows, 3].astype(float)

            sub_epochs_spo[:, index] = df[1 + (i * number_of_rows):(i + 1) * number_of_rows, 4]
            dfs.append(df)
        regrets_spo.append(regrets_spo_np)

    fig = plt.figure()
    ax2 = fig.add_subplot(1, 1, 1)
    ax2.set_title('Load12 3 Machines 10 Jobs')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('Regret')

    # ax2.hlines(17663, 0, max([np.max(array) for array in x_axis]))
    ax2.hlines(17663, 0, 6)
    N = 2
    plot_no = 8
    # print predict opt
    min_len = min([len(arr) for arr in test_regrets])
    test_regrets = [n_iter_regrets[:min_len] for n_iter_regrets in test_regrets]
    test_regrets = np.array(test_regrets)
    print(sub_epochs)
    sub_epochs = sub_epochs[0]
    sub_epochs = sub_epochs[:min_len]
    x_axis = sub_epochs

    print(test_regrets)
    y_axis = test_regrets
    mean_regret = np.median(y_axis, axis=0)
    plt.plot(x_axis, mean_regret)
    std = np.std(y_axis, axis=0)
    error = std
    ax2.errorbar(x_axis, np.median(y_axis, axis=0), yerr=error, fmt='-o')

    for index, file in enumerate(files_spo):
        y_axis = regrets_spo[index]
        x_axis = 6 * sub_epochs_spo[:, index] / np.max(sub_epochs_spo[:, index])
        # mean spo relax
        plt.plot(x_axis, np.median(y_axis, axis=1), color='r', ls='--')
        # individual spo relax
        # plt.plot(x_axis, y_axis[:,plot_no], color='r', ls='--')
        mean_regret = np.median(y_axis, axis=1)
        error = [mean_regret - np.min(y_axis, axis=1), np.max(y_axis, axis=1) - mean_regret]
        std = np.std(y_axis, axis=1)
        error = std
        print(error)
        ax2.errorbar(x_axis, np.median(y_axis, axis=1), yerr=error, fmt='-o')
        # for j in range(len(x_axis)-1):
        #
        #     print(y_axis.shape)
        #     ax2.add_patch(
        #         patches.Rectangle(
        #             (0.1, 0.1),
        #             0.5,
        #             0.5,
        #             fill=False  # remove background
        #         ))
        # ax2.axhspan(xmin=x_axis[j], xmax=x_axis[j + 1], ymin=np.min(y_axis[j, :]),
        #             ymax=np.max(y_axis[j, :]), color='r', alpha=0.5)

    labels = ['dnc', 'dnc_max', 'greedy', 'SPO', 'baseline']
    plt.legend(labels=labels)
    plt.plot()
    plt.show()


def plot_minibatch_icon_mean():
    file_folder_predict = 'icon/Easy'
    file_folder_spo = '../SPO'

    files_predict = ['icon-l12DIVIDE_AND_CONQUER-0-1.csv', 'icon-l12-N0-DIVIDE_AND_CONQUER-0-1.csv',
                     'icon-l12-N1-DIVIDE_AND_CONQUER-0-1.csv', 'icon-l12k0N0-DIVIDE_AND_CONQUER-0-1.csv']
    # files_spo = ['Load1_SPO_warmstart_corrected.csv', 'Load8_SPO_warmstart_corrected.csv',
    #              'Load12_SPO_warmstart_corrected.csv', 'Load20_SPO_warmstart_corrected.csv']

    files_spo = ['Load12_SPO_warmstart_corrected.csv']

    titles = ['ICON Load1 - 3 Machines 10 Jobs']

    colors = ['y', 'g', 'c', 'm']
    capacities = ['1', '8', '12']
    models = ['Divide and Conquer']

    number_of_files = len(files_predict)
    colors = ['y']
    dfs = []
    run_times = []
    sub_epochs = []
    test_regrets = []
    for index, file in enumerate(files_predict):
        df = read_file(filename=file, folder_path=file_folder_predict, delimiter=' ')
        df = np.array(df[3:]).astype(float)
        run_times.append(df[:, 2])
        sub_epochs.append(3 * df[:, 1] / np.max(df[:, 1]))
        test_regrets.append(df[:, 4].astype(float))
        # print(df[0])

    iter_no = 10
    iter_range = range(iter_no)
    number_of_rows = 66
    regrets_spo = []
    # regrets_spo = np.zeros((number_of_rows - 1, number_of_files))
    sub_epochs_spo = np.zeros((number_of_rows - 1, number_of_files))
    val_spo = np.zeros((number_of_rows - 1, number_of_files))
    styles = ['-', '-x', '-o']

    for index, file in enumerate(files_spo):
        regrets_spo_np = np.zeros((number_of_rows - 1, iter_no))
        for i in iter_range:
            df = np.array(read_file(filename=file, folder_path=file_folder_spo, delimiter=','))
            # print(df[1])
            # print(index)
            # print(df)

            regrets_spo_np[:, i] = df[1 + (i * number_of_rows):(i + 1) * number_of_rows, 3].astype(float)

            sub_epochs_spo[:, index] = df[1 + (i * number_of_rows):(i + 1) * number_of_rows, 4]
            dfs.append(df)
        regrets_spo.append(regrets_spo_np)

    fig = plt.figure()
    ax2 = fig.add_subplot(1, 1, 1)
    ax2.set_title('Load12 3 Machines 10 Jobs')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('Regret')

    # ax2.hlines(17663, 0, max([np.max(array) for array in x_axis]))
    ax2.hlines(17663, 0, 6)
    N = 2
    plot_no = 8
    # print predict opt
    min_len = min([len(arr) for arr in test_regrets])
    test_regrets = [n_iter_regrets[:min_len] for n_iter_regrets in test_regrets]
    test_regrets = np.array(test_regrets)
    print(sub_epochs)
    sub_epochs = sub_epochs[0]
    sub_epochs = sub_epochs[:min_len]
    x_axis = sub_epochs

    print(test_regrets)
    y_axis = test_regrets
    mean_regret = np.median(y_axis, axis=0)
    plt.plot(x_axis, mean_regret)
    std = np.std(y_axis, axis=0)
    error = std
    ax2.errorbar(x_axis, np.median(y_axis, axis=0), yerr=error, fmt='-o')

    for index, file in enumerate(files_spo):
        y_axis = regrets_spo[index]
        x_axis = 6 * sub_epochs_spo[:, index] / np.max(sub_epochs_spo[:, index])
        # mean spo relax
        plt.plot(x_axis, np.median(y_axis, axis=1), color='r', ls='--')
        # individual spo relax
        # plt.plot(x_axis, y_axis[:,plot_no], color='r', ls='--')
        mean_regret = np.median(y_axis, axis=1)
        error = [mean_regret - np.min(y_axis, axis=1), np.max(y_axis, axis=1) - mean_regret]
        std = np.std(y_axis, axis=1)
        error = std
        print(error)
        ax2.errorbar(x_axis, np.median(y_axis, axis=1), yerr=error, fmt='-o')
        # for j in range(len(x_axis)-1):
        #
        #     print(y_axis.shape)
        #     ax2.add_patch(
        #         patches.Rectangle(
        #             (0.1, 0.1),
        #             0.5,
        #             0.5,
        #             fill=False  # remove background
        #         ))
        # ax2.axhspan(xmin=x_axis[j], xmax=x_axis[j + 1], ymin=np.min(y_axis[j, :]),
        #             ymax=np.max(y_axis[j, :]), color='r', alpha=0.5)

    labels = ['dnc', 'SPO', 'baseline']
    plt.legend(labels=labels)
    plt.plot()
    plt.show()


def plot_minibatch_icon_mean_kfold0():
    file_folder_predict = 'icon/Easy/load12/spartan'
    file_folder_spo = 'icon/Easy/load12/spo'

    files_predict = ['icon-l12k0N0-DIVIDE_AND_CONQUER-0-1.csv', 'icon-l12k0N0v2-DIVIDE_AND_CONQUER-0-1.csv',
                     'icon-l12k0N1-DIVIDE_AND_CONQUER-0-1.csv', 'icon-l12k0N1v2-DIVIDE_AND_CONQUER-0-1.csv',
                     'icon-l12k0N2-DIVIDE_AND_CONQUER-0-1.csv', 'icon-l12k0N2v2-DIVIDE_AND_CONQUER-0-1.csv',
                     'icon-l12k0N3-DIVIDE_AND_CONQUER-0-1.csv',
                     'icon-l12k0N3v2-DIVIDE_AND_CONQUER-0-1.csv', 'icon-l12k0N4-DIVIDE_AND_CONQUER-0-1.csv',
                     'icon-l12k0N4v2-DIVIDE_AND_CONQUER-0-1.csv']

    files_spo = ['12Load12_SPO_warmstart_corrected_kfold0.csv']

    titles = ['ICON Load1 - 3 Machines 10 Jobs']

    number_of_files = 1
    colors = ['y', 'g', 'c', 'm']
    capacities = ['1', '8', '12']
    models = ['Divide and Conquer']

    number_of_files = len(files_predict)

    run_times = []
    sub_epochs = []
    test_regrets = []
    val_regrets = []
    for index, file in enumerate(files_predict):
        df = read_file(filename=file, folder_path=file_folder_predict, delimiter=' ')
        df = np.array(df[3:]).astype(float)
        run_times.append(df[:, 2])
        sub_epochs.append(3 * df[:, 1] / np.max(df[:, 1]))
        test_regrets.append(df[:, 4].astype(float))
        val_regrets.append(df[:, 5].astype(float))
        # print(df[0])

    iter_no = 10
    iter_range = range(iter_no)
    number_of_rows = 66
    regrets_spo = []
    sub_epochs_spo = np.zeros((number_of_rows - 1, number_of_files))

    val_spo = []

    for index, file in enumerate(files_spo):
        regrets_spo_np = np.zeros((number_of_rows - 1, iter_no))
        val_regrets_spo_np = np.zeros((number_of_rows - 1, iter_no))
        for i in iter_range:
            df = np.array(read_file(filename=file, folder_path=file_folder_spo, delimiter=','))

            regrets_spo_np[:, i] = df[1 + (i * number_of_rows):(i + 1) * number_of_rows, 3].astype(float)

            sub_epochs_spo[:, index] = df[1 + (i * number_of_rows):(i + 1) * number_of_rows, 4]

            val_regrets_spo_np[:, i] = df[1 + (i * number_of_rows):(i + 1) * number_of_rows, 1].astype(float)
        regrets_spo.append(regrets_spo_np)
        val_spo.append(val_regrets_spo_np)

    fig = plt.figure()
    ax2 = fig.add_subplot(1, 1, 1)
    ax2.set_title('Load12 3 Machines 10 Jobs Fold 0')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('Regret')

    ax2.hlines(19618, 0, 6)
    # print predict opt
    min_len = min([len(arr) for arr in test_regrets])
    test_regrets = [n_iter_regrets[:min_len] for n_iter_regrets in test_regrets]
    test_regrets = np.array(test_regrets)
    sub_epochs = sub_epochs[0]
    sub_epochs = sub_epochs[:min_len]
    x_axis = sub_epochs

    # print(test_regrets)
    y_axis = test_regrets
    mean_regret = np.median(y_axis, axis=0)
    plt.plot(x_axis, mean_regret)
    std = np.std(y_axis, axis=0)
    plt.fill_between(x_axis, mean_regret - std, mean_regret + std,
                     alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')

    val_regrets = [n_iter_regrets[:min_len] for n_iter_regrets in val_regrets]
    val_regrets = np.array(val_regrets)
    # if you want to find test regrets for each iteration use this
    # tmp_val_regrets = val_regrets
    # min_val_ind = np.argmin(tmp_val_regrets, axis=1)
    # min_val_test_regrets = tmp_val_regrets[np.arange(len(y_axis)), min_val_ind]
    # min_val_test_regret = np.amin(min_val_test_regrets)

    tmp_val_regrets = val_regrets
    min_val_row, min_val_col = np.unravel_index(tmp_val_regrets.argmin(), tmp_val_regrets.shape)
    min_val_test_regret = y_axis[min_val_row, min_val_col]
    plt.scatter(x_axis[min_val_col], min_val_test_regret, color='b', ls='--')
    us_regret = min_val_test_regret
    print('us', min_val_test_regret)
    for index, file in enumerate(files_spo):
        y_axis = regrets_spo[index]
        x_axis = 6 * sub_epochs_spo[:, index] / np.max(sub_epochs_spo[:, index])

        # tmp_val_regrets = val_spo[index]
        # min_val_ind = np.argmin(tmp_val_regrets, axis=0)
        # min_val_test_regrets = y_axis[min_val_ind, np.arange(iter_no)]
        # min_val_test_regret = np.amin(min_val_test_regrets)

        tmp_val_regrets = val_spo[index]
        min_val_row, min_val_col = np.unravel_index(tmp_val_regrets.argmin(), tmp_val_regrets.shape)
        min_val_test_regret = y_axis[min_val_row, min_val_col]

        plt.scatter(x_axis[min_val_col], min_val_test_regret, color='r', ls='--')
        spo_regret = min_val_test_regret
        # mean spo relax
        plt.plot(x_axis, np.median(y_axis, axis=1), color='r', ls='--')
        mean_regret = np.median(y_axis, axis=1)

        std = np.std(y_axis, axis=1)

        plt.fill_between(x_axis, mean_regret - std, mean_regret + std,
                         alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        print('spo', min_val_test_regret)
    labels = ['Us', 'SPO-relax', 'baseline']
    plt.legend(labels=labels)
    plt.plot()
    plt.savefig('figs/icon_l12_k0.pdf')
    plt.show()

    return us_regret, spo_regret


def plot_minibatch_icon_mean_kfold1():
    file_folder_predict = 'icon/Easy/load12/spartan'
    file_folder_spo = 'icon/Easy/load12/spo'

    files_predict = ['icon-l12k1N0-DIVIDE_AND_CONQUER-0-1.csv',
                     'icon-l12k1N0v2-DIVIDE_AND_CONQUER-0-1.csv',
                     'icon-l12k1N1-DIVIDE_AND_CONQUER-0-1.csv',
                     'icon-l12k1N1v2-DIVIDE_AND_CONQUER-0-1.csv',
                     'icon-l12k1N2-DIVIDE_AND_CONQUER-0-1.csv',
                     'icon-l12k1N2v2-DIVIDE_AND_CONQUER-0-1.csv',
                     'icon-l12k1N3-DIVIDE_AND_CONQUER-0-1.csv',
                     'icon-l12k1N3v2-DIVIDE_AND_CONQUER-0-1.csv',
                     'icon-l12k1N4-DIVIDE_AND_CONQUER-0-1.csv',
                     'icon-l12k1N4v2-DIVIDE_AND_CONQUER-0-1.csv']

    files_spo = ['12Load12_SPO_warmstart_corrected_kfold1.csv']

    titles = ['ICON Load1 - 3 Machines 10 Jobs']

    number_of_files = 1
    colors = ['y', 'g', 'c', 'm']
    capacities = ['1', '8', '12']
    models = ['Divide and Conquer']

    number_of_files = len(files_predict)

    run_times = []
    sub_epochs = []
    test_regrets = []
    val_regrets = []
    for index, file in enumerate(files_predict):
        df = read_file(filename=file, folder_path=file_folder_predict, delimiter=' ')
        df = np.array(df[3:]).astype(float)
        run_times.append(df[:, 2])
        sub_epochs.append(3 * df[:, 1] / np.max(df[:, 1]))
        test_regrets.append(df[:, 4].astype(float))
        val_regrets.append(df[:, 5].astype(float))
        # print(df[0])

    iter_no = 10
    iter_range = range(iter_no)
    number_of_rows = 66
    regrets_spo = []
    sub_epochs_spo = np.zeros((number_of_rows - 1, number_of_files))

    val_spo = []

    for index, file in enumerate(files_spo):
        regrets_spo_np = np.zeros((number_of_rows - 1, iter_no))
        val_regrets_spo_np = np.zeros((number_of_rows - 1, iter_no))
        for i in iter_range:
            df = np.array(read_file(filename=file, folder_path=file_folder_spo, delimiter=','))

            regrets_spo_np[:, i] = df[1 + (i * number_of_rows):(i + 1) * number_of_rows, 3].astype(float)

            sub_epochs_spo[:, index] = df[1 + (i * number_of_rows):(i + 1) * number_of_rows, 4]

            val_regrets_spo_np[:, i] = df[1 + (i * number_of_rows):(i + 1) * number_of_rows, 1].astype(float)
        regrets_spo.append(regrets_spo_np)
        val_spo.append(val_regrets_spo_np)

    fig = plt.figure()
    ax2 = fig.add_subplot(1, 1, 1)
    ax2.set_title('Load12 3 Machines 10 Jobs Fold 1')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('Regret')

    ax2.hlines(16585.576724999584, 0, 6)
    # print predict opt
    min_len = min([len(arr) for arr in test_regrets])
    test_regrets = [n_iter_regrets[:min_len] for n_iter_regrets in test_regrets]
    test_regrets = np.array(test_regrets)
    sub_epochs = sub_epochs[0]
    sub_epochs = sub_epochs[:min_len]
    x_axis = sub_epochs

    # print(test_regrets)
    y_axis = test_regrets
    mean_regret = np.median(y_axis, axis=0)
    plt.plot(x_axis, mean_regret)
    std = np.std(y_axis, axis=0)
    plt.fill_between(x_axis, mean_regret - std, mean_regret + std,
                     alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')

    val_regrets = [n_iter_regrets[:min_len] for n_iter_regrets in val_regrets]
    val_regrets = np.array(val_regrets)

    # tmp_val_regrets = val_regrets
    # min_val_ind = np.argmin(tmp_val_regrets, axis=1)
    # min_val_test_regrets = y_axis[np.arange(len(y_axis)), min_val_ind]
    # min_val_test_regret = np.amin(min_val_test_regrets)

    tmp_val_regrets = val_regrets
    min_val_row, min_val_col = np.unravel_index(tmp_val_regrets.argmin(), tmp_val_regrets.shape)
    min_val_test_regret = y_axis[min_val_row, min_val_col]
    plt.scatter(x_axis[min_val_col], min_val_test_regret, color='b', ls='--')
    us_regret = min_val_test_regret
    print('us', min_val_test_regret)
    for index, file in enumerate(files_spo):
        y_axis = regrets_spo[index]
        x_axis = 6 * sub_epochs_spo[:, index] / np.max(sub_epochs_spo[:, index])

        # tmp_val_regrets = val_spo[index]
        # min_val_ind = np.argmin(tmp_val_regrets, axis=0)
        # min_val_test_regrets = y_axis[min_val_ind, np.arange(iter_no)]
        # min_val_test_regret = np.amin(min_val_test_regrets)

        tmp_val_regrets = val_spo[index]
        min_val_row, min_val_col = np.unravel_index(tmp_val_regrets.argmin(), tmp_val_regrets.shape)
        min_val_test_regret = y_axis[min_val_row, min_val_col]
        plt.scatter(x_axis[min_val_col], min_val_test_regret, color='r', ls='--')

        spo_regret = min_val_test_regret
        # mean spo relax
        plt.plot(x_axis, np.median(y_axis, axis=1), color='r', ls='--')
        mean_regret = np.median(y_axis, axis=1)

        std = np.std(y_axis, axis=1)

        plt.fill_between(x_axis, mean_regret - std, mean_regret + std,
                         alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        print('spo', min_val_test_regret)
    labels = ['Us', 'SPO-relax', 'baseline']
    plt.legend(labels=labels)
    plt.plot()
    plt.savefig('figs/icon_l12_k1.pdf')
    plt.show()
    return us_regret, spo_regret


def find_indexes(f_list):
    start_indexes = []
    end_indexes = []
    end_index = 0
    start_index = 0
    for index, file in enumerate(f_list):
        if not is_number(file[0]):
            if end_index > start_index:
                end_indexes.append(end_index)
            start_index = index + 1
        else:
            if float(file[0]) < 1000:
                if start_index > end_index:
                    start_indexes.append(start_index)
                end_index = index
    end_indexes.append(end_index)
    return start_indexes, end_indexes


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def mean_std_template(file_folder_predict, file_folder_spo, file_folder_qptl, files_predict, files_spo, files_qptl,
                      dest_file_name, title,
                      predict_model_name='us', is_show=False, is_save=True, number_of_rows_spo=84,
                      number_of_rows_qptl=73,
                      iter_no_pdo=2, is_run_time = False):
    number_of_files = len(files_predict)

    header_length_pdo = 3
    run_times = []
    sub_epochs = []
    test_regrets = []
    val_regrets = []
    for index, file in enumerate(files_predict):
        df_list = read_file(filename=file, folder_path=file_folder_predict, delimiter=' ')
        start_indexes, end_indexes = find_indexes(df_list)
        number_of_rows_pdo = int((len(df_list) - 2 * header_length_pdo) / 2) - 1
        iter_no_pdo = int(len(start_indexes))
        for i in range(iter_no_pdo):
            start_index = start_indexes[i]
            end_index = end_indexes[i]
            df = np.array(df_list[start_index:end_index]).astype(float)
            run_times.append(df[:, 2])
            sub_epochs.append((np.max(df[:, 0]) + 1) * df[:, 1] / np.max(df[:, 1]))
            test_regrets.append(df[:, 4])
            val_regrets.append(df[:, 5])

    run_times_spo = []
    regrets_spo = []
    sub_epochs_spo = []
    val_spo = []

    for index, file in enumerate(files_spo):

        df = np.array(read_file(filename=file, folder_path=file_folder_spo, delimiter=','))
        iter_no = int((len(df) - 1) / number_of_rows_spo)
        regrets_spo_np = np.zeros((number_of_rows_spo - 1, iter_no))
        val_regrets_spo_np = np.zeros((number_of_rows_spo - 1, iter_no))
        sub_epoch_spo = np.zeros((number_of_rows_spo - 1, iter_no))
        run_time_spo = np.zeros((number_of_rows_spo - 1, iter_no))
        iter_range = range(int((len(df) - 1) / number_of_rows_spo))
        for i in iter_range:
            regrets_spo_np[:, i] = df[1 + (i * number_of_rows_spo):(i + 1) * number_of_rows_spo, 3].astype(float)

            sub_epoch_spo[:, i] = df[1 + (i * number_of_rows_spo):(i + 1) * number_of_rows_spo, 4]

            val_regrets_spo_np[:, i] = df[1 + (i * number_of_rows_spo):(i + 1) * number_of_rows_spo, 1].astype(float)

            run_time_spo [:, i] = df[1 + (i * number_of_rows_spo):(i + 1) * number_of_rows_spo, 5].astype(float)
        run_times_spo.append(run_time_spo)
        regrets_spo.append(regrets_spo_np)
        val_spo.append(val_regrets_spo_np)
        sub_epochs_spo.append(sub_epoch_spo)

    run_times_qptl = []
    regrets_qptl = []
    sub_epochs_qptl = []
    val_qptl = []

    for index, file in enumerate(files_qptl):

        df = np.array(read_file(filename=file, folder_path=file_folder_qptl, delimiter=','))
        iter_no = int((len(df) - 1) / number_of_rows_qptl)
        regrets_qptl_np = np.zeros((number_of_rows_qptl - 1, iter_no))
        val_regrets_qptl_np = np.zeros((number_of_rows_qptl - 1, iter_no))
        sub_epoch_qptl = np.zeros((number_of_rows_qptl - 1, iter_no))
        run_time_qptl = np.zeros((number_of_rows_qptl - 1, iter_no))
        iter_range = range(int((len(df) - 1) / number_of_rows_qptl))
        for i in iter_range:
            regrets_qptl_np[:, i] = df[1 + (i * number_of_rows_qptl):(i + 1) * number_of_rows_qptl, 3].astype(float)

            sub_epoch_qptl[:, i] = df[1 + (i * number_of_rows_qptl):(i + 1) * number_of_rows_qptl, 6]

            val_regrets_qptl_np[:, i] = df[1 + (i * number_of_rows_qptl):(i + 1) * number_of_rows_qptl, 0].astype(float)

            run_time_qptl[:, i] = df[1 + (i * number_of_rows_qptl):(i + 1) * number_of_rows_qptl, 7].astype(float)

        run_times_qptl.append(run_time_qptl)
        regrets_qptl.append(regrets_qptl_np)
        val_qptl.append(val_regrets_qptl_np)
        sub_epochs_qptl.append(sub_epoch_qptl)

    if is_run_time:
        test_regrets = run_times
        regrets_spo = run_times_spo
        regrets_qptl = run_times_qptl

    baseline_regret = test_regrets[0][0]

    fig = plt.figure()
    ax2 = fig.add_subplot(1, 1, 1)
    ax2.set_title(title)
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('Regret')

    ax2.hlines(baseline_regret, 0, 6)
    # print predict opt
    min_len = min([len(arr) for arr in test_regrets])
    test_regrets = [n_iter_regrets[:min_len] for n_iter_regrets in test_regrets]
    test_regrets = np.array(test_regrets)
    sub_epochs = sub_epochs[0]
    sub_epochs = sub_epochs[:min_len]
    x_axis = sub_epochs

    y_axis = test_regrets
    mean_regret = np.median(y_axis, axis=0)
    plt.plot(x_axis, mean_regret)
    std = np.std(y_axis, axis=0)
    plt.fill_between(x_axis, mean_regret - std, mean_regret + std,
                     alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')

    val_regrets = [n_iter_regrets[:min_len] for n_iter_regrets in val_regrets]
    val_regrets = np.array(val_regrets)

    # tmp_val_regrets = val_regrets
    # min_val_ind = np.argmin(tmp_val_regrets, axis=1)
    # min_val_test_regrets = y_axis[np.arange(len(y_axis)), min_val_ind]
    # min_val_test_regret = np.amin(min_val_test_regrets)
    # plt.scatter(x_axis[min_val_ind[np.argmin(min_val_test_regrets)]], min_val_test_regret, color='b', ls='--')

    tmp_val_regrets = val_regrets
    min_val_row, min_val_col = np.unravel_index(tmp_val_regrets.argmin(), tmp_val_regrets.shape)
    min_val_test_regret = y_axis[min_val_row, min_val_col]
    # plt.scatter(x_axis[min_val_col], min_val_test_regret, color='b', ls='--')
    us_regret = min_val_test_regret
    print('val regret: {}: {}'.format(predict_model_name, tmp_val_regrets.min()))
    for index, file in enumerate(files_spo):
        sub_epoch_spo = sub_epochs_spo[0]
        y_axis = regrets_spo[index]

        x_axis = sub_epoch_spo[:, index] / 550

        # tmp_val_regrets = val_spo[index]
        # min_val_ind = np.argmin(tmp_val_regrets, axis=0)
        # min_val_test_regrets = y_axis[min_val_ind, np.arange(iter_no)]
        # min_val_test_regret = np.amin(min_val_test_regrets)
        # plt.scatter(x_axis[min_val_ind[np.argmin(min_val_test_regrets)]], min_val_test_regret, color='r', ls='--')

        tmp_val_regrets = val_spo[index]
        min_val_row, min_val_col = np.unravel_index(tmp_val_regrets.argmin(), tmp_val_regrets.shape)
        min_val_test_regret = y_axis[min_val_row, min_val_col]
        plt.scatter(x_axis[min_val_row], min_val_test_regret, color='r', ls='--')
        spo_regret = min_val_test_regret
        # mean spo relax
        plt.plot(x_axis, np.median(y_axis, axis=1), color='r', ls='--')
        mean_regret = np.median(y_axis, axis=1)

        std = np.std(y_axis, axis=1)

        plt.fill_between(x_axis, mean_regret - std, mean_regret + std,
                         alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        # plt.scatter(x_axis[])
        print('spo', tmp_val_regrets.min())

    for index, file in enumerate(files_qptl):
        sub_epoch_qptl = sub_epochs_qptl[0]
        y_axis = regrets_qptl[index]
        print(file)
        x_axis = sub_epoch_qptl[:, index] / 550

        # tmp_val_regrets = val_spo[index]
        # min_val_ind = np.argmin(tmp_val_regrets, axis=0)
        # min_val_test_regrets = y_axis[min_val_ind, np.arange(iter_no)]
        # min_val_test_regret = np.amin(min_val_test_regrets)
        # plt.scatter(x_axis[min_val_ind[np.argmin(min_val_test_regrets)]], min_val_test_regret, color='r', ls='--')

        tmp_val_regrets = val_qptl[index]
        min_val_row, min_val_col = np.unravel_index(tmp_val_regrets.argmin(), tmp_val_regrets.shape)
        min_val_test_regret = y_axis[min_val_row, min_val_col]
        plt.scatter(x_axis[min_val_row], min_val_test_regret, color='r', ls='--')
        qptl_regret = min_val_test_regret
        # mean spo relax
        plt.plot(x_axis, np.median(y_axis, axis=1), color='m', ls='--')
        mean_regret = np.median(y_axis, axis=1)

        std = np.std(y_axis, axis=1)

        plt.fill_between(x_axis, mean_regret - std, mean_regret + std,
                         alpha=0.5, edgecolor='#999999', facecolor='#000000')
        # plt.scatter(x_axis[])
        print('qptl', tmp_val_regrets.min())

    labels = [predict_model_name, 'SPO-relax', 'QPTL', 'baseline']
    plt.legend(labels=labels)
    plt.plot()

    if is_save:
        plt.savefig('figs/' + str(dest_file_name))
    if is_show:
        plt.show()
    regrets = [baseline_regret, us_regret, spo_regret, qptl_regret]
    return regrets


if __name__ == '__main__':
    # plot_knapsack_weighted()
    plot_run_time()
# test_knapsack_unit()
# plot_ICON()
# plot_spo_relax_og()

# plot_minibatch_coordinate()
# plot_minibatch_weighted_knapsack()
# plot_minibatch_icon_mean()
