from Tests.plot_templates import weighted_knapsack_template, find_regrets_knapsack_dp, get_min_val_test_regret, \
    read_knapsack_files_predict, read_knapsack_files_dp
import numpy as np
import matplotlib.pyplot as plt

facecolors = ['#a6cee3',
              '#1f78b4',
              '#b2df8a',
              '#33a02c',
              '#a6cee3',
              '#1f78b4',
              '#b2df8a',
              '#33a02c'
              ]


def find_runtimes_knapsack_single(file_dp):
    # number_of_rows_spo

    run_times, sub_epochs, test_regrets, val_regrets = read_knapsack_files_predict(file_dp)
    one_epoch_run_time = np.mean([run_time[-1] / len(run_time) for run_time in run_times]) * 18

    return one_epoch_run_time


def find_runtimes_knapsack_single_dp(file_dp):
    run_times_dp, sub_epochs_dp, test_regrets_dp, val_regrets_dp = read_knapsack_files_dp(file_dp)

    one_epoch_run_time = np.mean(np.array(
        [run_time_dp[-1] / len(run_time_dp) * 9 if len(run_time_dp) > 9 else run_time_dp[-1] for i, run_time_dp in
         enumerate(run_times_dp)]))

    return one_epoch_run_time


def bar_plot_runtime(file_folder, capacities, is_save, is_show, dest_file_name, plot_title=""):
    total_folds = 5
    files_dp_folders = []
    files_dnl_folders = []

    edgecolors = ['#089FFF', '#2ba14d', '#eb6c65', '#bd78de']

    for c in capacities:
        file_names = []
        for kfold in range(total_folds):
            file_name = file_folder + "c" + str(c) + "/spartan/knapsack-wsparc" + str(c) + '-k' + str(
                kfold) + "-DIVIDE_AND_CONQUER_GREEDY-0-1.csv"
            file_names.append(file_name)
        files_dnl_folders.append(file_names)

    for c in capacities:
        file_names = []
        for kfold in range(total_folds):
            file_name = file_folder + "c" + str(c) + "/dp/knapsack-w" + "c" + str(c) + '-k' + str(
                kfold) + "-DP.csv"
            file_names.append(file_name)
        files_dp_folders.append(file_names)

    dnl_run_times = np.zeros((len(capacities), total_folds))
    dp_run_times = np.zeros((len(capacities), total_folds))
    for index, c in enumerate(capacities):
        for kfold in range(total_folds):
            file_dnl = files_dnl_folders[index]
            file_dp = files_dp_folders[index]

            dnl_run_time = find_runtimes_knapsack_single(file_dnl[kfold])

            dp_run_time = find_runtimes_knapsack_single_dp(file_dp[kfold])
            dp_run_times[index, kfold] = dp_run_time
            dnl_run_times[index, kfold] = dnl_run_time

    fig = plt.figure()
    ax2 = fig.add_subplot(1, 1, 1)

    xtick_labels = capacities
    ind = np.arange(0, len(xtick_labels)) + 1
    xtickslocs = ind
    ax2.set_xticks(ind)
    ax2.set_xticklabels(xtick_labels)
    ax2.set_title(plot_title)
    ax2.set_xlabel('Capacities')
    ax2.set_ylabel('Run Time Per Epoch in Log Scale(s)')

    regrets = np.array([np.mean(dnl_run_times, axis=1),
                        np.mean(dp_run_times, axis=1)])
    errors = np.array([np.std(dnl_run_times, axis=1), np.std(dp_run_times, axis=1)])
    for index in range(2):
        ind_plot = index
        ax2.bar(ind + ((+0.15 * ind_plot) - 0.22), regrets[index, :], yerr=errors[index, :], color=facecolors[index],
                width=0.15)

    # ax2.xaxis.set_major_locator(mticker.FixedLocator(xtickslocs))
    labels = ['Dnl-Greedy', "DP"]
    ax2.legend(labels)
    ax2.set_yscale('log')
    if is_save:
        plt.savefig('figs/' + str(dest_file_name))
    if is_show:
        plt.show()


def bar_plot_runtime_dnl(file_folder, capacities, is_save, is_show, dest_file_name, file_prefix="", weight_fix="",
                         plot_title=""):
    total_folds = 1
    files_dnl_folders = [[] for i in range(3)]
    files_dnlexh_folders = [[] for i in range(3)]
    files_dnlgreedy_folders = [[] for i in range(3)]
    edgecolors = ['#089FFF', '#2ba14d', '#eb6c65', '#bd78de']
    for i in range(3):
        for c in capacities:
            file_names = []
            if not (c > 120 and i > 1):
                for kfold in range(total_folds):
                    file_name = file_folder + file_prefix + "knap" + weight_fix + "-c" + str(c) + '-k' + str(
                        kfold) + "-DIVIDE_AND_CONQUER-" + str(i) + "-1.csv"
                    file_names.append(file_name)
                files_dnl_folders[i].extend(file_names)

        for c in capacities:
            file_names = []
            if not (c > 120 and i > 1):
                for kfold in range(total_folds):
                    file_name = file_folder + file_prefix + "knap" + weight_fix + "-c" + str(c) + '-k' + str(
                        kfold) + "-DIVIDE_AND_CONQUER_GREEDY-" + str(i) + "-1.csv"
                    file_names.append(file_name)
                files_dnlgreedy_folders[i].extend(file_names)
        for c in capacities:
            file_names = []
            if not (c > 12 and i > 1):
                for kfold in range(total_folds):
                    file_name = file_folder + file_prefix + "knap" + weight_fix + "-c" + str(c) + '-k' + str(
                        kfold) + "-EXHAUSTIVE-" + str(i) + "-1.csv"
                    file_names.append(file_name)
                files_dnlexh_folders[i].extend(file_names)

    dnl_run_times = np.zeros((len(capacities), 3))
    dnlgreedy_run_times = np.zeros((len(capacities), 3))
    exh_run_times = np.zeros((len(capacities), 3))

    for index, c in enumerate(capacities):
        for ub in range(3):

            if not (c > 120 and ub > 1):
                file_dnl = files_dnl_folders[ub]
                dnl_run_time = find_runtimes_knapsack_single(file_dnl[index])
                dnl_run_times[index, ub] = dnl_run_time
            if not (c > 120 and ub > 1):
                file_dnlgreedy = files_dnlgreedy_folders[ub]
                dnlgreedy_run_time = find_runtimes_knapsack_single(file_dnlgreedy[index])
                dnlgreedy_run_times[index, ub] = dnlgreedy_run_time


            if not (c > 12 and ub > 1):
                file_exh = files_dnlexh_folders[ub]
                exh_run_time = find_runtimes_knapsack_single(file_exh[index])
                exh_run_times[index, ub] = exh_run_time

        fig = plt.figure()
        ax2 = fig.add_subplot(1, 1, 1)

        xtick_labels = [30, 300, 3000]
        ind = np.arange(0, len(xtick_labels)) + 1
        xtickslocs = ind
        ax2.set_xticks(ind)
        ax2.set_xticklabels(xtick_labels)
        ax2.set_title(plot_title + " (C:{})".format(c))
        ax2.set_xlabel('Sample Points')
        ax2.set_ylabel('Run Time Per Epoch in Log Scale (s)')

        regrets = np.array([
            dnlgreedy_run_times[index], dnl_run_times[index], exh_run_times[index]])
        for index in range(3):
            ind_plot = index
            ax2.bar(ind + ((+0.15 * ind_plot) - 0.22), regrets[index, :], color=facecolors[index],
                    width=0.15)

        # ax2.xaxis.set_major_locator(mticker.FixedLocator(xtickslocs))
        labels = ['Dnl-Greedy', "Dnl", "Exhaustive"]
        ax2.legend(labels)
        ax2.set_yscale('log')

        if is_save:
            plt.savefig('figs/' + "c" + str(c) + str(dest_file_name))
        if is_show:
            plt.show()


def dp_and_dnl():
    file_folder = 'Tests/Knapsack/weighted/'
    plot_title = 'Weighted Knapsack Runtime Comparison(DP vs Dnl)'
    dest_file_name = 'weighted_dp_runtime.pdf'
    capacities = [12, 24, 48, 72, 96, 120, 144, 172, 196, 220]
    bar_plot_runtime(file_folder=file_folder, capacities=capacities, is_show=True, is_save=True,
                     dest_file_name=dest_file_name, plot_title=plot_title)


def dnl_and_exhaustive():
    file_folder = 'Tests/knapsack_runtime/'
    plot_title = 'Knapsack-Weighted'
    dest_file_name = 'weighted_dnlexh_runtime.pdf'
    capacities = [12, 120, 196]
    bar_plot_runtime_dnl(file_folder=file_folder, capacities=capacities, is_show=True, is_save=True,
                         dest_file_name=dest_file_name, file_prefix="N0", weight_fix='w', plot_title=plot_title)


if __name__ == '__main__':
    dnl_and_exhaustive()
    dp_and_dnl()
