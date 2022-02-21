from Tests.plot_templates import weighted_knapsack_template

is_run_time=True
is_normalize = False
if __name__ == '__main__':
    file_folder = 'Tests/Knapsack/weighted/'
    plot_title = 'Weighted Knapsack Run Time Comparison'
    dest_file_name = 'strong_corr_w_knapsackrun_qptlthin.pdf'
    capacities = [12, 24, 48, 72, 96, 120, 144, 172, 196, 220]
    # capacities =[12, 24,96,172,196,220]
    weighted_knapsack_template(file_folder=file_folder, models=[1, 0, 1, 1, 1, 1, 1, 1], capacities=capacities, is_plot=False,
                               plot_title=plot_title, is_save=True, dest_file_name=dest_file_name,is_run_time=is_run_time, is_normalize=is_normalize)

    file_folder = 'Tests/Knapsack/unit/'
    plot_title = 'Unit Knapsack Run Time Comparison'
    dest_file_name = 'unweighted_knapsackrun_qptlthin.pdf'
    capacities = [5, 10, 15, 20, 25, 30, 35, 40]
    weighted_knapsack_template(file_folder=file_folder, capacities=capacities, models=[1, 0, 1, 1, 1, 1, 1, 1], w_tag_str="",
                               is_plot=False, plot_title=plot_title, is_save=True, dest_file_name=dest_file_name, isUnit=True,is_run_time = is_run_time, is_normalize=is_normalize, unit_tag_intopt='_unit')

    file_folder = 'Tests/Knapsack/weighted/'
    plot_title = 'Knapsack-Weighted'
    dest_file_name = 'strong_corr_w_knapsack_qptlthin.pdf'

    capacities = [12, 24, 48, 72, 96, 120, 144, 172, 196, 220]
    # capacities =[12, 24,96,196,220]
    weighted_knapsack_template(file_folder=file_folder, models=[1, 0, 1, 1, 1, 1, 1, 1], capacities=capacities, is_plot=False,
                               plot_title=plot_title, is_save=True, dest_file_name=dest_file_name,ylim=2,is_run_time = False)

    file_folder = 'Tests/Knapsack/unit/'
    plot_title = 'Knapsack-Unit Weights'
    dest_file_name = 'unweighted_knapsack_qptlthin.pdf'
    capacities = [5, 10, 15, 20, 25, 30, 35, 40]
    # capacities = [5, 20, 25, 35, 40]
    weighted_knapsack_template(file_folder=file_folder, capacities=capacities, models=[1, 0, 1, 1, 1, 1, 1, 1], w_tag_str="",
                               is_plot=False, plot_title=plot_title, is_save=True, dest_file_name=dest_file_name,ylim=2, isUnit=True, is_run_time=False, unit_tag_intopt='_unit')
