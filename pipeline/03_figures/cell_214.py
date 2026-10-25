fig = plot_grouped_boxplot_combined(
    final_df,
    isa_col='far_isa',
    building_col='far_building',
    error_col='abs_error_node',
    min_samples=500,
    interval = 2,
    # y_max=40,
    save=True,
    show=True
)