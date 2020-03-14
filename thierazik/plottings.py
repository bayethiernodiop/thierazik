import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Regression chart, we will see more of this chart in the next class.
def chart_regression(pred, y):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    t.sort_values(by=['y'], inplace=True)
    _ = plt.plot(t['y'].tolist(), label='expected')
    _ = plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()

# Radar graphic
def kiviat_diagram(cluster_index,data_kiviat, features_names, clusters_labels):
    """
    Example of Usage :
    scaled_features.iloc[:,:] = MinMaxScaler(feature_range=(0, 100)).fit_transform(scaled_features)
    features_names = scaled_features.columns

    # Initializing a pd.DataFrame to store aggregate values
    clusters_agg_df = pd.DataFrame(
        columns=[feature_name for feature_name in features_names]
    )
    list_clusters = [0,1]
    # Iterating upon clusters
    for cluster_index in list_clusters:

        # Getting elements of the cluster
        mask = Xtrain['Y'] == cluster_index

        # For each feature. Could be vectorized.
        for feature in scaled_features.columns:
            # compute the mean
            mean_of_feature_for_cluster = scaled_features[mask][feature].mean()
            clusters_agg_df.loc[cluster_index, feature] = mean_of_feature_for_cluster

    # Compute the data for the 'mean customer'
    for feature in scaled_features.columns:
        # compute the mean
        mean_of_feature = scaled_features[feature].mean()
        clusters_agg_df.loc['mean', feature] = mean_of_feature
        
        median_of_feature = scaled_features[feature].median()
        clusters_agg_df.loc['median', feature] = median_of_feature
    kiviat_diagram(0,clusters_agg_df,list(scaled_features.columns),list_clusters)
    """
    # Get number of variables
    num_vars = len(features_names)

    # Split the circle into even parts and save the angles
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    angles += angles[:1]

    _, ax = plt.subplots( subplot_kw=dict(polar=True))
    

    # Set color
    # ----------
    # cmap = plt.cm.Dark2  # ListedColormap
    cmap = plt.cm.get_cmap('hsv')  # LinearSegmentedColormap
    NB_CLUSTERS = len(clusters_labels) - (1 if -1 in clusters_labels else 0)
    color = cmap(cluster_index/NB_CLUSTERS)

    # Add cluster to the chart
    # -------------------------
    # Get values and complete the circle
    values = data_kiviat[features_names].loc[cluster_index].tolist()
    values += values[:1]
    ax.plot(
        angles,
        values,
        color=color,
        linewidth=1,
        label="Cluster #"+str(cluster_index)
    )
    ax.fill(angles, values, color=color, alpha=0.25)


    # Add 'mean ' to the chart
    # Get values and complete the circle
    values = data_kiviat[features_names].loc['mean'].tolist()
    values += values[:1]
    ax.plot(angles,
            values,
            color='black',
            linewidth=0.6,
            label='mean')

    # Fix axis to go in the right order and start at 12 o'clock.
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label.
    ax.set_thetagrids(np.degrees(angles), features_names)

    # Go through labels and adjust alignment based on position
    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

    # Ensure radar goes from 0 to 100
    ax.set_ylim(0, 100)

    # Set position of y-labels (0-100)
    ax.set_rlabel_position(180 / num_vars)

    # Add some custom styling.
    # Change the color of the tick labels.
    ax.tick_params(colors='#222222')
    # Make the y-axis (0-100) labels smaller.
    ax.tick_params(axis='y', labelsize=8)
    # Change the color of the circular gridlines.
    ax.grid(color='#AAAAAA')
    # Change the color of the outermost gridline (the spine).
    ax.spines['polar'].set_color('#222222')
    # Change the background color inside the circle itself.
    ax.set_facecolor('#FAFAFA')

    # Add title
    ax.set_title('Groupe #{} '.format(cluster_index), y=1.08)

    # Add a legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.36, 1.15))