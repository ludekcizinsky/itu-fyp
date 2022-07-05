# Standard packages from 1st semester
import numpy as np
from matplotlib import pyplot as plt

# Pandas - mainly used to lookup categorical variable names
import pandas as pd

# Needed for Task 2
from scipy.stats import chi2_contingency

# Needed for the Task 3
import folium
from branca.element import Template, MacroElement

# Needed only if using Deepnote
from IPython.core.display import display, HTML

# Needed for Task 4
from geopy.distance import great_circle
from sklearn.cluster import DBSCAN as dbscan

# =================== GENERAL FUNCTIONS =======================


def get_proper_axs(n_subplots, n_rows=None):
    """
    Returns a figure and axs based on number of subplots.
    :n_subplots: int
    :return: fig, axs
    """

    # Set the number of rows
    if not n_rows:
        n_rows = int(n_subplots / 3)
    if n_rows < 1:
        n_rows = 1

    fig, axs = plt.subplots(n_rows, 3)
    if type(axs[0]) != np.ndarray:
        axs = [axs]

    return fig, axs


def get_relevant_lookup_values(df, categories):
    """
    Takes lookup pandas dataframe and based on category codes returns corresponding
    category names. NOTE: This function is specific to a structure of excel sheet
    located in references/variable lookup
    :df: pandas dataframe
    :categories: 1D numpy array of items
    :return: Python list with relevant code names
    """

    relevant_categories_names = []
    for category in categories:
        row = df.loc[df.iloc[:, 0] == category]
        val = row.iloc[:, 1].to_numpy()
        relevant_categories_names.append(val[0])
    return relevant_categories_names


def map_num_cat_to_named_cat(var_lookup, column, column_name):
    """
    Maps raw data from columns to its corresponding categories specified in provided lookup table.
    :var_lookup: pandas dataframe --> project specific --> expects to get excel sheet from references/var lookup.xls
    :column: 1D numpy array --> must correspond to column name
    :column_name: Name of the column whose values are supposed to be replaced with lookup values
    :return: 1D numpy array
    """

    # Adjust column name if neccessary
    if "_" in column_name:
        column_name = " ".join(column_name.split("_"))

    # Replace the number labels with names from lookup table
    # Get corresponding sheet as pandas data frame
    df = var_lookup.get(column_name)

    # Create a dictionary which maps code from dataset to the corresponding value in lookup table
    num_categories = np.unique(column)
    column_dic = {num_category: named_category for num_category, named_category in zip(
        num_categories, get_relevant_lookup_values(df, num_categories))}

    # Replace the num categories with named categories using the lookup dictionary
    column_named_cats = np.array([column_dic[num_cat] for num_cat in column])

    return column_named_cats


# =================== LOAD FUNCTIONS =======================


def load_data_numpy_csv(filepath, encoding='utf-8-sig', names=True, usemask=True, skip_header=0, usecols=None, max_rows=None, dtype=None):
    """
    Loads in CSV file using numpy
    :filename: full path to the given file, string
    :return: structured masked numpy array
    """

    data = np.genfromtxt(filepath,
                         delimiter=',',
                         dtype=dtype,
                         names=names,
                         encoding=encoding,
                         usemask=usemask,
                         skip_header=skip_header,
                         usecols=usecols,
                         max_rows=max_rows,
                         missing_values={'-1', ''}
                         )

    return data


def load_pandas_df_from_excel(filepath):
    """
    Loads data from excel sheet to pandas dataframe.
    :filepath: path to the excel sheet relative to the current file location
    :return: pandas dataframe
    """

    df = pd.read_excel(filepath, sheet_name=None)

    return df


# =================== CLEANING DATA FUNCTIONS =======================


def get_city_specific_records_by_lad(data, authority_code):
    """
    Returns records from the given dataset that are relevant to specified city.
    :data: structured numpy array which has to include column Local_Authority_District
    :authority_code: code from the lookup table for the given aurhority
    :return: structured numpy array with records relevant only to given city
    """

    # Mask the values for the given city based on authority code
    city_only_mask_district = data['Local_Authority_District'] == authority_code

    # Use the mask to select the records
    city_only_records = data[city_only_mask_district]

    return city_only_records


def is_accident_id_existent(accident_table, other_table):
    """
    Checks that column Accident_id (its values) in other table have corresponding IDs
    in accident table.
    :other_table: structured 1D array
    :return: 1d array with ids not present in accident table
    """

    # Get unique accident ids from other table
    unique_accident_ids = np.unique(other_table['Accident_Index'])

    # Check whether there are any different values in other table compare to accident
    # if that is the case, it should be returned in difference, otherwise empty set will be returned
    A = set(unique_accident_ids)
    B = set(accident_table['Accident_Index'])
    difference = A.difference(B)

    return np.array(difference)


def select_only_relevant_columns(data, selected_columns):
    """
    Selects given columns from the provided dataset.
    :data: structured numpy array
    :selected_columns: List object with names of the columns that need to be chosen from the given dataset.
    :return: Masked numpy array
    """

    return data[selected_columns]


def strip_number_from_zero(number):
    """
    Modifies numbers which are submitted as strings and have
    the following form: 00, 01, .... --> gets rid of the leading 0.
    :number: str
    :return: Modified str
    """

    first, second = list(number)
    if first == '0':
        return second
    else:
        return (first + second)


def get_accident_data_from_given_year(filepath):

    # Load the data to structured numpy array
    arr = load_data_numpy_csv(filepath, encoding='utf-8-sig', names=True,
                              usemask=True, skip_header=0, usecols=None, max_rows=None, dtype=None)

    # Get accidents relevant to Manchester
    accidents_in_manchester = get_city_specific_records_by_lad(arr, 102)

    return accidents_in_manchester

# =================== EXPLROING DATA FUNCTIONS =======================


def get_median(data):
    """
    :data: structured numpy array
    :return: median value, float
    """
    try:

        # Sort the array first
        data = np.sort(data)

        # Get len of array
        arr_len = int(data.size)

        # Even number of records
        if arr_len % 2 == 0:
            median = (data[arr_len//2] + data[arr_len//2-1])/2

        # Odd number of records
        else:
            median = data[arr_len//2]

        return median
    except Exception:
        return None


def get_quartiles(data):
    """
    :data: structured numpy array
    :return: first and second quartile values, float
    """
    try:

        # Sort the array first
        data = np.sort(data)

        # Get len of array
        arr_len = int(data.size)

        # Odd number of records - adjust the list with data
        if arr_len % 2 != 0:
            np.delete(data, arr_len//2)

        # Compute the quartiles
        q1 = get_median(data[:arr_len//2])
        q2 = get_median(data[arr_len//2:])

        return q1, q2

    except Exception:
        return None, None


def get_info_datasets(RAW_DATA):
    """
    :RAW_DATA: dictionary where value is 2D numpy arrays (datasets)
    :return: dictionary with the following format: {
        "key": {
            "row numbers" : int
            "column numbers": int
            "column names": tuple
        }
        ....
    }
    """
    assert RAW_DATA, "You need to specify filepaths"
    assert type(RAW_DATA) == dict, "RAW_DATA must be submitted as a dictionary"

    summary_dic = {
    }

    for key, arr in RAW_DATA.items():

        # Get metrics
        n_rows = arr.shape[0]
        column_names = RAW_DATA[key].dtype.names
        n_cols = len(column_names)

        # Save it to the summary dict
        summary_dic[key] = {
            "row numbers": n_rows,
            "column numbers": n_cols,
            "column names": column_names
        }

    return summary_dic


def get_5_num_sum(data, columns=None):
    """
    Computes five number summary. It uses the definition of all metrics from
    the book we used used in the course "Introduction to Data Science and Programming".
    :data: structured Numpy array representing given table
    :columns: Python list with column names
    :return: min, Q1, median, Q3, max (all floats)
    """

    # If column indexes are not specified, it will be assumed that all columns should be considered
    if not columns:
        columns = list(data.dtype.names)

    # Calculate the metrics for each column and save it to a dictionary which will then be returned as a result
    # NOTE: In case column is not numeric, the column is skipped
    summary_dict = dict()
    for column_name in columns:

        # Get 1D numpy array (column)
        column = data[column_name]

        # Filter out missing values
        column = column[~column.mask]

        # Check that input is numerical
        dtype_str = str(column.dtype)
        is_numerical = 'int' in str(dtype_str) or 'float' in dtype_str

        if is_numerical:
            # calculate median
            median = get_median(column)

            # Calculate Q1 and Q2
            q1, q2 = get_quartiles(column)

            # calculate min/max
            try:
                data_min, data_max = min(column), max(column)
            except Exception:
                data_min, data_max = None, None

            # Save the results into a dictionary
            summary_dict[column_name] = {
                "min": data_min,
                "q1": q1,
                "median": median,
                "q2": q2,
                "max": data_max
            }

    return summary_dict


def boxplots(data, selected_column_names=None, n_rows=None):
    '''
    Shows boxplots for numerical data. (If not numeric - only sets title with colum name + NOT NUMERICAL
    :data: structured numpy array
    :selected_column_names: list with selected column names, if None all columns from data are considered
    :return: None
    '''

    if not selected_column_names:
        selected_column_names = data.dtype.names

    num_boxplots = len(selected_column_names)
    fig, axs = get_proper_axs(num_boxplots, n_rows)

    fig.set_figheight(5)
    fig.set_figwidth(10)

    index = 0

    for row in axs:
        for ax in row:
            if index < num_boxplots:
                # Get column
                column = data[selected_column_names[index]]

                # Filter out missing values
                column = column[~column.mask]

                # Check that input is numerical
                dtype_str = str(column.dtype)
                is_numerical = 'int' in str(dtype_str) or 'float' in dtype_str

                if is_numerical:
                    ax.boxplot(column)
                    ax.set_title(
                        selected_column_names[index], fontweight="bold", fontsize=14)
                else:
                    ax.set_title(
                        f'{selected_column_names[index]}\n- NOT NUMERICAL', fontweight="bold", fontsize=14)

                index += 1
            else:
                ax.axis('off')


def freqBar(data, VAR_LOOKUP, selected_column_names=None, figheight=15, figwidth=15, hspace=0.5, nrows=None, xlabels=None, use_spec_count=None,
            special_col_for_lookup=None, special_titles=None, save_fn=None, check_missing_vals=False):
    """
    Creates barplots for categorical data. Takes data as input
    and a list consisting of the column names of the desired variables.
    :data: structured numpy array or dictionary
    :selected_column_namested: List with column names
    :hspace: Horizontal space between rows (px)
    :nrows: Number of rows within subplot
    :xlabels: List of lables for x-axis
    :use_spec_count: do not use np.unique to count categories but use specified count
    :special_col_for_lookup: when lookin up the variable in VAR LOOKUP do not use the specified column name, but use a different name
    :special_titles: customize the title
    :save_fn: Save figure - provide a filename (fn)
    :check_missing_vals: If true, then it expects the array will be masked
    :return: figure object
    """

    if not selected_column_names:
        selected_column_names = data.dtype.names

    num_boxplots = len(selected_column_names)
    fig, axs = get_proper_axs(num_boxplots, nrows)

    fig.set_figheight(figheight)
    fig.set_figwidth(figwidth)
    fig.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=None, hspace=hspace)

    index = 0

    for row in axs:
        for ax in row:
            if index < num_boxplots:

                # Get column and its name
                if num_boxplots == 1 and not use_spec_count:
                    column_name = selected_column_names[index]
                    column = data
                else:
                    column_name = selected_column_names[index]
                    column = data[column_name]

                # Filter out missing values if needed
                if check_missing_vals:
                    column = column[~column.mask]

                # Get categories and their count
                if use_spec_count:
                    categories, counts = column, column
                else:
                    categories, counts = np.unique(column, return_counts=True)

                # Set labels
                ax.set_ylabel('Count')

                # title
                if special_titles:
                    title = special_titles[index]
                    ax.set_title(f'Barplot of\n{title}',
                                 fontweight="bold", fontsize=14)
                else:
                    ax.set_title(f'Barplot of\n{column_name}',
                                 fontweight="bold", fontsize=14)

                # Set special column name for lookup if needed
                if special_col_for_lookup:
                    column_name = special_col_for_lookup

                # Set ticks if possible
                xticks = [i for i in range(1, len(categories) + 1)]
                column_name_no_under_score = " ".join(column_name.split('_'))

                # Before setting ticks, catch an exception and correct it
                # (For some reason, excel has limit on len of sheet name)
                if column_name_no_under_score == 'Pedestrian CrossingHuman Control':
                    column_name_no_under_score = 'Pedestrian CrossingHuman Contro'

                if VAR_LOOKUP and VAR_LOOKUP.get(column_name_no_under_score) is not None:
                    try:
                        # Get corresponding sheet as pandas data frame
                        df = VAR_LOOKUP.get(column_name_no_under_score)

                        # Use this dataframe lookup to get relevant xtick labels
                        xticks_labels = get_relevant_lookup_values(
                            df, categories)

                        # Set xticks
                        ax.set_xticks(xticks)
                        ax.set_xticklabels(
                            xticks_labels, rotation=45, ha='right')
                    except Exception as e:
                        print(f'{column_name_no_under_score} - {e}')
                elif xlabels:
                    # Set xticks
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(
                        xlabels, rotation=45, ha='right')

                # plot data
                ax.bar(xticks, counts)
                index += 1
            else:
                ax.axis('off')

    if save_fn:
        fig.savefig(save_fn, dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait')


def histograms(data, selected_column_names=None, figheight=15, figwidth=15, hspace=0.5, nrows=None, nbins=None, xticks=None, range=None):
    """
    Plots histogram for submited numerical data.
    :data: structured numpy array or dictionary
    :selected_column_namested: List with column names
    :hspace: Horizontal space between rows (px)
    :return: None
    """

    if not selected_column_names:
        selected_column_names = data.dtype.names

    num_boxplots = len(selected_column_names)
    fig, axs = get_proper_axs(num_boxplots, nrows)

    fig.set_figheight(figheight)
    fig.set_figwidth(figwidth)
    fig.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=None, hspace=hspace)

    index = 0

    for row in axs:
        for ax in row:
            if index < num_boxplots:

                # Get column and its name
                if num_boxplots == 1:
                    column_name = selected_column_names[index]
                    column = data
                else:
                    column_name = selected_column_names[index]
                    column = data[column_name]

                # Plot data
                ax.hist(column, bins=nbins, edgecolor='black', range=range)

                # Add labels and descriptions to plot
                ax.set_title(
                    f'Histogram of\n{column_name}\n', fontweight="bold", fontsize=14)
                ax.set_xlabel(column_name)
                ax.set_ylabel('Frequncy')

                # Add xticks if specifiied
                if xticks:
                    ax.set_xticks(xticks)

                index += 1

            else:
                ax.axis('off')


# =================== HYPOTHESIS TESTING =======================


def do_pearson_chi_square_test(cat_var_A, cat_var_B, rownames, colnames):
    """
    Conducts pearson chi-square hypothesis test on two categorical variables.
    :cat_var_A: structured array
    :cat_var_B: structured array
    :rownames: name of var A
    :colnames: name of var B
    :return: observed and expected values via pd dataframe, pVal
    """

    # Create pandas crosstab table with observed values
    observed_pd = pd.crosstab(cat_var_A, cat_var_B,
                              rownames=rownames, colnames=colnames)
    observed = observed_pd.to_numpy()

    # Calculate chiValue, pValue, degree of freedom, expected values
    chiVal, pVal, df, expected = chi2_contingency(observed, correction=False)

    # Get categories from both vars
    cat_A = np.unique(cat_var_A)
    cat_B = np.unique(cat_var_B)

    # Create a crosstab table
    expected_pd = pd.DataFrame(
        expected,
        index=cat_A,
        columns=cat_B,
        dtype=int
    )

    return observed_pd, expected_pd, chiVal, pVal, df


def get_cramers_v(chiVal, observed):
    """
    Computes Cramer's V which is a measure of assocination
    between two nominal variables giving a value in a range <0, 1>.
    Close to zero means a weak relationship and vice versa.
    :chiVal: float or int
    :observed: pandas dataframe of observed values
    :return: Cramer's V, float
    """

    rowTotals = observed.sum(axis=1)
    N = rowTotals.sum()
    V = np.sqrt((chiVal/N) / (min(observed.shape)-1))
    return V


# =================== MAP VISUALIZATION =======================
def create_folium_map(data, color_map=None, column_to_highlight=None, radius=5):
    """
    Creates a folium map object from the submitted values using the
    longitude and lantitude coordinates.
    :data: structured numpy array
    :color_map: dictionary which maps values to colors
    :radius: size of the circle marker
    :return: folium map object
    """

    # First, get center of the map
    location = data['Latitude'].mean(), data['Longitude'].mean()

    # Start with creating folium map object
    m = folium.Map(location=location,
                   zoom_start=12)

    for row in data:

        # Get coordinates for map
        coord_lat = row['Latitude']
        coord_long = row['Longitude']

        # Get color of the marker if possible
        if column_to_highlight and color_map:
            key = row[column_to_highlight]
            if key in color_map:
                color = color_map[key]
        else:
            color = None

        marker = folium.CircleMarker(
            (coord_lat, coord_long),
            color=color, radius=radius,
            fill=color
        )
        marker.add_to(m)

    return m


def folium_show(m, use_deepnote=False):
    """
    Helper function for plotting follium in Deepnote.
    """
    if use_deepnote:
        data = m.get_root().render()
        data_fixed_height = data.replace('width: 100%;height: 100%', 'width: 100%').replace(
            'height: 100.0%;', 'height: 609px;', 1)
        display(HTML(data_fixed_height))
    else:
        return m


def map_location_to_year(data, lookup_data):
    """
    Maps location to the given year based on the lookup table.
    :data: structured array
    :lookup_data: structured array
    :return: structured array with columns Year, Latitude, Longitude, Label
    """
    # Get the needed data
    years = []
    for row in data:
        lat, lon = np.array(row['Latitude']), np.array(row['Longitude'])
        point_mask = (np.isin(lookup_data['Latitude'], lat)) & (
            np.isin(lookup_data['Longitude'], lon))
        year = lookup_data[point_mask]['Year'].data[0]
        years.append(year)

    # Create a structured array
    vals = [(year, lat, lon, label) for year, lat, lon, label
            in zip(years,
                   data['Latitude'],
                   data['Longitude'],
                   data['Cluster'])]

    result = np.array(vals, dtype=[('Year', int),
                                   ('Latitude', float),
                                   ('Longitude', float),
                                   ('Label', int)])
    return result


def greatcircle(x, y):
    """
    Computes distance between two points using geopy's model 'Great-circle'.
    Find detail info about the computation here:
    https://geopy.readthedocs.io/en/release-0.96.3/
    :x: point 1
    :y: point 2
    :return: distance in meters (float)
    """

    lat1, long1 = x[0], x[1]
    lat2, long2 = y[0], y[1]
    dist = great_circle((lat1, long1), (lat2, long2)).meters
    return dist


def get_color_map(labels):
    """
    Maps labels to corresponding color.
    :return: dict
    """

    # Define available colors
    colors = {
        0: '#e35656',
        1: '#e38c56',
        2: '#5ca641',
        3: '#039491',
        4: '#033d94',
        5: '#3b0394',
        6: '#810394',
        7: '#94034e',
        8: '#940322',
        9: '#ff0a0a',
    }

    # Create the mapping
    color_map = {
        label: colors[label % len(colors)] for label in labels if label != -1
    }

    return color_map


def create_folium_map_clusters(data, max_distance=100, min_cluster_size=10, radius=10):
    """
    Groups points on the map by their distance from each other. But there is also another parameter (min_cluster_size)
    which defines the minimum cluster size. We use sci-kit learn's dbscan function:
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

    For computing distance, we explicitly state a function greatcircle -–> see its doc string for more info.

    NOTE: The code is ours, but the original idea and "HOW  TO DO IT" was adopted from the following source:
    https://www.kaggle.com/hjnotter/using-clustering-to-locate-accident-hotspots, especially knowing that
    we need to use dbscan function and how to use it was important to know for us and the source explained it well.

    :data: structured numpy array with columns 'Latitude' and 'Longitude'
    :max_distance: maximum distance between two points on the map in meters
    :min_cluster_size: minimum number of points for cluster to be considered
    :return: folium map object and meta data via structured array
    """

    # Create pandas dataframe from the two latitude and longitude
    valid_vals_mask = (~data['Latitude'].mask) & (~data['Longitude'].mask)
    data = {
        'Latitude': data['Latitude'][valid_vals_mask].astype(float),
        'Longitude': data['Longitude'][valid_vals_mask].astype(float)
    }
    df = pd.DataFrame(data, columns=['Latitude', 'Longitude'])

    # Get a label for each datapoint, if there was not cluster find for given point, it will be given label -1
    labels = dbscan(eps=max_distance, min_samples=min_cluster_size,
                    metric=greatcircle).fit(df).labels_

    # Define colors for clusters
    color_map = get_color_map(labels)

    # Convert data into structured numpy array and filter out -1
    vals = [(lab, lat, lon) for lab, lat, lon
            in zip(labels,
                   df['Latitude'],
                   df['Longitude']) if lab != -1]
    if vals:
        structured_arr = np.array(vals,
                                  dtype=[('Cluster', int),
                                         ('Latitude', float),
                                         ('Longitude', float)])

        M = create_folium_map(
            structured_arr, color_map=color_map, column_to_highlight='Cluster', radius=radius)

        return M, structured_arr
    else:
        print('No clusters found, try to loose the conditions')


def map_points_to_cluster_centers(data):
    """
    First, Computes center of the submitted cluster by finding mean of the latitude and longitude of all points.
    NOTE: This calculation makes assumption that given points are being quite close to each other, it is fair
    to treat Earth as being flat.

    Second, maps label of the given cluster to the center using a dictionary.

    :data: structured numpy array with the following columns 'Latitude', 'Longitude', 'Cluster'
    :return: dictionary which represents a map between cluster lables and its centers
    """

    # Get cluster labels
    cluster_labels = np.unique(data['Cluster'])

    # Create a dictionary which will map the cluster label to the corresponding center
    map_cluster_to_center = dict()
    for cluster_label in cluster_labels:

        # Ignore -1 since this signifies points which do not belong to any cluster
        if cluster_label != -1:

            # Get cluster points
            cluster_points_mask = data['Cluster'] == cluster_label
            cluster_points_lat = data[cluster_points_mask]['Latitude'].astype(
                float)
            cluster_points_lon = data[cluster_points_mask]['Longitude'].astype(
                float)

            # Compute the center
            center_location = cluster_points_lat.mean(), cluster_points_lon.mean()

            # Save the center to the map
            map_cluster_to_center[cluster_label] = center_location

    return map_cluster_to_center


def map_centroid_to_locations(centroid_location, year, all_tables, max_perimeter=100):
    """
    The goal of this function is to filter out accidents
    which appeared at certain location within certain perimeter from
    the centroid and get the details of these accidents.

    :centroid location: tuple with latitude and longitude
    :year: string denoting given year
    :label: Cluster label, int
    :all_tables: a dictionary where key is a year and value is structured array with accident data
    :max_perimiter: maximum distance between centroid and given point
    :return: relevant accident data as a structured array
    """

    # Get relevant table
    lookup_table = all_tables[year]

    # Create a mask
    mask = list()
    for row in lookup_table:
        coordinates_row = row[['Latitude', 'Longitude']]
        distance = greatcircle(centroid_location, coordinates_row)
        if distance <= max_perimeter:
            mask.append(True)
        else:
            mask.append(False)

    # Use this mask to filter out data
    relevant_data = lookup_table[np.array(mask)]

    return relevant_data


def add_legend(M, color_map):
    """
    Adopted code from here:
    https://nbviewer.jupyter.org/gist/talbertc-usgs/18f8901fc98f109f2b71156cf3ac81cd
    NOTE: The majority of template snippet is NOT our work, we adjusted it to our needs to display
    given labels.
    :M: folium map object
    :return: map object with added legend
    """

    # Build labels html code
    labels = ""
    for label, color in color_map.items():
        labels = f"{labels}<li><span style='background:{color};opacity:0.7;'</span>{str(label)}</li>"

    template = """
    {% macro html(this, kwargs) %}

    <!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>jQuery UI Draggable - Default functionality</title>
    <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">

    <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
    <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
    
    <script>
    $( function() {
        $( "#maplegend" ).draggable({
                        start: function (event, ui) {
                            $(this).css({
                                right: "auto",
                                top: "auto",
                                bottom: "auto"
                            });
                        }
                    });
    });

    </script>
    </head>
    <body>

    
    <div id='maplegend' class='maplegend' 
        style='position: absolute; z-index:9999; border:2px solid grey; background-color:rgba(255, 255, 255, 0.8);
        border-radius:6px; padding: 10px; font-size:14px; right: 20px; bottom: 20px;'>
        
    <div class='legend-title'>Legend</div>
    <div class='legend-scale'>
    <ul class='legend-labels'>
    SPLIT_HERE
    </ul>
    </div>
    </div>
    
    </body>
    </html>

    <style type='text/css'>
    .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
    .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: right;
        list-style: none;
        }
    .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 10x;
        line-height: 18px;
        margin-bottom: 2px;
        }
    .maplegend ul.legend-labels li span {
        display: block;
        float: right;
        color: #ffffff;
        height: 15px;
        width: 45px;
        margin-right: 5px;
        margin-left: 10px;
        border: 1px solid #999;
        }
    .maplegend .legend-source {
        font-size: 80%;
        color: #ffffff;
        clear: both;
        }
    .maplegend a {
        color: #777;
        }
    </style>
    {% endmacro %}"""

    # Adjust the string
    part1, part2 = template.split('SPLIT_HERE')
    template = part1 + labels + part2

    macro = MacroElement()
    macro._template = Template(template)

    M.get_root().add_child(macro)

    return M
