from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, lit, mean, stddev, when, first

def analyze_outliers_fixed_window(df, detail_cols, date_col, feature_col, control_limit):
    """
    Analyzes outliers in a feature column of a Spark DataFrame over fixed 25-month windows,
    grouped by specified detail columns. Unlike a rolling window, this function calculates
    a constant mean and standard deviation for each 25-month period for the entire group,
    and then evaluates each record in the window against these fixed statistics.

    Parameters:
    - df (DataFrame): Spark DataFrame containing the data to be analyzed.
    - detail_cols (list): List of column names to group by for the analysis (e.g., ['location', 'type', 'form']).
    - date_col (str): Name of the column containing the date. Assumes data is sorted by this column if necessary.
    - feature_col (str): Name of the column containing the feature to analyze for outliers.
    - control_limit (float): Scale value multiplied by standard deviation for threshold of outlier.

    Returns:
    - DataFrame: The input DataFrame augmented with a '25_month_control_analysis' column indicating
                 whether each row is considered an 'Outlier' or 'Normal' based on the fixed window
                 analysis. Intermediate columns used for calculations are dropped.

    The function first partitions the data by the specified detail columns and orders it by the date column.
    It then identifies the start of each 25-month window for grouping. For each group defined by the detail
    columns and the window start, it calculates a fixed mean and standard deviation of the feature column.
    Each record's feature value is compared against these fixed values to determine if it is an outlier,
    using the criterion of being outside fixed_mean ± 1.5 * fixed_stddev. The result is a new column in the
    DataFrame labeling each record as 'Outlier' or 'Normal'.
    """
    # Define the window specification for partitioning the data
    windowSpec = Window.partitionBy(*detail_cols).orderBy(date_col)

    # Assign a unique window ID based on concatenation of window start and detail columns
    # This is a simplistic approach and might need customization for more complex scenarios
    analysis_date = F.date_format(F.current_timestamp(), "yyyy-MM-dd")
    concat_cols = F.concat_ws("|", lit(analysis_date), *map(lambda cols:   F.col(cols),detail_cols))
    df = df.withColumn("window_id", concat_cols)

    # Calculate mean and stddev for the entire window now, based on window_start
    windowAggSpec = Window.partitionBy(*detail_cols, 'window_id')
    df = df.withColumn('fixed_mean', mean(col(feature_col)).over(windowAggSpec))
    df = df.withColumn('fixed_stddev', stddev(col(feature_col)).over(windowAggSpec))

    # Define bounds for outliers: values outside fixed_mean ± 1.5 * fixed_stddev are considered outliers
    df = df.withColumn('is_outlier', when(
        (col(feature_col) < (col('fixed_mean') - control_limit * col('fixed_stddev'))) | 
        (col(feature_col) > (col('fixed_mean') + control_limit * col('fixed_stddev'))), lit(True)).otherwise(lit(False)))

    # Adding a column "25_month_control_analysis" to mark outliers within the control
    df = df.withColumn('25_month_control_analysis', when(col('is_outlier') == True, 'Outlier').otherwise('Normal'))

    # Optionally, drop intermediate columns
    df = df.drop('mean', 'stddev', 'window_id', 'fixed_mean', 'fixed_stddev', 'is_outlier')

    return df
