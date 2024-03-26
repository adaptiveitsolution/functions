from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, lit, mean, stddev, current_date, expr, when, first, concat_ws, date_format

def analyze_outliers_fixed_window(df, detail_cols, date_col, feature_col, control_limit):
    """
    Analyzes and returns control statistics for a given feature across fixed 25-month windows, 
    partitioned by specified detail columns in a Spark DataFrame. The function calculates 
    control statistics including the mean, standard deviation, upper control limit (UCL), 
    and lower control limit (LCL) for each unique combination of detail columns within each 
    window. These statistics are intended for use in Statistical Process Control (SPC) analysis.
    
    The function first generates a unique identifier for each window based on the current date 
    and detail columns. It then calculates the fixed mean and standard deviation for the feature 
    column within each group defined by the detail columns and window identifier. Finally, it 
    computes the UCL and LCL for each group as the fixed mean plus or minus the control limit 
    times the fixed standard deviation, respectively.
    
    Parameters:
    - df (DataFrame): The Spark DataFrame containing the data to be analyzed. Must include the 
      columns specified by `detail_cols`, a date column named as per `date_col`, and a feature 
      column as per `feature_col`.
    - detail_cols (list of str): The names of the columns to partition the data by for the analysis. 
      These columns define the granularity of the analysis and the resulting statistics.
    - date_col (str): The name of the column containing date information. This column is used to 
      order the data within each partition and to define the fixed windows for analysis.
    - feature_col (str): The name of the column containing the feature to analyze for outliers 
      and to calculate control statistics for.
    - control_limit (float): The multiplier for the standard deviation to define the upper and lower 
      control limits. A common value is 3, which corresponds to limits set at ±3 standard deviations 
      from the mean, under the assumption of normal distribution of the data.

    Returns:
    - DataFrame: A Spark DataFrame containing the unique combinations of detail columns, each 
      associated with its calculated mean, standard deviation, upper control limit, and lower 
      control limit for the specified feature across fixed 25-month windows. Additional columns 
      may include a unique window identifier, depending on the implementation.

    Notes:
    - The returned DataFrame does not include the original data points or any outlier classification. 
      It is a summary table useful for lookup or join operations in SPC or further analysis.
    - This function assumes that the data is pre-sorted by the date column if necessary. If the 
      data is not sorted, the window identification and subsequent calculations may not accurately 
      reflect the intended 25-month periods.
    """
    # Define the window specification for partitioning and ordering the data
    windowSpec = Window.partitionBy(*detail_cols).orderBy(date_col)

    # Generate a window_id for each 25-month period
    analysis_date = date_format(current_date(), "yyyy-MM-dd")
    concat_cols = concat_ws(" | ", lit(analysis_date), *map(col, detail_cols))
    df = df.withColumn("window_id", concat_cols)

    # Calculate mean, stddev, upper and lower control limits for each group + window
    windowAggSpec = Window.partitionBy(*detail_cols, 'window_id')
    df = df.withColumn('fixed_mean', mean(col(feature_col)).over(windowAggSpec))
    df = df.withColumn('fixed_stddev', stddev(col(feature_col)).over(windowAggSpec))
    df = df.withColumn('upper_control_limit', col('fixed_mean') + control_limit * col('fixed_stddev'))
    df = df.withColumn('lower_control_limit', col('fixed_mean') - control_limit * col('fixed_stddev'))

    # Select distinct detail columns and their associated statistics
    stat_columns = ['fixed_mean', 'fixed_stddev', 'upper_control_limit', 'lower_control_limit']
    df_stats = df.select(*detail_cols, *stat_columns, 'window_id').distinct()

    # Optionally, you might want to drop or modify the window_id column if it doesn't match your needs
    # For simplicity, I'm leaving it as is, but it can be customized to represent actual 25-month windows or removed

    return df_stats
