from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, mean, stddev, when
from pyspark.sql.window import Window

def analyze_outliers(df: DataFrame, detail_cols: list, date_col: str, feature_col: str) -> DataFrame:
    """
    Analyzes outliers in a feature column of a Spark DataFrame over a 25-month window,
    grouped by specified detail columns.

    Parameters:
    - df: Spark DataFrame containing the data.
    - detail_cols: List of column names to group by (e.g., ['location', 'type', 'form']).
    - date_col: Name of the column containing the date.
    - feature_col: Name of the column containing the feature to analyze.

    Returns:
    - DataFrame with an additional column '25_month_control_analysis' indicating outliers.
    """
    # Define the window spec for 25 months, grouped by detail columns and ordered by date
    windowSpec = Window.partitionBy(*detail_cols).orderBy(date_col).rowsBetween(-24, 0)

    # Calculate rolling mean and standard deviation for the feature column
    df = df.withColumn('mean', mean(col(feature_col)).over(windowSpec))
    df = df.withColumn('stddev', stddev(col(feature_col)).over(windowSpec))

    # Define bounds for outliers: values outside mean ± 1.5 * stddev are considered outliers
    df = df.withColumn('is_outlier', when(
        (col(feature_col) < (col('mean') - 1.5 * col('stddev'))) | 
        (col(feature_col) > (col('mean') + 1.5 * col('stddev'))), lit(True)).otherwise(lit(False)))

    # Adding a column "25_month_control_analysis" to mark outliers within the control
    df = df.withColumn('25_month_control_analysis', when(col('is_outlier') == True, 'Outlier').otherwise('Normal'))

    # Drop intermediate columns if not needed
    df = df.drop('mean', 'stddev', 'is_outlier')

    return df

# Example usage:
# Assuming `df` is your DataFrame loaded with data
# detail_cols = ['location', 'type', 'form']
# date_col = 'date'
# feature_col = 'feature'
# df_with_outliers = analyze_outliers(df, detail_cols, date_col, feature_col)

