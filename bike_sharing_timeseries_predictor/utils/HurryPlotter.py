import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.dates as mdates
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_categorical_dtype

class HurryPlotter:
    def __init__(self, dataframe):
        # Store the DataFrame directly
        self.data = dataframe

        # Ensure all datetime columns are converted to datetime objects
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                try:
                    self.data[col] = pd.to_datetime(self.data[col])
                except ValueError:
                    pass

    def plot(self, x, y, title='Title', x_label='x-axis', y_label='y-axis', groupby=None, filter_condition=None, resample=None, show_statistics=False):
        # Make a copy of the DataFrame to avoid modifying the original data
        data_to_plot = self.data.copy()
    
        # Filter data if a condition is provided
        if filter_condition:
            data_to_plot = data_to_plot.query(filter_condition)
        
        # If resample is specified, perform resampling and sum the data
        if resample and is_datetime64_any_dtype(data_to_plot[x]):
            data_to_plot = data_to_plot.resample(resample, on=x)[y].sum().reset_index()
            data_to_plot[x] = data_to_plot[x].dt.strftime('%Y-%m')  # Format the datetime column
        
        # Group data if groupby is specified and perform sum
        elif groupby:
            data_to_plot = data_to_plot.groupby(groupby)[y].sum().reset_index()

        # Determine the type of the x-axis
        x_type = data_to_plot[x].dtype

        if is_datetime64_any_dtype(x_type) or np.issubdtype(x_type, np.float64): #is_numeric_dtype(x_type)
            plot_func = sns.lineplot
            is_date = is_datetime64_any_dtype(x_type)
        elif resample or is_categorical_dtype(x_type) or np.issubdtype(x_type, np.integer):
            plot_func = sns.barplot
            is_date = False
        else:
            raise ValueError("x-axis type not supported for plotting")

        plt.figure(figsize=(10, 6))
        plot_func(data=data_to_plot, x=x, y=y)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if is_date:
            # Set the locator and formatter for x-axis to show dates nicely
            ax = plt.gca()
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.set_xlim(data_to_plot[x].min(), data_to_plot[x].max())

        if (groupby or resample) and show_statistics:
            weighted_stats = self.calculate_weighted_stats(data_to_plot, groupby, y)
            self.display_stats(plt.gca(), weighted_stats)

        plt.xticks(rotation=45)
        plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
        plt.show()

    @staticmethod
    def calculate_weighted_stats(data, groupby_col, value_col):
        weighted_mean_value = np.average(data[groupby_col], weights=data[value_col])
        weighted_variance_value = np.average((data[groupby_col] - weighted_mean_value) ** 2, weights=data[value_col])
        return {'mean': weighted_mean_value, 'variance': weighted_variance_value}

    @staticmethod
    def display_stats(ax, stats):
        textstr = f'Mean: {stats["mean"]:.2f}\nVariance: {stats["variance"]:.2f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.8, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)


def inverse_min_max_scaling(scaled_val, original_min, original_max):
    return scaled_val * (original_max - original_min) + original_min

if __name__ == '__main__':

    # Loading data and preprocessing
    df = pd.read_csv("./data/hour.csv")
    df.drop_duplicates()

    # Convert time to datetime format for better visulization
    df['dteday'] = pd.to_datetime(df['dteday'])
    df['dtehr'] = df.apply(lambda row: row['dteday'] + pd.Timedelta(hours=row['hr']), axis=1)

    # Restore min max scaled variable to the original for better visulization
    df['temp_origin'] = df['temp'].apply(lambda x: inverse_min_max_scaling(x, -8, 39))
    df['atemp_origin'] = df['atemp'].apply(lambda x: inverse_min_max_scaling(x, -16, 50))
    df['windspeed_origin'] = df['windspeed'].apply(lambda x: x*67)

    # Initialization of HurryPlotter
    plotter = HurryPlotter(df)
    #plotter.plot(x='dteday', y='cnt', title='Time Series Plot of Bike Rentals', x_label='Time', y_label='Number of Bike Rentals',resample='Q')
    #plotter.plot(x='season', y='cnt', groupby='season', title='Total Bike Rentals by Season', x_label='Season', y_label='Total Number of Bike Rentals', show_statistics=True)
    #plotter.plot(x='temp', y='cnt', groupby='temp', title='Temperature Rental', x_label='Temperature', y_label='Total Number of Bike Rentals')
    plotter.plot(x='season', y='cnt', groupby='season', filter_condition='yr==1', title='Total Bike Rentals by Season', x_label='Season', y_label='Total Number of Bike Rentals', show_statistics=True)


