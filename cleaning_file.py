import pandas as pd

def convertEpisodeResults(df):
    # Group by "State" and "Action", calculate count, sum, and average
    grouped_data = df.groupby(['State', 'Action']).agg({'Value': ['count', 'sum']}).reset_index()
    grouped_data.columns = ['State', 'Action', 'Count', 'Sum']

    # Calculate average based on the count
    grouped_data['Average'] = grouped_data['Sum'] / grouped_data['Count']
    grouped_data = grouped_data.round(4)

    return grouped_data

df = pd.read_csv("episode_results.csv")
cleaned_df = convertEpisodeResults(df)
cleaned_df.to_csv("master_df.csv")