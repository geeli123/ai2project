# Data retreived from http://www.cs.cmu.edu/~ark/personas/
import pandas as pd
import numpy as np
import re

# Local store for plot and movie data
plot_summaries_path = 'MovieSummaries/plot_summaries.txt'
movie_path = 'MovieSummaries/movie.metadata.tsv'
csv_name = 'movie_data.csv'
plot_labels = ['Wiki_ID', 'Summary']
plot_summary_data = pd.read_csv(plot_summaries_path, sep='\t', names=plot_labels)

movie_labels = ['Wiki_ID', 'Free_ID', 'Movie_Name', 'Release_Date', 'Revenue', 'Runtime', 'Languages', 'Countries', 'Genres']
movie_data = pd.read_csv(movie_path, sep='\t', names=movie_labels)

# Join Dataframes on Wikipedia ID
movie_data = movie_data.set_index('Wiki_ID').join(plot_summary_data.set_index('Wiki_ID'))
# Drop unnecessary data columns
movie_data = movie_data.drop(['Free_ID', 'Movie_Name', 'Release_Date', 'Revenue', 'Runtime', 'Languages', 'Countries'], axis=1)
# Remove all non-matched plot-to-movie rows
movie_data = movie_data.dropna().reset_index(drop=True)

# Keep only most popular movie genres
genre = {'action', 'adventure', 'drama', 'comedy', 'family', 'science fiction', 'horror', 'animation', 'mystery', 'thriller', 'romance', 'crime', 'fantasy'}

for index, row in movie_data.iterrows():
    # Set genre to lower case, extract the genre from data using regex
    line_values = [x.lower() for x in re.findall(r'\"[A-Za-z]+\s?[A-Za-z]*\"', row.Genres) if re.sub('"', '', x.lower()) in genre]
    # Remove 'film' and '"' strings to standardize genre output, join as a string with colon delimiter
    movie_data.iloc[index].Genres = re.sub(r'\"|\sfilm', '', ':'.join(line_values))
    # Remove non-plot related information from summary
    summary = re.sub(r'{.*}|<.*', '', row.Summary)
    movie_data.iloc[index].Summary = summary

movie_data.replace('', np.nan, inplace=True)
movie_data.dropna()
movie_data.to_csv(csv_name, encoding='utf-8', index=False)
