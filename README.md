# MovieTweetings ALS Collaborative Filtering 
## Business Objective: Increase Recommendation Accuracy for our active Twitter Users

![Business Mockup](Business_Mockup.png)

Our users will be able to log into Twitter where their previous movie ratings will provide the basis for new movie recommendations to be made. This objective will be accomplished by exploiting the relationships between users and between movies.We will further addres the problem of when users have not rated a sufficient amount of movies by implementing an average based solution, so these users, too, will have recommendations avaliable to them. 


# Dataset:

[MovieTweetings](https://github.com/sidooms/MovieTweetings) is a dataset obtained by scraping Twitter for well-structured tweets for movie ratings of the form similar to the following:
"I rated The Matrix 9/10 http://www.imdb.com/title/tt0133093/ #IMDb"
According to the documentation, “On a daily basis the Twitter API is queried for the term ‘I rated #IMDb’.”

## users.dat
Contains the mapping of the users ids on their true Twitter id in the following format: userid::twitter_id. 

## ratings.dat
Contains the extracted ratings are stored in the following format: user_id::movie_id::rating::rating_timestamp.

## movies. dat
Contains the movies that were rated in the tweets followed by the year of release and associated genres in the following format: movie_id::movie_title (movie_year)::genre|genre|genre.

# Model Selection Criteria:
We aim to minimize RMSE while providing a minimized computation time to provide the most accurate recommendations as fast as possible. When tuning the model this shall be taken into consideration. 

# Methodology:

Primary exploratory data analysis and visualization will be done to explore the dataset at hand.

We will be undertaking a Spark implementation of ALS (Alternating Least Squares) with implicit feedback and explicit numerical ratings models taken into consideration. Implicit feedback will be of the form like versus dislike based on a threshold of a rating of 5. 

The assumptions underlying the implicit feedback model is that an individual may exhibit voluntary response bias, only rating and subsequently tweeting when they strongly feel a certain way about
a movie though then the numerical rating may be arbitrary.  Namely this can be thought of as a user will tweet their rating about a movie when they love the movie or hate it. Further a priori to their rating, the user may have felt that they would enjoy they movie, which is why they watched it. Taking into account these factors, the implicit feedback model is considered. 

The explicit model will take the explicit ratings from the scraped tweets into account. 

Based on the resulting metrics of the two models and their computational complexity, a model will be chosen and hyperparameters will be tuned via grid search. 
