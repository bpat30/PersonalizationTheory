# MovieTweetings ALS Collaborative Filtering 
## Business Objective: Increase Recommendation Accuracy for our active Twitter Users

![Business Mockup](Business_Mockup.png)

Our users will be able to log into Twitter where their previous movie ratings will provide the basis for new movie recommendations to be made. This objective will be accomplished by exploiting the relationships between users and between movies.We will further addres the problem of when users have not rated a sufficient amount of movies by implementing an average based solution, so these users, too, will have recommendations avaliable to them. 


# Dataset:
Bibtex: @conference{Dooms13crowdrec, author = {Dooms, Simon and De Pessemier, Toon and Martens, Luc}, title = {MovieTweetings: a Movie Rating Dataset Collected From Twitter}, booktitle = {Workshop on Crowdsourcing and Human Computation for Recommender Systems, CrowdRec at RecSys 2013}, year = {2013} }


MovieTweetings is a dataset obtained by scraping Twitter for well-structured tweets for movie ratings of the form similar to the following:
"I rated The Matrix 9/10 http://www.imdb.com/title/tt0133093/ #IMDb"
According to the documentation, “On a daily basis the Twitter API is queried for the term ‘I rated #IMDb’.”

## users.dat
Contains the mapping of the users ids on their true Twitter id in the following format: userid::twitter_id. 

## ratings.dat
Contains the extracted ratings are stored in the following format: user_id::movie_id::rating::rating_timestamp.

## movies. dat
Contains the movies that were rated in the tweets followed by the year of release and associated genres in the following format: movie_id::movie_title (movie_year)::genre|genre|genre.
