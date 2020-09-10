# CS6375 Group Project: Box Office Forecast
### Team Blockbuster
#### Team Member: Jin Liu, Rakesh Reddy, Prakshita Nag

### Progress Check

####Data Pre-Processing
1. To train models on the profitability of movies, we need both the budget and Box office number, therefore any movie missing one or both of these two numbers was removed from the dataset.
2. The dataset provided the original budget and box office number. However, given the fact that the dataset contains movies over two decades (1997-2017), the inflation factor must be considered. Therefore, we normalized the budget and box office numbers year-by-year using the inflation rate of 2017, so that we can make valid analyses on movies across two decades.
3. The profitability of a movie is calculated as such: the ratio of box office over budget, based upon which we divided the movies into four categories:
Poor(1): ratio < 1 These movies definitely lose money from box office.
Fair(2): ratio between 1 and 2. These movies can earn budget back from box office, but are barely profitable.
Good(3): ratio between 2 and 4. These movies are guaranteed profitable.
Great(4): ratio above 4. These movies are the true blockbusters that earned more than expected.
4. Some features that are either obviously not relevant to their box office numbers, or too complicated to calculate in this project, were removed, including movie ID, movie name, posters, keywords, DVD release date, length, and awards.
5. Since the movies released in different times in a year may fair differently, we extracted the months from the release date to be included in our further analysis.
6. To accomodate the features like producer, actors, director we replaced the data in them with the net box office gross for each of them.
	A. For actors we summed up the net box office gross of each of the actors.
	B. For production we replaced it with the total gross of all the movies of the production house.
	C. For Directors we replaced the director with the net box office gross of all the movies directed by him.

####Logistic Thinking on Modeling choice
1. From the producer's perspective, it is essential to know if they can earn their budget back from box office, whereas for the distributers, who take money directly from box office, the box office numbers matter more to them. To build models that can satisfy both requirements, we decided to create two models: one is a regression model to predict the box office number, the other is a classification model to predict the profitability of a movie.
2. Movies released between 1997 and 2015 will be used for training, and movies in 2016 and 2017 will be used to test our model.

####Preliminary Analysis
1. The average box office was calculated for each genre. The animation,adventure and Sci-Fi movies earn the highest average box office, while the short, news and documentary films fair worst.
2. The average box office was also calculated for each movie rating. G, PG and PG-13 movies fair better than R and NC17 on average.
3. The average box office was calculated by month. Movies released in May, June and December earn higher in average box office compare to other months of a year.
4. If we build a Boost Tree model using just the ratings and genre to classify the profitability of movies, based upon the feature importance analysis, the top five features are: Comedy, Drama, Romance, PG-13 and R.

####Room for improvement
1. Find a way to use the movie plot as a feature by creating a tf-idf score, also using word2vec to find the cosine distance by comparing a movie plot with successful and unsuccessful movies and assigning it a score.