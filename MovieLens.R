if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.r-project.org")

library(stringr)
library(ggplot2)
library(caret)
library(readr)
library(dplyr)
library(stats)
library(corrplot)
library(tidyr)
library(lubridate)
library(data.table)
library(DT)
library(methods)
library(knitr)
library(grid)
library(gridExtra)
library(Matrix)
library(corrplot)
library(RColorBrewer)
library(corrplot)
library(magrittr)
library(viridis)
library(stringi)
library(matrixStats)
library(heuristica)
library(gam)
library(modelr)
library(tidyr)
library(tidyselect)
library(broom)
library(tibble)
library(purrr)
library(forcats)
library(DBI)
library(hexbin)

# Download the data
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", 
                                  readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

# Load and tidy data
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")
# Validation set will be 10% of MovieLens data

#set.seed(1) # if using R 3.6.0: set.seed(1, sample.kind = "Rounding")
if (paste(version$major,".", version$minor, sep="") < '3.6.0') {
  set.seed(1)
} else {
  set.seed(1, sample.kind = "Rounding", warning)
}

test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Find the dimension 
print("Dimension: ")
dim(edx)
print("Rows:")
nrow(edx)
print("Columns:")
ncol(edx)

# Inspect the data
head(edx)
print("The number of unique movies: ")
n_distinct(edx$movieId)

print("The number of unique users")
n_distinct(edx$userId)

print("Movie ratings by genre")
edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

print ("Movie with highest ratings in descending order")
edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

print ("Top 5 ratings")
edx %>% group_by(rating) %>% summarize(count = n()) %>% top_n(5) %>%
  arrange(desc(count))

print("Graph of ratings the movies")
edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line()

# Initial model - Find the average to start with
mu <- mean(edx$rating)
mu

# Find the naive root mean square error
naive_rmse <- RMSE(validation$rating, mu)
naive_rmse

# Present the data as a table

rmse_results <- tibble(method = "Average movie rating model", RMSE = naive_rmse)
rmse_results %>% knitr::kable()

movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

movie_avgs %>% ggplot(mapping = aes(x = b_i)) +
                  geom_histogram(bins = 10, fill="grey", colour = "black") +
                  labs(x = "bias(b_i)", y = "Movie count", title = "Number of movies with the computed b_i")

# Penalty term Movie effect
predicted_ratings <- mu +  validation %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

movie_effect_model_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results, tibble(method="Movie effect model", RMSE = movie_effect_model_rmse ))
rmse_results %>% knitr::kable()

# Model with movie influence
user_avgs<- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating - mu - b_i))

user_avgs %>% ggplot(mapping = aes(x = b_u)) +
  geom_histogram(bins = 30, fill="grey", colour = "black") +
  labs(x = "bias(b_u)", y = "Movie count", title = "Number of movies with the computed b_u")

user_avgs <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- validation%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

models_user_movies_rmse <- RMSE(predicted_ratings, validation$rating)

rmse_results <- bind_rows(rmse_results, tibble(method="Movie and user effect model", RMSE = models_user_movies_rmse))
rmse_results %>% knitr::kable()

# calculate lambdas to find out mimimum RMSE
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l) {
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n() + l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n() + l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))
})

# Plot lambdas vs RMSE
qplot(lambdas, rmses)

# Find out the lambda for minimum RMSE
lambda <- lambdas[which.min(rmses)]
lambda

# Add all RMSE to compare
rmse_results <- bind_rows(rmse_results, tibble(method="Improvized movie and user effect model", RMSE = min(rmses)))
rmse_results %>% knitr::kable()

print("R software version details:")
version
