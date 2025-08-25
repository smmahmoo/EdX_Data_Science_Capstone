##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Add movie release year to the data set
edx <- edx %>% mutate(release_year = suppressWarnings(as.numeric(stringr::str_extract(title, "(\\d{4})"))))

# Quick check
dim(edx); dim(final_holdout_test); length(unique(edx$userId)); length(unique(edx$movieId))

###################################################################################################

tibble(
  n_rows_edx = nrow(edx),
  n_rows_final_holdout = nrow(final_holdout_test),
  n_users = n_distinct(edx$userId),
  n_movies = n_distinct(edx$movieId),
  rating_mean = mean(edx$rating),
  rating_sd   = sd(edx$rating)
)
summary(edx$rating)

edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.5, color = "white") +
  labs(title = "Distribution of Ratings (edx)",
       x = "Rating", y = "Count")

edx %>%
  count(title, sort = TRUE) %>%
  slice_head(n = 10) %>%
  ggplot(aes(x = reorder(title, n), y = n)) +
  geom_col() +
  coord_flip() +
  labs(title = "Top 10 Most-Rated Movies (edx)", x = "Movie", y = "Number of ratings")

edx %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(binwidth = 0.1, color = "white") +
  scale_x_log10() +
  labs(title = "Distribution of Number of Ratings per User (log scale)", 
       x = "Ratings per user (log10)", y = "Users")

edx %>%
  separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarise(mean_rating = mean(rating), n = n(), .groups = "drop") %>%
  arrange(desc(mean_rating)) %>%
  ggplot(aes(x = reorder(genres, mean_rating), y = mean_rating)) +
  geom_col() +
  coord_flip() +
  labs(title = "Average Rating by Genre", x = "Genre", y = "Average rating")

edx %>%
  filter(!is.na(release_year), dplyr::between(release_year, 1920, 2010)) %>%
  group_by(release_year) %>%
  summarise(mean_rating = mean(rating), .groups = "drop") %>%
  ggplot(aes(release_year, mean_rating)) +
  geom_line() +
  geom_point(size = 0.8) +
  labs(title = "Average Rating by Release Year", x = "Release year", y = "Average rating")

RMSE <- function(true, pred) sqrt(mean((true - pred)^2))

set.seed(2, sample.kind = "Rounding")
idx <- createDataPartition(edx$rating, p = 0.9, list = FALSE)
edx_train <- edx[idx, ]
edx_val   <- edx[-idx, ]

# Ensure IDs overlap
edx_val <- edx_val %>%
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

dim(edx_train); dim(edx_val)

mu_hat <- mean(edx_train$rating)

# Movie effects
b_i <- edx_train %>%
  group_by(movieId) %>%
  summarise(b_i = mean(rating - mu_hat), .groups = "drop")

# Predict movie effect on edx_val
pred_movie_val <- edx_val %>%
  left_join(b_i, by = "movieId") %>%
  mutate(pred = mu_hat + ifelse(is.na(b_i), 0, b_i)) %>%
  pull(pred)
rmse_movie_val <- RMSE(edx_val$rating, pred_movie_val)

# User effects (after movie)
b_u <- edx_train %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu_hat - b_i), .groups = "drop")

pred_user_val <- edx_val %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu_hat + ifelse(is.na(b_i), 0, b_i) + ifelse(is.na(b_u), 0, b_u)) %>%
  pull(pred)
rmse_user_val <- RMSE(edx_val$rating, pred_user_val)

tibble(Model = c("Movie Effect (val)", "Movie + User (val)"),
       RMSE = c(rmse_movie_val, rmse_user_val))

lambdas <- seq(0, 10, 0.25)

rmse_by_lambda <- sapply(lambdas, function(l) {
  # Regularized movie effect
  b_i_reg <- edx_train %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu_hat) / (n() + l), .groups = "drop")
  
  # Regularized user effect
  b_u_reg <- edx_train %>%
    left_join(b_i_reg, by = "movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - mu_hat - b_i) / (n() + l), .groups = "drop")
  
  # Predict on validation split
  pred_val <- edx_val %>%
    left_join(b_i_reg, by = "movieId") %>%
    left_join(b_u_reg, by = "userId") %>%
    mutate(pred = mu_hat + ifelse(is.na(b_i), 0, b_i) + ifelse(is.na(b_u), 0, b_u)) %>%
    pull(pred)
  
  RMSE(edx_val$rating, pred_val)
})

lambda_best <- lambdas[which.min(rmse_by_lambda)]
lambda_best

tibble(lambda = lambdas, RMSE = rmse_by_lambda) %>%
  ggplot(aes(lambda, RMSE)) +
  geom_line() + geom_point(size = 0.8) %>%
  labs(title = "Validation RMSE vs. Regularization (λ)",
       x = "λ", y = "Validation RMSE")

mu_hat_full <- mean(edx$rating)

b_i_full <- edx %>%
  group_by(movieId) %>%
  summarise(b_i = sum(rating - mu_hat_full) / (n() + lambda_best), .groups = "drop")

b_u_full <- edx %>%
  left_join(b_i_full, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - mu_hat_full - b_i) / (n() + lambda_best), .groups = "drop")

# 1) Global mean baseline
rmse_global <- RMSE(final_holdout_test$rating, rep(mu_hat_full, nrow(final_holdout_test)))

# 2) Movie effect only
pred_movie_test <- final_holdout_test %>%
  left_join(b_i_full, by = "movieId") %>%
  mutate(pred = mu_hat_full + ifelse(is.na(b_i), 0, b_i)) %>%
  pull(pred)
rmse_movie_test <- RMSE(final_holdout_test$rating, pred_movie_test)

# 3) Movie + user effect (un-regularized)
b_i_full_noreg <- edx %>% group_by(movieId) %>% summarise(b_i = mean(rating - mu_hat_full), .groups = "drop")
b_u_full_noreg <- edx %>%
  left_join(b_i_full_noreg, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu_hat_full - b_i), .groups = "drop")

pred_user_test <- final_holdout_test %>%
  left_join(b_i_full_noreg, by = "movieId") %>%
  left_join(b_u_full_noreg, by = "userId") %>%
  mutate(pred = mu_hat_full + ifelse(is.na(b_i), 0, b_i) + ifelse(is.na(b_u), 0, b_u)) %>%
  pull(pred)
rmse_user_test <- RMSE(final_holdout_test$rating, pred_user_test)

# 4) Regularized movie + user
pred_reg_test <- final_holdout_test %>%
  left_join(b_i_full, by = "movieId") %>%
  left_join(b_u_full, by = "userId") %>%
  mutate(pred = mu_hat_full + ifelse(is.na(b_i), 0, b_i) + ifelse(is.na(b_u), 0, b_u)) %>%
  pull(pred)
rmse_reg_test <- RMSE(final_holdout_test$rating, pred_reg_test)

rmse_results <- tibble(
  Model = c("Global Average",
            "Movie Effect",
            "Movie + User Effect (unregularized)",
            "Regularized Movie + User"),
  RMSE  = c(rmse_global, rmse_movie_test, rmse_user_test, rmse_reg_test)
)

rmse_results

rmse_results %>%
  ggplot(aes(x = reorder(Model, RMSE), y = RMSE)) +
  geom_col() +
  coord_flip() +
  geom_text(aes(label = round(RMSE, 5)), hjust = -0.1, size = 3.5) +
  labs(title = "RMSE by Model (Evaluated on Final Hold-Out Test)",
       x = "Model", y = "RMSE")