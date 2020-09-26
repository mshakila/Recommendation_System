# ASSIGNMENT RECOMMENDATION ENGINE

# Business Problem: To provide interesting book recommendations to readers

#importing library for Recommendation system
library(recommenderlab)

book1 <- read.csv("E:\\Recommendation_System\\book_recommendation.csv")
head(book1,3)
book1 <- book1[,-1] # removing 1st column as it is index-numbers
dim(book1)
str(book1)
length(book1$Book.Title)
a <- unique(book1$Book.Title)
# the dataset has names of books, their ratings and the users (id) who have rated them.

table(book1$Book.Rating)
prop.table(table(book1$Book.Rating))
#  the ratings are on a scale of  1 to 10
# About 20% of books have been given a rating of 5 or 6, 
# about 40% of # the books have been rated 7 or 8. 
# About 30% of the books have been rated 9 or 10. 

# the histogram also shows that more number of books have been rated 7 or 8
hist(book1$Book.Rating)


title_group <- aggregate(x=book1$Book.Rating,by=list(book1$Book.Title),FUN=mean)
title_group[9585,]
head(title_group)
# pipe command ctrl+shift+M
#the datatype should be realRatingMatrix inorder to build recommendation engine
book1_matrix <- as(book1, 'realRatingMatrix')

##### Recommendation based on popularity
model_popular1 <- Recommender(book1_matrix, method='POPULAR')
recommend_popular1 <- predict(model_popular1, book1_matrix[c(201,230)],n=5)
as(recommend_popular1,'list')
book1[c(201,230),]
# in popular method all readers are given same generic recommendations

##### Recommendation based on User based collaborative filtering
model_ubcf1 <- Recommender(book1_matrix, method='UBCF')
recommend_ubcf1 <- predict(model_ubcf1, book1_matrix[c(201,230)],n=5)
as(recommend_ubcf1,'list')

# can also give top 3 from the predicted one
recommend_ubcf1_top3 <- bestN(recommend_ubcf1,n=3)
as(recommend_ubcf1_top3,'list')
# since both these users have different tastes they are given specific recommendations

# affinity value for user 201
'''
Now, for the same user "201", let's have a look at the affinity 
value computed for all items we didn't have any value in the original data:
'''
# predicting affinity of all non-rated items
predicted_affinity_201 <- predict(model_ubcf1,book1_matrix[201,],type='ratings')
as(predicted_affinity_201,'list')
# real affinity of user-201
as(book1_matrix[201],'list')

# we can also specify normalization method, minrating
model_ubcf2 <- Recommender(book1_matrix, method='UBCF',
                           param=list(normalize='Z-score',method='Cosine',
                                      nn=5,minRating=1))
recommend_ubcf2 <- predict(model_ubcf2,book1_matrix[c(201,230)],n=5)

as(recommend_ubcf2,'list')
# we get the same recommendations as above

##### Recommendation based on Item based collaborative filtering
# since IBCF is computation-intensive let us consider first 2000 records only

book2 <- book1[1:2000,]
book2_matrix <- as(book2,'realRatingMatrix')
model_ibcf <- Recommender(book2_matrix, method='IBCF')
recommend_ibcf <- predict(model_ibcf, book2_matrix[c(201,230)],n=5)
as(recommend_ibcf,'list')
book2[c(201,230),]

# IBCF considers similar books and ratings and then gives user-specific recommendations

# considering same 2000 records and recommending using popular method
model_popular2 <- Recommender(book2_matrix, method='POPULAR')
recommend_popular2 <- predict(model_popular2, book2_matrix[c(201,230)],n=5)
as(recommend_popular2,'list')

# considering same 2000 records and recommending using UBCF method
model_ubcf2 <- Recommender(book2_matrix, method='UBCF')
recommend_ubcf2 <- predict(model_ubcf2, book2_matrix[c(201,230)],n=5)
as(recommend_ubcf2,'list')

# from above we see that the recommendations are different for different methods

# userId 277427 and UBCF method
# [1] "What to Expect When You're Expecting (Revised Edition)"
# [2] "Fatal Voyage"                                          
# [3] "Only Love (Magical Love)"                              
# [4] "Private Screening"                                     
# [5] "Diamond Spur" 
# 
# userId 277427 and IBCF method
# [1] "Fatal Voyage"              "Flashback"                 "Good Omens"               
# [4] "Il Dio Delle Piccole Cose" "Praying for Sleep"  

############## evaluation
# using evaluation scheme splitting 80% train and 20% validation
set.seed(111)
data_eval <- evaluationScheme(book1_matrix,method='split',
                              train=0.8,given=1)
dim(book1_matrix)
# recommendation model using UBCF
model_train_ubcf <- Recommender(getData(data_eval,'train'),'UBCF')
# predictions on test data for UBCF
# recommend_test_ubcf takes around 5 mins to run
recommend_test_ubcf <- predict(model_train_ubcf,getData(data_eval,'known'),
                               type='ratings')
# error metrics for UBCF
error_ubcf <- calcPredictionAccuracy(recommend_test_ubcf,
                                     getData(data_eval,'unknown'))
error_ubcf
'''
when set.seed(111) for all records, then these are error metrics
    RMSE      MSE      MAE 
2.552483 6.515168 1.896552

when set.seed(111)  for data_eval(1000 records), then these are error metrics
    RMSE      MSE      MAE 
2.045773 4.185185 1.518519 

when set.seed(123)  for data_eval, then these are error metrics
     RMSE       MSE       MAE 
 5.458938 29.800000  5.400000 
 
for 2000 records
    RMSE      MSE      MAE 
2.566928 6.589121 1.991192 

when we set seed to 111 the error is far less than seed(123) '''

######### CONCLUSIONS
'''
We have to recommend books to our users. The dataset contains 10,000 entries.
I have used Recommendation system to give recommendations to the readers.

Here I have used popular method, UBCF and IBCF methods. Here used cosine 
similarity matrix. I have also used evaluation scheme to find the goodness
of our model by finding the error.
'''

