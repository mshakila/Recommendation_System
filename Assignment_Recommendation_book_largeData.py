#### ASSIGNMENT RECOMMENDATION ENGINE - larger book dataset
'''

The book dataset contains 1.1 million ratings of 270,000 books by 90,000 users. 
The ratings are on a scale from 1 to 10, 
also has 0 rating which indicates implicit rating, I have removed this 
because there are other methods to recommend based on implicit ratings.

The dataset is downloaded from this below link
http://www2.informatik.uni-freiburg.de/~cziegler/BX/
'''
# Business Problem: To provide interesting book recommendations to readers

#importing library for Recommendation system
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
pd.set_option('display.max_columns',10)

# loading books dataset
books = pd.read_csv("E:/Recommendation_System/BX-CSV-Dump/BX-Books.csv",
                   sep=";", encoding="ISO-8859-1", error_bad_lines=False)
books.columns
books.columns = ['ISBN', 'BookTitle', 'BookAuthor', 'YearOfPublication', 
                 'Publisher', 'ImageURLS', 'ImageURLM', 'ImageURLL']
books.head(3)
books.shape #  271,360 books

# loading books_rating dataset
ratings_raw = pd.read_csv("E:/Recommendation_System/BX-CSV-Dump/BX-Book-Ratings.csv",
                   sep=";", encoding="latin-1", error_bad_lines=False)
ratings_raw.columns
ratings_raw.columns = ['UserID', 'ISBN', 'BookRating']
ratings_raw.head()
ratings_raw.shape # 1,149,780  ratings

''' it has implicit rating which is denoted by 0. For recommendation using
implicit ratings, different methods can be used. Here we are not using
implicit ratings. so we are removing them '''
ratings = ratings_raw[ratings_raw['BookRating']>0]
ratings.head()
ratings.shape
# now we have 433,671 explicit ratings
ratings['BookRating'].describe() 
# it ranges from 1-to-10 and mean rating is 7.60

# plotting the distribution of rating
plt.rc("font",size=15)
ratings.BookRating.value_counts().plot(kind='bar');plt.xlabel('Rating');plt.ylabel('Count')
# maximum of the books have been rated 8

########### RECOMMENDATION BASED ON RATING COUNT
rating_count = pd.DataFrame(ratings.groupby('ISBN')['BookRating'].count())
rating_count_sorted=rating_count.sort_values('BookRating',ascending=False)

# top 5 popular books based on count
top_5_books = rating_count_sorted.iloc[0:6,]
top_5_books_recommend = pd.merge(top_5_books,books,on='ISBN')
top_5_books_recommend
'''
The top 5 books recommended based on high rating-count are:
1. The Lovely Bones
2. Wild Animus
3. The Da Vinci Code
4. The Red Tent
5. Divine Secrets of the Ya-Ya Sisterhood
'''

########### RECOMMENDATION BASED ON HIGHEST MEAN RATING 
rating_mean = pd.DataFrame(ratings.groupby('ISBN')['BookRating'].mean())
rating_mean_sorted=rating_mean.sort_values('BookRating',ascending=False)
rating_mean_sorted.head()
# top 5 popular books based on count
top_5_books_rating_mean = rating_mean_sorted.iloc[0:6,]
top_5_books_rating_mean_recommend = pd.merge(top_5_books_rating_mean,books,on='ISBN')
top_5_books_rating_mean_recommend
pd.merge(top_5_books_rating_mean_recommend,rating_count,on='ISBN').head()
''' 
The top 5 books recommended based on high rating-count are:
1. Alone: The Classic Polar Adventure
2. Auschwitz : A Doctor's Eyewitness Account
3. La Conciencia Sin Fronteras
4. The Loser
5. Intangible Evidence  
It shows top 5 books with mean rating of 10. when we look at the last
column of count (BookRating_y), it is showing rating of just 1. so only by
considering high rating, we cannot give proper recommendations. '''

 
######## RECOMMENDATION BASED ON Item-Item collaborative filtering using CORRELATION
# getting both mean-rating and rating-count in one dataset
average_rating = pd.DataFrame(ratings.groupby('ISBN')['BookRating'].mean())
average_rating['RatingCount'] = pd.DataFrame(
    ratings.groupby('ISBN')['BookRating'].count())
average_rating.sort_values('RatingCount',ascending=False).head(6)
''' now when  we look at both count and average-rating, we see that the 
book in 2nd position has just a average-rating of 4.0, so when we look at
only high counts or high ratings, we are liable to make mistakes. 
'''

## Filtering books rated by atleast a few users  and users who have rated atleast a few books

''' using users who have rated atleast 50 (times) books, we get 
175023 (ratings) records '''
counts_user = ratings['UserID'].value_counts()
ratings_new = ratings[ratings['UserID'].isin(counts_user[counts_user >=50].index)]

''' of above 175023 rcords, we are filtering books that have been rated 
by atleast 10 users. we get 26698 (ratings) records '''
counts_book = ratings_new['ISBN'].value_counts()
ratings_new = ratings_new[ratings_new['ISBN'].isin(counts_book[counts_book >=10].index)]

######## getting rating matrix for above data using Item-Item similarity
ratings_pivot_new = ratings_new.pivot(index='UserID',columns='ISBN').BookRating
ratings_pivot_new.shape
# 1257 users, 1497 books
ratings_pivot_new.head()
''' mostly sparse matrix since many users would not have rated all books'''

# let us select a book and give recommendation 
# ISBN 0385504209 is for book "The Da Vinci Code"  
DaVinciCode_ratings = ratings_pivot_new['0385504209']
similar_to_DaVinciCode = ratings_pivot_new.corrwith(DaVinciCode_ratings)
type(similar_to_DaVinciCode)
corr_DaVinciCode = pd.DataFrame(similar_to_DaVinciCode,columns=['Pearson_corr'])

# removing missing values i.e., books that have not been rated
corr_DaVinciCode.dropna(inplace=True) 


# getting both mean-rating and rating-count in one dataset
user_rating_count = pd.DataFrame(ratings.groupby('UserID')['BookRating'].count())
average_rating = pd.DataFrame(ratings.groupby('ISBN')['BookRating'].mean())
average_rating['RatingCount'] = pd.DataFrame(
    ratings.groupby('ISBN')['BookRating'].count())
average_rating.sort_values('RatingCount',ascending=False).head(6)


corr_DaVinciCode_summary = corr_DaVinciCode.join(average_rating['RatingCount'])
corr_DaVinciCode_summary.head()


DaVinciCode_recomm =pd.merge(corr_DaVinciCode_summary,books,on='ISBN')
DaVinciCode_recomm.drop(DaVinciCode_recomm.columns[[7,8,9]],axis=1,inplace=True)

# Books similar to DaVinciCode that have high corr as well more than 100 rating count
DaVinciCode_recomm_topN = DaVinciCode_recomm[DaVinciCode_recomm['RatingCount']>=100].sort_values(
    'Pearson_corr', ascending=False)
DaVinciCode_recomm_topN.head(5)

''' 
The top 5 books recommended based on correlation: Item-Item-collaborative filtering
1. Silence of the Lambs 
2. Harry Potter and the Prisoner of Azkaban (Book 3)
3. The Da Vinci Code 
4. The Divine Secrets of the Ya-Ya Sisterhood: A 
5. The Beach House '''

DaVinciCode_recomm_topN.tail()
''' The Hours, Little Altars everywher, etc
these books  have high negative corr with DaVinciCode. 
These should not at all be recommended to readers of DaVinciCode '''

############ CONCLUSIONS
'''
We have to recommend books to our users. The dataset contains 1.1 million 
ratings of 270,000 books by 90,000 users.

I have used Recommendation system to give recommendations to the readers.

I have given recommendations based on highest rating count, highest mean 
rating. I used IBCF for a single book "The Da Vinci Code" (as computation 
intensive if do for all books) using correlation similarity matrix. I have done
filtering to improve the recommendations.
'''


