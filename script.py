import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

'''With hold some of the row entries for testing'''
from sklearn.model_selection import train_test_split

from sklearn.metrics.pairwise import cosine_similarity

class GET_RECOMMENDATION(object):

    def __init__(self, user, book_title, percentage, split_size, current_user) -> None:
        '''user = user_info, 
        books_title,
        Find users who've read x % of books that you've read
        split_size = train-test split float 0-1'''
        self.user = user # pandas.core.frame.DataFrame
        self.user['book_id'] = user['book_id'].astype(str)
        self.train, self.test = train_test_split(self.user, test_size = split_size, random_state = 2053)                # TODO: perhaps its better to remove seed?
        self.user_books = self.train
        self.books_title = book_title # read books_title.json out this module
        self.books_title['book_id'] = self.books_title['book_id'].astype(str)
        self.k = 100//percentage
        self.user_name = current_user





    def book_mapping(self):
        '''Maps book_id to csv_id'''
        csv_book_mapping = {}
        with open('./data/book_id_map.csv', 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                csv_id, book_id = line.strip().split(',')
                csv_book_mapping[csv_id] = book_id
        return csv_book_mapping




    def similar_users(self, book_map):
        ''' Args: csv_book_mapping from book_mapping'''
        
        print ('Finding similar users...')

        book_set = set(self.user_books['book_id'])
        overlap_users = {}

        # Add users in overlap_users if they have read the same books as you
        with open('./data/goodreads_interactions.csv') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                user_id, csv_id, _, rating, _ = line.strip().split(',')

                book_id = book_map.get(csv_id)
                if book_id in book_set:
                    if user_id not in overlap_users:
                        overlap_users[user_id] = 1
                    else:
                        overlap_users[user_id] += 1
        # Total users who have read the same books as you
        print ('Total users who\'ve read the same books as ' + self.user_name + ': {}'.format(len(overlap_users)))

        # Users who have read at least 20% of the books that you've read
        filtered_overlap_users = set([k for k in overlap_users if overlap_users[k] > self.user_books.shape[0]/self.k])
        print ('Total users who\'ve read at least {}% of the books that '.format(100/self.k) + self.user_name + ' has read: {}'.format(len(filtered_overlap_users)))

        return filtered_overlap_users




    def interactions(self, filtered_users, book_map):
        '''args: set > filtered_overlap_users from similar_users function
               csv_book_mapping_'''
        interactions_list = []
        with open('./data/goodreads_interactions.csv') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                user_id, csv_id, _, rating, _ = line.strip().split(',')

                if user_id in filtered_users:
                    book_id = book_map[csv_id]
                    interactions_list.append([user_id, book_id, rating])

        print ('Total books in filtered user list: {}\n'.format(len(interactions_list)))

        return interactions_list




    def collab_filter(self, interaction_list):
        '''args: interactions_list'''
        '''
        Build Collaborative Filtering Features i.e. user-book matrix containing ratings in each cell
        '''
        # Create a dataframe for the above list
        print ('Applying Collaborative Filtering')

        interactions = pd.DataFrame(interaction_list, columns = ['user_id', 'book_id', 'rating'])

        # Concatenate your book ratings to the ratings from everyone else
        interactions = pd.concat([self.user_books[['user_id', 'book_id', 'rating']], interactions])
        interactions['book_id'] = interactions['book_id'].astype(str)
        interactions['user_id'] = interactions['user_id'].astype(str)
        interactions['rating'] = pd.to_numeric(interactions['rating'])
        # We want each user_id to correspond to a single row in a matrix. So we're assigning index to each of these user_ids 
        interactions['user_index'] = interactions['user_id'].astype('category').cat.codes
        # We do the same with book_id so that we can create a matrix user_index X book_index
        interactions['book_index'] = interactions['book_id'].astype('category').cat.codes

        print ('Number of unique users: {}\nNumber of unique books: {}'.format(len(interactions['user_index'].unique()), len(interactions['book_index'].unique())))

        matrix = (interactions['rating'], (interactions['user_index'], interactions['book_index']))
        ratings_mat_coo = coo_matrix(matrix)
        ratings_mat = ratings_mat_coo.tocsr()  # Convert to compressed sparse row format

        interactions[interactions['user_id'] == '-1']
        my_index = 0  # Corresponds to row 0 of the matrix i.e. your information

        # Use Cosine Similarity to find user who are similar to us

        similarity = cosine_similarity(ratings_mat[my_index,:], ratings_mat).flatten()

        # Find indices of top n users who are more similar to us
        indices = np.argpartition(similarity, -10)[-10:]

        # Find their user_id
        similar_users = interactions[interactions['user_index'].isin(indices)].copy()

        # Remove all entries that corresponds to you
        similar_users = similar_users[similar_users['user_id'] != '-1']

        book_recs = similar_users.groupby('book_id').rating.agg(['count', 'mean'])

        books_recs = book_recs.merge(self.books_title, how = 'inner', on = 'book_id')
        books_recs["adjusted_count"] = books_recs["count"] * (books_recs["count"] / books_recs['ratings'])

        books_recs['score'] = books_recs['mean'] * books_recs['adjusted_count']
        books_recs = books_recs[~books_recs['book_id'].isin(self.user_books['book_id'])] # take out books that you've already read

        self.user_books['mod_title'] = self.user_books['title'].str.replace('[a-zA-Z0-9 ]', '', regex = True).str.lower()
        self.user_books['mod_title'] = self.user_books['mod_title'].str.replace('\s+', ' ', regex = True)

        book_recs = books_recs[~books_recs['mod_title'].isin(self.user_books['mod_title'])]
        book_recs = book_recs[book_recs['count']>2]                                                                     # TODO: possible hyperparameter
        book_recs = book_recs[book_recs['mean']>4]                                                                      # TODO: possible hyperparameter

        top_recs = book_recs.sort_values('score', ascending = False)

        return top_recs



    def show_img(x):
        return '<img src="{}" width=50></img>'.format(x)



    def get_train_set(self):
        return self.train



    def get_test_set(self):
        return self.test