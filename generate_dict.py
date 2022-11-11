'''
    This generates dictionary containing 10 users and their corresponding train set, test set and top recommendations
    Author: Bipin Koirala
'''
import os
import json
import pandas as pd
from tqdm import tqdm
from script import GET_RECOMMENDATION
import time

start = time.time()

# Initialize book title:
books_title = pd.read_json('./data/books_title.json')
books_title['book_id'] = books_title['book_id'].astype(str)

# file path for 10 test users
path = './data/users/'

# store test-train-top recommendation for each user. key = user_xxx
dictionary = {} 

for file in tqdm(os.listdir(path), total = 10, desc = 'Processing', mininterval = 0.2):

    file_path = os.path.join(path, file)
    user_info = pd.read_csv(file_path)

    # construct model class
    recommend = GET_RECOMMENDATION(user = user_info, book_title = books_title, percentage = 25, split_size = 0.3, current_user = file[:-4])

    csv_book_map = recommend.book_mapping()
    filtered_users = recommend.similar_users(book_map = csv_book_map)
    interaction_list = recommend.interactions(filtered_users, csv_book_map)

    top_recommendations = recommend.collab_filter(interaction_list)
    top_recommendations = top_recommendations['book_id'].to_numpy().astype(float)
    
    # create dictionary entry per user for evaluation purpose
    train_data = recommend.get_train_set()
    train_data = train_data['book_id'].to_numpy().astype(float)

    test_data = recommend.get_test_set()
    test_data = test_data['book_id'].to_numpy().astype(float)

    print ('Creating dictionary for ' + file[:-4])
    dictionary[file[:-4]] = [test_data, train_data, top_recommendations]

# save dictionary as JSON file
with open('./data/dict/top_recs.json', 'w') as f:
    json.dump(dictionary, f)

print ('Finished Process in {:.3f} min'.format((time.time() - start)/60.0))
# To load this
# with open('./data/dict/top_recs.json') as file:
#       data = json.load(file)