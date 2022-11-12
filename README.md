# Book Recommendation System

## (As a part of ML-Group Project)
This branch deals with Collaborative Filtering approach on recommending books for user.

### Dataset

We make use of GoodReads Dataset available on [UCSD Book Graph](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home). 

### Before you begin

- Download [Complete Book Graph](https://drive.google.com/uc?id=1LXpK1UfqtP89H1tYy0pBGHjYk8IhigUK) and save it inside `./data/` folder in root directory
- Download [Book IDs](https://drive.google.com/uc?id=1CHTAaNwyzvbi1TR08MJrJ03BxA266Yxr) and [Complete User-Book Interactions.csv](https://drive.google.com/open?id=1zmylV7XW2dfQVCLeg1LbllfQtHD2KUon) in the same folder
- Create a folder named `dict` inside `./data/` & move `users` inside `./data/`

Some parts of the code has been adapted from [VikParuchuri](https://github.com/dataquestio/project-walkthroughs/tree/master/books)

---

### Collaborative Filtering
This is commonly adapted technique in when building various recommendation systems. Collaborative filtering primarily uses similarity metric (usually cosine similarity) between users and items simultaneously to provide recommendation based on specific input user. It can be regarded as the process of evaluating/filtering items based on the opinions of other people i.e. 'word of mouth' phenomenon. It is regarded as unsupervised learning because no domain knowledge is necessary to recommend items to the user as the embeddings are automatically learned during the recommendation process. 

We use the following approach in building collaborative filtering:-
* Find similar users
  
  Given a list of liked books pertaining to a specific user; we find other users who've read at least x % of those books. We explore the performance of the model at various percentages 20% and 35%.
  
* Create User-Book interaction matrix
  
  We then proceed to create a user-book matrix containing ratings in each cell. Each of the users only interact with a very small subset of the entire dataset and therefore, we have a very huge matrix with entries mostly filled up in zeros. From computational cost perspective, this take a huge amount of time and memory to process the matrix. Therefore, we converse this dense matrix into a sparse matrix for further data processing.
  
* Cosine Similarity

  For a given user, we compute cosine similarity between them and all the other users in order to find users who have similar taste in books. We used top 10 users based on cosine similarity score and explore books they've read. Book recommendation is done based on a <b>score</b> which is computed as below: 
 
  `score = mean x adjusted_count` <br> 
  `adjusted_count = count x (count/ratings)` <br>
  
  Here, <b>mean</b> is an average rating out of 5 and <b>ratings</b> is the total number of rating the book received. <b>adjusted count</b> quantifies the number of times a specific book appeared among the users with similar taste relative to other users. <br>
  
Below is a snapshot of top 7 books recommended based on the preference of a test user.


![Figure](https://github.com/b53k/Book-Recommendation-System/blob/main/figs/recs.png)

---
We randomly selected 10 user with their top 40 liked books (might increase the number at later stage of the project) to see how good the model is when it comes to recommending new books. For this purpose, we split a given instance into 70% train-set and 30% test-set. So the test-set essentially contains 12 books. Recommended books were well-above 12.

Figure below shows the performance of the model when selecting other users who've read at least 35% of books for a given user. Here, the cut-off mean rating for recommended book is 4 i.e. only books that have mean rating of 4 or above is shown.

![Figure1](https://github.com/b53k/Book-Recommendation-System/blob/main/figs/35%25%20result.jpg)

For other users who've read at least 20% of books for a given user and the cut-off mean rating at 3.5 we can see slight improvement in performance.

![Figure2](https://github.com/b53k/Book-Recommendation-System/blob/main/figs/20%25%20result.jpg) 

The apparent improvement in performance can be attributed to the fact that when selecting other users who have read at least 20% of the books and selecting books with average rating of 3.5 or above; the model explores more number of books and the probability of a book being in both testing dataset and recommended books is higher.
