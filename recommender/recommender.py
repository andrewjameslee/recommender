import pandas as pd
from pandas.core.indexes.base import Index
import scipy.sparse as sparse
import numpy as np
import random
import implicit
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

import yaml
import logging

grouped_df = ''
df = ''
event_type_strength = ''
interactions_df = ''
articles_df = ''
def importdata():
    print("[RECOMMENDATION ENGINE] Importing Data")
    global grouped_df
    global df
    global event_type_strength
    global interactions_df
    global articles_df
    articles_df = pd.read_csv(shared_articles_file)
    interactions_df = pd.read_csv(users_interactions_file)
    articles_df.drop(['authorUserAgent', 'authorRegion', 'authorCountry'], axis=1, inplace=True)
    interactions_df.drop(['userAgent', 'userRegion', 'userCountry'], axis=1, inplace=True)

    articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
    articles_df.drop('eventType', axis=1, inplace=True)

    
    event_type_strength = {
    'VIEW': 1.0,
    'LIKE': 2.0, 
    'BOOKMARK': 3.0, 
    'FOLLOW': 2.0,
    'COMMENT CREATED': 2.0,  
    'DISLIKE': -10.0,
    }
    


sparse_content_person = ''
sparse_person_content = ''
person_vecs = ''
content_vecs = ''
similar = ''

def recommendcore():
    print("[RECOMMENDATION ENGINE] Building model")
    global df
    global grouped_df
    global sparse_content_person
    global sparse_person_content
    global person_vecs
    global content_vecs
    global similar
    global event_type_strength
    df = pd.merge(interactions_df[['contentId','personId', 'eventType']], articles_df[['contentId', 'title','url']], how = 'inner', on = 'contentId')

    df['eventStrength'] = df['eventType'].apply(lambda x: event_type_strength[x])

    #debugging
    if DEBUG == 'True':
        print(df.head(10))
        print(df.info())
        print(df['eventType'].value_counts())
        
    #debugging
    if DEBUG == 'True':
        print(df.head(10))

    df = df.drop_duplicates()
    grouped_df = df.groupby(['personId', 'contentId', 'title','url']).sum().reset_index()

    #debugging
    if DEBUG == 'True':
        print(grouped_df.sample(10))
        print(grouped_df.dtypes)

    grouped_df['title'] = grouped_df['title'].astype("category")
    grouped_df['personId'] = grouped_df['personId'].astype("category")
    grouped_df['contentId'] = grouped_df['contentId'].astype("category")
    grouped_df['person_id'] = grouped_df['personId'].cat.codes
    grouped_df['content_id'] = grouped_df['contentId'].cat.codes

    sparse_content_person = sparse.csr_matrix((grouped_df['eventStrength'].astype(float), (grouped_df['content_id'], grouped_df['person_id'])))
    sparse_person_content = sparse.csr_matrix((grouped_df['eventStrength'].astype(float), (grouped_df['person_id'], grouped_df['content_id'])))

    model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=50)
    alpha = 15
    data = (sparse_content_person * alpha).astype('double')

    # Fit the model
    print("[RECOMMENDATION ENGINE] Fitting model to data")
    model.fit(data)

    content_id = 450
    n_similar = 10

    person_vecs = model.user_factors
    content_vecs = model.item_factors

    content_norms = np.sqrt((content_vecs * content_vecs).sum(axis=1))

    scores = content_vecs.dot(content_vecs[content_id]) / content_norms
    top_idx = np.argpartition(scores, -n_similar)[-n_similar:]
    similar = sorted(zip(top_idx, scores[top_idx] / content_norms[content_id]), key=lambda x: -x[1])

    # Get the trained person and content vectors. We convert them to csr matrices

    person_vecs = sparse.csr_matrix(model.user_factors)
    content_vecs = sparse.csr_matrix(model.item_factors)





def eventcounts():
    return(df['eventType'].value_counts())

def eventstrength():
    print (event_type_strength)
    return(event_type_strength)

def recommend(person_id, sparse_person_content, person_vecs, content_vecs, num_contents=10):
    try:   
    # Get the interactions scores from the sparse person content matrix
            person_interactions = sparse_person_content[person_id,:].toarray()
    # Add 1 to everything, so that articles with no interaction yet become equal to 1
            person_interactions = person_interactions.reshape(-1) + 1
    # Make articles already interacted zero
            person_interactions[person_interactions > 1] = 0
    # Get dot product of person vector and all content vectors
            rec_vector = person_vecs[person_id,:].dot(content_vecs.T).toarray()
    
    # Scale this recommendation vector between 0 and 1
            min_max = MinMaxScaler()
            rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0]
    # Content already interacted have their recommendation multiplied by zero
            recommend_vector = person_interactions * rec_vector_scaled
    # Sort the indices of the content into order of best recommendations
            content_idx = np.argsort(recommend_vector)[::-1][:num_contents]
    
    # Start empty list to store titles and scores
            titles = []
            urls = []
            scores = []

            for idx in content_idx:
        # Append titles and scores to the list
                titles.append(grouped_df.title.loc[grouped_df.content_id == idx].iloc[0])
                urls.append(grouped_df.url.loc[grouped_df.content_id == idx].iloc[0])
                scores.append(recommend_vector[idx])

            recommendations = pd.DataFrame({'title': titles, 'url': urls, 'score': scores})
            
            return recommendations
    except:
            return ("an error has occured")




def make_train(ratings, pct_test = 0.2):
    import random
    test_set = ratings.copy() # Make a copy of the original set to be the test set. 
    test_set[test_set != 0] = 1 # Store the test set as a binary preference matrix
    
    training_set = ratings.copy() # Make a copy of the original data we can alter as our training set. 
    
    nonzero_inds = training_set.nonzero() # Find the indices in the ratings data where an interaction exists
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1])) # Zip these pairs together of item,user index into list

    
    random.seed(0) # Set the random seed to zero for reproducibility
    
    num_samples = int(np.ceil(pct_test*len(nonzero_pairs))) # Round the number of samples needed to the nearest integer
    samples = random.sample(nonzero_pairs, num_samples) # Sample a random number of item-user pairs without replacement

    content_inds = [index[0] for index in samples] # Get the item row indices

    person_inds = [index[1] for index in samples] # Get the user column indices

    
    training_set[content_inds, person_inds] = 0 # Assign all of the randomly chosen user-item pairs to zero
    training_set.eliminate_zeros() # Get rid of zeros in sparse array storage after update to save space
    
    return training_set, test_set, list(set(person_inds))

    content_train, content_test, content_persons_altered = make_train(sparse_content_person, pct_test = 0.2)




def auc_score(predictions, test):
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr)

def calc_mean_auc(training_set, altered_persons, predictions, test_set):
    store_auc = [] # An empty list to store the AUC for each user that had an item removed from the training set
    popularity_auc = [] # To store popular AUC scores
    pop_contents = np.array(test_set.sum(axis = 1)).reshape(-1) # Get sum of item iteractions to find most popular
    content_vecs = predictions[1]
    for person in altered_persons: # Iterate through each user that had an item altered
        training_column = training_set[:,person].toarray().reshape(-1) # Get the training set column
        zero_inds = np.where(training_column == 0) # Find where the interaction had not yet occurred
        
        # Get the predicted values based on our user/item vectors
        person_vec = predictions[0][person,:]
        pred = person_vec.dot(content_vecs).toarray()[0,zero_inds].reshape(-1)
        
        # Get only the items that were originally zero
        # Select all ratings from the MF prediction for this user that originally had no iteraction
        actual = test_set[:,person].toarray()[zero_inds,0].reshape(-1)
        
        # Select the binarized yes/no interaction pairs from the original full data
        # that align with the same pairs in training 
        pop = pop_contents[zero_inds] # Get the item popularity for our chosen items
        
        store_auc.append(auc_score(pred, actual)) # Calculate AUC for the given user and store
        
        popularity_auc.append(auc_score(pop, actual)) # Calculate AUC using most popular and score
    # End users iteration
    
    return float('%.3f'%np.mean(store_auc)), float('%.3f'%np.mean(popularity_auc))   



class dataworker:

    def getuserstats(self, person_id,dataformat=None):
        try:
            #lookup personid64 from personid
            person_id64 = grouped_df.personId.loc[grouped_df['person_id'] == person_id ].iloc[0]
            #get stats from personid
            stats = df.loc[df['personId'] == person_id64 ]
        
            if dataformat == 'html':
                return stats.to_html()
            else:
                return stats.to_json(orient ='index')
        except (IndexError) as e:
            print('EXCEPTION RAISED: ' + str(e))
            print(e.args)
            return ('cannot return data stats for person_id ' + str(person_id) )     
          

    def getrecommend(self, person_id,dataformat=None):
        try:
            recommendations = recommend(person_id, sparse_person_content, person_vecs, content_vecs)
            if dataformat == 'html':
                return recommendations.to_html()
            else:
                return recommendations.to_json(orient ='index')
        
        except (IndexError, AttributeError) as e:
            print('EXCEPTION RAISED: ' + str(e))
            print(e.args)
            return ('cannot return recommendations for person_id ' + str(person_id) ) 

    def getstatstrength(self):
        try:
            stats = eventstrength()
            return stats
        
        except (IndexError, AttributeError) as e:
            print('EXCEPTION RAISED: ' + str(e))
            print(e.args)
            return ('cannot return eventstrength' ) 

    def updateevents(self,eventtype,eventstrength):
        try:
            global event_type_strength
            event_type_strength[eventtype] = float(eventstrength)
            return event_type_strength
        
        except (IndexError, AttributeError) as e:
            print('EXCEPTION RAISED: ' + str(e))
            print(e.args)
            return ('cannot update event type and strength' ) 

    def geteventcounts(self):
        try:
            stats = eventcounts()
            return stats.to_json(orient ='index')
        
        except (IndexError, AttributeError) as e:
            print('EXCEPTION RAISED: ' + str(e))
            print(e.args)
            return ('cannot return eventcounts' )

    def regeneraterecommendation(self):
        try:
            results = recommendcore()
            return  ('recommendations have been regenerated' )
        
        except (IndexError, AttributeError) as e:
            print('EXCEPTION RAISED: ' + str(e))
            print(e.args)
            return ('cannot regenerate recommendations' )

print("[RECOMMENDATION ENGINE] Initialising")
with open('recommender/recommender-conf.yml') as stream:
    config = yaml.safe_load(stream)

DEBUG = config['DEBUG']
shared_articles_file = config['shared_articles_file']
users_interactions_file = config['users_interactions_file']

if DEBUG == 'True':
    print('DEBUG set to', DEBUG)

importdata()
recommendcore()

#debugging
if DEBUG == 'True':
    person_id = 1
    recommendations = recommend(person_id, sparse_person_content, person_vecs, content_vecs)
    print('recommendations for', person_id)
    print(recommendations)
    print(grouped_df.loc[grouped_df['person_id'] == 1].sort_values(by=['eventStrength'], ascending=False)[['title', 'url', 'eventStrength', 'person_id']])

if DEBUG == 'True':
    print('similar content')
    for content in similar:
        idx, score = content
        print(grouped_df.title.loc[grouped_df.content_id == idx].iloc[0])
        
if DEBUG == 'True':
    print("[RECOMMENDATION ENGINE] store_auc and popularity_auc are", calc_mean_auc(content_train, content_persons_altered,
              [person_vecs, content_vecs.T], content_test))

    #debugging
    if DEBUG == 'True':
        print(articles_df['eventType'].value_counts())

    #debugging
    if DEBUG == 'True':
        print(articles_df.info())
        print(interactions_df.info())        