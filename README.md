# recommender
Collaborative Filtering Recommender System based on Susan Li's examples (https://towardsdatascience.com/building-a-collaborative-filtering-recommender-system-with-clickstream-data-dffc86c8c65)
Modified to include a swagger and restful interface to query user interactions and retreive recommendations 


::Repository Contents::
│   .gitignore
│   requirements.txt
│   run.py (used to run the application)
│
├───data
│       shared_articles.zip
│       users_interactions.zip
│
├───recommender
│   │   apicontroller.py (contains all of the flask / swagger code)
│   │   recommender-conf.yml (config file for the recommender engine)
│   │   recommender.py (main recommender engine)
│   │   __init__.py
