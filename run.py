from recommender import app
import recommender.recommender
from flask import make_response

#start the main app
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=80,debug=True)
