#!flask/bin/python

from flask import Flask, jsonify, Blueprint,request,make_response
from flask_restx import Api, Resource, reqparse
import recommender

#initialise the API and namespace
apicontroller = Blueprint('apicontroller', __name__)

api_extension = Api(apicontroller,
    title='Recommender API',
    version='1.0',
    description='The recommender application is an API for retreiving user web page recommendations based on previous interactons and machine learning',
    doc='/'
    )

name_space = api_extension.namespace('api', description='the data endpoint returns information on users interactions where the recommendation endpoint returns recommendations for the user')

#setup parser for dataformat variable
parser = reqparse.RequestParser()
parser.add_argument('dataformat', type=str, help='can be set to "html". Any other value or left blank will return JSON')

#setup URL routing
@name_space.route("/v1/recommender/person/<int:person_id>")
@name_space.doc(params={'person_id': 'id of person'})
@name_space.doc(parser=parser)
class RecommenderClass(Resource):
	def get(self,person_id):
            #check if variable dataformat exists and is set to html - if not set to none
            dataformat = request.args.get( 'dataformat',default=None)  
            #get recommendations from recommender 
            results = recommender.recommender.dataworker().getrecommend(person_id,dataformat)
            #work out how to display the results
            if dataformat == 'html':
                response = make_response(results)
                response.headers['Content-type'] = 'text/html'
                return response
            else:
                return jsonify(results)

#setup URL routing
@name_space.route("/v1/data/person/<int:person_id>")
@name_space.doc(params={'person_id': 'id of person'})
@name_space.doc(parser=parser)
class StatsClass(Resource):

	def get(self,person_id):
		    #check if variable dataformat exists and is set to html - if not set to none
            dataformat = request.args.get( 'dataformat',default=None)
            #get userstats from recommender 
            results = recommender.recommender.dataworker().getuserstats(person_id,dataformat)
            #work out how to display the results
            if dataformat == 'html':
                response = make_response(results)
                response.headers['Content-type'] = 'text/html'
                return response
            else:
                return jsonify(results)


