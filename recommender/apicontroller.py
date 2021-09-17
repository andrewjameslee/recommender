#!flask/bin/python

from flask import Flask, jsonify, Blueprint,request,make_response
from flask_restx import Api, Resource, reqparse
from numpy import true_divide
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
parserapi = reqparse.RequestParser()
parserapi.add_argument('dataformat', type=str, help='can be set to "html". Any other value or left blank will return JSON')


#setup URL routing
@name_space.route("/v1/recommender/person/<int:person_id>")

@name_space.doc(params={'person_id': 'id of person'})
@name_space.doc(parser=parserapi)
class RecommenderClass(Resource):

	def get(self,person_id):
            '''get recommendations for a person'''
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
@name_space.doc(parser=parserapi)
class StatsClass(Resource):

	def get(self,person_id):
            '''get interactions for a person'''        
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


admin_name_space = api_extension.namespace('admin', description='admin endpoint for administration functions')

my_resource_parser = reqparse.RequestParser()
my_resource_parser.add_argument('eventtype', type=str, default='')
my_resource_parser.add_argument('eventstrength', type=float, default='')

#setup URL routing
@admin_name_space.route('/v1/admin/event/counts')
class AdminEventClass(Resource):
    #geteventstrength values
    #puteventstrength values
    #posteventstrengh values
	def get(self):
        
            '''get event counts from dataset for all users by event type'''        
            #get eventcounts from recommender 
            results = recommender.recommender.dataworker().geteventcounts();

            return results


@admin_name_space.route('/v1/admin/event/eventstrength')
@admin_name_space.expect(my_resource_parser)
class AdminEventClass(Resource):
	def get(self):
            '''display event strength values'''        
            #check if variable dataformat exists and is set to html - if not set to none
            dataformat = request.args.get( 'dataformat',default=None)

            #get eventcounts from recommender 
            results = recommender.recommender.dataworker().getstatstrength();
            #work out how to display the results
            if dataformat == 'html':
                response = make_response(results)
                response.headers['Content-type'] = 'text/html'
                return response;

            else:
                return results

	def put(self):
            '''modify event and strength'''   
            eventtype = request.args.get( 'eventtype',default=None)
            eventstrength = request.args.get( 'eventstrength',default=None)
            
            #check if variable dataformat exists and is set to html - if not set to none
            dataformat = request.args.get( 'dataformat',default=None)

            results = recommender.recommender.dataworker().getstatstrength();
            if eventtype not in results:
                return ('cannot modify '+ str(eventtype) + ' not in eventtypes ' + str(results) ) 
            
            #get eventcounts from recommender 
            results = recommender.recommender.dataworker().updateevents(eventtype,eventstrength);
            #work out how to display the results
            if dataformat == 'html':
                response = make_response(results)
                response.headers['Content-type'] = 'text/html'
                return response;

            else:
                return jsonify(results);                

@admin_name_space.route('/v1/admin/recommender/regenerate')
class AdminModelClass(Resource):
	def post(self):
            '''regenerate recommendations'''        
            results = recommender.recommender.dataworker().regeneraterecommendation();
            return results