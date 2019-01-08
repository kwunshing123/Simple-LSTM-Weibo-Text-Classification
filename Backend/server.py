import jieba as jb
import numpy as np
import tensorflow as tf
import urllib.request as urlreq
from keras.models import model_from_json
from flask import Flask, Response, request, json
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)
bingMapsKey = "AlazyxVPvdz9Nv1jkdNDeik5C2SLUXbj0muL9RcyaoSqtmg0p0I6AYrhWW1wjwBK"
  
class Classifier(Resource):
    def get(self):
        # load json and create model
        json_file = open('keras_lstm.json', 'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        # load weights into new model
        model.load_weights("keras_lstm.h5")
        print("Loaded model from disk")
        
        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(30, min_frequency=1)
        
        sen = request.args.get('content')
        sen_prosessed = " ".join(jb.cut(sen, cut_all=True))
        sen_prosessed = vocab_processor.transform([sen_prosessed])
        sen_prosessed = np.array(list(sen_prosessed))
        result = model.predict(sen_prosessed)
        
        catalogue = list(result[0]).index(max(result[0]))
        threshold=0.8
        if max(result[0]) > threshold:
            if catalogue == 0:
                category = "This is an article about health"
            elif catalogue == 1:
                category = "This is an article about technology"
            elif catalogue == 2:
                category = "This is an article about design"
        else:
          category = "Cannot classify" 
        data = {'input': sen, 'data': category, 'health_prob': float(result[0][0]), 'tech_prob': float(result[0][1]), 'design_prob': float(result[0][2])}
        result = json.dumps(data)
        return Response(result)
    
class LocationsIdentify(Resource):
    def get(self):
        city = request.args.get('city')
        routeUrl = "http://dev.virtualearth.net/REST/V1/Locations?q=" + str(city) + "&o=json&key=" + bingMapsKey
        req = urlreq.Request(routeUrl)
        response = urlreq.urlopen(req)
        json_res = json.loads(response.read())['resourceSets'][0]['resources'][0]['point']
        result = json.dumps(json_res)
        return Response(result)
        
api.add_resource(Classifier, '/weibo')
api.add_resource(LocationsIdentify, '/city')

if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(port=5002)