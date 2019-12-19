from flask import Flask, render_template,request, Response
from flask_restful import Resource, Api
from tensorflow.keras.backend import get_session, set_session
import base64
import os
from PIL import Image
import numpy as np
from skimage import transform
import numpy as np
import re
import sys 
import os
from io import BytesIO
sys.path.append(os.path.abspath("./model"))
from load import * 
app = Flask(__name__)
api=Api(app)
global model, graph, sess
model, graph, sess = init()

# def convertImage(imgData):
# 	# print(type(imgData))
# 	# imgstr = re.search(b'base64,(.*)', imgData).group(1)

# 	with open('output.png','wb') as output:
# 		output.write(base64.decodebytes(imgData.encode()))
	


class index(Resource):
	def get(self):
		return {'hello':'world'}


class test(Resource):
	def get(self):
		return {'result':'test passed'}


class predict(Resource):
	def get(self):
		return {'result':'Predictor is working'}
	def post(self):
		content=request.json
		
		# convertImage(content['image'])
		
		x = Image.open(BytesIO(base64.b64decode(content['image']))).convert('L')
		
		x = np.array(x).astype('float32')/255
		x = transform.resize(x,(224,224,3))
		x = np.expand_dims(x,axis=0)

		with graph.as_default():
			set_session(sess)
			out = model.predict_classes(x)
			confidence=round(np.max(model.predict_proba(x))*100)
			# prediction=np.array2string(out)
			prediction=out.item()

			return {'prediction':prediction,'confidence':confidence}
			# response = np.array_str(out) 
			# print("Predicted {} with a confidence of {}".format(np.array_str(out),np.max(model.predict_proba(x))))
			# if (confidence>50):
			# 	return "Predicted {} with a confidence of {}%".format(prediction,confidence)
			# else:
			# 	return "Not sure"
	
api.add_resource(index,'/')
api.add_resource(test,'/test')
api.add_resource(predict,'/predict')


# if __name__ == "__main__":
# 	port = int(os.environ.get('PORT', 5000))
# 	app.run(host='0.0.0.0', port=port)
