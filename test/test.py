import requests

# resp = requests.post("http://localhost:5000/predict", files={'file': open('test/dog.jpg', 'rb')})
resp = requests.post("https://image-classifier-flask-demo.herokuapp.com/predict", files={'file': open('cat.jpg', 'rb')})
# https://image-classifier-flask-demo.herokuapp.com/

print(resp.text)
