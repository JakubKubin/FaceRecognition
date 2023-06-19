import requests

url = 'http://localhost:8080/get_dict_emotions'
data = {'path': "full\\path\\to\\picture"}

response = requests.post(url, json=data)

print(response.json())

url = 'http://localhost:8080/get_dict_backgrounds'
data = {'path': "full\\path\\to\\picture"}

response = requests.post(url, json=data)

print(response.json())

url = 'http://localhost:8080/get_recognized_faces'
data = {'path1': "full\\path\\to\\picture1", 'path2': "full\\path\\to\\picture"}

response = requests.post(url, json=data)

print(response.json())