import requests

url = "http://localhost:8000/chain"
data = {
    "language": "Spanish",
    "text": "Hello, how are you?"
}

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print("响应结果:", result)
else:
    print("请求失败，状态码:", response.status_code)