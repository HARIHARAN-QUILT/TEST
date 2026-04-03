import base64
import requests

url="http://0.0.0.0:8000/api/document-analyze"

headers={
 "Content-Type":"application/json",
 "x-api-key":"sk_track2_987654321"
}

with open("sample.pdf","rb") as f:
    b64=base64.b64encode(f.read()).decode()

payload={
 "fileName":"sample.pdf",
 "fileType":"pdf",
 "fileBase64":b64
}
r=requests.post(url,headers=headers,json=payload)
# print(payload)
print("-------------------------------------------------------------------------------------------------------------------")

print(r.json())