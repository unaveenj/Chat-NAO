from ibm_watson import ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from pandas.io.json import json_normalize

import json
#Authentication
API_KEY = "GZT7tn7BpHaDVYvDqj1dLX5i4RhHgTVYRfgQ3D_Xfvrv"
URL = "https://api.au-syd.tone-analyzer.watson.cloud.ibm.com/instances/b404c5c1-fbcd-409c-97c5-326612bcbdab"
authenticator = IAMAuthenticator(API_KEY)
ta = ToneAnalyzerV3(version='2020-10-24', authenticator=authenticator)
ta.set_service_url(URL)

#Analyse tone
# res = ta.tone('I feel great, its sunny outside, and I have got all my work done. But still, this sucks, I have like 500 hundred hours more coding to do. This is going to take all weekend').get_result()
res = ta.tone("My dog passed away and i feel sad").get_result()
print(res)
# for i in range(len(res["sentences_tone"])):
#     print(json.dumps(res["sentences_tone"][i]["tones"][0]["tone_name"], indent=2))
#     print(json.dumps(res["sentences_tone"][i]["tones"][0]["score"], indent=2))

# mydata = json_normalize(res['sentences_tone'])
# print(mydata)

tones_data = json_normalize(data=res['sentences_tone'], record_path='tones')
tones_data=tones_data.sort_values(by=['score'], ascending=False)
print(tones_data)
if '' in tones_data['tone_name'].tolist():
    print("I am goin to dance for you")

