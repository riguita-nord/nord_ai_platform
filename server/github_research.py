import json
import urllib.request

REPOS = {'qbcore': 'qbcore-framework/qb-core'}

def recent(question):
    q = question.lower()
    if 'qbcore' not in q and 'qb-core' not in q:
        return None
    repo = REPOS['qbcore']
    req = urllib.request.Request('https://api.github.com/repos/'+repo+'/commits?per_page=3',headers={'User-Agent':'NordAI-Research','Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(req,timeout=10) as response:
        commits = json.load(response)
    results=[]
    for item in commits[:3]:
        info=item.get('commit',{})
        results.append({'date':info.get('committer',{}).get('date',''),'message':info.get('message','')[:250],'url':item.get('html_url','')})
    return results
