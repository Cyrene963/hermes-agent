import urllib.request
import re
from bs4 import BeautifulSoup
import sys

url = "https://cloud.google.com/blog/products/ai-machine-learning/claude-mythos-preview-on-vertex-ai"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    html = urllib.request.urlopen(req).read().decode('utf-8')
    date_match = re.search(r'"datePublished":"(.*?)"', html)
    if date_match:
        print("Google Date:", date_match.group(1))
    
    title_match = re.search(r'<title>(.*?)</title>', html)
    if title_match:
        print("Google Title:", title_match.group(1))
except Exception as e:
    print(e)
