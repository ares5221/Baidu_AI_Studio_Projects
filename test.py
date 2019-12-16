#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import requests

url = 'https://mp.weixin.qq.com/s/ojHFMpPwThGsc5jGZMnLOg'
useragent = 'Mozilla/5.0 (Linux; U; Android 4.4.2; zh-cn; PE-TL20 Build/HuaweiPE-TL20) AppleWebKit/537.36 (KHTML, like Gecko)Version/4.0 MQQBrowser/5.3 Mobile Safari/537.36'
hd = {'user-agent': useragent}
ss = requests.get(url, headers=hd)
con = ss.text
print(con)


