import json
import ssl
import urllib
from urllib import parse, request


def 小麦病害识别(服务器路径):
    # 识农API
    host = 'https://senseagro.market.alicloudapi.com'
    # url方式
    path = '/api/senseApi'
    # 阿里云appcode
    appcode = '342414c27ad142fb9b9068d1aa508cba'
    url = host + path

    header = {
        "User-Agent": "Opera/9.80 (Windows NT 6.1; U; en) Presto/2.7.62 Version/11.01",
        'Authorization': 'APPCODE ' + appcode,
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
    }

    formData = {
        'crop_id': '''132''',
        'image_url': 服务器路径
    }

    data = urllib.parse.urlencode(formData).encode("utf-8")

    request = urllib.request.Request(url, data=data, headers=header)

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    返回体 = {
        "状态": "正常",
        "错误信息": "",
        "识别结果": "XXX",
        "准确率": "0",
    }
    try:
        response = urllib.request.urlopen(request, context=ctx)
        content = response.read().decode("utf-8")
        if (content):
            content = json.loads(content)
            if content["content"].get("msg") == None:
                返回体["识别结果"] = content["content"]["result"]
                返回体["准确率"] = content["content"]["score"]
            else:
                返回体["状态"] = "错误"
                if content["status"] == "1":
                    返回体["状态"] = "图片非小麦"
                返回体["错误信息"] = content["content"].get("msg")
    except BaseException:
        返回体["状态"] = "错误"
        返回体["错误信息"] = "识农接口504请求超时"
    return 返回体


if __name__ == '__main__':
    图片url = "https://i.loli.net/2021/03/18/MNvxCrwtOZHjkmh.jpg"
    图片url = "http://ldttwebserver.chinanorth2.cloudapp.chinacloudapi.cn/static/images/logo2.jpg"
    print(小麦病害识别(图片url))
