# -*- coding: utf-8 -*-

import requests
import re

def Get_image_url(headers, key):
    print("正在获取图片下载链接......")
    url = 'https://image.baidu.com/search/index?tn=baiduimage&ie=utf-8&word={}'.format('%s' % key)
    request = requests.get(url, headers=headers, timeout=10)  # timeout实现网页未返回值的情况
    # 获取网页的代码
    image_urls = re.findall('"objURL":"(.*?)",', request.text, re.S)
    # 获取图片的下载链接，列表形式
    if not image_urls:
        print("Error:图片下载链接获取失败！")
    else:
        return image_urls

def Write_image(image_urls, num):
    for i in range(0, num):
        image_data = requests.get(image_urls[i])
        # 获取下载链接中的图片信息
        print("正在下载第%s张图片：" % (i + 1))
        image_path = "C:/Users/lmx/Desktop/虫/%s.jpg" % (i + 1)
        # 保存图片的路径，可以修改
        with open(image_path, 'wb') as fp:
            fp.write(image_data.content)
            # 写入图片信息


if __name__ == '__main__':
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome'
                      '/112.0.0.0 Safari/537.36 Edg/112.0.1722.58',
        'Host': 'image.baidu.com',
        'Cookie': 'BIDUPSID=6096EFD12C571F1D6231034147921FB8; PSTM=1682383713; BAIDUID=6096EFD12C571F1D5581B126'
                  '79EA8E7D:FG=1; BD_UPN=12314753; BDORZ=FFFB88E999055A3F8A630C64834BD6D0; delPer=0; BD_CK_SAM=1'
                  '; PSINO=5; BDRCVFR[dG2JNJb_ajR]=mk3SLVN4HKm; BAIDUID_BFESS=6096EFD12C571F1D5581B12679EA8E7D:FG'
                  '=1; userFrom=null; BDRCVFR[-pGxjrCMryR]=mk3SLVN4HKm; BDRCVFR[tox4WRQ4-Km]=mk3SLVN4HKm; BDRCVFR'
                  '[A24tJn4Wkd_]=mk3SLVN4HKm; shifen[598151295075_76725]=1682411441; BCLID=10450312018116497963; B'
                  'CLID_BFESS=10450312018116497963; BDSFRCVID=z9_OJeC62lsu0DJfOkenUsu36Pzw6K3TH6bHQI-qy-1kcJagoI4a'
                  'EG0PUx8g0KuMDFkVogKK0eOTHktF_2uxOjjg8UtVJeC6EG0Ptf8g0f5; BDSFRCVID_BFESS=z9_OJeC62lsu0DJfOkenUsu'
                  '36Pzw6K3TH6bHQI-qy-1kcJagoI4aEG0PUx8g0KuMDFkVogKK0eOTHktF_2uxOjjg8UtVJeC6EG0Ptf8g0f5; H_BDCLCKID'
                  '_SF=tbFqoK8bJKL3qJTph47hqR-8MxrK2JT3KC_X3b7Ef-FB_p7_bf--D4Ay5H3RBt592KTX-4OatKQmJ40CyTbxy5KVybQA'
                  'eRo8HR6W3hcq5b7zMbjHQT3m3JvbbN3i-xrR3D3pWb3cWKJq8UbSMnOPBTD02-nBat-OQ6npaJ5nJq5nhMJmb67JD-50exbH5'
                  '5uHtb-e3H; H_BDCLCKID_SF_BFESS=tbFqoK8bJKL3qJTph47hqR-8MxrK2JT3KC_X3b7Ef-FB_p7_bf--D4Ay5H3RBt592K'
                  'TX-4OatKQmJ40CyTbxy5KVybQAeRo8HR6W3hcq5b7zMbjHQT3m3JvbbN3i-xrR3D3pWb3cWKJq8UbSMnOPBTD02-nBat-OQ6n'
                  'paJ5nJq5nhMJmb67JD-50exbH55uHtb-e3H; BDRCVFR[Q5XHKaSBNfR]=mk3SLVN4HKm; BA_HECTOR=0005ah85ah01akak'
                  'ah218kdl1i4f4661m; ZFY=2wMwDt78vksPrYmFMrRHpQ0FDKAW:BwWKHieg1S7DwzI:C; Hm_lvt_aec699bb6442ba076c89'
                  '81c6dc490771=1682412342; Hm_lpvt_aec699bb6442ba076c8981c6dc490771=1682412342; COOKIE_SESSION=297_0'
                  '_8_8_21_12_1_0_8_6_0_1_4899_0_356_0_1682412370_0_1682412014%7C9%230_0_1682412014%7C1; ZD_ENTRY=bai'
                  'du; ab_sr=1.0.1_MTQ3MDNkZDUwMWVlMDBiOTUwOTNmZTIyZWYxOTI5MjA5OGY2ZDE3MjZhODhkZTNkMjg0YjY2MDMwYjhiZDI'
                  '2YTZhY2Y3MjRkZTQ0ZDVlNjJlNzQyZTg1NTYwMmU4MDg0MWVlOGYxYjljYzAxZmEyZTc1NDc2NTBjYjczMjBhZmY1MTcyYWQyYT'
                  'g0YTE1Mzc2NmUxODA3ZWU2YmE5MDM5MQ==; __bid_n=187b632a23dacdbd374207; H_PS_645EC=1b0eaVb4%2FdPDyC6op6N'
                  'C0mbno0FhzDP1g9C0LK2F9fx137fXB7h1o3RqkSjaSbV12NqWTbs; BD_HOME=1; H_PS_PSSID=38516_36554_38469_38368_'
                  '38468_38485_37928_37709_38356_26350_38546'

    }
    key = str(input("图片的关键字："))
    num = int(input("图片的数量："))
    image_urls = Get_image_url(headers, key)
    if image_urls:
        Write_image(image_urls, num)
    else:
        print("程序结束！")