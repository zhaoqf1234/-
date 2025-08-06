import requests
from bs4 import BeautifulSoup


response = requests.get("https://xyq.163.com/introduce/")
response.encoding = response.apparent_encoding
html = response.text
soup = BeautifulSoup(html,"html.parser")

all_titles = soup.find_all("div",attrs={"class":"link-cont"})

for title in all_titles:
    links = title.find_all("a", attrs={"href": True, "title": True})
    for link in links:
        if link:
            # link = link.find("a", attrs={"href": True, "title": True})
            href = link['href']  # 获取 a 标签的 href 属性
            title_text = link['title']  # 获取 a 标签的 title 属性
            if not href.startswith('http'):
                href = 'https:' + href

            filename = f"{title_text}.txt"

            response = requests.get(href)
            response.encoding = response.apparent_encoding
            html = response.text
            soup = BeautifulSoup(html, "html.parser")
            content = soup.find_all("div",attrs={"class":"artText"})

            with open(filename, "w", encoding="utf-8") as file:
                for item in content:
                    file.write(item.get_text())
                print(f"成功爬取数据，创建{title_text}数据文件")
