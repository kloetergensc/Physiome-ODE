from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


def save_code(url, folder_path):
    # Send a GET request to the URL
    url += "@@cellml_codegen/Python"
    name = url.split("/")[5].split(".")[0]
    url = url.replace("slow", "fast")
    response = requests.get(url)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text()
        try:
            start = text.index("# Size of variable arrays:")
            end_key = """if __name__ == "__main__":
    (voi, states, algebraic) = solve_model()
    plot_model(voi, states, algebraic)"""
            end = text.index(end_key) + len(end_key)
        except:
            print("FAIL: no valid python file", folder_path, name)
            print(url)
            return url, name

        Path(folder_path + "/" + name).mkdir(parents=True, exist_ok=True)
        with open(f"{folder_path}/{name}/model.py", "w") as file:
            file.write(text[start:end])
        print("SUCCCESS", folder_path, name)

    else:
        print("FAIL: dead link", folder_path, name)
    return url, name


# URL of the website you want to crawl
url_list = []
name_list = []
with open("resources/links_fields.txt", "r") as file:
    fields = file.read()
    fields = fields.split("\n")
for topic in fields:
    topic_name = topic.split("/")[-1]
    folder_path = f"models/{topic_name}"
    # if not Path(folder_path).exists():
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    response = requests.get(topic)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        anchor_tags = soup.find_all("a")
        for tag in anchor_tags:
            link = tag.get("href")
            if link:
                if "/exposure/" in link:
                    link = link[:-4]
                    url, name = save_code(link, folder_path=folder_path)
                    url_list.append(url)
                    name_list.append(name)
    else:
        print("Failed to retrieve the web page.")

df = pd.DataFrame(list(zip(name_list, url_list)))
df.to_csv("links.csv")
