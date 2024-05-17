
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Base URL template with common parameters
base_url_template_1 = "https://omaha.craigslist.org/search/nebraska-city-ne/mis?lat=40.658&lon=-95.86&search_distance=1000#search=1~list~{page}"
base_url_template_2 = "https://omaha.craigslist.org/search/nebraska-city-ne/act?lat=40.658&lon=-95.86&search_distance=1000#search=1~list~{page}"
# Initialize the list to store post titles and labels
data = []

# Iterate through 40 pages for url_1
for page_number in range(0, 41):
    page_url = base_url_template_1.format(page=page_number)
    response = requests.get(page_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # print(page_url)

    # Extract and store the content of the posting title and label
    result_nodes = soup.find_all('div', class_='title')
    # print(page_number)
    # print(result_nodes)
    for node in result_nodes:
        post_title = node.get_text(strip=True)
        data.append({"Title": post_title, "Label": "Missed Connections"})

# Iterate through 7 pages for url_2
for page_number in range(0, 8):
    page_url = base_url_template_2.format(page=page_number)
    response = requests.get(page_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract and store the content of the posting title and label
    result_nodes = soup.find_all('div', class_='title')
    for node in result_nodes:
        post_title = node.get_text(strip=True)
        data.append({"Title": post_title, "Label": "Activity Partners"})

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv("craigslist_data.csv", index=False)

