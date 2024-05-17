## Flag-derogatory-information-on-Craigslist
Developed a text analytics model to identify and flag derogatory, hate comments on the Craigslist platform

### Project Overview

In today's digital landscape, online platforms face a pressing challenge in moderating user-generated content to maintain a safe and welcoming environment. Craigslist, is no exception. The platform's open nature, while fostering a vibrant community, also renders it susceptible to misuse through the posting of inappropriate or adult content. To address this issue and enhance user experience, we propose the development of an Automated Content Moderation System leveraging machine learning technology.

The Automated Content Moderation System is designed to analyse the textual content of posts on Craigslist and categorize them into two distinct groups: safe and containing adult/not safe for work (NSFW) content. This binary classification serves as a foundational mechanism to filter out posts that may be offensive or inappropriate for general audiences. 

The system will be powered by advanced machine learning algorithms capable of understanding and interpreting the nuances of human language. The core of the system will be a text classification model, trained on a large dataset of labelled posts, categorizing them as either 'safe' or 'adult/NSFW'. The training process will involve natural language processing (NLP) techniques to handle the diverse and unstructured nature of text data on Craigslist.

### Data Collection 

The success of our Automated Content Moderation System hinges on the quality and comprehensiveness of the data used to train it. To this end, we propose a meticulous data
collection strategy focused on Craigslist's community section. This section is a vibrant hub of user-generated content, making it an ideal source for gathering a diverse range of text data.

Our approach utilizes sophisticated web scraping tool **BeautifulSoup** to extract the data. We have narrowed down the scope of the project to the sub-sections ‘missed connections’ & ‘activity’ in the community section of Craigslist, capturing relevant text data from various posts. The scraping algorithm is designed to meticulously comb through each page, identifying and extracting the ‘titles’ of the posts.

From the sections specified, we have scraped around **4652 records** to create our sample space of titles of users’ posts.

### Data Anotation

Once the data is collected, the next critical step is the manual annotation process. We have manually labelled all the 4652 records as derogatory (1) or non-derogatory (2). This manual intervention is crucial for creating an accurately annotated dataset, which forms the backbone of our training process.

<img width="864" alt="image" src="https://github.com/hchitnen/Flag-derogatory-information-on-Craigslist/assets/148294077/7e98a710-cb2b-43aa-a08b-b5c3b740dcd2">

### Model Development

The data analysis phase of this project involves the development and implementation of models tailored to optimize Craigslist's platform. The focus is on deploying predictive models that can handle the diverse and unstructured data characteristic of Craigslist.

The development process involves several stages, beginning with data preprocessing to clean and structure Craigslist's diverse datasets. We also tried to extract information as to how users are able to get around the algorithm and misuse the platform and find a trend in such behaviours. The core of the process is the modelling stage, where various algorithms are applied and tuned to predict such behaviours and flag such content before it is posted on the website.

<img width="877" alt="image" src="https://github.com/hchitnen/Flag-derogatory-information-on-Craigslist/assets/148294077/abcaf19c-f109-4bd3-a9cb-50a4f87bf433">


#### Text Representation Techniques:

1. **Tokenization:** This is the first step in text processing where the text is split into
individual words or tokens.
2. **Lemmatization:** Here, words are reduced to their base or dictionary form
3.  **Stop Words Removal:** Common words like 'is', 'and', 'the', etc., which may not contain important meaning, are removed. This helps in focusing on words which contribute to the understanding of the text's sentiment or classification.

#### Document Embedding:

**TF-IDF (Term Frequency-Inverse Document Frequency):** This technique evaluates how relevant a word is to a document in a collection of documents. It's useful in weighting terms and understanding the importance of words in different contexts.

#### Word Embedding:

**GloVe (Global Vectors for Word Representation):** It's an unsupervised learning algorithm for obtaining vector representations for words. By capturing word co-occurrences, GloVe provides word embeddings that provide meanings based on global word-word co-occurrence matrix.
**BERT (Bidirectional Encoder Representations from Transformers):** A more advanced technique that uses Transformers to understand the context of a word in a sentence. BERT's bidirectional nature allows it to capture the meaning of a word based on the entire context, which is a significant advancement over previous methods that looked at words in isolation.

### Model Performance

<img width="851" alt="image" src="https://github.com/hchitnen/Flag-derogatory-information-on-Craigslist/assets/148294077/d5fc1d1f-8304-4e48-91f6-f9777c64c1db">

The performance of the models as indicated by the AUC scores would suggest that the Multi-Layer Perceptron (MLP) is the most effective model for this specific task. Its ability to learn from vast amounts of data and capture complex relationships makes it well-suited for the nuanced task of content moderation. However, the trade-off is that Neural Networks require more data and computational power, and they can be less interpretable than simpler models like Logistic Regression or Decision Trees.



