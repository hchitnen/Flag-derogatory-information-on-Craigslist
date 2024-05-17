    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from transformers import BertTokenizer, BertModel
    import torch

    #-------------------TF-IDF MODEL--------------------------------------------------------#

    import nltk
    import numpy as np

    comments = pd.read_csv('Data.csv', encoding='ISO-8859-1')
    print(comments.head())

    corpus = comments.iloc[:,0].tolist()
    label = comments.iloc[:,1].tolist()

    #tokenize each review in the list.
    token_corpus = []
    for doc in corpus:
        token_doc = nltk.word_tokenize(doc)
        token_corpus.append(token_doc)

    #lemmatize the tokenized corpus and lower the case.
    nltk.download('wordnet')
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lem_corpus = []
    for doc in token_corpus:
        lemmatized_token = [lemmatizer.lemmatize(token).lower() for token in doc if token.isalpha()]
        lem_corpus.append(lemmatized_token)

    #remove stopwords and punctuations
    from nltk.corpus import stopwords
    import string
    norm_corpus = []
    for doc in lem_corpus:
        stop_words_removed = [token for token in doc if not token in stopwords.words('english') if token.isalpha()]
        norm_corpus.append(stop_words_removed)

    X_train, X_test, y_train, y_test, comments_train, comments_test = train_test_split(
        norm_corpus, label, corpus, test_size=0.3, random_state=42)

    from sklearn.feature_extraction.text import TfidfVectorizer
    norm_train_strings = [' '.join(doc) for doc in X_train]
    norm_test_strings = [' '.join(doc) for doc in X_test]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3)
    vectorizer.fit(norm_train_strings)
    train_vector = vectorizer.transform(norm_train_strings)
    test_vector = vectorizer.transform(norm_test_strings)

    classifiers = {
        'MLPClassifier': MLPClassifier(solver='lbfgs', hidden_layer_sizes=(3, 10), max_iter=500, random_state=1),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=1),
        'LogisticRegression': LogisticRegression(random_state=1),
        'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100, random_state=1),
        'SVM': SVC(kernel='linear', random_state=1),
        'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=1)
    }

    # Train, predict, and evaluate each model
    for name, model in classifiers.items():
        # Training
        model.fit(train_vector, y_train)

        # Prediction
        y_pred = model.predict(test_vector)

        # Evaluation
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)

        print(f"{name} Accuracy: {acc:.2f}%")
        print(f"{name} AUC: {auc:.2f}")

    # # Create a DataFrame with test comments, original class variable values, and predicted class variable values
    # test_data = pd.DataFrame({'Comments': comments_test, 'Original_Class': y_test, 'Predicted_Class': y_pred})
    #
    # # Save the test data to a CSV file
    # test_data.to_csv('TFIDF_test_data.csv', index=False)

    #-------------------GLOVE MODEL--------------------------------------------------------#

    import nltk
    import numpy as np

    comments = pd.read_csv('Data.csv', encoding='ISO-8859-1')
    print(comments.head())

    corpus = comments.iloc[:,0].tolist()
    label = comments.iloc[:,1].tolist()

    #tokenize each review in the list.
    token_corpus = []
    for doc in corpus:
        token_doc = nltk.word_tokenize(doc)
        token_corpus.append(token_doc)

    #lemmatize the tokenized corpus and lower the case.
    nltk.download('wordnet')
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lem_corpus = []
    for doc in token_corpus:
        lemmatized_token = [lemmatizer.lemmatize(token).lower() for token in doc if token.isalpha()]
        lem_corpus.append(lemmatized_token)

    #remove stopwords and punctuations
    from nltk.corpus import stopwords
    import string
    norm_corpus = []
    for doc in lem_corpus:
        stop_words_removed = [token for token in doc if not token in stopwords.words('english') if token.isalpha()]
        norm_corpus.append(stop_words_removed)

    #glove - word embedding
    from gensim.scripts.glove2word2vec import glove2word2vec
    glove_input_file = 'glove.6B.100d.txt'
    word2vec_output_file = 'finalproject_glove.txt.word2vec'
    glove2word2vec(glove_input_file, word2vec_output_file)

    from gensim.models import KeyedVectors
    filename = 'finalproject_glove.txt.word2vec'
    model = KeyedVectors.load_word2vec_format(filename, binary=False)

    #converting the corpus
    glove_vectors=[]
    for doc in norm_corpus:
        word_glove=[]
        for word in doc:
            if word in model:
                word2glove = model[word]
                word_glove.append(word2glove)
        glove_vectors.append(np.array(word_glove))

    #PADDING
    # Determine the maximum lengths in each dimension
    max_rows = max(arr.shape[0] if len(arr.shape) > 0 else 0 for arr in glove_vectors)
    max_columns = max(arr.shape[1] if len(arr.shape) > 1 else 0 for arr in glove_vectors)
    # Initialize an empty 3D array filled with zeros
    padded_array = np.zeros((len(glove_vectors), max_rows, max_columns), dtype=glove_vectors[0].dtype)

    #Pad each 2D array and populate the 3D array
    for i, arr in enumerate(glove_vectors):
        if arr.size > 0:
            padded_array[i, :arr.shape[0], :arr.shape[1]] = arr

    #flattening the 3d array
    num_comments, max_sequence_length, vector_dimension = padded_array.shape
    glove_vectors_2d = padded_array.reshape((num_comments, -1))

    # PARTITION: Split the data into training and testing sets
    X_train, X_test, y_train, y_test, comments_train, comments_test = train_test_split(
        glove_vectors_2d, label, corpus, test_size=0.3, random_state=42)

    classifiers = {
        'MLPClassifier': MLPClassifier(solver='lbfgs', hidden_layer_sizes=(3, 10), max_iter=500, random_state=1),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=1),
        'LogisticRegression': LogisticRegression(random_state=1),
        'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100, random_state=1),
        'SVM': SVC(kernel='linear', random_state=1),
        'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=1)
    }

    # Train, predict, and evaluate each model
    for name, model in classifiers.items():
        # Training
        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)

        # Evaluation
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)

        print(f"{name} Accuracy: {acc:.2f}%")
        print(f"{name} AUC: {auc:.2f}")

    # # Create a DataFrame with test comments, original class variable values, and predicted class variable values
    # test_data = pd.DataFrame({'Comments': comments_test, 'Original_Class': y_test, 'Predicted_Class': y_pred})
    #
    # # Save the test data to a CSV file
    # test_data.to_csv('GLOVE_test_data.csv', index=False)


    #-------------------BERT MODEL--------------------------------------------------------#

    # Load the dataset
    file_path = 'Data.csv'  # Replace with the actual path
    df = pd.read_csv(file_path, encoding='ISO-8859-1')

    # Check the structure of the dataframe
    print(df.head())

    # Split the dataset into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    # Tokenize and embed the training comments
    train_tokens = tokenizer(list(train_df['Community']), padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        train_output = bert_model(**train_tokens)

    # Extract [CLS] token embeddings
    train_embeddings = train_output.last_hidden_state[:, 0, :].numpy()
    print(train_embeddings.shape)

    classifiers = {
        'LogisticRegression': LogisticRegression(random_state=1),
        'MLPClassifier': MLPClassifier(solver='lbfgs', hidden_layer_sizes=(2, 10), random_state=1),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=1),
        'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100, random_state=1),
        'SVM': SVC(kernel='linear', random_state=1, probability=True),
        'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=1)
    }

    # Train, predict, and evaluate each model
    for name, classifier in classifiers.items():
        # Train the model
        classifier.fit(train_embeddings, train_df['Class'])

        # Tokenize and embed the testing comments
        test_tokens = tokenizer(list(test_df['Community']), padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            test_output = bert_model(**test_tokens)

        # Extract [CLS] token embeddings for the testing set
        test_embeddings = test_output.last_hidden_state[:, 0, :].numpy()

        # Make predictions on the testing set
        predictions = classifier.predict(test_embeddings)

        # Calculate AUC
        auc = roc_auc_score(test_df['Class'], classifier.predict_proba(test_embeddings)[:, 1])

        # Display the AUC and accuracy
        print(f"{name} AUC: {auc:.2f}")
        accuracy = accuracy_score(test_df['Class'], predictions)
        print(f"{name} Accuracy: {accuracy:.2f}\n")

    # # Create a DataFrame with test comments, original class variable values, and predicted class variable values
    # test_df['predictions'] = predictions
    #
    # # Save the test data to a CSV file
    # test_df.to_csv('BERT_test_data.csv', index=False)
