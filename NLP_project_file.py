#!/usr/bin/env python
# coding: utf-8

# In[1]:


data = [
    {
        "Student ID": 1,
        "Student Name": "Ram",
        "Question ID": "Q1",
        "Question": "Define supervised learning and explain its key characteristics and principles.",
        "Answer": "Supervised learning is a machine learning paradigm where models are trained using labeled data to make predictions based on input features and corresponding output labels. It requires annotated datasets. It is further divided into “Classification” and “Regression”."
    },
    {
        "Student ID": 1,
        "Student Name": "Ram",
        "Question ID": "Q2",
        "Question": "How do you calculate the area of a rectangle?",
        "Answer": "The area of a rectangle is length times width."
    },
    {
        "Student ID": 1,
        "Student Name": "Ram",
        "Question ID": "Q3",
        "Question": "Explain the importance of feature scaling in machine learning.",
        "Answer": "Feature scaling is essential in machine learning to ensure that all features contribute equally to the learning process. It helps in improving the convergence speed of gradient-based algorithms and prevents features with larger magnitudes from dominating those with smaller magnitudes."
    },
    {
        "Student ID": 1,
        "Student Name": "Ram",
        "Question ID": "Q4",
        "Question": "What is the difference between classification and regression?",
        "Answer": "Classification involves predicting discrete labels or categories, while regression predicts continuous numeric values."
    },
    {
        "Student ID": 1,
        "Student Name": "Ram",
        "Question ID": "Q5",
        "Question": "What are the key components of a neural network?",
        "Answer": "The key components of a neural network include neurons (or nodes), which are connected in layers (input, hidden, and output layers), activation functions, weights, biases, and the overall architecture (number of layers and neurons per layer)."
    },
    {
        "Student ID": 2,
        "Student Name": "Mohit",
        "Question ID": "Q1",
        "Question": "Define supervised learning and explain its key characteristics and principles.",
        "Answer": "Supervised learning is a method where models are trained with labeled data to predict outcomes. Labeled datasets help models learn patterns for accurate predictions on new data. Some of its algorithms are Decision Trees, Random Forest, Support Vector Machines (SVM), Naive Bayes, etc."
    },
    {
        "Student ID": 2,
        "Student Name": "Mohit",
        "Question ID": "Q2",
        "Question": "How do you calculate the area of a rectangle?",
        "Answer": "A rectangle's area is calculated by length * width."
    },
    {
        "Student ID": 2,
        "Student Name": "Mohit",
        "Question ID": "Q3",
        "Question": "What is cross-validation and why is it important in machine learning?",
        "Answer": "Cross-validation is a technique used to assess the performance of a machine learning model. It involves partitioning the dataset into multiple subsets, training the model on some subsets, and testing it on others. This helps to evaluate the model's performance and generalization ability."
    },
    {
        "Student ID": 2,
        "Student Name": "Mohit",
        "Question ID": "Q4",
        "Question": "Explain the concept of overfitting in machine learning.",
        "Answer": "Overfitting occurs when a model learns the training data too well, capturing noise and random fluctuations rather than the underlying pattern. As a result, the model performs well on the training data but poorly on unseen data."
    },
    {
        "Student ID": 2,
        "Student Name": "Mohit",
        "Question ID": "Q5",
        "Question": "What is the purpose of regularization in machine learning?",
        "Answer": "Regularization is a technique used to prevent overfitting in machine learning models by adding a penalty term to the loss function. It helps to keep the model's weights small, thereby reducing the complexity of the model and improving its generalization ability."
    },
    {
        "Student ID": 3,
        "Student Name": "Rohan",
        "Question ID": "Q1",
        "Question": "Define supervised learning and explain its key characteristics and principles.",
        "Answer": "Supervised learning involves several algorithms which predict outcomes based on inputs."
    },
    {
        "Student ID": 3,
        "Student Name": "Rohan",
        "Question ID": "Q2",
        "Question": "How do you calculate the area of a rectangle?",
        "Answer": "The area of a rectangle is equal to its length multiplied by its width."
    },
    {
        "Student ID": 3,
        "Student Name": "Rohan",
        "Question ID": "Q3",
        "Question": "What are the advantages of using decision trees in machine learning?",
        "Answer": "Decision trees are interpretable, easy to understand, and can handle both numerical and categorical data. They require little data preprocessing, handle nonlinear relationships, and provide feature importances."
    },
    {
        "Student ID": 3,
        "Student Name": "Rohan",
        "Question ID": "Q4",
        "Question": "Explain the concept of underfitting in machine learning.",
        "Answer": "Underfitting occurs when a model is too simple to capture the underlying structure of the data. It performs poorly on both the training and test datasets, indicating that it has not learned enough from the data."
    },
    {
        "Student ID": 3,
        "Student Name": "Rohan",
        "Question ID": "Q5",
        "Question": "What is the difference between Bagging and Boosting?",
        "Answer": "Bagging (Bootstrap Aggregating) involves training multiple independent models on different subsets of the training data and averaging their predictions, while Boosting involves sequentially training models, where each subsequent model focuses on the errors made by the previous ones."
    },
    {
        "Student ID": 4,
        "Student Name": "Gaurav",
        "Question ID": "Q1",
        "Question": "Define supervised learning and explain its key characteristics and principles.",
        "Answer": "Supervised learning is a type of machine learning where models learn from labeled data to make predictions about unseen or future data. It involves two phases: training and inference."
    },
    {
        "Student ID": 4,
        "Student Name": "Gaurav",
        "Question ID": "Q2",
        "Question": "How do you calculate the area of a rectangle?",
        "Answer": "To calculate the area of a rectangle, you multiply its length by its width."
    },
    {
        "Student ID": 4,
        "Student Name": "Gaurav",
        "Question ID": "Q3",
        "Question": "What are the different types of kernels used in SVM?",
        "Answer": "Some of the commonly used kernels in Support Vector Machines (SVM) are linear, polynomial, radial basis function (RBF), and sigmoid kernels."
    },
    {
        "Student ID": 4,
        "Student Name": "Gaurav",
        "Question ID": "Q4",
        "Question": "Explain the Bias-Variance tradeoff in machine learning.",
        "Answer": "The bias-variance tradeoff is a fundamental concept in machine learning that refers to the tradeoff between the bias (error due to overly simplistic assumptions) and variance (error due to sensitivity to fluctuations in the training data) of a model."
    },
    {
        "Student ID": 4,
        "Student Name": "Gaurav",
        "Question ID": "Q5",
        "Question": "What is the purpose of feature engineering in machine learning?",
        "Answer": "Feature engineering involves selecting, transforming, and creating new features from the raw data to improve the performance of machine learning models. It helps in capturing relevant information and reducing noise, ultimately leading to better model accuracy."
    },
    {
        "Student ID": 5,
        "Student Name": "Priya",
        "Question ID": "Q1",
        "Question": "Define supervised learning and explain its key characteristics and principles.",
        "Answer": "Supervised learning is a type of machine learning where the algorithm learns from labeled data to predict the output for unseen data. It requires a dataset with input-output pairs."
    },
    {
        "Student ID": 5,
        "Student Name": "Priya",
        "Question ID": "Q2",
        "Question": "How do you calculate the area of a rectangle?",
        "Answer": "The area of a rectangle can be calculated by multiplying its length by its width."
    },
    {
        "Student ID": 5,
        "Student Name": "Priya",
        "Question ID": "Q3",
        "Question": "What are the steps involved in building a machine learning model?",
        "Answer": "The steps involved in building a machine learning model include data collection, data preprocessing (cleaning, feature engineering, scaling), model selection, model training, model evaluation, hyperparameter tuning, and deployment."
    },
    {
        "Student ID": 5,
        "Student Name": "Priya",
        "Question ID": "Q4",
        "Question": "Explain the difference between L1 and L2 regularization.",
        "Answer": "L1 regularization adds a penalty term equal to the absolute value of the coefficients, promoting sparsity, while L2 regularization adds a penalty term equal to the square of the coefficients, preventing large coefficients."
    },
    {
        "Student ID": 5,
        "Student Name": "Priya",
        "Question ID": "Q5",
        "Question": "What is the purpose of ensemble learning in machine learning?",
        "Answer": "Ensemble learning combines multiple individual models to improve the overall performance and robustness of the system. It helps in reducing bias and variance, enhancing generalization, and handling complex relationships in the data."
    }
]

# Printing the first dictionary in the list
print(data[0])


# In[2]:


import pandas as pd
df = pd.DataFrame(data)


# In[3]:


df


# In[ ]:


df.info()


# # Access WordNet data: NLTK provides easy access to the WordNet database, allowing you to look up words, retrieve their definitions, and explore their relationships with other words in WordNet.
# 
# Synonym and antonym lookup: You can find synonyms (words with similar meanings) and antonyms (words with opposite meanings) for a given word.
# 
# Word sense disambiguation: WordNet can be used to disambiguate word senses. Given a word with multiple meanings, WordNet provides information about different senses of that word, helping to determine which sense is appropriate in a given context.
# 
# Hyponym and hypernym relationships: WordNet organizes words into hierarchies where more specific terms (hyponyms) are grouped under more general terms (hypernyms). NLTK allows you to navigate these hierarchical relationships.
# 
# Lemmatization: NLTK's WordNet module can be used for lemmatization, which is the process of reducing words to their base or dictionary form (lemmas).

# In[6]:


import  nltk
nltk.download('punkt') #NLTK is a popular library for natural language processing tasks in Python. 
                            #The punkt module within NLTK is responsible for tokenizing text, i.e., splitting text into individual words or tokens.
nltk.download("wordnet")

from nltk.stem import WordNetLemmatizer


# In[7]:


def word_token_lema(text):
    tokens=nltk.word_tokenize(text)
    lema=WordNetLemmatizer()
    lema_tokens=[lema.lemmatize(words) for words in tokens]
    tokenized_string = " ".join(lema_tokens)

    return lema.lemmatize(tokenized_string)
    
    
df["lemma"]=df['Answer'].apply(word_token_lema)


# In[8]:


df["lemma"][0]


# In[9]:


df["Answer"][0]


# In[10]:


df.head()


# In[11]:


import nltk
nltk.download('maxent_ne_chunker')

nltk.download('words')


text="Supervised learning is  build by Piyush , a machine learning paradigm where models are trained using labeled data to make predictions based on input features and corresponding output labels. It requires annotated datasets. It is further divided into “Classification” and “Regression"
tokens=nltk.word_tokenize(text)
pos_tags=nltk.pos_tag(tokens)
pos_tags
name_entity=nltk.ne_chunk(pos_tags)


# In[12]:


def extract_name(text):
    tokens=nltk.word_tokenize(text)
    pos_tags=nltk.pos_tag(tokens)
    name_entity=nltk.ne_chunk(pos_tags)
    
    name=[]
    for entity in name_entity:
        if isinstance(entity,nltk.Tree):
            if entity.label() == "PERSON":
                names=" ".join([word for word,tag in entity.leaves()])
                name.append(names)
    return name


# In[13]:


df["name"]=df["Answer"].apply(extract_name)


# In[14]:



def score_grammar(text):
    grammar = "NP: {<DT>?<JJ>*<NN>}" 
    parser=nltk.RegexpParser(grammar)
    tokens=nltk.word_tokenize(text)
    tagged_words=nltk.pos_tag(tokens)
    parsed_tree=parser.parse(tagged_words)
    NP_length=0
    
    
    for subtree in parsed_tree.subtrees():
        if subtree.label() == "NP":
            NP_length =NP_length+len(subtree.leaves())
            
            
    sentence_length=len(tagged_words)
    NP_percent=(NP_length/sentence_length)*100
    return NP_percent

        
    


# In[15]:


df["Grammar_score"]=df["Answer"].apply(score_grammar)


# In[16]:


df.head()


# In[18]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def calculate_similarity(questions, answers):
    question_tokens=nltk.word_tokenize(questions.lower())
    question=" ".join(question_tokens)
    
    preprocess_answer=nltk.word_tokenize(answers.lower())
    answer=" ".join(preprocess_answer)
    
    document=[question,answer]
    
    tfidf_vector=TfidfVectorizer()
    
    tfidf_matrix=tfidf_vector.fit_transform(document)
    
    similarity_Score=cosine_similarity(tfidf_matrix[0],tfidf_matrix[1:]).flatten()
    
    return similarity_Score[0]


# In[19]:


df["similarity_score"]=df.apply(lambda row: calculate_similarity(row["Question"],row["Answer"]),axis=1)


# In[20]:


df.head()


# In[21]:


def score_information_completeness(answer):
    # List of keywords related to the question based on it will find how many keywords are present and give score
    keywords = ["define", "explain", "supervised learning", "key characteristics", 
            "principles", "important", "factor", "impact", "effect", "cause", 
            "significance", "consequence", "influence", "role", 
            "reason", "signify", "purpose", "advantage", "disadvantage", 
            "difference", "steps", "steps involved", "building", 
            "concept", "technique", "method", "algorithm"]

    
    answer = answer.lower()
    answer_token=nltk.word_tokenize(answer)
    keyword_count=0
    for i in answer_token:
        if i in keywords:
            keyword_count+=1
            
    completness_percentage=(keyword_count/len(answer))*100

    


    return completness_percentage


# In[22]:


df["score_information_completeness_percentage"]=df["Answer"].apply(score_information_completeness)


# In[23]:


df


# In[24]:


def measure_engagement(answer, min_length=10, max_length=150):
    word_count = len(nltk.word_tokenize(answer))
    
    if word_count < min_length:
        engagement_score = 0  # If the answer is too short, assign a score of 0
    elif word_count > max_length:
        engagement_score = 100  # If the answer is too long, assign a score of 100
    else:
        # Normalize the engagement score between 0 and 100 based on the specified range
        engagement_score = ((word_count - min_length) / (max_length - min_length)) * 100
    
    return engagement_score


# In[ ]:





# In[25]:


df["Engagement_score"]=df["Answer"].apply(measure_engagement)


# In[26]:


df


# In[27]:


def calculate_answer_score(row):
    # Define weights for each feature
    grammar_weight = 0.3
    similarity_weight = 0.4
    completeness_weight = 0.15
    engagement_weight = 0.15
    
    # Calculate the weighted sum of feature scores
    total_score = (row['Grammar_score'] * grammar_weight +
                   row['similarity_score'] * similarity_weight +
                   row['score_information_completeness_percentage'] * completeness_weight +
                   row['Engagement_score'] * engagement_weight)
    
    return total_score


# In[28]:


df["Answer_Score"]=df.apply(calculate_answer_score, axis=1)


# In[31]:


df


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

# Initialize the standard scaler
scaler = StandardScaler()

# Fit the scaler on the 'Answer_score' column and transform it
df['Answer_score_standardized'] = scaler.fit_transform(df[['Answer_Score']])

# Initialize the min-max scaler
min_max_scaler = MinMaxScaler(feature_range=(0, 100))

# Fit the min-max scaler on the standardized values and transform it
df['Answer_score_scaled'] = min_max_scaler.fit_transform(df[['Answer_score_standardized']])


# In[33]:


# Drop the previous column 'Answer_score_standardized'
df.drop(columns=['Answer_score_standardized'], inplace=True)


# In[ ]:


df


# In[ ]:




