# Personality-prediction-through-CV
This project aims to predict the personality of an individual based on their OCEAN values and resume. The system utilizes Natural Language Processing (NLP) techniques to analyze resumes and extract relevant information. The predictions are made using the Big Five Personality Traits model, which measures Openness, Conscientiousness, Extraversion, Agreeableness, and Natural Reactions (Neuroticism).


# Objectives
The system serves several objectives:

Efficiency in Candidate Selection: Reduce the workload of hiring, training, and human resources departments by automating the personality prediction process.

Holistic Candidate Evaluation: Focus not only on qualifications and experience but also on personality traits relevant to specific job profiles.

Data Storage and Comparison: Allow administrators to store data in an Excel sheet for further analysis, comparison, and sorting.


# Background of Personality Perception
The Big Five Personality Traits model, introduced by Lewis Goldberg, measures five key dimensions:

Openness: Creativity, desire for knowledge, and openness to new experiences.

Conscientiousness: Level of care, organization, and thoroughness in life and work.

Extraversion/Introversion: Level of sociability and energy derived from social interactions.

Agreeableness: How well one gets along with others, including consideration and willingness to compromise.

Natural Reactions (Neuroticism): Emotional reactions and stability in stressful situations.


# Dependencies of the System
os

pandas

numpy

tkinter

functools

pyresparser

sklearn

nltk

spaCy


# Description

The system utilizes logistic regression for model training and pyresparser for extracting information from resumes. The flow of the system involves:

train_model class: Contains methods for training the model and predicting results based on various values.

main method: Initializes the train_model class, trains the model, and designs the landing page of the system using Tkinter.

predict_person method: Creates a new window for predicting personality, takes user inputs, and calls the prediction_result method.

OpenFile method: Allows the user to choose a resume file and updates the button text with the selected file's name.

prediction_result method: Predicts personality, extracts information from the resume, and displays the results on a full-screen window.

check_type method: Converts various data types into the desired format.

This system enhances the candidate selection process by considering both traditional qualifications and personality traits, providing a comprehensive approach to hiring decisions.
