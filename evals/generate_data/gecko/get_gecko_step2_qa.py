from openai import OpenAI
import json

def get_qa(
    client,
    prompt,
    types
):

    contents_qa=\
    """
    Given a image description, generate one or two multiple-choice questions that verifies if the image description is correct.
    Classify each concept into a type (object, human, animal, food, activity, attribute, counting, color, material , spatial, location, shape, other), and then generate a question for each type.
    Strictly return the outputs strictly in a json format as shown in the examples. Do not output any other text.
    
    ## Examples
    Description: A man posing for a selfie in a jacket and bow tie.
    The visual-groundable words and their scores are labelled below:
    A {1}[Man, human] {2}[posing, activity] for a {3}[selfie, object] in a {4}[jacket, object] and a {5}[bow tie, object].
    Generated questions and answers are below:
    Output:
    [
      {
        "id": 1,
        "question": "is there a man in the image?",
        "choices": ["yes", "no"],
        "answer": "yes"
      },
      {
        "id": 2,
        "question": "is the man posing for the selfie?",
        "choices": ["yes", "no"],
        "answer": "yes"
      },
      {
        "id": 3,
        "question": "is the man taking a selfie?",
        "choices": ["yes", "no"],
        "answer": "yes"
      },
      {
        "id": 4,
        "question": "is the man wearing a jacket?",
        "choices": ["yes", "no"],
        "answer": "yes"
      },
      {
        "id": 5,
        "question": "is the man wearing a bow tie?",
        "choices": ["yes", "no"],
        "answer": "yes"
      }
    ]

    Description: A horse and several cows feed on hay.
    The visual-groundable words and their scores are labelled below:
    A {1}[horse, animal] and {2}[several, count] {3}[cows, animal] {4}[feed, activity] on a {5}[hay, object].
    Generated questions and answers are below:
    Output:
    [
      {
        "id": 1,
        "question": "is there a horse?",
        "choices": ["yes", "no"],
        "answer": "yes"
      },
      {
        "id": 2,
        "question": "are there several cows?",
        "choices": ["yes", "no"],
        "answer": "yes"
      },
      {
        "id": 3,
        "question": "are there cows?",
        "choices": ["yes", "no"],
        "answer": "yes"
      },
      {
        "id": 4,
        "question": "are the horse and cows feeding on hay?",
        "choices": ["yes", "no"],
        "answer": "yes"
      },
      {
        "id": 5,
        "question": "is there hay?",
        "choices": ["yes", "no"],
        "answer": "yes"
      }
    ]


    Now, it is your turn.
    Description: ###
    ###
    Generated questions and answers are below:
    Output:
    """
    
    query = contents_qa.replace("###", prompt, 1)
    query = query.replace("###", types)

    response_qa = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
        {"role": "user", "content": query}
      ]
    )

    response_qa = json.loads(response_qa.choices[0].message.content)

    return response_qa
