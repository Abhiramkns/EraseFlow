from openai import OpenAI

def get_coverage(
    client,
    prompt
):
    contents_coverage=\
    """
    Given a image description, label the visually groundable words in the description, and a score indicating how visually groundable it is. Classify each word into a type (entity, activity, attribute, counting, color, material, spatial, location, shape, style, other).

    Description:
    Portrait of a gecko wearing a train conductor’s hat and holding a flag that has a yin-yang symbol on it. Woodcut.
    The visual-groundable words and their scores are labelled below:
    {1}[Portrait, style, 0.8] of {2}[a, count, 1.0] {3}[gecko, entity, 1.0] {4}[wearing, activity, 1.0] {5}[a, count, 1.0] {6}[train conductor’s hat, entity, 1.0] and {7}[holding, entity, 1.0] {8}[a, count, 1.0] {9}[flag, entity, 1.0] that has {10}[a yin-yang symbol, entity, 1.0] on it. {11}[Woodcut, material, 1.0].

    Description:
    square blue apples on a tree with circular yellow leaves
    The visual-groundable words and their scores are labelled below:
    {1}[square, shape,1.0] {2}[blue, color, 1.0] {3}[apples, entities, 1.0] {4}[on, spatial, 1.0] {5}[a, count, 1.0] {6}[tree, entity, 1.0] with {7}[circular, shape, 1.0] {8}[yellow, color, 1.0] {9}[leaves, entity, 1.0]

    Description:
    A small dog running on a beach happily on a sunny day
    The visual-groundable words and their scores are labelled below:
    {1}[A, count, 1.0] {2}[small, attribute, 0.3] {3}[dog, entity, 1.0] {4}[running, activity, 1.0] {5}[on, spatial, 1.0] {6}[a, count, 0.0] {7}[beach, entity, 1.0] happily on {8}[a, count, 0.0] {9}[sunny, attribute, 0.5] day.

    Description: acrylic drawing, illustration, multiple mushrooms and pink jello, naive, flat, sketchy, purple background.
    The visual-groundable words and their scores are labelled below:
    {1}[crylic drawing, style, 1.0], {2}[illustration, style, 0.2], {3}[multiple, count, 0.5] {4}[mushrooms, eneity, 1.0] and {5}[pink, color, 1.0] {6}[jell, entity, 1.0], {7}[naive, attribute, 0.1], {8}[flat, attribute, 0.8] {9}[sketchy, style, 0.8] {10}[purple, color, 1.0] {11}[background, entity, 1.0].

    Description: a girl with many braids, riding away on her bike through the city, children’s book cover illustration, detailed background, vibrant colors
    The visual-groundable words and their scores are labelled below:
    {1}[a, count, 1.0] {2}[girl, entity, 1.0], with {3}[many, counts, 0.8] {4}[braids, entity, 1.0], {5}[riding away, activity, 1.0] on her {6}[bike, entity, 1.0] through the {7}[city, place, 0.8], {8}[children’s book cover illustration, style, 0.8], {9}}[detailed background, entity, 1.0] {10}[vibrant colors, colors , 1.0].

    Description: ###
    """

    query = contents_coverage.replace('###', prompt)

    completion = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
        {"role": "user", "content": query}
      ]
    )

    return completion.choices[0].message.content
    