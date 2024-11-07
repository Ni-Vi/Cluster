# ideology_annotation_dataset
Dataset from 'What Sounds "Right" to Me? Experiential Factors in the Perception of Political Ideology'

Data files:
  - annotation_task_responses.csv: Annotator responses to the paired ideology ranking task:

        - worker: ID of the annotator giving a response
        - Input.page(question#): Metadata related to question # in the task
            - difficulty: Whether the question is a screening or a main task question
            - entity: Key entity of the question
            - perspective: Ideological perspective from where the question is asked
              (e.g. more left vs. more right)
            - post_X_id: Unique ID associated with a post in slot X of the question
            - post_X_paragraph: Paragraph of the post in slot X of the question
            - post_X_alignment: Ideological alignment of the subreddit containing the post
              in slot X of the question
            - post_X_test: Text of the post in slot X of the question
        - Answer.page(question#): Responses given by the annotator to the question
  - annotator_attributes.csv: This csv file contains the labels assigned to annotators
    based on their responses to the post-survey. The annotator groups of interest include:
	
        - ideology: Annotator's personal political beliefs
        - reddit: Annotator's familiarity/experience with Reddit
        - news: Annotator's familiarity with US political news

Annotator information in all files have been replaced by anonymized numeric ids.
