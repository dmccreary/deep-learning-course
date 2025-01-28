# Concept Dependencies

!!! prompt
    Below is a CSV file of concepts in a course on Deep Learning for junior or senior college students.  Please use this CSV file of 200 concepts for the next step.

    Next, please create a fully connected concept dependency graph
    for the concepts in this course.
    In this step, for each Concept in the CSV file, create a new
    list of other concepts on which it depends.  If a concept
    depends on a specific concept that is not listed,
    then we will need to add it as a new concept.
    This will help us determine the teaching order of the Concepts.

    Return the new list as a CSV file using the following format:

    1. The first column in the CSV file is a ConceptID you assigned in the list above.
    2. The second column is the ConceptLabel
    3. The third column is called Dependencies.  It's format is a pipe-delimited list of the Concept IDs that the concept depends upon.

    Check your results to make sure that all concepts are connected to at least one other concept.

    Foundation Concepts are concepts that have no dependant concepts in this set.  They have an empty third column of Dependencies.


    ## Response

    Foundation Concepts
In this example, the following concepts have no dependencies (empty Dependencies column). They serve as possible “starting points” in the curriculum:

- 1: AI Definition
- 3: History Of AI
- 10: Biological Inspiration
- 11: Python Setup
- 191: Project Scoping

All other concepts depend on at least one earlier concept—ensuring every concept is connected in one unified graph. Feel free to modify or expand these dependencies to match your exact course structure.