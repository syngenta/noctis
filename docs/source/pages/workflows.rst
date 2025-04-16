Typical Workflows
--------------------
The two typical workflows in NOTICS are a bulk upload from a CSV to an empty Graph Data Base and getting a python object with a query result. We hope that these diagrams will help to understand what is happening behind the curtains.

Graph Database Creation Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This diagram outlines the process when a user wants to create a Graph Database from scratch using a CSV file containing reaction strings. The workflow begins with the instantiation of a Preprocessor and a GraphSchema objects, followed by the ingestion of the initial CSV file. The data is then transformed into a Neo4j-compatible CSV format, where each row is expanded into a set of nodes and relationships. Finally, a Neo4jRepository object is instantiated and a query is executed to upload the formatted data into the Graph Database.

.. mermaid:: ../static/diagrams/CSV_to_GDB_workflow.mmd
    :align: center



Graph Database to PythonObject Workflow
^^^^^^^^^^^^^^^^^^^^^

The second diagram depicts the sequence of events when a user queries the graph database and requests a Python object as the output. This workflow illustrates how the user's query is processed, the interaction between NOCTIS and the Graph Database through Neo4jRepository object, the transformation of the query results into a Python object using the classes of data transformation module, and the return of the object to the user.

.. mermaid:: ../static/diagrams/GDB_to_PyObj_workflow.mmd
    :align: center
