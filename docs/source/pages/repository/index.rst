Repository Module
=================

.. currentmodule:: noctis

Overview
--------

The Repository Module is a crucial component of the Noctis project, primarily focused on interactions with Neo4j databases. This module provides a comprehensive set of tools and functionalities for managing, querying, and manipulating graph data within a Neo4j environment.

Neo4j Queries
-------------

.. automodule:: noctis.repository.neo4j.neo4j_queries
    :no-members:
    :no-inherited-members:

Abstract Classes
^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: ../../generated

    AbstractQuery
    Neo4jQueryRegistry

----

Constraints
^^^^^^^^^^^

.. autosummary::
    :toctree: ../../generated

    CreateUniquenessConstraints
    DropUniquenessConstraints
    ShowUniquenessConstraints

----

Retrieve Graph Queries
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: ../../generated


    GetNode
    GetTree
    GetRoutes
    GetPathsThroughIntermediates

----

Modify Graph Queries
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: ../../generated

    AddNodesAndRelationships
    LoadNodesFromCsv
    LoadRelationshipsFromCsv
    ImportDbFromCsv
    DeleteAllNodes

----

Retrieve Stats Query
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: ../../generated

    GetGDBSchema

----

Custom Query
^^^^^^^^^^^^

.. autosummary::
    :toctree: ../../generated

    CustomQuery

Neo4j Repository
----------------
.. automodule:: noctis.repository.neo4j.neo4j_repository
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: ../../generated

    Neo4jRepository

Neo4j Functions
---------------

.. automodule:: noctis.repository.neo4j.neo4j_functions
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: ../../generated

    _convert_datacontainer_to_query
    _convert_record_to_query_neo4j
    _create_node_queries
    _create_relationship_queries
    _generate_properties_assignment
    _get_dict_keys_from_csv
    _create_neo4j_import_path
    _generate_files_string
    _generate_nodes_files_string
    _generate_relationships_files_string
