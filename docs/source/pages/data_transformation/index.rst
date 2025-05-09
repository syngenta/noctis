Data Transformation
===================

.. currentmodule:: noctis

Overview
--------

The Data Transformation module is a key component of the NOCTIS, focusing on various aspects of data manipulation, formatting, and processing. This module provides tools for styling data, formatting Neo4j results, and both preprocessing and postprocessing of chemical data.

Data Styles
-----------

Data Frame Stylers
^^^^^^^^^^^^^^^^^^

.. automodule:: noctis.data_transformation.data_styles.dataframe_stylers
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: ../../generated
    :template: pydantic_class_template.rst

    NodesRelationshipsStyle
    PandasExportStyle

Neo4j
-----

Neo4j Formatter
^^^^^^^^^^^^^^^

.. automodule:: noctis.data_transformation.neo4j.neo4j_formatter
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: ../../generated

    Neo4jResultFormatter
    format_result

Postprocessing
--------------

Chem Data Generators
^^^^^^^^^^^^^^^^^^^^

.. automodule:: noctis.data_transformation.postprocessing.chemdata_generators
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: ../../generated

    ChemDataGeneratorInterface
    ChemDataGeneratorFactory
    PandasGenerator
    NetworkXGenerator
    ReactionStringGenerator
    SyngraphGenerator

Preprocessing
-------------



Core Graph Builder
^^^^^^^^^^^^^^^^^^

.. automodule:: noctis.data_transformation.preprocessing.core_graph_builder
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: ../../generated

    FormatValidator
    CoreGraphBuilder
    ValidatedStringBuilder
    UnvalidatedStringBuilder

Data Preprocessing
^^^^^^^^^^^^^^^^^^
.. automodule:: noctis.data_transformation.preprocessing.data_preprocessing
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: ../../generated

    PreprocessorConfig
    PandasRowPreprocessorBase
    CSVPreprocessor
    PythonObjectPreprocessorInterface
    ChemicalStringPreprocessorBase
    PythonObjectPreprocessorFactory
    DataFramePreprocessor
    ReactionStringsPreprocessor
    SynGraphPreprocessor
    Preprocessor

GraphExpander
^^^^^^^^^^^^^
.. automodule:: noctis.data_transformation.preprocessing.graph_expander
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: ../../generated

    GraphExpander

Utilities
---------

.. automodule:: noctis.data_transformation.preprocessing.utilities
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: ../../generated

    _update_partition_dict_with_row
    _save_dataframes_to_partition_csv
    _save_list_to_partition_csv
    _merge_partition_files
    create_noctis_relationship
    _delete_tmp_folder
    create_noctis_node
    explode_smiles_like_reaction_string
    explode_v3000_reaction_string
    dict_to_list
    create_data_container
