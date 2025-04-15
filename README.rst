⭒˚.⋆ NOCTIS ⭒˚.⋆
=======================================

.. image:: https://img.shields.io/pypi/v/noctis
   :target: https://pypi.python.org/pypi/noctis
   :alt: PyPI - Version
.. image:: https://static.pepy.tech/badge/noctis/month
   :target: https://pepy.tech/project/noctis
   :alt: Downloads monthly
.. image:: https://static.pepy.tech/badge/noctis
   :target: https://pepy.tech/project/noctis
   :alt: Downloads total
.. image:: https://img.shields.io/github/actions/workflow/status/syngenta/noctis/test_suite.yml?branch=main
   :alt: GitHub Workflow Status
.. image:: https://readthedocs.org/projects/noctis/badge/?version=latest
   :target: https://noctis.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://img.shields.io/badge/contributions-welcome-blue
   :target: https://github.com/syngenta/noctis/blob/main/CONTRIBUTING.md
   :alt: Contributions
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT

---------------------

.. image:: docs/source/static/noctis.png
   :alt: Noctis Logo
   :align: center
   :width: 250px

What is Noctis?
---------------

**NOCTIS** is a Python package for modeling and analyzing chemical reaction data as graph structures.

It allows you to build and explore complex reaction networks — beyond just molecules and chemical equations — by supporting additional node types, metadata, and relationships.

**NOCTIS** provides tools to:

- Preprocess and validate chemical reaction datasets
- Build graph representations from reaction data
- Query and navigate reaction networks
- Mine synthesis routes using a custom `Java plugin <https://github.com/syngenta/noctis-route-miner>`_
- Interact programmatically with graph databases
- Transform query results into formats suitable for further analysis (e.g., NetworkX, Pandas)



To learn more, visit the `documentation <https://noctis.readthedocs.io/>`_.

Try it interactively with our example notebook!
📓 `Example Notebooks <https://github.com/syngenta/noctis/tree/main/jupyters>`_

Installation
------------

Create a dedicated Python environment for this package with your favorite environment manager.

.. code-block:: shell

   conda create -n noctis python=3.9
   conda activate noctis

* Option 1: Install the package from the GitHub repository:

.. code-block:: shell

   pip install git+ssh://git@github.com/syngenta/noctis.git@main

* Option 2: Install the package from the Python Package Index (PyPI):

.. code-block:: shell

   pip install noctis

Configuration
-------------

This package requires some configuration parameters to work,
including some secrets to store access credentials to database and services.

After installation, and before the first usage, run the following command:

.. code-block:: shell

    noctis_configure

This command creates a `<home>/noctis` directory and places into it:

1. `settings.yaml` – populated with default settings. You can review and modify them.
2. `.secrets.yaml` – contains placeholders for secrets. Fill them with your actual values.
3. `schema.yaml` – a description of the graph schema used in the database.

🔧 For more details, refer to the **Configuration** section of the documentation.

Development Installation
---------------------------

If you're working on the development of Noctis and want to run directly from source:

.. code-block:: shell

   git clone git@gitlab.com:syngentagroup/scientific-computing-team/noctis.git
   pip install -e noctis/[dev]
