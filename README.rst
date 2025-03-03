noctis
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



Installation
------------

Create a dedicated Python environment for this package with your favorite environment manager.

.. code-block:: shell

   conda create -n noctis python=3.9
   conda activate noctis


* Option 1: Install the package from the github repository:

.. code-block:: shell

   pip install git+ssh://git@github.com/syngenta/noctis.git@main

* Option 2: Install the package from the Python Package Index (PyPI):

.. code-block:: shell

   pip install noctis


Configuration
-------------
This package requires some configuration parameters to work,
including some secretes to store access credentials to database and services.

After installation, and before the first usage, the use should run the following command

.. code-block:: shell

    noctis_configure
..

| This command generates the <home>/noctis directory and places into it two files:

1. settings.yaml populated with defaults settings. The user can review and modify these values if necessary.
2. .secrets.yaml containing the keys for the necessary secrets. The user must replace the placeholders with the correct values
3. schema.yaml a description of the schema used in the database

| For more details please refer to the Configuration section of the documentation


Development Installation
---------------------------

When working on the development of this package, the developer wants to work
directly on the source code while still using the packaged installation. For
that, run:

.. code-block:: shell

   git clone git@gitlab.com:syngentagroup/scientific-computing-team/noctis.git
   pip install -e noctis/[dev]
