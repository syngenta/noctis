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

What is **NOCTIS**?
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
📓 `Example Notebook <https://github.com/syngenta/noctis/tree/main/jupyters>`_

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

Both **NOCTIS** and its companion package **Linchemin** need a small on-disk configuration before you can use them.

After installation, and before the first usage, run the following command:

.. code-block:: shell

    noctis_configure && linchemin_configure

What each command does
----------------------

| ``noctis_configure``
| Creates the directory ``~/noctis/`` and writes three files

    |  ``settings.yaml`` – default runtime settings (edit as needed)
    |  ``.secrets.yaml`` – placeholders for credentials (replace with the real
      values)
    |  ``schema.yaml`` – description of the graph schema used by the database

| ``linchemin_configure``
| Creates the directory ``~/linchemin/`` and writes two files

    |  ``settings.toml`` – default settings you may tweak
    |  ``.secrets.toml`` – placeholders for the required secrets

For more details:

| 🔧 **NOCTIS** – see the *Configuration* chapter of this `documentation <https://noctis.readthedocs.io/>`_.
| 🔧 **Linchemin** – see the `Linchemin repo <https://github.com/syngenta/linchemin>`_ for full details.
|
|
| ⚠️ **ALERT** ⚠️:
| If you skip this step you’ll run straight into mysterious import errors (e.g. *“Settings object has no attribute ‘CONSTRUCTORS’”*). The fix is simply to run the two commands above once.


Development Installation
---------------------------

If you're working on the development of **NOCTIS** and want to run directly from source:

.. code-block:: shell

   git clone git@gitlab.com:syngentagroup/scientific-computing-team/noctis.git
   pip install -e noctis/[dev]
