noctis
=======================================



Installation
------------

Installing this package depends on a working `installation of Python`_ with pip.
To install the package, run:

.. code-block:: shell

   pip install git+ssh://git@gitlab.com:syngentagroup/scientific-computing-team//noctis.git

.. _installation of Python: https://www.python.org/downloads/

For Development
---------------

When working on the development of this package, the developer wants to work
directly on the source code while still using the packaged installation. For
that, run:

.. code-block:: shell

   git clone git@gitlab.com:syngentagroup/scientific-computing-team/noctis.git
   pip install -e noctis/[dev]
