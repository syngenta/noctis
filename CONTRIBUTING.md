# Contributing TO pumas

First off, thanks for taking the time to contribute! :tada::+1:

The following is a set of guidelines for contributing to pumas, which are hosted in the [Syngenta Organization](https://github.com/syngenta) on GitHub.
These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Contribution Terms and License

The code and documentation of pumas is contained in this repository. To contribute
to this project or any of the elements of pumas we recommend you start by reading this
contributing guide.

## Contributing to pumas codebase

If you would like to contribute to the package, we recommend the following development setup.

1. Create a copy of the [repository](https://github.com/syngenta/pumas) via the "_Fork_" button.

2. Clone the linchemin repository:

    ```sh
    git clone git@github.com:${GH_ACCOUNT_OR_ORG}/pumas.git
    ```

3. Add remote linchemin repo as an "upstream" in your local repo, so you can check/update remote changes.

   ```sh
   git remote add upstream git@github.com:syngenta/pumas.git
   ```

4. Create a dedicated branch:

    ```sh
    cd pumas
    git checkout -b a-super-nice-feature-we-all-need
    ```

5. Create and activate a dedicated conda environment (any other virtual environment management would work):

    ```sh
    conda env create pumas
    conda activate pumas
    ```

6. Install pumas in editable mode:

    ```sh
    pip install -e .[dev]
    ```

7. Implement your changes and once you are ready run the tests:

    ```sh
    # this can take quite long
    cd pumas/tests
    python -m pytest
    ```

   And add style checks (be aware that running isort might change your files!):
   ```sh
    cd pumas
    # sorting the imports
    python -m isort src/pumas
    # checking black
    python -m black src/pumas
    # checking flake8
    python -m flake8 --ignore E501 src/pumas
    ```
   Or use the pre-commit hooks provided in the repository:
   ```sh
    pre-commit install
    pre-commit run --all-files
    ```

8. Once the tests and checks passes, but most importantly you are happy with the implemented feature, commit your changes.

    ```sh
    # add the changes
    git add
    # commit them
    git commit -s -m "feat: implementing super nice feature." -m "A feature we all need."
    # check upstream changes
    git fetch upstream
    git rebase upstream/main
    # push changes to your fork
    git push -u origin a-super-nice-feature-we-all-need
    ```

9. From your fork, open a pull request via the "_Contribute_" button, the maintainers will be happy to review it.
