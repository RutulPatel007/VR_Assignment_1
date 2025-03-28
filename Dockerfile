# Use an official Python runtime as a parent image
FROM python:3.12

# Set the working directory
WORKDIR /app


COPY pyproject.toml setup.py /app/


# Install required tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel build twine
# Copy the project files
COPY . /app


# Build the package
RUN python -m build

# Upload to PyPI (Replace with environment variables in production)
CMD twine upload dist/* -u $PYPI_USERNAME -p $PYPI_PASSWORD