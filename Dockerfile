# Use an official Python runtime as a parent image
FROM python:3.12

# Set the working directory
WORKDIR /app

# Copy the project files
COPY . /app

# Install required tools
RUN pip install --upgrade pip setuptools wheel twine build

# Build the package
RUN python -m build

# Upload to PyPI (Replace with environment variables in production)
CMD twine upload dist/* -u $PYPI_USERNAME -p $PYPI_PASSWORD