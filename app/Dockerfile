
# Use the official Python image as the base image
FROM python:3.10-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY /app /app

# Install the required packages
RUN pip install fastapi uvicorn pymongo pydantic

# Expose port 80 for the FastAPI application
EXPOSE 8080

# Start the FastAPI application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]