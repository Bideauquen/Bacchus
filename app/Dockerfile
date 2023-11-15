
# Use the official Python image as the base image
FROM python:3.10-slim-buster

# Set the working directory to /
WORKDIR /

# Copy the current directory contents into the container at /app
COPY app/ app/
COPY db/ db/

# Install the required packages
RUN pip install --upgrade pip
RUN pip install fastapi[all] pymongo

# Expose port 80 for the FastAPI application
EXPOSE 8080

# Start the FastAPI application using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]