# pipeline/Dockerfile

# Use the official Python image
#FROM python:3.9-slim

# Set environment variables
#ENV PYTHONDONTWRITEBYTECODE=1
#ENV PYTHONUNBUFFERED=1

# Set the working directory
#WORKDIR /app

# Install system dependencies
#RUN apt-get update && apt-get install -y build-essential

# Copy requirements.txt
#COPY requirements.txt .

# Install Python dependencies
#RUN pip install --upgrade pip
#RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
#COPY . .

# Expose application and metrics ports
#EXPOSE 8000
#EXPOSE 8001

# Run the application
#CMD ["python", "fetch_live_data.py"]


# pipeline/Dockerfile

# Use the official Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code, including the ML model
COPY . .

# Expose application and metrics ports
EXPOSE 8000
EXPOSE 8001

# Run the fetch_live_data.py script by default
CMD ["python", "fetch_live_data.py"]

